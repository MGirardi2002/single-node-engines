"""
Gerador sintético paramétrico — modelo fato + dimensão.

Produz três tabelas por chunk:

  clientes    (dimensão, ~40k linhas)  atributos cadastrais correlacionados
  transacoes  (fato, ~4M linhas)       uma linha por transação, com timestamp
  labels      (rótulos, ~40k linhas)   is_anomaly / anomaly_type por cliente

Os rótulos ficam em arquivo separado de propósito: a matriz de features nunca
os carrega, o que elimina por construção o risco de vazamento (leakage).

A metodologia de fator de escala segue a prática de benchmarks padronizados
(TPC-H / TPC-DS): os dados são gerados por um processo paramétrico, e o volume
é controlado por um parâmetro de escala, mantendo as distribuições constantes.

Uso:
    python -m src.gerador.gerar --escala 100mb
    python -m src.gerador.gerar --escala 2gb
    python -m src.gerador.gerar --calibrar
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from . import config as cfg


# ===========================================================================
# DIMENSÃO: CLIENTES
# ===========================================================================

def _gerar_clientes(
    rng: np.random.Generator, n: int, id_inicial: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera a dimensão de clientes com atributos correlacionados.

    As dependências (idade -> renda -> score -> limite) são intencionais: um
    gerador de colunas independentes produziria uma base em que qualquer
    anomalia multivariada seria indistinguível de ruído.
    """
    idade = np.clip(rng.normal(38, 12, n), 18, 80).astype(np.int16)

    renda = np.clip(1800 + idade * 120 + rng.normal(0, 1500, n), 1200, None)

    score_credito = np.clip(
        450 + renda / 100 + rng.normal(0, 50, n), 300, 1000
    ).round().astype(np.int16)

    limite_credito = np.clip(
        renda * 1.5 + (score_credito - 500) * 20 + rng.normal(0, 1000, n),
        500, None,
    )

    tempo_cliente_meses = np.clip(
        rng.gamma(3, 15, n), 1, 120
    ).astype(np.int16)

    qtd_dependentes = np.clip(rng.poisson(1.5, n), 0, 8).astype(np.int8)

    patrimonio_estimado = renda * tempo_cliente_meses * rng.uniform(0.3, 1.5, n)

    # Ticket médio esperado do cliente — usado para gerar o valor das
    # transações. Não é exposto na dimensão: é um parâmetro latente, e as
    # engines terão que reconstruí-lo via agregação.
    ticket_latente = np.clip(40 + renda * 0.015 + rng.normal(0, 20, n), 10, None)

    # Perfil comportamental — variável LATENTE: não entra em nenhuma tabela.
    # As engines só observam seus efeitos nas transações, como aconteceria com
    # dados reais.
    perfil = rng.choice(cfg.PERFIS, size=n, p=cfg.PERFIS_P)

    # Cesta de consumo — também latente. Define quais categorias o cliente
    # frequenta, criando correlação entre categorias na população normal.
    cesta = rng.integers(0, len(cfg.CESTAS), size=n)

    df = pd.DataFrame({
        "cliente_id": np.arange(id_inicial, id_inicial + n, dtype=np.int64),
        "idade": idade,
        "renda_mensal": renda.round(2),
        "score_credito": score_credito,
        "limite_credito": limite_credito.round(2),
        "tempo_cliente_meses": tempo_cliente_meses,
        "qtd_dependentes": qtd_dependentes,
        "patrimonio_estimado": patrimonio_estimado.round(2),
        "regiao": rng.choice(cfg.REGIOES, size=n, p=cfg.REGIOES_P),
        "tipo_cliente": rng.choice(cfg.TIPOS_CLIENTE, size=n, p=cfg.TIPOS_CLIENTE_P),
    })

    return df, ticket_latente, perfil, cesta


# ===========================================================================
# FATO: TRANSAÇÕES
# ===========================================================================

def _matriz_cestas() -> np.ndarray:
    """
    Matriz (n_cestas x n_categorias) com a distribuição de gasto de cada cesta.
    PESO_NUCLEO fica nas 5 categorias-núcleo; o resto se espalha nas demais.
    """
    n_cat = len(cfg.CATEGORIAS)
    m = np.zeros((len(cfg.CESTAS), n_cat))
    for i, nucleo in enumerate(cfg.CESTAS.values()):
        m[i, :] = (1.0 - cfg.PESO_NUCLEO) / (n_cat - len(nucleo))
        m[i, list(nucleo)] = cfg.PESO_NUCLEO / len(nucleo)
    return m


def _sortear_categorias(
    rng: np.random.Generator, cesta: np.ndarray, n_tx: np.ndarray
) -> np.ndarray:
    """
    Sorteia a categoria de cada transação conforme a cesta do seu cliente.

    Faz um laço sobre as cestas (8 iterações) em vez de montar uma matriz de
    probabilidades por transação — esta teria ~6M x 20 floats e não caberia
    confortavelmente em memória nas escalas maiores.
    """
    probs = _matriz_cestas()
    cesta_por_tx = np.repeat(cesta, n_tx)
    out = np.empty(cesta_por_tx.size, dtype=np.int16)
    for i in range(probs.shape[0]):
        mask = cesta_por_tx == i
        k = int(mask.sum())
        if k:
            out[mask] = rng.choice(probs.shape[1], size=k, p=probs[i])
    return out

def _gerar_transacoes(
    rng: np.random.Generator,
    clientes: pd.DataFrame,
    ticket_latente: np.ndarray,
    perfil: np.ndarray,
    cesta: np.ndarray,
    chunk_id: int,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """
    Gera a tabela fato. Retorna também o índice local do cliente de cada
    transação (usado para injetar as anomalias de forma vetorizada) e a
    origem da linha do tempo em segundos epoch.

    `data_hora` sai daqui como offset inteiro em segundos; a conversão para
    timestamp acontece só depois da injeção de anomalias, que desloca datas.
    """
    n_clientes = len(clientes)

    # Clientes mais antigos e de tipo superior transacionam um pouco mais.
    peso_tipo = pd.Series(clientes["tipo_cliente"]).map(
        {"Bronze": 0.8, "Prata": 1.0, "Ouro": 1.3, "Platina": 1.7}
    ).to_numpy()

    mult_freq = pd.Series(perfil).map(cfg.PERFIL_FREQ).to_numpy()
    mult_ticket = pd.Series(perfil).map(cfg.PERFIL_TICKET).to_numpy()

    lam = cfg.TX_POR_CLIENTE * peso_tipo * mult_freq * rng.uniform(0.6, 1.4, n_clientes)
    n_tx = np.maximum(rng.poisson(lam), 5).astype(np.int64)

    total = int(n_tx.sum())
    idx_local = np.repeat(np.arange(n_clientes, dtype=np.int64), n_tx)

    # --- timestamps -------------------------------------------------------
    t0 = np.datetime64(cfg.DATA_INICIO, "s").astype(np.int64)
    segundos_janela = cfg.DIAS_JANELA * 86_400
    offset_seg = rng.integers(0, segundos_janela, size=total, dtype=np.int64)

    # --- valores ----------------------------------------------------------
    # Lognormal em torno do ticket latente do cliente: cauda direita pesada,
    # como em gastos reais.
    base = np.repeat(ticket_latente * mult_ticket, n_tx)
    valor = base * rng.lognormal(mean=0.0, sigma=0.45, size=total)

    # --- efeitos de perfil ------------------------------------------------
    canais = rng.choice(cfg.CANAIS, size=total, p=[0.55, 0.33, 0.12])

    # 'digital': nunca usa canal presencial. Produz pct_web/pct_app extremos
    # numa população perfeitamente normal — é o falso positivo que derruba um
    # detector univariado sobre o perfil de canal.
    mask_digital = np.repeat(perfil == "digital", n_tx)
    canais[mask_digital] = rng.choice(
        ["app", "web"], size=int(mask_digital.sum()), p=[0.6, 0.4]
    )

    # 'sazonal': concentra compras em três janelas do ano (datas comemorativas).
    # Gera burstiness alta de forma legítima, competindo com a anomalia
    # 'rajada' — que passa a só se distinguir pela combinação com outros sinais.
    mask_sazonal = np.repeat(perfil == "sazonal", n_tx)
    n_saz = int(mask_sazonal.sum())
    if n_saz:
        centros = np.array([0.30, 0.62, 0.94]) * segundos_janela
        escolha = rng.choice(len(centros), size=n_saz)
        offset_seg[mask_sazonal] = np.clip(
            centros[escolha] + rng.normal(0, 8 * 86_400, n_saz),
            0, segundos_janela - 1,
        ).astype(np.int64)

    tx = pd.DataFrame({
        "transacao_id": np.arange(total, dtype=np.int64) + chunk_id * cfg.STRIDE_TX_ID,
        "cliente_id": np.repeat(clientes["cliente_id"].to_numpy(), n_tx),
        "data_hora": offset_seg,  # offset em segundos; convertido no final
        "valor": valor,
        "categoria": np.asarray(cfg.CATEGORIAS)[_sortear_categorias(rng, cesta, n_tx)],
        "canal": canais,
        "status": rng.choice(
            ["aprovada", "negada", "estornada"], size=total, p=[0.94, 0.05, 0.01]
        ),
    })

    return tx, idx_local, int(t0)


# ===========================================================================
# INJEÇÃO DE ANOMALIAS
# ===========================================================================

def _injetar_anomalias(
    rng: np.random.Generator,
    clientes: pd.DataFrame,
    tx: pd.DataFrame,
    idx_local: np.ndarray,
    t0: int,
) -> pd.DataFrame:
    """
    Marca ~PCT_ANOMALIA dos clientes e altera suas transações.

    Nenhum dos quatro padrões é um outlier univariado grosseiro. Todos exigem
    agregação, janela temporal ou razão com atributo da dimensão para se
    tornarem visíveis — é o que dá propósito à etapa de feature engineering e o
    que separa Isolation Forest/LOF de um Z-Score por coluna.
    """
    n_clientes = len(clientes)
    n_anom = int(n_clientes * cfg.PCT_ANOMALIA)

    is_anomaly = np.zeros(n_clientes, dtype=np.int8)
    anomaly_type = np.full(n_clientes, "normal", dtype=object)

    escolhidos = rng.choice(n_clientes, size=n_anom, replace=False)
    grupos = np.array_split(escolhidos, len(cfg.TIPOS_ANOMALIA))

    # copy=True é obrigatório: sob copy-on-write (pandas 2.x+) os arrays
    # devolvidos por to_numpy() são somente-leitura.
    valor = tx["valor"].to_numpy(copy=True)
    offset = tx["data_hora"].to_numpy(copy=True)
    canal = tx["canal"].to_numpy(copy=True)

    # -- 1. RAJADA ---------------------------------------------------------
    # Volume total e ticket normais; o que é anormal é a concentração
    # temporal. Só aparece depois de uma window function.
    alvo = grupos[0]
    mask = np.isin(idx_local, alvo)
    pos = np.flatnonzero(mask)
    # 40% das transações do cliente colapsam numa janela de 3 dias.
    # 50% das transações num intervalo de 2 dias. O perfil 'sazonal' também
    # concentra compras, mas em três janelas com dispersão de ~8 dias — um pico
    # de 2 dias permanece distinguível dele sem ser um outlier trivial.
    #
    # O início da rajada é sorteado POR CLIENTE e depois difundido para as
    # transações dele. Sortear por transação apenas redistribuiria as datas de
    # forma uniforme, sem formar rajada nenhuma.
    inicio_por_cliente = np.zeros(n_clientes, dtype=np.int64)
    inicio_por_cliente[alvo] = (
        rng.integers(0, cfg.DIAS_JANELA - 2, size=alvo.size) * 86_400
    )
    sel = pos[rng.random(pos.size) < 0.50]
    offset[sel] = (
        inicio_por_cliente[idx_local[sel]]
        + rng.integers(0, 2 * 86_400, size=sel.size)
    )
    is_anomaly[alvo] = 1
    anomaly_type[alvo] = "rajada"

    # -- 2. TICKET DESPROPORCIONAL ----------------------------------------
    # Valores elevados em relação à renda do cliente, mas dentro da faixa
    # global de valores (um cliente de renda alta teria esses mesmos valores
    # sem ser anômalo). Anomalia contextual.
    alvo = grupos[1]
    mask = np.isin(idx_local, alvo)
    # ×1.8–3.0 apenas: o perfil 'premium_raro' já tem ticket 4x acima do
    # esperado pela renda de forma legítima, então uma magnitude maior faria
    # este tipo virar um outlier univariado trivial.
    valor[mask] *= rng.uniform(1.8, 3.0, size=mask.sum())
    is_anomaly[alvo] = 1
    anomaly_type[alvo] = "ticket_desproporcional"

    # -- 3. MIGRAÇÃO DE CANAL ---------------------------------------------
    # Cliente historicamente presencial migra abruptamente para web na segunda
    # metade da janela, com elevação moderada de valor.
    alvo = grupos[2]
    mask = np.isin(idx_local, alvo)
    metade = cfg.DIAS_JANELA * 86_400 // 2
    recente = mask & (offset > metade)
    antigo = mask & (offset <= metade)
    canal[antigo] = "pos"
    canal[recente] = "web"
    # Elevação modesta: o perfil 'digital' já concentra 100% em app/web, então
    # o que distingue esta anomalia não é o canal em si, e sim a *mudança* no
    # tempo somada à elevação de valor — só visível cruzando janela temporal
    # com perfil de canal.
    valor[recente] *= rng.uniform(1.3, 2.0, size=recente.sum())
    is_anomaly[alvo] = 1
    anomaly_type[alvo] = "migracao_canal"

    # -- 4. MULTIVARIADA ---------------------------------------------------
    # Deliberadamente SEM nenhuma marginal extrema. O cliente mistura duas
    # cestas de consumo que nunca coocorrem na população normal: cada
    # participação por categoria fica dentro da faixa usual, mas o vetor
    # conjunto é improvável. Além disso, a correlação score<->limite (que vale
    # para toda a população) é quebrada de forma suave — ambos os valores
    # permanecem em faixas comuns, só que incompatíveis entre si.
    #
    # Um detector univariado é cego para isso por construção: não existe
    # coluna cujo |z| seja alto. É o único tipo em que Isolation Forest e LOF
    # têm vantagem estrutural sobre o Z-Score.
    alvo = grupos[3]
    mask = np.isin(idx_local, alvo)

    # Recategoriza as transações desses clientes com a mistura de duas cestas
    # distantes, 50/50.
    probs = _matriz_cestas()
    cats = np.asarray(cfg.CATEGORIAS)
    pos = np.flatnonzero(mask)
    par_por_cliente = rng.integers(0, len(cfg.PARES_DISTANTES), size=alvo.size)
    nomes = list(cfg.CESTAS)
    mapa_cliente = dict(zip(alvo.tolist(), par_por_cliente.tolist()))
    for p, (a, b) in enumerate(cfg.PARES_DISTANTES):
        clientes_do_par = [c for c, pp in mapa_cliente.items() if pp == p]
        if not clientes_do_par:
            continue
        sub = pos[np.isin(idx_local[pos], clientes_do_par)]
        mistura = 0.5 * probs[nomes.index(a)] + 0.5 * probs[nomes.index(b)]
        categoria_nova = rng.choice(len(cats), size=sub.size, p=mistura)
        tx.loc[sub, "categoria"] = cats[categoria_nova]

    # Quebra suave da correlação score <-> limite: valores individualmente
    # comuns (score na faixa 480-560, limite 15k-25k), combinação rara.
    clientes.loc[alvo, "score_credito"] = rng.integers(
        480, 560, size=alvo.size
    ).astype(np.int16)
    clientes.loc[alvo, "limite_credito"] = rng.uniform(
        15_000, 25_000, size=alvo.size
    ).round(2)
    is_anomaly[alvo] = 1
    anomaly_type[alvo] = "multivariada"

    # -- consolida ---------------------------------------------------------
    tx["valor"] = valor.round(2)
    tx["canal"] = canal
    tx["data_hora"] = pd.to_datetime(t0 + offset, unit="s")

    labels = pd.DataFrame({
        "cliente_id": clientes["cliente_id"].to_numpy(),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type.astype(str),
    })

    return labels


# ===========================================================================
# ORQUESTRAÇÃO
# ===========================================================================

def gerar_chunk(chunk_id: int, base: str = cfg.DIR_RAW) -> dict:
    """Gera e persiste um chunk. Idempotente para um mesmo chunk_id."""
    rng = np.random.default_rng(cfg.seed_do_chunk(chunk_id))
    id_inicial = chunk_id * cfg.CHUNK_CLIENTES + 1

    clientes, ticket_latente, perfil, cesta = _gerar_clientes(
        rng, cfg.CHUNK_CLIENTES, id_inicial
    )
    tx, idx_local, t0 = _gerar_transacoes(
        rng, clientes, ticket_latente, perfil, cesta, chunk_id
    )
    labels = _injetar_anomalias(rng, clientes, tx, idx_local, t0)

    # Embaralha a fato: em dados reais a ordem física não acompanha o
    # cliente, e ordenação prévia favoreceria artificialmente algumas engines.
    tx = tx.sample(frac=1, random_state=cfg.seed_do_chunk(chunk_id)).reset_index(drop=True)

    destino = cfg.dir_do_chunk(chunk_id, base)
    os.makedirs(destino, exist_ok=True)
    clientes.to_parquet(f"{destino}/clientes.parquet", index=False)
    tx.to_parquet(f"{destino}/transacoes.parquet", index=False)
    labels.to_parquet(f"{destino}/labels.parquet", index=False)

    mb = sum(
        os.path.getsize(f"{destino}/{f}")
        for f in ("clientes.parquet", "transacoes.parquet", "labels.parquet")
    ) / 1024**2

    return {"chunk_id": chunk_id, "n_transacoes": len(tx), "mb": round(mb, 2)}


def gerar_escala(escala: str, base: str = cfg.DIR_RAW) -> None:
    """
    Gera todos os chunks necessários para uma escala.

    Chunks já existentes são reaproveitados — é o que torna o nested scaling
    barato: gerar "2gb" depois de "500mb" só produz os chunks que faltam.
    """
    if escala not in cfg.ESCALAS:
        raise ValueError(f"Escala desconhecida: {escala}. Opções: {list(cfg.ESCALAS)}")

    n_chunks = cfg.ESCALAS[escala]
    print(f"Escala '{escala}': {n_chunks} chunk(s) x {cfg.CHUNK_CLIENTES:,} clientes")

    total_mb, total_tx = 0.0, 0
    for c in range(n_chunks):
        destino = cfg.dir_do_chunk(c, base)
        if os.path.exists(f"{destino}/transacoes.parquet"):
            mb = os.path.getsize(f"{destino}/transacoes.parquet") / 1024**2
            print(f"  chunk {c:04d}  [reaproveitado]")
            total_mb += mb
            continue
        info = gerar_chunk(c, base)
        total_mb += info["mb"]
        total_tx += info["n_transacoes"]
        print(f"  chunk {c:04d}  {info['n_transacoes']:>10,} tx  {info['mb']:>8.2f} MB")

    print(f"\n[OK] Escala '{escala}': ~{total_mb:.1f} MB em {n_chunks} chunk(s)")
    _escrever_manifesto(base)


def _escrever_manifesto(base: str = cfg.DIR_RAW) -> None:
    """
    Mapa escala -> lista de caminhos. As pipelines leem daqui em vez de
    duplicar os dados por escala (economiza ~4x de disco e de tempo).
    """
    manifesto = {}
    for escala, n in cfg.ESCALAS.items():
        chunks = [cfg.dir_do_chunk(c, base) for c in range(n)]
        if all(os.path.exists(f"{d}/transacoes.parquet") for d in chunks):
            manifesto[escala] = {
                "n_chunks": n,
                "transacoes": [f"{d}/transacoes.parquet" for d in chunks],
                "clientes": [f"{d}/clientes.parquet" for d in chunks],
                "labels": [f"{d}/labels.parquet" for d in chunks],
            }
    os.makedirs(os.path.dirname(cfg.MANIFESTO), exist_ok=True)
    with open(cfg.MANIFESTO, "w", encoding="utf-8") as fp:
        json.dump(manifesto, fp, indent=2)
    print(f"[OK] Manifesto: {cfg.MANIFESTO} ({', '.join(manifesto) or 'vazio'})")


def calibrar() -> None:
    """
    Mede bytes/linha reais para ajustar CHUNK_CLIENTES às escalas nominais.
    Roda num diretório temporário e não afeta os dados do benchmark.
    """
    tmp = "data/_calibracao"
    shutil.rmtree(tmp, ignore_errors=True)
    info = gerar_chunk(0, base=tmp)
    mb_por_chunk = info["mb"]
    print(f"\n1 chunk ({cfg.CHUNK_CLIENTES:,} clientes) = {mb_por_chunk:.2f} MB "
          f"/ {info['n_transacoes']:,} transações")
    print(f"  -> {mb_por_chunk * 1024**2 / info['n_transacoes']:.1f} bytes por linha\n")
    print("Chunks necessários por escala nominal:")
    for escala, alvo_mb in [("100mb", 100), ("200mb", 200), ("400mb", 400),
                            ("800mb", 800), ("1600mb", 1600)]:
        print(f"  {escala:>6}: {max(1, round(alvo_mb / mb_por_chunk)):>3} chunks "
              f"(config atual: {cfg.ESCALAS[escala]})")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gerador sintético do TCC")
    ap.add_argument("--escala", choices=list(cfg.ESCALAS), help="escala a gerar")
    ap.add_argument("--calibrar", action="store_true",
                    help="mede bytes/linha e sugere CHUNK_CLIENTES")
    args = ap.parse_args()

    if args.calibrar:
        calibrar()
    elif args.escala:
        gerar_escala(args.escala)
    else:
        ap.error("informe --escala ou --calibrar")
