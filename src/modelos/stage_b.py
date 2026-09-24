"""
Stage B — detecção de anomalias sobre a matriz de features.

Idêntico para as quatro engines: lê a matriz produzida pelo Stage A (que o
teste de equivalência já garantiu ser igual entre elas), aplica três modelos e
calcula as métricas.

============================================================================
EXERCÍCIO
============================================================================
A infraestrutura (carregar, alinhar rótulos, gravar resultado) está pronta.
Faltam as cinco funções marcadas com `TODO`. Elas são o conteúdo de ML do
trabalho — a parte que a banca vai perguntar.

Ordem sugerida, da mais simples para a mais complexa:

    1. zscore_score
    2. precision_at_k
    3. iforest_score
    4. lof_score
    5. avaliar_por_tipo

Para conferir cada função à medida que avança:

    ~/tcc-venv/bin/python scripts/verificar_stage_b.py

O verificador testa uma função por vez, com exemplos pequenos em que dá para
calcular a resposta de cabeça. Quando os testes passarem, rode com os dados
reais:

    ~/tcc-venv/bin/python scripts/verificar_stage_b.py --real

Material de apoio: Parte 6 do notebooks/guia_do_projeto.ipynb. Tudo o que você
precisa já foi executado lá — o exercício é trazer para cá entendendo cada
linha.
============================================================================

Uso (depois de pronto):
    python -m src.modelos.stage_b --escala 100mb
"""
from __future__ import annotations
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.gerador import config as cfg

# Proporção conhecida de anomalias. Passada ao `contamination` dos modelos.
CONTAMINACAO = cfg.PCT_ANOMALIA

# Semente fixa: sem ela, o Isolation Forest e a amostragem do LOF dariam
# resultados diferentes a cada execução, e o experimento não seria reprodutível.
SEMENTE = 42

# O LOF compara cada ponto com seus vizinhos, com custo ~O(n²). Acima deste
# tamanho ele roda sobre uma amostra — limitação declarada na metodologia.
AMOSTRA_LOF = 20_000


# ===========================================================================
# INFRAESTRUTURA — pronta, não precisa mexer
# ===========================================================================

def carregar(escala: str, engine: str = "pandas"):
    """
    Lê a matriz de features e alinha os rótulos pelo cliente_id.

    O alinhamento é o ponto delicado: se os rótulos ficassem numa ordem
    diferente da matriz, todas as métricas sairiam erradas sem nenhum erro
    aparente. Por isso é feito por chave, e não por posição.

    Devolve
    -------
    X     : matriz padronizada (média 0, desvio 1 por coluna)
    y     : 1 = anômalo, 0 = normal
    tipos : anomaly_type de cada linha ('normal', 'rajada', ...)
    """
    feats = pd.read_parquet(
        f"{cfg.DIR_DADOS}/features/{escala}/{engine}.parquet"
    ).set_index("cliente_id").sort_index()

    import json
    with open(cfg.MANIFESTO, encoding="utf-8") as fp:
        manifesto = json.load(fp)
    labels = pd.concat(
        [pd.read_parquet(p) for p in manifesto[escala]["labels"]],
        ignore_index=True,
    ).set_index("cliente_id").loc[feats.index]

    # Padronização: sem ela, patrimonio_estimado (milhões) dominaria qualquer
    # cálculo de distância sobre pct_web (0 a 1).
    X = StandardScaler().fit_transform(feats.to_numpy(dtype="float64"))
    y = labels["is_anomaly"].to_numpy()
    tipos = labels["anomaly_type"].to_numpy()
    return X, y, tipos


# ===========================================================================
# EXERCÍCIO — implemente as funções abaixo
# ===========================================================================
#
# Convenção usada pelas três funções de score: MAIOR = MAIS ANÔMALO.
# Todas as métricas assumem isso. Se um modelo devolver o contrário, inverta.


def zscore_score(X: np.ndarray) -> np.ndarray:
    """
    Baseline univariado.

    X já está padronizado, ou seja, cada valor JÁ É um z-score. O score do
    cliente é o maior |z| entre todas as suas features: se qualquer
    característica estiver muito longe da média, ele é suspeito.

    Entrada : X com formato (n_clientes, n_features)
    Saída   : vetor com formato (n_clientes,)

    Dica: np.abs e .max com o parâmetro axis. Qual axis percorre as colunas
    de cada linha?
    """
    # 1: Torna todos os valores em positivo
    z_abs = np.abs(X)

    # 2: Pega o maior valor de cada linha
    return z_abs.max(axis=1)


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Dos k clientes com MAIOR score, que fração é realmente anômala?

    É a métrica que um gestor entende: "se um analista revisar os k casos
    mais suspeitos, quantos serão fraude de verdade?".

    Exemplo: y = [1, 0, 1, 0], scores = [0.9, 0.8, 0.1, 0.7], k = 2
             os 2 maiores scores são os índices 0 e 1 -> y = [1, 0] -> 0.5

    Dica: np.argsort ordena do MENOR para o maior. Como pegar os k maiores?
    """
    # 1: Ordena os scores do menor para o maior
    scores_ordenados = np.argsort(scores)

    # 2: Pega os k maiores
    k_maiores = scores_ordenados[-k:]

    # 3: Calcula a precisão
    return y[k_maiores].mean()


def iforest_score(X: np.ndarray) -> np.ndarray:
    """
    Isolation Forest: anomalias são isoladas com poucos cortes aleatórios.

    Use sklearn.ensemble.IsolationForest com contamination=CONTAMINACAO,
    random_state=SEMENTE e n_jobs=-1.

    ATENÇÃO À CONVENÇÃO: o método score_samples devolve valores MENORES para
    os pontos mais anômalos — o oposto do que as métricas esperam.

    Pergunta para responder antes de implementar: por que é preciso chamar
    fit antes de score_samples, se o modelo é não supervisionado e nunca vê
    os rótulos? O que ele "aprende"?

    Resposta: Sem o fit, a estrutura de árvores aleatórias não é criada e,
    portanto, não há como isolar pontos. O Isolation Forest "aprende" a estrutura
    dos dados através do fit, que cria as árvores de decisão aleatórias que
    serão usadas para calcular a anomalia de cada ponto.
    """
    if_model = IsolationForest(
        contamination=CONTAMINACAO,
        random_state=SEMENTE,
        n_jobs=-1
    )

    if_model.fit(X)

    return -if_model.score_samples(X)


def lof_score(X: np.ndarray) -> np.ndarray:
    """
    Local Outlier Factor: um ponto é anômalo se sua vizinhança é muito menos
    densa que a vizinhança dos seus vizinhos.

    Use sklearn.neighbors.LocalOutlierFactor com n_neighbors=20,
    contamination=CONTAMINACAO e n_jobs=-1. Chame fit_predict(X) e depois leia
    o atributo negative_outlier_factor_.

    ATENÇÃO À CONVENÇÃO: negative_outlier_factor_ é, como o nome diz,
    NEGATIVO, e mais negativo = mais anômalo.

    Esta função recebe X já amostrado quando necessário (ver `main`); não
    precisa amostrar aqui dentro.

    Pergunta: por que o LOF não tem um score_samples como o Isolation Forest?
    (Pista: o que ele precisaria saber sobre um cliente novo para calcular a
    densidade dele?)

    Resposta: Pois o LOF calcula a densidade local de cada ponto em relação
    aos seus vizinhos.  
    """
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINACAO,
        n_jobs=-1)
    
    lof.fit_predict(X)

    return -lof.negative_outlier_factor_


def avaliar_por_tipo(tipos: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """
    PR-AUC separado para cada tipo de anomalia.

    Para cada tipo em cfg.TIPOS_ANOMALIA:
      1. selecione só as linhas daquele tipo MAIS as linhas 'normal'
         (as dos outros tipos de anomalia ficam de fora);
      2. monte o rótulo binário: 1 se for daquele tipo, 0 se for normal;
      3. calcule average_precision_score com os scores dessas mesmas linhas.

    Devolve algo como {'rajada': 0.49, 'ticket_desproporcional': 0.03, ...}

    Pergunta: por que tirar os outros tipos de anomalia da conta, em vez de
    tratá-los como 0? O que aconteceria com o PR-AUC da 'rajada' se um
    cliente 'multivariada' com score alto contasse como falso positivo?
    """
    
    metricas = {}

    for tipo in cfg.TIPOS_ANOMALIA:
        # 1: filtra apenas as linhas com tipo atual ou normais
        sel = (tipos == tipo) | (tipos == 'normal')
        # 2: cria rótulo binário, 1 para o tipo atual, 0 para normal
        y_true = (tipos[sel] == tipo).astype(int)
        y_score = scores[sel]
        
        metricas[tipo] = average_precision_score(y_true, y_score)
        
    return metricas


# ===========================================================================
# EXECUÇÃO — pronta, usa as funções acima
# ===========================================================================

def avaliar(X: np.ndarray, y: np.ndarray, tipos: np.ndarray) -> pd.DataFrame:
    """Roda os três modelos e monta a tabela de resultados."""
    rng = np.random.default_rng(SEMENTE)
    linhas = []

    modelos = [("Z-Score", zscore_score), ("IForest", iforest_score),
               ("LOF", lof_score)]

    for nome, funcao in modelos:
        if nome == "LOF" and len(X) > AMOSTRA_LOF:
            idx = np.sort(rng.choice(len(X), size=AMOSTRA_LOF, replace=False))
        else:
            idx = np.arange(len(X))

        s = funcao(X[idx])
        yy, tt = y[idx], tipos[idx]
        # k = nº de anomalias presentes no conjunto avaliado (na amostra do
        # LOF, é menor que o total).
        kk = int(yy.sum())

        linha = {
            "modelo": nome,
            "n": len(idx),
            "pr_auc": average_precision_score(yy, s),
            "acaso": yy.mean(),
            "precision_at_k": precision_at_k(yy, s, kk),
            "k": kk,
        }
        linha |= {f"pr_auc_{t}": v for t, v in avaliar_por_tipo(tt, s).items()}
        linhas.append(linha)

    return pd.DataFrame(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage B — detecção de anomalias")
    ap.add_argument("--escala", required=True, choices=list(cfg.ESCALAS))
    ap.add_argument("--engine", default="pandas",
                    help="de qual engine ler a matriz (são equivalentes)")
    ap.add_argument("--saida", default="results/stage_b.csv")
    args = ap.parse_args()

    X, y, tipos = carregar(args.escala, args.engine)
    print(f"escala {args.escala}: {X.shape[0]:,} clientes x {X.shape[1]} "
          f"features, {y.sum():,} anomalias ({y.mean():.2%})\n")

    res = avaliar(X, y, tipos)
    res.insert(0, "escala", args.escala)

    cols = ["modelo", "n", "pr_auc", "acaso", "precision_at_k"]
    print(res[cols].round(4).to_string(index=False))

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    res.to_csv(args.saida, mode="a", index=False,
               header=not os.path.exists(args.saida))
    print(f"\ngravado em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
