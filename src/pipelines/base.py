"""
Contrato do Stage A — a especificação que as quatro engines devem honrar.

Cada engine implementa o mesmo pipeline lógico de forma **idiomática** (Polars
em lazy, DuckDB em SQL, Spark em DataFrame API). O que NÃO pode variar é o
resultado: a matriz de features precisa sair idêntica, a menos de erro de ponto
flutuante. Se divergir, comparar tempos de execução não significa nada.

Por isso toda definição ambígua está fixada aqui, e não deixada ao default de
cada biblioteca:

  * desvio-padrão é **amostral** (ddof=1), com 0.0 quando n < 2;
  * o bucket de 7 dias é `(epoch_seg - t_min) // 604800` — inteiro e portável,
    em vez de `floor('7D')`, que ancora no epoch de forma diferente conforme a
    biblioteca;
  * divisões por zero têm valor de preenchimento explícito;
  * a saída é ordenada por `cliente_id`, com as colunas na ordem de COLUNAS.

Etapas do pipeline (todas medidas separadamente):

  ingestao -> limpeza -> agregacao -> janela -> join -> derivadas -> escrita
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.gerador import config as cfg

# ---------------------------------------------------------------------------
# Constantes do contrato
# ---------------------------------------------------------------------------

SEGUNDOS_POR_DIA = 86_400
SEGUNDOS_POR_BUCKET = 7 * SEGUNDOS_POR_DIA  # 604_800

# Valores de preenchimento quando o denominador é zero. Fixados para que as
# quatro engines produzam o mesmo número em vez de NaN/NULL/Inf conforme o
# tratamento de cada uma.
FILL_STD = 0.0
FILL_DELTA_TICKET = 1.0
FILL_DELTA_WEB = 0.0
FILL_RAZAO = 0.0

# Tolerância do teste de equivalência. Não pode ser apertada demais: somas de
# milhões de floats acumulam em ordens diferentes conforme o plano de execução
# de cada engine, e o erro relativo da soma ingênua cresce com n (~n*eps).
# 1e-6 é folgado o bastante para absorver isso e apertado o bastante para
# detectar qualquer divergência semântica real.
RTOL = 1e-6
ATOL = 1e-9

CANAIS = tuple(sorted(cfg.CANAIS))
CATEGORIAS = tuple(sorted(cfg.CATEGORIAS))

COLUNAS: tuple[str, ...] = (
    "cliente_id",
    # --- agregações da tabela fato ---
    "n_tx",
    "valor_total",
    "valor_medio",
    "valor_std",
    "valor_max",
    "valor_min",
    "n_categorias",
    "n_canais",
    "recencia_dias",
    "dias_ativo",
    "pct_negada",
    # --- window function temporal ---
    "max_tx_7d",
    "burstiness",
    # --- perfil de canal ---
    *(f"pct_{c}" for c in CANAIS),
    "delta_pct_web",
    "delta_ticket",
    # --- perfil de categoria ---
    *(f"cat_{c}" for c in CATEGORIAS),
    # --- atributos da dimensão ---
    "idade",
    "renda_mensal",
    "score_credito",
    "limite_credito",
    "tempo_cliente_meses",
    "qtd_dependentes",
    "patrimonio_estimado",
    # --- razões derivadas ---
    "razao_gasto_renda",
    "razao_ticket_renda",
    "razao_limite_renda",
    "uso_limite",
)

ETAPAS: tuple[str, ...] = (
    "ingestao",
    "limpeza",
    "agregacao",
    "janela",
    "join",
    "derivadas",
    "escrita",
)


@dataclass
class Cronometro:
    """
    Mede o tempo de cada etapa do pipeline.

    Engines com avaliação preguiçosa (Polars lazy, Spark) não executam nada até
    haver uma materialização, de modo que cronometrar a construção do plano
    mediria zero. As implementações dessas engines forçam a materialização ao
    final de cada etapa e registram isso em `materializado`, para que a
    comparação entre engines seja honesta e a limitação fique documentada.
    """

    tempos: dict[str, float] = field(default_factory=dict)
    materializado: bool = True

    @contextmanager
    def etapa(self, nome: str):
        if nome not in ETAPAS:
            raise ValueError(f"Etapa desconhecida: {nome}. Esperadas: {ETAPAS}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.tempos[nome] = time.perf_counter() - t0

    @property
    def total(self) -> float:
        return sum(self.tempos.values())


@dataclass(frozen=True)
class Contexto:
    """Entradas e saída de uma execução do Stage A."""

    engine: str
    escala: str
    transacoes: list[str]
    clientes: list[str]
    saida: str

    @classmethod
    def do_manifesto(cls, engine: str, escala: str, manifesto: dict, saida: str):
        m = manifesto[escala]
        return cls(engine=engine, escala=escala, transacoes=m["transacoes"],
                   clientes=m["clientes"], saida=saida)


# ---------------------------------------------------------------------------
# Especificação semântica — referência única para as quatro implementações
# ---------------------------------------------------------------------------

ESPECIFICACAO = """
LIMPEZA
    - descarta transações com valor <= 0 ou data_hora nula
    - remove duplicatas por transacao_id (mantém a primeira ocorrência na
      ordem de leitura — o gerador não produz duplicatas, então a escolha não
      afeta o resultado; a operação existe porque tem custo e é parte do que
      se mede)

PARÂMETROS GLOBAIS (derivados da tabela fato, após a limpeza)
    t_min = min(data_hora) em segundos epoch
    t_max = max(data_hora) em segundos epoch
    janela_dias = (t_max - t_min) / 86400
    metade = t_min + (t_max - t_min) // 2

AGREGAÇÃO (por cliente_id)
    n_tx           = contagem
    valor_total    = soma(valor)
    valor_medio    = média(valor)
    valor_std      = desvio amostral(valor), ddof=1; 0.0 se n_tx < 2
    valor_max      = máximo(valor)
    valor_min      = mínimo(valor)
    n_categorias   = contagem distinta(categoria)
    n_canais       = contagem distinta(canal)
    recencia_dias  = (t_max - máximo(data_hora)) / 86400
    dias_ativo     = (máximo(data_hora) - mínimo(data_hora)) / 86400
    pct_negada     = contagem(status = 'negada') / n_tx

JANELA
    bucket     = (data_hora_epoch - t_min) // 604800
    max_tx_7d  = máximo, sobre os buckets do cliente, da contagem de transações
    esperado   = n_tx / (janela_dias / 7)
    burstiness = max_tx_7d / esperado ; 0.0 se esperado = 0

PERFIL DE CANAL E CATEGORIA
    pct_<canal>     = contagem(canal) / n_tx
    cat_<categoria> = contagem(categoria) / n_tx
    delta_pct_web   = pct_web(data_hora > metade) - pct_web(data_hora <= metade)
                      metade vazia contribui 0.0
    delta_ticket    = média(valor | data_hora > metade)
                      / média(valor | data_hora <= metade)
                      1.0 se qualquer metade estiver vazia ou o denominador = 0

JOIN
    junção interna com a dimensão `clientes` por cliente_id

DERIVADAS
    razao_gasto_renda  = valor_total / renda_mensal
    razao_ticket_renda = valor_medio / renda_mensal
    razao_limite_renda = limite_credito / renda_mensal
    uso_limite         = valor_total / limite_credito
    (0.0 quando o denominador é 0)

ESCRITA
    Parquet único, ordenado por cliente_id, colunas na ordem de COLUNAS
"""
