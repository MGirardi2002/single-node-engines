"""
Parâmetros do gerador sintético.

A base é gerada em *chunks* independentes de clientes. Uma escala é definida
como "os primeiros N chunks", o que garante que a base de 100 MB seja um
subconjunto genuíno da base de 1,6 GB (nested scaling) — condição para que a
comparação entre escalas isole o efeito do volume.

Cada chunk usa uma seed derivada de forma determinística do seu índice, de modo
que o conteúdo do chunk 7 é sempre o mesmo, independentemente da escala que o
inclui.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Escala
# ---------------------------------------------------------------------------

# Clientes por chunk. Calibrado empiricamente em 13/09/2026: com ~100
# transações/cliente o Parquet fica em ~18,2 bytes/linha, de modo que 57k
# clientes (~5,8M transações) produzem ~100 MB. Reconferir com
# `python -m src.gerador.gerar --calibrar` se o esquema mudar.
CHUNK_CLIENTES = 57_000

# Média de transações por cliente ao longo da janela observada.
TX_POR_CLIENTE = 100

# Quantidade de chunks por escala nominal (1 chunk ~= 110 MB).
#
# Progressão geométrica (o volume dobra a cada escala), decisão de 14/09. Dobrar
# é o padrão em estudos de escalabilidade: permite ler direto se o custo cresce
# proporcionalmente ao volume, e os pontos ficam igualmente espaçados em escala
# logarítmica. Com o limite de 8,5 GB por execução, as engines que operam só em
# memória devem falhar entre 400 e 800 MB, e as duas maiores escalas cobrem a
# faixa em que só engines com processamento out-of-core (DuckDB, Spark) seguem.
ESCALAS: dict[str, int] = {
    "100mb": 1,
    "200mb": 2,
    "400mb": 4,
    "800mb": 8,
    "1600mb": 16,
}

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

SEED_BASE = 42

# Espaçamento entre os ids de transação de chunks distintos. Precisa ser maior
# que o número máximo de transações de um chunk para evitar colisão de ids.
STRIDE_TX_ID = 100_000_000


def seed_do_chunk(chunk_id: int) -> int:
    """Seed determinística por chunk — garante o nested scaling."""
    return SEED_BASE * 1_000 + chunk_id


# ---------------------------------------------------------------------------
# Janela temporal
# ---------------------------------------------------------------------------

DATA_INICIO = "2025-01-01"
DIAS_JANELA = 365

# ---------------------------------------------------------------------------
# Anomalias
# ---------------------------------------------------------------------------

PCT_ANOMALIA = 0.02

# Os quatro tipos têm peso igual. Note que nenhum deles é um outlier univariado
# grosseiro: todos exigem feature engineering (agregação temporal, razão com a
# renda, ou combinação de atributos) para se tornarem detectáveis. Essa é uma
# decisão deliberada — anomalias triviais fariam o Z-Score empatar com
# Isolation Forest e LOF, esvaziando o eixo de ML do trabalho.
TIPOS_ANOMALIA = (
    "rajada",  # concentração temporal anormal, volume total normal
    "ticket_desproporcional",  # alto em relação à renda, normal em absoluto
    "migracao_canal",  # mudança abrupta de canal + elevação de valor
    "multivariada",  # nenhum atributo extremo, a combinação é que é rara
)

# ---------------------------------------------------------------------------
# Subpopulações legítimas
# ---------------------------------------------------------------------------

# Variável latente (não exposta em nenhuma tabela) que dá heterogeneidade à
# população normal. Cada perfil tem UMA característica extrema sem ser anômalo:
# o varejista transaciona muitíssimo, o digital nunca usa canal presencial, o
# sazonal concentra compras em poucas janelas, o premium tem ticket altíssimo.
#
# Sem isso, "ter alguma feature extrema" equivale a "ser anômalo", e um
# Z-Score univariado (max |z|) resolve o problema sozinho — medido: PR-AUC
# 0.624 do Z-Score contra 0.589 do Isolation Forest. Os perfis produzem os
# falsos positivos que só um modelo multivariado consegue descartar, que é a
# condição para o eixo de ML do trabalho ter conteúdo.
PERFIS = ("padrao", "varejista", "digital", "sazonal", "premium_raro")
PERFIS_P = (0.85, 0.05, 0.04, 0.04, 0.02)

# Multiplicadores (frequência de transações, ticket médio) por perfil.
PERFIL_FREQ = {"padrao": 1.0, "varejista": 3.0, "digital": 1.2, "sazonal": 1.0,
               "premium_raro": 0.3}
PERFIL_TICKET = {"padrao": 1.0, "varejista": 0.3, "digital": 0.9, "sazonal": 1.1,
                 "premium_raro": 4.0}

# ---------------------------------------------------------------------------
# Cestas de consumo
# ---------------------------------------------------------------------------

# Cada cliente tem uma cesta latente: 5 categorias "núcleo" que concentram a
# maior parte do gasto, e o restante distribuído entre as demais. Isso cria
# CORRELAÇÃO entre categorias na população normal — quem gasta em educação
# também gasta em livraria, quem gasta em combustível também gasta em seguros.
#
# É esse acoplamento que dá à anomalia 'multivariada' um mecanismo honesto:
# ela mistura duas cestas distantes, produzindo um vetor de participação por
# categoria improvável SEM que nenhuma participação isolada seja extrema.
# Um detector univariado (max |z| por coluna) é cego para esse padrão por
# construção; Isolation Forest e LOF, não.
#
# Índices referentes à tupla CATEGORIAS acima.
CESTAS: dict[str, tuple[int, ...]] = {
    "familia":         (0, 3, 14, 13, 10),   # supermercado, farmácia, casa, pet, saúde
    "jovem_urbano":    (1, 8, 6, 12, 15),    # restaurante, lazer, transporte, assinatura, beleza
    "motorista":       (2, 6, 19, 7, 1),     # combustível, transporte, seguros, serviços, restaurante
    "estudante":       (9, 16, 12, 6, 1),    # educação, livraria, assinatura, transporte, restaurante
    "casa_propria":    (14, 5, 7, 19, 0),    # casa, eletrônicos, serviços, seguros, supermercado
    "viajante":        (11, 8, 1, 4, 6),     # viagem, lazer, restaurante, vestuário, transporte
    "bem_estar":       (10, 17, 3, 15, 0),   # saúde, esporte, farmácia, beleza, supermercado
    "tecnologia":      (5, 12, 18, 16, 7),   # eletrônicos, assinatura, telefonia, livraria, serviços
}

# Fração do gasto que vai para as categorias-núcleo da cesta.
PESO_NUCLEO = 0.80

# Pares de cestas sem sobreposição de núcleo — a mistura de um par destes é o
# que caracteriza a anomalia multivariada.
PARES_DISTANTES = (
    ("familia", "tecnologia"),
    ("estudante", "casa_propria"),
    ("motorista", "bem_estar"),
    ("viajante", "familia"),
)

# ---------------------------------------------------------------------------
# Domínios categóricos
# ---------------------------------------------------------------------------

# Mantidos como string (não category) no Parquet: o custo de manipular strings
# de cardinalidade média/alta é justamente uma das dimensões medidas pelo
# benchmark, e pré-otimizar isso apagaria uma diferença real entre as engines.

CATEGORIAS = (
    "supermercado", "restaurante", "combustivel", "farmacia", "vestuario",
    "eletronicos", "transporte", "servicos", "lazer", "educacao",
    "saude", "viagem", "assinatura", "pet", "casa",
    "beleza", "livraria", "esporte", "telefonia", "seguros",
)

CANAIS = ("pos", "app", "web")

REGIOES = ("Sul", "Sudeste", "Centro-Oeste", "Nordeste", "Norte")
REGIOES_P = (0.15, 0.45, 0.12, 0.20, 0.08)

TIPOS_CLIENTE = ("Bronze", "Prata", "Ouro", "Platina")
TIPOS_CLIENTE_P = (0.45, 0.30, 0.18, 0.07)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

# Os dados NÃO devem ficar no diretório do repositório quando este está sob
# /mnt/c: o driver 9p do WSL2 torna o I/O em /mnt/c ordens de grandeza mais
# lento que o sistema de arquivos nativo, o que contaminaria todas as medições
# de leitura. Defina TCC_DATA_DIR apontando para um caminho dentro do WSL
# (ex.: ~/tcc-data). O código continua rodando no Windows sem a variável.
DIR_DADOS = os.environ.get("TCC_DATA_DIR", "data")

DIR_RAW = f"{DIR_DADOS}/raw"
MANIFESTO = f"{DIR_DADOS}/escalas.json"


def dir_do_chunk(chunk_id: int, base: str = DIR_RAW) -> str:
    return f"{base}/chunk={chunk_id:04d}"
