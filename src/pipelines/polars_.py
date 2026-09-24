"""
Stage A em Polars — idiomático: API lazy, `scan_parquet` e motor de streaming.

Sobre a medição por etapa
-------------------------
Polars constrói um plano e só o executa na materialização. Cronometrar a
construção do plano mediria essencialmente zero. Há dois modos:

  staged=True   força `.collect()` ao fim de cada etapa. Dá o detalhamento por
                etapa, mas impede o otimizador de fundir operações e obriga a
                tabela inteira a caber em memória — portanto **penaliza** o
                Polars e não deve ser usado para comparar totais nem nas
                escalas grandes.

  staged=False  uma única consulta lazy, executada em streaming na escrita. É
                o número justo para comparação entre engines, sem
                detalhamento por etapa (o trabalho aparece quase todo em
                `escrita`).

Sobre memória (decisão de 14/09)
--------------------------------
A versão anterior materializava a tabela fato limpa com `.collect()` também no
modo idiomático, para evitar recomputá-la nos dois ramos que a consomem. Na
escala de 2 GB isso exigiu mais de 10 GB de RAM e terminou em OOM: em memória,
as 126 milhões de transações ocupam cerca de 10 GB, contra 2,2 GB comprimidos
em disco.

Isso não refletia uma limitação do Polars, e sim da escolha de implementação:
DuckDB e Spark, com a mesma materialização, transbordam para disco quando falta
memória; o `.collect()` do Polars não. O Polars tem um motor de streaming feito
para dados maiores que a memória, e o uso idiomático para esse volume é
recorrer a ele. Por isso, no modo idiomático, a consulta roda em streaming.

Uma segunda tentativa confiou a reutilização da tabela limpa ao otimizador
(`comm_subplan_elim`). Também terminou em OOM: o plano mostrou um nó CACHE
guardando a tabela limpa inteira — já com as colunas indicadoras — para servir
aos dois ramos. A solução adotada é calcular os buckets da janela numa passada
de streaming separada e reaproveitar apenas o RESULTADO, que tem uma linha por
cliente. A tabela é lida duas vezes, mas nunca fica inteira em memória: é o
padrão para dados maiores que a memória — reaproveitar resultados pequenos,
não a tabela grande.

A terceira tentativa ainda estourou, e a medição isolada de cada operação
apontou a deduplicação: `unique(subset=...)` guarda a linha inteira de cada
chave e custou 8,4 GB em 500 MB — igual em streaming e em memória. Trocada por
`is_first_distinct`, que chegou a 2,6 GB. No Polars, a escolha da API para a
mesma operação pode triplicar a memória.

A quarta correção desfez um erro introduzido na própria migração para
streaming: os indicadores por categoria e canal tinham sido tirados de dentro
do `agg` e calculados antes, como colunas, na suposição (não medida) de que
isso ajudaria o streaming. As colunas largas eram materializadas e custavam
2,6x mais memória. Todas as expressões voltaram para dentro do `agg`.
"""

from __future__ import annotations

import os

import polars as pl

from src.gerador import config as cfg

from . import base

MOTOR = "streaming"

# Diretório de transbordo do motor de streaming.
#
# Mesmo motivo do `spark.local.dir` e do `temp_directory` do DuckDB: o padrão é
# o temporário do sistema, e neste ambiente `/tmp` é um tmpfs — memória RAM.
# Transbordar para lá não alivia a pressão de memória, ela a aumenta, e o
# harness contabiliza o spill como consumo da engine.
#
# O Polars não chegou a transbordar nas escalas medidas até aqui (pico de
# 1,5 GB em 200 MB), então nenhuma medição existente está comprometida. A
# configuração é preventiva: é a partir de 800 MB e 1,6 GB que o streaming
# passa a transbordar, e é exatamente onde a comparação entre engines importa.
DIR_TEMP = f"{cfg.DIR_DADOS}/_polars_tmp"


def executar(ctx: base.Contexto, staged: bool = False) -> base.Cronometro:
    cron = base.Cronometro(materializado=staged)
    os.makedirs(DIR_TEMP, exist_ok=True)
    os.environ.setdefault("POLARS_TEMP_DIR", DIR_TEMP)

    def marco(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Materializa a etapa quando em modo instrumentado."""
        return lf.collect().lazy() if staged else lf

    # -- ingestão ----------------------------------------------------------
    with cron.etapa("ingestao"):
        tx = pl.scan_parquet(ctx.transacoes)
        tx = marco(tx)

    # -- limpeza -----------------------------------------------------------
    with cron.etapa("limpeza"):
        validas = tx.filter(
            (pl.col("valor") > 0) & pl.col("data_hora").is_not_null()
        )
        # is_first_distinct e não unique(subset=...). Medido em 500 MB (32
        # milhões de linhas): unique(subset) atingiu 8,4 GB de pico, tanto em
        # streaming quanto em memória, porque guarda a linha inteira de cada
        # chave; is_first_distinct atingiu 2,6 GB, pois só precisa lembrar as
        # chaves já vistas. O resultado é o mesmo: mantém a primeira ocorrência.
        limpa = validas.filter(pl.col("transacao_id").is_first_distinct())
        tx = marco(limpa)

    # -- agregação ---------------------------------------------------------
    with cron.etapa("agregacao"):
        # Parâmetros globais: uma passada rápida, em streaming. É calculada
        # antes da deduplicação porque duplicatas têm a mesma data_hora e não
        # alteram mínimo nem máximo — e assim a passada dispensa a tabela hash
        # da deduplicação.
        lim = validas.select(
            pl.col("data_hora").dt.epoch(time_unit="s").min().alias("t_min"),
            pl.col("data_hora").dt.epoch(time_unit="s").max().alias("t_max"),
        ).collect(engine=MOTOR)
        t_min = int(lim["t_min"][0])
        t_max = int(lim["t_max"][0])
        janela_dias = (t_max - t_min) / base.SEGUNDOS_POR_DIA
        metade = t_min + (t_max - t_min) // 2

        seg = pl.col("data_hora").dt.epoch(time_unit="s")
        novo = seg > metade
        eh_web = (pl.col("canal") == "web").cast(pl.Float64)

        # Todas as expressões ficam DENTRO do agg, sem colunas intermediárias.
        # Uma versão anterior calculava os ~30 indicadores antes, com
        # with_columns, supondo que isso ajudaria o streaming. Medido em 200 MB,
        # só as 20 colunas de categoria assim custaram 2,89 GB, contra 1,13 GB
        # com as mesmas expressões dentro do agg: as colunas largas eram
        # materializadas inteiras.
        #
        # when/then sem otherwise gera nulo na outra metade, e mean() ignora
        # nulos: equivale a filtrar a metade antes da média.
        agg = tx.group_by("cliente_id").agg(
            pl.len().alias("n_tx"),
            pl.col("valor").sum().alias("valor_total"),
            pl.col("valor").mean().alias("valor_medio"),
            pl.col("valor").std().alias("valor_std"),
            pl.col("valor").max().alias("valor_max"),
            pl.col("valor").min().alias("valor_min"),
            # n_categorias e n_canais NÃO são calculados aqui — ver o
            # with_columns logo abaixo. Um `n_unique()` dentro do agg mantém um
            # conjunto de distintos POR GRUPO, e o custo é o número de grupos,
            # não o de valores distintos.
            (pl.col("status") == "negada").mean().alias("pct_negada"),
            seg.max().alias("_ultima"),
            seg.min().alias("_primeira"),
            *[(pl.col("canal") == c).mean().alias(f"pct_{c}")
              for c in base.CANAIS],
            *[(pl.col("categoria") == c).mean().alias(f"cat_{c}")
              for c in base.CATEGORIAS],
            pl.when(novo).then(eh_web).mean().alias("_web_novo"),
            pl.when(~novo).then(eh_web).mean().alias("_web_velho"),
            pl.when(novo).then(pl.col("valor")).mean().alias("_v_novo"),
            pl.when(~novo).then(pl.col("valor")).mean().alias("_v_velho"),
        ).with_columns(
            pl.col("valor_std").fill_null(base.FILL_STD),
            # Contagem de distintos derivada dos indicadores que o agg JÁ
            # produziu, em vez de um n_unique() com estado próprio.
            #
            # `cat_<c>` é a média do indicador (categoria == c) sobre as
            # transações do cliente, logo é > 0 exatamente quando o cliente tem
            # ao menos uma transação naquela categoria. Contar quantos
            # indicadores são positivos dá a mesma contagem distinta, sem
            # guardar conjunto nenhum.
            #
            # Medido em 800 MB: os dois n_unique() custavam 1.905 MB, contra
            # 576 MB em 400 MB — crescimento de 3,3x para o dobro de dados,
            # enquanto todo o resto da agregação cresce 1,5x. O custo não vinha
            # dos 20 e 3 valores distintos, e sim dos 456 mil conjuntos, um por
            # cliente. Era o que tirava o Polars das escalas grandes.
            pl.sum_horizontal(
                [(pl.col(f"cat_{c}") > 0).cast(pl.UInt32)
                 for c in base.CATEGORIAS]).alias("n_categorias"),
            pl.sum_horizontal(
                [(pl.col(f"pct_{c}") > 0).cast(pl.UInt32)
                 for c in base.CANAIS]).alias("n_canais"),
            ((t_max - pl.col("_ultima")) / base.SEGUNDOS_POR_DIA)
                .alias("recencia_dias"),
            ((pl.col("_ultima") - pl.col("_primeira")) / base.SEGUNDOS_POR_DIA)
                .alias("dias_ativo"),
        ).drop("_ultima", "_primeira")
        agg = marco(agg)

    # -- janela ------------------------------------------------------------
    with cron.etapa("janela"):
        bucket = ((pl.col("data_hora").dt.epoch(time_unit="s") - t_min)
                  // base.SEGUNDOS_POR_BUCKET)

        def pico_semanal(fonte: pl.LazyFrame) -> pl.LazyFrame:
            return (fonte
                    .select("cliente_id", bucket.alias("_b"))
                    .group_by("cliente_id", "_b")
                    .agg(pl.len().alias("_n"))
                    .group_by("cliente_id")
                    .agg(pl.col("_n").max().cast(pl.Float64).alias("max_tx_7d")))

        if staged:
            buckets = pico_semanal(tx)
        else:
            # Passada própria, a partir da tabela limpa ainda não expandida
            # com as colunas indicadoras, materializando só o resultado (uma
            # linha por cliente). Compartilhar `tx` entre os dois ramos fazia o
            # otimizador guardar a tabela inteira num CACHE em memória.
            buckets = pico_semanal(limpa).collect(engine=MOTOR).lazy()

        esperado = pl.col("n_tx") / (janela_dias / 7.0)
        agg = agg.join(buckets, on="cliente_id", how="inner").with_columns(
            pl.when(esperado > 0)
              .then(pl.col("max_tx_7d") / esperado)
              .otherwise(0.0)
              .alias("burstiness"),
            (pl.col("_web_novo").fill_null(base.FILL_DELTA_WEB)
             - pl.col("_web_velho").fill_null(base.FILL_DELTA_WEB))
                .alias("delta_pct_web"),
            pl.when((pl.col("_v_velho").is_not_null())
                    & (pl.col("_v_velho") != 0)
                    & (pl.col("_v_novo").is_not_null()))
              .then(pl.col("_v_novo") / pl.col("_v_velho"))
              .otherwise(base.FILL_DELTA_TICKET)
              .alias("delta_ticket"),
        ).drop("_web_novo", "_web_velho", "_v_novo", "_v_velho")
        agg = marco(agg)

    # -- join --------------------------------------------------------------
    with cron.etapa("join"):
        clientes = pl.scan_parquet(ctx.clientes)
        df = agg.join(clientes, on="cliente_id", how="inner")
        df = marco(df)

    # -- derivadas ---------------------------------------------------------
    with cron.etapa("derivadas"):
        def _div(num: str, den: str, nome: str) -> pl.Expr:
            return (pl.when(pl.col(den) != 0)
                      .then(pl.col(num) / pl.col(den))
                      .otherwise(base.FILL_RAZAO)
                      .alias(nome))

        df = (df.with_columns(
                  _div("valor_total", "renda_mensal", "razao_gasto_renda"),
                  _div("valor_medio", "renda_mensal", "razao_ticket_renda"),
                  _div("limite_credito", "renda_mensal", "razao_limite_renda"),
                  _div("valor_total", "limite_credito", "uso_limite"),
              )
              .sort("cliente_id")
              .select(base.COLUNAS))
        df = marco(df)

    # -- escrita -----------------------------------------------------------
    with cron.etapa("escrita"):
        if staged:
            df.collect().write_parquet(ctx.saida)
        else:
            # Toda a consulta executa aqui, em streaming, lote a lote.
            df.sink_parquet(ctx.saida, engine=MOTOR)

    return cron
