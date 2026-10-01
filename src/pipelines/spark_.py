"""
Stage A em PySpark — DataFrame API, `local[*]`.

Sobre a medição por etapa
-------------------------
O Spark é preguiçoso e, diferente de Polars e DuckDB, forçar a materialização
entre etapas exige `persist()` — o que altera o próprio perfil de memória que o
benchmark pretende medir. Por isso:

  staged=False  (padrão)  um único plano, materializado na escrita. É o número
                honesto para comparar com as demais engines, e é como um job
                real seria escrito.

  staged=True   persiste e conta ao fim de cada etapa. Fornece o detalhamento,
                mas os tempos e o pico de memória NÃO são comparáveis aos das
                outras engines — serve para entender onde o tempo é gasto
                dentro do Spark, não para a comparação entre engines.

O `spark.stop()` no final é obrigatório: sem ele a JVM permanece viva e o
harness contabilizaria sua memória na execução seguinte.
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.gerador import config as cfg

from . import base


# Diretório de shuffle e spill.
#
# ESTA É A CONFIGURAÇÃO MAIS IMPORTANTE DESTE ARQUIVO, e sua ausência invalidava
# as medições anteriores do Spark.
#
# O padrão de `spark.local.dir` é o diretório temporário do sistema. Neste
# ambiente, `/tmp` é um **tmpfs**, ou seja, memória RAM — verificado em 23/09
# com `findmnt`. Tudo que o Spark "transborda para disco" ia, na verdade, para a
# RAM; e como o swap está desligado, essas páginas não podiam sequer ser
# liberadas. O resultado é que o spill, que deveria ALIVIAR a pressão de
# memória, era contabilizado como consumo pelo harness — que mede a memória em
# uso no sistema.
#
# Isso inverte a leitura das medições de 14/09: reduzir o heap PIORAVA o
# consumo, porque heap menor produz mais spill, e mais spill significava mais
# RAM ocupada. Foi por isso que 400 MB com heap de 4g (7.786 MB) consumiu mais
# fora do heap do que 200 MB com heap de 5g (7.499 MB).
#
# É o mesmo defeito que o DuckDB tinha e que foi corrigido em 14/09 com
# `temp_directory`; aqui ele passou despercebido por mais tempo.
DIR_TEMP = f"{cfg.DIR_DADOS}/_spark_tmp"

# Heap da JVM do driver (em local[*] o driver é também o executor).
#
# Levantado por medição em 400 MB, DEPOIS da correção do DIR_TEMP acima:
#
#   heap | pico do sistema | fora do heap | tempo   | margem até a trava
#   -----|-----------------|--------------|---------|-------------------
#   4g   |     4.966 MB    |   ~870 MB    | 153,9 s |     3.303 MB
#   5g   |     5.975 MB    |   ~880 MB    | 133,8 s |     2.294 MB
#   6g   |     6.990 MB    |   ~850 MB    | 149,8 s |     1.279 MB
#
# (o ponto de 6g é a mediana de 3 repetições, com amplitude de 66 MB)
#
# Duas leituras importam. A memória fora do heap é CONSTANTE em ~870 MB — não é
# mais função da escala, como parecia antes da correção do spill. E o pico
# passou a ser limitado pelo heap, e não pelo volume de dados: o Spark deve
# caber em 800 MB e 1,6 GB, gastando mais tempo em disco em vez de estourar.
#
# Adotado 5g, e não 6g. O tempo é equivalente (a diferença de 16 s está perto
# da amplitude de 8 s entre repetições) e sobra 1 GB a mais de margem. O
# argumento decisivo é de simetria com o DuckDB: o `memory_limit=6GB` dele é o
# orçamento TOTAL; o equivalente aqui é heap + fora do heap, que com 5g dá
# ~6 GB. As duas engines ficam com o mesmo orçamento e transbordam para disco
# real além dele.
HEAP_PADRAO = "5g"

# Limites da memória FORA do heap, ilimitados por padrão:
#
#   * memória direta (NIO/Netty): o teto padrão de MaxDirectMemorySize é o
#     próprio -Xmx, o que liberaria outros 4 GB;
#   * arenas do malloc do glibc: até 8 por núcleo — com 10 núcleos, dezenas de
#     arenas de 64 MB que o processo retém em vez de devolver ao sistema.
#
# Medido em 400 MB: 7.786 -> 7.538 MB, ou seja, 248 MB (3%). O efeito é pequeno
# e está dentro da variação entre execuções — NÃO foi isto que resolveu o
# problema de memória do Spark, e sim o DIR_TEMP. Os limites são mantidos porque
# tornam o consumo previsível: sem eles, ele é função do número de núcleos da
# máquina, e não do orçamento definido pelo experimento.
DIRECT_PADRAO = "1g"
ARENAS_PADRAO = "4"


def _sessao() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    # Precisa estar no ambiente ANTES de a JVM subir: ela é lançada como
    # processo filho deste e herda o valor.
    os.environ.setdefault("MALLOC_ARENA_MAX",
                          os.environ.get("TCC_SPARK_ARENAS", ARENAS_PADRAO))
    heap = os.environ.get("TCC_SPARK_HEAP", HEAP_PADRAO)
    direta = os.environ.get("TCC_SPARK_DIRECT", DIRECT_PADRAO)
    os.makedirs(DIR_TEMP, exist_ok=True)
    spark = (
        SparkSession.builder
        .appName("tcc-stage-a")
        .master("local[*]")
        .config("spark.driver.memory", heap)
        .config("spark.local.dir", DIR_TEMP)
        # MaxMetaspaceSize: o Spark gera código em tempo de execução e cria
        # muitas classes; o metaspace é ilimitado por padrão.
        .config("spark.driver.extraJavaOptions",
                f"-XX:MaxDirectMemorySize={direta} -XX:MaxMetaspaceSize=512m")
        .config("spark.sql.shuffle.partitions", "24")
        # O gerador grava timestamps ingênuos cujo valor é o epoch em UTC, e é
        # assim que Pandas, Polars e DuckDB os interpretam. Sem fixar UTC aqui,
        # o Spark aplicaria o fuso local na conversão e produziria uma
        # `recencia_dias` deslocada, reprovando no teste de equivalência.
        .config("spark.sql.session.timeZone", "UTC")
        # Sem o marcador _SUCCESS o diretório de saída contém apenas Parquet,
        # e pode ser lido diretamente por pandas/pyarrow no teste de
        # equivalência.
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs",
                "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def executar(ctx: base.Contexto, staged: bool = False) -> base.Cronometro:
    cron = base.Cronometro(materializado=staged)
    spark = _sessao()
    persistidos: list[DataFrame] = []

    def marco(df: DataFrame) -> DataFrame:
        if not staged:
            return df
        df.persist()
        df.count()
        while persistidos:
            persistidos.pop().unpersist()
        persistidos.append(df)
        return df

    try:
        # -- ingestão ------------------------------------------------------
        with cron.etapa("ingestao"):
            tx = spark.read.parquet(*ctx.transacoes)
            tx = marco(tx)

        # -- limpeza -------------------------------------------------------
        with cron.etapa("limpeza"):
            tx = (tx
                  .filter((F.col("valor") > 0) & F.col("data_hora").isNotNull())
                  .dropDuplicates(["transacao_id"])
                  # TIMESTAMP_NTZ não admite cast direto para BIGINT no
                  # Spark 4; o passo por TIMESTAMP resolve, e com a sessão em
                  # UTC o resultado é o mesmo epoch das outras engines.
                  .withColumn("_seg",
                              F.col("data_hora").cast("timestamp").cast("long")))
            # persist() nos DOIS modos: `tx` é consumida duas vezes adiante
            # (agregação e buckets da janela) e, sem cache, o Spark reexecuta
            # leitura, filtro e deduplicação a cada ramo — medido: 123s sem
            # cache contra 46s com. Nenhum job real seria escrito sem isto, e
            # deixá-lo de fora seria comparar as engines com um espantalho.
            tx.persist()
            tx.count()

        # -- agregação -----------------------------------------------------
        with cron.etapa("agregacao"):
            lim = tx.select(
                F.min("_seg").alias("t_min"), F.max("_seg").alias("t_max")
            ).first()
            t_min, t_max = int(lim["t_min"]), int(lim["t_max"])
            janela_dias = (t_max - t_min) / base.SEGUNDOS_POR_DIA
            metade = t_min + (t_max - t_min) // 2

            novo = F.col("_seg") > metade
            velho = F.col("_seg") <= metade
            eh_web = F.col("canal") == "web"

            agg = tx.groupBy("cliente_id").agg(
                F.count(F.lit(1)).alias("n_tx"),
                F.sum("valor").alias("valor_total"),
                F.avg("valor").alias("valor_medio"),
                F.coalesce(F.stddev_samp("valor"), F.lit(base.FILL_STD))
                    .alias("valor_std"),
                F.max("valor").alias("valor_max"),
                F.min("valor").alias("valor_min"),
                F.countDistinct("categoria").alias("n_categorias"),
                F.countDistinct("canal").alias("n_canais"),
                F.avg(F.when(F.col("status") == "negada", 1.0).otherwise(0.0))
                    .alias("pct_negada"),
                F.max("_seg").alias("_ultima"),
                F.min("_seg").alias("_primeira"),
                *[F.avg(F.when(F.col("canal") == c, 1.0).otherwise(0.0))
                    .alias(f"pct_{c}") for c in base.CANAIS],
                *[F.avg(F.when(F.col("categoria") == c, 1.0).otherwise(0.0))
                    .alias(f"cat_{c}") for c in base.CATEGORIAS],
                F.avg(F.when(novo, F.when(eh_web, 1.0).otherwise(0.0)))
                    .alias("_web_novo"),
                F.avg(F.when(velho, F.when(eh_web, 1.0).otherwise(0.0)))
                    .alias("_web_velho"),
                F.avg(F.when(novo, F.col("valor"))).alias("_v_novo"),
                F.avg(F.when(velho, F.col("valor"))).alias("_v_velho"),
            ).withColumn(
                "recencia_dias",
                (F.lit(t_max) - F.col("_ultima")) / base.SEGUNDOS_POR_DIA,
            ).withColumn(
                "dias_ativo",
                (F.col("_ultima") - F.col("_primeira")) / base.SEGUNDOS_POR_DIA,
            ).drop("_ultima", "_primeira")
            agg = marco(agg)

        # -- janela --------------------------------------------------------
        with cron.etapa("janela"):
            pico = (tx
                    .withColumn("_b", ((F.col("_seg") - F.lit(t_min))
                                       / base.SEGUNDOS_POR_BUCKET).cast("long"))
                    .groupBy("cliente_id", "_b").agg(F.count(F.lit(1)).alias("_n"))
                    .groupBy("cliente_id")
                    .agg(F.max("_n").cast("double").alias("max_tx_7d")))

            esperado = F.col("n_tx") / F.lit(janela_dias / 7.0)
            agg = (agg.join(pico, on="cliente_id", how="inner")
                   .withColumn("burstiness",
                               F.when(esperado > 0, F.col("max_tx_7d") / esperado)
                                .otherwise(F.lit(0.0)))
                   .withColumn("delta_pct_web",
                               F.coalesce(F.col("_web_novo"),
                                          F.lit(base.FILL_DELTA_WEB))
                               - F.coalesce(F.col("_web_velho"),
                                            F.lit(base.FILL_DELTA_WEB)))
                   .withColumn("delta_ticket",
                               F.when(F.col("_v_velho").isNotNull()
                                      & (F.col("_v_velho") != 0)
                                      & F.col("_v_novo").isNotNull(),
                                      F.col("_v_novo") / F.col("_v_velho"))
                                .otherwise(F.lit(base.FILL_DELTA_TICKET)))
                   .drop("_web_novo", "_web_velho", "_v_novo", "_v_velho"))
            agg = marco(agg)

        # -- join ----------------------------------------------------------
        with cron.etapa("join"):
            clientes = (spark.read.parquet(*ctx.clientes)
                        .drop("regiao", "tipo_cliente"))
            df = agg.join(clientes, on="cliente_id", how="inner")
            df = marco(df)

        # -- derivadas -----------------------------------------------------
        with cron.etapa("derivadas"):
            def _div(num: str, den: str):
                return F.when(F.col(den) != 0, F.col(num) / F.col(den)) \
                        .otherwise(F.lit(base.FILL_RAZAO))

            df = (df
                  .withColumn("razao_gasto_renda",
                              _div("valor_total", "renda_mensal"))
                  .withColumn("razao_ticket_renda",
                              _div("valor_medio", "renda_mensal"))
                  .withColumn("razao_limite_renda",
                              _div("limite_credito", "renda_mensal"))
                  .withColumn("uso_limite",
                              _div("valor_total", "limite_credito"))
                  .select(*base.COLUNAS)
                  # Ordenação exigida pelo contrato (base, seção ESCRITA), que
                  # Pandas, Polars e DuckDB já cumpriam. Estava ausente aqui, e
                  # o teste de equivalência não a detectava porque ordenava os
                  # dois lados antes de comparar (corrigido em 28/09).
                  #
                  # Não é gratuita: no Spark, ordenar exige particionamento por
                  # faixa e, portanto, um shuffle. Sem ela, o Spark realizava
                  # menos trabalho que as demais engines, e os tempos medidos
                  # até 23/09 estão subestimados nessa proporção.
                  .orderBy("cliente_id"))
            df = marco(df)

        # -- escrita -------------------------------------------------------
        with cron.etapa("escrita"):
            df.write.mode("overwrite").parquet(ctx.saida)
    finally:
        # Imprescindível: a JVM sobreviveria ao processo Python e sua memória
        # seria contabilizada na execução seguinte do harness.
        spark.stop()

    return cron
