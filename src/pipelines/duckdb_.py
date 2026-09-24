"""
Stage A em DuckDB — idiomático: SQL puro sobre `read_parquet`, sem trazer
dados para o Python em momento algum. A escrita final sai por `COPY ... TO`.

Assim como em Polars, `staged=True` materializa cada etapa numa tabela
intermediária para permitir o detalhamento por etapa; `staged=False` deixa tudo
como CTEs numa única consulta, que é como se escreveria de verdade e é o número
justo para comparação entre engines.
"""

from __future__ import annotations

import os

import duckdb

from src.gerador import config as cfg

from . import base

# Orçamento de memória informado ao DuckDB. Por padrão ele se permite usar 80%
# da RAM do sistema (7,7 GiB neste ambiente) sem saber que o harness encerra a
# execução ao cruzar 95% de 8,5 GB. Informar um orçamento abaixo da trava
# permite que ele transborde para disco antes de ser encerrado — é o mesmo papel
# do `spark.driver.memory` no Spark. Medido em 200 MB: pico de 7,13 GB sem
# limite contra 6,07 GB com 6 GB, a um custo de 8,6 s -> 11,7 s.
MEMORY_LIMIT: str | None = "6GB"

# Forma de deduplicar por transacao_id: "janela" (row_number sobre partições)
# ou "distinct_on" (DISTINCT ON, por tabela hash). Medido em 100 e 200 MB, a
# janela usou menos memória (2,0 GB contra 4,4 GB em 100 MB) e foi mais rápida;
# as duas produzem a mesma matriz.
DEDUP = "janela"

# Diretório de transbordo. O padrão do DuckDB é ".tmp" relativo à pasta atual,
# que no harness é o repositório em /mnt/c: gravar lá seria lento (driver 9p
# do WSL) e encheria a pasta do projeto de arquivos temporários.
DIR_TEMP = f"{cfg.DIR_DADOS}/_duckdb_tmp"


def _lista_sql(paths: list[str]) -> str:
    itens = ", ".join(f"'{p}'" for p in paths)
    return f"[{itens}]"


def _colunas_perfil() -> str:
    """Participações por canal e categoria, como médias de indicadores."""
    partes = [
        f"avg(CASE WHEN canal = '{c}' THEN 1.0 ELSE 0.0 END) AS pct_{c}"
        for c in base.CANAIS
    ]
    partes += [
        f"avg(CASE WHEN categoria = '{c}' THEN 1.0 ELSE 0.0 END) AS cat_{c}"
        for c in base.CATEGORIAS
    ]
    return ",\n        ".join(partes)


def executar(ctx: base.Contexto, staged: bool = False) -> base.Cronometro:
    cron = base.Cronometro(materializado=staged)
    os.makedirs(DIR_TEMP, exist_ok=True)
    config = {"temp_directory": DIR_TEMP}
    if MEMORY_LIMIT:
        config["memory_limit"] = MEMORY_LIMIT
    con = duckdb.connect(config=config)

    # -- ingestão ----------------------------------------------------------
    with cron.etapa("ingestao"):
        con.execute(f"""
            CREATE OR REPLACE VIEW tx_raw AS
            SELECT * FROM read_parquet({_lista_sql(ctx.transacoes)})
        """)
        if staged:
            con.execute("CREATE OR REPLACE TABLE tx_mat AS SELECT * FROM tx_raw")
            fonte = "tx_mat"
        else:
            fonte = "tx_raw"

    # -- limpeza -----------------------------------------------------------
    with cron.etapa("limpeza"):
        # TABLE, e não VIEW, nos dois modos: a tabela fato limpa é consumida
        # duas vezes adiante (agregação e buckets da janela). Como VIEW, o
        # DuckDB releria e relimparia o Parquet a cada uso, e o mesmo valeria
        # de formas diferentes nas outras engines — uma variável não
        # controlada. Materializar uma vez em todas é a escolha equivalente.
        if DEDUP == "distinct_on":
            sql = f"""
                CREATE OR REPLACE TABLE tx AS
                SELECT DISTINCT ON (transacao_id)
                       *, CAST(epoch(data_hora) AS BIGINT) AS _seg
                FROM {fonte}
                WHERE valor > 0 AND data_hora IS NOT NULL
            """
        else:
            sql = f"""
                CREATE OR REPLACE TABLE tx AS
                SELECT * EXCLUDE (rn) FROM (
                    SELECT *,
                           row_number() OVER (PARTITION BY transacao_id) AS rn,
                           CAST(epoch(data_hora) AS BIGINT) AS _seg
                    FROM {fonte}
                    WHERE valor > 0 AND data_hora IS NOT NULL
                ) WHERE rn = 1
            """
        con.execute(sql)

    # -- agregação ---------------------------------------------------------
    with cron.etapa("agregacao"):
        t_min, t_max = con.execute(
            "SELECT min(_seg), max(_seg) FROM tx"
        ).fetchone()
        t_min, t_max = int(t_min), int(t_max)
        janela_dias = (t_max - t_min) / base.SEGUNDOS_POR_DIA
        metade = t_min + (t_max - t_min) // 2

        con.execute(f"""
            CREATE OR REPLACE {'TABLE' if staged else 'VIEW'} agg AS
            SELECT
                cliente_id,
                count(*)                              AS n_tx,
                sum(valor)                            AS valor_total,
                avg(valor)                            AS valor_medio,
                coalesce(stddev_samp(valor), {base.FILL_STD}) AS valor_std,
                max(valor)                            AS valor_max,
                min(valor)                            AS valor_min,
                count(DISTINCT categoria)             AS n_categorias,
                count(DISTINCT canal)                 AS n_canais,
                avg(CASE WHEN status = 'negada' THEN 1.0 ELSE 0.0 END)
                                                      AS pct_negada,
                ({t_max} - max(_seg)) / {base.SEGUNDOS_POR_DIA}.0
                                                      AS recencia_dias,
                (max(_seg) - min(_seg)) / {base.SEGUNDOS_POR_DIA}.0
                                                      AS dias_ativo,
                {_colunas_perfil()},
                avg(CASE WHEN _seg >  {metade} AND canal = 'web' THEN 1.0
                         WHEN _seg >  {metade} THEN 0.0 END) AS _web_novo,
                avg(CASE WHEN _seg <= {metade} AND canal = 'web' THEN 1.0
                         WHEN _seg <= {metade} THEN 0.0 END) AS _web_velho,
                avg(CASE WHEN _seg >  {metade} THEN valor END) AS _v_novo,
                avg(CASE WHEN _seg <= {metade} THEN valor END) AS _v_velho
            FROM tx
            GROUP BY cliente_id
        """)

    # -- janela ------------------------------------------------------------
    with cron.etapa("janela"):
        con.execute(f"""
            CREATE OR REPLACE {'TABLE' if staged else 'VIEW'} com_janela AS
            WITH buckets AS (
                SELECT cliente_id,
                       (_seg - {t_min}) // {base.SEGUNDOS_POR_BUCKET} AS b,
                       count(*) AS n
                FROM tx
                GROUP BY cliente_id, b
            ), pico AS (
                SELECT cliente_id, CAST(max(n) AS DOUBLE) AS max_tx_7d
                FROM buckets GROUP BY cliente_id
            )
            SELECT
                a.* EXCLUDE (_web_novo, _web_velho, _v_novo, _v_velho),
                p.max_tx_7d,
                CASE WHEN a.n_tx / ({janela_dias} / 7.0) > 0
                     THEN p.max_tx_7d / (a.n_tx / ({janela_dias} / 7.0))
                     ELSE 0.0 END AS burstiness,
                coalesce(a._web_novo, {base.FILL_DELTA_WEB})
                  - coalesce(a._web_velho, {base.FILL_DELTA_WEB})
                                                    AS delta_pct_web,
                CASE WHEN a._v_velho IS NOT NULL AND a._v_velho <> 0
                          AND a._v_novo IS NOT NULL
                     THEN a._v_novo / a._v_velho
                     ELSE {base.FILL_DELTA_TICKET} END AS delta_ticket
            FROM agg a JOIN pico p USING (cliente_id)
        """)

    # -- join --------------------------------------------------------------
    with cron.etapa("join"):
        con.execute(f"""
            CREATE OR REPLACE {'TABLE' if staged else 'VIEW'} juntado AS
            SELECT j.*, c.* EXCLUDE (cliente_id, regiao, tipo_cliente)
            FROM com_janela j
            JOIN read_parquet({_lista_sql(ctx.clientes)}) c USING (cliente_id)
        """)

    # -- derivadas + escrita ----------------------------------------------
    def _div(num: str, den: str, nome: str) -> str:
        return (f"CASE WHEN {den} <> 0 THEN {num} / {den} "
                f"ELSE {base.FILL_RAZAO} END AS {nome}")

    derivadas = ",\n                ".join([
        _div("valor_total", "renda_mensal", "razao_gasto_renda"),
        _div("valor_medio", "renda_mensal", "razao_ticket_renda"),
        _div("limite_credito", "renda_mensal", "razao_limite_renda"),
        _div("valor_total", "limite_credito", "uso_limite"),
    ])

    with cron.etapa("derivadas"):
        con.execute(f"""
            CREATE OR REPLACE {'TABLE' if staged else 'VIEW'} final AS
            SELECT * FROM (
                SELECT *, {derivadas} FROM juntado
            ) ORDER BY cliente_id
        """)

    with cron.etapa("escrita"):
        cols = ", ".join(base.COLUNAS)
        con.execute(f"""
            COPY (SELECT {cols} FROM final ORDER BY cliente_id)
            TO '{ctx.saida}' (FORMAT PARQUET)
        """)

    con.close()
    return cron
