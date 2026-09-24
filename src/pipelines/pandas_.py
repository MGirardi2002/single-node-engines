"""
Stage A em Pandas — implementação de referência.

É contra esta saída que as demais engines são comparadas no teste de
equivalência. Escrita de forma idiomática em Pandas: tudo em memória, eager,
`groupby().agg()` e `crosstab`.

Ver `base.ESPECIFICACAO` para a semântica exata de cada feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import base


def _segundos(col: pd.Series) -> np.ndarray:
    """
    datetime64 -> segundos epoch (int64).

    Passa por `datetime64[s]` em vez de dividir nanossegundos: o Parquet pode
    devolver a coluna em resolução de us ou ns conforme quem a escreveu, e a
    divisão daria resultados diferentes nos dois casos.
    """
    return col.to_numpy().astype("datetime64[s]").astype("int64")


def executar(ctx: base.Contexto) -> base.Cronometro:
    cron = base.Cronometro()

    # -- ingestão ----------------------------------------------------------
    with cron.etapa("ingestao"):
        tx = pd.concat(
            [pd.read_parquet(p) for p in ctx.transacoes], ignore_index=True
        )

    # -- limpeza -----------------------------------------------------------
    with cron.etapa("limpeza"):
        tx = tx[(tx["valor"] > 0) & tx["data_hora"].notna()]
        tx = tx.drop_duplicates(subset="transacao_id", keep="first")
        tx = tx.reset_index(drop=True)

    # -- agregação ---------------------------------------------------------
    with cron.etapa("agregacao"):
        seg = _segundos(tx["data_hora"])
        t_min, t_max = int(seg.min()), int(seg.max())
        janela_dias = (t_max - t_min) / base.SEGUNDOS_POR_DIA
        metade = t_min + (t_max - t_min) // 2

        tx["_seg"] = seg
        tx["_negada"] = (tx["status"] == "negada").astype("float64")

        agg = tx.groupby("cliente_id").agg(
            n_tx=("transacao_id", "count"),
            valor_total=("valor", "sum"),
            valor_medio=("valor", "mean"),
            valor_std=("valor", "std"),  # ddof=1 por padrão
            valor_max=("valor", "max"),
            valor_min=("valor", "min"),
            n_categorias=("categoria", "nunique"),
            n_canais=("canal", "nunique"),
            pct_negada=("_negada", "mean"),
            _ultima=("_seg", "max"),
            _primeira=("_seg", "min"),
        )
        agg["valor_std"] = agg["valor_std"].fillna(base.FILL_STD)
        agg["recencia_dias"] = (t_max - agg["_ultima"]) / base.SEGUNDOS_POR_DIA
        agg["dias_ativo"] = (
            (agg["_ultima"] - agg["_primeira"]) / base.SEGUNDOS_POR_DIA
        )
        agg = agg.drop(columns=["_ultima", "_primeira"])

    # -- janela ------------------------------------------------------------
    with cron.etapa("janela"):
        bucket = (tx["_seg"] - t_min) // base.SEGUNDOS_POR_BUCKET
        por_bucket = tx.groupby(["cliente_id", bucket]).size()
        agg["max_tx_7d"] = por_bucket.groupby(level=0).max().astype("float64")

        esperado = agg["n_tx"] / (janela_dias / 7.0)
        agg["burstiness"] = np.where(
            esperado > 0, agg["max_tx_7d"] / esperado, 0.0
        )

        # Perfis de canal e categoria.
        canal = pd.crosstab(tx["cliente_id"], tx["canal"], normalize="index")
        for c in base.CANAIS:
            agg[f"pct_{c}"] = canal[c] if c in canal else 0.0

        cat = pd.crosstab(tx["cliente_id"], tx["categoria"], normalize="index")
        for c in base.CATEGORIAS:
            agg[f"cat_{c}"] = cat[c] if c in cat else 0.0

        # Variação entre as duas metades da janela — separa uma migração de
        # canal de um cliente que sempre foi digital.
        recente = tx["_seg"] > metade
        meias = (
            pd.DataFrame({
                "cliente_id": tx["cliente_id"],
                "recente": recente,
                "valor": tx["valor"],
                "web": (tx["canal"] == "web").astype("float64"),
            })
            .groupby(["cliente_id", "recente"])
            .agg(valor=("valor", "mean"), web=("web", "mean"))
            .unstack()
        )
        web_novo = meias.get(("web", True), pd.Series(0.0, index=agg.index)).fillna(0.0)
        web_velho = meias.get(("web", False), pd.Series(0.0, index=agg.index)).fillna(0.0)
        agg["delta_pct_web"] = (web_novo - web_velho).reindex(
            agg.index, fill_value=base.FILL_DELTA_WEB
        )

        v_novo = meias.get(("valor", True), pd.Series(np.nan, index=agg.index))
        v_velho = meias.get(("valor", False), pd.Series(np.nan, index=agg.index))
        razao = (v_novo / v_velho.replace(0.0, np.nan)).reindex(agg.index)
        agg["delta_ticket"] = razao.fillna(base.FILL_DELTA_TICKET)

    # -- join --------------------------------------------------------------
    with cron.etapa("join"):
        clientes = pd.concat(
            [pd.read_parquet(p) for p in ctx.clientes], ignore_index=True
        ).set_index("cliente_id")
        df = agg.join(clientes, how="inner")

    # -- derivadas ---------------------------------------------------------
    with cron.etapa("derivadas"):
        def _div(a: pd.Series, b: pd.Series) -> np.ndarray:
            return np.where(b != 0, a / b.where(b != 0, 1.0), base.FILL_RAZAO)

        df["razao_gasto_renda"] = _div(df["valor_total"], df["renda_mensal"])
        df["razao_ticket_renda"] = _div(df["valor_medio"], df["renda_mensal"])
        df["razao_limite_renda"] = _div(df["limite_credito"], df["renda_mensal"])
        df["uso_limite"] = _div(df["valor_total"], df["limite_credito"])

        df = df.reset_index().sort_values("cliente_id")
        df = df[list(base.COLUNAS)]

    # -- escrita -----------------------------------------------------------
    with cron.etapa("escrita"):
        df.to_parquet(ctx.saida, index=False)

    return cron
