"""
Teste de equivalência entre engines.

Pré-condição de todo o benchmark: se as engines não produziram a MESMA matriz
de features, comparar seus tempos de execução não significa nada — uma pode
estar sendo rápida por estar fazendo menos, ou coisa diferente.

Compara cada engine contra a implementação de referência (Pandas), verificando
schema, número de linhas, chaves e valores. A tolerância numérica é folgada de
propósito: somas de milhões de floats acumulam em ordens distintas conforme o
plano de execução, e exigir igualdade exata reprovaria engines corretas. Ver
`base.RTOL`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines import base

REFERENCIA = "pandas"


def comparar(caminho_ref: str, caminho_alvo: str) -> tuple[bool, list[str]]:
    """Retorna (equivalente, lista de divergências)."""
    ref = pd.read_parquet(caminho_ref)
    alvo = pd.read_parquet(caminho_alvo)
    problemas: list[str] = []

    # -- schema ------------------------------------------------------------
    if list(alvo.columns) != list(base.COLUNAS):
        faltando = set(base.COLUNAS) - set(alvo.columns)
        sobrando = set(alvo.columns) - set(base.COLUNAS)
        if faltando or sobrando:
            problemas.append(
                f"colunas divergentes — faltando: {sorted(faltando) or '—'}, "
                f"inesperadas: {sorted(sobrando) or '—'}")
            return False, problemas
        problemas.append("ordem das colunas difere do contrato")

    # -- cardinalidade e chaves -------------------------------------------
    if len(ref) != len(alvo):
        problemas.append(f"nº de linhas: referência {len(ref):,} x alvo {len(alvo):,}")
        return False, problemas

    ref = ref.sort_values("cliente_id").reset_index(drop=True)
    alvo = alvo.sort_values("cliente_id").reset_index(drop=True)

    if not ref["cliente_id"].equals(alvo["cliente_id"]):
        n = int((ref["cliente_id"].to_numpy() != alvo["cliente_id"].to_numpy()).sum())
        problemas.append(f"conjunto de cliente_id difere em {n:,} posições")
        return False, problemas

    # -- valores -----------------------------------------------------------
    for col in base.COLUNAS:
        if col == "cliente_id":
            continue
        a = ref[col].to_numpy(dtype="float64")
        b = alvo[col].to_numpy(dtype="float64")

        nan_a, nan_b = np.isnan(a), np.isnan(b)
        if not np.array_equal(nan_a, nan_b):
            problemas.append(f"{col}: posições nulas divergem "
                             f"({int(nan_a.sum())} x {int(nan_b.sum())})")
            continue

        ok = np.isclose(a, b, rtol=base.RTOL, atol=base.ATOL, equal_nan=True)
        if not ok.all():
            i = int(np.flatnonzero(~ok)[0])
            dif = int((~ok).sum())
            problemas.append(
                f"{col}: {dif:,} valor(es) divergem — ex. linha {i} "
                f"(cliente {ref['cliente_id'][i]}): "
                f"referência {a[i]!r} x alvo {b[i]!r}")

    return not problemas, problemas


def main() -> int:
    import argparse
    import json
    import os

    from src.gerador import config as cfg

    ap = argparse.ArgumentParser(description="Teste de equivalência entre engines")
    ap.add_argument("--escala", required=True, choices=list(cfg.ESCALAS))
    args = ap.parse_args()

    base_dir = f"{cfg.DIR_DADOS}/features/{args.escala}"
    ref = f"{base_dir}/{REFERENCIA}.parquet"
    if not os.path.exists(ref):
        raise SystemExit(f"Referência ausente: {ref}. Rode a engine "
                         f"'{REFERENCIA}' primeiro.")

    alvos = sorted(f for f in os.listdir(base_dir)
                   if f.endswith(".parquet") and f != f"{REFERENCIA}.parquet")
    if not alvos:
        raise SystemExit(f"Nenhuma outra engine encontrada em {base_dir}.")

    print(f"Referência: {REFERENCIA} (escala {args.escala})")
    print(f"Tolerância: rtol={base.RTOL}, atol={base.ATOL}\n")

    tudo_ok = True
    for arquivo in alvos:
        engine = arquivo.removesuffix(".parquet")
        ok, problemas = comparar(ref, f"{base_dir}/{arquivo}")
        print(f"{'[OK]' if ok else '[X] '} {engine}")
        for p in problemas:
            print(f"       {p}")
        tudo_ok &= ok

    print("\n" + ("EQUIVALENTES — a comparação de desempenho é válida."
                  if tudo_ok else
                  "DIVERGÊNCIA — corrigir antes de comparar tempos."))
    return 0 if tudo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
