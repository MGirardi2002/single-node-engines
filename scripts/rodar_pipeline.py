"""
Executa o Stage A de uma engine numa escala e reporta o tempo por etapa.

Este script roda o pipeline diretamente, sem instrumentação de memória — é a
ferramenta de desenvolvimento. As medições oficiais do benchmark passam pelo
harness (`src/benchmark/`), que isola cada execução em subprocesso e amostra o
RSS do processo e de seus filhos.

Uso:
    python scripts/rodar_pipeline.py --engine pandas --escala 100mb
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gerador import config as cfg  # noqa: E402
from src.pipelines import base  # noqa: E402

ENGINES = {
    "pandas": "src.pipelines.pandas_",
    "polars": "src.pipelines.polars_",
    "duckdb": "src.pipelines.duckdb_",
    "spark": "src.pipelines.spark_",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Executa o Stage A de uma engine")
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--escala", required=True, choices=list(cfg.ESCALAS))
    ap.add_argument("--saida", default=None,
                    help="caminho do Parquet de features (padrão: derivado)")
    ap.add_argument("--staged", dest="staged", action="store_true", default=None,
                    help="força materialização por etapa (detalha, mas penaliza "
                         "engines preguiçosas)")
    ap.add_argument("--no-staged", dest="staged", action="store_false",
                    help="modo idiomático: uma única materialização")
    args = ap.parse_args()

    with open(cfg.MANIFESTO, encoding="utf-8") as fp:
        manifesto = json.load(fp)
    if args.escala not in manifesto:
        sys.exit(f"Escala '{args.escala}' não gerada. Rode o gerador primeiro.")

    saida = args.saida or (
        f"{cfg.DIR_DADOS}/features/{args.escala}/{args.engine}.parquet"
    )
    os.makedirs(os.path.dirname(saida), exist_ok=True)

    ctx = base.Contexto.do_manifesto(args.engine, args.escala, manifesto, saida)
    modulo = importlib.import_module(ENGINES[args.engine])

    print(f"engine={args.engine}  escala={args.escala}  "
          f"chunks={len(ctx.transacoes)}")

    aceita_staged = "staged" in inspect.signature(modulo.executar).parameters
    if args.staged is not None and aceita_staged:
        cron = modulo.executar(ctx, staged=args.staged)
    else:
        cron = modulo.executar(ctx)

    print(f"\n{'etapa':<14}{'segundos':>10}{'%':>8}")
    print("-" * 32)
    for etapa in base.ETAPAS:
        t = cron.tempos.get(etapa)
        if t is None:
            continue
        print(f"{etapa:<14}{t:>10.3f}{100 * t / cron.total:>7.1f}%")
    print("-" * 32)
    print(f"{'TOTAL':<14}{cron.total:>10.3f}")

    if not cron.materializado:
        print("\n  ATENÇÃO: execução em modo idiomático (avaliação preguiçosa).")
        print("  O TOTAL é válido e comparável entre engines, mas a divisão por")
        print("  etapa NÃO é: o trabalho todo aparece na etapa que dispara a")
        print("  materialização. Use --staged para obter o detalhamento — ao")
        print("  custo de impedir a otimização entre etapas.")
    # O Spark escreve um diretório de part-files; as demais, um arquivo único.
    if os.path.isdir(saida):
        tamanho = sum(os.path.getsize(os.path.join(saida, f))
                      for f in os.listdir(saida))
    else:
        tamanho = os.path.getsize(saida)
    print(f"\nsaída: {saida} ({tamanho / 1024**2:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
