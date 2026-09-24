"""
Executa UMA combinação (engine, escala, modo) e grava o resultado em JSON.

Roda sempre como subprocesso, lançado pelo `runner`. Dois motivos:

  1. Isolamento de falha — um estouro de memória do Pandas na maior escala
     encerra apenas este processo; o runner registra `status=OOM` e segue.
     O ponto de falha de cada engine é um dos resultados esperados do
     trabalho, então ele precisa ser capturável, não catastrófico.

  2. Isolamento de medição — cada execução parte de um processo limpo, sem
     memória residual, cache de import ou JVM remanescente da anterior.

O resultado sai em arquivo, não em stdout: as engines escrevem log próprio
(o Spark é especialmente verboso) e misturar isso com a saída estruturada
tornaria o parsing frágil.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.gerador import config as cfg  # noqa: E402
from src.pipelines import base  # noqa: E402

ENGINES = {
    "pandas": "src.pipelines.pandas_",
    "polars": "src.pipelines.polars_",
    "duckdb": "src.pipelines.duckdb_",
    "spark": "src.pipelines.spark_",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--escala", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--resultado", required=True,
                    help="arquivo JSON onde gravar tempos e status")
    ap.add_argument("--staged", action="store_true")
    args = ap.parse_args()

    resultado: dict = {"status": "erro"}
    try:
        with open(cfg.MANIFESTO, encoding="utf-8") as fp:
            manifesto = json.load(fp)

        os.makedirs(os.path.dirname(args.saida), exist_ok=True)
        ctx = base.Contexto.do_manifesto(
            args.engine, args.escala, manifesto, args.saida
        )
        modulo = importlib.import_module(ENGINES[args.engine])

        if "staged" in inspect.signature(modulo.executar).parameters:
            cron = modulo.executar(ctx, staged=args.staged)
        else:
            cron = modulo.executar(ctx)

        resultado = {
            "status": "ok",
            "tempos": cron.tempos,
            "total": cron.total,
            "materializado": cron.materializado,
            "n_chunks": len(ctx.transacoes),
        }
    except MemoryError:
        # MemoryError capturável é raro: no Linux o OOM killer normalmente
        # encerra o processo com SIGKILL antes disso, e quem detecta é o
        # runner, pelo código de retorno.
        resultado = {"status": "OOM", "erro": "MemoryError"}
    except Exception as exc:
        resultado = {"status": "erro", "erro": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc()}

    with open(args.resultado, "w", encoding="utf-8") as fp:
        json.dump(resultado, fp)

    return 0 if resultado["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
