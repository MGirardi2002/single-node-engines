"""
Teste de equivalência entre engines.

Pré-condição de todo o benchmark: se as engines não produziram a MESMA matriz
de features, comparar seus tempos de execução não significa nada — uma pode
estar sendo rápida por estar fazendo menos, ou coisa diferente.

Compara cada engine contra uma implementação de referência, verificando schema,
ORDENAÇÃO das linhas, número de linhas, chaves e valores. A tolerância numérica
é folgada de propósito: somas de milhões de floats acumulam em ordens distintas
conforme o plano de execução, e exigir igualdade exata reprovaria engines
corretas. Ver `base.RTOL`.

Sobre a verificação de ordenação
--------------------------------
O contrato (`base`, seção ESCRITA) exige saída ordenada por `cliente_id`. Até
23/09 este teste ordenava AMBOS os lados antes de comparar, e por isso não
conseguia, por construção, detectar divergência de ordenação. O Spark passou
dois meses gravando a saída fora de ordem sem que o portão acusasse — e ordenar
exige shuffle, ou seja, ele pulava trabalho que as outras três faziam.

A ordenação passou a ser verificada ANTES de qualquer reordenação. A comparação
de valores continua exigindo os dois lados ordenados, senão uma divergência de
ordem se propagaria como divergência em todas as colunas.

Sobre a escolha da referência
-----------------------------
O Pandas é a referência natural por ser a implementação mais legível, mas ele
não conclui as escalas de 800 MB e 1,6 GB. Nessas, a referência precisa ser
outra — daí `--referencia`. A escolha não privilegia engine nenhuma: o teste
verifica concordância mútua, e qualquer uma serve de eixo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines import base

REFERENCIA = "pandas"


def _esta_ordenado(df: pd.DataFrame) -> bool:
    """Verifica a ordenação exigida pelo contrato: `cliente_id` crescente."""
    return bool(df["cliente_id"].is_monotonic_increasing)


def comparar(caminho_ref: str, caminho_alvo: str) -> tuple[bool, list[str]]:
    """Retorna (equivalente, lista de divergências)."""
    ref = pd.read_parquet(caminho_ref)
    alvo = pd.read_parquet(caminho_alvo)
    problemas: list[str] = []

    # -- ordenação ---------------------------------------------------------
    # Verificada ANTES de reordenar. Depois da reordenação a informação está
    # perdida — foi assim que a saída fora de ordem do Spark passou despercebida.
    if not _esta_ordenado(alvo):
        problemas.append(
            "saída NÃO ordenada por cliente_id, como o contrato exige "
            "(a engine está pulando a ordenação que as demais executam)")

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

    # A partir daqui os dois lados são reordenados: uma divergência de ordem
    # já foi registrada acima, e mantê-la aqui a propagaria como divergência
    # em todas as colunas, escondendo eventuais divergências de valor reais.
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
    ap.add_argument("--referencia", default=None,
                    help=f"engine usada como eixo da comparação (padrão: "
                         f"{REFERENCIA}; nas escalas em que ele não conclui, "
                         f"a primeira disponível)")
    args = ap.parse_args()

    base_dir = f"{cfg.DIR_DADOS}/features/{args.escala}"
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Escala sem matrizes: {base_dir}")

    presentes = sorted(f.removesuffix(".parquet") for f in os.listdir(base_dir)
                       if f.endswith(".parquet"))
    if len(presentes) < 2:
        raise SystemExit(f"São necessárias ao menos duas engines em "
                         f"{base_dir}; encontrada(s): {presentes or 'nenhuma'}")

    if args.referencia:
        referencia = args.referencia
        if referencia not in presentes:
            raise SystemExit(f"Referência '{referencia}' ausente em {base_dir}. "
                             f"Disponíveis: {', '.join(presentes)}")
    elif REFERENCIA in presentes:
        referencia = REFERENCIA
    else:
        # Nas escalas grandes o Pandas não conclui. Qualquer engine serve de
        # eixo: o teste verifica concordância mútua, não conformidade a uma
        # implementação privilegiada.
        referencia = presentes[0]
        print(f"AVISO: '{REFERENCIA}' não produziu matriz nesta escala; "
              f"usando '{referencia}' como referência.\n")

    ref = f"{base_dir}/{referencia}.parquet"
    alvos = [f"{e}.parquet" for e in presentes if e != referencia]

    print(f"Referência: {referencia} (escala {args.escala})")
    print(f"Comparando: {', '.join(a.removesuffix('.parquet') for a in alvos)}")
    print(f"Tolerância: rtol={base.RTOL}, atol={base.ATOL}\n")

    # A referência é comparada contra si mesma em tudo, menos na ordenação:
    # sem esta verificação, uma referência fora de ordem passaria sem exame e
    # ainda serviria de eixo para as demais.
    tudo_ok = True
    if not _esta_ordenado(pd.read_parquet(ref, columns=["cliente_id"])):
        print(f"[X]  {referencia} (referência)")
        print("       saída NÃO ordenada por cliente_id, como o contrato exige")
        tudo_ok = False

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
