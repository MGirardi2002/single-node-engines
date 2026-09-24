"""
Verificador do exercício do Stage B.

Testa cada função de src/modelos/stage_b.py com exemplos pequenos, em que a
resposta certa dá para conferir de cabeça. Funções ainda não implementadas
aparecem como PENDENTE; as outras, como OK ou FALHOU com uma dica.

Uso:
    python scripts/verificar_stage_b.py          # exemplos pequenos
    python scripts/verificar_stage_b.py --real   # + dados reais de 100 MB
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gerador import config as cfg  # noqa: E402
from src.modelos import stage_b as sb  # noqa: E402

resultado: dict[str, str] = {}


def teste(nome):
    def deco(f):
        def rodar():
            try:
                f()
                resultado[nome] = "OK"
                print(f"  [OK]       {nome}")
            except NotImplementedError:
                resultado[nome] = "PENDENTE"
                print(f"  [PENDENTE] {nome}")
            except AssertionError as e:
                resultado[nome] = "FALHOU"
                print(f"  [FALHOU]   {nome}\n             {e}")
            except Exception as e:
                resultado[nome] = "FALHOU"
                print(f"  [ERRO]     {nome}\n             {type(e).__name__}: {e}")
        return rodar
    return deco


def perto(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------

@teste("1. zscore_score")
def t_zscore():
    X = np.array([[0.0, 0.0],
                  [3.0, -4.0],
                  [1.0, 1.0]])
    s = sb.zscore_score(X)
    assert np.shape(s) == (3,), (
        f"formato {np.shape(s)}; esperado (3,) — um score por LINHA. "
        "Confira o axis do max.")
    assert np.allclose(s, [0, 4, 1]), (
        f"devolveu {list(np.round(s, 3))}; esperado [0, 4, 1]. "
        "A linha [3, -4] deve dar 4: o maior valor ABSOLUTO.")


@teste("2. precision_at_k")
def t_patk():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    p = sb.precision_at_k(y, s, 2)
    assert perto(p, 1.0), (
        f"k=2 deu {p}; esperado 1.0 (os 2 maiores scores são os 2 anômalos). "
        "Se deu 0.0, você pegou os MENORES scores — lembre que argsort é "
        "crescente.")
    p3 = sb.precision_at_k(y, s, 3)
    assert perto(p3, 2 / 3, 1e-6), f"k=3 deu {p3}; esperado 0.667"


def _nuvem_com_outlier():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(300, 2))
    X = np.vstack([X, [[8.0, 8.0]]])      # índice 300 é o anômalo óbvio
    return X


@teste("3. iforest_score")
def t_iforest():
    X = _nuvem_com_outlier()
    s = sb.iforest_score(X)
    assert np.shape(s) == (301,), f"formato {np.shape(s)}; esperado (301,)"
    assert int(np.argmax(s)) == 300, (
        "o ponto (8, 8), claramente isolado, não recebeu o MAIOR score. "
        "Provavelmente o sinal está invertido: score_samples devolve valores "
        "menores para os mais anômalos.")
    assert np.allclose(s, sb.iforest_score(X)), (
        "duas chamadas deram resultados diferentes — faltou random_state=SEMENTE?")


@teste("4. lof_score")
def t_lof():
    X = _nuvem_com_outlier()
    s = sb.lof_score(X)
    assert np.shape(s) == (301,), f"formato {np.shape(s)}; esperado (301,)"
    assert int(np.argmax(s)) == 300, (
        "o ponto (8, 8) não recebeu o MAIOR score. negative_outlier_factor_ "
        "é negativo e mais negativo = mais anômalo; é preciso inverter.")
    assert (s > 0).all(), (
        "há scores <= 0; depois de inverter o sinal, o LOF deveria ser "
        "positivo (em torno de 1 para pontos normais).")


@teste("5. avaliar_por_tipo")
def t_por_tipo():
    tipos = np.array(["normal", "normal", "normal",
                      "rajada", "multivariada",
                      "ticket_desproporcional", "migracao_canal"])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.05, 0.25])
    r = sb.avaliar_por_tipo(tipos, s)

    assert set(r) == set(cfg.TIPOS_ANOMALIA), (
        f"chaves {sorted(r)}; esperado uma por tipo de anomalia")
    assert perto(r["rajada"], 1.0, 1e-6), (
        f"rajada deu {r['rajada']:.3f}; esperado 1.0. A 'rajada' (0.8) tem o "
        "maior score entre ela e os normais. Se deu 0.5, a 'multivariada' "
        "(0.9) entrou na conta como normal — ela deveria ter sido excluída.")
    assert perto(r["ticket_desproporcional"], 0.25, 1e-6), (
        f"ticket_desproporcional deu {r['ticket_desproporcional']:.3f}; "
        "esperado 0.25 (score 0.05, abaixo dos 3 normais).")
    assert perto(r["migracao_canal"], 0.5, 1e-6), (
        f"migracao_canal deu {r['migracao_canal']:.3f}; esperado 0.5.")


# ---------------------------------------------------------------------------

def real():
    print("\nDados reais — escala 100mb (leva alguns minutos por causa do LOF)\n")
    X, y, tipos = sb.carregar("100mb")
    res = sb.avaliar(X, y, tipos)
    print(res[["modelo", "n", "pr_auc", "acaso", "precision_at_k"]]
          .round(4).to_string(index=False))

    # Faixas obtidas no notebook-guia. Pequenas variações são normais; algo
    # muito fora indica erro de implementação.
    esperado = {"Z-Score": (0.40, 0.48), "IForest": (0.10, 0.18),
                "LOF": (0.08, 0.35)}
    print()
    ok = True
    for _, linha in res.iterrows():
        lo, hi = esperado[linha["modelo"]]
        dentro = lo <= linha["pr_auc"] <= hi
        ok &= dentro
        print(f"  [{'OK' if dentro else 'CONFERIR'}] {linha['modelo']:<8} "
              f"PR-AUC {linha['pr_auc']:.3f}  (esperado entre {lo} e {hi})")
    return ok


if __name__ == "__main__":
    print("Verificando src/modelos/stage_b.py\n")
    for t in (t_zscore, t_patk, t_iforest, t_lof, t_por_tipo):
        t()

    feitos = sum(v == "OK" for v in resultado.values())
    print(f"\n{feitos}/{len(resultado)} funções corretas")

    if "--real" in sys.argv:
        if feitos < len(resultado):
            print("\nConclua as 5 funções antes de rodar com dados reais.")
        else:
            real()
