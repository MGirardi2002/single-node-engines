"""
Validação da base sintética antes de gerar as escalas grandes.

Responde a três perguntas, nesta ordem de importância:

  1. A base é íntegra?  (proporção de anomalias, integridade referencial,
     distribuição dos tipos)
  2. As anomalias são *difíceis o bastante*?  Se um Z-Score univariado já
     resolve o problema, o eixo de ML do TCC fica vazio — Isolation Forest e
     LOF não teriam o que acrescentar.
  3. O feature engineering tem efeito?  Compara detecção sobre atributos
     brutos da dimensão vs. features agregadas da tabela fato.

Critério de aceite: Z-Score claramente abaixo do Isolation Forest, e nenhum dos
dois perto da perfeição. PR-AUC (average precision) é a métrica usada — com 2%
de anomalias, acurácia e F1 em threshold fixo enganam.

Uso:
    python scripts/validar_base.py [escala]
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.gerador import config as cfg  # noqa: E402

ESCALA = sys.argv[1] if len(sys.argv) > 1 else "100mb"

with open(cfg.MANIFESTO, encoding="utf-8") as fp:
    manifesto = json.load(fp)
if ESCALA not in manifesto:
    sys.exit(f"Escala '{ESCALA}' não encontrada em {cfg.MANIFESTO}. Gere-a primeiro.")

paths = manifesto[ESCALA]
tx = pd.concat([pd.read_parquet(p) for p in paths["transacoes"]], ignore_index=True)
clientes = pd.concat([pd.read_parquet(p) for p in paths["clientes"]], ignore_index=True)
labels = pd.concat([pd.read_parquet(p) for p in paths["labels"]], ignore_index=True)

print("=" * 62)
print(f" 1. INTEGRIDADE — escala '{ESCALA}'")
print("=" * 62)
print(f"transacoes : {len(tx):>12,} linhas")
print(f"clientes   : {len(clientes):>12,} linhas")
print(f"labels     : {len(labels):>12,} linhas")

orfas = ~tx["cliente_id"].isin(set(clientes["cliente_id"]))
print(f"\nintegridade referencial : {'OK' if not orfas.any() else f'FALHA ({orfas.sum():,} órfãs)'}")
print(f"ids duplicados (tx)     : {'OK' if tx['transacao_id'].is_unique else 'FALHA'}")
print(f"ids duplicados (cliente): {'OK' if clientes['cliente_id'].is_unique else 'FALHA'}")
print(f"nulos                   : {'OK' if tx.isna().sum().sum() == 0 else 'FALHA'}")

pct = labels["is_anomaly"].mean() * 100
print(f"\nproporção de anomalias  : {pct:.2f}%  (alvo {cfg.PCT_ANOMALIA*100:.1f}%)")
print("\ndistribuição por tipo:")
print(labels["anomaly_type"].value_counts().to_string())

janela = (tx["data_hora"].max() - tx["data_hora"].min()).days
print(f"\njanela temporal         : {janela} dias")

# ---------------------------------------------------------------------------
# 2. Feature engineering — prévia do que o Stage A fará em cada engine
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print(" 2. FEATURE ENGINEERING")
print("=" * 62)

fim = tx["data_hora"].max()

agg = tx.groupby("cliente_id").agg(
    n_tx=("transacao_id", "count"),
    valor_total=("valor", "sum"),
    valor_medio=("valor", "mean"),
    valor_std=("valor", "std"),
    valor_max=("valor", "max"),
    n_categorias=("categoria", "nunique"),
    ultima_compra=("data_hora", "max"),
)
agg["recencia_dias"] = (fim - agg["ultima_compra"]).dt.days
agg = agg.drop(columns="ultima_compra")

# Concentração temporal: maior nº de transações numa mesma janela de 7 dias.
# É o que revela a anomalia de "rajada" — invisível em qualquer agregado total.
semana = tx["data_hora"].dt.floor("7D")
por_semana = tx.groupby(["cliente_id", semana]).size()
agg["max_tx_7d"] = por_semana.groupby(level=0).max()
agg["burstiness"] = agg["max_tx_7d"] / (agg["n_tx"] / (janela / 7))

# Perfil de canal.
canal = pd.crosstab(tx["cliente_id"], tx["canal"], normalize="index")
for c in cfg.CANAIS:
    agg[f"pct_{c}"] = canal.get(c, 0.0)

# Participação por categoria — é o vetor em que a anomalia multivariada vive.
# Nenhuma coluna isolada fica extrema; o que é raro é a combinação.
cat = pd.crosstab(tx["cliente_id"], tx["categoria"], normalize="index")
for c in cfg.CATEGORIAS:
    agg[f"cat_{c}"] = cat.get(c, 0.0)

# Variação entre a primeira e a segunda metade da janela. É o que separa a
# anomalia de migração de canal do perfil 'digital' legítimo: o digital sempre
# foi digital, o anômalo mudou. Exige cruzar tempo com perfil de canal —
# nenhuma agregação simples captura isso.
inicio = tx["data_hora"].min()
recente = tx["data_hora"] > (inicio + (fim - inicio) / 2)
eh_web = (tx["canal"] == "web").astype("float64")
metades = (pd.DataFrame({"cliente_id": tx["cliente_id"], "recente": recente,
                         "valor": tx["valor"], "web": eh_web})
           .groupby(["cliente_id", "recente"])
           .agg(valor=("valor", "mean"), web=("web", "mean"))
           .unstack(fill_value=0.0))
agg["delta_pct_web"] = metades[("web", True)] - metades[("web", False)]
agg["delta_ticket"] = metades[("valor", True)] / metades[("valor", False)].replace(0, np.nan)

df = agg.join(clientes.set_index("cliente_id")).join(labels.set_index("cliente_id"))

# Razões com a dimensão — revelam as anomalias contextuais.
df["razao_gasto_renda"] = df["valor_total"] / df["renda_mensal"]
df["razao_ticket_renda"] = df["valor_medio"] / df["renda_mensal"]
df["razao_limite_renda"] = df["limite_credito"] / df["renda_mensal"]
df["uso_limite"] = df["valor_total"] / df["limite_credito"]

df = df.fillna(0.0)
print(f"matriz de features: {df.shape[0]:,} clientes x "
      f"{df.select_dtypes('number').shape[1] - 1} features numéricas")

y = df["is_anomaly"].to_numpy()
FEATS_BRUTAS = ["idade", "renda_mensal", "score_credito", "limite_credito",
                "tempo_cliente_meses", "patrimonio_estimado"]
FEATS_ENG = [c for c in df.select_dtypes("number").columns if c != "is_anomaly"]

# ---------------------------------------------------------------------------
# 3. Dificuldade
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print(" 3. DIFICULDADE DAS ANOMALIAS  (PR-AUC, baseline = %.4f)" % y.mean())
print("=" * 62)


def zscore_score(X: np.ndarray) -> np.ndarray:
    """Maior |z| entre as features — o baseline univariado."""
    return np.abs(StandardScaler().fit_transform(X)).max(axis=1)


def iforest_score(X: np.ndarray) -> np.ndarray:
    m = IsolationForest(contamination=cfg.PCT_ANOMALIA, random_state=42, n_jobs=-1)
    m.fit(X)
    return -m.score_samples(X)


def lof_score(X: np.ndarray) -> np.ndarray:
    """
    LOF é baseado em kNN (~O(n^2)) e não escala. Aqui roda inteiro porque a
    validação usa só a menor escala; no benchmark ele ficará restrito às
    escalas menores, como limitação declarada.
    """
    m = LocalOutlierFactor(n_neighbors=20, contamination=cfg.PCT_ANOMALIA, n_jobs=-1)
    m.fit_predict(X)
    return -m.negative_outlier_factor_


X_brutas = StandardScaler().fit_transform(df[FEATS_BRUTAS].to_numpy())
X_eng = StandardScaler().fit_transform(df[FEATS_ENG].to_numpy())

MODELOS = {"Z-Score": zscore_score, "IForest": iforest_score, "LOF": lof_score}

scores = {("brutas", n): f(X_brutas) for n, f in MODELOS.items()}
scores |= {("eng", n): f(X_eng) for n, f in MODELOS.items()}

print(f"\n{'features':<12}{'modelo':<10}{'PR-AUC (geral)':>16}")
print("-" * 38)
for (feats, modelo), s in scores.items():
    print(f"{feats:<12}{modelo:<10}{average_precision_score(y, s):>16.4f}")

# -- desempenho por tipo de anomalia ----------------------------------------
# Exigir que o IForest vença no agregado seria desenhar os dados para produzir
# um vencedor pré-definido. O que interessa é que tipos DIFERENTES favoreçam
# modelos diferentes — aí a comparação tem conteúdo, e o resultado é honesto.
print("\nPR-AUC por tipo de anomalia (features engenheiradas).")
print("'ganho' = quantas vezes o melhor modelo supera o acaso.\n")
print(f"{'tipo':<26}" + "".join(f"{m:>10}" for m in MODELOS)
      + f"{'melhor':>10}{'ganho':>8}")
print("-" * 74)

tipos = list(cfg.TIPOS_ANOMALIA)
por_tipo = {}
for t in tipos:
    # Um tipo por vez contra a população normal, para isolar seu efeito.
    sel = ((df["anomaly_type"] == t) | (df["anomaly_type"] == "normal")).to_numpy()
    yt = (df["anomaly_type"] == t).to_numpy()[sel].astype(int)
    aps = {m: average_precision_score(yt, scores[("eng", m)][sel]) for m in MODELOS}
    melhor = max(aps, key=aps.get)
    ganho = aps[melhor] / yt.mean()
    por_tipo[t] = (aps, melhor, ganho)
    print(f"{t:<26}" + "".join(f"{aps[m]:>10.4f}" for m in MODELOS)
          + f"{melhor:>10}{ganho:>7.1f}x")

print("\n" + "=" * 62)
print(" VEREDITO")
print("=" * 62)

ap_eng = max(average_precision_score(y, scores[("eng", m)]) for m in MODELOS)
ap_brutas = max(average_precision_score(y, scores[("brutas", m)]) for m in MODELOS)

ok = True

if ap_eng > 0.95:
    print(f"[X] Detecção quase perfeita ({ap_eng:.3f}) — anomalias FÁCEIS "
          "DEMAIS. Reduzir magnitudes no gerador.")
    ok = False
else:
    print(f"[OK] Nenhum modelo resolve o problema sozinho (melhor: {ap_eng:.3f}).")

if ap_eng <= ap_brutas * 1.5:
    print(f"[X] Features engenheiradas ({ap_eng:.3f}) pouco acrescentam sobre "
          f"as brutas ({ap_brutas:.3f}) — o pipeline não está agregando valor.")
    ok = False
else:
    print(f"[OK] Feature engineering eleva o PR-AUC de {ap_brutas:.3f} para "
          f"{ap_eng:.3f} — o pipeline justifica sua existência.")

# Cada tipo precisa ser detectável de forma não-trivial por ALGUM modelo.
# Sem esse piso, um tipo indetectável passaria despercebido e viraria ruído
# puro nos resultados.
fracos = [t for t, (_, _, ganho) in por_tipo.items() if ganho < 5.0]
if fracos:
    print(f"[X] Tipo(s) praticamente indetectável(is) (ganho < 5x sobre o "
          f"acaso): {', '.join(fracos)}. Aumentar a magnitude desses padrões.")
    ok = False
else:
    print("[OK] Todos os tipos são detectáveis por algum modelo com ganho "
          ">= 5x sobre o acaso.")

vencedores = {melhor for _, (_, melhor, _) in por_tipo.items()}
if len(vencedores) < 2:
    print(f"[X] Um único modelo ({vencedores.pop()}) vence em TODOS os tipos — "
          "o eixo de comparação entre modelos fica vazio.")
    ok = False
else:
    linha = ", ".join(f"{t}->{m}" for t, (_, m, _) in por_tipo.items())
    print(f"[OK] Modelos diferentes vencem em tipos diferentes ({linha}) — "
          "a comparação entre modelos tem conteúdo.")

print("\n" + ("BASE APROVADA — pode gerar as escalas maiores."
              if ok else "BASE REPROVADA — ajustar o gerador antes de escalar."))
sys.exit(0 if ok else 1)
