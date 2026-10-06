"""
Gráficos comparativos e tabela de recomendações, a partir de
`results/benchmark.csv`.

Saída em SVG e PNG dentro de `results/figuras/`. O SVG é o que entra na
monografia: vetorial, sem perda ao ampliar e sem depender da resolução de tela
em que foi gerado.

Decisões de visualização
------------------------
**Modo claro apenas.** As figuras vão para um documento impresso; não há modo
escuro em papel.

**Nunca dois eixos y.** Tempo e memória são grandezas de escalas diferentes e
saem em figuras separadas. Sobrepô-las num par de eixos permitiria sugerir
qualquer correlação pela simples escolha das escalas.

**Cor nunca sozinha.** Cada engine tem cor, marcador e rótulo direto na ponta da
linha. Três razões: a monografia pode ser impressa em preto e branco; dois dos
quatro tons da paleta ficam abaixo de 3:1 de contraste com o fundo; e o par
amarelo/laranja fica perto do limite de discriminação para visão normal quando
todas as linhas são comparadas entre si.

**Eixo x logarítmico**, porque as escalas dobram (100 MB -> 1,6 GB). No gráfico
de tempo o eixo y também é logarítmico: numa escala log-log, um custo que dobra
quando o volume dobra vira uma reta, e o que se lê na inclinação é o expoente de
escalonamento — que é a pergunta do trabalho. Os valores absolutos não se perdem
porque cada linha carrega o próprio número na ponta.

**Memória em eixo linear**, com a trava do harness marcada. Ali o que importa é
a distância até o teto, e o log a comprimiria justamente onde ela é decisiva.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

# Backend sem janela: o script roda no WSL, sem servidor gráfico.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ARQUIVO_PADRAO = "results/benchmark.csv"
DIR_FIGURAS = "results/figuras"

# Volume nominal de cada escala, em MB — dá as posições no eixo x.
ESCALAS_MB = {"100mb": 100, "200mb": 200, "400mb": 400,
              "800mb": 800, "1600mb": 1600}
ROTULO_ESCALA = {"100mb": "100 MB", "200mb": "200 MB", "400mb": "400 MB",
                 "800mb": "800 MB", "1600mb": "1,6 GB"}

# Ordem fixa das engines. A cor de cada uma vem do slot correspondente da
# paleta categórica e NÃO é reatribuída quando uma engine falta no gráfico:
# a cor acompanha a entidade, nunca a sua posição.
ENGINES = ("polars", "pandas", "duckdb", "spark")
NOME = {"polars": "Polars", "pandas": "Pandas",
        "duckdb": "DuckDB", "spark": "PySpark"}
COR = {"polars": "#2a78d6",   # slot 1 — azul
       "pandas": "#eb6834",   # slot 2 — laranja
       "duckdb": "#1baf7a",   # slot 3 — aqua
       "spark":  "#eda100"}   # slot 4 — amarelo
# Codificação secundária: sobrevive à impressão em preto e branco e distingue
# o par amarelo/laranja, que é o mais próximo da paleta.
MARCADOR = {"polars": "o", "pandas": "s", "duckdb": "^", "spark": "D"}

# Cromo do gráfico: recessivo por princípio — a tinta forte é dos dados.
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_MUDA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
FUNDO = "#fcfcfb"

# Trava do harness: 95% de 8,5 GB. É o teto contra o qual "estourar a memória"
# é definido neste experimento, então aparece explicitamente na figura.
TRAVA_MB = 8.5 * 0.95 * 1024


def carregar(caminho: str) -> dict[tuple[str, str], dict]:
    """
    Agrega o CSV por (engine, escala).

    Só linhas `status=ok` entram nas medianas. Linhas com aviso no campo `erro`
    são descartadas: são justamente as que o harness marcou como suspeitas
    (linha de base contaminada), e incluí-las anularia o propósito do aviso.
    """
    brutos: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    falhou: dict[tuple[str, str], tuple[str, float]] = {}

    with open(caminho, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            chave = (r["engine"], r["escala"])
            if r["status"] == "ok" and not r.get("erro"):
                brutos[chave].append(
                    (float(r["tempo_total"]), float(r["pico_mem_mb"])))
            elif r["status"] != "ok":
                # O pico de uma execução encerrada É uma medição válida: é o
                # ponto em que a engine bateu no teto. Não existe tempo válido
                # (a execução não terminou), mas o consumo até ali foi medido.
                falhou[chave] = (r["status"], float(r["pico_mem_mb"]))

    dados: dict[tuple[str, str], dict] = {}
    for chave, obs in brutos.items():
        tempos = [o[0] for o in obs]
        memorias = [o[1] for o in obs]
        dados[chave] = {
            "tempo": statistics.median(tempos),
            "memoria": statistics.median(memorias),
            "amplitude": max(tempos) - min(tempos) if len(tempos) > 1 else 0.0,
            "n": len(tempos),
            "status": "ok",
        }
    # Um par que só tem falhas não entra nas medianas, mas precisa aparecer no
    # gráfico: o ponto de ruptura de cada engine é um resultado do trabalho.
    for chave, (status, pico) in falhou.items():
        dados.setdefault(chave, {"tempo": None, "memoria": None,
                                 "pico_falha": pico, "amplitude": 0.0,
                                 "n": 0, "status": status})
    return dados


def _escalas_presentes(dados: dict) -> list[str]:
    presentes = {esc for _, esc in dados}
    return [e for e in ESCALAS_MB if e in presentes]


def _moldura(ax, titulo: str, rotulo_y: str, escalas: list[str],
             pad_titulo: int = 72) -> None:
    """Eixos, grade e rótulos — o cromo recessivo, aplicado igual nas figuras."""
    # O pad só precisa ser grande onde existe a faixa de execuções não
    # concluídas acima da área de dados (gráfico de tempo).
    ax.set_title(titulo, color=TINTA, fontsize=13, pad=pad_titulo, loc="left")
    ax.set_xlabel("Volume da base", color=TINTA_2, fontsize=10, labelpad=10)
    ax.set_ylabel(rotulo_y, color=TINTA_2, fontsize=10, labelpad=10)

    ax.set_xscale("log", base=2)
    ax.set_xticks([ESCALAS_MB[e] for e in escalas])
    ax.set_xticklabels([ROTULO_ESCALA[e] for e in escalas])
    ax.minorticks_off()

    ax.grid(True, which="major", color=GRADE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EIXO)
        ax.spines[lado].set_linewidth(1.0)
    ax.tick_params(colors=TINTA_MUDA, labelsize=9, length=0)


def _rotulos_sem_colisao(ax, pontos: list[tuple[float, float, str, str]]) -> None:
    """
    Rótulo direto na ponta de cada linha, afastando os que se sobrepõem.

    Com quatro séries, a legenda sozinha obriga o leitor a ir e voltar entre a
    linha e a caixa. O rótulo direto elimina esse trajeto — mas se duas pontas
    caem quase no mesmo y, os textos se sobrepõem. Aqui eles são ordenados por
    altura e empurrados para cima quando ficam perto demais.
    """
    if not pontos:
        return
    # O afastamento é calculado em PIXELS, não em unidades do eixo. Num eixo
    # logarítmico a mesma diferença numérica ocupa alturas completamente
    # diferentes conforme a região do gráfico: perto do topo, dois valores a
    # 40 s de distância ficam colados, enquanto lá embaixo a mesma diferença
    # separa demais. Sobreposição é um fato de tela, então mede-se na tela.
    ax.figure.canvas.draw()
    para_tela = ax.transData.transform
    para_dados = ax.transData.inverted().transform

    AFASTAMENTO_PX = 15.0
    ordenados = sorted(pontos, key=lambda p: p[1])
    ys_px: list[float] = []
    for x, y, _, _ in ordenados:
        y_px = para_tela((x, y))[1]
        if ys_px and y_px - ys_px[-1] < AFASTAMENTO_PX:
            y_px = ys_px[-1] + AFASTAMENTO_PX
        ys_px.append(y_px)

    for (x, _, engine, texto), y_px in zip(ordenados, ys_px):
        x_px = para_tela((x, 1))[0]
        y_ajustado = para_dados((x_px, y_px))[1]
        ax.annotate(f"{NOME[engine]}  {texto}",
                    xy=(x, y_ajustado), xytext=(10, 0),
                    textcoords="offset points", annotation_clip=False,
                    color=TINTA_2, fontsize=9, va="center")


def _marcar_falhas(ax, falhas: list[tuple[float, str, str]]) -> bool:
    """
    Marca, no topo da área do gráfico, a escala em que cada engine falhou.

    O marcador fica no topo de propósito: uma execução que estourou a memória
    não tem tempo nem pico válidos, e desenhá-la na altura do último ponto bom
    sugeriria um valor que não foi medido.
    """
    if not falhas:
        return False
    # x em coordenadas de dados, y em fração da área — o marcador fica colado
    # no topo independentemente da escala do eixo y.
    transformacao = ax.get_xaxis_transform()

    # Empilha as engines que falharam na MESMA escala. Sem isto elas caem no
    # mesmo ponto e uma esconde a outra — foi o que aconteceu na primeira
    # versão, onde Pandas e Polars estouraram juntos em 800 MB e 1,6 GB e só um
    # marcador aparecia.
    por_escala: dict[float, list[tuple[str, str]]] = defaultdict(list)
    for x, engine, status in falhas:
        por_escala[x].append((engine, status))

    # A faixa fica ACIMA da área de dados, e não dentro dela. Dentro, os
    # marcadores caíam em cima dos rótulos das linhas — em 1,6 GB eles ficam a
    # ~95% da altura — e, pior, um "X" desenhado entre os dados sugere um valor
    # naquela altura. Fora da área, a leitura é inequívoca: não há valor.
    for x, itens in por_escala.items():
        for i, (engine, status) in enumerate(sorted(itens)):
            y = 1.04 + i * 0.075
            ax.plot([x], [y], marker="X", markersize=10, color=COR[engine],
                    markeredgecolor=FUNDO, markeredgewidth=1.5,
                    transform=transformacao, clip_on=False, zorder=5)
            # O nome da engine acompanha o marcador: a cor sozinha não pode
            # carregar a identidade, ainda mais numa figura que será impressa.
            ax.annotate(f"{NOME[engine]} · {status}",
                        xy=(x, y), xycoords=transformacao,
                        xytext=(9, 0), textcoords="offset points",
                        color=TINTA_2, fontsize=8, va="center",
                        annotation_clip=False)
    return True


# --- Etapas do pipeline (gráfico da QP2) ----------------------------------
#
# Cores CATEGÓRICAS, na ordem fixa da paleta, e não uma rampa de um tom só.
#
# A primeira versão usava quatro passos de azul, com o argumento de que as
# etapas têm ordem natural (a sequência do pipeline) e de que isso liberaria as
# cores das engines, usadas nas outras figuras. Na prática os passos vizinhos
# ficaram indistinguíveis nos segmentos pequenos — o encoding falhou no que
# tinha de essencial, que é permitir identificar cada fatia.
#
# A perda de informação é menor do que parece: numa barra empilhada ordenada
# pelo pipeline, a SEQUÊNCIA já está na posição, e a cor não precisa carregá-la.
# A coincidência com as cores das engines é um risco apenas entre figuras; aqui
# dentro a legenda nomeia as etapas e o eixo nomeia as engines.
#
# Junção, derivadas e escrita somam menos de 6% em todas as engines e viram
# uma categoria "outras", em cinza fora da paleta categórica: agrupá-las evita
# fatias ilegíveis, e o cinza sinaliza resíduo, não etapa de interesse.
ETAPAS_PRINCIPAIS = ("ingestao", "limpeza", "agregacao", "janela")
ETAPAS_RESIDUAIS = ("join", "derivadas", "escrita")
ROTULO_ETAPA = {"ingestao": "Ingestão", "limpeza": "Limpeza",
                "agregacao": "Agregação", "janela": "Janela",
                "outras": "Outras (junção, derivadas, escrita)"}
COR_ETAPA = {"ingestao": "#2a78d6",   # slot 1 — azul
             "limpeza": "#eb6834",    # slot 2 — laranja
             "agregacao": "#1baf7a",  # slot 3 — aqua
             "janela": "#eda100",     # slot 4 — amarelo
             "outras": "#a8a69e"}     # cinza neutro, fora da paleta

# Em qual fundo o rótulo percentual precisa de texto claro. Azul e laranja são
# escuros o bastante para texto branco; aqua, amarelo e cinza exigem tinta
# escura, senão o número some.
ETAPAS_TEXTO_CLARO = {"ingestao", "limpeza"}


def carregar_etapas(caminho: str, escala: str) -> dict[str, dict[str, float]]:
    """Mediana do tempo de cada etapa, por engine, numa escala."""
    bruto: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    with open(caminho, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if r["escala"] != escala or r["status"] != "ok" or r.get("erro"):
                continue
            for e in (*ETAPAS_PRINCIPAIS, *ETAPAS_RESIDUAIS):
                bruto[r["engine"]][e].append(float(r[f"t_{e}"] or 0.0))

    dados: dict[str, dict[str, float]] = {}
    for engine, etapas in bruto.items():
        med = {e: statistics.median(v) for e, v in etapas.items()}
        dados[engine] = {e: med[e] for e in ETAPAS_PRINCIPAIS}
        dados[engine]["outras"] = sum(med[e] for e in ETAPAS_RESIDUAIS)
    return dados


def grafico_etapas(dados: dict, escala: str,
                   dir_saida: str = DIR_FIGURAS) -> list[str]:
    """
    Composição do tempo por etapa — responde à QP2.

    Barras empilhadas NORMALIZADAS A 100%, e a normalização é decisão
    metodológica, não estética. Os totais do modo instrumentado não são
    comparáveis entre engines: a materialização forçada ao fim de cada etapa
    penaliza as preguiçosas de formas diferentes. Uma barra em valores
    absolutos convidaria à comparação de alturas, que é inválida. Normalizar
    torna essa leitura impossível por construção e deixa apenas a comparação de
    composições, que é o que a QP2 pergunta.
    """
    engines = [e for e in ENGINES if e in dados]
    fig, ax = plt.subplots(figsize=(10.5, 4.9), facecolor=FUNDO)
    ax.set_facecolor(FUNDO)
    # Título no nível da figura, e não do eixo: no nível do eixo ele disputa a
    # mesma faixa horizontal com a legenda, que precisa ficar logo acima das
    # barras para que o leitor associe cor e etapa sem percorrer a página.
    fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.26)

    chaves = (*ETAPAS_PRINCIPAIS, "outras")
    ys = range(len(engines))
    esquerda = [0.0] * len(engines)

    for etapa in chaves:
        larguras = []
        for engine in engines:
            total = sum(dados[engine].values()) or 1.0
            larguras.append(dados[engine][etapa] / total * 100)
        ax.barh(list(ys), larguras, left=esquerda, height=0.62,
                color=COR_ETAPA[etapa], label=ROTULO_ETAPA[etapa],
                # Fio da cor do fundo entre segmentos: separa as fatias sem
                # introduzir uma linha de contorno que competiria com os dados.
                edgecolor=FUNDO, linewidth=2.0, zorder=3)
        for i, (x0, w) in enumerate(zip(esquerda, larguras)):
            # Rótulo só onde cabe: abaixo de ~8% o número encosta nas bordas.
            if w >= 8:
                ax.text(x0 + w / 2, i, f"{w:.0f}%", ha="center", va="center",
                        color=FUNDO if etapa in ETAPAS_TEXTO_CLARO else TINTA,
                        fontsize=9, fontweight="bold", zorder=4)
        esquerda = [a + b for a, b in zip(esquerda, larguras)]

    ax.set_yticks(list(ys))
    ax.set_yticklabels([NOME[e] for e in engines], fontsize=10, color=TINTA)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Participação no tempo total da engine", color=TINTA_2,
                  fontsize=10, labelpad=10)
    fig.text(0.085, 0.935,
             f"Onde cada engine gasta o tempo — {ROTULO_ESCALA[escala]}",
             color=TINTA, fontsize=13, ha="left", va="top")

    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(EIXO)
    ax.tick_params(colors=TINTA_MUDA, labelsize=9, length=0)
    ax.grid(False)

    ax.legend(loc="lower left", bbox_to_anchor=(-0.06, 1.02), ncol=5,
              frameon=False, fontsize=8.5, labelcolor=TINTA_2,
              handlelength=1.1, columnspacing=1.4, handletextpad=0.5)

    ax.annotate("Modo instrumentado: cada etapa é materializada ao terminar, o "
                "que torna o detalhamento visível.\nAs proporções são "
                "comparáveis entre engines; os tempos absolutos NÃO são — a "
                "materialização\nforçada penaliza as engines de avaliação "
                "preguiçosa de formas diferentes.",
                xy=(0.0, -0.30), xycoords="axes fraction",
                color=TINTA_MUDA, fontsize=8, va="top")

    return _salvar(fig, f"etapas_{escala}", dir_saida)


def _tracar_variante(ax, dados: dict, escalas: list[str], rotulo: str,
                     eixo: str, falhas: list | None = None
                     ) -> list[tuple[float, float, str, str]]:
    """
    Desenha uma VARIANTE de implementação de uma engine já presente no gráfico.

    A cor continua sendo a da engine, porque a entidade é a mesma: é o Polars
    nos dois casos. O que distingue as duas séries é o traço (tracejado) e o
    rótulo. Trocar a cor sugeriria que se trata de outra ferramenta, e violaria
    a regra de que a cor acompanha a entidade, não a variante.
    """
    pontas = []
    for engine in ENGINES:
        xs, ys = [], []
        for esc in escalas:
            d = dados.get((engine, esc))
            if not d:
                continue
            valor = d.get(eixo)
            if valor is not None:
                xs.append(ESCALAS_MB[esc])
                ys.append(valor / 1024 if eixo == "memoria" else valor)
            elif falhas is not None:
                # A variante também precisa mostrar onde falhou: sem isso, a
                # linha tracejada simplesmente termina, e o leitor não
                # distingue "não foi medido" de "não concluiu".
                falhas.append((ESCALAS_MB[esc], engine,
                               f"{d['status']} · {rotulo}"))
        if not xs:
            continue
        ax.plot(xs, ys, color=COR[engine], linewidth=1.8,
                linestyle=(0, (6, 3)), marker=MARCADOR[engine], markersize=7,
                markerfacecolor=FUNDO, markeredgecolor=COR[engine],
                markeredgewidth=1.6, zorder=4)
        sufixo = f"{ys[-1]:.1f} GB" if eixo == "memoria" else f"{ys[-1]:.1f} s"
        pontas.append((xs[-1], ys[-1], engine, f"{sufixo} · {rotulo}"))
    return pontas


def _figura():
    fig, ax = plt.subplots(figsize=(10.5, 6.0), facecolor=FUNDO)
    ax.set_facecolor(FUNDO)
    # Margem à direita para os rótulos diretos, que ficam fora da última
    # escala; margem superior para a faixa de execuções não concluídas.
    fig.subplots_adjust(left=0.085, right=0.78, top=0.74, bottom=0.15)
    return fig, ax


def _salvar(fig, nome: str, dir_saida: str) -> list[str]:
    os.makedirs(dir_saida, exist_ok=True)
    caminhos = []
    for ext in ("svg", "png"):
        caminho = os.path.join(dir_saida, f"{nome}.{ext}")
        fig.savefig(caminho, format=ext, dpi=200, facecolor=FUNDO)
        caminhos.append(caminho)
    plt.close(fig)
    return caminhos


def grafico_tempo(dados: dict, dir_saida: str = DIR_FIGURAS,
                  variantes: list[tuple[str, dict]] | None = None) -> list[str]:
    escalas = _escalas_presentes(dados)
    if variantes:
        for _, d in variantes:
            escalas = sorted(set(escalas) | set(_escalas_presentes(d)),
                             key=lambda e: ESCALAS_MB[e])
    fig, ax = _figura()

    pontas, falhas = [], []
    for engine in ENGINES:
        xs, ys = [], []
        for esc in escalas:
            d = dados.get((engine, esc))
            if d and d["tempo"] is not None:
                xs.append(ESCALAS_MB[esc])
                ys.append(d["tempo"])
            elif d:
                falhas.append((ESCALAS_MB[esc], engine, d["status"]))
        if not xs:
            continue
        ax.plot(xs, ys, color=COR[engine], linewidth=2.0,
                marker=MARCADOR[engine], markersize=8,
                markeredgecolor=FUNDO, markeredgewidth=1.2, zorder=3)
        pontas.append((xs[-1], ys[-1], engine, f"{ys[-1]:.1f} s"))

    for rotulo, d in (variantes or []):
        pontas += _tracar_variante(ax, d, escalas, rotulo, "tempo", falhas)

    ax.set_yscale("log")
    _moldura(ax, "Tempo do Stage A por volume de dados",
             "Tempo (s) — mediana de 3 execuções", escalas)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50",
                        "100", "200", "500", "1000"])
    _rotulos_sem_colisao(ax, pontas)
    _marcar_falhas(ax, falhas)

    ax.annotate("Eixos logarítmicos: uma reta significa custo proporcional ao "
                "volume; inclinação maior,\ncusto que cresce mais rápido que "
                "os dados. O X marca a escala em que a engine não concluiu.",
                xy=(0.0, -0.17), xycoords="axes fraction",
                color=TINTA_MUDA, fontsize=8, va="top")
    return _salvar(fig, "tempo_por_escala", dir_saida)


def grafico_memoria(dados: dict, dir_saida: str = DIR_FIGURAS,
                    variantes: list[tuple[str, dict]] | None = None) -> list[str]:
    escalas = _escalas_presentes(dados)
    if variantes:
        for _, d in variantes:
            escalas = sorted(set(escalas) | set(_escalas_presentes(d)),
                             key=lambda e: ESCALAS_MB[e])
    fig, ax = _figura()

    pontas = []
    for engine in ENGINES:
        xs, ys = [], []
        rupturas: list[tuple[float, float, str]] = []
        for esc in escalas:
            d = dados.get((engine, esc))
            if d and d["memoria"] is not None:
                xs.append(ESCALAS_MB[esc])
                ys.append(d["memoria"] / 1024)
            elif d:
                rupturas.append((ESCALAS_MB[esc],
                                 d.get("pico_falha", TRAVA_MB) / 1024,
                                 d["status"]))
        if not xs:
            continue
        ax.plot(xs, ys, color=COR[engine], linewidth=2.0,
                marker=MARCADOR[engine], markersize=8,
                markeredgecolor=FUNDO, markeredgewidth=1.2, zorder=3)

        # Aqui — diferente do gráfico de tempo — a falha TEM valor medido: é o
        # consumo no instante em que a trava encerrou a execução. O traço é
        # tracejado porque a engine ainda queria mais memória; o que se mede é
        # onde ela bateu, não onde teria parado.
        if rupturas:
            x_r, y_r, status = rupturas[0]
            ax.plot([xs[-1], x_r], [ys[-1], y_r], color=COR[engine],
                    linewidth=1.6, linestyle=(0, (4, 3)), zorder=3)
            for x_r, y_r, status in rupturas:
                ax.plot([x_r], [y_r], marker="X", markersize=11,
                        color=COR[engine], markeredgecolor=FUNDO,
                        markeredgewidth=1.5, zorder=5)
            x_ult, y_ult, status_ult = rupturas[-1]
            pontas.append((x_ult, y_ult, engine,
                           f"{y_ult:.1f} GB · {status_ult}"))
            continue
        pontas.append((xs[-1], ys[-1], engine, f"{ys[-1]:.1f} GB"))

    for rotulo, d in (variantes or []):
        pontas += _tracar_variante(ax, d, escalas, rotulo, "memoria")
    # No grafico de memoria a variante nao entra na faixa de falhas: ali o
    # ponto de ruptura tem valor medido e e desenhado dentro da area.

    ax.axhline(TRAVA_MB / 1024, color=TINTA_MUDA, linewidth=1.4,
               linestyle=(0, (5, 4)), zorder=2)
    ax.annotate("trava do harness — 8,3 GB",
                xy=(0.005, TRAVA_MB / 1024), xycoords=("axes fraction", "data"),
                xytext=(0, 6), textcoords="offset points",
                color=TINTA_2, fontsize=8, fontweight="bold")

    ax.set_ylim(0, max(TRAVA_MB / 1024 * 1.12, 1))
    _moldura(ax, "Pico de memória do Stage A por volume de dados",
             "Pico de memória (GB) — mediana de 3 execuções", escalas,
             pad_titulo=18)
    _rotulos_sem_colisao(ax, pontas)

    ax.annotate("O X marca onde a engine foi encerrada por atingir a trava. O "
                "consumo ali foi medido,\nmas é um piso: a engine ainda pedia "
                "memória quando foi interrompida.",
                xy=(0.0, -0.15), xycoords="axes fraction",
                color=TINTA_MUDA, fontsize=8, va="top")
    return _salvar(fig, "memoria_por_escala", dir_saida)


def tabela_recomendacoes(dados: dict) -> str:
    """
    Tabela de recomendações, derivada dos dados — não da impressão do autor.

    Para cada escala, ordena as engines que concluíram e registra as que
    falharam. A recomendação é a mais rápida entre as que couberam.
    """
    escalas = _escalas_presentes(dados)
    linhas = ["| Volume | Recomendada | Tempo | 2º lugar | Não concluíram |",
              "|---|---|---|---|---|"]
    for esc in escalas:
        ok = sorted(
            ((d["tempo"], eng) for (eng, e), d in dados.items()
             if e == esc and d["tempo"] is not None),
        )
        falhas = sorted(NOME[eng] for (eng, e), d in dados.items()
                        if e == esc and d["tempo"] is None)
        if not ok:
            linhas.append(f"| {ROTULO_ESCALA[esc]} | — | — | — | "
                          f"{', '.join(falhas) or '—'} |")
            continue
        melhor_t, melhor = ok[0]
        segundo = (f"{NOME[ok[1][1]]} ({ok[1][0]:.1f} s, "
                   f"{ok[1][0] / melhor_t:.1f}x)") if len(ok) > 1 else "—"
        linhas.append(f"| {ROTULO_ESCALA[esc]} | **{NOME[melhor]}** | "
                      f"{melhor_t:.1f} s | {segundo} | "
                      f"{', '.join(falhas) or '—'} |")
    return "\n".join(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gráficos e tabela do benchmark")
    ap.add_argument("--entrada", default=ARQUIVO_PADRAO)
    ap.add_argument("--saida", default=DIR_FIGURAS)
    ap.add_argument("--etapas", metavar="CSV",
                    help="CSV do modo instrumentado (--staged); gera o gráfico "
                         "de composição do tempo por etapa, que responde à QP2")
    ap.add_argument("--etapas-escalas", nargs="+", default=["200mb"],
                    help="escalas do gráfico de etapas (padrão: 200mb)")
    ap.add_argument("--variante", action="append", default=[], metavar="RÓTULO=CSV",
                    help="série adicional de uma implementação alternativa da "
                         "mesma engine, desenhada tracejada na mesma cor "
                         "(ex.: 'reformulado=results/benchmark_polars_otimizado.csv')")
    args = ap.parse_args()

    if not os.path.exists(args.entrada):
        print(f"não encontrado: {args.entrada}")
        return 1

    dados = carregar(args.entrada)
    if not dados:
        print("nenhuma execução utilizável no CSV")
        return 1

    variantes: list[tuple[str, dict]] = []
    for spec in args.variante:
        if "=" not in spec:
            print(f"--variante espera RÓTULO=CSV, recebido: {spec}")
            return 1
        rotulo, caminho = spec.split("=", 1)
        if not os.path.exists(caminho):
            print(f"não encontrado: {caminho}")
            return 1
        variantes.append((rotulo, carregar(caminho)))
        print(f"variante '{rotulo}': {caminho}")

    for caminho in grafico_tempo(dados, args.saida, variantes):
        print(f"gerado: {caminho}")
    for caminho in grafico_memoria(dados, args.saida, variantes):
        print(f"gerado: {caminho}")

    if args.etapas:
        if not os.path.exists(args.etapas):
            print(f"não encontrado: {args.etapas}")
            return 1
        for escala in args.etapas_escalas:
            por_etapa = carregar_etapas(args.etapas, escala)
            if not por_etapa:
                print(f"sem execuções utilizáveis em {escala}")
                continue
            for caminho in grafico_etapas(por_etapa, escala, args.saida):
                print(f"gerado: {caminho}")

    print("\n" + tabela_recomendacoes(dados))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
