"""
Acompanhamento ao vivo da bateria do benchmark.

Duas saídas, para dois jeitos de acompanhar:

* **Terminal** — uma linha que se reescreve a cada segundo com a execução
  atual, o tempo decorrido, a memória e a CPU. Só aparece quando a saída é um
  terminal de verdade; se o runner estiver gravando em arquivo, a linha
  sobrescrita viraria lixo no log e é omitida.

* **Arquivo `results/andamento.txt`** — reescrito a cada 2 segundos com a
  execução atual e a lista das já concluídas. Serve para acompanhar uma bateria
  rodando em segundo plano: aberto no VS Code, o arquivo se atualiza sozinho.

Escrever um arquivo pequeno a cada 2 s não interfere nas medições: é uma fração
de milissegundo de I/O, contra execuções de segundos a minutos.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

ARQUIVO = "results/andamento.txt"


def _mmss(segundos: float) -> str:
    m, s = divmod(int(segundos), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Andamento:
    INTERVALO_TERMINAL = 1.0
    INTERVALO_ARQUIVO = 2.0

    def __init__(self, total: int, limite_mb: float, descricao: str,
                 caminho: str = ARQUIVO):
        self.total = total
        self.feitas = 0
        self.limite_mb = limite_mb
        self.descricao = descricao
        self.caminho = caminho
        self.inicio = datetime.now()
        self.t0 = time.perf_counter()
        self.concluidas: list[str] = []
        self.terminal = sys.stdout.isatty()
        self._ult_terminal = 0.0
        self._ult_arquivo = 0.0
        self._largura = 0
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        self._escrever(atual=None)

    # -- durante uma execução ----------------------------------------------

    def tick(self, rotulo: str, decorrido: float, mem_mb: float,
             pico_mb: float, cpu: float) -> None:
        agora = time.perf_counter()
        n = self.feitas + 1

        if self.terminal and agora - self._ult_terminal >= self.INTERVALO_TERMINAL:
            linha = (f"  ▶ [{n}/{self.total}] {rotulo}  {_mmss(decorrido)}  "
                     f"mem {mem_mb / 1024:.1f}G (pico {pico_mb / 1024:.1f}G)  "
                     f"cpu {cpu:.0f}%")
            self._largura = max(self._largura, len(linha))
            print("\r" + linha.ljust(self._largura), end="", flush=True)
            self._ult_terminal = agora

        if agora - self._ult_arquivo >= self.INTERVALO_ARQUIVO:
            self._escrever(atual=(
                f"[{n}/{self.total}] {rotulo}\n"
                f"        tempo decorrido  {_mmss(decorrido)}\n"
                f"        memória          {mem_mb / 1024:.2f} GB agora · "
                f"pico {pico_mb / 1024:.2f} GB · trava em {self.limite_mb / 1024:.2f} GB\n"
                f"        CPU              {cpu:.0f}%  (100% = 1 núcleo)"))
            self._ult_arquivo = agora

    # -- entre execuções ------------------------------------------------------

    def limpar_linha(self) -> None:
        """Apaga a linha ao vivo antes de o runner imprimir um resultado."""
        if self.terminal and self._largura:
            print("\r" + " " * self._largura + "\r", end="", flush=True)
            self._largura = 0

    def concluir(self, rotulo: str, r: dict, observacao: str = "") -> None:
        self.limpar_linha()
        self.feitas += 1
        extra = f"  {observacao}" if observacao else ""
        self.concluidas.append(
            f"{rotulo:<32} {r['status']:<8} {r['tempo_total']:>8.1f}s  "
            f"{r['pico_mem_mb']:>7.0f} MB{extra}")
        self._escrever(atual=None)

    def pular(self, quantidade: int, motivo: str) -> None:
        """Execuções que não vão acontecer (ex.: aquecimento falhou)."""
        self.feitas += quantidade
        self.concluidas.append(f"  ↳ {quantidade} repetição(ões) pulada(s): {motivo}")
        self._escrever(atual=None)

    def finalizar(self) -> None:
        self.limpar_linha()
        self._escrever(atual=None, fim=True)

    # -- arquivo ----------------------------------------------------------------

    def _escrever(self, atual: str | None, fim: bool = False) -> None:
        decorrido = time.perf_counter() - self.t0
        estado = (f"CONCLUÍDO às {datetime.now():%H:%M:%S}" if fim
                  else "EM ANDAMENTO")
        partes = [
            f"Benchmark — {estado}",
            f"Iniciado em {self.inicio:%d/%m %H:%M:%S} · decorrido {_mmss(decorrido)}",
            self.descricao,
            f"Execuções: {self.feitas} de {self.total}",
            "",
        ]
        if atual and not fim:
            partes += ["AGORA", f"  {atual}", ""]
        partes += ["CONCLUÍDAS"]
        partes += [f"  {c}" for c in self.concluidas] or ["  (nenhuma ainda)"]
        texto = "\n".join(partes) + "\n"

        # Escreve num temporário e renomeia: quem estiver lendo o arquivo nunca
        # vê uma versão pela metade.
        tmp = self.caminho + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            fp.write(texto)
        os.replace(tmp, self.caminho)
