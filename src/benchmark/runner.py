"""
Harness do benchmark: executa a matriz (engine x escala x repetição),
isolando cada execução em subprocesso e medindo tempo, memória e CPU.

Política de medição
-------------------
* **Aquecimento**: a primeira execução de cada par (engine, escala) é
  descartada. Ler o mesmo Parquet duas vezes seguidas faz a segunda vir do
  cache de página do SO e ser bem mais rápida; limpar esse cache de forma
  confiável não é viável, então a alternativa honesta é garantir que *todas*
  as execuções medidas estejam igualmente quentes.
* **Repetições**: 3 execuções medidas; reporta-se mediana e IQR.
* **Estabilização da linha de base**: antes de cada execução, espera-se que a
  memória do sistema volte ao patamar ocioso. Como o pico é medido ACIMA da
  base registrada no início da execução, medir antes de o sistema ter devolvido
  a memória da execução anterior produz um pico subestimado — em 23/09 isso
  registrou 1,4 GB numa execução do Spark com heap de 6 GB. Ver
  `metrics.aguardar_estabilizacao`.
* **Timeout**: encerra execuções que não terminam em tempo hábil, registrando
  `status=TIMEOUT` em vez de travar a bateria.
* **OOM**: um subprocesso encerrado por SIGKILL sem que o runner tenha pedido
  é registrado como `status=OOM`. Com o swap desligado, é o desfecho típico de
  estouro de memória no Linux — e é um resultado esperado do trabalho, não uma
  falha do experimento.
* **Isolamento de memória**: cada execução roda num *scope* do systemd com
  limite próprio (LIMITE_GB_PADRAO). Sem isso, na primeira tentativa em 2 GB, o
  OOM foi global: o kernel matou o worker e o systemd, em seguida, encerrou
  todos os processos do mesmo grupo — inclusive o próprio runner, que morreu
  sem gravar nada, e um processo do WSL. Com o scope, o OOM fica contido na
  execução medida, e o limite de memória de cada execução passa a ser explícito.
* **Gravação incremental**: cada resultado vai para o CSV assim que sai, para
  que uma interrupção no meio da bateria não perca o que já foi medido.

Uso:
    python -m src.benchmark.runner --escalas 100mb --engines pandas polars
    python -m src.benchmark.runner --escalas 100mb 500mb --repeticoes 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

from src.gerador import config as cfg
from src.pipelines import base

from .andamento import Andamento
from .metrics import (PICO_MINIMO_PLAUSIVEL_MB, Amostrador,
                      aguardar_estabilizacao, memoria_em_uso)

ENGINES = ("pandas", "polars", "duckdb", "spark")
TIMEOUT_PADRAO = 1800  # 30 min
ARQUIVO_RESULTADOS = "results/benchmark.csv"

# Memória disponível para cada execução, em GB. O WSL tem 10 GB (~9,7 GiB
# utilizáveis); o restante fica para o sistema e para o próprio runner.
LIMITE_GB_PADRAO = 8.5

# Fração do limite em que o PRÓPRIO runner encerra a execução.
#
# Por que não deixar só o kernel agir: perto do limite, o kernel passa minutos
# tentando liberar memória (descartando e relendo páginas de disco) antes de
# desistir. Num teste do Polars em 500 MB isso levou 14 minutos com disco e CPU
# no máximo, e deixou o Windows inutilizável. O runner já amostra a memória a
# cada 0,1 s, então pode encerrar a execução assim que ela cruza o limiar —
# o desfecho é o mesmo (OOM), só que em segundos. O limite do cgroup continua
# valendo como segunda barreira.
FRACAO_LIMIAR = 0.95


def _systemd_disponivel() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    teste = subprocess.run(["systemctl", "--user", "is-system-running"],
                           capture_output=True, text=True)
    return teste.stdout.strip() in ("running", "degraded")


SYSTEMD = _systemd_disponivel()


def _isolamento(limite_gb: float) -> list[str]:
    """
    Prefixo que lança o worker num scope do systemd com limite de memória.

    `OOMPolicy=continue` impede o systemd de encerrar o scope inteiro quando o
    kernel mata o processo; `MemorySwapMax=0` mantém a política sem swap.
    Se o systemd de usuário não estiver disponível, roda sem isolamento.
    """
    if not SYSTEMD:
        return []
    return ["systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={int(limite_gb * 1024)}M",
            "-p", "MemorySwapMax=0",
            "-p", "OOMPolicy=continue"]

CAMPOS = (
    "timestamp", "engine", "escala", "staged", "repeticao", "status",
    "tempo_total", "pico_mem_mb", "cpu_medio", "cpu_pico", "n_chunks",
    *(f"t_{e}" for e in base.ETAPAS),
    "erro",
)


def executar_um(engine: str, escala: str, staged: bool, timeout: int,
                limite_gb: float = LIMITE_GB_PADRAO,
                painel: Andamento | None = None, rotulo: str = "") -> dict:
    """Executa uma vez, em subprocesso, e devolve a linha de resultado."""
    saida = f"{cfg.DIR_DADOS}/features/{escala}/{engine}.parquet"
    fd, caminho_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # A memória da execução anterior precisa ter sido devolvida ao sistema
    # antes de o amostrador registrar a linha de base — senão o pico desta
    # execução sai subestimado. Ver `metrics.aguardar_estabilizacao`.
    assentada, estabilizou = aguardar_estabilizacao()
    aviso_base = "" if estabilizou else (
        f"linha de base não estabilizou em 120s (assentou em "
        f"{assentada / 1024**2:.0f} MB): o pico pode estar subestimado")

    cmd = [*_isolamento(limite_gb), sys.executable, "-m", "src.benchmark.worker",
           "--engine", engine, "--escala", escala,
           "--saida", saida, "--resultado", caminho_json]
    if staged:
        cmd.append("--staged")

    limiar = int(limite_gb * FRACAO_LIMIAR * 1024**3)

    # O amostrador registra a linha de base de memória ao ser criado, então
    # precisa existir ANTES de o subprocesso começar a alocar.
    amostrador = Amostrador()
    amostrador.start()
    # stderr vai para arquivo, não para PIPE: como o runner fica consultando o
    # processo em laço, um PIPE cheio (o Spark escreve muito log) travaria o
    # worker esperando alguém ler.
    with tempfile.TemporaryFile() as arq_stderr:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=arq_stderr,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        t0 = time.perf_counter()
        motivo = None  # None | "TIMEOUT" | "LIMIAR"
        while proc.poll() is None:
            time.sleep(0.2)
            if painel:
                cpu = amostrador.amostras_cpu[-1] if amostrador.amostras_cpu else 0.0
                painel.tick(rotulo, time.perf_counter() - t0,
                            amostrador.atual_mem / 1024**2,
                            amostrador.pico_mem_mb, cpu)
            if time.perf_counter() - t0 > timeout:
                motivo = "TIMEOUT"
            elif amostrador.atual_mem >= limiar:
                motivo = "LIMIAR"
            if motivo:
                proc.kill()
                proc.wait()
                break
        wall = time.perf_counter() - t0
        amostrador.parar()

        arq_stderr.seek(0)
        stderr = arq_stderr.read()[-4000:]

    linha = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "engine": engine, "escala": escala, "staged": staged,
        "pico_mem_mb": round(amostrador.pico_mem_mb, 1),
        "cpu_medio": round(amostrador.cpu_medio, 1),
        "cpu_pico": round(amostrador.cpu_pico, 1),
        "tempo_total": round(wall, 3),
        "erro": aviso_base,
    }

    if motivo == "TIMEOUT":
        linha["status"] = "TIMEOUT"
        return linha

    if motivo == "LIMIAR":
        linha["status"] = "OOM"
        linha["erro"] = (f"encerrado pelo harness: memória cruzou "
                         f"{limiar / 1024**2:.0f} MB "
                         f"({FRACAO_LIMIAR:.0%} de {limite_gb} GB)")
        return linha

    # SIGKILL sem pedido nosso: o kernel encerrou por falta de memória.
    if proc.returncode == -9:
        linha["status"] = "OOM"
        linha["erro"] = "encerrado pelo kernel (limite do cgroup)"
        return linha

    try:
        with open(caminho_json, encoding="utf-8") as fp:
            r = json.load(fp)
    except (OSError, json.JSONDecodeError):
        linha["status"] = "erro"
        linha["erro"] = stderr.decode(errors="replace")[-500:]
        return linha
    finally:
        os.unlink(caminho_json)

    linha["status"] = r["status"]
    if r["status"] == "ok":
        # Rede de segurança contra linha de base contaminada: uma execução que
        # terminou bem não pode ter ocupado quase nada. Sem esta marca, o
        # defeito passa despercebido — foi o que aconteceu em 23/09, quando um
        # aquecimento do Pandas em 200 MB registrou 12 MB de pico.
        if linha["pico_mem_mb"] < PICO_MINIMO_PLAUSIVEL_MB:
            linha["erro"] = (
                f"pico implausível ({linha['pico_mem_mb']:.0f} MB < "
                f"{PICO_MINIMO_PLAUSIVEL_MB:.0f} MB): linha de base "
                f"provavelmente contaminada — descartar esta execução")
        linha["n_chunks"] = r["n_chunks"]
        # Tempo de parede do subprocesso inteiro, incluindo import das
        # bibliotecas e (no Spark) a subida da JVM. É o custo real de rodar a
        # engine, e omiti-lo favoreceria artificialmente as mais pesadas.
        for etapa, t in r["tempos"].items():
            linha[f"t_{etapa}"] = round(t, 3)
    else:
        linha["erro"] = str(r.get("erro", ""))[:500]
    return linha


def gravar(linhas: list[dict], caminho: str) -> None:
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=CAMPOS, extrasaction="ignore")
        if novo:
            w.writeheader()
        for linha in linhas:
            w.writerow(linha)


def main() -> int:
    ap = argparse.ArgumentParser(description="Harness do benchmark")
    ap.add_argument("--engines", nargs="+", default=list(ENGINES),
                    choices=list(ENGINES))
    ap.add_argument("--escalas", nargs="+", required=True,
                    choices=list(cfg.ESCALAS))
    ap.add_argument("--repeticoes", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_PADRAO)
    ap.add_argument("--staged", action="store_true",
                    help="materializa por etapa (detalha, mas penaliza "
                         "engines preguiçosas — não use para comparar totais)")
    ap.add_argument("--sem-aquecimento", action="store_true",
                    help="pula a execução de aquecimento (só para depuração)")
    ap.add_argument("--saida", default=ARQUIVO_RESULTADOS)
    ap.add_argument("--limite-gb", type=float, default=LIMITE_GB_PADRAO,
                    help=f"memória por execução (padrão {LIMITE_GB_PADRAO} GB); "
                         "valores baixos servem para testar a trava do harness")
    args = ap.parse_args()

    total_execucoes = (len(args.engines) * len(args.escalas)
                       * (args.repeticoes + (0 if args.sem_aquecimento else 1)))
    print(f"{total_execucoes} execuções | timeout {args.timeout}s | "
          f"modo {'instrumentado' if args.staged else 'idiomático'}")
    limiar_mb = args.limite_gb * FRACAO_LIMIAR * 1024
    if SYSTEMD:
        print(f"limite de {args.limite_gb} GB por execução (scope do systemd); "
              f"o harness encerra ao cruzar {limiar_mb:.0f} MB\n")
    else:
        print(f"AVISO: systemd de usuário indisponível — sem isolamento de "
              f"cgroup; só a trava do harness ({limiar_mb:.0f} MB) protege.\n")

    print(f"memória em uso ao iniciar: {memoria_em_uso() / 1024**2:.0f} MB\n")

    n_gravadas = 0
    painel = Andamento(
        total=total_execucoes, limite_mb=limiar_mb,
        descricao=(f"Engines: {', '.join(args.engines)} · escalas: "
                   f"{', '.join(args.escalas)} · {args.repeticoes} repetição(ões)"
                   f"{'' if args.sem_aquecimento else ' + aquecimento'}"))
    print(f"acompanhe também em {os.path.abspath(painel.caminho)}\n")

    def registrar(r: dict) -> None:
        nonlocal n_gravadas
        gravar([r], args.saida)
        n_gravadas += 1

    for escala in args.escalas:
        for engine in args.engines:
            tempos: list[float] = []

            if not args.sem_aquecimento:
                rotulo = f"{engine} {escala} aquecimento"
                r = executar_um(engine, escala, args.staged, args.timeout,
                                args.limite_gb, painel, rotulo)
                painel.concluir(rotulo, r, "(descartado)")
                print(f"  {engine:<8} {escala:<6} aquecimento  {r['status']:<8} "
                      f"{r['tempo_total']:>8.2f}s  {r['pico_mem_mb']:>8.0f} MB  "
                      f"(descartado)")
                if r["status"] != "ok":
                    # Se nem o aquecimento passa, repetir 3x não ajuda.
                    r["repeticao"] = 0
                    registrar(r)
                    painel.pular(args.repeticoes, f"aquecimento terminou em {r['status']}")
                    print(f"           -> {r['status']}, pulando repetições")
                    if r["erro"]:
                        print(f"           {r['erro'][:200]}")
                    continue

            for i in range(1, args.repeticoes + 1):
                rotulo = f"{engine} {escala} rep {i}/{args.repeticoes}"
                r = executar_um(engine, escala, args.staged, args.timeout,
                                args.limite_gb, painel, rotulo)
                r["repeticao"] = i
                registrar(r)
                painel.concluir(rotulo, r)
                if r["status"] == "ok":
                    tempos.append(r["tempo_total"])
                print(f"  {engine:<8} {escala:<6} rep {i}/{args.repeticoes}  "
                      f"{r['status']:<8} {r['tempo_total']:>8.2f}s  "
                      f"{r['pico_mem_mb']:>8.0f} MB")

            if tempos:
                mediana = statistics.median(tempos)
                iqr = (max(tempos) - min(tempos)) if len(tempos) > 1 else 0.0
                print(f"           -> mediana {mediana:.2f}s "
                      f"(amplitude {iqr:.2f}s)\n")

    painel.finalizar()
    print(f"\n{n_gravadas} linhas gravadas em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
