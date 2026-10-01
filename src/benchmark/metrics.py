"""
Amostragem de memória e CPU durante uma execução do benchmark.

Mede o SISTEMA inteiro (o WSL dedicado ao experimento), descontando o consumo
de base registrado imediatamente antes de a execução começar.

Por que não `tracemalloc`
-------------------------
Ele instrumenta apenas o heap do interpretador Python. Polars aloca em Rust,
DuckDB em C++ e o Spark na JVM — em três das quatro engines ele reportaria
essencialmente zero.

Por que não somar a memória de cada processo
--------------------------------------------
Foram tentadas duas versões por processo, e ambas falharam no Spark, que roda
como uma árvore de processos (Python, JVM e workers Python criados por fork):

1. **Soma de RSS.** O RSS inclui páginas compartilhadas entre processos, que
   entram na soma várias vezes. Resultado: pico de 20 GB num ambiente limitado
   a 12 GB — fisicamente impossível.

2. **Soma de PSS.** O PSS divide as páginas compartilhadas entre os processos,
   mas lê-lo exige percorrer /proc/<pid>/smaps, que numa JVM tem milhares de
   entradas. O amostrador passou a disputar CPU com a engine e deixou o Spark
   quase 4x mais lento (37 s -> 140 s), além de ainda produzir valores acima do
   limite físico. Um instrumento que altera o que mede invalida o benchmark.

Por que medir o sistema funciona
--------------------------------
`total - disponível` (de /proc/meminfo) é uma leitura única e barata, não conta
nada em dobro e captura automaticamente todos os processos que a engine criar.
O cache de páginas de arquivos fica fora da conta, pois entra como memória
disponível.

A premissa é que o ambiente esteja dedicado ao experimento: nenhum outro
processo relevante rodando no WSL durante as medições. O consumo ocioso do
próprio sistema é removido pela subtração da linha de base.
"""

from __future__ import annotations

import os
import threading
import time

import psutil


def memoria_em_uso() -> int:
    """Bytes efetivamente ocupados no sistema (exclui cache recuperável)."""
    vm = psutil.virtual_memory()
    return vm.total - vm.available


# Menor memória em uso já observada no processo. Estimativa do patamar ocioso
# do sistema, usada por `aguardar_estabilizacao`. Começa no infinito e só cai:
# a primeira leitura da bateria já a define.
_piso_observado: float = float("inf")

# Pico abaixo do qual uma execução bem-sucedida é considerada implausível.
# Qualquer engine lendo ao menos 100 MB de Parquet ocupa centenas de MB; um
# valor menor que isto denuncia linha de base contaminada, não economia de
# memória. Serve de rede de segurança independente da espera acima.
PICO_MINIMO_PLAUSIVEL_MB = 200.0

# Jiffies por segundo (100 no Linux). /proc/stat contabiliza em jiffies.
try:
    _JIFFIES_POR_SEG = os.sysconf("SC_CLK_TCK") or 100
except (ValueError, AttributeError, OSError):
    _JIFFIES_POR_SEG = 100


def _cpu_acumulado() -> tuple[int, int]:
    """
    Tempo de CPU acumulado desde o boot, em jiffies: (ocupado, espera_io).

    `ocupado` soma user, nice, system, irq, softirq e steal — exclui idle e
    iowait, que não são trabalho. `espera_io` é reportado à parte porque
    distingue carga de processador de carga de disco, o que importa nas engines
    que transbordam.

    Fora do Linux devolve (0, 0): o harness roda no WSL, mas o módulo é
    importado por ferramentas de análise que rodam no Windows.
    """
    try:
        with open("/proc/stat", encoding="utf-8") as fp:
            campos = fp.readline().split()
    except OSError:
        return 0, 0
    # cpu user nice system idle iowait irq softirq steal guest guest_nice
    v = [int(x) for x in campos[1:9]]
    ocupado = v[0] + v[1] + v[2] + v[5] + v[6] + v[7]
    return ocupado, v[4]


def aguardar_estabilizacao(tolerancia_mb: float = 100.0, estaveis: int = 3,
                           timeout: float = 120.0,
                           intervalo: float = 0.5) -> tuple[int, bool]:
    """
    Espera a memória da execução anterior ser devolvida ao sistema.

    Por que isto é necessário
    -------------------------
    O `Amostrador` reporta o pico ACIMA da linha de base que registra ao ser
    criado. Em execuções encadeadas — e a bateria oficial encadeia aquecimento
    e três repetições —, o sistema pode ainda não ter devolvido a memória da
    execução anterior quando essa base é tomada. A base sai inflada e o pico
    medido sai baixo demais.

    Medido em 23/09, três execuções do Spark em 400 MB, em sequência:

        heap 4g -> 4.966 MB
        heap 5g -> 5.956 MB
        heap 6g -> 1.400 MB   <- impossível para uma JVM de 6 GB

    O terceiro valor não é um resultado, é a contaminação da linha de base pela
    execução anterior. Sem esta espera, as repetições posteriores de cada par
    sairiam sistematicamente subestimadas, e nada na saída denunciaria o erro.

    O critério
    ----------
    Exige DUAS condições ao mesmo tempo:

    1. a leitura parou de cair (a memória anterior foi devolvida);
    2. ela está perto do **menor valor já observado** nesta bateria, que serve
       de estimativa do patamar ocioso e se recalibra sozinho para baixo.

    As duas são necessárias, e cada uma cobre a falha da outra. Uma primeira
    versão comparava com uma referência tomada no início da bateria e falhou
    quando a própria referência nasceu contaminada. A segunda exigia apenas
    "parar de cair" e também falhou: a liberação de memória acontece em
    degraus, e um platô intermediário satisfazia o critério. Foi assim que a
    bateria de 23/09 registrou 12 MB de pico num aquecimento do Pandas em
    200 MB — impossível, e sintoma de base tomada alto demais.

    Só a QUEDA é aguardada: ela indica memória sendo liberada. Uma subida vem
    de outra atividade no sistema, e esperar não a resolveria.

    Devolve a memória em uso ao fim da espera e se ela convergiu dentro do
    prazo. Um `False` não interrompe a bateria: é registrado como aviso, para
    que a execução afetada possa ser descartada na análise.
    """
    global _piso_observado
    tol = int(tolerancia_mb * 1024**2)
    prazo = time.monotonic() + timeout
    anterior = memoria_em_uso()
    consecutivas = 0
    janela: list[int] = []

    while time.monotonic() < prazo:
        time.sleep(intervalo)
        atual = memoria_em_uso()

        # O piso só desce com um nível SUSTENTADO. Tomar o menor valor já visto
        # torna a estimativa refém de um mergulho instantâneo: uma única
        # leitura baixa passaria a reprovar todas as esperas seguintes, que
        # nunca mais alcançariam aquele nível. Usar o MAIOR valor de uma janela
        # de leituras consecutivas exige que a queda tenha se mantido.
        #
        # Isto não é hipotético: na bateria de 23/09 a última execução do Spark
        # em 1,6 GB foi marcada com aviso porque o piso havia ficado preso
        # abaixo do patamar ocioso real — a medição estava correta, o detector
        # é que estava estrito demais.
        janela.append(atual)
        if len(janela) > estaveis:
            janela.pop(0)
        if len(janela) == estaveis:
            _piso_observado = min(_piso_observado, max(janela))

        parou_de_cair = anterior - atual < tol
        perto_do_piso = atual <= _piso_observado + tol
        if parou_de_cair and perto_do_piso:
            consecutivas += 1
            if consecutivas >= estaveis:
                return atual, True
        else:
            consecutivas = 0
        anterior = atual
    return memoria_em_uso(), False


class Amostrador(threading.Thread):
    """
    Registra, enquanto está ativo, o pico de memória acima da linha de base, o
    uso de CPU do sistema e o TEMPO DE CPU acumulado.

    Deve ser iniciado ANTES de lançar a execução medida, para que a linha de
    base não inclua memória da própria engine.

    Por que tempo de CPU, além da utilização percentual
    ---------------------------------------------------
    A utilização percentual não denuncia contaminação externa. Medido em 28/09,
    mesmo trabalho (DuckDB em 800 MB), duas ocasiões:

        23/09: 257,2 s de parede, CPU média 185%, memória 6.111 MB
        28/09: 175,7 s de parede, CPU média 183%, memória 6.118 MB

    Utilização e memória praticamente idênticas, 32% de diferença no tempo. A
    execução de 23/09 consumiu ~48% mais núcleo-segundos para produzir o mesmo
    resultado — sinal de disputa por CPU com o sistema hospedeiro ou de redução
    de clock. Nada disso aparecia no CSV, e a medição parecia boa.

    O tempo acumulado torna o problema visível: duas execuções do mesmo
    trabalho com consumo de núcleo-segundos diferente são incomparáveis, e
    agora isso se lê direto no resultado.

    (O campo `steal` de /proc/stat resolveria de forma mais direta, mas o
    Hyper-V não o preenche neste ambiente — verificado, vem zerado.)
    """

    def __init__(self, intervalo: float = 0.1):
        super().__init__(daemon=True)
        self.intervalo = intervalo
        self.n_cpus = psutil.cpu_count(logical=True) or 1
        self.base = memoria_em_uso()
        self.pico_mem = 0
        # Última leitura acima da base — consultada pelo runner para encerrar
        # a execução antes de o kernel entrar em disputa por memória.
        self.atual_mem = 0
        self.amostras_cpu: list[float] = []
        self._cpu0, self._io0 = _cpu_acumulado()
        self._cpu_jiffies = 0
        self._io_jiffies = 0
        self._parar = threading.Event()
        # cpu_percent(interval=None) mede em relação à chamada anterior e
        # devolve 0.0 na primeira; esta chamada só prepara a referência.
        psutil.cpu_percent(interval=None)

    def run(self) -> None:
        while not self._parar.is_set():
            time.sleep(self.intervalo)
            self.atual_mem = memoria_em_uso() - self.base
            self.pico_mem = max(self.pico_mem, self.atual_mem)
            # Convertido para "100% = um núcleo", a mesma convenção do `top`,
            # para que 1200% signifique os 12 núcleos saturados.
            self.amostras_cpu.append(psutil.cpu_percent(interval=None) * self.n_cpus)

    def parar(self) -> None:
        self._parar.set()
        self.join(timeout=2.0)
        cpu1, io1 = _cpu_acumulado()
        self._cpu_jiffies = max(cpu1 - self._cpu0, 0)
        self._io_jiffies = max(io1 - self._io0, 0)

    @property
    def pico_mem_mb(self) -> float:
        return max(self.pico_mem, 0) / 1024**2

    @property
    def cpu_segundos(self) -> float:
        """Núcleo-segundos de trabalho efetivo consumidos durante a execução."""
        return self._cpu_jiffies / _JIFFIES_POR_SEG

    @property
    def io_segundos(self) -> float:
        """
        Núcleo-segundos de espera por disco, do contador `iowait`.

        CUIDADO com a interpretação: `iowait` NÃO é tempo gasto fazendo I/O. É
        tempo em que o núcleo esteve OCIOSO havendo alguma tarefa bloqueada
        esperando disco — ou seja, um subconjunto do tempo ocioso, não do
        ocupado. Daí duas ressalvas:

        * não somar com `cpu_segundos`: descrevem estados distintos do núcleo
          (ocupado e ocioso), e a soma não significa nada;
        * o valor cai se outra carga ocupar os núcleos que estariam ociosos,
          mesmo que a espera por disco não tenha mudado.

        Serve como indicador DIRECIONAL. A distinção que ele sustenta com
        segurança é de ordem de grandeza — nas medições de 28/09, o Polars
        marcou 0 em todas as escalas enquanto o DuckDB marcou 2.947 em 1,6 GB.
        Quantificar com precisão o tempo em disco exigiria outra instrumentação.
        """
        return self._io_jiffies / _JIFFIES_POR_SEG

    @property
    def cpu_medio(self) -> float:
        """Média percentual de CPU; 100% = um núcleo saturado."""
        if not self.amostras_cpu:
            return 0.0
        return sum(self.amostras_cpu) / len(self.amostras_cpu)

    @property
    def cpu_pico(self) -> float:
        return max(self.amostras_cpu, default=0.0)
