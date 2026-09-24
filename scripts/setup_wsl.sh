#!/usr/bin/env bash
#
# Prepara o ambiente experimental do TCC dentro do WSL2 / Ubuntu.
#
# Por que WSL2 e não Windows: o PySpark depende das bibliotecas nativas do
# Hadoop (winutils.exe / hadoop.dll) para acessar o sistema de arquivos local no
# Windows, e não existe build comunitário correspondente ao Hadoop 3.5.0
# empacotado no PySpark 4.2. Além disso, Linux é o ambiente em que as quatro
# engines efetivamente rodam em produção — medir nele é mais representativo.
#
# Uso:
#   bash scripts/setup_wsl.sh
#
set -euo pipefail

VENV="$HOME/tcc-venv"
DATA="$HOME/tcc-data"

echo "=============================================="
echo " 1/4  Java (JDK 21) — exige sudo"
echo "=============================================="
if command -v java >/dev/null 2>&1; then
    echo "[skip] java já instalado: $(java -version 2>&1 | head -1)"
else
    sudo apt-get update -qq
    sudo apt-get install -y openjdk-21-jdk-headless
fi

JAVA_HOME_DETECTADO="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
echo "[ok] JAVA_HOME = $JAVA_HOME_DETECTADO"

echo
echo "=============================================="
echo " 2/4  Ambiente Python 3.13 (via uv, sem sudo)"
echo "=============================================="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "[erro] uv não encontrado. Instale com:"
    echo "       curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Python 3.13 e não o 3.14 do sistema: 3.13 é a versão mais nova em que o
# PySpark 4.2 foi verificado funcionando neste projeto.
uv python install 3.13
[ -d "$VENV" ] || uv venv --python 3.13 "$VENV"
VIRTUAL_ENV="$VENV" uv pip install \
    pandas polars duckdb pyarrow scikit-learn psutil pyspark

echo
echo "=============================================="
echo " 3/4  Diretório de dados (fora de /mnt/c)"
echo "=============================================="
# Crítico: dados em /mnt/c passam pelo driver 9p do WSL2 e ficam ordens de
# grandeza mais lentos, o que contaminaria todas as medições de I/O.
mkdir -p "$DATA"
echo "[ok] $DATA  ($(df -h "$DATA" | tail -1 | awk '{print $4}') livres)"

echo
echo "=============================================="
echo " 4/4  Variáveis de ambiente"
echo "=============================================="
MARCA="# --- TCC: ambiente do benchmark ---"
if grep -qF "$MARCA" "$HOME/.bashrc" 2>/dev/null; then
    echo "[skip] .bashrc já configurado"
else
    {
        echo ""
        echo "$MARCA"
        echo "export JAVA_HOME=\"$JAVA_HOME_DETECTADO\""
        echo "export TCC_DATA_DIR=\"$DATA\""
        echo "export PYSPARK_PYTHON=\"$VENV/bin/python\""
        echo "export PYSPARK_DRIVER_PYTHON=\"$VENV/bin/python\""
        # O hostname do WSL resolve para um endereço de loopback; sem isto o
        # Spark se liga ao IP da interface virtual, o que introduz latência e
        # falhas intermitentes de bind entre execuções.
        echo "export SPARK_LOCAL_IP=127.0.0.1"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    } >> "$HOME/.bashrc"
    echo "[ok] variáveis acrescentadas ao ~/.bashrc"
fi

echo
echo "=============================================="
echo " Pronto. Abra um shell novo (ou: source ~/.bashrc) e valide com:"
echo "   \$HOME/tcc-venv/bin/python scripts/check_spark.py"
echo "=============================================="
