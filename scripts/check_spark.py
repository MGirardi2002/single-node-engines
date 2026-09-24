"""
Go/no-go do PySpark: verifica que a engine consegue executar todas as operações
que o pipeline do benchmark exige.

Não basta a SparkSession subir — no Windows ela sobe e só falha na primeira
leitura de arquivo (UnsatisfiedLinkError por falta das bibliotecas nativas do
Hadoop). Por isso este teste exercita read / groupBy / window / join / write.

Uso (dentro do WSL):
    $HOME/tcc-venv/bin/python scripts/check_spark.py
"""

import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

tmp = tempfile.mkdtemp(prefix="sparkcheck_")
src = os.path.join(tmp, "tx")
os.makedirs(src)

rng = np.random.default_rng(0)
N = 200_000
for c in range(3):
    pd.DataFrame({
        "transacao_id": np.arange(N) + c * N,
        "cliente_id": rng.integers(1, 5_000, N),
        "data_hora": pd.to_datetime(
            np.datetime64("2025-01-01", "s").astype("int64")
            + rng.integers(0, 365 * 86_400, N), unit="s"),
        "valor": rng.lognormal(4, 0.5, N).round(2),
        "categoria": rng.choice(["mercado", "farmacia", "lazer", "viagem"], N),
        "canal": rng.choice(["pos", "app", "web"], N),
    }).to_parquet(f"{src}/part-{c:04d}.parquet", index=False)

print(f"[setup] 3 arquivos Parquet em {src}")
print(f"[setup] JAVA_HOME = {os.environ.get('JAVA_HOME', '(não definido)')}\n")

try:
    import pyspark
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    print(f"pyspark {pyspark.__version__} | python {sys.version.split()[0]}")

    spark = (
        SparkSession.builder
        .appName("gonogo")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"[OK] SparkSession        Spark {spark.version}\n")

    df = spark.read.parquet(src)
    print(f"[OK] read.parquet        {df.count():,} linhas")

    agg = df.groupBy("cliente_id").agg(
        F.count("*").alias("n_tx"),
        F.sum("valor").alias("total"),
        F.avg("valor").alias("ticket"),
        F.countDistinct("categoria").alias("n_cat"),
        F.max("data_hora").alias("ultima"),
    )
    print(f"[OK] groupBy / agg       {agg.count():,} clientes")

    w = Window.partitionBy("cliente_id").orderBy("data_hora")
    win = (df.withColumn("anterior", F.lag("valor").over(w))
             .withColumn("delta", F.col("valor") - F.col("anterior")))
    print(f"[OK] window (lag)        {win.filter(F.col('delta').isNotNull()).count():,} linhas")

    joined = win.join(agg, on="cliente_id", how="inner")
    print(f"[OK] join                {joined.count():,} linhas")

    out = os.path.join(tmp, "out")
    agg.write.mode("overwrite").parquet(out)
    n_out = len([f for f in os.listdir(out) if f.endswith(".parquet")])
    print(f"[OK] write.parquet       {n_out} arquivo(s)")

    spark.stop()
    print("\n" + "=" * 46)
    print(" GO — PySpark apto para o benchmark")
    print("=" * 46)
except Exception:
    print("\n" + "=" * 46)
    print(" NO-GO")
    print("=" * 46)
    traceback.print_exc()
    sys.exit(1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
