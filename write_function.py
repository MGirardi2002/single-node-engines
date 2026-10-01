import re
import boto3

from typing import Dict, Optional, Any, List
from urllib.parse import urlparse
from datetime import datetime

from awsglue.context import GlueContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

def write_text_single_file(df, final_uri: str, tmp_suffix: str = "_tmp") -> None:
    s3_client = boto3.client("s3")

    parsed = urlparse(final_uri)
    bucket = parsed.netloc
    final_key = parsed.path.lstrip("/")

    # diretorio temp
    tmp_prefix = final_key + tmp_suffix
    tmp_uri = f"s3://{bucket}/{tmp_prefix}"

    # 1) escreve como texto no dir temp
    df.coalesce(1).write.mode("overwrite").text(tmp_uri)

    # 2) acha o unico part-*.txt gerado
    resp = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=tmp_prefix + "/"
        )
    contents = resp.get("Contents", [])
    
    if not contents:
        raise RuntimeError(f"Não encontrou arquivos em {tmp_uri}")

    part_obj = None
    for obj in contents:
        if obj["Key"].endswith(".txt"):
            part_obj = obj
            break
    
    if not part_obj:
        raise RuntimeError(f"Não encontrou arquivo part-*.txt em {tmp_uri}")

    # 3) lê o conteúdo, remove o \n final e grava no destino
    body = s3_client.get_object(Bucket=bucket, Key=part_obj["Key"])["Body"].read()
    body = body.rstrip(b"\n")

    s3_client.put_object(
        Bucket=bucket,
        Key=final_key,
        Body=body
    )

    # 4) apaga o diretorio temp
    s3_client.delete_objects(
        Bucket=bucket,
        Delete={
            "Objects": [
                {"Key": obj["Key"]} for obj in contents
            ]
        }
    )

    print(f"[INFO] Arquivo de texto gravado em S3 como objeto único: {final_uri}")
    
    
     
