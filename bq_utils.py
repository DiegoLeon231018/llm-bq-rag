from typing import List, Optional
import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from settings import BQ_PROJECT_ID, SERVICE_ACCOUNT_JSON

def get_bq_client() -> bigquery.Client:
    if not BQ_PROJECT_ID:
        raise ValueError("BQ_PROJECT_ID no está definido en .env")
    if not SERVICE_ACCOUNT_JSON or not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise ValueError(f"SERVICE_ACCOUNT_JSON no apunta a un archivo válido: {SERVICE_ACCOUNT_JSON}")
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
    return bigquery.Client(project=BQ_PROJECT_ID, credentials=creds)

def fetch_table_df(
    client: bigquery.Client,
    table: str,
    include_columns: Optional[List[str]] = None,
    where_sql: Optional[str] = None,
    limit_rows: Optional[int] = None,
) -> pd.DataFrame:
    cols = "*"
    if include_columns:
        safe_cols = [f"`{c.strip()}`" for c in include_columns if c.strip()]
        cols = ", ".join(safe_cols) if safe_cols else "*"

    query = [f"SELECT {cols} FROM `{client.project}.{table}`"]
    if where_sql and where_sql.strip():
        query.append(f"WHERE {where_sql}")
    if limit_rows and int(limit_rows) > 0:
        query.append(f"LIMIT {int(limit_rows)}")

    sql = "\n".join(query)
    job = client.query(sql)
    return job.result().to_dataframe()

def get_table_schema(client: bigquery.Client, table: str) -> str:
    """Devuelve 'col(tipo), col(tipo), ...' para prompt SQL."""
    table_ref = f"{client.project}.{table}"
    tbl = client.get_table(table_ref)
    return ", ".join([f"{f.name}({f.field_type})" for f in tbl.schema])

def run_query_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    """Ejecuta SELECTs. Rechaza DML/DDL."""
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise ValueError("Solo se permiten consultas SELECT.")
    job = client.query(sql)
    return job.result().to_dataframe()
