# bq_utils.py (compat, sin warnings y más rápido si tienes Storage API)
from typing import List, Optional, Dict
import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from settings import BQ_PROJECT_ID, BQ_DATASET, SERVICE_ACCOUNT_JSON

def get_bq_client() -> bigquery.Client:
    if not BQ_PROJECT_ID:
        raise ValueError("BQ_PROJECT_ID no está definido en .env")
    if SERVICE_ACCOUNT_JSON and os.path.exists(SERVICE_ACCOUNT_JSON):
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
        return bigquery.Client(project=BQ_PROJECT_ID, credentials=creds)
    return bigquery.Client(project=BQ_PROJECT_ID)

def list_dataset_tables(client: bigquery.Client, dataset: str) -> List[str]:
    full = f"{client.project}.{dataset}"
    tables = [t.table_id for t in client.list_tables(full)]
    if not tables:
        raise ValueError(f"No se encontraron tablas en {full}")
    return tables

def table_fqn(project: str, dataset: str, table_id: str) -> str:
    return f"{project}.{dataset}.{table_id}"

def fetch_table_df(
    client: bigquery.Client,
    table_fqn_str: str,
    include_columns: Optional[List[str]] = None,
    where_sql: Optional[str] = None,
    limit_rows: Optional[int] = None,
) -> pd.DataFrame:
    cols = "*"
    if include_columns:
        safe_cols = [f"`{c.strip()}`" for c in include_columns if c.strip()]
        cols = ", ".join(safe_cols) if safe_cols else "*"
    query = [f"SELECT {cols} FROM `{table_fqn_str}`"]
    if where_sql and where_sql.strip():
        query.append(f"WHERE {where_sql}")
    if limit_rows and int(limit_rows) > 0:
        query.append(f"LIMIT {int(limit_rows)}")
    sql = "\n".join(query)
    return run_query_df(client, sql)

def get_table_schema(client: bigquery.Client, table_fqn_str: str) -> str:
    tbl = client.get_table(table_fqn_str)
    return ", ".join([f"{f.name}({f.field_type})" for f in tbl.schema])

def build_schema_registry(client: bigquery.Client, dataset: str) -> Dict[str, str]:
    schemas = {}
    for t in list_dataset_tables(client, dataset):
        fqn = table_fqn(client.project, dataset, t)
        schemas[fqn] = get_table_schema(client, fqn)
    return schemas

def run_query_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise ValueError("Solo se permiten consultas SELECT.")
    job = client.query(sql)
    rows = job.result()

    # Intentar BigQuery Storage API si está instalado (rápido, sin warnings)
    try:
        from google.cloud import bigquery_storage
        if getattr(client, "_credentials", None) is not None:
            bqs = bigquery_storage.BigQueryReadClient(credentials=client._credentials)
        else:
            bqs = bigquery_storage.BigQueryReadClient()
        return rows.to_dataframe(bqstorage_client=bqs)
    except Exception:
        # Fallback forzado por REST, sin intentar crear el cliente de Storage => sin warning
        return rows.to_dataframe(create_bqstorage_client=False)
