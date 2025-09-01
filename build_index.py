import os
import json
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from settings import (
    OPENAI_API_KEY, EMBEDDING_MODEL, INDEX_DIR,
    BQ_TABLE, INCLUDE_COLUMNS, WHERE_SQL, LIMIT_ROWS
)
from bq_utils import get_bq_client, fetch_table_df

def row_to_text(row: dict, order: list) -> str:
    return " | ".join([f"{col}: {row.get(col, '')}" for col in order])

def main():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY no está configurado.")

    client = get_bq_client()
    print(f"Cargando datos de {BQ_TABLE}...")
    df = fetch_table_df(
        client,
        BQ_TABLE,
        include_columns=INCLUDE_COLUMNS if INCLUDE_COLUMNS else None,
        where_sql=WHERE_SQL,
        limit_rows=LIMIT_ROWS if LIMIT_ROWS > 0 else None,
    )
    if df.empty:
        raise ValueError("La consulta no devolvió filas.")

    cols = list(df.columns)
    print(f"Columnas indexadas: {cols}")

    docs = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        rec = row.to_dict()
        text = row_to_text(rec, cols)
        metadata = {"row_index": int(i), "row_json": json.dumps(rec, ensure_ascii=False, default=str)}
        docs.append(Document(page_content=text, metadata=metadata))

    print("Creando embeddings y FAISS...")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(docs, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectordb.save_local(INDEX_DIR)
    print(f"Índice guardado en: {INDEX_DIR}")

if __name__ == "__main__":
    main()
