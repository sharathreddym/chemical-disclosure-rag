"""Data ingestion: CSV → SQLite + ChromaDB."""

import sqlite3
import pandas as pd
import chromadb
from pathlib import Path

from app.config import DATABASE_PATH, CHROMA_PATH, CSV_PATH
from app.data.schema import CREATE_TABLE_SQL, CREATE_INDEXES_SQL


def load_csv(csv_path: str = None) -> pd.DataFrame:
    """Load and clean the CSV dataset.

    Cleaning steps:
    1. Strip whitespace from string columns
    2. Replace NaN with empty string for string columns
    3. Drop rows that are missing the primary key (CDPHId)
    """
    path = csv_path or CSV_PATH
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # Strip whitespace from string columns and fill NaN
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("").str.strip()

    # Drop rows missing the primary key
    before = len(df)
    df = df[df["CDPHId"].notna()]
    dropped_no_pk = before - len(df)
    if dropped_no_pk > 0:
        print(f"  [data quality] dropped {dropped_no_pk} rows missing CDPHId")

    return df


def deduplicate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove ONLY full-row exact duplicates from the dataset.

    A row is removed only if EVERY column value is identical to another row.
    Rows that share an ID but differ in any field (date, status, count, etc.)
    are kept as-is - they represent legitimate distinct reporting events.

    Returns the deduplicated dataframe and a quality report dict.
    """
    initial = len(df)

    # Drop exact duplicates - all columns must match
    df_clean = df.drop_duplicates(keep="first")
    full_dupes_removed = initial - len(df_clean)
    final = len(df_clean)

    report = {
        "initial_rows": initial,
        "full_row_duplicates_removed": full_dupes_removed,
        "final_rows": final,
        "duplicate_pct": round((full_dupes_removed / initial * 100) if initial else 0, 2),
    }
    return df_clean.reset_index(drop=True), report


def init_sqlite(df: pd.DataFrame, db_path: str = None) -> str:
    """Create SQLite database from DataFrame.

    Assumes the input dataframe has already been deduplicated of full-row
    duplicates by `deduplicate_dataframe`. Rows that share IDs but differ
    in any field (date, status, count) are KEPT as legitimate distinct
    reporting events.
    """
    path = db_path or DATABASE_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # Create table
    cursor.execute("DROP TABLE IF EXISTS chemicals_in_cosmetics;")
    cursor.execute(CREATE_TABLE_SQL)

    cols = [
        "CDPHId", "ProductName", "CSFId", "CSF", "CompanyId", "CompanyName",
        "BrandName", "PrimaryCategoryId", "PrimaryCategory", "SubCategoryId",
        "SubCategory", "CasId", "CasNumber", "ChemicalId", "ChemicalName",
        "InitialDateReported", "MostRecentDateReported", "DiscontinuedDate",
        "ChemicalCreatedAt", "ChemicalUpdatedAt", "ChemicalDateRemoved", "ChemicalCount",
    ]
    placeholders = ", ".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO chemicals_in_cosmetics ({', '.join(cols)}) VALUES ({placeholders})"

    records = df[cols].values.tolist()
    cursor.executemany(insert_sql, records)

    # Create performance indexes
    for idx_sql in CREATE_INDEXES_SQL:
        cursor.execute(idx_sql)

    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM chemicals_in_cosmetics;")
    final_count = cursor.fetchone()[0]
    conn.close()

    print(f"SQLite database created at {path} with {final_count} rows.")
    return path


def init_chromadb(df: pd.DataFrame, chroma_path: str = None) -> str:
    """Create ChromaDB vector store with product-chemical text embeddings."""
    import shutil

    path = chroma_path or CHROMA_PATH

    # If existing store is locked, use a new path
    if Path(path).exists():
        try:
            shutil.rmtree(path)
        except (PermissionError, OSError):
            import time
            path = path + f"_{int(time.time())}"
            print(f"  Old store locked, using new path: {path}")
    Path(path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=path)

    collection = client.get_or_create_collection(
        name="cosmetics_chemicals",
        metadata={"hnsw:space": "cosine"},
    )

    # SMART INDEX: Instead of embedding all 114K rows (slow), we index unique
    # entity values. The SQL agent handles exact lookups; vector search only needs
    # to resolve fuzzy/partial names to exact entity values.
    documents = []
    metadatas = []
    ids = []
    doc_id = 0

    # 1. Unique chemicals with their CAS numbers
    chemicals = df.drop_duplicates(subset=["ChemicalName", "CasNumber"])[["ChemicalName", "CasNumber", "ChemicalId"]].fillna("")
    for _, row in chemicals.iterrows():
        documents.append(f"Chemical: {row['ChemicalName']}. CAS Number: {row['CasNumber']}.")
        metadatas.append({"type": "chemical", "ChemicalName": str(row["ChemicalName"]),
                          "CasNumber": str(row["CasNumber"]), "ChemicalId": str(row["ChemicalId"])})
        ids.append(f"chem_{doc_id}")
        doc_id += 1
    print(f"  Chemicals: {len(chemicals)} unique entries")

    # 2. Unique companies
    companies = df.drop_duplicates(subset=["CompanyName"])[["CompanyName", "CompanyId"]].fillna("")
    for _, row in companies.iterrows():
        documents.append(f"Company: {row['CompanyName']}.")
        metadatas.append({"type": "company", "CompanyName": str(row["CompanyName"])})
        ids.append(f"comp_{doc_id}")
        doc_id += 1
    print(f"  Companies: {len(companies)} unique entries")

    # 3. Unique brands
    brands = df.drop_duplicates(subset=["BrandName"])[["BrandName", "CompanyName"]].fillna("")
    for _, row in brands.iterrows():
        documents.append(f"Brand: {row['BrandName']}. Company: {row['CompanyName']}.")
        metadatas.append({"type": "brand", "BrandName": str(row["BrandName"]),
                          "CompanyName": str(row["CompanyName"])})
        ids.append(f"brand_{doc_id}")
        doc_id += 1
    print(f"  Brands: {len(brands)} unique entries")

    # 4. Unique products (with their key associations)
    products = df.drop_duplicates(subset=["ProductName", "CompanyName", "BrandName"])[
        ["ProductName", "CompanyName", "BrandName", "PrimaryCategory", "SubCategory", "CDPHId"]
    ].fillna("")
    # Cap at 5000 to keep indexing fast; SQL handles the full dataset
    products = products.head(5000)
    for _, row in products.iterrows():
        documents.append(
            f"Product: {row['ProductName']}. Brand: {row['BrandName']}. "
            f"Company: {row['CompanyName']}. Category: {row['PrimaryCategory']} - {row['SubCategory']}."
        )
        metadatas.append({
            "type": "product", "ProductName": str(row["ProductName"]),
            "CompanyName": str(row["CompanyName"]), "BrandName": str(row["BrandName"]),
            "PrimaryCategory": str(row["PrimaryCategory"]), "SubCategory": str(row["SubCategory"]),
            "CDPHId": str(row["CDPHId"]),
        })
        ids.append(f"prod_{doc_id}")
        doc_id += 1
    print(f"  Products: {len(products)} entries (capped)")

    # 5. Unique categories
    cats = df.drop_duplicates(subset=["PrimaryCategory", "SubCategory"])[
        ["PrimaryCategory", "SubCategory", "PrimaryCategoryId", "SubCategoryId"]
    ].fillna("")
    for _, row in cats.iterrows():
        documents.append(f"Category: {row['PrimaryCategory']} - Subcategory: {row['SubCategory']}.")
        metadatas.append({"type": "category", "PrimaryCategory": str(row["PrimaryCategory"]),
                          "SubCategory": str(row["SubCategory"])})
        ids.append(f"cat_{doc_id}")
        doc_id += 1
    print(f"  Categories: {len(cats)} unique entries")

    # Batch insert all
    total = len(documents)
    print(f"  Total documents to index: {total}")
    batch_size = 500
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  ChromaDB: indexed {end}/{total}...")

    print(f"ChromaDB vector store created at {path} with {total} documents.")
    return path


def ingest_all(csv_path: str = None):
    """Run full data ingestion pipeline with quality reporting."""
    print("Loading CSV...")
    df = load_csv(csv_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    print("\nChecking for full-row duplicates...")
    df_clean, dq_report = deduplicate_dataframe(df)
    print(f"  Initial rows:                {dq_report['initial_rows']:,}")
    print(f"  Full-row duplicates removed: {dq_report['full_row_duplicates_removed']:,} "
          f"({dq_report['duplicate_pct']}%)")
    print(f"  Final rows:                  {dq_report['final_rows']:,}")
    print("  Note: rows that share IDs but differ in any field are kept as distinct.")

    print("\nCreating SQLite database...")
    init_sqlite(df_clean)

    print("\nCreating ChromaDB vector store...")
    init_chromadb(df_clean)

    print("\nData ingestion complete!")
    return df_clean


if __name__ == "__main__":
    ingest_all()
