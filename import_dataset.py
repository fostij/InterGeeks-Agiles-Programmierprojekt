import logging
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Professional Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

CSV_FILE = "dataset_prepared_merged.csv"
DB_URL = "postgresql+psycopg2://projekt_user:105105@localhost:5432/auto_versicherung_db"


def run_bulk_etl():
    logger.info("=== Starting Bulk Dataset Ingestion Pipeline ===")
    
    # 1. Extraction Layer
    if not os.path.exists(CSV_FILE):
        logger.error(f"Source file '{CSV_FILE}' not found in current directory.")
        return
        
    logger.info(f"Extracting records from '{CSV_FILE}'...")
    df = pd.read_csv(CSV_FILE)
    logger.info(f"Extraction successful. Found {len(df)} records to process.")
    
    # 2. Loading Layer (Bulk Persist)
    logger.info("Initializing connection to PostgreSQL 'auto_versicherung_db'...")
    engine = create_engine(DB_URL)
    
    try:
        with engine.begin() as connection:
            df.to_sql(
                name="unfaelle_dataset",
                con=connection,
                if_exists="replace",  # Changes to 'replace' to safely overwrite previous partial runs
                index=False
            )
        logger.info("Bulk upload operation successful! All database entries committed safely.")
        
        # 3. Verification Layer (FIXED using text())
        with engine.connect() as check_conn:
            query_count = text("SELECT COUNT(*) FROM unfaelle_dataset;")
            total_rows = check_conn.execute(query_count).scalar()
            
            query_select = text("SELECT auto_make_model, auto_year, claim_amount_eur  FROM unfaelle_dataset LIMIT 3;")
            sample_data = pd.read_sql(query_select, con=check_conn)
            
            print("\n" + "=" * 70)
            print(f" LIVE DATABASE VERIFICATION SNAPSHOT - TOTAL ENTRIES: {total_rows}")
            print("=" * 70)
            print(sample_data.to_string(index=False))
            print("=" * 70 + "\n")
            
    except SQLAlchemyError as error:
        logger.critical(f"Database tracking query failed: {error}")


if __name__ == "__main__":
    run_bulk_etl()
