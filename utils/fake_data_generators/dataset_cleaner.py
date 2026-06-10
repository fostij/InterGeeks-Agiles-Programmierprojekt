import pandas as pd
from pathlib import Path

from utils.fake_data_generators.accident_description_generator import generate_accident_description


USD_TO_EUR_RATE = 0.92
INPUT_FILE = Path("dataset.csv")
OUTPUT_FILE = Path("dataset_prepared.csv")
CLEANUP_LOG_FILE = Path("dataset_cleanup_report.txt")

def clean_dataset(dataframe: pd.DataFrame, log_path: Path = CLEANUP_LOG_FILE) -> pd.DataFrame:
    cleaned = dataframe.copy()
    original_rows = len(cleaned)

    null_columns = [column for column in cleaned.columns if cleaned[column].isna().all()]
    if null_columns:
        cleaned = cleaned.drop(columns=null_columns)

    rows_with_nulls = cleaned[cleaned.isna().any(axis=1)]
    cleaned = cleaned.dropna(axis=0, how="any")

    duplicate_rows = cleaned[cleaned.duplicated(keep="first")]
    cleaned = cleaned.drop_duplicates(keep="first")

    report_lines = [
        f"Original rows: {original_rows}",
        f"Removed fully empty columns ({len(null_columns)}): {', '.join(null_columns) if null_columns else 'none'}",
        f"Removed rows with null values: {len(rows_with_nulls)}",
        f"Removed duplicate rows: {len(duplicate_rows)}",
        f"Final rows: {len(cleaned)}",
    ]

    if not rows_with_nulls.empty:
        report_lines.append("")
        report_lines.append("Rows removed because of null values:")
        report_lines.extend(f"- row_index={index}" for index in rows_with_nulls.index.tolist())

    if not duplicate_rows.empty:
        report_lines.append("")
        report_lines.append("Duplicate rows removed:")
        report_lines.extend(f"- row_index={index}" for index in duplicate_rows.index.tolist())

    log_path.write_text("\n".join(report_lines), encoding="utf-8")
    return cleaned

def build_prepared_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(
        {
            "auto_make_model": dataframe["auto_make"].astype(str).str.strip() + " " + dataframe["auto_model"].astype(str).str.strip(),
            "auto_year": dataframe["auto_year"],
            "incident_severity": dataframe["incident_severity"],
        }
    )

    collision_candidates = dataframe[["collision_type", "incident_type"]].astype("string").apply(lambda column: column.str.strip())
    collision_candidates = collision_candidates.replace({"": pd.NA, "?": pd.NA})
    prepared["collision_type"] = collision_candidates.bfill(axis=1).iloc[:, 0].fillna("Unknown")

    claim_candidates = pd.DataFrame(
        {
            "total_claim_amount": pd.to_numeric(dataframe["total_claim_amount"], errors="coerce"),
            "vehicle_claim": pd.to_numeric(dataframe["vehicle_claim"], errors="coerce"),
        }
    )
    claim_amount_usd = claim_candidates.bfill(axis=1).iloc[:, 0].fillna(0.0)
    prepared["claim_amount_eur"] = claim_amount_usd.mul(USD_TO_EUR_RATE).round(2)
    prepared["generated_description"] = prepared.apply(generate_accident_description, axis=1)

    return prepared

def fix_name_errors(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["auto_make"] = dataframe["auto_make"].replace(
        {"Accura": "Acura", "Suburu": "Subaru"}
        )
    return dataframe

def get_prepared_dataset() -> pd.DataFrame:
    dataframe = pd.read_csv(INPUT_FILE)
    cleaned = clean_dataset(dataframe)
    fixed = fix_name_errors(cleaned)
    prepared = build_prepared_dataset(fixed)
    #prepared.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    return prepared