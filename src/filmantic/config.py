from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
MOVIELENS_PATH = RAW_DATA_PATH / "ml-latest-small"
