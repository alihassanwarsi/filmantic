from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = PROCESSED_DATA_PATH / "embeddings.npy"
