import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = PROCESSED_DATA_PATH / "embeddings.npy"

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"
