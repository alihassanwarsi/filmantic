import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDINGS_PATH


def build_movie_text(df):
    """Combine movie metadata into text for semantic embedding."""
    texts = []

    for _, movie in df.iterrows():
        parts = (
            f"{movie['title']} "
            f"{movie['overview']} "
            f"{' '.join(movie['genres'])} "
            f"{' '.join(movie['keywords'])} "
            f"{' '.join(movie['cast'])} "
            f"{movie['director']}"
        )
        texts.append(parts)

    return texts


def generate_embeddings(texts, batch_size=64):
    """Generate normalized semantic embeddings."""

    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


def build_embeddings(df, rebuild=False, batch_size=64):
    """Load cached embeddings or generate them."""

    if EMBEDDINGS_PATH.exists() and not rebuild:
        embeddings = np.load(EMBEDDINGS_PATH)

        if embeddings.shape == (len(df), 384):
            return embeddings

    texts = build_movie_text(df)
    embeddings = generate_embeddings(texts, batch_size=batch_size)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    return embeddings


if __name__ == "__main__":
    from .recommender import build_recommender

    df, _ = build_recommender()
    embeddings = build_embeddings(df)

    print("Embeddings shape:", embeddings.shape)
