from collections import defaultdict

import numpy as np

from .embeddings import build_embeddings
from .recommender import (
    build_recommender,
    recommend_embedding,
    recommend_hybrid,
    recommend_movies,
)

K = 10


def build_relevant_sets(df):
    """For each movie, find ids of movies that share at least one genre."""

    genre_to_movies = defaultdict(set)

    for movie_id, genres in zip(df["movie_id"], df["genres"]):
        for genre in genres:
            genre_to_movies[genre].add(movie_id)

    relevant_sets = {}

    for movie_id, genres in zip(df["movie_id"], df["genres"]):
        relevant = set()
        for genre in genres:
            relevant |= genre_to_movies[genre]
        relevant.discard(movie_id)
        relevant_sets[movie_id] = relevant

    return relevant_sets


def precision_at_k(recommended_ids, relevant_ids, k=K):
    top_k = recommended_ids[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids, relevant_ids, k=K):
    top_k = recommended_ids[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(recommended_ids, relevant_ids, k=K):
    top_k = recommended_ids[:k]
    return 1.0 if any(mid in relevant_ids for mid in top_k) else 0.0


def ndcg_at_k(recommended_ids, relevant_ids, k=K):
    top_k = recommended_ids[:k]

    dcg = sum(
        1 / np.log2(rank + 2) for rank, mid in enumerate(top_k) if mid in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(recommend_fn, df, relevant_sets, k=K):
    """Evaluate a recommender function across all movies in the dataset."""

    scores = {"precision": [], "recall": [], "hit_rate": [], "ndcg": []}

    for movie_id in df["movie_id"]:
        relevant_ids = relevant_sets[movie_id]

        if not relevant_ids:
            continue

        recommended_ids = recommend_fn(movie_id, k)["movie_id"].tolist()

        scores["precision"].append(precision_at_k(recommended_ids, relevant_ids, k))
        scores["recall"].append(recall_at_k(recommended_ids, relevant_ids, k))
        scores["hit_rate"].append(hit_rate_at_k(recommended_ids, relevant_ids, k))
        scores["ndcg"].append(ndcg_at_k(recommended_ids, relevant_ids, k))

    return {metric: float(np.mean(values)) for metric, values in scores.items()}


if __name__ == "__main__":
    df, similarity = build_recommender()
    embeddings = build_embeddings(df)
    relevant_sets = build_relevant_sets(df)

    methods = {
        "TF-IDF": lambda movie_id, n: recommend_movies(movie_id, df, similarity, n=n),
        "Embedding": lambda movie_id, n: recommend_embedding(
            movie_id, df, embeddings, n=n
        ),
        "Hybrid": lambda movie_id, n: recommend_hybrid(
            movie_id, df, similarity, embeddings, n=n
        ),
    }

    print(f"Evaluating at k={K} (relevance = shares at least one genre)\n")

    for name, recommend_fn in methods.items():
        results = evaluate(recommend_fn, df, relevant_sets, k=K)
        print(f"{name}:")
        for metric, value in results.items():
            print(f"  {metric:<10} {value:.4f}")
        print()
