from pathlib import Path

from flask import Flask, abort, jsonify, request

from filmantic.embeddings import build_embeddings
from filmantic.poster import get_poster_url
from filmantic.recommender import build_recommender, recommend_hybrid

FRONTEND_FOLDER = Path(__file__).resolve().parents[3] / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_FOLDER), static_url_path="")

df = None
similarity = None
embeddings = None
all_genres = None
all_keywords = None


def load_data():
    """Lazily build the recommender, embeddings, and genre/keyword indexes."""

    global df, similarity, embeddings, all_genres, all_keywords

    if df is None:
        df, similarity = build_recommender()
        embeddings = build_embeddings(df)
        all_genres = sorted({g for genres in df["genres"] for g in genres})
        all_keywords = sorted({k for keywords in df["keywords"] for k in keywords})

    return df, similarity, embeddings


def attach_posters(records):
    """Mutate a list of movie records in place, adding a poster_url to each."""

    for record in records:
        record["poster_url"] = get_poster_url(record["movie_id"])

    return records


@app.get("/")
def home():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/api/genres")
def list_genres():
    """List every unique genre in the dataset, for the multi-select UI."""

    load_data()

    return jsonify(all_genres)


@app.get("/api/keywords/suggest")
def suggest_keywords():
    """Suggest up to 10 keywords matching a partial query, for the tag input."""

    load_data()

    q = request.args.get("q", "").strip().lower()

    if not q:
        return jsonify([])

    matches = [k for k in all_keywords if q in k.lower()][:10]

    return jsonify(matches)


@app.get("/api/movies/search")
def search_movies():
    """Search for movies by title."""

    df, _, _ = load_data()

    q = request.args.get("q", "")

    if not q:
        abort(400, description="Query parameter 'q' is required.")

    matches = df["title"].str.contains(q, case=False, na=False, regex=False)

    results = df.loc[matches, ["movie_id", "title"]]

    return jsonify(results.head(10).to_dict(orient="records"))


@app.get("/api/movies/<int:movie_id>")
def get_movie(movie_id):
    """Get details of a movie by its ID."""

    df, _, _ = load_data()

    movie = df[df["movie_id"] == movie_id]

    if movie.empty:
        abort(404, description="Movie not found")

    record = movie.iloc[0].to_dict()
    record["poster_url"] = get_poster_url(movie_id)

    return jsonify(record)


@app.get("/api/recommendations/<int:movie_id>")
def get_recommendations(movie_id):
    """Get hybrid recommendations for a movie."""

    df, similarity, embeddings = load_data()

    n = request.args.get("n", default=5, type=int)

    if n is None or n < 1 or n > 50:
        abort(400, description="Query parameter 'n' must be between 1 and 50.")

    movie = df[df["movie_id"] == movie_id]

    if movie.empty:
        abort(404, description="Movie not found")

    results = recommend_hybrid(movie_id, df, similarity, embeddings, n=n)
    records = results.to_dict(orient="records")
    attach_posters(records)

    return jsonify(records)


@app.get("/api/recommendations/discover")
def discover_movies():
    """Find movies based on matching genres and keywords."""

    df, _, _ = load_data()

    genres = [g.strip() for g in request.args.get("genres", "").split(",") if g.strip()]
    keywords = [
        k.strip() for k in request.args.get("keywords", "").split(",") if k.strip()
    ]

    n = request.args.get("n", default=5, type=int)

    if n is None or n < 1 or n > 50:
        abort(400, description="Query parameter 'n' must be between 1 and 50.")

    if not genres and not keywords:
        abort(400, description="Enter at least one genre or keyword.")

    genre_set = {g.lower() for g in genres}
    keyword_set = {k.lower() for k in keywords}

    results = df.copy()
    results["match_count"] = 0

    if genre_set:
        results["match_count"] += results["genres"].apply(
            lambda movie_genres: sum(1 for g in movie_genres if g.lower() in genre_set)
        )

    if keyword_set:
        results["match_count"] += results["keywords"].apply(
            lambda movie_keywords: sum(
                1 for k in movie_keywords if k.lower() in keyword_set
            )
        )

    results = results[results["match_count"] > 0]
    results = results.sort_values("match_count", ascending=False).head(n)

    records = results[
        ["movie_id", "title", "genres", "keywords", "match_count"]
    ].to_dict(orient="records")
    attach_posters(records)

    return jsonify(records)


@app.errorhandler(400)
@app.errorhandler(404)
def handle_error(error):
    return jsonify({"error": error.description}), error.code


if __name__ == "__main__":
    app.run(debug=True)
