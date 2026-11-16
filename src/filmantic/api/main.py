from flask import Flask, abort, jsonify, request

from filmantic.embeddings import build_embeddings
from filmantic.recommender import build_recommender, recommend_hybrid

app = Flask(__name__)

df = None
similarity = None
embeddings = None


def load_data():
    global df, similarity, embeddings

    if df is None:
        df, similarity = build_recommender()
        embeddings = build_embeddings(df)

    return df, similarity, embeddings


@app.get("/api/health")
def health_check():
    return jsonify({"status": "ok"})


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

    return jsonify(movie.iloc[0].to_dict())


@app.get("/api/recommendations/<int:movie_id>")
def get_recommendations(movie_id):
    """Get hybrid recommendations for a movie."""

    df, similarity, embeddings = load_data()

    n = request.args.get("n", default=10, type=int)

    if n is None or n < 1 or n > 50:
        abort(400, description="Query parameter 'n' must be between 1 and 50.")

    movie = df[df["movie_id"] == movie_id]

    if movie.empty:
        abort(404, description="Movie not found")

    results = recommend_hybrid(movie_id, df, similarity, embeddings, n=n)

    return jsonify(results.to_dict(orient="records"))


@app.errorhandler(400)
@app.errorhandler(404)
def handle_error(error):
    return jsonify({"error": error.description}), error.code


if __name__ == "__main__":
    app.run(debug=True)
