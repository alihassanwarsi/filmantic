import pandas as pd
import pytest

from filmantic.api import main as api

client = api.app.test_client()


@pytest.fixture
def fake_data(monkeypatch):
    movies = pd.DataFrame(
        [
            {"movie_id": 157336, "title": "Interstellar"},
            {"movie_id": 27205, "title": "Inception"},
            {"movie_id": 155, "title": "The Dark Knight"},
            {"movie_id": 19995, "title": "Avatar"},
            {"movie_id": 597, "title": "Titanic"},
            {"movie_id": 603, "title": "The Matrix"},
        ]
    )

    monkeypatch.setattr(api, "load_data", lambda: (movies, None, None))

    monkeypatch.setattr(
        api,
        "recommend_hybrid",
        lambda movie_id, df, similarity, embeddings, n: df.head(n),
    )


def test_health_check():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_movie_search(fake_data):
    response = client.get("/api/movies/search?q=interstellar")

    assert response.status_code == 200
    assert response.json[0]["title"] == "Interstellar"


def test_empty_search(fake_data):
    response = client.get("/api/movies/search")

    assert response.status_code == 400


def test_movie_details(fake_data):
    response = client.get("/api/movies/157336")

    assert response.status_code == 200
    assert response.json["title"] == "Interstellar"


def test_movie_not_found(fake_data):
    response = client.get("/api/movies/999999999")

    assert response.status_code == 404


def test_recommendations(fake_data):
    response = client.get("/api/recommendations/157336?n=5")

    assert response.status_code == 200
    assert len(response.json) == 5


def test_invalid_recommendation_count(fake_data):
    response = client.get("/api/recommendations/157336?n=100")

    assert response.status_code == 400
