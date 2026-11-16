from functools import lru_cache

import requests

from .config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL

PLACEHOLDER_POSTER = "https://via.placeholder.com/342x513?text=No+Poster"


@lru_cache(maxsize=4096)
def get_poster_url(movie_id):
    """Fetch a movie's poster URL from TMDB."""

    if not TMDB_API_KEY:
        return PLACEHOLDER_POSTER

    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5,
        )
        response.raise_for_status()

        poster_path = response.json().get("poster_path")

        if not poster_path:
            return PLACEHOLDER_POSTER

        return f"{TMDB_IMAGE_BASE_URL}{poster_path}"

    except requests.RequestException:
        return PLACEHOLDER_POSTER
