import os
import pickle
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("TMDB_API_KEY not set. Add it to your .env file.")
    st.stop()

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US".format(movie_id, TMDB_API_KEY)
    data = requests.get(url)

def recommend(movie):
    match = movies[movies['title'].str.lower() == movie.lower()]

    if match.empty:
        return [], []

    index = match.index[0]

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

st.header('Movie Recommender System')
movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(recommended_movie_posters[0])
        st.text(recommended_movie_names[0])
    with col2:
        st.image(recommended_movie_posters[1])
        st.text(recommended_movie_names[1])
    with col3:
        st.image(recommended_movie_posters[2])
        st.text(recommended_movie_names[2])
    with col4:
        st.image(recommended_movie_posters[3])
        st.text(recommended_movie_names[3])
    with col5:
        st.image(recommended_movie_posters[4])
        st.text(recommended_movie_names[4])




