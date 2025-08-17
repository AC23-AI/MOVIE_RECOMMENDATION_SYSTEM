import pickle
import streamlit as st
import requests
import time
from PIL import Image
import gdown
import os
def fetch_poster(movie_id):
    time.sleep(0.5)  # Delay to prevent rate limiting

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=b3d0d5942a2933d87615da57631ec5c8&language=en-US"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            print(f"No poster found for movie ID {movie_id}")
            return "https://via.placeholder.com/150"

    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for movie ID {movie_id}: {e}")
        return "https://via.placeholder.com/150"

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters



url = "https://drive.google.com/uc?id=1fkikMjbvIhaERr8iv9bLj-EZx8fw_egI"
output = "similarity.pkl"
gdown.download(url, output, quiet=False)

similarity = pickle.load(open(output, "rb"))
movies = pickle.load(open("movies.pkl",'rb'))



st.title('🎬MOVIE RECOMMENDATION SYSTEM🎬')
image = Image.open(r"C:/Users/USER/Desktop/MRS/images.png") 
st.image(image,width=250)
st.header("The Place Where You Find Your Favourite Movies!!🌟🍟🌟")
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

