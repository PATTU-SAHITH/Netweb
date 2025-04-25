import pandas as pd
from surprise import Dataset, Reader, SVD
import requests

# TMDb API Key (replace with your own key)
TMDB_API_KEY = 'YOUR_TMDB_API_KEY'

movies = {
    101: "The Matrix",
    102: "Inception",
    103: "Interstellar",
    104: "The Dark Knight",
    105: "Toy Story",
    106: "Finding Nemo",
    107: "The Godfather",
    108: "Pulp Fiction"
}

df = pd.DataFrame([
    [1, 101, 5], [1, 102, 5], [1, 103, 4],
    [2, 105, 5], [2, 106, 4],
    [3, 107, 5], [3, 108, 5], [3, 104, 4],
], columns=["userId", "movieId", "rating"])

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

def get_poster_url(movie_title):
    search_url = 'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_title
    }
    response = requests.get(search_url, params=params)
    data = response.json()
    if data['results']:
        poster_path = data['results'][0].get('poster_path')
        if poster_path:
            return f'https://image.tmdb.org/t/p/w500{poster_path}'
    return "https://via.placeholder.com/150x225?text=No+Image"

def recommend_movies(user_id, n=3):
    all_movie_ids = set(movies.keys())
    seen = set(df[df['userId'] == user_id]['movieId'])
    unseen = all_movie_ids - seen

    predictions = [(mid, algo.predict(user_id, mid).est) for mid in unseen]
    predictions.sort(key=lambda x: x[1], reverse=True)

    results = []
    for mid, score in predictions[:n]:
        title = movies[mid]
        poster_url = get_poster_url(title)
        results.append((title, round(score, 2), poster_url))
    return results
