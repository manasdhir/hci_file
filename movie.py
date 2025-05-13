import kagglehub
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                return pd.read_csv(os.path.join(root, file))
    return None

# Preprocessing
@st.cache_data
def preprocess_data(df):
    # Safely extract numeric years
    df['Released_Year'] = pd.to_numeric(df['Released_Year'].astype(str).str.extract(r'(\d{4})', expand=False), errors='coerce')
    
    # Compute median from cleaned column
    median_year = int(df['Released_Year'].median(skipna=True))
    
    # Fill missing values with median and convert to int
    df['Released_Year'] = df['Released_Year'].fillna(median_year).astype(int)

    # Combine features for text similarity
    df['Combined_Features'] = df['Genre'] + ' ' + df['Director'] + ' ' + df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3']

    # Clean certificate field
    df['Certificate'] = df['Certificate'].fillna('NR').replace({'U': 'G', 'UA': 'PG', 'A': 'R', 'Not Rated': 'NR'})

    return df

@st.cache_resource
def get_genre_encoder(df):
    features = ['IMDB_Rating', 'Released_Year']
    X = df[features].apply(pd.to_numeric, errors='coerce').dropna()
    genre_encoder = LabelEncoder()
    df['Genre_Label'] = genre_encoder.fit_transform(df['Genre'].str.split(',').str[0])
    y = df['Genre_Label'].loc[X.index]  # Fixed: using square brackets instead of parentheses
    return genre_encoder

@st.cache_data
def get_recommendations(movie_title, df, sim_matrix):
    idx = df[df['Series_Title'] == movie_title].index[0]
    sim_scores = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)[1:6]
    return df.iloc[[i[0] for i in sim_scores]][['Series_Title', 'Genre', 'IMDB_Rating', 'Released_Year', 'Director', 'Certificate', 'Star1', 'Star2', 'Star3']]

@st.cache_data
def get_enhanced_recommendations(movie_title, df, sim_matrix, prefs):
    idx = df[df['Series_Title'] == movie_title].index[0]
    sim_scores = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)[:20]
    scored = []
    for i, score in sim_scores:
        movie = df.iloc[i]
        if prefs:
            p_score = score * 0.5
            if prefs['year_range'][0] <= movie['Released_Year'] <= prefs['year_range'][1]: p_score += 0.15
            if prefs['min_rating'] <= movie['IMDB_Rating']: p_score += 0.15
            if any(g.strip() in movie['Genre'] for g in prefs['genres']): p_score += 0.1
            if movie['Certificate'] in prefs['age_ratings']: p_score += 0.1
            scored.append((i, p_score))
        else:
            scored.append((i, score))
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
    return df.iloc[[i[0] for i in top]][['Series_Title', 'Genre', 'IMDB_Rating', 'Released_Year', 'Director', 'Certificate', 'Star1', 'Star2', 'Star3']]

@st.cache_resource
def create_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])
    return cosine_similarity(tfidf_matrix)

# Load and process
with st.spinner("Loading and processing data..."):
    df = preprocess_data(load_data())
    sim_matrix = create_similarity_matrix(df)
    genre_encoder = get_genre_encoder(df)

# Sidebar Inputs - Move preferences to sidebar only
st.sidebar.title("🔧 Preferences")
genres = sorted({g.strip() for sub in df['Genre'].dropna().str.split(',') for g in sub})
selected_genres = st.sidebar.multiselect("Preferred Genres", genres)
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 7.0, 0.1)
year_range = st.sidebar.slider("Year Range", int(df['Released_Year'].min()), int(df['Released_Year'].max()), (2000, 2022))
age_ratings = st.sidebar.multiselect("Age Ratings", ['G', 'PG', 'PG-13', 'R', 'NR'], default=['G', 'PG', 'PG-13'])

# Main Area
st.header("🎞️ Your Movie Matchmaker")

# Move movie selector to main page
movie = st.selectbox("📽️ Pick a movie", options=[""] + list(df['Series_Title'].values))

# Add a search button instead of apply preferences checkbox
search_clicked = st.button("🔍 Search")

# Only show movie details and recommendations when a movie is picked and search is clicked
if movie and search_clicked:
    # Create preferences dict
    prefs = {
        'genres': selected_genres,
        'min_rating': min_rating,
        'year_range': year_range,
        'age_ratings': age_ratings
    }
    
    info = df[df['Series_Title'] == movie].iloc[0]
    st.subheader(f"🎬 {info['Series_Title']}")
    st.text(f"Genre: {info['Genre']} | Rating: {info['IMDB_Rating']} | Year: {info['Released_Year']} | Certificate: {info['Certificate']}")
    st.markdown(f"**Director**: {info['Director']}")
    st.markdown(f"**Stars**: {info['Star1']}, {info['Star2']}, {info['Star3']}")

    st.divider()
    st.subheader("✨ Top Recommendations")
    results = get_enhanced_recommendations(movie, df, sim_matrix, prefs)
    for _, r in results.iterrows():
        with st.container():
            st.markdown(f"**{r['Series_Title']}**")
            st.text(f"Genre: {r['Genre']} | Rating: {r['IMDB_Rating']} | Year: {r['Released_Year']} | Certificate: {r['Certificate']}")
            st.text(f"Stars: {r['Star1']}, {r['Star2']}, {r['Star3']}")
            st.markdown("---")
