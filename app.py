import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
def load_data():
    df = pd.read_csv('imdb_top_1000.csv')
    return df

def build_tfidf_matrix(df):
    # Combine relevant features for content-based filtering
    df['combined'] = df['Genre'].fillna('') + ' ' + df['Overview'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return tfidf_matrix

def get_recommendations_by_genre(selected_genres, df, tfidf_matrix, top_n=5, year_filter=None, duration_filter=None):
    # Filter movies by selected genres
    genre_mask = df['Genre'].apply(lambda x: any(g.strip() in x for g in selected_genres))
    filtered_df = df[genre_mask].reset_index(drop=True)
    if year_filter:
        # Ensure Released_Year is numeric for comparison
        filtered_df['Released_Year'] = pd.to_numeric(filtered_df['Released_Year'], errors='coerce')
        if year_filter == 'New':
            filtered_df = filtered_df[filtered_df['Released_Year'] >= 2000]
        elif year_filter == 'Old':
            filtered_df = filtered_df[filtered_df['Released_Year'] < 2000]
    if duration_filter:
        # Assume 'Runtime' column is in format '142 min'
        filtered_df['Runtime_Min'] = pd.to_numeric(filtered_df['Runtime'].str.replace(' min', ''), errors='coerce')
        if duration_filter == 'Short':
            filtered_df = filtered_df[filtered_df['Runtime_Min'] < 90]
        elif duration_filter == 'Normal':
            filtered_df = filtered_df[(filtered_df['Runtime_Min'] >= 90) & (filtered_df['Runtime_Min'] <= 120)]
        elif duration_filter == 'Long':
            filtered_df = filtered_df[filtered_df['Runtime_Min'] > 120]
    if filtered_df.empty:
        return pd.DataFrame()
    # Rebuild tfidf for filtered set
    tfidf = TfidfVectorizer(stop_words='english')
    filtered_df['combined'] = filtered_df['Genre'].fillna('') + ' ' + filtered_df['Overview'].fillna('')
    tfidf_matrix_filtered = tfidf.fit_transform(filtered_df['combined'])
    # Recommend top_n movies by IMDB rating
    filtered_df = filtered_df.sort_values(by='IMDB_Rating', ascending=False).head(top_n)
    return filtered_df[['Series_Title', 'IMDB_Rating', 'Genre', 'Overview', 'Released_Year', 'Runtime', 'Poster_Link']].reset_index(drop=True)

def main():
    st.markdown('<h1 style="color:#111;font-size:2.5em;">🎬 Movie Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#444;font-size:1.1em;">Pick your favorite genres and get the top 5 movies with the highest IMDB ratings!</p>', unsafe_allow_html=True)
    df = load_data()
    tfidf_matrix = build_tfidf_matrix(df)
    # Genre selection
    all_genres = sorted(set(g.strip() for sublist in df['Genre'].dropna().str.split(',') for g in sublist))
    selected_genres = st.multiselect('Select genre(s):', all_genres)
    year_filter = st.radio('Movie Age', ['All', 'New', 'Old'], horizontal=True)
    duration_filter = st.radio('Movie Duration', ['All', 'Normal', 'Long'], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        recommend_btn = st.button('🎯 Recommend', key='recommend', use_container_width=True)
    with col2:
        surprise_btn = st.button('🎲 Surprise Me', key='surprise', use_container_width=True)

    if recommend_btn and selected_genres:
        year_logic = None
        if year_filter == 'New':
            year_logic = 'New'
        elif year_filter == 'Old':
            year_logic = 'Old'
        duration_logic = None
        if duration_filter == 'Normal':
            duration_logic = 'Normal'
        elif duration_filter == 'Long':
            duration_logic = 'Long'
        recommendations = get_recommendations_by_genre(selected_genres, df, tfidf_matrix, top_n=5, year_filter=year_logic, duration_filter=duration_logic)
        if recommendations.empty:
            st.warning('No movies found for the selected filters.')
        else:
            st.markdown('<h3 style="color:#111;">Top 5 Recommendations</h3>', unsafe_allow_html=True)
            for idx, row in recommendations.iterrows():
                with st.container():
                    cols = st.columns([5, 1])
                    with cols[0]:
                        st.markdown(f"""
                            <div style='background-color:#f4f4f4;padding:1em;margin-bottom:1em;border-radius:10px;border-left:5px solid #111;'>
                                <span style='font-size:1.3em;font-weight:bold;color:#111;'>{idx+1}.</span>
                                <span style='font-size:1.2em;font-weight:bold;'>{row['Series_Title']}</span><br>
                                <span style='color:#888;'>IMDB Rating: <b style='color:#111;'>{row['IMDB_Rating']}</b></span><br>
                                <span style='color:#666;'>Genre: {row['Genre']}</span><br>
                                <span style='color:#666;'>Year: {int(row['Released_Year']) if pd.notnull(row['Released_Year']) else 'N/A'}</span><br>
                                <span style='color:#666;'>Duration: {row['Runtime'] if pd.notnull(row['Runtime']) else 'N/A'}</span><br>
                                <span style='color:#444;font-size:0.98em;'>{row['Overview']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        if pd.notnull(row['Poster_Link']):
                            st.image(row['Poster_Link'], width=100)
    elif recommend_btn and not selected_genres:
        st.info('Please select at least one genre.')

    if surprise_btn:
        st.markdown('<h3 style="color:#111;">🎲 Surprise Picks</h3>', unsafe_allow_html=True)
        surprise_df = df.sample(5)
        for idx, row in surprise_df.iterrows():
            with st.container():
                cols = st.columns([5, 1])
                with cols[0]:
                    st.markdown(f"""
                        <div style='background-color:#f4f4f4;padding:1em;margin-bottom:1em;border-radius:10px;border-left:5px solid #111;'>
                            <span style='font-size:1.3em;font-weight:bold;color:#111;'>{idx+1}.</span>
                            <span style='font-size:1.2em;font-weight:bold;'>{row['Series_Title']}</span><br>
                            <span style='color:#888;'>IMDB Rating: <b style='color:#111;'>{row['IMDB_Rating']}</b></span><br>
                            <span style='color:#666;'>Genre: {row['Genre']}</span><br>
                            <span style='color:#666;'>Year: {int(row['Released_Year']) if pd.notnull(row['Released_Year']) else 'N/A'}</span><br>
                            <span style='color:#666;'>Duration: {row['Runtime'] if pd.notnull(row['Runtime']) else 'N/A'}</span><br>
                            <span style='color:#444;font-size:0.98em;'>{row['Overview']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                with cols[1]:
                    if pd.notnull(row['Poster_Link']):
                        st.image(row['Poster_Link'], width=100)

if __name__ == '__main__':
    main()
