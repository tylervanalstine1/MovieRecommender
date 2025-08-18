# Movie Recommender App

This is a Streamlit application that recommends movies based on your selection using the IMDB Top 1000 dataset from Kaggle.

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Start the app:
   ```
   streamlit run app.py
   ```

## Features
- Select a movie and get recommendations for similar movies.
- Uses content-based filtering (based on genres, description, etc.).

## Dataset
- Place your `imdb_top_1000.csv` file in the project directory.
