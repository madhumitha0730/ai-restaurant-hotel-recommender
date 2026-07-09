# AI-Based Restaurant & Hotel Recommendation System

A content-based recommendation system that suggests restaurants and hotels
based on a user's free-text preferences (cuisine, ambience, amenities, city,
budget). Built with Python, pandas, and scikit-learn (TF-IDF + cosine
similarity), with a Streamlit web interface.

## How it works

1. **Profile building** – each restaurant/hotel record is converted into a
   short text "profile" combining its cuisine/type, tags/amenities, and city.
2. **Vectorization** – all profiles are converted to TF-IDF vectors, a
   standard NLP technique that represents text as weighted numeric vectors.
3. **Matching** – a user's query (e.g. *"romantic rooftop italian dinner"*)
   is vectorized the same way, and **cosine similarity** ranks every item by
   how closely it matches.
4. **Hybrid scoring** – the final score blends text similarity (70%) with
   the item's normalized rating (30%), so well-matched *and* well-rated
   places rank highest. Optional filters (city, max price) are applied
   before ranking.
5. A **similar_items()** method also supports "you may also like" style
   item-to-item recommendations.

This is a classic **content-based filtering** approach (as opposed to
collaborative filtering, which needs historical user–item ratings). It works
well with no user history at all — perfect for a cold-start recommender.

## Project structure

```
project/
├── data/
│   ├── generate_data.py   # generates sample datasets
│   ├── restaurants.csv    # 60 sample restaurants
│   └── hotels.csv         # 50 sample hotels
├── recommender.py         # core recommendation engine
├── demo.py                 # CLI demo (no Streamlit needed)
├── app.py                  # Streamlit web app
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the CLI demo (fastest way to see it work)

```bash
python demo.py
```

## Run the web app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Using your own data

Replace `data/restaurants.csv` / `data/hotels.csv` with real data, keeping
the same column names:

- **restaurants.csv**: `id, name, city, cuisine, price_range, rating, tags`
  (`tags` is pipe-separated, e.g. `romantic|rooftop|pure veg`)
- **hotels.csv**: `id, name, city, hotel_type, star_rating, price_per_night, rating, amenities`
  (`amenities` is pipe-separated, e.g. `spa|pool|free wifi`)

No code changes are needed — just point `recommender.py`'s loader functions
at your CSVs (or overwrite the sample files in place).

## Extending the project

- **Collaborative filtering**: add a `ratings.csv` of `user_id, item_id, rating`
  and implement matrix factorization (e.g. `surprise` library or
  `scipy.sparse.linalg.svds`) to recommend based on similar users, not just
  item text.
- **LLM-powered query understanding**: `recommender.py` includes a stub
  `parse_query_with_llm()` — wire it to the Anthropic API to turn natural
  language like *"somewhere cheap and cheerful for a family dinner in
  Chennai"* into structured filters automatically.
- **Real data**: connect to a live source (Zomato/Swiggy-style API, Google
  Places, Booking.com API) instead of the CSVs.
- **Persistence**: add a small database (SQLite/PostgreSQL) if you need
  user accounts, saved favorites, or booking history.

## Tech stack

- Python 3.10+
- pandas — data handling
- scikit-learn — TF-IDF vectorization + cosine similarity
- Streamlit — web UI
