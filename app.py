"""
app.py
------
Streamlit web app for the AI-based Restaurant & Hotel Recommendation System.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import streamlit as st
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
}
h1 {
    background: linear-gradient(90deg, #ff6b6b, #f8b500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton>button {
    background: linear-gradient(90deg, #ff6b6b, #f8b500);
    color: white;
    border: none;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
from recommender import load_restaurant_recommender, load_hotel_recommender

st.set_page_config(page_title="AI Restaurant & Hotel Recommender", page_icon="🍽️", layout="centered")

st.title("🍽️🏨 AI Restaurant & Hotel Recommendation System")
st.caption(
    "Content-based recommender using TF-IDF text similarity + rating, "
    "built with scikit-learn."
)


@st.cache_resource
def get_recommenders():
    return load_restaurant_recommender(), load_hotel_recommender()


rest_rec, hotel_rec = get_recommenders()

mode = st.radio("What are you looking for?", ["Restaurants", "Hotels"], horizontal=True)

cities = sorted(rest_rec.df["city"].unique()) if mode == "Restaurants" else sorted(hotel_rec.df["city"].unique())
city = st.selectbox("City", ["Any"] + cities)

query = st.text_input(
    "Describe what you want",
    placeholder="e.g. romantic rooftop italian dinner" if mode == "Restaurants"
    else "e.g. spa resort with pool and sea view",
)

top_n = st.slider("Number of recommendations", 3, 10, 5)

if mode == "Restaurants":
    max_price = st.select_slider("Max price level (1 = budget, 4 = fine dining)", options=[1, 2, 3, 4], value=4)
    price_col = "price_range"
    rec_engine = rest_rec
else:
    max_price = st.slider("Max price per night (₹)", 1000, 20000, 20000, step=500)
    price_col = "price_per_night"
    rec_engine = hotel_rec

if st.button("Get recommendations", type="primary"):
    if not query.strip():
        st.warning("Please describe what you're looking for.")
    else:
        results = rec_engine.recommend_from_query(
            query=query,
            city=None if city == "Any" else city,
            max_price=max_price,
            price_col=price_col,
            top_n=top_n,
        )
        if results.empty:
            st.info("No matches found — try widening your filters.")
        else:
            st.subheader(f"Top {len(results)} picks for you")
            for _, row in results.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{row['name']}** · {row['city']}")
                        if mode == "Restaurants":
                            st.write(f"Cuisine: {row['cuisine']} · Tags: {row['tags'].replace('|', ', ')}")
                        else:
                            st.write(f"Type: {row['hotel_type']} · Amenities: {row['amenities'].replace('|', ', ')}")
                    with c2:
                        st.metric("Rating", f"{row['rating']:.1f} ★")
                        st.caption(f"Match: {row['similarity']*100:.0f}%")

st.divider()
st.caption(
    "Data is synthetic sample data (see data/generate_data.py). "
    "Swap in your own restaurants.csv / hotels.csv with the same columns to use real data."
)
