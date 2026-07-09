"""
demo.py
-------
Quick command-line demo of the recommendation engine — no Streamlit needed.

Run:
    python demo.py
"""

from recommender import load_restaurant_recommender, load_hotel_recommender
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


def main():
    rest_rec = load_restaurant_recommender()
    hotel_rec = load_hotel_recommender()

    print("=" * 80)
    print("RESTAURANTS — query: 'romantic rooftop italian dinner', city=Chennai")
    print("=" * 80)
    results = rest_rec.recommend_from_query(
        query="romantic rooftop italian dinner",
        city="Chennai",
        max_price=4,
        price_col="price_range",
        top_n=5,
    )
    print(results.to_string(index=False) if not results.empty else "No matches.")

    print()
    print("=" * 80)
    print("HOTELS — query: 'spa resort with pool and sea view', city=Mumbai, max ₹10000/night")
    print("=" * 80)
    results = hotel_rec.recommend_from_query(
        query="spa resort with pool and sea view",
        city="Mumbai",
        max_price=10000,
        price_col="price_per_night",
        top_n=5,
    )
    print(results.to_string(index=False) if not results.empty else "No matches.")

    print()
    print("=" * 80)
    print("ITEM-TO-ITEM — restaurants similar to id=1")
    print("=" * 80)
    print(rest_rec.similar_items(1, top_n=5).to_string(index=False))


if __name__ == "__main__":
    main()
