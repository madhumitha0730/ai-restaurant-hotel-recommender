"""
generate_data.py
-----------------
Creates synthetic-but-realistic sample datasets for the AI Restaurant &
Hotel Recommendation System (restaurants.csv, hotels.csv).

Run:
    python generate_data.py

Replace these CSVs with your own real data any time — the recommender
only needs the same column names to keep working.
"""

import random
import csv
import os

random.seed(42)

CITIES = ["Chennai", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune"]

# ---------------------------------------------------------------- Restaurants
CUISINES = [
    "South Indian", "North Indian", "Chinese", "Italian", "Continental",
    "Mexican", "Japanese", "Thai", "Street Food", "Mughlai", "Bakery", "Seafood"
]
REST_TAGS = [
    "family friendly", "romantic", "rooftop", "budget", "fine dining",
    "outdoor seating", "live music", "pure veg", "buffet", "quick bite",
    "pet friendly", "late night"
]
REST_NAME_PARTS_1 = ["Spice", "Royal", "Green", "Urban", "Golden", "Coastal",
                      "Copper", "Bombay", "Madras", "Saffron", "The Table", "Sunset"]
REST_NAME_PARTS_2 = ["Kitchen", "Bistro", "Diner", "House", "Corner", "Grill",
                      "Cafe", "Restaurant", "Terrace", "Junction", "Hub", "Kitchenette"]


def make_restaurants(n=60):
    rows = []
    used_names = set()
    for i in range(1, n + 1):
        while True:
            name = f"{random.choice(REST_NAME_PARTS_1)} {random.choice(REST_NAME_PARTS_2)}"
            if name not in used_names:
                used_names.add(name)
                break
        cuisine = random.choice(CUISINES)
        city = random.choice(CITIES)
        price_range = random.randint(1, 4)          # 1=$, 4=$$$$
        rating = round(random.uniform(3.0, 5.0), 1)
        tags = random.sample(REST_TAGS, k=random.randint(2, 4))
        rows.append({
            "id": i,
            "name": name,
            "city": city,
            "cuisine": cuisine,
            "price_range": price_range,
            "rating": rating,
            "tags": "|".join(tags),
        })
    return rows


# --------------------------------------------------------------------- Hotels
AMENITIES = [
    "free wifi", "swimming pool", "spa", "gym", "free breakfast",
    "parking", "pet friendly", "airport shuttle", "bar", "business center",
    "sea view", "rooftop restaurant", "family rooms", "pet friendly"
]
HOTEL_TYPES = ["Business", "Boutique", "Resort", "Budget", "Heritage", "Luxury"]
HOTEL_NAME_PARTS_1 = ["Grand", "Royal", "The Pearl", "Sunrise", "Emerald",
                       "Regency", "Palm", "Silver", "Horizon", "The Oak", "Blue Lagoon", "Zenith"]
HOTEL_NAME_PARTS_2 = ["Palace", "Residency", "Inn", "Suites", "Towers",
                       "Retreat", "Plaza", "Grand Hotel", "Court", "Stay", "Heights", "Manor"]


def make_hotels(n=50):
    rows = []
    used_names = set()
    for i in range(1, n + 1):
        while True:
            name = f"{random.choice(HOTEL_NAME_PARTS_1)} {random.choice(HOTEL_NAME_PARTS_2)}"
            if name not in used_names:
                used_names.add(name)
                break
        city = random.choice(CITIES)
        star_rating = random.randint(2, 5)
        price_per_night = random.choice([1500, 2200, 3200, 4500, 6000, 8500, 12000, 18000])
        rating = round(random.uniform(3.0, 5.0), 1)
        htype = random.choice(HOTEL_TYPES)
        amenities = random.sample(AMENITIES, k=random.randint(3, 6))
        rows.append({
            "id": i,
            "name": name,
            "city": city,
            "hotel_type": htype,
            "star_rating": star_rating,
            "price_per_night": price_per_night,
            "rating": rating,
            "amenities": "|".join(sorted(set(amenities))),
        })
    return rows


def write_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    restaurants = make_restaurants(60)
    write_csv(
        restaurants,
        os.path.join(here, "restaurants.csv"),
        ["id", "name", "city", "cuisine", "price_range", "rating", "tags"],
    )

    hotels = make_hotels(50)
    write_csv(
        hotels,
        os.path.join(here, "hotels.csv"),
        ["id", "name", "city", "hotel_type", "star_rating", "price_per_night", "rating", "amenities"],
    )
