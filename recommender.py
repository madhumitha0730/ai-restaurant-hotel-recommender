"""
recommender.py
---------------
AI-based content recommendation engine for restaurants and hotels.

How it works (in plain English):
1. Every restaurant/hotel is turned into a short "profile" string that
   describes it (cuisine, city, tags/amenities, price bucket, etc).
2. All profiles are converted into TF-IDF vectors (a classic NLP technique
   that turns text into numbers, weighting distinctive words more heavily).
3. When a user gives preferences (free-text + optional filters), we vectorize
   the query the same way and compute cosine similarity against every item.
4. We blend that similarity score with the item's own rating (normalised)
   to produce a final ranked list. This is a simple hybrid
   content-based + popularity-based recommender.

You can also ask "give me items similar to item X" (item-to-item
recommendations) which is handy for "You may also like..." features.

No external AI/ML API is required — everything runs locally with
scikit-learn. If you want to swap in a real LLM (e.g. the Anthropic API)
for natural-language query understanding, see the `parse_query_with_llm`
stub near the bottom.
"""

from __future__ import annotations
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """
    Generic content-based recommender that works for either restaurants
    or hotels, depending on the column mapping you give it.
    """

    def __init__(self, csv_path: str, profile_columns: list[str], id_col: str = "id",
                 rating_col: str = "rating", name_col: str = "name", city_col: str = "city"):
        self.df = pd.read_csv(csv_path)
        self.id_col = id_col
        self.rating_col = rating_col
        self.name_col = name_col
        self.city_col = city_col
        self.profile_columns = profile_columns

        self.df["profile"] = self.df.apply(self._build_profile, axis=1)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["profile"])

        # normalised rating 0..1 used for the popularity component of the score
        r = self.df[self.rating_col].astype(float)
        self.df["_rating_norm"] = (r - r.min()) / (r.max() - r.min() + 1e-9)

    def _build_profile(self, row) -> str:
        parts = []
        for col in self.profile_columns:
            val = row[col]
            if pd.isna(val):
                continue
            # pipe-separated multi-value fields (tags, amenities) -> spaces
            parts.append(str(val).replace("|", " "))
        return " ".join(parts)

    # ------------------------------------------------------------------
    def recommend_from_query(self, query: str, city: str | None = None,
                              max_price=None, price_col: str | None = None,
                              top_n: int = 5, sim_weight: float = 0.7) -> pd.DataFrame:
        """
        Recommend items matching a free-text preference query, e.g.
        "romantic rooftop italian dinner" or "spa resort with pool".

        city        : optional exact-match filter
        max_price   : optional upper bound filter
        price_col   : which column max_price applies to (e.g. 'price_range'
                      or 'price_per_night') — required if max_price is set
        sim_weight  : how much weight (0..1) to give text-similarity vs rating
        """
        candidates = self.df.copy()

        if city:
            candidates = candidates[candidates[self.city_col].str.lower() == city.lower()]
        if max_price is not None and price_col is not None:
            candidates = candidates[candidates[price_col] <= max_price]

        if candidates.empty:
            return candidates

        query_vec = self.vectorizer.transform([query])
        idx = candidates.index
        sims = cosine_similarity(query_vec, self.tfidf_matrix[idx]).flatten()

        candidates = candidates.copy()
        candidates["similarity"] = sims
        candidates["score"] = (
            sim_weight * candidates["similarity"] +
            (1 - sim_weight) * candidates["_rating_norm"]
        )

        return candidates.sort_values("score", ascending=False).head(top_n)[
            [self.id_col, self.name_col, self.city_col, self.rating_col, "similarity", "score"] +
            [c for c in self.profile_columns if c not in (self.city_col,)]
        ]

    # ------------------------------------------------------------------
    def similar_items(self, item_id, top_n: int = 5) -> pd.DataFrame:
        """
        'You may also like...' — find items most similar to a given item_id.
        """
        if item_id not in self.df[self.id_col].values:
            raise ValueError(f"No item with id={item_id}")

        row_idx = self.df.index[self.df[self.id_col] == item_id][0]
        sims = cosine_similarity(self.tfidf_matrix[row_idx], self.tfidf_matrix).flatten()

        result = self.df.copy()
        result["similarity"] = sims
        result = result[result[self.id_col] != item_id]

        return result.sort_values("similarity", ascending=False).head(top_n)[
            [self.id_col, self.name_col, self.city_col, self.rating_col, "similarity"] +
            [c for c in self.profile_columns if c not in (self.city_col,)]
        ]


# ---------------------------------------------------------------------------
def load_restaurant_recommender(csv_path=None) -> ContentRecommender:
    csv_path = csv_path or os.path.join(os.path.dirname(__file__), "data", "restaurants.csv")
    return ContentRecommender(
        csv_path=csv_path,
        profile_columns=["cuisine", "tags", "city"],
        rating_col="rating",
    )


def load_hotel_recommender(csv_path=None) -> ContentRecommender:
    csv_path = csv_path or os.path.join(os.path.dirname(__file__), "data", "hotels.csv")
    return ContentRecommender(
        csv_path=csv_path,
        profile_columns=["hotel_type", "amenities", "city"],
        rating_col="rating",
    )


# ---------------------------------------------------------------------------
def parse_query_with_llm(free_text: str) -> dict:
    """
    OPTIONAL EXTENSION (not required to run the project):
    Instead of / in addition to TF-IDF matching, you could send `free_text`
    to the Anthropic API and ask Claude to extract structured filters, e.g.
    {"city": "Chennai", "cuisine": "Italian", "max_price": 3}.
    That structured output can then be fed into recommend_from_query() as
    the `city` / `max_price` args, while the raw text still drives the
    TF-IDF `query` for the fuzzy/semantic part. Left as a stub so the
    project runs fully offline by default.
    """
    raise NotImplementedError("Wire this up to the Anthropic API if you want LLM-based query parsing.")
