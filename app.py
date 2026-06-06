import pickle
import os
import re
import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page configuration ---
st.set_page_config(
    page_title="Movies Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- OMDb API configuration (no key required) ---
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "thewdb")
OMDB_BASE = "https://www.omdbapi.com/"


# --- Helper functions ---
def _omdb_get(params, timeout=10):
    """Do a GET against OMDb and return (data, error_message)."""
    full_params = {**params, "apikey": OMDB_API_KEY}
    try:
        r = requests.get(OMDB_BASE, params=full_params, timeout=timeout)
        if r.status_code != 200:
            return None, f"OMDb error {r.status_code}: {r.text[:200]}"
        data = r.json()
        if isinstance(data, dict) and data.get("Response") == "False":
            return None, data.get("Error", "Unknown OMDb error")
        return data, None
    except requests.exceptions.Timeout:
        return None, "Request timed out (OMDb did not respond in time)."
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to OMDb. Check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {e}"


@st.cache_data(show_spinner=False)
def fetch_poster_by_title(title, year=None):
    """Fetch a poster URL for a movie by title using OMDb."""
    params = {"t": title, "type": "movie"}
    if year:
        params["y"] = str(year)
    data, err = _omdb_get(params)
    if data is None:
        return None, err
    poster = data.get("Poster")
    if poster and poster != "N/A":
        return poster, None
    return None, None


@st.cache_data(show_spinner=False)
def search_movies(query):
    """Search movies on OMDb by keyword."""
    data, err = _omdb_get({"s": query, "type": "movie"})
    if data is None:
        return [], err
    return data.get("Search", [])[:12], None


def _clean_title_for_search(title):
    """Make a title more likely to match on OMDb by removing year suffixes etc."""
    # Strip trailing year like "(2012)" or "(2008)"
    cleaned = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title))
    # Strip trailing " - The Movie" etc.
    cleaned = re.sub(r"\s*-\s*the movie\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


# --- Load pickled data ---
movies = pickle.load(open('movie_list.pkl', 'rb'))

# Normalize the column that holds the movie id
if 'movie_id' not in movies.columns and 'id' in movies.columns:
    movies = movies.rename(columns={'id': 'movie_id'})

movie_list = movies['title'].values


# --- Build / load similarity matrix ---
# similarity.pkl in this project is sometimes a corrupted/duplicate of movie_list.pkl.
# We rebuild a real 4806x4806 cosine-similarity matrix from the 'tags' column.
@st.cache_resource
def build_similarity():
    """Build (or load) a 4806x4806 cosine-similarity matrix."""
    sim_path = "similarity.pkl"
    sim = None

    if os.path.exists(sim_path):
        try:
            candidate = pickle.load(open(sim_path, "rb"))
            # Only accept the file if it's a real square matrix of the right size
            if hasattr(candidate, "shape"):
                shape = candidate.shape
                if len(shape) == 2 and shape[0] == shape[1] == len(movies):
                    sim = np.asarray(candidate)
        except Exception:
            sim = None

    if sim is None:
        with st.spinner("Building similarity matrix from tags (first run only)..."):
            tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
            tag_strings = movies["tags"].fillna("").astype(str)
            vectors = tfidf.fit_transform(tag_strings)
            sim = cosine_similarity(vectors)
            # Persist for next time
            try:
                pickle.dump(sim, open(sim_path, "wb"))
            except Exception:
                pass

    return sim


similarity = build_similarity()


def recommend(movie):
    """Return 5 recommended movies (names + posters + years) similar to the chosen one."""
    idx_list = movies.index[movies["title"] == movie].tolist()
    if not idx_list:
        return [], [], []
    index = idx_list[0]

    # `similarity` is a (n, n) numpy array
    row = similarity[index]
    distances = sorted(
        list(enumerate(row)),
        reverse=True,
        key=lambda x: x[1],
    )
    recommended_names, recommended_posters, recommended_years = [], [], []
    for i, _score in distances[1:6]:
        row_data = movies.iloc[i]
        title = str(row_data["title"])
        year = _clean_title_for_search(title)  # nothing yet, just to keep var
        # Try to pull a year from a 'release_date' or similar column if it exists
        year_val = None
        for col in ("release_date", "year", "release_year"):
            if col in movies.columns and pd.notna(row_data[col]):
                v = str(row_data[col])
                m = re.search(r"(\d{4})", v)
                if m:
                    year_val = m.group(1)
                    break

        search_title = _clean_title_for_search(title)
        poster, _ = fetch_poster_by_title(search_title, year_val)
        recommended_posters.append(poster)
        recommended_names.append(title)
        recommended_years.append(year_val or "—")
    return recommended_names, recommended_posters, recommended_years


# --- HTML helpers for evenly-sized cards ---
def _poster_html(poster_url, height=320):
    if poster_url:
        return (
            f"<div style='height:{height}px;overflow:hidden;border-radius:8px;"
            f"background:#1f1f1f;display:flex;align-items:center;justify-content:center;'>"
            f"<img src='{poster_url}' style='width:100%;height:100%;object-fit:cover;'/>"
            f"</div>"
        )
    return (
        f"<div style='height:{height}px;background:#1f1f1f;"
        f"display:flex;align-items:center;justify-content:center;"
        f"color:#888;border-radius:8px;font-size:48px;'>🎬</div>"
    )


def _title_html(title, max_lines=2):
    """Truncate long titles and show a tooltip with the full title.
    The wrapper div enforces a fixed height so all cards line up."""
    safe = str(title).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    line_height = 1.25
    height_px = int(max_lines * 16 * line_height) + 4
    return (
        f"<div title='{safe}' style='height:{height_px}px;line-height:{line_height};"
        f"overflow:hidden;display:-webkit-box;-webkit-line-clamp:{max_lines};"
        f"-webkit-box-orient:vertical;font-weight:600;margin-top:8px;'>"
        f"{safe}</div>"
    )


def _meta_html(year=None, rating=None):
    bits = []
    if year and year != "—":
        bits.append(f"📅 {year}")
    if rating:
        bits.append(f"⭐ {rating:.1f}")
    if not bits:
        return ""
    return (
        f"<div style='color:#9aa0a6;font-size:12px;margin-top:4px;'>"
        f"{' | '.join(bits)}</div>"
    )


def _movie_card_html(poster_url, title, year=None, rating=None, height=320):
    return (
        _poster_html(poster_url, height=height)
        + _title_html(title)
        + _meta_html(year, rating)
    )


def render_grid(cards_data, columns=6, poster_height=320):
    """Render a list of card dicts evenly across `columns` columns.
    Each card dict can have: poster, title, year, rating."""
    if not cards_data:
        st.info("No movies to display.")
        return
    cols = st.columns(columns, gap="small")
    for idx, card in enumerate(cards_data):
        with cols[idx % columns]:
            st.markdown(
                _movie_card_html(
                    card.get("poster"),
                    card.get("title", "Unknown"),
                    card.get("year"),
                    card.get("rating"),
                    height=poster_height,
                ),
                unsafe_allow_html=True,
            )


def _omdb_card(title, year=None):
    """Build a card dict for a movie looked up by title via OMDb."""
    data, _ = _omdb_get({"t": _clean_title_for_search(title), "type": "movie"})
    if not data or data.get("Response") == "False":
        return {"title": title, "poster": None, "year": year or "—", "rating": None}
    poster = data.get("Poster")
    if poster == "N/A":
        poster = None
    return {
        "title": data.get("Title", title),
        "poster": poster,
        "year": data.get("Year", year or "—"),
        "rating": _safe_float(data.get("imdbRating")),
    }


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_dashboard_section(titles):
    """Fetch OMDb records for a curated list of titles."""
    cards = []
    for t in titles:
        cards.append(_omdb_card(t))
    return cards


# --- Sidebar ---
with st.sidebar:
    st.title("🎬 Movie Hub")
    st.markdown("---")
    page = st.radio(
        "📍 Navigation",
        ["🏠 Dashboard", "🔍 Recommend", "🔥 Trending", "⭐ Popular",
         "🏆 Top Rated", "🔎 Search"],
    )
    st.markdown("---")
    st.subheader("📊 Library Stats")
    st.metric("Total Movies", len(movies))
    st.metric("Recommendation Engine", "Cosine Similarity")
    st.markdown("---")
    st.subheader("⚙️ API Settings")
    api_key_input = st.text_input(
        "OMDb API Key (optional)",
        value="",
        type="password",
        help="Leave blank to use the default 'thewdb' key.",
    )
    if api_key_input:
        os.environ["OMDB_API_KEY"] = api_key_input
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption("Current key: `" + (OMDB_API_KEY[:4] + "…") + "`")
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write(
        "A movie recommendation dashboard powered by **Machine Learning** "
        "and the **OMDb API**."
    )
    st.write("Built with ❤️ using Streamlit")


# --- Curated title lists for the homepage ---
HOME_TITLES = [
    "Inception", "The Dark Knight", "Interstellar", "The Matrix",
    "Avengers: Endgame", "Spider-Man: No Way Home", "Joker", "Parasite",
    "The Shawshank Redemption", "Pulp Fiction", "Forrest Gump",
    "The Godfather", "Fight Club", "Gladiator", "Titanic",
    "The Lord of the Rings: The Return of the King", "Star Wars",
    "Harry Potter and the Sorcerer's Stone", "Iron Man", "Black Panther",
]


# --- Page: Dashboard ---
if page == "🏠 Dashboard":
    st.title("🎬 Movies Recommendation Dashboard")
    st.markdown("##### Discover, explore, and get personalized movie recommendations")
    st.markdown("---")

    st.subheader("🔥 Trending Picks")
    with st.spinner("Loading movies..."):
        cards = get_dashboard_section(HOME_TITLES[:12])
    render_grid(cards, columns=6)

    st.markdown("---")
    st.subheader("⭐ All-Time Favorites")
    with st.spinner("Loading movies..."):
        cards = get_dashboard_section(HOME_TITLES[6:18])
    render_grid(cards, columns=6)

# --- Page: Recommend ---
elif page == "🔍 Recommend":
    st.title("🔍 Movie Recommender")
    st.markdown("Pick a movie you like, and we'll suggest 5 similar titles.")
    st.markdown("---")

    selected_movie = st.selectbox(
        "🎥 Type or select a movie to get recommendation",
        movie_list,
    )

    if st.button('🎯 Show Recommendation', use_container_width=True):
        with st.spinner("Finding similar movies..."):
            names, posters, years = recommend(selected_movie)

        if not names:
            st.error("No recommendations could be generated. "
                     "Check that movie_list.pkl is valid.")
        else:
            st.success(f"Top 5 movies similar to **{selected_movie}**:")
            cards = [
                {"title": n, "poster": p, "year": y}
                for n, p, y in zip(names, posters, years)
            ]
            render_grid(cards, columns=5, poster_height=380)

# --- Page: Trending ---
elif page == "🔥 Trending":
    st.title("🔥 Trending Picks")
    st.markdown("##### A curated list of must-watch movies")
    st.markdown("---")
    with st.spinner("Loading movies..."):
        cards = get_dashboard_section(HOME_TITLES[:12])
    render_grid(cards, columns=6)

# --- Page: Popular ---
elif page == "⭐ Popular":
    st.title("⭐ Popular Movies")
    st.markdown("##### Movies everyone loves")
    st.markdown("---")
    with st.spinner("Loading movies..."):
        cards = get_dashboard_section(HOME_TITLES[6:18])
    render_grid(cards, columns=6)

# --- Page: Top Rated ---
elif page == "🏆 Top Rated":
    st.title("🏆 Top Rated Classics")
    st.markdown("##### Highest rated films of all time")
    st.markdown("---")
    with st.spinner("Loading movies..."):
        cards = get_dashboard_section([
            "The Shawshank Redemption", "The Godfather", "The Dark Knight",
            "12 Angry Men", "Schindler's List",
            "The Lord of the Rings: The Return of the King",
            "Pulp Fiction", "Forrest Gump", "Fight Club", "Inception",
            "The Matrix", "Goodfellas",
        ])
    render_grid(cards, columns=6)

# --- Page: Search ---
elif page == "🔎 Search":
    st.title("🔎 Search Movies")
    st.markdown("##### Find any movie using OMDb's search")
    st.markdown("---")

    query = st.text_input("🎬 Enter a movie name", placeholder="e.g. Inception")
    if query:
        with st.spinner(f"Searching for '{query}'..."):
            results, err = search_movies(query)
        if err:
            st.error(err)
        if results:
            st.write(f"Found **{len(results)}** results")
            cards = [
                {
                    "title": r.get("Title", "Unknown"),
                    "poster": r.get("Poster") if r.get("Poster") != "N/A" else None,
                    "year": r.get("Year", "—"),
                }
                for r in results
            ]
            render_grid(cards, columns=6)
        elif not err:
            st.warning("No results found. Try a different keyword.")
