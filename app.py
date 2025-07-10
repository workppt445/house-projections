import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import json
import os
import base64
from io import BytesIO

# ================= Configuration =================
st.set_page_config(
    page_title="Melbourne House Price Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Constants ----------------
SECRET_CODE = "7477"
EDITOR_STATE_FILE = "editor_state.json"
CONTENT_FILE = "editable_content.md"
MAX_FUTURE_YEARS = 5

# -------------- Session State Init --------------
if 'live_editor' not in st.session_state:
    st.session_state.live_editor = False
if 'homepage_md' not in st.session_state:
    st.session_state.homepage_md = (
        "# 🏡 Melbourne House Price Explorer\n"
        "Explore historical and forecasted house prices across Melbourne.\n"
        "Select a map point, set filters, and dive into the data!"
    )
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# ============ Editor Mode Functions ============
def load_editor_state():
    if os.path.exists(EDITOR_STATE_FILE):
        try:
            return json.load(open(EDITOR_STATE_FILE))
        except:
            return {'live_editor': False}
    return {'live_editor': False}

def save_editor_state(state):
    with open(EDITOR_STATE_FILE, 'w') as f:
        json.dump(state, f)

editor_state = load_editor_state()

# --------------- Sidebar: Editor Toggle ---------------
st.sidebar.title("🔐 Unlock Developer Mode")
code_input = st.sidebar.text_input("Enter secret code:", type="password")
if code_input == SECRET_CODE:
    editor_state['live_editor'] = not editor_state.get('live_editor', False)
    save_editor_state(editor_state)
    st.experimental_rerun()
if editor_state.get('live_editor'):
    st.sidebar.success("🔓 Developer Mode Active")

# ------------- Editable Homepage -------------
if editor_state.get('live_editor'):
    md_text = st.sidebar.text_area("Edit Homepage Markdown:", value=st.session_state.homepage_md, height=250)
    if st.sidebar.button("Save Homepage"):
        st.session_state.homepage_md = md_text
        with open(CONTENT_FILE, 'w') as f:
            f.write(md_text)
        st.sidebar.success("Homepage updated!")

# Display homepage
if editor_state.get('live_editor') and os.path.exists(CONTENT_FILE):
    st.markdown(open(CONTENT_FILE).read(), unsafe_allow_html=True)
else:
    st.markdown(st.session_state.homepage_md, unsafe_allow_html=True)

st.markdown("---")

# ------------- Sidebar: Theme & Navigation -------------
theme = st.sidebar.selectbox("Theme:", ["Light", "Dark"])
bg = "#FFFFFF" if theme == "Light" else "#121212"
fg = "#000000" if theme == "Light" else "#EEEEEE"
st.markdown(f"<style>body{{background-color:{bg};color:{fg}}}</style>", unsafe_allow_html=True)

page = st.sidebar.radio("Go to:", ["Map & Trends", "Heatmap", "Comparison", "Favorites & Notes", "About"])

# ================= Data Loading =================
@st.cache_data
def load_data():
    try:
        st.write("📦 Loading price data...")
        prices = pd.read_csv("house-prices-by-small-area-sale-year.csv")

        st.write("🏠 Loading dwellings data...")
        dwellings = pd.read_csv("city-of-melbourne-dwellings-and-household-forecasts-by-small-area-2020-2040.csv")

        st.write("🔧 Converting data...")
        for col in ['sale_year', 'median_price']:
            if col in prices.columns:
                prices[col] = pd.to_numeric(prices[col], errors='coerce')
            if col in dwellings.columns:
                dwellings[col] = pd.to_numeric(dwellings[col], errors='coerce')

        st.write("✅ Done loading data.")
        return prices.dropna(subset=['sale_year', 'median_price']), dwellings

    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame(), pd.DataFrame()

prices_df, dwellings_df = load_data()

# ================= Pages =================
if page == "Map & Trends":
    st.header("📍 Interactive Map & Trends")

    st.subheader("🧪 Debug Info")
    st.write("Available columns in price data:")
    st.write(prices_df.columns.tolist())

    if prices_df.empty or 'latitude' not in prices_df.columns or 'longitude' not in prices_df.columns:
        st.warning("Map data is missing latitude/longitude columns; map disabled.")
    else:
        midpoint = (prices_df['latitude'].mean(), prices_df['longitude'].mean())
        view = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=prices_df,
            get_position='[longitude, latitude]',
            get_fill_color='[0, 150, 255, 180]',
            get_radius=200,
            pickable=True
        )
        deck = pdk.Deck(layers=[layer], initial_view_state=view,
                        tooltip={"html": "<b>Area:</b> {small_area}<br/>"
                                         "<b>Year:</b> {sale_year}<br/>"
                                         "<b>Price:</b> ${median_price:,.0f}"})
        st.pydeck_chart(deck)

# ---- Add other pages (Heatmap, Comparison, etc.) here if needed ----

st.markdown("---")
st.write("*Data source: City of Melbourne Open Data Portal.*")
