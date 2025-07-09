import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import json
import os
import base64
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import altair as alt

# ================= Configuration =================
st.set_page_config(
    page_title="Melbourne House Price Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Constants ------------------
DATA_PATH = "house-prices-by-small-area-sale-year.csv"  # adjust if needed
EDITOR_STATE_FILE = "editor_state.json"
CONTENT_FILE = "editable_content.md"
SECRET_CODE = "7477"
MAX_FUTURE_YEARS = 5

# ================= Helper Functions =================

@st.cache_data
def load_data(path=DATA_PATH):
    """Load and preprocess house price data from CSV."""
    df = pd.read_csv(path)
    # Standardize columns
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Ensure essential columns exist
    required_cols = ['small_area', 'sale_year', 'median_price', 'latitude', 'longitude', 'property_type']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in data: {missing}")
        return pd.DataFrame()
    # Clean numeric columns
    for col in ['latitude', 'longitude', 'sale_year', 'median_price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop incomplete rows
    df = df.dropna(subset=['latitude', 'longitude', 'sale_year', 'median_price', 'small_area', 'property_type'])
    # Filter data to City of Melbourne if jurisdiction present
    if 'jurisdiction' in df.columns:
        df = df[df['jurisdiction'].str.lower() == 'city of melbourne']
    return df

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

def get_homepage_content(editor_state):
    if editor_state.get('live_editor') and os.path.exists(CONTENT_FILE):
        return open(CONTENT_FILE).read()
    return (
        "# 🏡 Melbourne House Price Explorer\n\n"
        "Welcome! Explore Melbourne house prices historically and forecast the next 5 years.\n\n"
        "Use the sidebar to navigate, select areas on the map, and save favorites."
    )

def generate_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'

def add_favorite(area):
    if 'favorites' not in st.session_state:
        st.session_state['favorites'] = []
    if area not in st.session_state['favorites']:
        st.session_state['favorites'].append(area)
        st.success(f"Added {area} to favorites.")
    else:
        st.info(f"{area} is already in your favorites.")

def remove_favorite(area):
    if 'favorites' in st.session_state and area in st.session_state['favorites']:
        st.session_state['favorites'].remove(area)
        st.success(f"Removed {area} from favorites.")

def fit_projection_model(df_sub):
    """Fit polynomial regression for price projection."""
    X = df_sub['sale_year'].values.reshape(-1, 1)
    y = df_sub['median_price'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    return model, poly

def predict_future_prices(model, poly, start_year, years_ahead=MAX_FUTURE_YEARS):
    future_years = np.arange(start_year + 1, start_year + years_ahead + 1)
    future_poly = poly.transform(future_years.reshape(-1, 1))
    preds = model.predict(future_poly)
    df_pred = pd.DataFrame({'sale_year': future_years, 'median_price': preds})
    return df_pred

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ==================== Load Data ====================
df = load_data()
editor_state = load_editor_state()

# ==================== Sidebar: Secret Code & Editor ====================
st.sidebar.title("🔐 Unlock Features")
code = st.sidebar.text_input("Enter secret code:", type="password")

if code == SECRET_CODE:
    editor_state['live_editor'] = not editor_state.get('live_editor', False)
    save_editor_state(editor_state)
    if editor_state['live_editor']:
        st.sidebar.success("Live Editor Mode ENABLED")
    else:
        # Reset content
        if os.path.exists(CONTENT_FILE):
            os.remove(CONTENT_FILE)
        st.sidebar.warning("Live Editor Mode DISABLED. Content reset.")
    st.experimental_rerun()

if editor_state.get('live_editor'):
    st.sidebar.subheader("Live Editor")
    current_md = open(CONTENT_FILE).read() if os.path.exists(CONTENT_FILE) else ""
    new_md = st.sidebar.text_area("Edit Homepage Markdown:", current_md, height=250)
    if st.sidebar.button("Save Content"):
        with open(CONTENT_FILE, 'w') as f:
            f.write(new_md)
        st.sidebar.success("Homepage content updated!")
        st.experimental_rerun()

# ==================== Main Page Content ====================
st.markdown(get_homepage_content(editor_state))
st.markdown("---")

# ==================== Navigation ====================
st.sidebar.title("🏠 Navigation")
nav_option = st.sidebar.radio("Select a page:", 
                             ['Map & Trends', 'Heatmap', 'Comparison', 'Favorites & Notes', 'About'])

# ==================== Theming ====================
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
if theme == "Light":
    bg_color = "#FFFFFF"
    fg_color = "#000000"
else:
    bg_color = "#121212"
    fg_color = "#EEEEEE"
st.markdown(f"<style>body{{background-color: {bg_color}; color: {fg_color};}}</style>", unsafe_allow_html=True)

# ==================== Pages ====================

if nav_option == 'Map & Trends':
    st.header("📍 Interactive Map & Price Trends")

    if df.empty:
        st.warning("No data loaded.")
    else:
        # Map center
        midpoint = (df['latitude'].mean(), df['longitude'].mean())
        view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11)

        # Scatter layer for houses
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_fill_color='[0, 120, 255, 180]',
            get_radius=250,
            pickable=True,
            auto_highlight=True,
        )

        r = pdk.Deck(
            layers=[scatter],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/light-v9',
            tooltip={
                "html": "<b>Area:</b> {small_area} <br/>"
                        "<b>Year:</b> {sale_year} <br/>"
                        "<b>Median Price:</b> ${median_price:,.0f} <br/>"
                        "<b>Property Type:</b> {property_type}"
            },
            mapbox_key=os.getenv('MAPBOX_TOKEN', ''),
        )
        st.pydeck_chart(r)

        # Filter options
        st.subheader("Filter Options")
        cols = st.columns([1,1,1])
        with cols[0]:
            area_selected = st.selectbox("Select Suburb", sorted(df['small_area'].unique()))
        with cols[1]:
            prop_selected = st.selectbox("Select Property Type", sorted(df['property_type'].unique()))
        with cols[2]:
            year_min = int(df['sale_year'].min())
            year_max = int(df['sale_year'].max())
            year_range = st.slider("Select Year Range", year_min, year_max, (year_min, year_max))

        filtered_df = df[
            (df['small_area'] == area_selected) &
            (df['property_type'] == prop_selected) &
            (df['sale_year'] >= year_range[0]) &
            (df['sale_year'] <= year_range[1])
        ]

        if filtered_df.empty:
            st.warning("No data for selected filters.")
        else:
            # Historical Price Plot
            st.subheader(f"Historical Prices for {area_selected} ({prop_selected})")
            hist_fig = px.line(
                filtered_df, x='sale_year', y='median_price', markers=True,
                labels={"sale_year": "Year", "median_price": "Median Price ($)"},
                title=f"Median House Price Over Years in {area_selected}"
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            # Fit and show projection
            model, poly = fit_projection_model(filtered_df)
            future_df = predict_future_prices(model, poly, filtered_df['sale_year'].max(), MAX_FUTURE_YEARS)

            st.subheader(f"{MAX_FUTURE_YEARS}-Year Price Projection")
            proj_fig = px.line(
                future_df, x='sale_year', y='median_price', markers=True,
                labels={"sale_year": "Year", "median_price": "Projected Median Price ($)"},
                title=f"Projected Median Prices for {area_selected}"
            )
            st.plotly_chart(proj_fig, use_container_width=True)

            # Show model accuracy RMSE
            train_preds = model.predict(poly.transform(filtered_df['sale_year'].values.reshape(-1,1)))
            rmse = calculate_rmse(filtered_df['median_price'], train_preds)
            st.metric("Model RMSE", f"${rmse:,.2f}")

            # Download filtered data CSV
            st.markdown(generate_download_link(filtered_df, f"{area_selected}_{prop_selected}_data.csv"), unsafe_allow_html=True)

            # Add to favorites
            if st.button("⭐ Add this area to Favorites"):
                add_favorite(area_selected)

            # Show favorites if any
            if 'favorites' in st.session_state and st.session_state['favorites']:
                st.markdown("### ⭐ Your Favorites")
                for fav in st.session_state['favorites']:
                    st.write(f"- {fav}")

elif nav_option == 'Heatmap':
    st.header("🌡️ Melbourne House Price Heatmap")
    if df

