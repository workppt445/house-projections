import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import json
import os

# Config
st.set_page_config(page_title="Melbourne House Price Explorer", layout="wide")

# Secret code and editor state
def load_editor_state():
    if os.path.exists('editor_state.json'):
        return json.load(open('editor_state.json'))
    return {'live_editor': False}

def save_editor_state(state):
    json.dump(state, open('editor_state.json', 'w'))

editor_state = load_editor_state()

# Secret code input
code = st.sidebar.text_input("Enter secret code to unlock features:", type="password")
if code == '7477' and not editor_state['live_editor']:
    editor_state['live_editor'] = True
    save_editor_state(editor_state)
    st.sidebar.success("Live editor mode ON. Changes are global until you turn it off or refresh.")
elif code == '7477' and editor_state['live_editor']:
    editor_state['live_editor'] = False
    save_editor_state(editor_state)
    # reset editable content
    if os.path.exists('editable_content.md'):
        os.remove('editable_content.md')
    st.sidebar.warning("Live editor mode OFF. Site reset to normal.")

# Editable homepage content
def get_home_content():
    if editor_state['live_editor'] and os.path.exists('editable_content.md'):
        return open('editable_content.md').read()
    return "# Welcome to the Melbourne House Price Explorer\n" \
           "Select a point on the map to view historical trends and projections."

# If in live editor mode, show editor
if editor_state['live_editor']:
    st.sidebar.subheader("Live Editor")
    text = st.sidebar.text_area("Edit homepage content:", get_home_content(), height=200)
    if st.sidebar.button("Save Content"):
        with open('editable_content.md', 'w') as f:
            f.write(text)
        st.sidebar.success("Content updated for all users.")

# Main page
st.markdown(get_home_content())

# Load preloaded data
df = pd.read_csv("data/house_prices_melbourne.csv")  # Replace with actual local path or remote file

# Preprocess
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Map plot
if 'latitude' in df.columns and 'longitude' in df.columns:
    st.subheader("📍 Select an area on the map")
    midpoint = (np.mean(df['latitude']), np.mean(df['longitude']))
    view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=12)
    tile_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[longitude, latitude]',
        get_fill_color='[0, 100, 255, 160]',
        get_radius=250,
        pickable=True,
        tooltip=True
    )
    r = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        layers=[tile_layer],
        initial_view_state=view_state,
        tooltip={"text": "{small_area} \nPrice: ${median_price}"}
    )
    st.pydeck_chart(r)

    # Area selector
    st.subheader("Selected Area Details")
    area = st.selectbox("Choose Small Area:", df['small_area'].unique())
    sub = df[df['small_area'] == area]

    # Historical chart
    fig1 = px.line(sub, x='sale_year', y='median_price', title=f"Historical Prices for {area}", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Projection (linear regression)
    X = sub['sale_year'].values.reshape(-1,1)
    y = sub['median_price'].values
    model = LinearRegression().fit(X, y)
    future_years = np.arange(sub['sale_year'].max()+1, sub['sale_year'].max()+6)
    preds = model.predict(future_years.reshape(-1,1))
    df_pred = pd.DataFrame({'sale_year': future_years, 'median_price': preds})
    fig2 = px.line(df_pred, x='sale_year', y='median_price', title="5-Year Projection", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Cool features
    st.subheader("🔧 Other Cool Features")
    st.markdown("1. Download Data 📥\n2. Heatmap View 🌡️\n3. Comparison Mode 🆚\n4. Export Report 📝\n5. Notifications 🔔\n6. Theme Switcher 🌗\n7. Favorites 🌟\n8. API Access 🔗\n9. Community Chat 💬\n10. [Secret] Surprise! 🎉")
    if editor_state['live_editor']:
        st.markdown("**Live editor is active!** Changes will reset on refresh.")
else:
    st.error("Map data missing required latitude and longitude columns.")
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

# ================= Configuration =================
st.set_page_config(
    page_title="Melbourne House Price Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Constants ------------------
DATA_PATH = "data/house_prices_melbourne.csv"
EDITOR_STATE_FILE = "editor_state.json"
CONTENT_FILE = "editable_content.md"
SECRET_CODE = "7477"
MAX_FUTURE_YEARS = 5

# ================= Caching Functions =================
@st.cache_data
 def load_data(path=DATA_PATH):
    """Load and preprocess house price data."""
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Filter jurisdiction if exists
    if 'jurisdiction' in df.columns:
        df = df[df['jurisdiction'] == 'City of Melbourne']
    # Ensure numeric
    for col in ['latitude', 'longitude', 'sale_year', 'median_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'sale_year', 'median_price'])
    return df

@st.cache_data
 def load_editor_state():
    if os.path.exists(EDITOR_STATE_FILE):
        try:
            return json.load(open(EDITOR_STATE_FILE))
        except:
            return {'live_editor': False}
    return {'live_editor': False}

# ================= State Management =================
editor_state = load_editor_state()

# Secret code input and toggle
code_input = st.sidebar.text_input("Enter secret code to unlock features:", type="password")
if code_input == SECRET_CODE:
    editor_state['live_editor'] = not editor_state.get('live_editor', False)
    json.dump(editor_state, open(EDITOR_STATE_FILE, 'w'))
    if editor_state['live_editor']:
        st.sidebar.success("Live Editor mode ON. Global edits enabled until toggled off.")
    else:
        # reset custom content
        if os.path.exists(CONTENT_FILE): os.remove(CONTENT_FILE)
        st.sidebar.warning("Live Editor mode OFF. Site reset to default.")
    st.experimental_rerun()

# Editable homepage content
if editor_state.get('live_editor', False):
    st.sidebar.subheader("Live Editor")
    current_md = open(CONTENT_FILE).read() if os.path.exists(CONTENT_FILE) else ""
    new_md = st.sidebar.text_area("Edit homepage markdown:", current_md, height=200)
    if st.sidebar.button("Save Content"):
        with open(CONTENT_FILE, 'w') as f:
            f.write(new_md)
        st.sidebar.success("Homepage content updated.")
        st.experimental_rerun()

# ================= Homepage =================
def get_homepage_markdown():
    if editor_state.get('live_editor') and os.path.exists(CONTENT_FILE):
        return open(CONTENT_FILE).read()
    return (
        "# 🏡 Melbourne House Price Explorer\n"
        "Explore past trends and future projections across Melbourne suburbs.\n"
        "Use the sidebar to navigate features!"
    )

st.markdown(get_homepage_markdown())
st.markdown("---")

# ================= Sidebar =================
st.sidebar.title("🔍 Navigation")
nav = st.sidebar.radio("Go to:", ["Map & Trends", "Heatmap", "Comparison", "Favorites & Notes"])

st.sidebar.title("✨ Cool Features")
features = [
    "Interactive PyDeck Map",
    "Historical Trends",
    "Polynomial Regression Projections",
    "Theme Switch (Light/Dark)",
    "Download Data as CSV",
    "Heatmap View with Folium",
    "Comparison Mode for Areas",
    "Session-based Favorites",
    "Community Notes Placeholder",
    "Live Editor Mode"
]
for f in features:
    st.sidebar.write(f)
# Theme switcher
theme = st.sidebar.selectbox("Theme:", ["Light", "Dark"], index=0)
bg = "#FFFFFF" if theme == "Light" else "#111111"
fg = "#000000" if theme == "Light" else "#EEEEEE"
st.markdown(f"<style>body{{background-color:{bg};color:{fg}}}</style>", unsafe_allow_html=True)

# ================= Load Data =================
df = load_data()

# ================= Helper Functions =================
def get_filtered_df(area, prop_type, year_min, year_max):
    return df[(df['small_area']==area)
              & (df['property_type']==prop_type)
              & (df['sale_year']>=year_min)
              & (df['sale_year']<=year_max)]

# ================= Main Sections =================

# ---- Section: Map & Trends ----
if nav == "Map & Trends":
    st.header("📍 Interactive Map & Trends")
    # PyDeck map
    midpoint = (df['latitude'].mean(), df['longitude'].mean())
    view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11)
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[longitude, latitude]',
        get_fill_color='[0, 120, 255, 140]',
        get_radius=300,
        pickable=True,
        auto_highlight=True
    )
    deck = pdk.Deck(layers=[scatter], initial_view_state=view_state,
                    tooltip={"html": "<b>Area:</b> {small_area}<br/><b>Year:</b> {sale_year}<br/><b>Price:</b> ${median_price}",
                             "style": {"color": "white"}},
                    map_style="mapbox://styles/mapbox/light-v9",
                    mapbox_key=os.getenv('MAPBOX_TOKEN', ''))
    st.pydeck_chart(deck)

    # Filters
    st.subheader("Filter Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        chosen_area = st.selectbox("Suburb:", sorted(df['small_area'].unique()))
    with col2:
        chosen_type = st.selectbox("Property Type:", sorted(df['property_type'].unique()))
    with col3:
        year_range = st.slider("Sale Year Range:", int(df['sale_year'].min()), int(df['sale_year'].max()),
                               (int(df['sale_year'].min()), int(df['sale_year'].max())))
    sub_df = get_filtered_df(chosen_area, chosen_type, year_range[0], year_range[1])

    if sub_df.empty:
        st.warning("No data available for these filters.")
    else:
        # Historical plot
        fig_hist = px.line(sub_df, x='sale_year', y='median_price', markers=True,
                           title=f"Historical Prices: {chosen_area} ({chosen_type})")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Projection
        X = sub_df['sale_year'].values.reshape(-1,1)
        y = sub_df['median_price'].values
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        future_years = np.arange(sub_df['sale_year'].max()+1,
                                  sub_df['sale_year'].max()+MAX_FUTURE_YEARS+1)
        future_poly = poly.transform(future_years.reshape(-1,1))
        preds = model.predict(future_poly)
        df_proj = pd.DataFrame({'sale_year': future_years, 'median_price': preds})
        fig_proj = px.line(df_proj, x='sale_year', y='median_price', markers=True,
                           title="5-Year Price Projection")
        st.plotly_chart(fig_proj, use_container_width=True)

        # RMSE
        train_preds = model.predict(X_poly)
        rmse = np.sqrt(mean_squared_error(y, train_preds))
        st.metric("Model RMSE", f"${rmse:,.0f}")

        # Download CSV
        csv_bytes = sub_df.to_csv(index=False).encode()
        b64str = base64.b64encode(csv_bytes).decode()
        st.markdown(f"[📥 Download Filtered Data](data:file/csv;base64,{b64str})")

# ---- Section: Heatmap ----
elif nav == "Heatmap":
    st.header("🌡️ Price Heatmap of Melbourne")
    m = folium.Map(location=midpoint, zoom_start=11, tiles="cartodbdark_matter")
    heat_data = df[['latitude','longitude','median_price']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    st_folium(m, width=700, height=500)

# ---- Section: Comparison ----
elif nav == "Comparison":
    st.header("🆚 Compare Two Suburbs")
    comp1 = st.selectbox("Suburb 1:", sorted(df['small_area'].unique()), key='c1')
    comp2 = st.selectbox("Suburb 2:", sorted(df['small_area'].unique()), key='c2')
    df1 = df[df['small_area']==comp1]
    df2 = df[df['small_area']==comp2]
    if df1.empty or df2.empty:
        st.warning("Select two suburbs with data.")
    else:
        df1['label'] = comp1
        df2['label'] = comp2
        df_cmp = pd.concat([df1, df2])
        fig_cmp = px.line(df_cmp, x='sale_year', y='median_price', color='label', markers=True,
                          title=f"Comparison: {comp1} vs {comp2}")
        st.plotly_chart(fig_cmp, use_container_width=True)

# ---- Section: Favorites & Notes ----
elif nav == "Favorites & Notes":
    st.header("⭐ Favorites & 💬 Community Notes")
    # Favorites
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    fav_area = st.selectbox("Mark Favorite Suburb:", sorted(df['small_area'].unique()))
    if st.button("Add to Favorites"):
        if fav_area not in st.session_state.favorites:
            st.session_state.favorites.append(fav_area)
            st.success(f"{fav_area} added to favorites.")
    st.write("Your Favorites:", st.session_state.favorites)
    
    # Community Notes placeholder
    note = st.text_area("Leave a note for the community:")
    if st.button("Post Note"):
        # Placeholder: in-memory only
        st.info(f"Note posted: {note}")

# ================= Footer =================
st.markdown("---")
st.write("*Data from City of Melbourne Open Data Portal. Built with Streamlit.*")
