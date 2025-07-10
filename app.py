import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
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
        "Explore Melbourne house prices historically and forecast the next 5 years.\n"
        "Use the sidebar to navigate between map, heatmap, comparison, and favorites."
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
    md_text = st.sidebar.text_area("Edit Homepage Markdown:", value=st.session_state.homepage_md, height=200)
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
    # Load price data
    prices = pd.read_csv("house-prices-by-small-area-sale-year.csv")
    prices.columns = [c.strip().lower().replace(' ', '_') for c in prices.columns]
    # Ensure property_type column exists
    if 'type' in prices.columns and 'property_type' not in prices.columns:
        prices.rename(columns={'type': 'property_type'}, inplace=True)
    # Add dummy coords if missing
    if 'latitude' not in prices.columns:
        prices['latitude'] = -37.8136
    if 'longitude' not in prices.columns:
        prices['longitude'] = 144.9631

    # Load dwellings data
    dwell = pd.read_csv("city-of-melbourne-dwellings-and-household-forecasts-by-small-area-2020-2040.csv")
    dwell.columns = [c.strip().lower().replace(' ', '_') for c in dwell.columns]
    # Ensure small_area and dwelling fields exist
    if 'geography' in dwell.columns:
        dwell.rename(columns={'geography': 'small_area'}, inplace=True)
    if 'category' in dwell.columns and 'dwelling_type' not in dwell.columns:
        dwell.rename(columns={'category': 'dwelling_type'}, inplace=True)
    if 'households' in dwell.columns and 'dwelling_number' not in dwell.columns:
        dwell.rename(columns={'households': 'dwelling_number'}, inplace=True)

    return prices, dwell

prices_df, dwell_df = load_data()

# Helper: download link
def download_csv(df, fname):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f"<a href='data:file/csv;base64,{b64}' download='{fname}'>📥 Download {fname}</a>"

# Pre-calculate midpoint
midpoint = (prices_df['latitude'].mean(), prices_df['longitude'].mean())

# ---- Map & Trends ----
if page == "Map & Trends":
    st.header("📍 Interactive Map & Trends")
    # Map
    view = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11)
    layer = pdk.Layer(
        "ScatterplotLayer", data=prices_df,
        get_position='[longitude, latitude]', get_fill_color='[0,120,255,160]', get_radius=200, pickable=True
    )
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
        tooltip={"text":"{small_area}\nYear: {sale_year}\nPrice: ${median_price:,.0f}"}
    ))
    # Filters
    st.subheader("Filters")
    area = st.selectbox("Suburb", sorted(prices_df['small_area'].dropna().unique()))
    ptype = st.selectbox("Type", sorted(prices_df['property_type'].dropna().unique()))
    yrs = st.slider("Year Range",
                     int(prices_df['sale_year'].min()),
                     int(prices_df['sale_year'].max()),
                     (int(prices_df['sale_year'].min()), int(prices_df['sale_year'].max())))
    sub = prices_df[
        (prices_df['small_area'] == area) &
        (prices_df['property_type'] == ptype) &
        (prices_df['sale_year'] >= yrs[0]) & (prices_df['sale_year'] <= yrs[1])
    ]
    if sub.empty:
        st.warning("No data for these filters.")
    else:
        # Pie chart
        if 'dwelling_type' in dwell_df.columns and 'dwelling_number' in dwell_df.columns:
            mix = dwell_df[dwell_df['small_area'] == area].groupby('dwelling_type')['dwelling_number'].sum().reset_index()
            pie = alt.Chart(mix).mark_arc().encode(theta='dwelling_number:Q', color='dwelling_type:N')
            st.altair_chart(pie, use_container_width=False)
        # Trend & Forecast
        st.subheader("Historical & 5-Year Forecast")
        base = alt.Chart(sub).mark_line(point=True).encode(x='sale_year:O', y='median_price:Q')
        model = LinearRegression().fit(sub[['sale_year']], sub['median_price'])
        future_years = np.arange(sub['sale_year'].max()+1, sub['sale_year'].max()+1+MAX_FUTURE_YEARS)
        future_df = pd.DataFrame({'sale_year': future_years})
        future_df['median_price'] = model.predict(future_df[['sale_year']])
        forecast = alt.Chart(future_df).mark_line(color='orange', strokeDash=[4,2]).encode(x='sale_year:O', y='median_price:Q')
        st.altair_chart((base + forecast).properties(title=f"{area} ({ptype})"), use_container_width=True)
        st.markdown(download_csv(sub, f"{area}_{ptype}.csv"), unsafe_allow_html=True)

# ---- Heatmap ----
elif page == "Heatmap":
    st.header("🌡️ Price Heatmap")
    m = folium.Map(location=midpoint, zoom_start=11)
    data = prices_df[['latitude','longitude','median_price']].dropna()
    data['intensity'] = (data['median_price'] - data['median_price'].min()) / (data['median_price'].max() - data['median_price'].min())
    HeatMap(data[['latitude','longitude','intensity']].values.tolist(), radius=15).add_to(m)
    st_folium(m, width=700, height=500)

# ---- Comparison ----
elif page == "Comparison":
    st.header("🔍 Compare Two Suburbs")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        suburb1 = st.selectbox("Suburb 1", sorted(prices_df['small_area'].dropna().unique()), key='comp1')
    with col2:
        suburb2 = st.selectbox("Suburb 2", sorted(prices_df['small_area'].dropna().unique()), index=1, key='comp2')
    with col3:
        st.markdown("#### Customize Comparison Chart")
        show_points = st.checkbox("Show data points", value=True)
        smooth = st.checkbox("Smooth lines (rolling avg)", value=False)

    df1 = prices_df[prices_df['small_area'] == suburb1].copy()
    df2 = prices_df[prices_df['small_area'] == suburb2].copy()
    if smooth:
        df1['median_price'] = df1['median_price'].rolling(3, min_periods=1).mean()
        df2['median_price'] = df2['median_price'].rolling(3, min_periods=1).mean()

    df1['Suburb'] = suburb1
    df2['Suburb'] = suburb2
    comp_df = pd.concat([df1, df2])

    chart = alt.Chart(comp_df).mark_line().encode(
        x=alt.X('sale_year:O', title='Year'),
        y=alt.Y('median_price:Q', title='Median Price (AUD)'),
        color=alt.Color('Suburb:N', title='Suburb'),
        tooltip=[alt.Tooltip('sale_year:O', title='Year'),
                 alt.Tooltip('median_price:Q', title='Median Price', format='$,.0f'),
                 alt.Tooltip('Suburb:N')]
    )
    if show_points:
        points = alt.Chart(comp_df).mark_point(size=50).encode(
            x='sale_year:O',
            y='median_price:Q',
            color='Suburb:N',
            tooltip=[alt.Tooltip('sale_year:O', title='Year'),
                     alt.Tooltip('median_price:Q', title='Median Price', format='$,.0f'),
                     alt.Tooltip('Suburb:N')]
        )
        chart = chart + points

    chart = chart.properties(
        width=700,
        height=400,
        title=f"House Price Comparison: {suburb1} vs {suburb2}"
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ---- Favorites & Notes ----
elif page == "Favorites & Notes":
    st.header("⭐ Favorites & Notes")
    for fav in st.session_state.favorites:
        st.write(fav)
        st.text_area(f"Notes for {fav}", key=f"note_{fav}")
    if st.button("Clear Favorites"):
        st.session_state.favorites = []

# ---- About ----
elif page == "About":
    st.header("ℹ️ About")
    st.write("Dashboard uses City of Melbourne open data to explore house prices.")

st.markdown("---")
st.write("*Data source: City of Melbourne.*")
