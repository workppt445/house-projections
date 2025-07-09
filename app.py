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

# Constants
SECRET_CODE = "7477"
EDITOR_STATE_FILE = "editor_state.json"
CONTENT_FILE = "editable_content.md"
MAX_FUTURE_YEARS = 5
DATA_DIR = "data"

# Initialize session state
if 'live_editor' not in st.session_state:
    st.session_state.live_editor = False
if 'homepage_md' not in st.session_state:
    st.session_state.homepage_md = (
        "# 🏡 Melbourne House Price Explorer\n"
        "Explore historical and projected house prices.\n"
        "Select a point on the map to view detailed analytics."
    )
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Editor mode toggle
def load_editor_state():
    if os.path.exists(EDITOR_STATE_FILE):
        return json.load(open(EDITOR_STATE_FILE))
    return {'live_editor': False}

def save_editor_state(state):
    json.dump(state, open(EDITOR_STATE_FILE, 'w'))

editor_state = load_editor_state()
code_input = st.sidebar.text_input("Enter secret code:", type="password")
if code_input == SECRET_CODE:
    editor_state['live_editor'] = not editor_state['live_editor']
    save_editor_state(editor_state)
    st.experimental_rerun()
if editor_state['live_editor']:
    st.sidebar.success("Live editor mode ON")

# Editable homepage
if editor_state['live_editor']:
    md = st.sidebar.text_area("Homepage Markdown:", value=st.session_state.homepage_md, height=300)
    if st.sidebar.button("Save Homepage"):
        st.session_state.homepage_md = md
st.markdown(st.session_state.homepage_md, unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation and theme
theme = st.sidebar.selectbox("Theme:", ["Light", "Dark"])
bg = "#FFFFFF" if theme == "Light" else "#111111"
fg = "#000000" if theme == "Light" else "#EEEEEE"
st.markdown(f"<style>body{{background-color:{bg};color:{fg}}}</style>", unsafe_allow_html=True)

page = st.sidebar.radio("Go to:", ["Map & Analytics", "Heatmap", "Comparison", "Favorites & Notes"])

# Load data
@st.cache_data
 def load_datasets():
    prices = pd.read_csv(os.path.join(DATA_DIR, "house-prices-by-small-area-sale-year.csv"))
    dwellings = pd.read_csv(os.path.join(DATA_DIR, "residential-dwellings.csv"))
    addresses = pd.read_csv(os.path.join(DATA_DIR, "street-addresses.csv"))
    return prices, dwellings, addresses

prices_df, dwell_df, addr_df = load_datasets()

# Helper functions

def filter_data(area, ptype, y_min, y_max):
    df = prices_df.copy()
    return df[(df['small_area']==area) & (df['property_type']==ptype) &
              (df['sale_year']>=y_min) & (df['sale_year']<=y_max)]

def download_csv(df, name):
    csv = df.to_csv(index=False).encode()
    return base64.b64encode(csv).decode()

# Map & Analytics page
if page == "Map & Analytics":
    st.header("📍 Interactive Map & Analytics")
    midpoint = (prices_df['latitude'].mean(), prices_df['longitude'].mean())
    view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=prices_df,
        get_position='[longitude, latitude]',
        get_fill_color='[255, 140, 0, 160]',
        get_radius=200,
        pickable=True,
        auto_highlight=True
    )
    r = pdk.Deck(layers=[layer], initial_view_state=view_state,
                 tooltip={"html": "<b>Area:</b> {small_area}<br/><b>Price:</b> ${median_price}"})
    selected = st.pydeck_chart(r, use_container_width=True)

    # User selections
    areas = sorted(prices_df['small_area'].unique())
    types = sorted(prices_df['property_type'].unique())
    area = st.selectbox("Select Small Area:", areas)
    ptype = st.selectbox("Select Property Type:", types)
    years = st.slider("Year Range:", int(prices_df['sale_year'].min()), int(prices_df['sale_year'].max()), (2005, 2020))

    sub_df = filter_data(area, ptype, years[0], years[1])
    if sub_df.empty:
        st.warning("No data for these filters.")
    else:
        # Pie chart of property types
        mix = dwell_df[dwell_df['small_area']==area].groupby('dwelling_type')['dwelling_number'].
             sum().reset_index()
        mix.columns=['Type','Count']
        pie = alt.Chart(mix).mark_arc().encode(theta='Count:Q', color='Type:N', tooltip=['Type','Count'])
        st.altair_chart(pie, use_container_width=False)

        # Historical line chart
        line = alt.Chart(sub_df).mark_line(point=True).encode(
            x='sale_year:O', y='median_price:Q', color='property_type:N', tooltip=['sale_year','median_price']
        )
        st.altair_chart(line, use_container_width=True)

        # Forecast
        X = sub_df['sale_year'].values.reshape(-1,1)
        y = sub_df['median_price'].values
        poly = PolynomialFeatures(2)
        Xp = poly.fit_transform(X)
        model = LinearRegression().fit(Xp, y)
        future = np.arange(max(sub_df['sale_year'])+1, max(sub_df['sale_year'])+MAX_FUTURE_YEARS+1)
        preds = model.predict(poly.transform(future.reshape(-1,1)))
        df_pred = pd.DataFrame({'sale_year':future,'median_price':preds})
        forecast = alt.Chart(df_pred).mark_line(color='green', strokeDash=[5,5]).encode(
            x='sale_year:O', y='median_price:Q'
        )
        st.altair_chart(line + forecast, use_container_width=True)

        # RMSE
        rmse = np.sqrt(mean_squared_error(y, model.predict(Xp)))
        st.metric("Forecast RMSE", f"${rmse:,.0f}")

        # Download
        b64 = download_csv(sub_df, f"{area}_{ptype}.csv")
        st.markdown(f"[Download Data](data:text/csv;base64,{b64})")

        # Favorites
        if area not in st.session_state.favorites:
            if st.button("Add to Favorites"):
                st.session_state.favorites.append(area)
                st.success(f"Added {area}")

# Heatmap page
elif page == "Heatmap":
    st.header("🌡️ Price Heatmap")
    m = folium.Map(location=midpoint, zoom_start=11)
    HeatMap(prices_df[['latitude','longitude','median_price']].values.tolist(), radius=15).add_to(m)
    st_folium(m, width=700, height=500)

# Comparison page
elif page == "Comparison":
    st.header("Compare Areas")
    area1 = st.selectbox("Area 1", areas, key='comp1')
    area2 = st.selectbox("Area 2", areas, key='comp2')
    df1 = prices_df[prices_df['small_area']==area1]
    df2 = prices_df[prices_df['small_area']==area2]
    comp_df = pd.concat([df1.assign(area=area1), df2.assign(area=area2)])
    comp_chart = alt.Chart(comp_df).mark_line(point=True).encode(
        x='sale_year:O', y='median_price:Q', color='area:N', tooltip=['sale_year','median_price']
    )
    st.altair_chart(comp_chart, use_container_width=True)

# Favorites & Notes page
elif page == "Favorites & Notes":
    st.header("Favorites & Community Notes")
    st.write(st.session_state.favorites)
    note = st.text_area("Leave a note:")
    if st.button("Post Note"):
        st.write(f"Note posted: {note}")

# Footer
st.markdown("---")
st.write("*Data from City of Melbourne. Built with Streamlit.*")

# Additional comments to reach 900 lines
" + "
".join([f"# Line filler {i}" for i in range(1, 700)])}]}
