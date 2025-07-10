import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import os
import base64
from io import BytesIO

st.set_page_config(
    page_title="Melbourne House Price Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

SECRET_CODE = "7477"
EDITOR_STATE_FILE = "editor_state.json"
CONTENT_FILE = "editable_content.md"
MAX_FUTURE_YEARS = 5

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

st.sidebar.title("🔐 Unlock Developer Mode")
code_input = st.sidebar.text_input("Enter secret code:", type="password")
if code_input == SECRET_CODE:
    editor_state['live_editor'] = not editor_state.get('live_editor', False)
    save_editor_state(editor_state)
    st.experimental_rerun()
if editor_state.get('live_editor'):
    st.sidebar.success("🔓 Developer Mode Active")

if editor_state.get('live_editor'):
    md_text = st.sidebar.text_area("Edit Homepage Markdown:", value=st.session_state.homepage_md, height=200)
    if st.sidebar.button("Save Homepage"):
        st.session_state.homepage_md = md_text
        with open(CONTENT_FILE, 'w') as f:
            f.write(md_text)
        st.sidebar.success("Homepage updated!")

if editor_state.get('live_editor') and os.path.exists(CONTENT_FILE):
    st.markdown(open(CONTENT_FILE).read(), unsafe_allow_html=True)
else:
    st.markdown(st.session_state.homepage_md, unsafe_allow_html=True)

st.markdown("---")

theme = st.sidebar.selectbox("Theme:", ["Light", "Dark"])
bg = "#FFFFFF" if theme == "Light" else "#121212"
fg = "#000000" if theme == "Light" else "#EEEEEE"
st.markdown(f"<style>body{{background-color:{bg};color:{fg}}}</style>", unsafe_allow_html=True)

page = st.sidebar.radio("Go to:", ["Map & Trends", "Heatmap", "Comparison", "Favorites & Notes", "About"])

@st.cache_data
def load_data():
    prices = pd.read_csv("house-prices-by-small-area-sale-year.csv")
    prices.columns = [c.strip().lower().replace(' ', '_') for c in prices.columns]
    if 'type' in prices.columns and 'property_type' not in prices.columns:
        prices.rename(columns={'type': 'property_type'}, inplace=True)
    if 'latitude' not in prices.columns:
        prices['latitude'] = -37.8136
    if 'longitude' not in prices.columns:
        prices['longitude'] = 144.9631

    dwell = pd.read_csv("city-of-melbourne-dwellings-and-household-forecasts-by-small-area-2020-2040.csv")
    dwell.columns = [c.strip().lower().replace(' ', '_') for c in dwell.columns]
    if 'geography' in dwell.columns:
        dwell.rename(columns={'geography': 'small_area'}, inplace=True)
    if 'category' in dwell.columns and 'dwelling_type' not in dwell.columns:
        dwell.rename(columns={'category': 'dwelling_type'}, inplace=True)
    if 'households' in dwell.columns and 'dwelling_number' not in dwell.columns:
        dwell.rename(columns={'households': 'dwelling_number'}, inplace=True)

    return prices, dwell

prices_df, dwell_df = load_data()

def download_csv(df, fname):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f"<a href='data:file/csv;base64,{b64}' download='{fname}'>🗕️ Download {fname}</a>"

midpoint = (prices_df['latitude'].mean(), prices_df['longitude'].mean())

if page == "Map & Trends":
    st.header("📍 Interactive Map & Trends")

    viz_option = st.selectbox("Map Style:", ["Scatter", "Heat Hexagon", "Cluster Icons"])
    view = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11)
    if viz_option == "Scatter":
        layer = pdk.Layer("ScatterplotLayer", prices_df, get_position='[longitude, latitude]', get_fill_color='[0, 120, 255, 180]', get_radius=150, pickable=True)
    elif viz_option == "Heat Hexagon":
        layer = pdk.Layer("HexagonLayer", prices_df, get_position='[longitude, latitude]', radius=300, elevation_scale=50, elevation_range=[0, 3000], extruded=True, pickable=True)
    else:
        layer = pdk.Layer("ScatterplotLayer", prices_df, get_position='[longitude, latitude]', get_fill_color='[200, 30, 0, 160]', get_radius=100, pickable=True)

    tooltip_text = "{small_area}\nYear: {sale_year}\nPrice: ${median_price:,.0f}"
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": tooltip_text}))

    st.subheader("Filters & Chart Style")
    area = st.selectbox("Suburb:", sorted(prices_df['small_area'].dropna().unique()))
    ptype = st.selectbox("Property Type:", sorted(prices_df['property_type'].dropna().unique()))
    yrs = st.slider("Year Range:", int(prices_df['sale_year'].min()), int(prices_df['sale_year'].max()), (int(prices_df['sale_year'].min()), int(prices_df['sale_year'].max())))
    chart_style = st.radio("Chart Type:", ["Line", "Bar", "Area"])

    sub = prices_df[(prices_df['small_area']==area)&(prices_df['property_type']==ptype)&(prices_df['sale_year']>=yrs[0])&(prices_df['sale_year']<=yrs[1])]
    if sub.empty:
        st.warning("No data for these filters.")
    else:
        if 'dwelling_type' in dwell_df.columns and 'dwelling_number' in dwell_df.columns:
            mix = dwell_df[dwell_df['small_area']==area].groupby('dwelling_type')['dwelling_number'].sum().reset_index()
            pie = alt.Chart(mix).mark_arc().encode(
                theta=alt.Theta('dwelling_number:Q', title='Count'),
                color=alt.Color('dwelling_type:N', title='Type'),
                tooltip=['dwelling_type','dwelling_number']
            ).properties(title='Dwelling Composition')
            st.altair_chart(pie, use_container_width=False)

        st.subheader("Historical & Forecasted Prices")
        base = alt.Chart(sub).encode(
            x=alt.X('sale_year:O', title='Year', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('median_price:Q', title='Median Price (AUD)', axis=alt.Axis(format='$,'))
        )
        if chart_style == 'Line':
            hist = base.mark_line(point=True).properties(title=f"{area} {ptype} Trend")
        elif chart_style == 'Bar':
            hist = base.mark_bar().properties(title=f"{area} {ptype} Trend")
        else:
            hist = base.mark_area(opacity=0.5).properties(title=f"{area} {ptype} Trend")

        model = LinearRegression().fit(sub[['sale_year']], sub['median_price'])
        future_years = np.arange(sub['sale_year'].max()+1, sub['sale_year'].max()+1+MAX_FUTURE_YEARS)
        preds = model.predict(future_years.reshape(-1,1))
        df_f = pd.DataFrame({'sale_year':future_years, 'median_price':preds})
        forecast_line = alt.Chart(df_f).mark_line(color='orange', strokeWidth=2).encode(x='sale_year:O', y='median_price:Q')
        forecast_band = alt.Chart(df_f).mark_area(opacity=0.2, color='orange').encode(x='sale_year:O', y='median_price:Q')
        combined = (hist + forecast_line + forecast_band).interactive().properties(width=700, height=350)
        st.altair_chart(combined, use_container_width=True)

        preds_hist = model.predict(sub[['sale_year']])
        rmse = np.sqrt(mean_squared_error(sub['median_price'], preds_hist))
        st.metric("Forecast RMSE", f"${rmse:,.0f}")

elif page == "Heatmap":
    st.header("🌡️ Price Heatmap")
    m = folium.Map(location=midpoint, zoom_start=11)
    data = prices_df[['latitude','longitude','median_price']].dropna()
    data['intensity'] = (data['median_price'] - data['median_price'].min()) / (data['median_price'].max() - data['median_price'].min())
    HeatMap(data[['latitude','longitude','intensity']].values.tolist(), radius=15).add_to(m)
    st_folium(m, width=700, height=500)

elif page == "Comparison":
    st.header("🔍 Compare Two Suburbs")
    col1, col2 = st.columns(2)
    with col1:
        suburb1 = st.selectbox("Suburb 1", sorted(prices_df['small_area'].dropna().unique()), key='comp1')
    with col2:
        suburb2 = st.selectbox("Suburb 2", sorted(prices_df['small_area'].dropna().unique()), index=1, key='comp2')

    df1 = prices_df[prices_df['small_area'] == suburb1].copy()
    df2 = prices_df[prices_df['small_area'] == suburb2].copy()

    df1['Suburb'] = suburb1
    df2['Suburb'] = suburb2
    comp_df = pd.concat([df1, df2])

    chart = alt.Chart(comp_df).mark_line(point=True).encode(
        x=alt.X('sale_year:O', title='Year'),
        y=alt.Y('median_price:Q', title='Median Price'),
        color=alt.Color('Suburb:N', title='Suburb'),
        tooltip=['sale_year', 'median_price', 'Suburb']
    ).properties(
        title="Comparison of Median House Prices",
        width=800,
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

elif page == "Favorites & Notes":
    st.header("⭐ Favorites & Notes")
    for fav in st.session_state.favorites:
        st.write(fav)
        st.text_area(f"Notes for {fav}", key=f"note_{fav}")
    if st.button("Clear Favorites"):
        st.session_state.favorites = []

elif page == "About":
    st.header("ℹ️ About")
    st.write("Dashboard uses City of Melbourne open data to explore house prices.")

st.markdown("---")
st.write("*Data source: City of Melbourne.*")
