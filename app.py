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
        prices.columns = [c.strip().lower().replace(' ', '_') for c in prices.columns]

        st.write("🏠 Loading dwellings data...")
        dwellings = pd.read_csv("city-of-melbourne-dwellings-and-household-forecasts-by-small-area-2020-2040.csv")
        dwellings.columns = [c.strip().lower().replace(' ', '_') for c in dwellings.columns]

        st.write("🔧 Converting data...")
        for col in ['sale_year', 'median_price']:
            if col in prices.columns:
                prices[col] = pd.to_numeric(prices[col], errors='coerce')
        for col in ['sale_year', 'dwelling_number']:
            if col in dwellings.columns:
                dwellings[col] = pd.to_numeric(dwellings[col], errors='coerce')

        # Drop rows missing critical data
        prices = prices.dropna(subset=['sale_year', 'median_price'])

        st.write("✅ Done loading data.")
        return prices, dwellings

    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# >>> Add the actual loading call here <<<
prices_df, dwellings_df = load_data()
prices_df, dwellings_df = load_data()

# ================ Helper Functions ================
def filter_data(area, ptype, y_min, y_max):
    df = prices_df.copy()
    return df[(df['small_area']==area)
              & (df['property_type']==ptype)
              & (df['sale_year']>=y_min)
              & (df['sale_year']<=y_max)]


def generate_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"<a href='data:file/csv;base64,{b64}' download='{filename}'>📥 Download {filename}</a>"


def fit_poly_model(df_sub):
    X = df_sub['sale_year'].values.reshape(-1,1)
    y = df_sub['median_price'].values
    poly = PolynomialFeatures(degree=2)
    Xp = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, y)
    return model, poly


def project_prices(model, poly, last_year, n_years=MAX_FUTURE_YEARS):
    years = np.arange(last_year+1, last_year+n_years+1)
    pred = model.predict(poly.transform(years.reshape(-1,1)))
    return pd.DataFrame({'sale_year': years, 'median_price': pred})

# ================= Pages =================

# ---- Map & Trends ----
if page == "Map & Trends":
    st.header("📍 Interactive Map & Trends")
    if prices_df.empty:
        st.error("No price data loaded.")
    else:
        # Debug: show columns
        st.write("Available columns in price data:", list(prices_df.columns))

        midpoint = None
        if 'latitude' in prices_df.columns and 'longitude' in prices_df.columns:
            midpoint = (prices_df['latitude'].mean(), prices_df['longitude'].mean())
        else:
            midpoint = (-37.8136, 144.9631)
            st.warning("Map data is missing latitude/longitude columns; map disabled.")

        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            # Determine area field
            if 'small_area' in prices_df.columns:
                area_field = 'small_area'
            elif 'area' in prices_df.columns:
                area_field = 'area'
            elif 'suburb' in prices_df.columns:
                area_field = 'suburb'
            else:
                area_field = prices_df.columns[0]
                st.warning(f"Using '{area_field}' as the area field.")
            areas = sorted(prices_df[area_field].dropna().unique())
            area = st.selectbox("Suburb:", areas)
        with col2:
            # Determine property type field
            if 'property_type' in prices_df.columns:
                ptype_field = 'property_type'
            elif 'type' in prices_df.columns:
                ptype_field = 'type'
            else:
                ptype_field = prices_df.columns[1]
                st.warning(f"Using '{ptype_field}' as the property type field.")
            ptypes = sorted(prices_df[ptype_field].dropna().unique())
            ptype = st.selectbox("Property Type:", ptypes)
        with col3:
            # Determine year field
            if 'sale_year' in prices_df.columns:
                year_field = 'sale_year'
            elif 'year' in prices_df.columns:
                year_field = 'year'
            else:
                year_field = prices_df.columns[2]
                st.warning(f"Using '{year_field}' as the sale year field.")
            yrs = st.slider("Sale Year Range:", int(prices_df[year_field].min()), int(prices_df[year_field].max()),
                            (int(prices_df[year_field].min()), int(prices_df[year_field].max())))

        # Filter data accordingly
        sub = prices_df[
            (prices_df[area_field] == area) &
            (prices_df[ptype_field] == ptype) &
            (prices_df[year_field] >= yrs[0]) &
            (prices_df[year_field] <= yrs[1])
        ]
        if sub.empty:
            st.warning("No data for these filters.")
        else:
            # Debug: show dwelling columns
            st.write("Dwelling data columns:", list(dwellings_df.columns))
            # Determine dwell field
            if area_field in dwellings_df.columns:
                dwell_field = area_field
            else:
                dwell_field = 'small_area' if 'small_area' in dwellings_df.columns else dwellings_df.columns[0]
                st.warning(f"Using '{dwell_field}' as the dwellings area field.")
            dwell = dwellings_df[dwellings_df[dwell_field] == area]
            if not dwell.empty and 'dwelling_type' in dwell.columns:
                mix = dwell.groupby('dwelling_type')['dwelling_number'].sum().reset_index()
                mix.columns = ['Type','Count']
                pie = alt.Chart(mix).mark_arc().encode(
                    theta='Count:Q', color='Type:N', tooltip=['Type','Count']
                ).properties(title="Dwelling Type Mix")
                st.altair_chart(pie, use_container_width=False)

            # Historical trend
            st.subheader("Historical Median Price")
            hist = px.line(sub, x=year_field, y='median_price', color=ptype_field, markers=True,
                           title=f"{area} - {ptype} Price Trend")
            st.plotly_chart(hist)

            # Forecast etc...
            model, poly = fit_poly_model(sub)
            future_df = project_prices(model, poly, sub[year_field].max())
            st.subheader(f"{MAX_FUTURE_YEARS}-Year Forecast")
            fc = px.line(future_df, x='sale_year', y='median_price', markers=True,
                         title="Projected Median Prices")
            st.plotly_chart(fc)

            preds = model.predict(poly.transform(sub[year_field].values.reshape(-1,1)))
            rmse = np.sqrt(mean_squared_error(sub['median_price'], preds))
            st.metric("Forecast RMSE", f"${rmse:,.2f}")

            st.markdown(generate_download_link(sub, f"{area}_{ptype}.csv"), unsafe_allow_html=True)

            if area not in st.session_state.favorites:
                if st.button("➕ Add to Favorites"):
                    st.session_state.favorites.append(area)
                    st.success(f"{area} bookmarked.")

# ---- Heatmap ----
elif page == "Heatmap":
    st.header("🌡️ Price Heatmap")
    if prices_df.empty:
        st.warning("No data to display heatmap.")
    else:
        # Calculate midpoint for map centering
        if 'latitude' in prices_df.columns and 'longitude' in prices_df.columns:
            midpoint = (prices_df['latitude'].mean(), prices_df['longitude'].mean())
        else:
            midpoint = (-37.8136, 144.9631)  # Default to Melbourne CBD

        m = folium.Map(location=[midpoint[0], midpoint[1]], zoom_start=11)
        heat_data = prices_df[['latitude','longitude','median_price']].dropna()
        # Normalize for intensity
        heat_data['intensity'] = (
            heat_data['median_price'] - heat_data['median_price'].min()
        ) / (
            heat_data['median_price'].max() - heat_data['median_price'].min()
        )
        points = heat_data[['latitude','longitude','intensity']].values.tolist()
        HeatMap(points, radius=15).add_to(m)
        st_folium(m, width=700, height=500)

# ---- Comparison ----
elif page == "Comparison":
    st.header("🔍 Compare Two Suburbs")
    areas = sorted(prices_df['small_area'].unique())
    colA, colB = st.columns(2)
    with colA:
        area1 = st.selectbox("Suburb 1:", areas, key='comp1')
    with colB:
        area2 = st.selectbox("Suburb 2:", areas, key='comp2')

    df1 = prices_df[prices_df['small_area']==area1]
    df2 = prices_df[prices_df['small_area']==area2]
    if df1.empty or df2.empty:
        st.warning("Select valid suburbs with data.")
    else:
        df1['area'] = area1
        df2['area'] = area2
        comp = pd.concat([df1, df2])
        fig = px.line(comp, x='sale_year', y='median_price', color='area', markers=True,
                      title=f"Comparison: {area1} vs {area2}")
        st.plotly_chart(fig)

# ---- Favorites & Notes ----
elif page == "Favorites & Notes":
    st.header("⭐ Favorites & 📝 Notes")
    if not st.session_state.favorites:
        st.info("No favorites yet. Bookmark from Map & Trends.")
    else:
        for fav in st.session_state.favorites:
            st.subheader(fav)
            note_key = f"note_{fav}"
            note = st.text_area("Your note:", key=note_key)
    if st.button("Clear Favorites"):
        st.session_state.favorites = []
        st.success("Cleared favorites.")

# ---- About ----
elif page == "About":
    st.header("ℹ️ About this Dashboard")
    st.markdown(
        """
        This interactive dashboard uses open data from the City of Melbourne to visualize and
        forecast house prices by suburb. Features include:
        - Interactive Map & Trends
        - Heatmap View
        - Suburb Comparison
        - Favorites & Notes
        - Editable homepage via secret code `7477`

        Data sources: City of Melbourne Open Data Portal.
        Built with Streamlit, PyDeck, Folium, Plotly, Altair, and Scikit-Learn.
        """
    )

# ---- Footer ----
st.markdown("---")
st.write("*Data source: City of Melbourne Open Data Portal.*")

# =========== End of App ===========
