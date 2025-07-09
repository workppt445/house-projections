import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

# Set page configuration
st.set_page_config(page_title="Melbourne House Price Explorer", layout="wide")

# Initialize session state for editor mode and favorites
if 'editor_mode' not in st.session_state:
    st.session_state['editor_mode'] = False
if 'homepage_md' not in st.session_state:
    st.session_state['homepage_md'] = (
        "# Melbourne House Price Explorer\n"
        "Explore Melbourne property data interactively. Use the navigation menu to switch between pages."
    )
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []

# Secret code to unlock editor mode
secret_code = st.sidebar.text_input("Enter code for editor mode", type="password")
if secret_code == "7477":
    st.session_state['editor_mode'] = True

# Theme selection (Light or Dark)
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Dashboard", "Favorites"])

# Load data (assume CSVs are in the /data/ directory)
@st.cache_data
def load_data():
    df_prices = pd.read_csv("data/house-prices-by-small-area-sale-year.csv")
    # Convert year to numeric
    if 'sale_year' in df_prices.columns:
        df_prices['sale_year'] = pd.to_numeric(df_prices['sale_year'], errors='coerce')
    df_dwell = pd.read_csv("data/residential-dwellings.csv")
    if 'dwelling_number' in df_dwell.columns:
        df_dwell['dwelling_number'] = pd.to_numeric(df_dwell['dwelling_number'], errors='coerce')
    df_addr = pd.DataFrame()
    try:
        df_addr = pd.read_csv("data/street-addresses.csv")
    except FileNotFoundError:
        df_addr = pd.DataFrame()
    return df_prices, df_dwell, df_addr

df_prices, df_dwell, df_addr = load_data()

# Convert coordinate fields to numeric
for col in ['latitude', 'longitude', 'lat', 'lon']:
    if col in df_dwell.columns or col in df_addr.columns:
        try:
            if col in df_dwell.columns:
                df_dwell[col] = pd.to_numeric(df_dwell[col], errors='coerce')
            if col in df_addr.columns:
                df_addr[col] = pd.to_numeric(df_addr[col], errors='coerce')
        except Exception:
            pass

# Home Page
if page == "Home":
    st.title("Melbourne Property Dashboard")
    if st.session_state['editor_mode']:
        new_md = st.text_area("Edit Homepage Markdown", st.session_state['homepage_md'], height=250)
        if new_md != st.session_state['homepage_md']:
            st.session_state['homepage_md'] = new_md
        st.markdown(st.session_state['homepage_md'], unsafe_allow_html=True)
    else:
        st.markdown(st.session_state['homepage_md'], unsafe_allow_html=True)
        st.write("This app uses City of Melbourne open data on house prices and dwellings.")
        st.write("Navigate to the Dashboard to interact with the map and see trends.")

# Dashboard Page
elif page == "Dashboard":
    st.header("Interactive Map and Analysis")
    st.write("Click on a point (property) in the map below to select a small area for analysis.")
    # Map view option: scatter or heatmap
    view_option = st.selectbox("Map View:", ["Points (Properties)", "Heatmap"])
    # Determine map data source
    if not df_dwell.empty and 'latitude' in df_dwell.columns and 'longitude' in df_dwell.columns:
        map_data = df_dwell.rename(columns={'longitude':'lon','latitude':'lat'})
    elif not df_addr.empty and 'Longitude' in df_addr.columns and 'Latitude' in df_addr.columns:
        map_data = df_addr.rename(columns={'Longitude':'lon','Latitude':'lat'})
    else:
        map_data = pd.DataFrame(columns=['lon','lat'])
    # Initial view state
    if not map_data.empty:
        lat_center = float(map_data['lat'].mean())
        lon_center = float(map_data['lon'].mean())
    else:
        lat_center, lon_center = -37.81, 144.96
    view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=12)
    layers = []
    # Create PyDeck layers based on view_option
    if view_option == "Points (Properties)":
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position='[lon, lat]',
            get_fill_color='[0, 120, 210, 180]',
            get_radius=10,
            pickable=True,
            auto_highlight=True,
            id="scatter"
        )
        layers.append(scatter)
    else:
        heatmap = pdk.Layer(
            "HeatmapLayer",
            data=map_data,
            get_position='[lon, lat]',
            aggregation='"SUM"',
            radiusPixels=50,
            id="heatmap"
        )
        layers.append(heatmap)
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=layers
    )
    select_event = st.pydeck_chart(deck, use_container_width=True, height=450,
                                    selection_mode="single-object", on_select="rerun")
    selected_area = None
    # Handle selection from map
    if select_event:
        sel = select_event.selection
        if sel:
            if 'scatter' in sel['indices'] and sel['indices']['scatter']:
                idx = sel['indices']['scatter'][0]
                sel_obj = sel['objects']['scatter'][0]
                selected_area = sel_obj.get('clue_small_area') or sel_obj.get('small_area')
    # If an area is selected, show analysis
    if selected_area:
        st.subheader(f"Analysis for Area: {selected_area}")
        # Filter dwellings data for selected area
        area_dw = df_dwell[df_dwell['clue_small_area'] == selected_area]
        # Pie chart: Dwelling type mix:contentReference[oaicite:6]{index=6}
        if not area_dw.empty:
            mix = area_dw.groupby('dwelling_type')['dwelling_number'].sum().reset_index()
            mix.columns = ['Type', 'Count']
            pie_chart = alt.Chart(mix).mark_arc().encode(
                theta='Count:Q',
                color='Type:N',
                tooltip=['Type','Count']
            ).properties(width=300, height=300, title="Dwelling Types")
            if theme == "Dark":
                pie_chart = pie_chart.configure_view(stroke=None).configure(background='#2c2c2c', titleColor='white').configure_axis(labelColor='white', domainColor='white', titleColor='white')
            st.altair_chart(pie_chart)
        else:
            st.write("No dwelling data available for this area.")
        # Price trends line chart
        area_prices = df_prices[df_prices['small_area'] == selected_area].copy()
        if not area_prices.empty:
            # Slider to filter years
            years = pd.to_numeric(area_prices['sale_year'], errors='coerce')
            min_year = int(years.min())
            max_year = int(years.max())
            year_range = st.slider("Sale Year Range", min_year, max_year, (min_year, max_year))
            mask = (area_prices['sale_year'] >= year_range[0]) & (area_prices['sale_year'] <= year_range[1])
            filtered_prices = area_prices[mask]
            filtered_prices = filtered_prices.sort_values('sale_year')
            filtered_prices['sale_year'] = pd.Categorical(filtered_prices['sale_year'])
            line_chart = alt.Chart(filtered_prices).mark_line(point=True).encode(
                x=alt.X('sale_year:O', title='Year'),
                y=alt.Y('median_price:Q', title='Median Price (AUD)'),
                color=alt.Color('type:N', title='Type'),
                tooltip=['sale_year','type','median_price']
            ).properties(width=650, height=400, title="Median Price Trends")
            # Polynomial regression forecast for dwellings:contentReference[oaicite:7]{index=7}
            dwell_prices = filtered_prices[filtered_prices['type'].str.contains('Dwelling', case=False)]
            if dwell_prices.shape[0] >= 2:
                x = pd.to_numeric(dwell_prices['sale_year'])
                y = pd.to_numeric(dwell_prices['median_price'])
                coeffs = np.polyfit(x, y, 2)
                poly_func = np.poly1d(coeffs)
                future_years = np.arange(max(x)+1, max(x)+6)
                predictions = poly_func(future_years)
                df_pred = pd.DataFrame({'sale_year': future_years.astype(str), 'median_price': predictions})
                line_chart = line_chart + alt.Chart(df_pred).mark_line(color='green', strokeDash=[5,5]).encode(
                    x='sale_year:O', y='median_price:Q'
                )
            if theme == "Dark":
                line_chart = line_chart.configure_view(stroke=None).configure(background='#2c2c2c', titleColor='white').configure_axis(labelColor='white', domainColor='white', titleColor='white')
            st.altair_chart(line_chart)
            # Area statistics
            total_sales = filtered_prices['transaction_count'].astype(float).sum()
            avg_price = filtered_prices.groupby('type')['median_price'].median().to_dict()
            st.write("#### Area Statistics")
            st.metric("Total Sales", f"{int(total_sales)}")
            # Show average prices by type
            for t, val in avg_price.items():
                st.metric(f"Median {t} Price", f"${val:,.0f}")
        else:
            st.write("No sales data for this area.")
        # Download filtered results:contentReference[oaicite:8]{index=8}
        if not area_prices.empty:
            to_download = filtered_prices[['sale_year','type','median_price','transaction_count']]
            csv_data = to_download.to_csv(index=False).encode('utf-8')
            st.download_button("Download Filtered Data", csv_data, f"{selected_area}_data.csv", mime="text/csv")
        # Bookmark / Favorites
        if selected_area and selected_area not in st.session_state['favorites']:
            if st.button("Add to Favorites"):
                st.session_state['favorites'].append(selected_area)
                st.success(f"Added {selected_area} to favorites.")
        # Comparison mode toggle
        if st.checkbox("Comparison Mode"):
            st.subheader("Comparison Mode: Select Two Areas")
            area_list = sorted(df_prices['small_area'].unique())
            area1 = st.selectbox("Area 1", area_list, key='comp1')
            area2 = st.selectbox("Area 2", area_list, key='comp2')
            colA, colB = st.columns(2)
            with colA:
                comp_data1 = df_prices[df_prices['small_area'] == area1].copy()
                comp_data1['sale_year'] = pd.to_numeric(comp_data1['sale_year'], errors='coerce')
                comp_data1 = comp_data1.sort_values('sale_year')
                chart1 = alt.Chart(comp_data1).mark_line(point=True).encode(
                    x='sale_year:O', y='median_price:Q', color='type:N',
                    tooltip=['sale_year','type','median_price']
                ).properties(width=300, height=300, title=area1)
                if theme == "Dark":
                    chart1 = chart1.configure_view(stroke=None).configure(background='#2c2c2c', titleColor='white').configure_axis(labelColor='white', domainColor='white', titleColor='white')
                st.altair_chart(chart1)
            with colB:
                comp_data2 = df_prices[df_prices['small_area'] == area2].copy()
                comp_data2['sale_year'] = pd.to_numeric(comp_data2['sale_year'], errors='coerce')
                comp_data2 = comp_data2.sort_values('sale_year')
                chart2 = alt.Chart(comp_data2).mark_line(point=True).encode(
                    x='sale_year:O', y='median_price:Q', color='type:N',
                    tooltip=['sale_year','type','median_price']
                ).properties(width=300, height=300, title=area2)
                if theme == "Dark":
                    chart2 = chart2.configure_view(stroke=None).configure(background='#2c2c2c', titleColor='white').configure_axis(labelColor='white', domainColor='white', titleColor='white')
                st.altair_chart(chart2)

# Favorites Page
elif page == "Favorites":
    st.header("Favorite Areas")
    if st.session_state['favorites']:
        for fav in st.session_state['favorites']:
            st.write(f"- {fav}")
        if st.button("Clear All Favorites"):
            st.session_state['favorites'] = []
            st.success("Favorites cleared.")
    else:
        st.write("No favorites yet. You can add the selected area to favorites in the Dashboard.")
