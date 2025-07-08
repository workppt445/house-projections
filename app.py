import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Melbourne House Price Dashboard", layout="wide")

st.title("🏘️ City of Melbourne House Price Trends")

# Upload your downloaded CSV file from https://data.melbourne.vic.gov.au
uploaded_file = st.file_uploader("📤 Upload 'Median House Prices by Type and Sale Year' CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.subheader("📊 Explore Trends")
    st.markdown("Use the filters below to explore housing price trends across property types.")

    # Dropdowns for filtering
    property_types = df["property_type"].unique()
    selected_type = st.selectbox("🏠 Property Type", sorted(property_types))

    filtered_df = df[df["property_type"] == selected_type]

    # Create chart
    fig = px.line(
        filtered_df,
        x="year_of_sale",
        y="median_price",
        color="property_type",
        markers=True,
        title=f"{selected_type} Median Price Over Time"
    )

    fig.update_layout(yaxis_title="Median Price ($)", xaxis_title="Year")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📂 Raw Data Preview")
    st.dataframe(filtered_df)

else:
    st.warning("👆 Please upload the housing price CSV file to begin.")
