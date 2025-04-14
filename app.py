import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- CONFIGURE PAGE ---
st.set_page_config(page_title="Sales Forecast & Customer Segmentation", layout="wide")
st.title("📊 Sales Forecasting & Customer Segmentation Dashboard")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📁 Upload your sales dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- BASIC PREPROCESSING ---
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['TotalPrice'] = df['Price'] * df['Quantity']

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filters")
    categories = df['Category'].unique()
    selected_categories = st.sidebar.multiselect("Filter by Category", categories, default=categories)

    min_date = df['OrderDate'].min()
    max_date = df['OrderDate'].max()
    date_range = st.sidebar.date_input("Filter by Date Range", [min_date, max_date])

    if len(date_range) == 2:
        df = df[
            (df['Category'].isin(selected_categories)) &
            (df['OrderDate'] >= pd.to_datetime(date_range[0])) &
            (df['OrderDate'] <= pd.to_datetime(date_range[1]))
        ]

    # Show data
    st.subheader("🧾 Filtered Data Preview")
    st.dataframe(df.head())

    # --- SALES FORECASTING ---
    st.header("📈 Sales Forecasting")

    monthly_sales = df.groupby(df['OrderDate'].dt.to_period('M')).sum(numeric_only=True)['TotalPrice']
    monthly_sales.index = monthly_sales.index.to_timestamp()

    sales_df = monthly_sales.reset_index()
    sales_df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(sales_df)

    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)

    st.subheader("📆 Forecasted Sales for Next 6 Months")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # --- CUSTOMER SEGMENTATION ---
    st.header("👥 Customer Segmentation (RFM + KMeans)")

    snapshot_date = df['OrderDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'OrderDate': lambda x: (snapshot_date - x.max()).days,
        'OrderID': 'count',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Apply KMeans
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(X)

    # Visualize Clusters
    st.subheader("🌀 Customer Clusters (Recency vs Monetary)")
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='Set1')
    ax2.set_xlabel("Recency")
    ax2.set_ylabel("Monetary Value")
    st.pyplot(fig2)

    st.subheader("📋 RFM + Cluster Data")
    st.dataframe(rfm)

else:
    st.info("Please upload a CSV file to begin.")
