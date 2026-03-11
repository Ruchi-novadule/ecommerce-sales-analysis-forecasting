import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_cleaning import load_and_clean_data
from src.forecasting_model import train_model, predict_future_sales

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("E-Commerce Sales Analysis Dashboard")

# Load dataset
df = load_and_clean_data("data/sales_data.csv")

# Data preprocessing
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["TotalSales"] = df["Quantity"] * df["Price"]
df["Month"] = df["OrderDate"].dt.month

# Sidebar dataset preview
st.sidebar.header("Dataset Preview")
st.sidebar.write(df.head())

st.subheader("Sales Dashboard")

# ================= FIRST ROW =================

col1, col2 = st.columns(2)

with col1:
    st.write("### Sales by Region")
    region_sales = df.groupby("Region")["TotalSales"].sum()

    fig1, ax1 = plt.subplots()
    region_sales.plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Total Sales")

    st.pyplot(fig1)

with col2:
    st.write("### Monthly Sales Trend")
    monthly_sales = df.groupby("Month")["TotalSales"].sum()

    fig2, ax2 = plt.subplots()
    monthly_sales.plot(kind="line", marker="o", ax=ax2)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Total Sales")

    st.pyplot(fig2)

# ================= SECOND ROW =================

col3, col4 = st.columns(2)

with col3:
    st.write("### Sales by Category")

    category_sales = df.groupby("Category")["TotalSales"].sum()

    fig3, ax3 = plt.subplots()
    category_sales.plot(kind="pie", autopct='%1.1f%%', ax=ax3)

    st.pyplot(fig3)

with col4:
    st.write("### Correlation Heatmap")

    corr = df[["Quantity","Price","TotalSales","Month"]].corr()

    fig5, ax5 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)

    st.pyplot(fig5)

# ================= THIRD ROW =================

col5, col6 = st.columns(2)

with col5:
    st.write("### Top Selling Products")

    top_products = df.groupby("Product")["TotalSales"].sum().sort_values(ascending=False).head(5)

    fig6, ax6 = plt.subplots()
    top_products.plot(kind="bar", ax=ax6)

    ax6.set_xlabel("Product")
    ax6.set_ylabel("Total Sales")

    st.pyplot(fig6)

with col6:
    st.write("### Sales Forecast")

    model = train_model(df)

    predictions = predict_future_sales(model)

    forecast_df = pd.DataFrame({
        "Month": ["April","May","June"],
        "Predicted Sales": predictions
    })

    fig7, ax7 = plt.subplots()

    ax7.plot(forecast_df["Month"], forecast_df["Predicted Sales"], marker="o")

    ax7.set_xlabel("Month")
    ax7.set_ylabel("Predicted Sales")

    st.pyplot(fig7)