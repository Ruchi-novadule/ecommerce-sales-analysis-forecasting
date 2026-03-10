import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_cleaning import load_and_clean_data
from src.forecasting_model import train_model
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("E-Commerce Sales Analysis Dashboard")

# Load dataset
df = load_and_clean_data("data/sales_data.csv")

# Data preprocessing
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["TotalSales"] = df["Quantity"] * df["Price"]
df["Month"] = df["OrderDate"].dt.month

# Dataset preview
st.subheader("Dataset Preview")
st.write(df.head())

# Sales by Region
st.subheader("Sales by Region")

region_sales = df.groupby("Region")["TotalSales"].sum()

fig1, ax1 = plt.subplots()
region_sales.plot(kind="bar", ax=ax1)

plt.xlabel("Region")
plt.ylabel("Total Sales")

st.pyplot(fig1)

# Monthly Sales Trend
st.subheader("Monthly Sales Trend")

monthly_sales = df.groupby("Month")["TotalSales"].sum()

fig2, ax2 = plt.subplots()
monthly_sales.plot(kind="line", marker="o", ax=ax2)

plt.xlabel("Month")
plt.ylabel("Total Sales")

st.pyplot(fig2)

# Sales by Category
st.subheader("Sales by Category")

category_sales = df.groupby("Category")["TotalSales"].sum()

fig3, ax3 = plt.subplots()
category_sales.plot(kind="pie", autopct='%1.1f%%', ax=ax3)

st.pyplot(fig3)

# Top Selling Products
st.subheader("Top Selling Products")

top_products = df.groupby("Product")["TotalSales"].sum().sort_values(ascending=False)

st.write(top_products.head())
st.subheader("Sales Forecast")

# Features and target
X = df[["Month","Quantity","Price"]]
y = df["TotalSales"]

# Train model
model = train_model(df)

# Future data
future_data = pd.DataFrame({
    "Month":[4,5,6],
    "Quantity":[10,12,15],
    "Price":[200,200,200]
})

# Prediction
future_predictions = model.predict(future_data)

forecast_df = pd.DataFrame({
    "Month":["April","May","June"],
    "Predicted Sales": future_predictions
})

st.write(forecast_df)
fig4, ax4 = plt.subplots()

ax4.plot(forecast_df["Month"], forecast_df["Predicted Sales"], marker="o")

ax4.set_title("Future Sales Forecast")
ax4.set_xlabel("Month")
ax4.set_ylabel("Predicted Sales")

st.pyplot(fig4)
st.subheader("Correlation Heatmap")

corr = df[["Quantity","Price","TotalSales","Month"]].corr()

fig5, ax5 = plt.subplots()

sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)

st.pyplot(fig5)
from src.forecasting_model import predict_future_sales

st.subheader("Sales Forecast")

predictions = predict_future_sales(model)

forecast_df = pd.DataFrame({
    "Month": ["April","May","June"],
    "Predicted Sales": predictions
})

st.write(forecast_df)
st.line_chart(forecast_df.set_index("Month"))
