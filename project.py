import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Page Config
st.set_page_config(page_title="Sales & Revenue Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('sales_data.csv')

df = load_data()

# 2. Scikit-Learn Regression Logic
def get_regression_data(series, steps):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression().fit(X, y)
    
    # Historical Trend Line
    trend_h = model.predict(X)
    
    # Future Forecast Values
    future_X = np.arange(len(series), len(series) + steps).reshape(-1, 1)
    forecast_v = model.predict(future_X)
    
    return trend_h, forecast_v

# Sidebar Settings
forecast_months = st.sidebar.slider("Forecast Months", 1, 6, 3)
s_trend, s_fore = get_regression_data(df['Sales'], forecast_months)
r_trend, r_fore = get_regression_data(df['Revenue'], forecast_months)

# --- X-Axis Label Logic ---
month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
last_month_name = str(df['Month'].iloc[-1])
try:
    start_idx = (month_list.index(last_month_name[:3].title()) + 1) % 12
except:
    start_idx = 0

future_labels = [month_list[(start_idx + i) % 12] for i in range(forecast_months)]
all_labels = list(df['Month']) + future_labels

# --- Main Dashboard ---
st.title("📊 Sales & Revenue Forecasting Dashboard")

st.markdown("""
This project provides a **data-driven visualization** of business performance. 
It analyzes historical trends using **Linear Regression** and predicts future growth.
""")

st.divider()

# --- NEWLY RESTORED METRICS COLUMNS ---
c1, c2, c3 = st.columns(3)
c1.metric("Total Sales", f"PKR {df['Sales'].sum():,}")
c2.metric("Total Revenue", f"PKR {df['Revenue'].sum():,}")
c3.metric("Avg Monthly Profit", f"PKR {df['Revenue'].mean():,.0f}")

st.divider()

# --- Charts Grid ---
col1, col2 = st.columns(2)

hist_indices = np.arange(len(df))
fore_indices = np.arange(len(df), len(df) + forecast_months)
connect_indices = np.append(hist_indices[-1], fore_indices)

with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(hist_indices, df['Sales'], label='Actual Sales', marker='o', color='#1f77b4', linewidth=2)
    ax1.plot(hist_indices, s_trend, linestyle=':', color='gray', label='Trend Line', alpha=0.8)
    connect_s_vals = np.append(df['Sales'].iloc[-1], s_fore)
    ax1.plot(connect_indices, connect_s_vals, label='Forecast', linestyle='--', marker='s', color='#ff7f0e')
    
    ax1.set_xticks(np.arange(len(all_labels)))
    ax1.set_xticklabels(all_labels, rotation=45)
    ax1.set_title("Sales Projection & Trend", fontsize=12, fontweight='bold')
    ax1.set_ylabel("PKR")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(hist_indices, df['Revenue'], label='Actual Revenue', marker='o', color='#2ca02c', linewidth=2)
    ax2.plot(hist_indices, r_trend, linestyle=':', color='gray', label='Trend Line', alpha=0.8)
    connect_r_vals = np.append(df['Revenue'].iloc[-1], r_fore)
    ax2.plot(connect_indices, connect_r_vals, label='Forecast', linestyle='--', marker='s', color='#d62728')
    
    ax2.set_xticks(np.arange(len(all_labels)))
    ax2.set_xticklabels(all_labels, rotation=45)
    ax2.set_title("Revenue Projection & Trend", fontsize=12, fontweight='bold')
    ax2.set_ylabel("PKR")
    ax2.legend()
    st.pyplot(fig2)

# Row 2 (Bar Charts)
row2_a, row2_b = st.columns(2)
with row2_a:
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(hist_indices - 0.2, df['Sales'], 0.4, label='Sales', color='#1f77b4')
    ax3.bar(hist_indices + 0.2, df['Expense'], 0.4, label='Expense', color='#aec7e8')
    ax3.set_xticks(hist_indices)
    ax3.set_xticklabels(df['Month'], rotation=45)
    ax3.set_title("Monthly Sales vs Expenses", fontsize=12, fontweight='bold')
    ax3.legend()
    st.pyplot(fig3)

with row2_b:
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    margin = (df['Revenue'] / df['Sales']) * 100
    ax4.bar(df['Month'], margin, color='#7f7f7f')
    ax4.set_title("Monthly Profit Margin (%)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("%")
    plt.xticks(rotation=45)
    st.pyplot(fig4)