import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import traceback

st.set_page_config(page_title="Sales Data Analysis & Forecasting", layout="wide")
st.title("📊 Sales Data Analysis & Forecasting (ARIMA)")

try:
    # Step 1: Data Collection/Generation
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(24)]
    sales = np.random.normal(10000, 2000, 24) + np.arange(24)*200

    # Introduce null values
    sales[3] = np.nan
    sales[10] = np.nan

    # Create DataFrame
    df = pd.DataFrame({'Date': dates, 'Sales': sales})
    df.set_index('Date', inplace=True)

    # Add duplicate rows
    duplicate_rows = df.iloc[[0, 5]]
    df = pd.concat([df, duplicate_rows]).sort_index()

    # Synthetic category & region data
    categories = ['Electronics', 'Clothing', 'Books', 'Home']
    regions = ['North', 'South', 'East', 'West']
    category_sales = np.random.normal(5000, 1000, (24, 4))
    region_sales = np.random.normal(3000, 500, (24, 4))
    category_df = pd.DataFrame(category_sales, index=dates, columns=categories)
    region_df = pd.DataFrame(region_sales, index=dates, columns=regions)

    # Step 2: Data Preprocessing
    st.header("📌 Data Preprocessing")

    st.subheader("Missing Values Before:")
    st.write(df.isnull().sum())

    # Fill nulls
    df['Sales'] = df['Sales'].fillna(method='ffill')

    st.subheader("Missing Values After:")
    st.write(df.isnull().sum())

    # Remove duplicates
    st.write(f"Number of duplicate rows before: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    st.write(f"Number of rows after removing duplicates: {len(df)}")

    # Standardization
    scaler = StandardScaler()
    df['Sales_Standardized'] = scaler.fit_transform(df[['Sales']])

    st.subheader("Data after standardization:")
    st.dataframe(df.head())

    # Train-Test Split
    train = df.iloc[:-6][['Sales', 'Sales_Standardized']]
    test = df.iloc[-6:][['Sales', 'Sales_Standardized']]

    st.write(f"Train set size: {len(train)}, Test set size: {len(test)}")

    # Save preprocessed (optional, for debugging)
    try:
        df.to_csv('preprocessed_sales_data.csv')
    except Exception as save_err:
        st.warning(f"Could not save CSV file: {save_err}")

    # Step 3: Data Analysis & Visualization
    st.header("📈 Data Analysis & Visualization")

    st.subheader("Sales Summary")
    st.write(df['Sales'].describe())

    st.subheader("Last Month Category Sales")
    st.write(category_df.iloc[-1])

    st.subheader("Last Month Regional Sales")
    st.write(region_df.iloc[-1])

    # Visualization 1: Monthly Sales Trend
    st.subheader("Monthly Sales Trend")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df['Sales'], marker='o', label='Actual Sales')
    ax1.set_title('Monthly Sales Trend')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales ($)')
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    # Visualization 2: Sales by Product Category
    st.subheader("Sales by Product Category")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.stackplot(category_df.index, category_df.T, labels=categories, alpha=0.6)
    ax2.set_title('Sales by Product Category')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    st.pyplot(fig2)

    # Visualization 3: Regional Sales Share (Pie Chart)
    st.subheader("Regional Sales Share (Last Month)")
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.pie(region_df.iloc[-1], labels=regions, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Regional Sales Share (Last Month)')
    st.pyplot(fig3)

    # Step 4: Model Training
    st.header("🤖 Model Training (ARIMA)")
    model = ARIMA(train['Sales'], order=(1,1,1))
    model_fit = model.fit()
    st.text(model_fit.summary())

    # Step 5: Model Evaluation and Forecasting
    pred = model_fit.forecast(steps=6)
    mae = mean_absolute_error(test['Sales'], pred)
    rmse = np.sqrt(mean_squared_error(test['Sales'], pred))
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Forecast next 6 months
    forecast_steps = 6
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_dates = [df.index[-1] + timedelta(days=30*(i+1)) for i in range(forecast_steps)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Sales': forecast})
    st.subheader("Forecasted Sales for Next 6 Months")
    st.dataframe(forecast_df)

    # Visualization 4: Actual vs. Forecasted Sales
    st.subheader("Actual vs. Forecasted Sales")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(test.index, test['Sales'], marker='o', label='Actual Sales')
    ax4.plot(test.index, pred, marker='x', label='Forecasted Sales', linestyle='--')
    ax4.set_title('Actual vs. Forecasted Sales')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sales ($)')
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)

    # Visualization 5: Sales Distribution
    st.subheader("Sales Distribution")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Sales'], kde=True, bins=10, ax=ax5)
    ax5.set_title('Sales Distribution')
    ax5.set_xlabel('Sales ($)')
    ax5.set_ylabel('Frequency')
    ax5.grid(True)
    st.pyplot(fig5)

except Exception as e:
    st.error(f"App crashed: {e}")
    st.code(traceback.format_exc())
