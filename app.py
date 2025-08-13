import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
st.set_page_config(layout="wide", page_title="Sales EDA & Regression")

st.title("📊 Sales Data — EDA & Linear Regression")
st.write("Upload `preprocessed_sales_data.csv` để thực hiện EDA và hồi quy tuyến tính.")

# Sidebar upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
test_size = st.sidebar.slider("Test set size (%)", 5, 50, 20)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

@st.cache_data
def load_data(buffer):
    return pd.read_csv(buffer)

def basic_overview(df):
    st.subheader("Dataset overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(df.head())

# Stop if no file
if uploaded_file is None:
    st.info("Vui lòng upload file `preprocessed_sales_data.csv`.")
    st.stop()

# Load data
df = load_data(uploaded_file)

# Convert Date
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

basic_overview(df)

# Numeric columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
st.write("Numeric columns:", num_cols)

# EDA
with st.expander("Histograms"):
    for c in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

if len(num_cols) >= 2:
    with st.expander("Correlation heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
        st.pyplot(fig)

# Model training
st.header("🤖 Model training")
features = st.multiselect("Chọn features (X)", options=num_cols, default=[c for c in num_cols if c != 'Sales'])
target = st.selectbox("Chọn target (y)", options=[c for c in num_cols if c not in features], index=0)

if st.button("Train model"):
    X = df[features]
    y = df[target]
    df_model = pd.concat([X, y], axis=1).dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df_model[features], df_model[target], test_size=test_size/100.0, random_state=int(random_state)
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"MSE: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"R²: {r2:.4f}")

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    st.pyplot(fig)

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", key=abs, ascending=False)
    st.table(coef_df)
