# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Thiết lập giao diện
sns.set_theme(style="whitegrid")
st.set_page_config(layout="wide", page_title="Video Games Sales — EDA & Regression", initial_sidebar_state="auto")

st.title("🎮 Video Games Sales — EDA & Linear Regression Demo")
st.write("Ứng dụng demo: upload `video games sales.csv` → tiền xử lý → EDA → huấn luyện LinearRegression → đánh giá.")

# -----------------------
# Sidebar: upload & options
# -----------------------
st.sidebar.header("1. Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload file CSV (video games sales.csv)", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample from repo (if available)", value=False)
test_size = st.sidebar.slider("Test set size (%)", 5, 50, 20)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

# -----------------------
# Helper functions
# -----------------------
@st.cache_data
def load_data_from_buffer(buffer):
    return pd.read_csv(buffer)

def basic_overview(df):
    st.subheader("Dataset overview")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    st.write("**Columns:**", list(df.columns))
    st.dataframe(df.head(5))

def convert_numeric_columns(df):
    # columns that likely represent numeric values
    cols_candidates = ['Rank', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    for col in cols_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def preprocess(df, drop_duplicates=True, dropna_subset=None):
    if drop_duplicates:
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        st.write(f"- Đã xóa duplicate: {before-after} dòng")
    if dropna_subset:
        before = df.shape[0]
        df = df.dropna(subset=dropna_subset)
        after = df.shape[0]
        st.write(f"- Đã dropna cho {dropna_subset}: {before-after} dòng")
    return df

def plot_hist(df, col):
    fig, ax = plt.subplots(figsize=(6,3.5))
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f'Histogram: {col}')
    st.pyplot(fig)

def plot_box(df, col):
    fig, ax = plt.subplots(figsize=(6,3))
    sns.boxplot(x=df[col].dropna(), ax=ax)
    ax.set_title(f'Boxplot: {col}')
    st.pyplot(fig)

def plot_count(df, col):
    fig, ax = plt.subplots(figsize=(8,3))
    order = df[col].value_counts().index[:30]
    sns.countplot(y=col, data=df, order=order, ax=ax)
    ax.set_title(f'Countplot: {col} (top categories)')
    st.pyplot(fig)

def plot_corr_heatmap(df, numeric_cols):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation matrix")
    st.pyplot(fig)

# -----------------------
# Main flow
# -----------------------
if uploaded_file is None and not use_sample:
    st.info("Upload `video games sales.csv` trên sidebar, hoặc bật 'Use sample' nếu repo có sample.")
    st.stop()

# Load data
if uploaded_file is not None:
    df = load_data_from_buffer(uploaded_file)
else:
    try:
        df = pd.read_csv("video games sales.csv")
        st.success("Loaded sample video games sales.csv from repo root.")
    except FileNotFoundError:
        st.error("No local sample found. Please upload a file.")
        st.stop()

# Basic overview
basic_overview(df)

st.markdown("---")
st.header("🔧 Step: Preprocessing")

# Convert numeric-like strings to numbers
df = convert_numeric_columns(df)
st.write("Kiểu dữ liệu sau khi convert:")
st.write(df.dtypes)

# Show missing values
st.write("Số giá trị missing theo cột:")
st.write(df.isnull().sum())

# Let user choose columns to require non-null before modeling
possible_numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
st.write("Các cột số hiện có:", possible_numeric_cols)

required_cols = st.multiselect("Các cột bắt buộc để huấn luyện (drop rows missing these):",
                               options=possible_numeric_cols,
                               default=[c for c in ['Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'] if c in possible_numeric_cols])

if st.button("Run preprocessing"):
    df = preprocess(df, drop_duplicates=True, dropna_subset=required_cols)
    st.success("Preprocessing hoàn tất.")
    st.write(df[required_cols].describe())

st.markdown("---")
st.header("🔎 Step: EDA")

# EDA suggestions
st.subheader("Gợi ý phân tích")
st.markdown("""
1. Phân phối doanh số ở từng khu vực và toàn cầu (NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales)  
2. Phân phối số lượng game theo Year  
3. Mối quan hệ giữa các biến doanh số (scatter, heatmap)  
4. Tương quan giữa các biến số (heatmap)  
5. Phân loại theo Platform, Genre, Publisher  
6. Kiểm tra outlier (boxplot)
""")

# Show interactive plots for numeric columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
st.write("Numerical columns detected:", num_cols)

with st.expander("Histograms"):
    for c in num_cols:
        plot_hist(df, c)

with st.expander("Boxplots"):
    for c in num_cols:
        plot_box(df, c)

# Category count plot
cat_cols = [c for c in df.columns if df[c].dtype == "object"]
if cat_cols:
    with st.expander("Top categorical counts"):
        sel_cat = st.selectbox("Chọn cột phân loại để plot count", options=cat_cols)
        plot_count(df, sel_cat)

# Correlation heatmap
if len(num_cols) >= 2:
    plot_corr_heatmap(df, num_cols)

st.markdown("---")
st.header("🤖 Step: Model training - Linear Regression")

# UI chọn features và target
features = st.multiselect(
    "Chọn features (X)",
    options=num_cols,
    default=[c for c in ['Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales'] if c in num_cols]
)
target = st.selectbox(
    "Chọn target (y)",
    options=[c for c in num_cols if c not in features],
    index=0 if 'Global_Sales' in num_cols else 0
)

if st.button("Train model"):
    try:
        if not features:
            st.error("❌ Bạn cần chọn ít nhất 1 feature.")
            st.stop()
        if target in features:
            st.error("❌ Target không được trùng với features.")
            st.stop()

        # Chuẩn bị dữ liệu
        X = df[features].copy()
        y = df[target].copy()

        df_model = pd.concat([X, y], axis=1).dropna()
        if df_model.shape[0] < 5:
            st.error("❌ Dữ liệu sau khi loại NaN quá ít, không đủ để train/test.")
            st.stop()

        X = df_model[features]
        y = df_model[target]

        # Chia tập
        test_pct = test_size / 100.0
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct, random_state=int(random_state)
        )

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Huấn luyện mô hình
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Dự đoán
        y_pred = model.predict(X_test_scaled)

        # Đánh giá
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.markdown("### 🔍 Evaluation")
        st.write(f"MSE: `{mse:.4f}`")
        st.write(f"RMSE: `{rmse:.4f}`")
        st.write(f"R²: `{r2:.4f}`")

        # Vẽ biểu đồ so sánh
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Trọng số features
        coef_df = pd.DataFrame({
            "feature": features,
            "coefficient": model.coef_
        }).sort_values(by="coefficient", key=abs, ascending=False)
        st.subheader("📌 Feature coefficients")
        st.table(coef_df)

    except Exception as e:
        st.error(f"❌ Lỗi khi huấn luyện mô hình: {str(e)}")
