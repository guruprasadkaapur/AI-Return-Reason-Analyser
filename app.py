import streamlit as st
import pandas as pd
import pickle

# ======================================
# PAGE CONFIGURATION
# ======================================

st.set_page_config(
    page_title="AI Return Analytics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# LOAD MODEL
# ======================================

@st.cache_resource
def load_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("📦 AI Dashboard")

st.sidebar.markdown("---")

st.sidebar.write("### Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
### AI Model

Model Used:
- Logistic Regression

NLP:
- TF-IDF Vectorizer

Purpose:
Automatically classify
customer return reasons.
"""
)

# ======================================
# MAIN TITLE
# ======================================

st.title("📦 AI Return Analytics Dashboard")

st.caption(
    "Analyze e-commerce return reasons using Machine Learning and NLP."
)

st.divider()

# ======================================
# IF FILE UPLOADED
# ======================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_columns = ["Product", "Return_Reason"]

    if not all(col in df.columns for col in required_columns):
        st.error("CSV must contain Product and Return_Reason columns.")
        st.stop()

    # ===============================
    # Prediction
    # ===============================

    X = vectorizer.transform(
        df["Return_Reason"].astype(str).str.lower()
    )

    predictions = model.predict(X)

    df["Predicted_Category"] = predictions

    # ===============================
    # KPIs
    # ===============================

    total_returns = len(df)

    total_products = df["Product"].nunique()

    top_category = df["Predicted_Category"].mode()[0]

    top_product = df["Product"].mode()[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📦 Total Returns",
        total_returns
    )

    col2.metric(
        "🛍 Products",
        total_products
    )

    col3.metric(
        "🏷 Top Category",
        top_category
    )

    col4.metric(
        "⭐ Top Product",
        top_product
    )

    st.divider()

    # ===============================
    # DATA PREVIEW
    # ===============================

    st.subheader("📄 Uploaded Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.divider()

    # ===============================
    # AI PREDICTIONS
    # ===============================

    st.subheader("🤖 AI Predictions")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # ===============================
    # CHARTS
    # ===============================

    chart1, chart2 = st.columns(2)

    with chart1:

        st.subheader("📊 Return Category Distribution")

        category_counts = (
            df["Predicted_Category"]
            .value_counts()
        )

        st.bar_chart(category_counts)

    with chart2:

        st.subheader("📈 Product-wise Returns")

        product_counts = (
            df["Product"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(product_counts)

    st.divider()

    # ===============================
    # SUMMARY TABLE
    # ===============================

    st.subheader("📋 Category Summary")

    summary = (
        df["Predicted_Category"]
        .value_counts()
        .reset_index()
    )

    summary.columns = [
        "Category",
        "Total Returns"
    ]

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.divider()

    # ===============================
    # DOWNLOAD
    # ===============================

    st.subheader("📥 Download Results")

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="predicted_returns.csv",
        mime="text/csv"
    )

else:

    st.info("👈 Upload a CSV file using the sidebar to begin analysis.")