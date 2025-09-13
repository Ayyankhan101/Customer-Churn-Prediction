# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(
    page_title="Customer Churn Insight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 5% 5% 5% 10%;
        box-shadow: 0 2px 4px rgba(0,0,0,.05);
    }
    section[data-testid="stSidebar"] {background-color: #f0f2f6;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("churn_model.pkl")
    except FileNotFoundError:
        st.error("Model artefact `churn_model.pkl` not found. Train it first.")
        st.stop()

@st.cache_data(show_spinner=False)
def load_sample_data():
    n = 1_000
    df = pd.DataFrame({
        "CustomerID": [f"C{str(i).zfill(4)}" for i in range(1, n+1)],
        "Age": np.random.randint(18, 75, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Tenure": np.random.randint(1, 60, n),
        "Usage Frequency": np.random.randint(1, 30, n),
        "Support Calls": np.random.randint(0, 10, n),
        "Payment Delay": np.random.randint(0, 30, n),
        "Subscription Type": np.random.choice(["Basic", "Standard", "Premium"], n),
        "Contract Length": np.random.choice(["Monthly", "Quarterly", "Annual"], n),
        "Total Spend": np.random.randint(100, 2500, n),
    })
    # synthetic date for trend
    df["Join Month"] = pd.date_range(start="2022-01-01", periods=n, freq="D")[:n].to_series().dt.to_period("M").astype(str)
    return df

def preprocess(df, model):
    df = df.drop("CustomerID", axis=1, errors="ignore")
    df = pd.get_dummies(
        df,
        columns=["Gender", "Subscription Type", "Contract Length"],
        drop_first=True,
    )
    training_cols = model.feature_names_in_
    for c in training_cols:
        if c not in df.columns:
            df[c] = 0
    return df[training_cols]

# --------------------------------------------------
# Arfefacts
# --------------------------------------------------
model = load_model()
raw_df = load_sample_data()
X = preprocess(raw_df, model)
proba = model.predict_proba(X)[:, 1]
raw_df["Churn_Prob"] = np.round(proba, 3)
raw_df["Churn_Pred"] = (proba >= 0.5).astype(int)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    top_n = st.slider("Top N high-risk customers", 10, 500, 100, 10)
    gender_filter = st.multiselect("Gender", ["Male", "Female"], ["Male", "Female"])
    sub_filter = st.multiselect("Subscription", ["Basic", "Standard", "Premium"], ["Basic", "Standard", "Premium"])
    contract_filter = st.multiselect("Contract", ["Monthly", "Quarterly", "Annual"], ["Monthly", "Quarterly", "Annual"])

filtered = raw_df[
    (raw_df["Gender"].isin(gender_filter))
    & (raw_df["Subscription Type"].isin(sub_filter))
    & (raw_df["Contract Length"].isin(contract_filter))
]

# --------------------------------------------------
# KPI
# --------------------------------------------------
st.title("📊 Customer Churn Insight")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Customers", f"{len(filtered):,}")
kpi2.metric("Predicted Churn", f"{filtered['Churn_Pred'].sum():,}")
kpi3.metric("Churn Rate", f"{(filtered['Churn_Pred'].mean() * 100):.1f} %")
kpi4.metric("Avg Spend", f"${filtered['Total Spend'].mean():.0f}")

# --------------------------------------------------
# Graph grid
# --------------------------------------------------

st.markdown("---")
st.subheader("🔍 Exploratory Analytics")

# --- define columns once ---
a1, a2 = st.columns(2)
b1, b2 = st.columns(2)
c1, c2 = st.columns(2)


# Row 1
a1, a2 = st.columns(2)
with a1:
    st.markdown("**Churn Probability Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(filtered["Churn_Prob"], bins=20, kde=True, color="#ff6b6b", ax=ax)
    ax.set_xlabel("Churn Probability")
    st.pyplot(fig)

# 1. Gender vs Churn (countplot)
with a2:
    st.markdown("**Gender vs Churn**")
    fig, ax = plt.subplots()
    sns.countplot(
        data=filtered.assign(Churn_Pred=filtered["Churn_Pred"].astype(str)),
        x="Gender",
        hue="Churn_Pred",
        palette={"0": "#54a24b", "1": "#ff6b6b"},
        ax=ax,
    )
    ax.set_ylabel("Count")
    st.pyplot(fig)

# 2. Total Spend vs Churn (boxplot)
with b1:
    st.markdown("**Total Spend vs Churn (Box)**")
    fig, ax = plt.subplots()
    sns.boxplot(
        data=filtered.assign(Churn_Pred=filtered["Churn_Pred"].astype(str)),
        x="Churn_Pred",
        y="Total Spend",
        palette={"0": "#54a24b", "1": "#ff6b6b"},
        ax=ax,
    )
    ax.set_xlabel("Churn Prediction")
    st.pyplot(fig)

# 3. Tenure vs Churn Probability (scatterplot hue)
with b2:
    st.markdown("**Tenure vs Churn Probability (Scatter)**")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=filtered.assign(Churn_Pred=filtered["Churn_Pred"].astype(str)),
        x="Tenure",
        y="Churn_Prob",
        hue="Churn_Pred",
        palette={"0": "#54a24b", "1": "#ff6b6b"},
        alpha=0.7,
        ax=ax,
    )
    st.pyplot(fig)

# Row 3
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Correlation Heat-map (Numerics)**")
    num_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Churn_Prob"]
    corr = filtered[num_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with c2:
    st.markdown("**Feature Importance (Tree-based)**")
    importances = model.feature_importances_
    feat_names = model.feature_names_in_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=fi_df.head(10), ax=ax, palette="viridis")
    ax.set_xlabel("Mean Decrease in Impurity")
    st.pyplot(fig)

# --------------------------------------------------
# Detailed table
# --------------------------------------------------
st.subheader(f"Top {top_n} High-Risk Customers")
top_churn = (
    filtered[filtered["Churn_Pred"] == 1]
    .nlargest(top_n, "Churn_Prob")
    .drop(columns=["Churn_Pred"])
)
st.dataframe(
    top_churn.style.background_gradient(cmap="Reds", subset=["Churn_Prob"]),
    use_container_width=True,
)

# --------------------------------------------------
# Download
# --------------------------------------------------
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

csv = convert_df(filtered)
st.download_button(
    label="⬇️ Download scored dataset",
    data=csv,
    file_name="churn_scores.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:0.9em;color:grey;">'
    "Built with Streamlit 🎈 | Model trained in Jupyter</div>",
    unsafe_allow_html=True,
)