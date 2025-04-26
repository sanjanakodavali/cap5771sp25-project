import os
# 0) force CPU‐only for SHAP force_plot
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    RocCurveDisplay
)
from streamlit_lottie import st_lottie
import requests

# 1) Page config & CSS
st.set_page_config(
    page_title="💊 MedRec Dashboard",
    layout="wide",
    page_icon="💊"
)
st.markdown("""
<div style="background-color:#4B91F6;padding:15px;border-radius:10px">
  <h1 style="color:white;text-align:center;margin:0">
    💊 Medication Recommendation System
  </h1>
</div>
""", unsafe_allow_html=True)

# 2) Lottie in sidebar
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

with st.sidebar:
    lottie_med = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jbrw3hcz.json")
    if lottie_med:
        st_lottie(lottie_med, height=200)

# 3) Load model & data (cached)
@st.cache_data(show_spinner=False)
def load_artifacts():
    model = joblib.load("best_rf_model.joblib")
    X_test = pd.read_csv("X_test.csv", index_col=0)
    y_test = pd.read_csv("y_test.csv", index_col=0).values.ravel()
    explainer = shap.TreeExplainer(model)
    return model, X_test, y_test, explainer

with st.spinner("🚀 Loading model & data..."):
    model, X_test, y_test, explainer = load_artifacts()
    y_probs = model.predict_proba(X_test)[:, 1]

# 4) Sidebar controls
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Classification threshold", 0.0, 1.0, 0.5, 0.01)
show_pr = st.sidebar.checkbox("Show PR Curve", True)
show_cm = st.sidebar.checkbox("Show Confusion Matrix", True)

# 5) Navigation
page = st.sidebar.radio("Go to", [
    "Metrics", "ROC Curve", "Feature Importances",
    "SHAP: Bar", "SHAP: Beeswarm", "SHAP: Force"
])

# 6) Page: Metrics
if page == "Metrics":
    st.header("Model Performance Metrics")
    metrics = pd.read_csv("model_metrics.csv")
    st.dataframe(metrics, use_container_width=True)

# 7) Page: ROC Curve
elif page == "ROC Curve":
    st.header("ROC Curve Comparison")

    st.image(
        "roc_curve.png",
        caption="ROC Curves for Random Forest, SVM, and Logistic Regression",
        use_container_width=False,
        width=600
    )


# 8) Page: Feature Importances
elif page == "Feature Importances":
    st.header("Random Forest Feature Importances")
    feat_imp = pd.read_csv("feature_importances.csv")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig, use_container_width=True)

# 9) Page: SHAP: Mean |SHAP| per Feature
elif page == "SHAP: Bar":
    st.header("SHAP: Mean |SHAP| per Feature")

    # 1) Sample your test set
    Xs = X_test.sample(100, random_state=42)

    # 2) Compute raw SHAP output
    raw_vals = explainer.shap_values(Xs)

    # 3) Unify into a 2-D array (n_samples × n_features)
    if isinstance(raw_vals, list):
        # old API: list of arrays per class
        shap_vals = raw_vals[1]
    elif isinstance(raw_vals, np.ndarray):
        shap_vals = raw_vals
        if shap_vals.ndim > 2:
            # new API: (samples, features, classes)
            shap_vals = shap_vals[..., 1]   # pick class-1 slice
    else:
        st.error(f"Unexpected SHAP output type: {type(raw_vals)}")
        shap_vals = np.zeros((len(Xs), Xs.shape[1]))

    # 4) Verify we now have exactly 2 dimensions
    assert shap_vals.ndim == 2, f"Expected 2D array, got {shap_vals.shape}"

    # 5) Compute mean absolute SHAP per feature
    mean_abs = np.abs(shap_vals).mean(axis=0)

    # 6) Build a clean DataFrame
    shap_df = pd.DataFrame({
        "Feature": Xs.columns.tolist(),
        "Mean |SHAP|": mean_abs.tolist()
    }).sort_values("Mean |SHAP|", ascending=False)

    # 7) (Optional) inspect the numbers
    st.dataframe(shap_df, use_container_width=True)

    # 8) Plot a horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["Feature"], shap_df["Mean |SHAP|"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("SHAP Feature Importances (Class 1)")
    st.pyplot(fig, use_container_width=True)


# 10) Page: SHAP Beeswarm
elif page == "SHAP: Beeswarm":
    st.header("SHAP Global Summary (Beeswarm)")

    # Display your static image at a narrower width so the y-axis labels show
    st.image(
        "shap_summary_beeswarm.png",
        caption="SHAP Global Summary (Beeswarm)",
        use_container_width=False,
        width=500  # shrink until your feature names are fully visible
    )


# 11) Page: SHAP Force
elif page == "SHAP: Force":
    st.header("SHAP Force Plot (Sample 0)")

    # Create three columns so we can center the middle one
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.image(
            "shap_force_plot.png",                # make sure this file lives next to app.py
            caption="SHAP Force Plot (pre-computed)",
            use_container_width=True              # fill the column width
        )




# 12) Footer in sidebar
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ • Powered by Streamlit")
