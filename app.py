"""
Streamlit web app for real-time fraud prediction.

Run from the project root:
    streamlit run app.py

The app loads the trained pipeline from models/fraud_detection_pipeline.pkl
(created by notebooks/02_improved_model.ipynb or `python -m src.train`).
All feature engineering happens inside the pipeline, so the app only has to
collect the raw transaction fields.
"""

import joblib
import pandas as pd
import streamlit as st

# Importing src.features is required: the saved pipeline references
# add_features() from this module, and joblib needs it to unpickle the model.
from src.features import MODEL_PATH, TRANSACTION_TYPES

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️")


@st.cache_resource  # load the model once, not on every user interaction
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["threshold"]


model, threshold = load_model()

# --------------------------------------------------------------------------
# Page header
# --------------------------------------------------------------------------
st.title("Fraud Detection Prediction App")
st.markdown(
    "Enter the details of a mobile-money transaction to predict whether it is "
    "fraudulent. The model is an XGBoost pipeline trained on 6.3M simulated "
    "transactions (PaySim)."
)
st.divider()

# --------------------------------------------------------------------------
# User inputs (the same raw columns the model was trained on)
# --------------------------------------------------------------------------
transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
amount = st.number_input("Amount", min_value=0.0, value=100.0)

col1, col2 = st.columns(2)
with col1:
    oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
    newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9900.0)
with col2:
    oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=100.0)

# In the dataset `step` = hours since start; the model only uses step % 24 (hour of day).
hour = st.slider("Hour of Day", min_value=0, max_value=23, value=12)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "step": hour,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }])

    # Probability of fraud, compared with the threshold tuned on validation data
    fraud_probability = float(model.predict_proba(input_data)[0, 1])
    is_fraud = fraud_probability >= threshold

    st.metric("Fraud probability", f"{fraud_probability:.1%}")
    st.caption(f"Decision threshold: {threshold:.1%}")

    if is_fraud:
        st.error("⚠️ The transaction is predicted to be **fraudulent**.")
    else:
        st.success("✅ The transaction is predicted to be **legitimate**.")

    with st.expander("Example of a typical fraud pattern"):
        st.markdown(
            "- Type **TRANSFER** or **CASH_OUT**\n"
            "- Amount equals the sender's whole balance (new sender balance = 0)\n"
            "- Receiver balance does not change accordingly\n\n"
            "e.g. TRANSFER, amount 181, sender 181 → 0, receiver 0 → 0"
        )
