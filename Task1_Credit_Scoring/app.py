
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"

st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("💳 Credit Scoring System")
st.write(
    "Estimate the probability of next-month credit-card default "
    "using historical financial and repayment information."
)

st.warning(
    "Educational demonstration only. This model is not validated "
    "for real lending decisions and may produce inaccurate predictions."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run python src/train_model.py first.")
    st.stop()

model = load_model()

st.subheader("Enter customer information")
st.caption(
    "Enter values in the units and coding used by the training dataset. "
    "The model uses historical Taiwanese credit-card client data."
)

with st.form("credit_scoring_form"):
    st.markdown("### Customer profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input(
            "Credit limit (LIMIT_BAL)",
            min_value=10000,
            max_value=1000000,
            value=200000,
            step=10000,
        )
        sex = st.selectbox("Sex code (SEX)", [1, 2])

    with col2:
        education = st.selectbox(
            "Education code (EDUCATION)",
            [1, 2, 3, 4, 5, 6, 0],
            index=0,
            help="Uses the dataset's original numeric coding.",
        )
        marriage = st.selectbox("Marriage code (MARRIAGE)", [1, 2, 3, 0])

    with col3:
        age = st.number_input("Age", 18, 100, 30)

    st.markdown("### Repayment status")

    st.caption(
        "Use the dataset's PAY codes for each historical month. "
        "These are encoded categories, not ordinary currency amounts."
    )

    pay_cols = st.columns(3)
    pay_values = []

    for i, month in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
        with pay_cols[i % 3]:
            pay_values.append(
                st.number_input(
                    f"{month} code",
                    min_value=-2,
                    max_value=8,
                    value=0,
                    key=month,
                )
            )

    st.markdown("### Bill amounts and previous payments")

    bill_cols = st.columns(3)
    bill_values = []

    for i in range(1, 7):
        with bill_cols[(i - 1) % 3]:
            bill_values.append(
                st.number_input(
                    f"BILL_AMT{i}",
                    min_value=-1000000,
                    max_value=10000000,
                    value=5000,
                    step=1000,
                    key=f"BILL_AMT{i}",
                )
            )

    payment_cols = st.columns(3)
    payment_values = []

    for i in range(1, 7):
        with payment_cols[(i - 1) % 3]:
            payment_values.append(
                st.number_input(
                    f"PAY_AMT{i}",
                    min_value=0,
                    max_value=10000000,
                    value=1000,
                    step=500,
                    key=f"PAY_AMT{i}",
                )
            )

    submitted = st.form_submit_button(
        "Assess Credit Default Risk",
        type="primary",
        use_container_width=True,
    )


if submitted:
    input_data = {
        "LIMIT_BAL": [limit_bal],
        "SEX": [sex],
        "EDUCATION": [education],
        "MARRIAGE": [marriage],
        "AGE": [age],
        "PAY_0": [pay_values[0]],
        "PAY_2": [pay_values[1]],
        "PAY_3": [pay_values[2]],
        "PAY_4": [pay_values[3]],
        "PAY_5": [pay_values[4]],
        "PAY_6": [pay_values[5]],
        "BILL_AMT1": [bill_values[0]],
        "BILL_AMT2": [bill_values[1]],
        "BILL_AMT3": [bill_values[2]],
        "BILL_AMT4": [bill_values[3]],
        "BILL_AMT5": [bill_values[4]],
        "BILL_AMT6": [bill_values[5]],
        "PAY_AMT1": [payment_values[0]],
        "PAY_AMT2": [payment_values[1]],
        "PAY_AMT3": [payment_values[2]],
        "PAY_AMT4": [payment_values[3]],
        "PAY_AMT5": [payment_values[4]],
        "PAY_AMT6": [payment_values[5]],
    }

    features = pd.DataFrame(input_data)

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    st.divider()
    st.subheader("Prediction result")

    result_col, probability_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.error("Predicted class: Higher default risk")
        else:
            st.success("Predicted class: Lower default risk")

    with probability_col:
        st.metric("Estimated default probability", f"{probability:.1%}")

    st.progress(probability)
    st.caption(
        "The probability is a model estimate, not a guarantee. "
        "No real-world lending decision should be made from this demo."
    )
