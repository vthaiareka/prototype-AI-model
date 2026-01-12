import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"  # change when deployed

st.set_page_config(page_title="ML Model Demo", layout="centered")

st.title("Expense Category Classifier")
st.caption("Type an expense description and see how the model predicts.")

text = st.text_input("Expense description", placeholder="e.g. cab booking")
top_k = st.slider("Top K predictions", 1, 5, 3)

if text:
    with st.spinner("Predicting..."):
        response = requests.post(
            API_URL,
            params={"text": text, "top_k": top_k},
            timeout=5
        )

    if response.status_code == 200:
        data = response.json()

        for i, p in enumerate(data["predictions"], start=1):
            st.write(
                f"**{i}. {p['category']}**  "
                f"(score = {p['score']:.3f})"
            )
    else:
        st.error("Prediction failed")
