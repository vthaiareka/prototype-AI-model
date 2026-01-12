import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="ML Model Demo", layout="centered")
st.title("Expense Category Classifier")
st.caption("Predictions update as you type")

# Session state
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "last_call_time" not in st.session_state:
    st.session_state.last_call_time = 0

# Inputs
text = st.text_input(
    "Expense description",
    placeholder="e.g. cab booking"
)

top_k = st.slider("Top K predictions", 1, 5, 3)

# --- Debounce settings (important) ---
DEBOUNCE_SECONDS = 0.4
now = time.time()

# Trigger prediction on text change
if text and text != st.session_state.last_text:
    if now - st.session_state.last_call_time > DEBOUNCE_SECONDS:
        st.session_state.last_text = text
        st.session_state.last_call_time = now

        try:
            response = requests.post(
                API_URL,
                params={"text": text, "top_k": top_k},
                timeout=5
            )
            if response.status_code == 200:
                st.session_state.predictions = response.json()
            else:
                st.session_state.predictions = None
        except Exception:
            st.session_state.predictions = None

# Display results
if st.session_state.predictions:
    st.markdown("**Predictions:**")
    for i, p in enumerate(st.session_state.predictions["predictions"], 1):
        st.write(
            f"**{i}. {p['category']}** "
            f"(score = {p['score']:.3f})"
        )
elif text:
    st.info("Typing… predicting…")
