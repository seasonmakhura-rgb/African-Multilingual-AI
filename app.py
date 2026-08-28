import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import joblib

# Load your saved vectorizer and model pipeline
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("saga_model.pkl")

def predict_language(text):
    # Vectorize and predict
    vec_text = vectorizer.transform([text])
    prediction = model.predict(vec_text)[0]
    probs = model.predict_proba(vec_text)
    confidence = float(max(probs[0]) * 100)
    return prediction, confidence


# Set Streamlit Page Config
st.set_page_config(
    page_title="Multilingual African Language AI",
    page_icon="🌍",
    layout="centered"
)

# ==========================================
# 1. LOAD RE-EXPORTED MODEL ARTIFACTS (CELL 7)
# ==========================================
@st.cache_resource
def load_artifacts():
    with open('african_lang_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

try:
    model, vectorizer, label_encoder = load_artifacts()
    st.sidebar.success("Model artifacts loaded successfully! (~12MB)")
except Exception as e:
    st.error(f"Error loading model artifacts from Cell 7: {e}")
    st.stop()

# Language ISO Mapping for Clean Display UI
LANG_NAMES = {
    'amh': 'Amharic (አማርኛ)',
    'fra': 'French (Français)',
    'hau': 'Hausa (Harshen Hausa)',
    'ibo': 'Igbo (Asụsụ Igbo)',
    'lin': 'Lingala (Lingála)',
    'lug': 'Luganda (Oluganda)',
    'orm': 'Oromo (Afaan Oromoo)',
    'pcm': 'Nigerian Pidgin',
    'run': 'Kirundi (Ikirundi)',
    'sna': 'Shona (chiShona)',
    'som': 'Somali (Soomaaliga)',
    'swa': 'Swahili (Kiswahili)',
    'tir': 'Tigrinya (ትግርኛ)',
    'xho': 'isiXhosa',
    'yor': 'Yoruba (Èdè Yorùbá)'
}

import streamlit as st
import google.generativeai as genai
from your_model_file import predict_language  # Your trained ML model

st.title("Baobab AI - Multilingual Assistant")

# 1. Initialize persistent chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render previous conversation thread
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Capture user chat input (replacing the single form button)
if prompt := st.chat_input("Ask a question in your language..."):
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step A: Run YOUR model to identify the language instantly
    detected_lang, confidence = predict_language(prompt)

    # Step B: Generate conversational response (e.g., via Gemini)
    # Using system instruction to force response in the detected language
    model = genai.GenerativeModel("gemini-1.5-flash")
    system_instruction = f"The user is speaking {detected_lang}. Reply back accurately in {detected_lang}."
    
    response = model.generate_content(f"{system_instruction}\nUser prompt: {prompt}")

    # Step C: Display Assistant response + metadata tag
    bot_reply = f"*(Detected: {detected_lang} - {confidence:.1f}% confidence)*\n\n{response.text}"
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
