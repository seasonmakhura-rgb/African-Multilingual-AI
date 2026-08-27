import os
import pickle
import numpy as np
import streamlit as st
from google import genai

st.set_page_config(page_title="BaoBab AI Assistant", page_icon="🌍")
st.title("🌍 BaoBab AI Assistant")

# Cache resources
@st.cache_resource
def load_assets():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('african_lang_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return vectorizer, model, label_encoder

vectorizer, model, label_encoder = load_assets()

ISO_MAP = {
    'amh': 'Amharic', 'fra': 'French', 'hau': 'Hausa', 'ibo': 'Igbo',
    'lin': 'Lingala', 'lug': 'Luganda', 'orm': 'Oromo', 'pcm': 'Nigerian Pidgin',
    'run': 'Kirundi', 'sna': 'Shona', 'som': 'Somali', 'swa': 'Swahili',
    'tir': 'Tigrinya', 'xho': 'Xhosa', 'yor': 'Yoruba'
}

api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

display_languages = sorted([ISO_MAP.get(code, code) for code in label_encoder.classes_])
target_language = st.sidebar.selectbox("Target Response Language", options=display_languages)

# Maintain Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# ⬇️ INSERT VOICE INPUT CODE RIGHT HERE (BEFORE CHAT INPUT) ⬇️
# =========================================================
audio_value = st.audio_input("Record your question")

if audio_value:
    st.info("🎙️ Audio recorded successfully!")
# =========================================================

# Text Chat Input
if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Predict language using TF-IDF + Logistic Regression
    vec_input = vectorizer.transform([user_input.lower().strip()])
    probs = model.predict_proba(vec_input)[0]
    idx = np.argmax(probs)
    conf = float(probs[idx]) * 100
    detected_code = label_encoder.classes_[idx]
    detected_lang_name = ISO_MAP.get(detected_code, detected_code)

    st.sidebar.info(f"🧠 Detected Input Language: **{detected_lang_name}** ({conf:.1f}% confidence)")

    # Call Gemini API
    system_instruction = (
        f"You are an expert multilingual conversational AI.\n"
        f"Detected user language: {detected_lang_name}.\n"
        f"TARGET OUTPUT LANGUAGE ENFORCEMENT: Compose your response strictly in {target_language}."
    )
    
    if client:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{system_instruction}\n\nUser Message: {user_input}"
            )
            bot_reply = response.text
        except Exception as e:
            bot_reply = f"Error generating response: {str(e)}"
    else:
        bot_reply = "⚠️ Please configure your GEMINI_API_KEY in Streamlit secrets."

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
