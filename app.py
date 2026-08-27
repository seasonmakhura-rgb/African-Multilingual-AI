import os
import pickle
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# Page Configuration
st.set_page_config(page_title="Multilingual AI Assistant", page_icon="🌍", layout="centered")
st.title("🌍 Multilingual AI Assistant")

# Cache resources to prevent reloading on every user interaction
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

# Language Mapping (ISO code -> Full Name)
ISO_MAP = {
    'amh': 'Amharic', 'fra': 'French', 'hau': 'Hausa', 'ibo': 'Igbo',
    'lin': 'Lingala', 'lug': 'Luganda', 'orm': 'Oromo', 'pcm': 'Nigerian Pidgin',
    'run': 'Kirundi', 'sna': 'Shona', 'som': 'Somali', 'swa': 'Swahili',
    'tir': 'Tigrinya', 'xho': 'Xhosa', 'yor': 'Yoruba'
}

# Fetch Gemini API Key from Secrets or Environment
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

# Sidebar Setup: Selection for target output language
display_languages = sorted([ISO_MAP.get(code, code) for code in label_encoder.classes_])
target_language = st.sidebar.selectbox("Target Response Language", options=display_languages)

# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior conversation turns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create clean unified input interface using Streamlit Tabs
tab_text, tab_audio = st.tabs(["💬 Type Message", "🎙️ Record Voice"])

user_input = None

with tab_text:
    text_prompt = st.chat_input("Type your message here...")
    if text_prompt:
        user_input = text_prompt

with tab_audio:
    audio_file = st.audio_input("Record audio question")
    if audio_file:
        if client:
            with st.spinner("Transcribing audio input via Gemini..."):
                try:
                    audio_bytes = audio_file.read()
                    
                    # Convert audio bytes into proper Part type for google-genai SDK
                    audio_part = types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type="audio/wav"
                    )
                    
                    transcribe_response = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=[
                            "Transcribe the following spoken audio verbatim into text. Return only the raw text response.",
                            audio_part
                        ]
                    )
                    user_input = transcribe_response.text.strip()
                except Exception as e:
                    st.error(f"Failed to transcribe audio: {str(e)}")
        else:
            st.error("⚠️ Gemini API Key is missing. Please configure GEMINI_API_KEY in Streamlit secrets.")

# Process input (either from text input or voice transcription)
if user_input:
    # Render and store user prompt
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1. Classification via TF-IDF + Logistic Regression Model
    vec_input = vectorizer.transform([user_input.lower().strip()])
    probs = model.predict_proba(vec_input)[0]
    idx = np.argmax(probs)
    conf = float(probs[idx]) * 100
    detected_code = label_encoder.classes_[idx]
    detected_lang_name = ISO_MAP.get(detected_code, detected_code)

    # Display detected language metrics in sidebar
    st.sidebar.info(f"🧠 Detected Input Language: **{detected_lang_name}** ({conf:.1f}% confidence)")

    # 2. Build system instruction and generate output using Gemini API
    system_instruction = (
        f"You are an expert multilingual conversational AI.\n"
        f"Detected user input language: {detected_lang_name}.\n"
        f"TARGET OUTPUT LANGUAGE ENFORCEMENT: Compose your response strictly in {target_language}."
    )
    
    if client:
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=f"{system_instruction}\n\nUser Message: {user_input}"
            )
            bot_reply = response.text
        except Exception as e:
            bot_reply = f"Error generating response: {str(e)}"
    else:
        bot_reply = "⚠️ Please configure your GEMINI_API_KEY in Streamlit secrets to enable AI responses."

    # Render and store assistant response
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
