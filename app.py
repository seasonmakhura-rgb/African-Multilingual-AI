import os
import io
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google import genai
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS

# Page Config
st.set_page_config(page_title="African Multilingual AI Assistant", page_icon="🌍")

# Dynamic TTS Language Mapping
TTS_LANG_MAP = {
    'Portuguese': 'pt',
    'French': 'fr',
    'Swahili': 'sw',
    'English': 'en',
    'Spanish': 'es'
}

# Load Classifier Artifacts
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('african_lang_classifier.keras')
    with open('word_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()

# Gemini API Client setup
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key)

# App UI
st.title("🌍 African Multilingual AI Assistant")
st.write("Detects 16 languages using a custom BiLSTM neural network and responds in your chosen language with voice synthesis.")

# Input Mode Selection
input_mode = st.radio("Choose Input Method:", ["💬 Text Input", "🎙️ Voice Input"], horizontal=True)

user_input = ""

if input_mode == "💬 Text Input":
    user_input = st.text_area("Type your message here:", placeholder="e.g., Habari yako, o kae?")
else:
    st.write("Click below, speak, and click stop to transcribe:")
    recorded_text = speech_to_text(
        start_prompt="🔴 Start Recording", 
        stop_prompt="⏹️ Stop Recording", 
        just_once=False, 
        key='stt'
    )
    if recorded_text:
        user_input = recorded_text
        st.success(f"Transcribed Text: **{user_input}**")

# Select Target Output Language
target_language = st.selectbox(
    "Select Target Output Language", 
    sorted(label_encoder.classes_), 
    index=sorted(label_encoder.classes_).index("Portuguese") if "Portuguese" in label_encoder.classes_ else 0
)

if st.button("Send Request", type="primary"):
    if not user_input.strip():
        st.warning("Please provide a text input or speak into the microphone.")
    else:
        # Step 1: Preprocess & Predict Language via BiLSTM
        with st.status("Running Dynamic Pipeline...", expanded=True) as status:
            st.write("🧠 Preprocessing text & classifying language with BiLSTM model...")
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=50)
            preds = model.predict(padded)
            detected_lang = label_encoder.inverse_transform([np.argmax(preds)])[0]
            confidence = float(np.max(preds)) * 100

            # Step 2: Confidence-Gated Dynamic Prompting
            st.write("⚙️ Assembling dynamic prompt for Gemini LLM...")
            if confidence >= 75.0:
                lang_context = f"Detected Input Language: {detected_lang} (High Confidence: {confidence:.1f}%)."
            else:
                lang_context = f"Detected Input Language might be {detected_lang} (Lower Confidence: {confidence:.1f}%). Please account for potential language or dialect ambiguity."

            prompt_sent = f"""SYSTEM INSTRUCTION:
You are an expert multilingual AI assistant.
{lang_context}
TARGET OUTPUT LANGUAGE ENFORCEMENT: Regardless of the input language, you MUST compose your entire response strictly in {target_language}.

USER QUERY: {user_input}
RESPONSE ({target_language}):"""

            # Step 3: Call Gemini API
            st.write("🚀 Requesting generative response from Gemini...")
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt_sent
                )
                ai_output = response.text
            except Exception as e:
                ai_output = f"Error: {str(e)}"

            status.update(label="Pipeline processing complete!", state="complete", expanded=False)

        # Display Outputs
        if confidence < 75.0:
            st.warning(f"🧠 **Classifier Output:** Detected **{detected_lang}** ({confidence:.1f}% confidence - Low Confidence Mode Active)")
        else:
            st.success(f"🧠 **Classifier Output:** Detected **{detected_lang}** ({confidence:.1f}% confidence)")

        st.markdown("### AI Response")
        st.write(ai_output)

        # Step 4: Dynamic Language Audio Routing (TTS)
        selected_tts_lang = TTS_LANG_MAP.get(target_language, 'en')
        try:
            tts = gTTS(text=ai_output, lang=selected_tts_lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.audio(fp, format='audio/mp3')
        except Exception:
            st.info("Audio playback unavailable for this language code.")

        with st.expander("🔍 View Full Backend System Prompt"):
            st.code(prompt_sent)
