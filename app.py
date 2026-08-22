import os
import pickle
import numpy as np
import streamlit as st
import sklearn
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google import genai

# Page Config
st.set_page_config(page_title="African Multilingual AI Assistant", page_icon="🌍")

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
st.write("Detects 16 languages using a custom BiLSTM neural network and responds in your chosen language.")

# User Inputs
user_input = st.text_area("Type your message here:", placeholder="e.g., Habari yako, o kae?")
target_language = st.selectbox(
    "Select Target Output Language", 
    sorted(label_encoder.classes_), 
    index=sorted(label_encoder.classes_).index("Portuguese") if "Portuguese" in label_encoder.classes_ else 0
)

if st.button("Send Request", type="primary"):
    if not user_input.strip():
        st.warning("Please type a message first.")
    else:
        # 1. Preprocess & Predict Language
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=50)
        preds = model.predict(padded)
        detected_lang = label_encoder.inverse_transform([np.argmax(preds)])[0]
        confidence = float(np.max(preds)) * 100

        # 2. Build Prompt
        prompt_sent = f"""SYSTEM INSTRUCTION:
You are an expert multilingual AI assistant.
Detected Input Language from User: {detected_lang} (Confidence: {confidence:.1f}%).
TARGET OUTPUT LANGUAGE ENFORCEMENT: Regardless of the input language, you MUST compose your entire response strictly in {target_language}.

USER QUERY: {user_input}
RESPONSE ({target_language}):"""

        # 3. Call Gemini
        with st.spinner("Generating AI response..."):
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt_sent
                )
                ai_output = response.text
            except Exception as e:
                ai_output = f"Error: {str(e)}"

        # Display Outputs
        st.success(f"🧠 **Classifier Output:** Detected **{detected_lang}** ({confidence:.1f}% confidence)")
        st.markdown("### AI Response")
        st.write(ai_output)

        with st.expander("🔍 View Full Backend System Prompt"):
            st.code(prompt_sent)
