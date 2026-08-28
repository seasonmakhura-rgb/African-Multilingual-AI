import joblib
import streamlit as st
import google.generativeai as genai

# 1. Load all three uploaded ML artifacts
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("african_classification.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_language(text):
    # Vectorize input text
    vec_text = vectorizer.transform([text])
    
    # Predict numeric class and map back to language name using label_encoder
    numeric_pred = model.predict(vec_text)
    language_name = label_encoder.inverse_transform(numeric_pred)[0]
    
    # Calculate confidence score
    probs = model.predict_proba(vec_text)
    confidence = float(max(probs[0]) * 100)
    
    return language_name, confidence

# 2. UI Layout & Setup
st.title("Baobab AI - Multilingual Conversational Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Conversational Loop
if prompt := st.chat_input("Ask a question in your language..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Classify language using your 3 loaded ML artifacts
    detected_lang, confidence = predict_language(prompt)

    # Generate Gemini response instructed to speak that language
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    
    system_prompt = f"The user is speaking {detected_lang}. Respond conversationally, accurately, and fluently in {detected_lang}."
    response = gemini_model.generate_content(f"{system_prompt}\nUser message: {prompt}")

    bot_reply = f"*(Detected Language: **{detected_lang}** | Confidence: **{confidence:.1f}%**)*\n\n{response.text}"
    
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
