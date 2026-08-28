import joblib
import streamlit as st
import google.generativeai as genai

# 1. Load your exact ML pipeline artifacts
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("african_lang_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_language(text):
    # Vectorize input text using TF-IDF subword features
    vec_text = vectorizer.transform([text])
    
    # Predict integer class and map to string language label
    numeric_pred = model.predict(vec_text)
    language_name = label_encoder.inverse_transform(numeric_pred)[0]
    
    # Calculate model probability confidence
    probs = model.predict_proba(vec_text)
    confidence = float(max(probs[0]) * 100)
    
    return language_name, confidence

# 2. Conversational UI Layout
st.title("Baobab AI - Multilingual Assistant")

# Maintain persistent chat thread
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Chat Input Loop
if prompt := st.chat_input("Type your message..."):
    
    # Render user prompt in chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step A: Run on-device classification using your loaded artifacts
    detected_lang, confidence = predict_language(prompt)

    # Step B: Generate conversational response via Gemini forced into the detected language
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    
    system_prompt = f"The user is speaking {detected_lang}. Respond to them conversationally, accurately, and fluently in {detected_lang}."
    response = gemini_model.generate_content(f"{system_prompt}\nUser message: {prompt}")

    # Step C: Render assistant reply with language metadata tag
    bot_reply = f"*(Detected Language: **{detected_lang}** | Confidence: **{confidence:.1f}%**)*\n\n{response.text}"
    
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
