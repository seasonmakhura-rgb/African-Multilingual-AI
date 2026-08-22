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
import faiss
from documents import KNOWLEDGE_DOCUMENTS

# Page Config
st.set_page_config(page_title="African Multilingual RAG AI", page_icon="🌍")

# Dynamic TTS Language Mapping
TTS_LANG_MAP = {
    'Portuguese': 'pt',
    'French': 'fr',
    'Swahili': 'sw',
    'English': 'en',
    'Spanish': 'es'
}

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Classifier Artifacts & Build FAISS RAG Index
@st.cache_resource
def load_system_resources():
    model = tf.keras.models.load_model('african_lang_classifier.keras')
    with open('word_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Build Lightweight FAISS Vector Index using Tokenizer Bag-of-Words Embeddings
    embedding_dim = 128
    np.random.seed(42)
    doc_vectors = []
    for doc in KNOWLEDGE_DOCUMENTS:
        seq = tokenizer.texts_to_sequences([doc['text']])
        vec = np.zeros(embedding_dim, dtype='float32')
        if seq[0]:
            for idx in seq[0][:embedding_dim]:
                vec[idx % embedding_dim] += 1.0
        doc_vectors.append(vec)
    
    doc_matrix = np.array(doc_vectors).astype('float32')
    faiss.normalize_L2(doc_matrix)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(doc_matrix)

    return model, tokenizer, label_encoder, index, embedding_dim

model, tokenizer, label_encoder, faiss_index, embedding_dim = load_system_resources()

# Gemini API Client setup
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key)

# App UI
st.title("🌍 African Multilingual AI Assistant (RAG Enabled)")
st.write("Integrates custom BiLSTM language classification with FAISS vector retrieval and Gemini LLM synthesis.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings & Controls")
    target_language = st.selectbox(
        "Select Target Output Language", 
        sorted(label_encoder.classes_), 
        index=sorted(label_encoder.classes_).index("Portuguese") if "Portuguese" in label_encoder.classes_ else 0
    )
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            st.caption(msg["metadata"])

# Input Mode Selection
input_mode = st.radio("Choose Input Method:", ["💬 Text Input", "🎙️ Voice Input"], horizontal=True)

user_input = ""

if input_mode == "💬 Text Input":
    user_input = st.text_input("Type your message here:", key="text_field")
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

if st.button("Send Request", type="primary"):
    if not user_input.strip():
        st.warning("Please provide a text input or speak into the microphone.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.status("Executing RAG & Multi-Turn Pipeline...", expanded=True) as status:
            # Step 1: BiLSTM Language Detection
            st.write("🧠 Classifying user language with BiLSTM model...")
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=50)
            preds = model.predict(padded)
            detected_lang = label_encoder.inverse_transform([np.argmax(preds)])[0]
            confidence = float(np.max(preds)) * 100

            # Step 2: RAG Retrieval via FAISS with Zero-Vector Guard
            st.write("🔍 Querying FAISS Vector Database for context...")
            q_vec = np.zeros(embedding_dim, dtype='float32')
            if seq[0]:
                for idx in seq[0][:embedding_dim]:
                    q_vec[idx % embedding_dim] += 1.0

            if np.sum(q_vec) > 0:
                q_matrix = np.array([q_vec]).astype('float32')
                faiss.normalize_L2(q_matrix)
                distances, indices = faiss_index.search(q_matrix, k=1)
                similarity_score = float(distances[0][0])
                
                SIMILARITY_THRESHOLD = 0.35
                if similarity_score >= SIMILARITY_THRESHOLD:
                    retrieved_doc = KNOWLEDGE_DOCUMENTS[indices[0][0]]
                    rag_context = retrieved_doc["text"]
                    doc_title = retrieved_doc["title"]
                else:
                    rag_context = "No relevant document found."
                    doc_title = "None (Low Similarity)"
            else:
                rag_context = "No relevant document found."
                doc_title = "None (No Match)"

            st.write(f"📄 Retrieved Document: **{doc_title}**")

            # Step 3: System Prompt Construction
            history_context = ""
            for past_msg in st.session_state.messages[:-1]:
                history_context += f"{past_msg['role'].upper()}: {past_msg['content']}\n"

            prompt_sent = f"""SYSTEM INSTRUCTION:
You are an expert multilingual AI assistant.
Detected Input Language from User: {detected_lang} (Confidence: {confidence:.1f}%).
TARGET OUTPUT LANGUAGE ENFORCEMENT: Regardless of the input language, you MUST compose your entire response strictly in {target_language}.

RETRIEVED KNOWLEDGE CONTEXT (RAG):
{rag_context}

CONVERSATION HISTORY:
{history_context if history_context else "No prior context."}

CURRENT USER QUERY: {user_input}
RESPONSE ({target_language}):"""

            # Step 4: Call Gemini API
            st.write("🚀 Generating grounded AI response...")
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt_sent
                )
                ai_output = response.text
            except Exception as e:
                ai_output = f"Error: {str(e)}"

            status.update(label="RAG Pipeline Execution Complete!", state="complete", expanded=False)

        metadata = f"🧠 Detected **{detected_lang}** ({confidence:.1f}%) | 📄 Knowledge Context: *{doc_title}*"
        st.session_state.messages.append({"role": "assistant", "content": ai_output, "metadata": metadata})

        # Step 5: TTS Speech Synthesis
        selected_tts_lang = TTS_LANG_MAP.get(target_language, 'en')
        try:
            tts = gTTS(text=ai_output, lang=selected_tts_lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.audio(fp, format='audio/mp3')
        except Exception:
            pass

        st.rerun()
