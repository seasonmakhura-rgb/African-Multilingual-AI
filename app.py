import os
import io
import time
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
from guardrails import validate_user_input

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

# Initialize Chat Memory & Performance Metrics State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "bilstm_latencies": [],
        "faiss_latencies": [],
        "total_requests": 0,
        "blocked_requests": 0
    }

# Load Classifier Artifacts & Build FAISS RAG Index
@st.cache_resource
def load_system_resources():
    model = tf.keras.models.load_model('african_lang_classifier.keras')
    with open('word_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Build Lightweight FAISS Vector Index
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
st.write("Integrates BiLSTM language classification, FAISS retrieval, safety guardrails, and latency monitoring.")

# Sidebar Controls & Analytics Monitor
with st.sidebar:
    st.header("⚙️ Settings & Controls")
    target_language = st.selectbox(
        "Select Target Output Language", 
        sorted(label_encoder.classes_), 
        index=sorted(label_encoder.classes_).index("Portuguese") if "Portuguese" in label_encoder.classes_ else 0
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.analytics = {
            "bilstm_latencies": [],
            "faiss_latencies": [],
            "total_requests": 0,
            "blocked_requests": 0
        }
        st.rerun()

    st.markdown("---")
    st.header("📊 Performance Analytics")
    
    total_reqs = st.session_state.analytics["total_requests"]
    blocked_reqs = st.session_state.analytics["blocked_requests"]
    
    col1, col2 = st.columns(2)
    col1.metric("Total Turns", total_reqs)
    col2.metric("Blocked", blocked_reqs)

    b_latencies = st.session_state.analytics["bilstm_latencies"]
    f_latencies = st.session_state.analytics["faiss_latencies"]

    if b_latencies:
        avg_bilstm = float(np.mean(b_latencies))
        avg_faiss = float(np.mean(f_latencies))
        st.write(f"⚡ **Avg BiLSTM Latency:** `{avg_bilstm:.2f} ms`")
        st.write(f"🔍 **Avg FAISS Latency:** `{avg_faiss:.2f} ms`")
        
        # Plot latency trend
        st.caption("Inference Latency Trend (ms)")
        st.line_chart({
            "BiLSTM (ms)": b_latencies,
            "FAISS (ms)": f_latencies
        })

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            st.caption(msg["metadata"])
        if "audio" in msg and msg["audio"] is not None:
            st.audio(msg["audio"], format="audio/mp3")

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
        st.session_state.analytics["total_requests"] += 1
        
        # Step 0: Execute Local Guardrails Check
        is_safe, guardrail_msg = validate_user_input(user_input)
        
        if not is_safe:
            st.session_state.analytics["blocked_requests"] += 1
            st.error(f"🚫 **Input Blocked by System Guardrails**: {guardrail_msg}")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.status("Executing Multimodal & RAG Pipeline...", expanded=True) as status:
                # Step 1: BiLSTM Language Detection + Benchmark
                st.write("🧠 Classifying user language with BiLSTM model...")
                t0_bilstm = time.perf_counter()
                
                seq = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=50)
                preds = model.predict(padded)
                detected_lang = label_encoder.inverse_transform([np.argmax(preds)])[0]
                confidence = float(np.max(preds)) * 100
                
                bilstm_latency_ms = (time.perf_counter() - t0_bilstm) * 1000
                st.session_state.analytics["bilstm_latencies"].append(round(bilstm_latency_ms, 2))

                # Step 2: RAG Retrieval via FAISS + Benchmark
                st.write("🔍 Querying FAISS Vector Database for context...")
                t0_faiss = time.perf_counter()
                
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

                faiss_latency_ms = (time.perf_counter() - t0_faiss) * 1000
                st.session_state.analytics["faiss_latencies"].append(round(faiss_latency_ms, 2))

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

                # Step 4: Call Gemini API (gemini-2.5-flash)
                st.write("🚀 Generating grounded AI response...")
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_sent
                    )
                    ai_output = response.text
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        ai_output = "⚠️ **Free-tier API limit reached.** Please wait a short moment and try again."
                    else:
                        ai_output = f"Error: {str(e)}"

                # Step 5: Speech Synthesis
                st.write("🔊 Synthesizing speech output...")
                audio_bytes = None
                selected_tts_lang = TTS_LANG_MAP.get(target_language, 'en')
                try:
                    tts = gTTS(text=ai_output, lang=selected_tts_lang)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    audio_bytes = fp.getvalue()
                except Exception:
                    pass

                status.update(label="Pipeline Execution Complete!", state="complete", expanded=False)

            # Store metadata with performance metrics
            metadata = f"🧠 Detected **{detected_lang}** ({confidence:.1f}%) | 📄 Knowledge Context: *{doc_title}* | ⏱️ BiLSTM: `{bilstm_latency_ms:.1f}ms` | FAISS: `{faiss_latency_ms:.1f}ms`"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_output, 
                "metadata": metadata,
                "audio": audio_bytes
            })

            st.rerun()
