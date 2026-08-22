import os
import io
import time
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google import genai
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import faiss
from PIL import Image
import pypdf
import docx
import pptx
from documents import KNOWLEDGE_DOCUMENTS
from guardrails import validate_user_input

# Page Config
st.set_page_config(page_title="African Multilingual RAG AI", page_icon="🌍")

# Ensure persistent storage directory exists for saved chats
SAVES_DIR = "saved_chats"
os.makedirs(SAVES_DIR, exist_ok=True)

# Dynamic TTS Language Mapping
TTS_LANG_MAP = {
    'Portuguese': 'pt',
    'French': 'fr',
    'Swahili': 'sw',
    'English': 'en',
    'Spanish': 'es'
}

# Initialize Memory, Analytics, & Dynamic Knowledge Base State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "bilstm_latencies": [],
        "faiss_latencies": [],
        "total_requests": 0,
        "blocked_requests": 0
    }

if "custom_documents" not in st.session_state:
    st.session_state.custom_documents = []

# Gemini API Client setup
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key)

# Helper function to extract text from multi-format files
def extract_text_from_file(uploaded_file, client_gemini=None):
    filename = uploaded_file.name.lower()
    
    # 1. Plain Text File (.txt)
    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8").strip()

    # 2. PDF File (.pdf)
    elif filename.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text_content = []
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_content.append(extracted)
        return "\n".join(text_content).strip()

    # 3. Word Document (.docx)
    elif filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()

    # 4. PowerPoint Presentation (.pptx)
    elif filename.endswith(".pptx"):
        prs = pptx.Presentation(uploaded_file)
        text_content = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_content.append(shape.text)
        return "\n".join(text_content).strip()

    # 5. Image Files (.jpg, .jpeg, .png) via Gemini Vision OCR
    elif filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(uploaded_file)
        prompt = "Extract and transcribe all readable text from this document image cleanly and accurately. Do not add commentary."
        response = client_gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt]
        )
        return response.text.strip()

    return ""

# Load Classifier Artifacts
@st.cache_resource
def load_system_resources():
    model = tf.keras.models.load_model('african_lang_classifier.keras')
    with open('word_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_system_resources()

# Helper function to dynamically build FAISS index based on active category filters and uploads
def build_faiss_index_for_docs(docs, tokenizer, embedding_dim=128):
    if not docs:
        return None, []
    
    np.random.seed(42)
    doc_vectors = []
    for doc in docs:
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
    return index, docs

# Combine static documents with dynamic custom uploads
all_active_documents = KNOWLEDGE_DOCUMENTS + st.session_state.custom_documents

# Extract unique document categories dynamically
ALL_CATEGORIES = sorted(list(set(doc["category"] for doc in all_active_documents)))

# App UI
st.title("🌍 African Multilingual AI Assistant (RAG Enabled)")
st.write("Integrates BiLSTM language classification, multi-format & camera knowledge ingestion, guardrails, and persistent chat memory.")

# Sidebar Controls & Dynamic Uploads
with st.sidebar:
    st.header("⚙️ Settings & Controls")
    target_language = st.selectbox(
        "Select Target Output Language", 
        sorted(label_encoder.classes_), 
        index=sorted(label_encoder.classes_).index("Portuguese") if "Portuguese" in label_encoder.classes_ else 0
    )
    
    st.markdown("---")
    st.header("💾 Session Memory & History")
    
    # Save Current Session
    session_name = st.text_input("Session Name", value="Session_1")
    if st.button("💾 Save Current Chat Session"):
        if st.session_state.messages:
            save_payload = {
                "messages": [
                    {k: v for k, v in msg.items() if k != "audio"} 
                    for msg in st.session_state.messages
                ],
                "analytics": st.session_state.analytics
            }
            file_path = os.path.join(SAVES_DIR, f"{session_name.strip()}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_payload, f, indent=2)
            st.success(f"Session saved as **{session_name}.json**!")
            st.rerun()
        else:
            st.warning("No messages to save yet.")

    # Load Saved Session
    saved_files = [f.replace(".json", "") for f in os.listdir(SAVES_DIR) if f.endswith(".json")]
    if saved_files:
        selected_file = st.selectbox("Load Saved History", ["-- Select Session --"] + saved_files)
        if st.button("📂 Load Selected Session"):
            if selected_file != "-- Select Session --":
                file_path = os.path.join(SAVES_DIR, f"{selected_file}.json")
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                st.session_state.messages = loaded_data.get("messages", [])
                st.session_state.analytics = loaded_data.get("analytics", st.session_state.analytics)
                st.success(f"Loaded **{selected_file}**!")
                st.rerun()
    else:
        st.caption("No saved sessions found on disk.")

    st.markdown("---")
    st.header("📁 Multi-Format Knowledge Ingestion")
    
    # Ingestion Source Selector
    ingest_source = st.radio("Choose Input Type:", ["📄 Upload Document", "📷 Capture Photo"], horizontal=True)
    upload_category = st.text_input("Assign Category for Upload", value="Custom Knowledge")

    extracted_text = ""
    doc_title_name = ""

    if ingest_source == "📄 Upload Document":
        uploaded_file = st.file_uploader(
            "Upload TXT, PDF, DOCX, PPTX, or JPG/PNG", 
            type=["txt", "pdf", "docx", "pptx", "jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            doc_title_name = uploaded_file.name
            with st.spinner("Parsing document contents..."):
                extracted_text = extract_text_from_file(uploaded_file, client_gemini=client)
    else:
        camera_image = st.camera_input("Take a photo of a document")
        if camera_image is not None:
            doc_title_name = f"Camera_Scan_{int(time.time())}.jpg"
            with st.spinner("Running Vision OCR on camera photo..."):
                img = Image.open(camera_image)
                prompt = "Extract and transcribe all readable text from this document image cleanly and accurately."
                ocr_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[img, prompt]
                )
                extracted_text = ocr_response.text.strip()

    if st.button("📥 Index into FAISS Vector DB"):
        if extracted_text:
            new_doc = {
                "id": len(all_active_documents) + 1,
                "title": doc_title_name,
                "category": upload_category.strip() if upload_category.strip() else "Custom Knowledge",
                "text": extracted_text
            }
            st.session_state.custom_documents.append(new_doc)
            st.success(f"Successfully indexed **{doc_title_name}** into FAISS!")
            st.rerun()
        else:
            st.warning("No readable text found to index.")

    st.markdown("---")
    # RAG Category Filter
    selected_categories = st.multiselect(
        "Filter RAG Knowledge Categories",
        options=ALL_CATEGORIES,
        default=ALL_CATEGORIES,
        help="Restrict vector retrieval to specific knowledge domains."
    )
    
    if st.button("🗑️ Clear Active Chat"):
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
        st.caption("Inference Latency Trend (ms)")
        st.line_chart({
            "BiLSTM (ms)": b_latencies,
            "FAISS (ms)": f_latencies
        })

    st.markdown("---")
    st.header("💾 Export Data")
    
    # 1. Export Chat Logs as JSON
    export_chat_data = []
    for msg in st.session_state.messages:
        export_chat_data.append({
            "role": msg["role"],
            "content": msg["content"],
            "metadata": msg.get("metadata", "")
        })
    
    chat_json_bytes = json.dumps(export_chat_data, indent=2).encode('utf-8')
    st.download_button(
        label="📥 Download Chat History (JSON)",
        data=chat_json_bytes,
        file_name="chat_history.json",
        mime="application/json",
        disabled=len(export_chat_data) == 0
    )

    # 2. Export Telemetry & Latencies as CSV
    if b_latencies:
        metrics_df = pd.DataFrame({
            "Turn": list(range(1, len(b_latencies) + 1)),
            "BiLSTM_Latency_ms": b_latencies,
            "FAISS_Latency_ms": f_latencies
        })
        csv_bytes = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Latency Metrics (CSV)",
            data=csv_bytes,
            file_name="latency_metrics.csv",
            mime="text/csv"
        )

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
                # Step 1: BiLSTM Language Detection
                st.write("🧠 Classifying user language with BiLSTM model...")
                t0_bilstm = time.perf_counter()
                
                seq = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=50)
                preds = model.predict(padded)
                detected_lang = label_encoder.inverse_transform([np.argmax(preds)])[0]
                confidence = float(np.max(preds)) * 100
                
                bilstm_latency_ms = (time.perf_counter() - t0_bilstm) * 1000
                st.session_state.analytics["bilstm_latencies"].append(round(bilstm_latency_ms, 2))

                # Step 2: RAG Retrieval via Filtered FAISS Index
                st.write("🔍 Filtering documents & querying FAISS Vector Database...")
                t0_faiss = time.perf_counter()
                
                # Filter documents across static and dynamic uploads by category
                filtered_docs = [doc for doc in all_active_documents if doc["category"] in selected_categories]
                embedding_dim = 128
                active_faiss_index, active_docs = build_faiss_index_for_docs(filtered_docs, tokenizer, embedding_dim)

                rag_context = "No relevant document found."
                doc_title = "None"
                
                if active_faiss_index is not None:
                    q_vec = np.zeros(embedding_dim, dtype='float32')
                    if seq[0]:
                        for idx in seq[0][:embedding_dim]:
                            q_vec[idx % embedding_dim] += 1.0

                    if np.sum(q_vec) > 0:
                        q_matrix = np.array([q_vec]).astype('float32')
                        faiss.normalize_L2(q_matrix)
                        distances, indices = active_faiss_index.search(q_matrix, k=1)
                        similarity_score = float(distances[0][0])
                        
                        SIMILARITY_THRESHOLD = 0.35
                        if similarity_score >= SIMILARITY_THRESHOLD:
                            retrieved_doc = active_docs[indices[0][0]]
                            rag_context = retrieved_doc["text"]
                            doc_title = f"[{retrieved_doc['category']}] {retrieved_doc['title']}"
                        else:
                            doc_title = "None (Low Similarity)"
                    else:
                        doc_title = "None (No Match)"
                else:
                    doc_title = "None (No Categories Selected)"

                faiss_latency_ms = (time.perf_counter() - t0_faiss) * 1000
                st.session_state.analytics["faiss_latencies"].append(round(faiss_latency_ms, 2))

                st.write(f"📄 Retrieved Context: **{doc_title}**")

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

            # Store metadata
            metadata = f"🧠 Detected **{detected_lang}** ({confidence:.1f}%) | 📄 Context: *{doc_title}* | ⏱️ BiLSTM: `{bilstm_latency_ms:.1f}ms` | FAISS: `{faiss_latency_ms:.1f}ms`"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_output, 
                "metadata": metadata,
                "audio": audio_bytes
            })

            st.rerun()
