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
st.set_page_config(page_title="African Multilingual RAG AI", page_icon="🌍", layout="wide")

# Ensure persistent storage directory exists
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

# Dynamic FAISS Index builder
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
ALL_CATEGORIES = sorted(list(set(doc["category"] for doc in all_active_documents)))

# --- Header Section ---
st.title("🌍 African Multilingual AI Assistant (RAG Enabled)")
st.caption("Integrates BiLSTM language classification, multi-format & camera knowledge ingestion, guardrails, and persistent chat memory.")

# --- Main Page Ingestion Panel (Expander) ---
with st.expander("📂 **Knowledge Ingestion Panel (Upload Documents or Take Photos)**", expanded=False):
    ingest_col1, ingest_col2 = st.columns([2, 1])
    
    with ingest_col1:
        ingest_source = st.radio("Select Input Method:", ["📄 Upload Document (PDF, DOCX, PPTX, TXT, Images)", "📷 Take Document Photo"], horizontal=True)
        extracted_text = ""
        doc_title_name = ""

        if ingest_source == "📄 Upload Document (PDF, DOCX, PPTX, TXT, Images)":
            uploaded_file = st.file_uploader(
                "Upload Document File", 
                type=["txt", "pdf", "docx", "pptx", "jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                doc_title_name = uploaded_file.name
                with st.spinner("Parsing document contents..."):
                    extracted_text = extract_text_from_file(uploaded_file, client_gemini=client)
        else:
            camera_image = st.camera_input("Take a photo of your document")
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

    with ingest_col2:
        upload_category = st.text_input("Assign Knowledge Category", value="Custom Knowledge")
        if extracted_text:
            st.info(f"**Extracted Character Count:** {len(extracted_text)}")
        
        if st.button("📥 Index into FAISS Vector DB", use_container_width=True):
            if extracted_text:
                new_doc = {
                    "id": len(all_active_documents) + 1,
                    "title": doc_title_name,
                    "category": upload_category.strip() if upload_category.strip() else "Custom Knowledge",
                    "text": extracted_text
                }
                st.session_state.custom_documents.append(new_doc)
                st.success(f"Indexed **{doc_title_name}**!")
                st.rerun()
            else:
                st.warning("No readable text found to index.")

st.markdown("---")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings & Controls")
    target_language = st.selectbox(
        "Target Output Language", 
        sorted(label_encoder.classes_), 
        index=sorted(label_encoder.classes_).index("Portuguese") if "Portuguese" in label_encoder.classes_ else 0
    )
    
    st.markdown("---")
    st.header("🔍 RAG Knowledge Filters")
    selected_categories = st.multiselect(
        "Filter Active Knowledge Domains",
        options=ALL_CATEGORIES,
        default=ALL_CATEGORIES,
        help="Restrict vector retrieval to specific categories."
    )

    st.markdown("---")
    st.header("💾 Session History")
    
    session_name = st.text_input("Session Name", value="Session_1")
    if st.button("💾 Save Session"):
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
            st.success(f"Saved **{session_name}.json**")
            st.rerun()

    saved_files = [f.replace(".json", "") for f in os.listdir(SAVES_DIR) if f.endswith(".json")]
    if saved_files:
        selected_file = st.selectbox("Load History", ["-- Select Session --"] + saved_files)
        if st.button("📂 Load Session"):
            if selected_file != "-- Select Session --":
                file_path = os.path.join(SAVES_DIR, f"{selected_file}.json")
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                st.session_state.messages = loaded_data.get("messages", [])
                st.session_state.analytics = loaded_data.get("analytics", st.session_state.analytics)
                st.rerun()

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
        st.write(f"⚡ **BiLSTM Avg:** `{np.mean(b_latencies):.2f} ms`")
        st.write(f"🔍 **FAISS Avg:** `{np.mean(f_latencies):.2f} ms`")
        st.line_chart({
            "BiLSTM (ms)": b_latencies,
            "FAISS (ms)": f_latencies
        })

# --- Main Chat UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            st.caption(msg["metadata"])
        if "audio" in msg and msg["audio"] is not None:
            st.audio(msg["audio"], format="audio/mp3")

input_mode = st.radio("Choose Input Method:", ["💬 Text Input", "🎙️ Voice Input"], horizontal=True)

user_input = ""

if input_mode == "💬 Text Input":
    user_input = st.text_input("Type your message here:", key="text_field")
else:
    st.write("Click below to record voice:")
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
        st.warning("Please provide input.")
    else:
        st.session_state.analytics["total_requests"] += 1
        
        # Step 0: Guardrails
        is_safe, guardrail_msg = validate_user_input(user_input)
        
        if not is_safe:
            st.session_state.analytics["blocked_requests"] += 1
            st.error(f"🚫 **Input Blocked**: {guardrail_msg}")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.status("Executing RAG Pipeline...", expanded=True) as status:
                # Step 1: BiLSTM Detection
                t0_bilstm = time.perf_counter()
                seq = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=50)
                preds = model.predict(padded)
                detected_lang = label_encoder.inverse_transform([np.argmax(preds)])[0]
                confidence = float(np.max(preds)) * 100
                
                bilstm_latency_ms = (time.perf_counter() - t0_bilstm) * 1000
                st.session_state.analytics["bilstm_latencies"].append(round(bilstm_latency_ms, 2))

                # Step 2: FAISS Vector Retrieval
                t0_faiss = time.perf_counter()
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
                        
                        if similarity_score >= 0.35:
                            retrieved_doc = active_docs[indices[0][0]]
                            rag_context = retrieved_doc["text"]
                            doc_title = f"[{retrieved_doc['category']}] {retrieved_doc['title']}"
                        else:
                            doc_title = "None (Low Similarity)"

                faiss_latency_ms = (time.perf_counter() - t0_faiss) * 1000
                st.session_state.analytics["faiss_latencies"].append(round(faiss_latency_ms, 2))

                # Step 3: Prompt Construction
                history_context = "".join([f"{m['role'].upper()}: {m['content']}\n" for m in st.session_state.messages[:-1]])

                prompt_sent = f"""SYSTEM INSTRUCTION:
You are an expert multilingual AI assistant.
Detected Input Language: {detected_lang} (Confidence: {confidence:.1f}%).
TARGET OUTPUT LANGUAGE: Compose your response strictly in {target_language}.

RETRIEVED CONTEXT:
{rag_context}

HISTORY:
{history_context if history_context else "None."}

QUERY: {user_input}
RESPONSE ({target_language}):"""

                # Step 4: Call Gemini API
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_sent
                    )
                    ai_output = response.text
                except Exception as e:
                    ai_output = f"Error: {str(e)}"

                # Step 5: TTS
                audio_bytes = None
                selected_tts_lang = TTS_LANG_MAP.get(target_language, 'en')
                try:
                    tts = gTTS(text=ai_output, lang=selected_tts_lang)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    audio_bytes = fp.getvalue()
                except Exception:
                    pass

                status.update(label="Complete!", state="complete", expanded=False)

            # Metadata
            metadata = f"🧠 Detected **{detected_lang}** ({confidence:.1f}%) | 📄 Context: *{doc_title}* | ⏱️ BiLSTM: `{bilstm_latency_ms:.1f}ms` | FAISS: `{faiss_latency_ms:.1f}ms`"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_output, 
                "metadata": metadata,
                "audio": audio_bytes
            })

            st.rerun()
