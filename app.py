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
from google.genai import types
from gtts import gTTS
import faiss
from PIL import Image
import pypdf
import docx
import pptx
from documents import KNOWLEDGE_DOCUMENTS
from guardrails import validate_user_input

# Page Config
st.set_page_config(page_title="BAOBAB AI", page_icon="🌳", layout="wide")

# Target Gemini Model Identifier
GEMINI_MODEL = "gemini-3.6-flash"

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

# Initialize State
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

if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""

# Gemini API Client setup
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key)

def transcribe_audio_callback():
    audio_data = st.session_state.get("live_mic_recorder")
    if audio_data is not None:
        try:
            audio_bytes = audio_data.read()
            mime_type = getattr(audio_data, "type", "audio/wav") or "audio/wav"
            
            audio_part = types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type
            )
            
            stt_response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    audio_part,
                    "Transcribe the spoken audio into text accurately without adding explanations or extra output."
                ]
            )
            if stt_response.text:
                st.session_state.pending_input = stt_response.text.strip()
        except Exception as e:
            st.error(f"Audio transcription error: {str(e)}")

def extract_text_from_file(uploaded_file, client_gemini=None):
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8").strip()
    elif filename.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()]).strip()
    elif filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()
    elif filename.endswith(".pptx"):
        prs = pptx.Presentation(uploaded_file)
        text_content = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_content.append(shape.text)
        return "\n".join(text_content).strip()
    elif filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(uploaded_file)
        response = client_gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image, "Extract and transcribe all readable text cleanly and accurately."]
        )
        return response.text.strip()
    return ""

@st.cache_resource
def load_system_resources():
    model = tf.keras.models.load_model('african_lang_classifier.keras')
    with open('word_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_system_resources()

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

def run_local_python_sandbox(code_str: str) -> str:
    """Executes python code locally in a captured standard output stream."""
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        # Restricted global scope for safe execution
        exec_globals = {"np": np, "pd": pd}
        exec(code_str, exec_globals)
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        return f"Execution Error: {str(e)}"
    finally:
        sys.stdout = old_stdout

all_active_documents = KNOWLEDGE_DOCUMENTS + st.session_state.custom_documents

# --- Header ---
st.title("🌳 BAOBAB AI")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    target_language = st.selectbox(
        "Output Language", 
        sorted(label_encoder.classes_), 
        index=sorted(label_encoder.classes_).index("Portuguese") if "Portuguese" in label_encoder.classes_ else 0
    )

    st.markdown("---")
    st.header("🌐 Active Capabilities")
    enable_web_search = st.toggle("Enable Live Web Search", value=False)
    
    # Tier 3 Feature: Code Interpreter Toggle
    enable_code_interpreter = st.toggle("Enable Python Code Sandbox", value=True, help="Allows the AI to execute Python code directly to solve math, data analysis, or algorithms.")
    
    custom_persona = st.text_area(
        "Custom Instructions / Persona", 
        value="You are BAOBAB AI, an expert, helpful assistant.",
        help="Define how the AI should behave or what role it should take."
    )

    st.markdown("---")
    st.header("🔍 Chat Navigation & Search")
    chat_search_query = st.text_input("Find in active chat", placeholder="Type keyword...", key="chat_search_key")
    
    if st.session_state.messages:
        st.subheader("📍 Jump to Message")
        user_msgs = [(idx, msg["content"]) for idx, msg in enumerate(st.session_state.messages) if msg["role"] == "user"]
        if user_msgs:
            for idx, text in user_msgs:
                label_text = text[:28] + ("..." if len(text) > 28 else "")
                st.markdown(f"👉 [{label_text}](#msg-{idx})")

    st.markdown("---")
    st.header("💾 Session History")
    session_name = st.text_input("Session Name", value="Session_1")
    if st.button("💾 Save Session"):
        if st.session_state.messages:
            save_payload = {
                "messages": [{k: v for k, v in msg.items() if k != "audio"} for msg in st.session_state.messages],
                "analytics": st.session_state.analytics
            }
            with open(os.path.join(SAVES_DIR, f"{session_name.strip()}.json"), "w", encoding="utf-8") as f:
                json.dump(save_payload, f, indent=2)
            st.success("Session saved!")
            st.rerun()

    saved_files = [f.replace(".json", "") for f in os.listdir(SAVES_DIR) if f.endswith(".json")]
    if saved_files:
        selected_file = st.selectbox("Load History", ["-- Select Session --"] + saved_files)
        if st.button("📂 Load Session"):
            if selected_file != "-- Select Session --":
                with open(os.path.join(SAVES_DIR, f"{selected_file}.json"), "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                st.session_state.messages = loaded_data.get("messages", [])
                st.session_state.analytics = loaded_data.get("analytics", st.session_state.analytics)
                st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.analytics = {"bilstm_latencies": [], "faiss_latencies": [], "total_requests": 0, "blocked_requests": 0}
        st.rerun()

    st.markdown("---")
    st.header("📊 Performance Analytics")
    col1, col2 = st.columns(2)
    col1.metric("Total Turns", st.session_state.analytics["total_requests"])
    col2.metric("Blocked", st.session_state.analytics["blocked_requests"])

# --- Local Interactive Sandbox Runner in Sidebar ---
with st.sidebar:
    st.markdown("---")
    st.header("💻 Quick Local Python Runner")
    with st.expander("Run Custom Python Snippet"):
        user_code = st.text_area("Python Code", value="import numpy as np\nprint(np.mean([10, 20, 30, 40]))")
        if st.button("Execute Snippet"):
            out = run_local_python_sandbox(user_code)
            st.code(out, language="text")

# --- Main Chat UI ---
search_active = bool(chat_search_query.strip())
matches_found = 0

for i, msg in enumerate(st.session_state.messages):
    is_match = search_active and chat_search_query.lower() in msg["content"].lower()
    if is_match:
        matches_found += 1
    
    if not search_active or is_match:
        st.markdown(f'<div id="msg-{i}"></div>', unsafe_allow_html=True)
        with st.chat_message(msg["role"]):
            if is_match:
                st.markdown(f"🔍 *Match found for '{chat_search_query}':*")
            st.markdown(msg["content"])
            if "metadata" in msg:
                st.caption(msg["metadata"])
            if "audio" in msg and msg["audio"] is not None:
                st.audio(msg["audio"], format="audio/mp3")

if search_active:
    st.info(f"🔎 Filtered results for '**{chat_search_query}**': Found **{matches_found}** matching message(s).")

st.markdown("---")

# --- Attachment Popover, Text Field, Live Audio Recorder & Send ---
input_col1, input_col2, input_col3 = st.columns([1, 6, 3])

with input_col1:
    with st.popover("➕", help="Add attachments"):
        tab_file, tab_photo = st.tabs(["📄 Upload", "📷 Camera"])
        with tab_file:
            uploaded_file = st.file_uploader("Upload File", type=["txt", "pdf", "docx", "pptx", "jpg", "jpeg", "png"], key="attachment_file")
            upload_category = st.text_input("Category", value="Custom Knowledge", key="file_cat")
            if st.button("📥 Attach File", key="btn_attach_file") and uploaded_file:
                ext_text = extract_text_from_file(uploaded_file, client_gemini=client)
                if ext_text:
                    st.session_state.custom_documents.append({
                        "id": len(all_active_documents) + 1,
                        "title": uploaded_file.name,
                        "category": upload_category.strip() or "Custom Knowledge",
                        "text": ext_text
                    })
                    st.success(f"Attached **{uploaded_file.name}**!")
        with tab_photo:
            camera_image = st.camera_input("Take photo", key="attachment_cam")
            photo_category = st.text_input("Category", value="Custom Knowledge", key="cam_cat")
            if st.button("📥 Attach Scan", key="btn_attach_cam") and camera_image:
                img = Image.open(camera_image)
                ext_text = client.models.generate_content(model=GEMINI_MODEL, contents=[img, "Extract text cleanly."]).text.strip()
                if ext_text:
                    st.session_state.custom_documents.append({
                        "id": len(all_active_documents) + 1,
                        "title": f"Scan_{int(time.time())}.jpg",
                        "category": photo_category.strip() or "Custom Knowledge",
                        "text": ext_text
                    })
                    st.success("Photo attached!")

with input_col2:
    user_input = st.text_input("Prompt", value=st.session_state.pending_input, placeholder="Ask BAOBAB AI or request Python computations...", label_visibility="collapsed", key="main_prompt_field")

with input_col3:
    st.audio_input("Record voice", label_visibility="collapsed", key="live_mic_recorder", on_change=transcribe_audio_callback)
    send_pressed = st.button("Send", type="primary", use_container_width=True)

# Process Prompt
if send_pressed:
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        st.session_state.pending_input = ""
        st.session_state.analytics["total_requests"] += 1
        
        is_safe, guardrail_msg = validate_user_input(user_input)
        if not is_safe:
            st.session_state.analytics["blocked_requests"] += 1
            st.error(f"🚫 **Input Blocked**: {guardrail_msg}")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Processing..."):
                # BiLSTM Language Detection
                t0_bilstm = time.perf_counter()
                seq = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=50)
                preds = model.predict(padded)
                detected_lang = label_encoder.inverse_transform([np.argmax(preds)])[0]
                confidence = float(np.max(preds)) * 100
                st.session_state.analytics["bilstm_latencies"].append(round((time.perf_counter() - t0_bilstm) * 1000, 2))

                # FAISS Retrieval
                t0_faiss = time.perf_counter()
                embedding_dim = 128
                active_faiss_index, active_docs = build_faiss_index_for_docs(all_active_documents, tokenizer, embedding_dim)
                rag_context, doc_title = "No relevant document found.", "None"
                
                if active_faiss_index is not None:
                    q_vec = np.zeros(embedding_dim, dtype='float32')
                    if seq[0]:
                        for idx in seq[0][:embedding_dim]:
                            q_vec[idx % embedding_dim] += 1.0
                    if np.sum(q_vec) > 0:
                        q_matrix = np.array([q_vec]).astype('float32')
                        faiss.normalize_L2(q_matrix)
                        distances, indices = active_faiss_index.search(q_matrix, k=1)
                        if float(distances[0][0]) >= 0.35:
                            retrieved_doc = active_docs[indices[0][0]]
                            rag_context = retrieved_doc["text"]
                            doc_title = f"[{retrieved_doc['category']}] {retrieved_doc['title']}"
                st.session_state.analytics["faiss_latencies"].append(round((time.perf_counter() - t0_faiss) * 1000, 2))

                history_context = "".join([f"{m['role'].upper()}: {m['content']}\n" for m in st.session_state.messages[:-1]])

                prompt_sent = f"""SYSTEM INSTRUCTION:
{custom_persona}
Detected Input Language: {detected_lang} (Confidence: {confidence:.1f}%).
TARGET OUTPUT LANGUAGE: Compose your response strictly in {target_language}.

RETRIEVED CONTEXT:
{rag_context}

HISTORY:
{history_context if history_context else "None."}

QUERY: {user_input}
RESPONSE ({target_language}):"""

                # Configure tools (Search Grounding & Code Execution Sandbox)
                tools_list = []
                if enable_web_search:
                    tools_list.append({"google_search": {}})
                if enable_code_interpreter:
                    tools_list.append(types.Tool(code_execution=types.CodeExecution()))

                config = types.GenerateContentConfig(tools=tools_list) if tools_list else None

            # Stream response directly into UI
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                ai_output = ""
                
                try:
                    response_stream = client.models.generate_content_stream(
                        model=GEMINI_MODEL,
                        contents=prompt_sent,
                        config=config
                    )
                    for chunk in response_stream:
                        if chunk.text:
                            ai_output += chunk.text
                            message_placeholder.markdown(ai_output + "▌")
                    message_placeholder.markdown(ai_output)
                except Exception as e:
                    ai_output = f"Error: {str(e)}"
                    message_placeholder.markdown(ai_output)

            # TTS Audio Generation
            audio_bytes = None
            try:
                tts = gTTS(text=ai_output, lang=TTS_LANG_MAP.get(target_language, 'en'))
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                audio_bytes = fp.getvalue()
            except Exception:
                pass

            metadata = f"🧠 **{detected_lang}** ({confidence:.1f}%) | 📄 Context: *{doc_title}* | 💻 Sandbox: `{'On' if enable_code_interpreter else 'Off'}`"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_output, 
                "metadata": metadata,
                "audio": audio_bytes
            })
            st.rerun()
