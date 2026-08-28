import streamlit as st
import pickle
import numpy as np

# Set Streamlit Page Config
st.set_page_config(
    page_title="Multilingual African Language AI",
    page_icon="🌍",
    layout="centered"
)

# ==========================================
# 1. LOAD RE-EXPORTED MODEL ARTIFACTS (CELL 7)
# ==========================================
@st.cache_resource
def load_artifacts():
    with open('african_lang_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

try:
    model, vectorizer, label_encoder = load_artifacts()
    st.sidebar.success("Model artifacts loaded successfully! (~12MB)")
except Exception as e:
    st.error(f"Error loading model artifacts from Cell 7: {e}")
    st.stop()

# Language ISO Mapping for Clean Display UI
LANG_NAMES = {
    'amh': 'Amharic (አማርኛ)',
    'fra': 'French (Français)',
    'hau': 'Hausa (Harshen Hausa)',
    'ibo': 'Igbo (Asụsụ Igbo)',
    'lin': 'Lingala (Lingála)',
    'lug': 'Luganda (Oluganda)',
    'orm': 'Oromo (Afaan Oromoo)',
    'pcm': 'Nigerian Pidgin',
    'run': 'Kirundi (Ikirundi)',
    'sna': 'Shona (chiShona)',
    'som': 'Somali (Soomaaliga)',
    'swa': 'Swahili (Kiswahili)',
    'tir': 'Tigrinya (ትግርኛ)',
    'xho': 'isiXhosa',
    'yor': 'Yoruba (Èdè Yorùbá)'
}

# ==========================================
# 2. STREAMLIT USER INTERFACE
# ==========================================
st.title("🌍 African Language Classifier")
st.write("Enter text in any supported African language to detect its origin and confidence score.")

user_input = st.text_area(
    "Input Text:",
    height=150,
    placeholder="e.g., Habari gani rafiki wangu or Ina kwana da fatan ka tashi lafiya..."
)

if st.button("Detect Language", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        # Preprocess & Vectorize
        text_vec = vectorizer.transform([user_input.lower()])
        
        # Predict Probabilities
        probabilities = model.predict_proba(text_vec)[0]
        top_idx = np.argmax(probabilities)
        top_confidence = probabilities[top_idx] * 100
        predicted_code = label_encoder.classes_[top_idx]
        display_name = LANG_NAMES.get(predicted_code, predicted_code.upper())
        
        # Output Main Result
        st.markdown("---")
        st.subheader("Classification Result")
        st.metric(label="Predicted Language", value=display_name)
        st.progress(float(probabilities[top_idx]))
        st.write(f"**Confidence:** `{top_confidence:.2f}%`")
        
        # Display Top 3 Predictions Breakdown
        st.markdown("---")
        st.subheader("Top Prediction Probabilities")
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        for idx in top_3_indices:
            code = label_encoder.classes_[idx]
            name = LANG_NAMES.get(code, code.upper())
            score = probabilities[idx] * 100
            st.write(f"- **{name}**: `{score:.2f}%`")
