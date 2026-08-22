# 🌍 African Multilingual AI Assistant

An end-to-end NLP and Generative AI application capable of classifying user text across **16 languages** (with focus on African languages) using a custom Deep Learning model, and routing responses through Google Gemini enforcing target output languages.

Live App: [African Multilingual AI Assistant](https://african-multilingual-ai-wjnlgrybsnpnbyr4kvbm2h.streamlit.app/)

---

## 📐 Architecture & Workflow

1. **Classification Brain**: Trained BiLSTM network (TensorFlow/Keras) classifying text across 16 languages using word-level tokenization.
2. **Orchestrator Layer**: Extracts detected language metadata and confidence scores to craft dynamic system prompts.
3. **Generative LLM**: Google Gemini API (`gemini-3.6-flash`) for target-language response generation.
4. **Deployment**: Streamlit Community Cloud integrated with GitHub version control.

---

## 📁 Repository Structure

* `app.py`: Streamlit web UI and backend orchestration pipeline.
* `african_lang_classifier.keras`: Saved trained BiLSTM neural network model.
* `word_tokenizer.pkl`: Pickled tokenizer for input sequence vectorization.
* `label_encoder.pkl`: Pickled target class encoder for 16 language classes.
* `requirements.txt`: Python package dependencies.
