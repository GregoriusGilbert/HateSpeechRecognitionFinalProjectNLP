import streamlit as st
import torch
import pandas as pd
import joblib
import os
import gdown
from transformers import BertTokenizer, BertForSequenceClassification

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Hate Speech Detector", page_icon="🚫", layout="wide")

# --- 2. CACHE MODEL LOADING ---

# A. Load Model BERT (Transformer) dari Google Drive
@st.cache_resource
def load_bert_model():
    model_name = 'bert-base-uncased' 
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    model_path = 'model_bert_hatespeech.pt'
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading BERT Model from Google Drive (418 MB)... Please wait."):
            id_drive = "1j-ALfhIH0L9gIkSFpggOicTYyZSOrXqU" 
            url = f'https://drive.google.com/uc?id={id_drive}'
            gdown.download(url, model_path, quiet=False)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    return tokenizer, model

# B. Load Model Baseline (ML Biasa via .pkl)
@st.cache_resource
def load_baseline_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    logreg_model = joblib.load('logistic_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    return vectorizer, logreg_model, svm_model, rf_model

# Initialize & Load All Models
tokenizer, bert_model = None, None
vectorizer, logreg_model, svm_model, rf_model = None, None, None, None

try:
    tokenizer, bert_model = load_bert_model()
    vectorizer, logreg_model, svm_model, rf_model = load_baseline_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")


# --- 3. SIDEBAR LEFT ---
with st.sidebar:
    st.header("📊 Dataset Information")
    st.metric(label="Total Dataset Tweets", value="36,340")
    st.metric(label="Features/Classes", value="3 Labels")
    st.write("---")
    st.subheader("🤖 Models Comparison (Accuracy)")
    st.markdown("- **BERT (Transformer):** `0.94` 🔥")
    st.markdown("- **Random Forest:** `0.94` 📈")
    st.markdown("- **SVM:** `0.90` 📝")
    st.markdown("- **Logistic Regression:** `0.88` 📋")


# --- 4. MAIN INTERFACE & NAVIGATION TABS ---
st.title("🚫 Hate Speech Recognition")
st.write("Aplikasi analisis otomatis ujaran kebencian dan kata kasar menggunakan perbandingan arsitektur Machine Learning dan Transformer.")

tab1, tab2, tab3 = st.tabs(["🔍 Hate Speech Detector", "📊 Dataset & Methodology", "👥 About Team"])


# --- TAB 1: LOGIKA DETECTOR (MULTI-MODEL) ---
with tab1:
    st.subheader("Input Text Analysis")
    
    # Fitur Interaktif: Pilih Model yang mau dipakai tes
    selected_model = st.radio(
        "Pilih Model Evaluasi:",
        ["BERT (Transformer)", "Random Forest", "SVM", "Logistic Regression"],
        horizontal=True
    )
    
    user_input = st.text_area("Sentence:", placeholder="Type a sentence to analyze (e.g., I hate this or Have a nice day)")

    if st.button("Detect"):
        if user_input.strip():
            prediction = None
            
            # --- PROSES PREDIKSI BERDASARKAN MODEL YANG DIPILIH ---
            if selected_model == "BERT (Transformer)":
                if tokenizer and bert_model:
                    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                        prediction = torch.argmax(outputs.logits, dim=1).item()
                else:
                    st.error("BERT Model or Tokenizer not properly loaded.")
                    
            else:
                if vectorizer:
                    text_vector = vectorizer.transform([user_input])
                    
                    if selected_model == "Logistic Regression" and logreg_model:
                        prediction = logreg_model.predict(text_vector)[0]
                    elif selected_model == "SVM" and svm_model:
                        prediction = svm_model.predict(text_vector)[0]
                    elif selected_model == "Random Forest" and rf_model:
                        prediction = rf_model.predict(text_vector)[0]
                    else:
                        st.error(f"Selected model ({selected_model}) is not loaded properly.")
                else:
                    st.error("TF-IDF Vectorizer is missing or not loaded.")
            
            # --- MAPPING & TAMPILAN OUTPUT LABEL ---
            if prediction is not None:
                if prediction == 0:
                    st.error(f"[{selected_model}] Result: **Hate Speech Detected** (Targeting Identity)")
                elif prediction == 1:
                    st.warning(f"[{selected_model}] Result: **Abusive / Offensive Language** (Profanity)")
                elif prediction == 2:
                    st.success(f"[{selected_model}] Result: **Safe (Non-Hate Speech)**")
        else:
            st.warning("Please enter some text first.")


# --- TAB 2: DATASET OVERVIEW ---
with tab2:
    st.subheader("Dataset Overview")
    st.write("Project ini dilatih menggunakan dataset klasifikasi sentimen ujaran kebencian dengan rincian kelas sebagai berikut:")
    st.markdown("""
    * **Class 0 - Hate Speech:** Tweet yang mengandung kebencian terhadap SARA atau identitas tertentu.
    * **Class 1 - Offensive Language:** Tweet yang menggunakan kata-kata kasar/kasar namun tidak menargetkan kelompok tertentu.
    * **Class 2 - Neither:** Tweet aman/netral yang tidak mengandung unsur kasar maupun ujaran kebencian.
    """)


# --- TAB 3: ABOUT TEAM ---
with tab3:
    st.subheader("About Team")
    st.write("Project ini dikembangkan oleh Tim Mahasiswa BINUS University untuk final assignment mata kuliah Natural Language Processing.")
    
    st.markdown("**Anggota Tim:**")
    st.write("- Gregorius Gilbert Susanto (2802420031)")
    st.write("- Willian Yehezkiel Alvin (2802419811)")
    st.write("- Vittorio Dinata (2802427542)")
    st.write("- Andrew Ong (2802420561)")