import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Hate Speech Detector", page_icon="🚫", layout="wide")

# --- 2. CACHE MODEL LOADING ---
@st.cache_resource
def load_model():
    model_name = 'bert-base-uncased' 
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Using 3 labels based on your .pt file mismatch error
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Load weights
    model.load_state_dict(torch.load('model_bert_hatespeech.pt', map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, model

# Initialize variables
tokenizer, model = None, None

# Load model
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")


# --- 3. SIDEBAR LEFT (Persis kayak image_f0d39e.png) ---
with st.sidebar:
    st.header("📊 Dataset Information")
    st.metric(label="Total Dataset Tweets", value="24,783")
    st.metric(label="Features/Classes", value="3 Labels")
    st.write("---")
    st.subheader("🤖 Machine Learning Model")
    st.markdown("- **Architecture:** BERT (bert-base-uncased)")
    st.markdown("- **Evaluation Accuracy:** **0.91**")


# --- 4. MAIN INTERFACE & NAVIGATION TABS ---
st.title("🚫 Hate Speech Recognition")
st.write("Aplikasi analisis otomatis ujaran kebencian dan kata kasar menggunakan arsitektur Transformer.")

# Bikin menu tab tengah ala image_f0d39e.png
tab1, tab2, tab3 = st.tabs(["🔍 Hate Speech Detector", "📊 Dataset & Methodology", "👥 About Team"])


# --- TAB 1: LOGIKA DETECTOR LO ---
with tab1:
    st.subheader("Input Text Analysis")
    user_input = st.text_area("Sentence:", placeholder="Type a sentence to analyze (e.g., I hate this or Have a nice day)")

    if st.button("Detect"):
        if not tokenizer or not model:
            st.error("Model or Tokenizer not properly loaded.")
        elif user_input.strip():
            # Tokenize input
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
            
            # Mapping output label
            if prediction == 0:
                st.error(f"Result: **Hate Speech Detected** (Targeting Identity)")
            elif prediction == 1:
                st.warning(f"Result: **Abusive / Offensive Language** (Profanity)")
            else:
                st.success(f"Result: **Safe (Non-Hate Speech)**")
        else:
            st.warning("Please enter some text first.")


# --- TAB 2: DATASET OVERVIEW ---
with tab2:
    st.subheader("Dataset Overview")
    st.write("Project ini dilatih menggunakan dataset klasifikasi sentimen ujaran kebencian dengan rincian kelas sebagai berikut:")
    st.markdown("""
    *   **Class 0 - Hate Speech:** Tweet yang mengandung kebencian terhadap SARA atau identitas tertentu.
    *   **Class 1 - Offensive Language:** Tweet yang menggunakan kata-kata kasar/kasar namun tidak menargetkan kelompok tertentu.
    *   **Class 2 - Neither:** Tweet aman/netral yang tidak mengandung unsur kasar maupun ujaran kebencian.
    """)


# --- TAB 3: ABOUT TEAM ---
with tab3:
    st.subheader("About Team")
    st.write("Project ini dikembangkan oleh Tim Mahasiswa BINUS University untuk final assignment mata kuliah Natural Language Processing.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Anggota Tim:**
        *   Gregorius Gilbert Susanto - 2802420031
        *   Willian Yehezkiel Alvin	- 2802419811
	    *   Vittorio Dinata - 2802427542
        *   Andrew Ong - 2802420561
        """)