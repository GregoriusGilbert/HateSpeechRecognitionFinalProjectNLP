import streamlit as st
import torch
import pandas as pd
import joblib
import os
import gdown
from transformers import BertTokenizer, BertForSequenceClassification

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Hate Speech Detector", page_icon="🚫", layout="wide")

st.markdown("""
    <style>
    /* Styling font dan background utama web */
    .stApp {
        background-color: #0f172a; /* Slate dark modern, gak bikin sakit mata */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Mengubah warna teks utama dan subheader */
    h1, h2, h3, p, span, label {
        color: #f8fafc !important;
    }
    
    /* Bikin text area input lebih aesthetic & smooth */
    textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 16px !important;
    }
    textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Custom tombol Detect biar gak kaku */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.3) !important;
    }
    
    /* Styling kotak Tab agar stylish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1e293b;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        border-radius: 6px;
        padding: 6px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155 !important;
        color: #38bdf8 !important;
        font-weight: bold !important;
    }
    
    /* Custom Card Box untuk hasil output */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        border-left: 6px solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .card-danger { background-color: rgba(239, 68, 68, 0.15); border-left-color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .card-warning { background-color: rgba(245, 158, 11, 0.15); border-left-color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .card-success { background-color: rgba(16, 185, 129, 0.15); border-left-color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    </style>
""", unsafe_allow_html=True)


# --- 2. CACHE MODEL LOADING ---
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

@st.cache_resource
def load_baseline_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    logreg_model = joblib.load('logistic_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    return vectorizer, logreg_model, svm_model, rf_model

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

# Navigasi utama dengan susunan 4 Tab
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Hate Speech Detector", 
    "⚙️ Model Pipeline", 
    "📊 Dataset Sample", 
    "👥 About Team"
])


# --- TAB 1: LOGIKA DETECTOR (MULTI-MODEL) ---
with tab1:
    st.subheader("Input Text Analysis")
    
    selected_model = st.radio(
        "Pilih Model Evaluasi:",
        ["BERT (Transformer)", "Random Forest", "SVM", "Logistic Regression"],
        horizontal=True
    )
    
    user_input = st.text_area("Sentence:", placeholder="Type a sentence to analyze (e.g., I hate this or Have a nice day)")

    if st.button("Detect"):
        if user_input.strip():
            prediction = None
            
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
            
            if prediction is not None:
                st.write("")
                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-card card-danger">
                        <h4 style="margin:0; color:#f87171 !important;">🚨 [{selected_model}] Hate Speech Detected</h4>
                        <p style="margin: 8px 0 0 0; color:#fca5a5 !important;">Kalimat teridentifikasi mengandung unsur ujaran kebencian yang menargetkan identitas/SARA.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown(f"""
                    <div class="result-card card-warning">
                        <h4 style="margin:0; color:#fbbf24 !important;">⚠️ [{selected_model}] Abusive / Offensive Language</h4>
                        <p style="margin: 8px 0 0 0; color:#fde047 !important;">Kalimat aman dari isu SARA, namun mengandung kata-kata kasar atau kurang pantas (profanity).</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == 2:
                    st.markdown(f"""
                    <div class="result-card card-success">
                        <h4 style="margin:0; color:#34d399 !important;">✅ [{selected_model}] Safe (Clean Text)</h4>
                        <p style="margin: 8px 0 0 0; color:#a7f3d0 !important;">Kalimat bersih. Tidak ditemukan unsur ujaran kebencian maupun kata-kata kasar.</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text first.")


# --- TAB 2: MODEL PIPELINE ---
with tab2:
    st.subheader("⚙️ Model Architecture & Pipeline")
    
    st.markdown("### **How the model works**")
    st.write("Sistem ini mengonstruksi pipeline NLP ujung-ke-ujung (end-to-end) yang membandingkan model statistik tradisional berbasis TF-IDF dengan arsitektur Deep Learning modern berbasis Transformer.")
    
    col_method1, col_method2 = st.columns([1, 1])
    
    with col_method1:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; min-height: 180px;">
            <strong style="color: #38bdf8; font-size: 18px;">TF-IDF Vectorization & Baseline</strong>
            <p style="color: #94a3b8; font-size: 14px; margin-top: 8px;">
                Mengekstrak fitur berbasis frekuensi statistik kata (Term Frequency-Inverse Document Frequency) dari 36k tweet. Fitur matriks ini kemudian dialirkan langsung ke model efisien seperti SVM, Random Forest, dan Logistic Regression.
            </p>
            <code style="color: #38bdf8; background-color: #0f172a; padding: 4px 8px; border-radius: 4px; font-size: 12px;">ngram_range=(1,2)</code>
        </div>
        """, unsafe_allow_html=True)
        
    with col_method2:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; min-height: 180px;">
            <strong style="color: #a78bfa; font-size: 18px;">BERT Transformer Architecture</strong>
            <p style="color: #94a3b8; font-size: 14px; margin-top: 8px;">
                Menggunakan <em>bert-base-uncased</em> dengan mekanisme Self-Attention untuk menangkap konteks semantik kalimat secara dua arah (bidirectional). Dilatih menggunakan akselerasi GPU Google Colab.
            </p>
            <code style="color: #a78bfa; background-color: #0f172a; padding: 4px 8px; border-radius: 4px; font-size: 12px;">num_labels=3, max_length=128</code>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### **Model Pipeline**")
    
    pipe1, pipe2, pipe3, pipe4, pipe5 = st.columns(5)
    
    with pipe1:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: bold; text-transform: uppercase;">Step 1</span>
            <strong style="color: #f8fafc; display: block; font-size: 14px; margin-top: 4px;">Load Data</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px;">Membaca 36,340 data tweet ter-merge dan diacak aman.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pipe2:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: bold; text-transform: uppercase;">Step 2</span>
            <strong style="color: #f8fafc; display: block; font-size: 14px; margin-top: 4px;">Preprocessing</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px;">Case folding, cleaning tokenisasi, & text padding.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pipe3:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: bold; text-transform: uppercase;">Step 3</span>
            <strong style="color: #f8fafc; display: block; font-size: 14px; margin-top: 4px;">Fit Vectors</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px;">Ekstraksi bobot TF-IDF & subword BERT Tokenizer.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pipe4:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: bold; text-transform: uppercase;">Step 4</span>
            <strong style="color: #f8fafc; display: block; font-size: 14px; margin-top: 4px;">Train Models</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px;">Bake ML ke .pkl & BERT di-train via GPU Colab.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pipe5:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: bold; text-transform: uppercase;">Step 5</span>
            <strong style="color: #f8fafc; display: block; font-size: 14px; margin-top: 4px;">Rank & Return</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px;">Akurasi diuji bersama, tembus skor tertinggi 94%.</p>
        </div>
        """, unsafe_allow_html=True)


# --- TAB 3: DATASET OVERVIEW & VISUALIZATION ---
with tab3:
    st.subheader("📊 Dataset Overview")
    
    col_data1, col_data2 = st.columns([4, 3])
    
    with col_data1:
        st.markdown("#### **Class Distributions & Labels**")
        st.markdown("""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; min-height: 235px; margin-bottom: 10px;">
            <ul style="color: #f8fafc; list-style-type: none; padding-left: 0; margin: 0;">
                <li style="margin-bottom: 14px;"><strong style="color: #f87171;">🔴 Class 0 - Hate Speech:</strong> Tweet mengandung kebencian eksplisit terhadap SARA, ras, agama, atau identitas kelompok tertentu.</li>
                <li style="margin-bottom: 14px;"><strong style="color: #fbbf24;">🟡 Class 1 - Offensive Language:</strong> Tweet mengandung kata makian, kasar, atau kurang pantas (profanity) namun tidak bermaksud mendiskriminasi golongan SARA.</li>
                <li><strong style="color: #34d399;">🟢 Class 2 - Safe (Neither):</strong> Tweet netral atau teks bersih yang bebas dari unsur kata kasar ataupun ujaran kebencian.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_data2:
        st.markdown("#### **Visual Target Distribution**")
        chart_data = pd.DataFrame(
            [5462, 23640, 7238], 
            index=['Class 0 (Hate Speech)', 'Class 1 (Offensive)', 'Class 2 (Safe)'], 
            columns=['Total Tweets']
        )
        st.bar_chart(chart_data, color="#38bdf8", use_container_width=True)

    st.markdown("#### **Preview Dataset Sample (36,340 Rows Total)**")
    st.write("Berikut adalah cuplikan data asli dari 10 baris pertama (Head) dan 10 baris terakhir (Tail) yang digunakan dalam proses training dan testing:")
    
    try:
        csv_path = 'data/merged_randomized_with_unnamed_sorted.csv'
        
        if os.path.exists(csv_path):
            full_df = pd.read_csv(csv_path)
            display_cols = ['tweet', 'class'] if 'tweet' in full_df.columns and 'class' in full_df.columns else full_df.columns
            
            df_head = full_df[display_cols].head(10)
            df_tail = full_df[display_cols].tail(10)
            
            sub_tab_head, sub_tab_tail = st.tabs(["🔝 First 10 Rows (Head)", "🔚 Last 10 Rows (Tail)"])
            
            with sub_tab_head:
                st.dataframe(df_head, use_container_width=True, hide_index=False)
                st.caption("💡 *Menampilkan data indeks awal pada dataset setelah di-merge.*")
                
            with sub_tab_tail:
                st.dataframe(df_tail, use_container_width=True, hide_index=False)
                st.caption("💡 *Menampilkan data indeks akhir pada dataset setelah di-merge.*")
        else:
            st.warning("⚠️ File dataset tidak ditemukan di path `data/merged_randomized_with_unnamed_sorted.csv`. Pastiin file CSV sudah lo push di folder data.")
            
    except Exception as e:
        st.error(f"Gagal memuat cuplikan dataset: {e}")


# --- TAB 4: ABOUT TEAM ---
with tab4:
    st.subheader("👥 About Team & Background")
    
    col_team1, col_team2 = st.columns([3, 2])
    
    with col_team1:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 25px; border-radius: 12px; border: 1px solid #334155; min-height: 330px; text-align: justify;">
            <strong style="color: #38bdf8; font-size: 20px; display: block; margin-bottom: 12px;">📌 Latar Belakang Proyek</strong>
            <p style="color: #cbd5e1; font-size: 14px; line-height: 1.7; margin-bottom: 12px;">
                Proyek ini diinisiasi berdasarkan realita digital saat ini, di mana kita semakin sering menemui maraknya penggunaan kata-kata kasar (abusive language) hingga ujaran kebencian (hate speech) yang tersebar bebas di berbagai platform media sosial, forum online, hingga komunitas gaming. 
            </p>
            <p style="color: #cbd5e1; font-size: 14px; line-height: 1.7; margin-bottom: 12px;">
                Berawal dari keresahan bersama melihat fenomena normalisasi toksisitas digital tersebut, kami berpikir: <em>"Mengapa kita tidak membangun sistem otomatis yang mampu memitigasi dan mendeteksi hal ini secara cerdas?"</em>
            </p>
            <p style="color: #cbd5e1; font-size: 14px; line-height: 1.7;">
                Melalui momentum final assignment mata kuliah <strong>Natural Language Processing</strong> ini, Kelompok 7 LA01 berkomitmen untuk mengembangkan sebuah sistem klasifikasi teks berbasis AI. Dengan memanfaatkan kekuatan model klasifikasi statistik tradisional serta melakukan fine-tuning pada arsitektur modern <strong>BERT (Bidirectional Encoder Representations from Transformers)</strong>, aplikasi ini diharapkan mampu mengidentifikasi batasan konteks yang jelas antara teks yang aman, kata makian biasa, dan ujaran kebencian berbahaya yang menargetkan identitas/SARA demi mendukung terciptanya ekosistem digital yang lebih sehat.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_team2:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 25px; border-radius: 12px; border: 1px solid #334155; min-height: 330px;">
            <strong style="color: #a78bfa; font-size: 20px; display: block; margin-bottom: 15px;">👥 Anggota Kelompok 7</strong>
            <p style="color: #94a3b8; font-size: 13px; margin-bottom: 15px; font-style: italic;">
                Kelas LA01 - Natural Language Processing
            </p>
            <ul style="color: #f8fafc; padding-left: 20px; line-height: 2.2; font-size: 15px;">
                <li><strong>Gregorius Gilbert Susanto</strong><br><span style="color: #94a3b8; font-size: 13px;">(2802420031)</span></li>
                <li><strong>Willian Yehezkiel Alvin</strong><br><span style="color: #94a3b8; font-size: 13px;">(2802419811)</span></li>
                <li><strong>Vittorio Dinata</strong><br><span style="color: #94a3b8; font-size: 13px;">(2802427542)</span></li>
                <li><strong>Andrew Ong</strong><br><span style="color: #94a3b8; font-size: 13px;">(2802420561)</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- 5. FOOTER  ---
st.markdown("""
    <br><hr style="border-color: #334155;"><br>
    <div style="text-align: center; color: #64748b; font-size: 12px; padding-bottom: 20px;">
        © 2026 Kelompok 7 LA01 - Hate Speech Recognition System. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)