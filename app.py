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
    /* Mengimpor Font Inter ala Web Premium */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #0b0f19 !important; /* Deep space dark */
        font-family: 'Inter', sans-serif;
    }
    
    /* Global Typography Reset */
    h1, h2, h3, h4, p, span, label, .stMetric div {
        color: #f8fafc !important;
    }
    
    /* Gedein Judul Utama ala Spotify Web */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.05em !important;
        background: linear-gradient(135deg, #fff 40%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px !important;
    }
    
    .main-subtitle {
        font-size: 1.15rem !important;
        color: #94a3b8 !important;
        font-weight: 400;
        margin-bottom: 35px !important;
        line-height: 1.6;
    }

    /* Badges / Chips Horizontal Style */
    .badge-container {
        display: flex;
        gap: 10px;
        margin-bottom: 25px;
        flex-wrap: wrap;
    }
    .custom-badge {
        background-color: #1e293b;
        color: #38bdf8 !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #334155;
    }
    .custom-badge-purple {
        background-color: #1e293b;
        color: #a78bfa !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #334155;
    }

    /* Modern Navigation Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent !important;
        padding: 0px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
        background-color: transparent !important;
        border-radius: 0px !important;
        padding: 10px 4px !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Container Box / Cards Grid Modern Layout */
    .premium-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .premium-card-title {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 8px !important;
    }
    .premium-card-desc {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        line-height: 1.6;
    }

    /* Input Text Area Container Modernization */
    textarea {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #1f2937 !important;
        border-radius: 14px !important;
        padding: 16px !important;
        font-size: 16px !important;
    }
    textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.15) !important;
    }
    
    /* Ganti Radio Button Biar Gak Kaku */
    .stRadio div[role="radiogroup"] {
        gap: 15px;
    }

    /* Glowing Predict/Detect Button ala Premium App */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.25s ease-out !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
        width: auto !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 20px rgba(56, 189, 248, 0.4) !important;
    }
    
    /* Output Result Cards Dynamic */
    .result-card {
        padding: 24px;
        border-radius: 16px;
        margin-top: 25px;
        border-left: 6px solid;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .card-danger { background: rgba(239, 68, 68, 0.08); border-left-color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2); }
    .card-warning { background: rgba(245, 158, 11, 0.08); border-left-color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.2); }
    .card-success { background: rgba(16, 185, 129, 0.08); border-left-color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); }
    </style>
""", unsafe_allow_html=True)


# --- 2. CACHE MODEL LOADING ---
@st.cache_resource
def load_bert_model():
    model_name = 'bert-base-uncased' 
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    model_path = 'model_bert_hatespeech.pt'
    
    if os.path.exists(model_path) and os.path.getsize(model_path) < 100 * 1024 * 1024:
        os.remove(model_path)
    
    if not os.path.exists(model_path):
        with st.spinner("📥 Fetching heavy Transformer weights from secure storage... Please wait."):
            id_drive = "1j-ALfhIH0L9gIkSFpggOicTYyZSOrXqU" 
            url = f'https://drive.google.com/uc?id={id_drive}'
            gdown.download(url, model_path, quiet=False, fuzzy=True)
    
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


# --- 3. SIDEBAR MODERN PANEL ---
with st.sidebar:
    st.markdown("<h2 style='font-weight:700; font-size:1.4rem; margin-bottom:20px;'>📊 Analytics Core</h2>", unsafe_allow_html=True)
    st.metric(label="Total Corrupted Datasets Enrolled", value="36,340")
    st.metric(label="Categorical Targeted Classes", value="3 Labels")
    st.markdown("<hr style='border-color: #1f2937;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-weight:600; font-size:1.05rem; margin-bottom:12px;'>🤖 Leaderboard Model Accuracy</h4>", unsafe_allow_html=True)
    st.markdown("- **BERT Architecture:** `0.94` 🔥")
    st.markdown("- **Random Forest Baseline:** `0.94` 📈")
    st.markdown("- **Support Vector Machine:** `0.90` 📝")
    st.markdown("- **Logistic Regression:** `0.88` 📋")


# --- 4. MAIN INTERFACE HEADERS ---
st.title("🚫 Hate Speech Recognition")
st.markdown("<p class='main-subtitle'>Aplikasi analisis otomatis ujaran kebencian dan kata kasar menggunakan perbandingan arsitektur Machine Learning tradisional dan Transformer dua arah.</p>", unsafe_allow_html=True)

st.markdown("""
    <div class='badge-container'>
        <div class='custom-badge'>⚡ 36,340 Tweets</div>
        <div class='custom-badge'>🌐 BERT Base Uncased</div>
        <div class='custom-badge-purple'>🛠️ TF-IDF Vectorizer</div>
        <div class='custom-badge-purple'>📊 4 Architecture Enrolled</div>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Hate Speech Detector", 
    "⚙️ Model Pipeline", 
    "📊 Dataset Sample", 
    "👥 About Team"
])


# --- TAB 1: DETECTOR INTERFACE ---
with tab1:
    st.markdown("<h3 style='font-weight:700; font-size:1.5rem; margin-bottom:15px;'>Search & Analyze Sentence</h3>", unsafe_allow_html=True)
    
    selected_model = st.radio(
        "Pilih Model Evaluasi:",
        ["BERT (Transformer)", "Random Forest", "SVM", "Logistic Regression"],
        horizontal=True
    )
    
    st.write("")
    user_input = st.text_area("Sentence Input Box:", placeholder="Type a sentence to analyze (e.g., I hate this or Have a nice day to everyone)")

    st.write("")
    if st.button("✨ Get Analytics Evaluation"):
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
            
            if prediction is not None:
                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-card card-danger">
                        <h4 style="margin:0; font-weight:700; color:#f87171 !important;">🚨 [{selected_model}] Hate Speech Detected</h4>
                        <p style="margin: 8px 0 0 0; color:#cbd5e1 !important;">Kalimat teridentifikasi kuat mengandung unsur ujaran kebencian aktif yang menyerang kelompok/SARA.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown(f"""
                    <div class="result-card card-warning">
                        <h4 style="margin:0; font-weight:700; color:#fbbf24 !important;">⚠️ [{selected_model}] Abusive / Offensive Language</h4>
                        <p style="margin: 8px 0 0 0; color:#cbd5e1 !important;">Aman dari isu SARA, namun sistem mendeteksi adanya penggunaan kata makian kasar (profanity).</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == 2:
                    st.markdown(f"""
                    <div class="result-card card-success">
                        <h4 style="margin:0; font-weight:700; color:#34d399 !important;">✅ [{selected_model}] Safe (Clean Text)</h4>
                        <p style="margin: 8px 0 0 0; color:#cbd5e1 !important;">Teks bersih. Tidak ditemukan indikasi kata kasar maupun ujaran kebencian di dalam kalimat.</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text first.")


# --- TAB 2: MODEL PIPELINE  ---
with tab2:
    st.markdown("<h3 style='font-weight:700; font-size:1.5rem; margin-bottom:10px;'>⚙️ Core Architecture Details</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:25px;'>Sistem membandingkan model statistik tradisional berbasis TF-IDF dengan arsitektur Deep Learning modern berbasis Transformer.</p>", unsafe_allow_html=True)
    
    col_method1, col_method2 = st.columns(2)
    
    with col_method1:
        st.markdown("""
        <div class="premium-card">
            <div class="premium-card-title">📐 TF-IDF Vectorization & Baselines</div>
            <p class="premium-card-desc">
                Mengekstrak fitur berbasis frekuensi statistik kata (Term Frequency-Inverse Document Frequency) dari 36k tweet. Fitur matriks ini dialirkan langsung ke model efisien seperti SVM, Random Forest, dan Logistic Regression.
            </p>
            <code style="color: #38bdf8; background: #0b0f19; padding: 4px 10px; border-radius: 6px; font-size: 11px; border:1px solid #1f2937;">max_features=5000</code>
        </div>
        """, unsafe_allow_html=True)
        
    with col_method2:
        st.markdown("""
        <div class="premium-card">
            <div class="premium-card-title">🧠 BERT Transformer Deep Learning</div>
            <p class="premium-card-desc">
                Menggunakan tokenisasi subword <em>bert-base-uncased</em> dengan mekanisme Self-Attention untuk menangkap konteks semantik kalimat secara dua arah (bidirectional) untuk hasil presisi tinggi.
            </p>
            <code style="color: #a78bfa; background: #0b0f19; padding: 4px 10px; border-radius: 6px; font-size: 11px; border:1px solid #1f2937;">num_labels=3, max_length=128</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='font-weight:700; font-size:1.4rem; margin-top:20px; margin-bottom:20px;'>Model Processing Pipeline</h3>", unsafe_allow_html=True)
    
    pipe1, pipe2, pipe3, pipe4, pipe5 = st.columns(5)
    
    with pipe1:
        st.markdown("""
        <div style="background: #111827; border:1px solid #1f2937; padding: 20px; border-radius: 12px; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing:0.05em;">Step 1</span>
            <strong style="color: #ffffff; display: block; font-size: 15px; margin-top: 4px; font-weight:700;">Load Dataset</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px; line-height:1.5;">Membaca 36,340 baris tweet ter-merge acak.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pipe2:
        st.markdown("""
        <div style="background: #111827; border:1px solid #1f2937; padding: 20px; border-radius: 12px; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing:0.05em;">Step 2</span>
            <strong style="color: #ffffff; display: block; font-size: 15px; margin-top: 4px; font-weight:700;">Text Cleaning</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px; line-height:1.5;">Case folding, filter regex, url & alphanumeric removal.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pipe3:
        st.markdown("""
        <div style="background: #111827; border:1px solid #1f2937; padding: 20px; border-radius: 12px; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing:0.05em;">Step 3</span>
            <strong style="color: #ffffff; display: block; font-size: 15px; margin-top: 4px; font-weight:700;">Extraction</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px; line-height:1.5;">Fitting bobot TF-IDF & subword Tokenizer BERT.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pipe4:
        st.markdown("""
        <div style="background: #111827; border:1px solid #1f2937; padding: 20px; border-radius: 12px; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing:0.05em;">Step 4</span>
            <strong style="color: #ffffff; display: block; font-size: 15px; margin-top: 4px; font-weight:700;">Bake Models</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px; line-height:1.5;">Simpan baseline ke .pkl & training BERT via GPU.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pipe5:
        st.markdown("""
        <div style="background: #111827; border:1px solid #1f2937; padding: 20px; border-radius: 12px; min-height: 140px;">
            <span style="color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing:0.05em;">Step 5</span>
            <strong style="color: #ffffff; display: block; font-size: 15px; margin-top: 4px; font-weight:700;">Rank Evaluation</strong>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px; line-height:1.5;">Validasi skor akurasi tertinggi sukses tembus 94%.</p>
        </div>
        """, unsafe_allow_html=True)


# --- TAB 3: DATASET OVERVIEW ---
with tab3:
    st.markdown("<h3 style='font-weight:700; font-size:1.5rem; margin-bottom:15px;'>📊 Data Target Exploration</h3>", unsafe_allow_html=True)
    
    col_data1, col_data2 = st.columns([4, 3])
    
    with col_data1:
        st.markdown("""
        <div style="background-color: #111827; padding: 24px; border-radius: 16px; border: 1px solid #1f2937; min-height: 250px;">
            <h4 style='font-weight:700; font-size:1.1rem; margin-top:0; margin-bottom:15px;'>Target Classification Definitions:</h4>
            <ul style="color: #f8fafc; list-style-type: none; padding-left: 0; margin: 0; line-height:1.7;">
                <li style="margin-bottom: 12px;"><strong style="color: #f87171;">🔴 Class 0 - Hate Speech:</strong> Ujaran kebencian rasial, SARA, agama, atau kelompok spesifik.</li>
                <li style="margin-bottom: 12px;"><strong style="color: #fbbf24;">🟡 Class 1 - Offensive Language:</strong> Penggunaan profanity / kata kasar verbal non-SARA.</li>
                <li><strong style="color: #34d399;">🟢 Class 2 - Safe Teks:</strong> Kalimat bersih, netral, dan aman untuk dikonsumsi publik.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_data2:
        chart_data = pd.DataFrame(
            [5462, 23640, 7238], 
            index=['Class 0 (Hate Speech)', 'Class 1 (Offensive)', 'Class 2 (Safe)'], 
            columns=['Total Tweets']
        )
        st.bar_chart(chart_data, color="#38bdf8", use_container_width=True)

    st.markdown("<h4 style='font-weight:700; margin-top:30px;'>Preview Dataset Sample</h4>", unsafe_allow_html=True)
    
    try:
        csv_path = 'data/merged_randomized_with_unnamed_sorted.csv'
        if os.path.exists(csv_path):
            full_df = pd.read_csv(csv_path)
            display_cols = ['tweet', 'class'] if 'tweet' in full_df.columns and 'class' in full_df.columns else full_df.columns
            
            df_head = full_df[display_cols].head(10)
            df_tail = full_df[display_cols].tail(10)
            
            sub_tab_head, sub_tab_tail = st.tabs(["🔝 First 10 Rows (Head)", "🔚 Last 10 Rows (Tail)"])
            with sub_tab_head:
                st.dataframe(df_head, use_container_width=True)
            with sub_tab_tail:
                st.dataframe(df_tail, use_container_width=True)
        else:
            st.warning("⚠️ File CSV data tidak ditemukan.")
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")


# --- TAB 4: ABOUT TEAM ---
with tab4:
    st.markdown("<h3 style='font-weight:700; font-size:1.5rem; margin-bottom:15px;'>👥 Project Ownership</h3>", unsafe_allow_html=True)
    
    col_team1, col_team2 = st.columns([7, 5])
    
    with col_team1:
        st.markdown("""
        <div style="background-color: #111827; padding: 25px; border-radius: 16px; border: 1px solid #1f2937; min-height: 310px; text-align: justify;">
            <strong style="color: #38bdf8; font-size: 1.25rem; display: block; margin-bottom: 12px; font-weight:700;">📌 Background Story</strong>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.7; margin-bottom: 12px;">
                Proyek ini diinisiasi berdasarkan realita digital saat ini, di mana maraknya penggunaan kata-kata kasar (abusive language) hingga ujaran kebencian (hate speech) yang tersebar bebas di berbagai platform media sosial, forum online, hingga komunitas gaming. Berawal dari keresahan bersama melihat fenomena normalisasi toksisitas digital tersebut, kami berpikir untuk membangun sistem otomatis yang mampu memitigasi hal ini secara cerdas.
            </p>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.7;">
                Melalui final assignment mata kuliah <strong>Natural Language Processing</strong> ini, Kelompok 7 LA01 berkomitmen mengembangkan sistem klasifikasi teks berbasis AI yang mampu mengidentifikasi batasan konteks jelas antara teks aman, makian biasa, dan ujaran kebencian berbahaya SARA demi ekosistem digital yang lebih sehat.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_team2:
        st.markdown("""
        <div style="background-color: #111827; padding: 25px; border-radius: 16px; border: 1px solid #1f2937; min-height: 310px;">
            <strong style="color: #a78bfa; font-size: 1.25rem; display: block; margin-bottom: 12px; font-weight:700;">👥 Group 7 Engineering</strong>
            <p style="color: #94a3b8; font-size: 13px; margin-bottom: 15px; font-style: italic;">Kelas LA01 - Natural Language Processing</p>
            <ul style="color: #f8fafc; padding-left: 15px; line-height: 2.2; font-size: 15px; list-style-type: square;">
                <li><strong>Gregorius Gilbert Susanto</strong> <span style="color: #64748b; font-size: 13px;">(2802420031)</span></li>
                <li><strong>Willian Yehezkiel Alvin</strong> <span style="color: #64748b; font-size: 13px;">(2802419811)</span></li>
                <li><strong>Vittorio Dinata</strong> <span style="color: #64748b; font-size: 13px;">(2802427542)</span></li>
                <li><strong>Andrew Ong</strong> <span style="color: #64748b; font-size: 13px;">(2802420561)</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- 5. PREMIUM FOOTER ---
st.markdown("""
    <br><br><hr style="border-color: #1f2937;"><br>
    <div style="text-align: center; color: #4b5563; font-size: 12px; padding-bottom: 20px; font-weight:500;">
        © 2026 Kelompok 7 LA01 — Hate Speech Recognition System Dashboard. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)