import streamlit as st
import pickle
import re
import string
import pandas as pd
import numpy as np
import random
import os
import html as _h

# --- Konfigurasi page ---
st.set_page_config(
    page_title="GitHub Issue Predictor",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJEKSI CUSTOM CSS UNTUK TAMPILAN PREMIUM ---
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Google Sans', sans-serif; }
    .main { background-color: #202225; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: #292b2f;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem !important;
        font-weight: 600 !important; 
        padding: 8px 0 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #40444b !important; }

    /* ── HERO ── */
    .hero-box {
        background: linear-gradient(135deg, #033c80 0%, #0054ba 50%, #006aeb 100%);
        border-radius: 24px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(26,107,60,0.25);
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: white;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-subtitle { font-size: 1.1rem; color: rgba(255,255,255,0.85); margin-top: 8px; }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.4);
        color: white;
        border-radius: 20px;
        padding: 4px 16px;
        font-size: 0.85rem;
        margin-top: 12px;
    }

    /* ── SECTION TITLE ── */
    .section-title { 
        font-family: 'Google Sans', serif;
        font-size: 1.8rem; 
        color: white; 
        padding-left: 16px;
        margin: 24px 0 20px 0; 
        position: relative;
    }
    .section-title::before {
        content: "";
        position: absolute;
        left: 0;
        top: 10%;
        bottom: 10%;
        width: 4px;
        background-color: #40444b; 
        border-radius: 50px;
    }
            
    .section-desc {
        background: #292b2f;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #40444b;
        color: white;
        line-height: 1.7;
        margin-bottom: 24px;
    }

    /* ── INFO CARDS ── */
    .info-card {
        background: #292b2f;
        border-radius: 14px;
        padding: 20px 22px;
        border: 1px solid #40444b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    .info-card h4 { color: white; margin-bottom: 8px; font-size: 1rem; }

    /* ── STEP BADGE ── */          
    .step-badge {
        display: inline-block;
        background: linear-gradient(135deg, #033c80, #0054ba);
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        line-height: 36px;
        text-align: center;
        font-weight: 900;
        font-size: 1rem;
        margin-right: 10px;
        vertical-align: middle;
    }

    /* ── BUTTONS ── */ 
    div.stButton > button {
        background: linear-gradient(135deg, red, #40444b);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-family: 'Nunito', sans-serif;
        font-weight: 800;
        font-size: 1rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 4px 16px rgba(26,107,60,0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26,107,60,0.4);
    }
    
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid #40444b;
    }
    .main .block-container { padding-bottom: 1rem !important; }
</style>  
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # 1. Deklarasi awal semua variabel agar tidak error jika file tidak ada
    data_splits = None
    lr_model = None
    mnb_model = None
    tuned_lr_model = None
    nb_biner_model = None
    svm_biner_model = None
    tfidf_vectorizer = None

    # 2. Setup path - gunakan __file__ untuk mendapatkan lokasi app.py
    # Kemudian naik 1 level ke project root, lalu akses folder data
    app_dir = os.path.dirname(os.path.abspath(__file__))  # D:\...\web
    project_root = os.path.dirname(app_dir)  # D:\...\Bug-Severity-Classification
    base_path = os.path.join(project_root, 'data', 'processed')

    # 3. Load secara independen satu per satu
    try:
        with open(os.path.join(base_path, 'data_splits_tfidf.pkl'), 'rb') as f:
            data_splits = pickle.load(f)
    except Exception:
        pass

    try:
        with open(os.path.join(base_path, 'model_lr_tuned_biner.pkl'), 'rb') as f:
            lr_model = pickle.load(f)
    except Exception:
        pass

    try:
        with open(os.path.join(base_path, 'model_nb_biner.pkl'), 'rb') as f:
            mnb_model = pickle.load(f)
    except Exception:
        pass

    try:
        with open(os.path.join(base_path, 'model_lr_tuned_biner.pkl'), 'rb') as f:
            tuned_lr_model = pickle.load(f)
    except Exception:
        pass

    try:
        with open(os.path.join(base_path, 'model_nb_biner.pkl'), 'rb') as f:
            nb_biner_model = pickle.load(f)
    except Exception:
        pass

    try:
        with open(os.path.join(base_path, 'model_svm_biner.pkl'), 'rb') as f:
            svm_biner_model = pickle.load(f)
    except Exception:
        pass

    try:
        with open(os.path.join(base_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
    except Exception:
        pass

    return data_splits, lr_model, mnb_model, tuned_lr_model, nb_biner_model, svm_biner_model, tfidf_vectorizer

# Unpacking 7 variabel (HARUS MATCH)
data_splits, lr_model, mnb_model, tuned_lr_model, nb_biner_model, svm_biner_model, tfidf_vectorizer = load_models()

# Label encoder (jika tersedia) untuk memastikan mapping label konsisten
label_encoder = None
if isinstance(data_splits, dict):
    label_encoder = data_splits.get('label_encoder')

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _map_class_to_label(class_value):
    """Map numeric/label class to UI label using label_encoder if available."""
    label = None
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        try:
            if isinstance(class_value, (int, np.integer)):
                label = label_encoder.classes_[int(class_value)]
            else:
                label = str(class_value)
        except Exception:
            label = str(class_value)
    else:
        # C ada sebelum N di alfabet, jadi Critical adalah 0
        label = "Critical" if class_value == 0 else "Non-Critical"

    if str(label).strip().lower() == "critical":
        return "🚨 CRITICAL"
    return "✅ NON-CRITICAL"

def _build_proba_dict(model, proba):
    probas = {}
    classes = getattr(model, "classes_", [])
    for cls, p in zip(classes, proba):
        probas[_map_class_to_label(cls)] = p * 100
    probas.setdefault("✅ NON-CRITICAL", 0.0)
    probas.setdefault("🚨 CRITICAL", 0.0)
    return probas

def convert_to_binary(pred_num):
    """Convert numeric class to label text."""
    return _map_class_to_label(pred_num)

# --- LOAD DATA SAMPLE UNTUK DASHBOARD ---
@st.cache_data
def load_sample_data():
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(app_dir)
        csv_path = os.path.join(project_root, 'data', 'processed', 'github_issues_preprocessed.csv')
        df = pd.read_csv(csv_path, nrows=1000)
        return df
    except FileNotFoundError:
        return pd.DataFrame() 

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-family:'Google Sans','Regular 400'; font-size:2.5rem; color:white; font-weight:700;">Buginator</div>
        <div style="font-size:0.85rem; color:rgba(255,255,255,0.7); margin-top:4px;">Github Issue Navigator</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        options=[
            "Home",
            "📊 Data Explorer & EDA",
            "⚙️ Feature Extraction",
            "🏆 Model Evaluation & Comparison"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 10px;">
        <div style="font-size:0.8rem; color:white; font-weight:700; margin-bottom:8px; opacity:0.9;">Kelompok 8</div>
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.6); line-height:1.6;">
            1. ANGELINA JOLIE CANDAYA 2802541644<br>
            2. JOSHUA KEVIN LIEM 2802535572<br>
            3. MAUREEN CALISTA SURJO 2802536392<br>
            4. NICHOLAS HUBERT SOEGIHONO 2802515564<br>
            5. ZAHRA' ZAKIYYAH PRIYONO 2802492554
        </div>
        <div style="margin-top:12px; font-size:0.65rem; color:rgba(255,255,255,0.4); text-align:center; border-top:1px solid rgba(255,255,255,0.1); padding-top:8px;">
            Natural Language Project<br>
            Binus University
        </div>
    </div>
    """, unsafe_allow_html=True)

def hero(title, subtitle, badge="Powered by Machine Learning"):
    st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
        <div class="hero-badge">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

# ===========================================================================
# PAGE: HOME
# ===========================================================================
if page == "Home":
    hero("Buginator", "Github issue predictor")
    st.markdown("<p></p>", unsafe_allow_html=True)

    st.write("") 
    
    # Info tentang binary classification
    st.info("""
    ℹ️ **Binary Classification:** Model memprediksi apakah GitHub issue adalah **🚨 CRITICAL** (serius, urgent) 
    atau **✅ NON-CRITICAL** (tidak serius). Menggunakan 3 algoritma: Logistic Regression, Naive Bayes, & SVM.
    """)
    
    issue_text = st.text_area("Masukkan Detail Issue di bawah ini:", height=180, placeholder="Cth: The application crashes whenever I try to upload a file larger than 5MB...")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        predict_button = st.button("🔥 Jalankan Prediksi")

    if predict_button:
        if not issue_text.strip():
            st.error("⚠️ Teks tidak boleh kosong!")
        else:
            if tfidf_vectorizer is None:
                st.error("❌ Gagal memprediksi: File tfidf_vectorizer.pkl belum termuat.")
            else:
                with st.spinner("Mengirim ke model AI..."):
                    try:
                        cleaned = clean_text(issue_text)
                        vec_text = tfidf_vectorizer.transform([cleaned])
                        
                        # Dictionary untuk menyimpan prediksi dari 3 model
                        predictions = {}
                        
                        # 1. Logistic Regression
                        if tuned_lr_model is not None:
                            try:
                                pred_lr_num = tuned_lr_model.predict(vec_text)[0]
                                pred_lr = convert_to_binary(pred_lr_num)
                                proba_lr = tuned_lr_model.predict_proba(vec_text)[0]
                                # BINARY: proba[0] = Non-Critical (class 0), proba[1] = Critical (class 1)
                                # BINARY: proba[0] = Critical (class 0), proba[1] = Non-Critical (class 1)
                                confidence_lr = max(proba_lr) * 100

                                predictions['Logistic Regression'] = {
                                    'prediction': pred_lr,
                                    'pred_num': pred_lr_num,
                                    'confidence': confidence_lr,
                                    'probas': {'🚨 CRITICAL': proba_lr[0]*100, '✅ NON-CRITICAL': proba_lr[1]*100}
                                }
                            except Exception as e:
                                pass
                        
                        # 2. Naive Bayes
                        if nb_biner_model is not None:
                            try:
                                pred_nb_num = nb_biner_model.predict(vec_text)[0]
                                pred_nb = convert_to_binary(pred_nb_num)
                                proba_nb = nb_biner_model.predict_proba(vec_text)[0]
                                # BINARY: proba[0] = Non-Critical (class 0), proba[1] = Critical (class 1)
                                # BINARY: proba[0] = Critical (class 0), proba[1] = Non-Critical (class 1)
                                confidence_nb = max(proba_nb) * 100

                                predictions['Naive Bayes'] = {
                                    'prediction': pred_nb,
                                    'pred_num': pred_nb_num,
                                    'confidence': confidence_nb,
                                    'probas': {'🚨 CRITICAL': proba_nb[0]*100, '✅ NON-CRITICAL': proba_nb[1]*100}
                                }
                            except Exception as e:
                                pass
                        
                        # 3. SVM
                        if svm_biner_model is not None:
                            try:
                                pred_svm_num = svm_biner_model.predict(vec_text)[0]
                                pred_svm = convert_to_binary(pred_svm_num)
                                # Untuk SVM (LinearSVC), gunakan decision function untuk confidence
                                try:
                                    decision = svm_biner_model.decision_function(vec_text)[0]
                                    # Hitung confidence dengan sigmoid function
                                    confidence_svm = (1 / (1 + np.exp(-decision))) * 100
                                    if confidence_svm < 50:
                                        confidence_svm = 100 - confidence_svm
                                except:
                                    confidence_svm = 50.0
                                
                                predictions['SVM (Best F1)'] = {
                                    'prediction': pred_svm,
                                    'pred_num': pred_svm_num,
                                    'confidence': confidence_svm,
                                    'probas': None
                                }
                            except Exception as e:
                                pass
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
                        
                        # Tampilkan hasil dari ketiga model
                        if predictions:
                            st.markdown("<h3 style='color:white; text-align:center; margin-bottom:20px;'>📊 Hasil Prediksi dari 3 Model</h3>", unsafe_allow_html=True)
                            
                            cols = st.columns(len(predictions))
                            
                            for idx, (model_name, result) in enumerate(predictions.items()):
                                with cols[idx]:
                                    pred = result['prediction']
                                    conf = result['confidence']
                                    
                                    # Tentukan warna & icon berdasarkan prediction (BINARY)
                                    # FIX: Check untuk icon, bukan substring (karena "NON-CRITICAL" mengandung "CRITICAL")
                                    if "🚨" in pred:
                                        color = "#FF0000"
                                        icon = "🚨"
                                        desc = "Segera diperhatikan - Issue serius!"
                                    else:  # NON-CRITICAL
                                        color = "#00FF00"
                                        icon = "✅"
                                        desc = "Prioritas rendah - Bukan masalah serius"
                                    
                                    # Tambah badge "Best F1" untuk SVM
                                    badge_html = ""
                                    if "Best F1" in model_name:
                                        badge_html = "<div style='background:#FFD700; color:black; padding:2px 8px; border-radius:8px; font-size:0.7rem; font-weight:bold; display:inline-block; margin-bottom:8px;'>⭐ Terbaik F1</div>"
                                    
                                    st.markdown(f"""
                                    <div class="info-card" style="border-left: 6px solid {color}; text-align: center;">
                                        <div style="margin-bottom:8px;">{badge_html}</div>
                                        <h4 style="margin:0; color:{color}; margin-bottom:8px; font-size:0.9rem;">{model_name}</h4>
                                        <div style="font-size:2.5rem; margin:12px 0;">{icon}</div>
                                        <h3 style="margin:8px 0; color:white; font-size:1.6rem;">{pred}</h3>
                                        <p style="color:#99aab5; font-size:0.85rem; margin:8px 0;">{desc}</p>
                                        <div style="background:#202225; padding:12px; border-radius:8px; margin-top:12px; border:2px solid {color};">
                                            <div style="font-size:2.2rem; color:{color}; font-weight:bold;">{conf:.1f}%</div>
                                            <div style="font-size:0.75rem; color:#99aab5; margin-top:4px;">Confidence</div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Tampilkan detail probabilitas
                            st.markdown("<h4 style='color:white; margin-top:24px;'>📈 Detail Probabilitas per Kelas</h4>", unsafe_allow_html=True)
                            
                            for model_name, result in predictions.items():
                                if result['probas'] is not None:
                                    st.markdown(f"**{model_name}:**", unsafe_allow_html=True)
                                    
                                    proba_dict = result['probas']
                                    
                                    col_prob1, col_prob2 = st.columns(2)
                                    
                                    with col_prob1:
                                        st.markdown(f"""
                                        <div class="info-card" style="text-align: center; border-left:4px solid #00FF00;">
                                            <div style="color:#99aab5; font-size:0.85rem; margin-bottom:6px; font-weight:bold;">✅ Non-Critical</div>
                                            <div style="font-size:2rem; color:#00FF00; font-weight:bold;">{proba_dict.get('✅ NON-CRITICAL', 0):.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col_prob2:
                                        st.markdown(f"""
                                        <div class="info-card" style="text-align: center; border-left:4px solid #FF0000;">
                                            <div style="color:#99aab5; font-size:0.85rem; margin-bottom:6px; font-weight:bold;">🚨 Critical</div>
                                            <div style="font-size:2rem; color:#FF0000; font-weight:bold;">{proba_dict.get('🚨 CRITICAL', 0):.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("")
                        else:
                            st.error("❌ Gagal memprediksi: Tidak ada model yang tersedia.")
                            
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi: {str(e)}")

    col1, col2 = st.columns([3,2])
    with col1:
        section("Background")
        st.markdown("""
        <div class="section-desc">        
            This code is designed to analyze and process the existing labels in the GitHub Issues dataset as part of the label construction stage.
            Since the dataset does not explicitly provide severity levels (Low, Medium, Critical), the available labels must first be explored to understand their structure and distribution. The labels are stored in a multi-label format as raw text, meaning that each issue may contain multiple labels combined in a single string.<br><br>
            The main purpose of this process is to:
            <ol>
                <li>Extract and clean the raw labels from the dataset</li>
                <li>Separate multiple labels into individual components</li>
                <li>Normalize the labels for consistency</li>
                <li>Analyze the frequency of each label</li>
            </ol>
            This analysis is important to identify which labels are relevant and can be used as the basis for constructing severity categories in the next step.
        </div>  
        """, unsafe_allow_html=True)

        section("Dataset we are working with")
        st.markdown("""
        <div class="info-card">
            <h4>sharjeelyunus/github-issues-dataset</h4>
            <div style="font-size:0.85rem;color:#fffff;">
                Sumber: <a href="https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset" target="_blank" style="color:#f7f91a;text-decoration:none;"><b>Hugging Face</b></a><br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        section("NLP project workflow")
        steps = [
            ("1", "Label Construction", "desc1"),
            ("2", "EDA", "desc2"),
            ("3", "Preprocessing", "desc3"),
            ("4", "Feature Extraction", "desc4"),
            ("5", "Modelling LR", "desc5"),
            ("6", "Modelling NB", "desc6"),
            ("7", "Evaluation", "desc7"),
            ("8", "Baseline Count Vectorizer", "desc8"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="info-card" style="padding:14px 18px; margin-bottom:10px;">
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <span class="step-badge">{num}</span>
                    <b style="color:#006aeb;"> {title}</b>
                </div>
                <div style="font-size:0.82rem;color:white;padding-left:46px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    section("Project Team")
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #202225 0%, #2f3136 100%);">
        <b style="color:#006aeb; font-size:1rem; display:block; margin-bottom:10px;">Kelompok 8</b>
        <table style="width:100%; font-size:0.85rem; color:white; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #40444b;">
                <td style="padding:8px 0; font-weight:600;">ANGELINA JOLIE CANDAYA</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802541644</td>
            </tr>
            <tr style="border-bottom:1px solid #40444b;">
                <td style="padding:8px 0; font-weight:600;">JOSHUA KEVIN LIEM</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802535572</td>
            </tr>
            <tr>
                <td style="padding:8px 0; font-weight:600;">MAUREEN CALISTA SURJO</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802536392</td>
            </tr>
            <tr>
                <td style="padding:8px 0; font-weight:600;">NICHOLAS HUBERT SOEGIHONO</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802515564</td>
            </tr>
            <tr>
                <td style="padding:8px 0; font-weight:600;">ZAHRA' ZAKIYYAH PRIYONO</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802492554</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: DATA EXPLORER & EDA
# ===========================================================================
elif page == "📊 Data Explorer & EDA":
    hero("Data Explorer & EDA", "Analisis karakteristik dataset, ketidakseimbangan kelas, dan proses transformasi teks", "Pipeline Tahap 1 & 2")
    
    section("Overview Dataset")
    df_sample = load_sample_data()
    
    if df_sample.empty:
        st.error("❌ File github_issues_preprocessed.csv tidak ditemukan di folder data/processed/")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="info-card" style="text-align: center; border-top: 4px solid #006aeb;">
                <h4 style="color: #99aab5; margin: 0; font-size:14px;">Total Dataset Bersih</h4>
                <h2 style="color: white; margin: 5px 0;">114,065 <span style="font-size:14px; color:#99aab5;">Baris</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("""
            <div class="info-card" style="text-align: center; border-top: 4px solid #006aeb;">
                <h4 style="color: #99aab5; margin: 0; font-size:14px;">Target Klasifikasi</h4>
                <h2 style="color: white; margin: 5px 0;">2 <span style="font-size:14px; color:#99aab5;">Kelas Severity</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown("""
            <div class="info-card" style="text-align: center; border-top: 4px solid #006aeb;">
                <h4 style="color: #99aab5; margin: 0; font-size:14px;">Maksimal Fitur Vocab</h4>
                <h2 style="color: white; margin: 5px 0;">50,000 <span style="font-size:14px; color:#99aab5;">Kata Penting</span></h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<p style='color:#99aab5; margin-bottom: 5px;'>Sampel representatif data hasil pemrosesan (1.000 baris pertama):</p>", unsafe_allow_html=True)
        df_sample_view = df_sample.copy()
        df_sample_view['severity'] = df_sample_view['severity'].replace({
            'Major': 'Non-Critical',
            'Minor': 'Non-Critical'
        })
        st.dataframe(df_sample_view, use_container_width=True, height=300)
        
        st.markdown("---")

        section("Exploratory Data Analysis")
        col_chart_text, col_chart_visual = st.columns([2, 3])
        
        with col_chart_text:
            st.markdown("""
            <div class="section-desc" style="height: 100%;">
                <h4 style="margin-top:0; color:#FFD700;">🚨 Masalah Class Imbalance</h4>
                Berdasarkan visualisasi distribusi target di samping, dataset ini memiliki masalah <b>Ketidakseimbangan Kelas (Class Imbalance)</b> yang sangat nyata.
                <br><br>
                Jumlah data dengan label <code>Critical</code> jauh lebih sedikit dibandingkan dengan kelas <code>Non-Critical</code>.
                <br><br>
                <b>Dampak Praktis:</b> Model cenderung akan lebih pintar menebak kelas Non-Critical karena contoh datanya melimpah. Oleh sebab itu, performa model akhir tidak boleh dinilai dari Akurasi saja, melainkan berpatokan pada nilai <b>F1-Macro Average</b>.
            </div>
            """, unsafe_allow_html=True)
            
        with col_chart_visual:
            st.markdown("<div class='info-card' style='margin-bottom:0;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center; color:white; font-size:0.95rem; margin-bottom:10px;'>Distribusi Frekuensi Target Severity</h4>", unsafe_allow_html=True)
            severity_binary = df_sample['severity'].replace({
                'Major': 'Non-Critical',
                'Minor': 'Non-Critical'
            })
            dist_data = severity_binary.value_counts().reindex(['Non-Critical', 'Critical']).fillna(0)
            st.bar_chart(dist_data, color="#006aeb")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='color:#99aab5; margin-bottom: 5px;'>Statistik sebaran panjang kata di dalam teks dokumen laporan:</p>", unsafe_allow_html=True)
        
        c_stat1, c_stat2, c_stat3 = st.columns(3)
        with c_stat1:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <div style="font-size:2.2rem; color:#006aeb; font-weight:900;">1</div>
                <div style="color:#99aab5; font-size:0.85rem; font-weight:600; margin-top:4px;">Kata (Teks Terpendek)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_stat2:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <div style="font-size:2.2rem; color:#006aeb; font-weight:900;">~108</div>
                <div style="color:#99aab5; font-size:0.85rem; font-weight:600; margin-top:4px;">Kata (Rata-rata per Teks)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_stat3:
            st.markdown("""
            <div class="info-card" style="text-align: center; border-left: 4px solid red;">
                <div style="font-size:2.2rem; color:red; font-weight:900;">12,000+</div>
                <div style="color:#99aab5; font-size:0.85rem; font-weight:600; margin-top:4px;">Kata (Outlier Terpanjang)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        section("Text Preprocessing Pipeline")
        st.markdown("""
        <div class="section-desc" style="display: flex; justify-content: space-between; text-align: center; padding: 15px 20px;">
            <div><span class="step-badge" style="width:24px; height:24px; line-height:24px; font-size:0.8rem;">1</span> <b>Lowercasing</b></div>
            <div><span class="step-badge" style="width:24px; height:24px; line-height:24px; font-size:0.8rem;">2</span> <b>Regex Cleaning</b></div>
            <div><span class="step-badge" style="width:24px; height:24px; line-height:24px; font-size:0.8rem;">3</span> <b>Stopwords Removal</b></div>
            <div><span class="step-badge" style="width:24px; height:24px; line-height:24px; font-size:0.8rem;">4</span> <b>Lemmatization</b></div>
        </div>
        """, unsafe_allow_html=True)

        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("""
            <div class="info-card" style="border-top: 4px solid red; height: 100%;">
                <h4 style="color:red; font-weight:700;">🔴 Raw Text Input (Before)</h4>
                <p style="color:#99aab5; font-size:0.8rem; margin-top:-5px;">Teks dokumen asli dari server GitHub</p>
                <div style="background:#202225; padding:15px; border-radius:8px; color:#eee; font-family:monospace; font-size:0.88rem; line-height:1.6; border: 1px solid #40444b;">
                    [Feature]: Invalid header error should show what the invalid header is.<br><br>
                    When passing an invalid header to fetch() it just throws a generic TypeError: Invalid name error... It would be very helpful if it included the name of the invalid header!
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_after:
            st.markdown("""
            <div class="info-card" style="border-top: 4px solid #006aeb; height: 100%;">
                <h4 style="color:#006aeb; font-weight:700;">🔵 Cleaned Tokens (After)</h4>
                <p style="color:#99aab5; font-size:0.8rem; margin-top:-5px;">Hasil normalisasi korpus menggunakan NLTK</p>
                <div style="background:#202225; padding:15px; border-radius:8px; color:#eee; font-family:monospace; font-size:0.88rem; line-height:1.6; border: 1px solid #40444b;">
                    feature invalid header error show invalid header <br><br>
                    passing invalid header fetch throw generic typeerror invalid name error would helpful included name invalid header
                </div>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================================
# PAGE: FEATURE ENGINEERING & BASELINE
# ===========================================================================
elif page == "⚙️ Feature Extraction":
    hero("Feature Engineering & Baseline", "Eksperimen saintifik representasi teks dan pembagian dataset", "Pipeline Tahap 3 & 7")
    
    section("Data Splitting")
    st.markdown("<p style='color:#99aab5;'>Proporsi pembagian dataset untuk proses pelatihan (Training), validasi (Validation), dan pengujian akhir (Testing):</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card" style="text-align: center; border-bottom: 4px solid #006aeb;">
            <h4 style="color: #99aab5; margin: 0; font-size:14px;">Data Train (Pelatihan)</h4>
            <h2 style="color: white; margin: 5px 0;">79,845 <span style="font-size:14px; color:#99aab5;">Baris</span></h2>
            <div style="font-size:0.8rem; color:#006aeb; font-weight:bold;">~70% Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="info-card" style="text-align: center; border-bottom: 4px solid #FFD700;">
            <h4 style="color: #99aab5; margin: 0; font-size:14px;">Data Validation (Validasi)</h4>
            <h2 style="color: white; margin: 5px 0;">17,110 <span style="font-size:14px; color:#99aab5;">Baris</span></h2>
            <div style="font-size:0.8rem; color:#FFD700; font-weight:bold;">~15% Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="info-card" style="text-align: center; border-bottom: 4px solid red;">
            <h4 style="color: #99aab5; margin: 0; font-size:14px;">Data Test (Pengujian)</h4>
            <h2 style="color: white; margin: 5px 0;">17,110 <span style="font-size:14px; color:#99aab5;">Baris</span></h2>
            <div style="font-size:0.8rem; color:red; font-weight:bold;">~15% Dataset</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    section("Eksperimen Representasi Teks (Baseline)")
    st.markdown("""
    <div class="section-desc">
        Komputer tidak bisa membaca teks mentah. Oleh karena itu, kita perlu melakukan eksperimen untuk membandingkan dua metode ekstraksi fitur: <b>TF-IDF Vectorizer</b> dan <b>CountVectorizer (sebagai Baseline)</b>. Keduanya dibatasi maksimal <b>50.000 kata</b> yang paling relevan untuk menjaga efisiensi memori.
    </div>
    """, unsafe_allow_html=True)

    col_table, col_text = st.columns([4, 5])
    
    with col_table:
        st.markdown("<h4 style='color:white; font-size:1rem; margin-bottom:10px;'>Tabel Komparasi Akurasi (Logistic Regression)</h4>", unsafe_allow_html=True)
        baseline_data = {
            "Metode Representasi": ["CountVectorizer", "TF-IDF Vectorizer"],
            "Akurasi": ["80.50%", "80.36%"],
            "Prinsip": ["Menghitung Frekuensi Murni", "Bobot Kepentingan Kata"]
        }
        st.dataframe(pd.DataFrame(baseline_data), hide_index=True, use_container_width=True)
        
    with col_text:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #FFD700; height: 100%;">
            <h4 style="color:#FFD700; margin-top:0;">💡 Kesimpulan Eksperimen</h4>
            Hasil pada data *Test* menunjukkan bahwa <b>TF-IDF</b> (80.36%) dan <b>CountVectorizer</b> (80.50%) menghasilkan tingkat akurasi yang nyaris identik.
            <br><br>
            Hal ini memberikan <i>insight</i> penting: <b>masalah utama (bottleneck) dalam proyek ini bukan terletak pada metode representasi teksnya</b>, melainkan pada <i>Class Imbalance</i> (yang telah kita lihat di EDA) dan kemampuan algoritma Machine Learning itu sendiri dalam memisahkan pola kelas yang minoritas.
        </div>
        """, unsafe_allow_html=True)

# ===========================================================================
# PAGE: MODEL EVALUATION & COMPARISON
# ===========================================================================
elif page == "🏆 Model Evaluation & Comparison":
    hero("Model Evaluation & Comparison", "Evaluasi mendalam, komparasi algoritma, dan performa prediksi final", "Tahap 4, 5, 6, & 7")
    
    section("Head-to-Head: LR vs Multinomial NB")
    st.markdown("""
    <div class="section-desc">
        Mengingat dataset kita memiliki masalah <b>Class Imbalance</b> (kelas Critical sangat minoritas), metrik Akurasi saja bisa menyesatkan. Oleh karena itu, kita membandingkan performa <b>Logistic Regression (LR)</b> dan <b>Multinomial Naïve Bayes (MNB)</b> dengan fokus utama pada metrik <b>F1-Score kelas Critical</b>.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-card" style="border-top: 4px solid #006aeb; height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="color:#006aeb; margin:0;">Logistic Regression</h3>
                <span style="background:#006aeb; color:white; padding:4px 12px; border-radius:12px; font-size:0.8rem; font-weight:bold;">WINNER 🏆</span>
            </div>
            <hr style="border-color:#40444b; margin:10px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#99aab5; font-size:1.1rem;">Akurasi Keseluruhan:</span>
                <span style="color:white; font-size:1.2rem; font-weight:bold;">80.36%</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; background:#202225; padding:10px; border-radius:8px; border:1px solid #006aeb;">
                <span style="color:#99aab5; font-size:1.1rem; font-weight:bold;">F1-Score Critical:</span>
                <span style="color:#006aeb; font-size:1.8rem; font-weight:900;">0.76</span>
            </div>
            <p style="color:#99aab5; font-size:0.85rem; margin-top:10px;">Algoritma ini sangat tangguh dalam mempelajari pola kelas minoritas dan menangani representasi teks yang <i>sparse</i> (renggang) dari TF-IDF.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card" style="border-top: 4px solid red; height: 100%;">
            <h3 style="color:red; margin:0;">Multinomial Naïve Bayes</h3>
            <hr style="border-color:#40444b; margin:10px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#99aab5; font-size:1.1rem;">Akurasi Keseluruhan:</span>
                <span style="color:white; font-size:1.2rem; font-weight:bold;">74.59%</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; background:#202225; padding:10px; border-radius:8px; border:1px solid red;">
                <span style="color:#99aab5; font-size:1.1rem; font-weight:bold;">F1-Score Critical:</span>
                <span style="color:red; font-size:1.8rem; font-weight:900;">0.46</span>
            </div>
            <p style="color:#99aab5; font-size:0.85rem; margin-top:10px;">Sangat sensitif terhadap ketidakseimbangan data. MNB lebih sering salah menebak kelas Critical sebagai kelas mayoritas (Non-Critical).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    section("Visualisasi Performa (Logistic Regression)")
    
    col_lc, col_cm = st.columns(2)
    
    with col_lc:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <h4 style="color:#FFD700; text-align:center;">📈 Learning Curve</h4>
        """, unsafe_allow_html=True)
        
        try:
            st.image("image_ff2747.png", use_container_width=True)
            st.markdown("""
            <div style="color:#99aab5; font-size:0.85rem; margin-top:10px; line-height:1.5;">
                <b>Interpretasi:</b> Grafik ini membuktikan bahwa model Logistic Regression <b>tidak mengalami overfitting</b>. Garis <i>Training Accuracy</i> dan <i>Validation Accuracy</i> berjalan stabil, berdekatan, dan konvergen seiring bertambahnya jumlah data latih.
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.error("⚠️ File 'image_ff2747.png' tidak ditemukan di direktori.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cm:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <h4 style="color:#FFD700; text-align:center;">🧩 Confusion Matrix</h4>
        """, unsafe_allow_html=True)
        
        try:
            st.image("image_1ba247.png", use_container_width=True)
            st.markdown("""
            <div style="color:#99aab5; font-size:0.85rem; margin-top:10px; line-height:1.5;">
                <b>Interpretasi:</b> Matriks ini memetakan prediksi yang benar (diagonal utama) dan yang meleset. Kesalahan klasifikasi terbanyak terjadi ketika model mengira laporan <code>Critical</code> sebagai <code>Non-Critical</code>, yang secara linguistik memang sering memiliki kemiripan kata (<i>overlapping keywords</i>).
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.error("⚠️ File 'image_1ba247.png' tidak ditemukan di direktori.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    section("Final Classification Report (Logistic Regression)")
    st.markdown("<p style='color:#99aab5;'>Laporan evaluasi klasifikasi rinci untuk model Logistic Regression terbaik pada Data Test (17.110 laporan):</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#99aab5; font-size:0.8rem;'>Catatan: Setelah konversi ke binary class, nilai metrik perlu dihitung ulang sesuai hasil evaluasi terbaru.</p>", unsafe_allow_html=True)
    
    report_data = {
        "Class / Label": ["Critical", "Non-Critical", "accuracy", "macro avg", "weighted avg"],
        "Precision": ["0.99", "0.87", "-", "0.93", "0.94"],
        "Recall": ["0.90", "0.99", "-", "0.94", "0.93"],
        "F1-Score": ["0.94", "0.93", "0.93", "0.93", "0.94"],
        "Support": ["13294", "9477", "22771", "22771", "22771"]
    }
    
    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True)
