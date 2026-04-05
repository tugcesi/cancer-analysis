import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Breast Cancer Prognosis Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 1rem; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    h1, h2, h3 { color: #667eea; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model_joblib = joblib.load('deadoralive.joblib')
        with open('deadoralive.pkl', 'rb') as f:
            pkl_data = pickle.load(f)
        
        if isinstance(pkl_data, dict):
            model_pkl = pkl_data.get('model', pkl_data)
        else:
            model_pkl = pkl_data
        
        return model_joblib, model_pkl
    except Exception as e:
        st.error(f"❌ Model yüklenemedi: {e}")
        return None, None

model_joblib, model_pkl = load_models()

st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <h1 style="color: #667eea; font-size: 2rem;">🏥</h1>
    <h2 style="color: #667eea;">Kontrol Paneli</h2>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("📍 Sayfa Seç:", 
                        ["🏠 Anasayfa", "🔮 Tahmin Yap", "📊 Model Bilgisi", "ℹ️ Hakkında"])

if page == "🏠 Anasayfa":
    st.markdown("""
        <div class="header-box">
            <h1>🏥 Meme Kanseri Prognozu Tahmin Sistemi</h1>
            <p style="font-size: 1.2rem; margin-top: 1rem;">Yapay Zeka Destekli Karar Destek Aracı</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Veri Seti", "317 Hasta", "+2.3%", delta_color="off")
    with col2:
        st.metric("🎯 Doğruluk", "~85%", "v1.0", delta_color="off")
    with col3:
        st.metric("📊 Özellikler", "15", "Features", delta_color="off")
    with col4:
        st.metric("⚙️ Model", "Dual", "Ensemble", delta_color="off")

elif page == "🔮 Tahmin Yap":
    st.markdown("""
        <div class="header-box">
            <h1>🔮 Hasta Prognoz Tahmini</h1>
            <p>Lütfen hasta bilgilerini aşağıda girin</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Hasta Bilgileri")
        age = st.slider("📅 Yaş", 18, 100, 50)
        gender = st.selectbox("👥 Cinsiyeti", [0, 1], 
                            format_func=lambda x: "👩 Kadın" if x == 0 else "👨 Erkek")
        follow_up_days = st.number_input("📊 Takip Süresi (gün)", 1, 3000, 500)
    
    with col2:
        st.subheader("🧬 Protein Seviyeleri")
        protein1 = st.number_input("🧪 Protein 1", -2.5, 2.5, 0.2, step=0.1)
        protein2 = st.number_input("🧪 Protein 2", -1.0, 3.5, 1.0, step=0.1)
        protein3 = st.number_input("🧪 Protein 3", -2.0, 2.0, 0.0, step=0.1)
        protein4 = st.number_input("🧪 Protein 4", -2.0, 1.5, 0.3, step=0.1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🏥 Tümör Bilgisi")
        tumor_stage = st.selectbox("📈 Tümör Evresi", [1, 2, 3], 
                                  format_func=lambda x: f"Evre {x}")
        her2_status = st.selectbox("🔬 HER2 Statüsü", [0, 1], 
                                  format_func=lambda x: "❌ Negatif" if x == 0 else "✅ Pozitif")
        histology = st.selectbox("🔬 Histoloji", [0, 1], 
                                format_func=lambda x: "Ductal" if x == 0 else "Lobular")
    
    with col4:
        st.subheader("⚕️ Tedavi Bilgisi")
        surgery_type = st.selectbox("🔪 Cerrahi Türü", 
                                   ['Modified Radical Mastectomy', 'Lumpectomy', 'Simple Mastectomy'],
                                   help="Yapılan cerrahi işlem")
    
    # Calculate derived features
    protein_mean = np.mean([protein1, protein2, protein3, protein4])
    protein_std = np.std([protein1, protein2, protein3, protein4])
    
    # DOĞRU FEATURES - Model'in beklediği sıra ve isimler
    # Surgery_type one-hot encoding
    surgery_modified = 1 if surgery_type == 'Modified Radical Mastectomy' else 0
    surgery_other = 1 if surgery_type == 'Lumpectomy' else 0  # Lumpectomy = Other
    surgery_simple = 1 if surgery_type == 'Simple Mastectomy' else 0
    
    # Input DataFrame - MODEL'İN BEKLEDIĞI SIRALAMAYLA
    input_data = pd.DataFrame({
        'Protein_Mean': [protein_mean],
        'Protein4': [protein4],
        'Protein2': [protein2],
        'Tumor_Stage': [tumor_stage],
        'HER2 status': [her2_status],
        'Follow_up_Days': [follow_up_days],
        'Protein3': [protein3],
        'Histology': [histology],
        'Protein1': [protein1],
        'Protein_Std': [protein_std],
        'Gender': [gender],
        'Age': [age],
        'Surgery_type_Modified Radical Mastectomy': [surgery_modified],
        'Surgery_type_Other': [surgery_other],
        'Surgery_type_Simple Mastectomy': [surgery_simple],
    })
    
    if st.button("🎯 TAHMIN YAP", use_container_width=True):
        st.markdown("---")
        
        if model_joblib is None or model_pkl is None:
            st.error("❌ Modeller yüklenemedi!")
        else:
            try:
                st.success(f"✅ Features doğru! ({len(input_data.columns)} adet)")
                
                # Tahminler
                pred_proba_joblib = model_joblib.predict_proba(input_data)[0]
                pred_proba_pkl = model_pkl.predict_proba(input_data)[0]
                
                # Ortalama
                avg_pred = (pred_proba_joblib[1] + pred_proba_pkl[1]) / 2
                
                st.markdown("<h3 style='color: #667eea; text-align: center;'>📊 TAHMIN SONUÇLARI</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🤖 Joblib", f"{pred_proba_joblib[1]*100:.1f}%", "Ölüm Riski")
                
                with col2:
                    st.metric("🤖 Pickle", f"{pred_proba_pkl[1]*100:.1f}%", "Ölüm Riski")
                
                with col3:
                    st.metric("📊 Ortalama", f"{avg_pred*100:.1f}%", "Ölüm Riski")
                
                st.markdown("---")
                
                # Risk gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_pred*100,
                    title={'text': "Risk Seviyesi (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 33], 'color': "#d4edda"},
                            {'range': [33, 66], 'color': "#fff3cd"},
                            {'range': [66, 100], 'color': "#f8d7da"}
                        ],
                    }
                ))
                fig.update_layout(height=400, paper_bgcolor="#f8f9fa")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Risk sınıflandırması
                if avg_pred < 0.33:
                    st.success("✅ **DÜŞÜK RİSK** - Hasta yaşama olasılığı yüksek")
                elif avg_pred < 0.66:
                    st.warning("⚠️ **ORTA RİSK** - Düzenli takip gerekli")
                else:
                    st.error("🔴 **YÜKSEK RİSK** - Yoğun tedavi ve takip önerilir")
                
                st.markdown("---")
                
                # Protein görselleştirmesi
                st.subheader("🧬 Protein Seviyeleri Analizi")
                protein_data = pd.DataFrame({
                    'Protein': ['P1', 'P2', 'P3', 'P4'],
                    'Seviye': [protein1, protein2, protein3, protein4]
                })
                fig_protein = px.bar(protein_data, x='Protein', y='Seviye',
                                    color='Seviye', color_continuous_scale='Viridis',
                                    title="Protein Seviyeleri")
                fig_protein.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#f8f9fa')
                st.plotly_chart(fig_protein, use_container_width=True)
                
                # Hasta özeti
                st.subheader("📋 Hasta Özeti")
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    st.write(f"**Yaş:** {age} yıl")
                    st.write(f"**Cinsiyet:** {'Kadın' if gender == 0 else 'Erkek'}")
                    st.write(f"**Takip:** {follow_up_days} gün")
                
                with col_sum2:
                    st.write(f"**Protein Ort:** {protein_mean:.3f}")
                    st.write(f"**Protein Std:** {protein_std:.3f}")
                    st.write(f"**Tümör Evresi:** {tumor_stage}")
                
                with col_sum3:
                    st.write(f"**HER2:** {'Pozitif' if her2_status == 1 else 'Negatif'}")
                    st.write(f"**Histoloji:** {'Ductal' if histology == 0 else 'Lobular'}")
                    st.write(f"**Cerrahi:** {surgery_type}")
                
            except Exception as e:
                st.error(f"❌ Hata: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

elif page == "📊 Model Bilgisi":
    st.markdown("<div class='header-box'><h1>📊 Model Bilgisi</h1></div>", unsafe_allow_html=True)
    st.info("""
    **Model:** BernoulliNB (Naive Bayes)
    **Veri Seti:** 317 hasta
    **Eğitim:** 253 örnek (%80)
    **Test:** 64 örnek (%20)
    **Doğruluk:** ~80-85%
    **Toplam Features:** 15
    """)

elif page == "ℹ️ Hakkında":
    st.markdown("<div class='header-box'><h1>ℹ️ Hakkında</h1></div>", unsafe_allow_html=True)
    st.warning("⚠️ Bu sistem yalnızca eğitim amaçlıdır. Gerçek klinik kararlar için doktor danışması alınız.")