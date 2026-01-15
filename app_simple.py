# app_simple.py
import os
import streamlit as st
import pandas as pd
from preprocessing import preprocess, preprocess_detailed
from similarity import jaccard_similarity
from utils import read_file
from collections import defaultdict

# Konfigurasi Halaman
st.set_page_config(
    page_title="Sistem Temu Balik Dokumen",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Konfigurasi")
    
    # Input Folder
    folder = st.text_input("📂 Folder Dokumen", "documents", help="Masukkan path folder tempat dokumen disimpan.")
    
    if not os.path.exists(folder):
        st.error("❌ Folder tidak ditemukan.")
        st.stop()
    
    # List Files
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    st.info(f"📂 Terdeteksi: **{len(files)} dokumen**")
    
    with st.expander("📜 Lihat Daftar Dokumen"):
        st.write(files)
        
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.caption("Sistem Temu Balik Informasi menggunakan **Jaccard Similarity** dan **Stemming AYS**.")
    st.caption("Dibuat untuk Tugas Besar Data Mining.")

# --- MAIN CONTENT ---
st.title("🔍 Sistem Temu Balik Dokumen")
st.markdown("### Cari dokumen relevan dengan cepat dan akurat")

# --- PREPROCESSING LOGIC (Cached) ---
if 'documents' not in st.session_state or st.session_state.get('folder') != folder:
    with st.spinner('🔄 Sedang memproses dokumen... Mohon tunggu sebentar.'):
        st.session_state.folder = folder
        documents = {}
        documents_full = {}
        documents_detailed = {}  # Menyimpan detail preprocessing
        raw_texts = {}
        
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(files):
            # Update progress
            progress = (idx + 1) / len(files)
            progress_bar.progress(progress)
            
            path = os.path.join(folder, file)
            text = read_file(path)
            raw_texts[file] = text
            
            tokens = preprocess(text)
            detailed = preprocess_detailed(text)
            documents[file] = [stem for original, stem in tokens]
            documents_full[file] = tokens
            documents_detailed[file] = detailed
        
        st.session_state.documents = documents
        st.session_state.documents_full = documents_full
        st.session_state.documents_detailed = documents_detailed
        st.session_state.raw_texts = raw_texts
        
        progress_bar.empty()
        st.success(f"✅ Berhasil memproses {len(files)} dokumen!")
else:
    documents = st.session_state.documents
    documents_full = st.session_state.documents_full
    documents_detailed = st.session_state.documents_detailed
    raw_texts = st.session_state.raw_texts

# --- SEARCH UI ---
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("🔎 Masukkan kata kunci pencarian:", placeholder="Contoh: ekonomi digital di indonesia...")
with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    search_button = st.button("Cari", type="primary")

if query:
    # Preprocessing Query
    query_tokens = preprocess(query)
    query_stems = [stem for original, stem in query_tokens]
    
    st.markdown("---")
    st.subheader("📊 Hasil Pencarian")
    
    # Tampilkan Query yang diproses
    with st.expander("ℹ️ Detail Query (Preprocessing)", expanded=False):
        st.write("**Query Asli:**", query)
        st.write("**Token Hasil Stemming:**", query_stems)

    # Hitung Similarity
    results = []
    for doc_name, doc_stems in documents.items():
        score = jaccard_similarity(
            [(w, w) for w in doc_stems],
            [(w, w) for w in query_stems]
        )
        
        relevansi_icon = "⭐⭐⭐" if score > 0.3 else "⭐⭐" if score > 0.1 else "⭐"
        if score == 0: relevansi_icon = "⚪"
            
        results.append({
            "Dokumen": doc_name,
            "Skor": score,
            "Relevansi": relevansi_icon
        })
    
    # Sorting
    results = sorted(results, key=lambda x: x["Skor"], reverse=True)
    relevant_results = [r for r in results if r["Skor"] > 0]
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Dokumen", len(files))
    m2.metric("Dokumen Relevan", len(relevant_results))
    m3.metric("Top Score", f"{relevant_results[0]['Skor']:.4f}" if relevant_results else "0.0000")
    
    if relevant_results:
        st.markdown("### 📄 Daftar Dokumen Relevan")
        
        for idx, result in enumerate(relevant_results):
            doc_name = result['Dokumen']
            score = result['Skor']
            
            # Card style expander
            with st.expander(f"#{idx+1} {doc_name} | Skor: {score:.4f} {result['Relevansi']}", expanded=(idx == 0)):
                
                tab1, tab2, tab3, tab4 = st.tabs(["📜 Cuplikan Teks", "🔍 Analisis Kata", "🧮 Perhitungan Similarity", "ℹ️ Info File"])
                
                with tab1:
                    raw_text = raw_texts.get(doc_name, "")
                    # Highlight query terms (simple approach)
                    # Note: This is case sensitive in simple replace, but good enough for visual
                    display_text = raw_text[:1500] + ("..." if len(raw_text) > 1500 else "")
                    st.markdown(f"```text\n{display_text}\n```")
                    st.caption("*Menampilkan 1500 karakter pertama.*")

                with tab2:
                    detailed_tokens = documents_detailed.get(doc_name, [])
                    
                    # Buat mapping untuk menghitung frekuensi berdasarkan stem
                    stem_mapping = defaultdict(lambda: {'originals': [], 'case_folding': [], 'filtering': [], 'count': 0})
                    
                    for detail in detailed_tokens:
                        stem = detail['stemming']
                        stem_mapping[stem]['originals'].append(detail['original'])
                        stem_mapping[stem]['case_folding'].append(detail['case_folding'])
                        stem_mapping[stem]['filtering'].append(detail['filtering'])
                        stem_mapping[stem]['count'] += 1
                    
                    # Buat data untuk tabel dengan tahapan proses
                    data = []
                    for stem, info in stem_mapping.items():
                        is_match = stem in query_stems
                        # Ambil contoh pertama untuk setiap tahap
                        original_examples = sorted(set(info['originals']))
                        
                        data.append({
                            "Kata Asli": ", ".join(original_examples[:3]) + ("..." if len(original_examples) > 3 else ""),
                            "Case Folding": original_examples[0].lower(),
                            "Filtering": original_examples[0].lower(),  # sama karena sudah lolos filtering
                            "Stemming": stem,
                            "Frekuensi": info['count'],
                            "Match": "✅" if is_match else ""
                        })
                    
                    df_analysis = pd.DataFrame(data)
                    # Sort by Match then Frequency
                    if not df_analysis.empty:
                        df_analysis = df_analysis.sort_values(by=["Match", "Frekuensi"], ascending=[False, False])
                    
                    st.caption("📊 **Tahapan Preprocessing:** Kata Asli → Case Folding → Filtering (Stopword Removal) → Stemming")
                    
                    st.dataframe(
                        df_analysis, 
                        column_config={
                            "Kata Asli": st.column_config.TextColumn("Kata Asli", width="medium"),
                            "Case Folding": st.column_config.TextColumn("Case Folding", width="medium"),
                            "Filtering": st.column_config.TextColumn("Filtering", width="medium"),
                            "Stemming": st.column_config.TextColumn("Stemming (Hasil)", width="medium"),
                            "Frekuensi": st.column_config.ProgressColumn("Frekuensi", format="%d", min_value=0, max_value=max([d['Frekuensi'] for d in data]) if data else 100),
                            "Match": st.column_config.TextColumn("Match", width="small")
                        },
                        use_container_width=True,
                        hide_index=True
                    )

                with tab3:
                    st.markdown("### 🧮 Perhitungan Jaccard Similarity")
                    
                    # Ambil set dokumen dan query
                    doc_stems = documents.get(doc_name, [])
                    doc_set = set(doc_stems)
                    query_set = set(query_stems)
                    
                    # Hitung intersection dan union
                    intersection = doc_set.intersection(query_set)
                    union = doc_set.union(query_set)
                    
                    # Tampilkan rumus
                    st.markdown("**Rumus Jaccard Similarity:**")
                    st.latex(r"J(A, B) = \frac{|A \cap B|}{|A \cup B|}")
                    
                    st.markdown("---")
                    
                    # Detail set
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Set Query (A):**")
                        st.info(f"Jumlah kata unik: **{len(query_set)}**")
                        with st.expander("Lihat kata-kata"):
                            st.write(sorted(list(query_set)))
                    
                    with col2:
                        st.markdown("**Set Dokumen (B):**")
                        st.info(f"Jumlah kata unik: **{len(doc_set)}**")
                        with st.expander("Lihat kata-kata"):
                            st.write(sorted(list(doc_set)))
                    
                    st.markdown("---")
                    
                    # Intersection
                    st.markdown("**Irisan (A ∩ B):**")
                    st.success(f"Jumlah kata yang sama: **{len(intersection)}**")
                    if intersection:
                        st.write("Kata-kata yang cocok:", ", ".join(sorted(list(intersection))))
                    else:
                        st.write("Tidak ada kata yang cocok")
                    
                    st.markdown("---")
                    
                    # Union
                    st.markdown("**Gabungan (A ∪ B):**")
                    st.info(f"Jumlah total kata unik: **{len(union)}**")
                    
                    st.markdown("---")
                    
                    # Perhitungan
                    st.markdown("**Perhitungan:**")
                    
                    if len(union) > 0:
                        st.latex(rf"J(A, B) = \frac{{{len(intersection)}}}{{{len(union)}}} = {score:.6f}")
                        
                        # Penjelasan dalam persentase
                        percentage = score * 100
                        st.metric("Tingkat Kemiripan", f"{percentage:.2f}%")
                        
                        # Interpretasi
                        if score > 0.5:
                            interpretation = "🟢 Sangat Mirip - Dokumen sangat relevan dengan query"
                        elif score > 0.3:
                            interpretation = "🟡 Cukup Mirip - Dokumen cukup relevan dengan query"
                        elif score > 0.1:
                            interpretation = "🟠 Sedikit Mirip - Dokumen memiliki keterkaitan dengan query"
                        else:
                            interpretation = "🔴 Kurang Mirip - Dokumen kurang relevan dengan query"
                        
                        st.info(interpretation)
                    else:
                        st.warning("Tidak dapat menghitung similarity (union kosong)")

                with tab4:
                    st.write(f"**Nama File:** {doc_name}")
                    st.write(f"**Ukuran Teks:** {len(raw_texts.get(doc_name, ''))} karakter")
                    st.write(f"**Jumlah Token:** {len(documents_full.get(doc_name, []))} kata")

    else:
        st.warning("⚠️ Tidak ditemukan dokumen yang cocok dengan kata kunci tersebut.")
        st.markdown("Cobalah menggunakan kata kunci yang lebih umum atau periksa kembali ejaan Anda.")
