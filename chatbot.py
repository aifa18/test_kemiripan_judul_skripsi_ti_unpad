import streamlit as st
import pandas as pd
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import requests
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyDE47svwoYcG7VI66MAydw8pNgSvopHeJo"
# Setup konfigurasi API
genai.configure(api_key=GEMINI_API_KEY)

def ask_gemini_api(user_input):
    prompt = f"""
    Kamu adalah chatbot untuk membantu mahasiswa memeriksa kemiripan judul skripsi mereka dan tugasmu adalah menjawab pertanyaan seputar skripsi, memberikan penilaian terhadap kemiripan judul, serta menyarankan perbaikan atau rekomendasi topik skripsi jika diminta.
    Judul skripsi yang sedang dicek: "{st.session_state.judul_user}". 
    Bidang minat terdeteksi: "{st.session_state.bidang_prediksi}". 
    Judul-judul skripsi yang mirip: {st.session_state.similar_titles}. 
    Pertanyaan pengguna: "{user_input}". 
    Jawablah dengan relevan sesuai konteks.
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        if response and response.text:
            # Bersihkan jika ada kata pembuka seperti "Jawaban:"
            clean_output = re.sub(r"(?i)^\s*(jawaban|contoh jawaban)\s*[:：-]\s", "", response.text.strip())
            return clean_output
        else:
            return "Maaf, aku tidak menerima jawaban dari Gemini."
    except Exception as e:
        return f"⚠ Error dari Gemini API: {str(e)}"

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv('dataset AI skripsi - Sheet1.csv')
    return df

df = load_data()

# ---------- Clean Function ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# ---------- Preprocess Gabungan Text ----------
df['gabungan'] = (df['Judul Skripsi'].fillna('') + ' ' + df['Metode Penelitian'].fillna('')).apply(clean_text)

# ---------- Train Bidang Minat Classifier ----------
# Pastikan kolom 'Bidang Minat' tidak kosong
df_latih = df[df['Bidang Minat'].notnull()]
classifier_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
classifier_model.fit(df_latih['gabungan'], df_latih['Bidang Minat'])

# ---------- TF-IDF Global Vectorizer (tetap pakai ini buat nanti per bidang) ----------
vectorizer = TfidfVectorizer()
vectorizer.fit(df['gabungan'])

# ---------- Prediksi Bidang Minat ----------
def prediksi_bidang_minat(judul, metode=''):
    gabungan_input = clean_text(judul + ' ' + metode)
    return classifier_model.predict([gabungan_input])[0]

# ---------- Cek Kemiripan Berdasarkan Bidang Minat ----------
def cek_kemiripan(judul, metode='', threshold=0.01):
    gabungan_input = clean_text(judul + ' ' + metode)
    input_vec = vectorizer.transform([gabungan_input])

    bidang_prediksi = prediksi_bidang_minat(judul, metode)
    df_filtered = df[df['Bidang Minat'] == bidang_prediksi].copy()

    if df_filtered.empty:
        return 0, [], [], bidang_prediksi, 0

    tfidf_filtered = vectorizer.transform(df_filtered['gabungan'])
    similarity_scores = cosine_similarity(input_vec, tfidf_filtered)[0]
    df_filtered['score'] = similarity_scores

    # Filter hanya yang di atas threshold
    df_mirip = df_filtered[df_filtered['score'] >= threshold].sort_values('score', ascending=False)

    # Ambil semua judul dan metode yang lolos threshold
    titles = df_mirip['Judul Skripsi'].tolist()
    methods = df_mirip['Metode Penelitian'].tolist()
    
    if not df_mirip.empty:
        max_score = df_mirip['score'].max()
    else:
        max_score = 0

    # Ambil judul dengan skor tertinggi
    if not df_mirip.empty:
        top_title = df_mirip.iloc[0]['Judul Skripsi']
    else:
        top_title = None
    
    return df_mirip['score'].tolist(), titles, methods, bidang_prediksi, max_score, top_title

intent_synonyms = {
    "greeting": ["halo", "hai", "pagi", "siang", "malam"],
    "goodbye": ["bye", "dadah", "sampai jumpa", "selamat tinggal"],
    "thanks": ["terima kasih", "makasih", "thanks", "thx"],
    "cek_kemiripan": ["persentase", "kemiripan", "similaritas", "sama", "seberapa"],
    "top_mirip": ["mirip", "semirip apa", "top mirip", "paling mirip"],
    "metode_mirip": ["metode", "pakai metode apa", "metode apa"],
    "rekomendasi": ["saran", "ganti judul", "unik atau tidak", "rekomendasi"],
    "help": ["bisa apa", "bantu apa", "fitur", "apa yang bisa"],
    "tahun_terbit": ["tahun", "kapan terbit", "tahun terbit", "tahun skripsi"],
    "pembimbing": ["dosen pembimbing", "siapa pembimbing", "pembimbingnya"],
    "mahasiswa": ["mahasiswa", "nama mahasiswa", "siapa yang buat", "penulis"]
}

# ---------- Intent Deteksi Sederhana ----------
def detect_intent(text):
    text = clean_text(text)
    highest_score = 0
    detected_intent = "unknown"

    for intent, keywords in intent_synonyms.items():
        for keyword in keywords:
            # Cek keseluruhan kalimat (frasa)
            score_phrase = fuzz.ratio(text, keyword)
            # Cek per kata dalam kalimat
            score_word = max(fuzz.ratio(word, keyword) for word in text.split())

            # Ambil score tertinggi
            score = max(score_phrase, score_word)

            if score > highest_score and score > 80:  # threshold 80%
                highest_score = score
                detected_intent = intent

    return detected_intent

# ---------- Streamlit Layout ----------
# ---------- Sidebar ----------
with st.sidebar:
    st.title("📚 Chatbot Title Match")
    st.markdown("👩‍🎓 Dibuat untuk mahasiswa Teknik Informatika UNPAD yang butuh cek judul skripsi.")
    st.markdown("---")
    st.markdown("### ℹ Tips Penggunaan:")
    st.markdown("""
    - Masukkan judul skripsi terlebih dahulu.
    - Tanyakan hal-hal seperti:
        - "Apakah ada judul yang mirip?"
        - "Seberapa mirip judulku?"
        - "Apa metode yang dipakai?"
        - "Siapa dosen pembimbingnya?"
        - "Bisa kasih saran langkah selanjutnya?"
    """)
    st.markdown("---")
    st.caption("Made by Latsa, Aifa, Ica")

# ---------- Halaman Utama ----------
st.markdown("<h1 style='text-align:center;'>🤖 Chatbot Title Match</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Cek kemiripan judul skripsi Teknik Informatika UNPAD dan dapatkan insight dari AI!</p>", unsafe_allow_html=True)
st.markdown("---")

# Inisialisasi state
if 'judul_user' not in st.session_state:
    st.session_state.judul_user = ''
if 'similarity_scores' not in st.session_state:
    st.session_state.similarity_scores = []
if 'similar_titles' not in st.session_state:
    st.session_state.similar_titles = []
if 'similar_methods' not in st.session_state:
    st.session_state.similar_methods = []
if 'bidang_prediksi' not in st.session_state:
    st.session_state.bidang_prediksi = ''
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Input Judul Awal
# ---------- Input Judul ----------
with st.container():
    with st.expander("📥 Masukkan Judul Skripsi Kamu", expanded=not st.session_state.judul_user):
        judul_input = st.text_input("📝 Judul Skripsi")
        metode_input = st.text_input("🔬 Metode Penelitian (Opsional)")
        if st.button("🚀 Submit Judul"):
            if judul_input.strip():
                st.session_state.judul_user = judul_input
                scores, titles, methods, bidang, max_score, top_title = cek_kemiripan(judul_input, metode_input)
                st.session_state.similarity_scores = scores
                st.session_state.similar_titles = titles
                st.session_state.similar_methods = methods
                st.session_state.bidang_prediksi = bidang
                st.session_state.max_similarity = max_score
                st.session_state.top_title = top_title
                st.success(f"✅ Judul berhasil diproses! Bidang Minat: {bidang}.Sekarang kamu bisa tanya-tanya ke chatbot 👇")

# ---------- Ringkasan Hasil ----------
if st.session_state.judul_user:
    st.markdown("### 📊 Hasil Analisis Judul")
    st.metric("📈 Kemiripan Tertinggi", f"{st.session_state.max_similarity*100:.2f}%")
    st.markdown("🎯 Judul Paling Mirip:")
    st.write(st.session_state.top_title or "Tidak ditemukan")
    st.markdown("---")

# Riwayat Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Tanya sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.judul_user:
            bot_msg = "Kamu perlu masukkan judul skripsi dulu ya sebelum nanya-nanya 😊"
        else:
            intent = detect_intent(prompt)
            if intent == "cek_kemiripan":
                bot_msg = (
                    f"📊 Kemiripan tertinggi: {st.session_state.max_similarity * 100:.2f}%  \n"
                    f"🎯 Judul yang paling mirip: {st.session_state.top_title}")
            elif intent == "top_mirip":
                if st.session_state.similar_titles:
                    bot_msg = (
                        f"🎯 Judul paling mirip di bidang {st.session_state.bidang_prediksi}:\n" +
                        "\n".join(
                            f"- {j} ({st.session_state.similarity_scores[i]*100:.2f}%)"
                            for i, j in enumerate(st.session_state.similar_titles)
                            if st.session_state.similarity_scores[i] >= 0.01
                        )
                    )
                else:
                    bot_msg = "🔍 Tidak ditemukan judul yang cukup mirip di bidang itu."
            elif intent == "metode_mirip":
                bot_msg = (
                    "🔧 Metode dari judul-judul yang mirip:\n" +
                    "\n".join(
                        f"- {st.session_state.similar_titles[i]}: {st.session_state.similar_methods[i]}"
                        for i in range(len(st.session_state.similar_titles))
                        if st.session_state.similarity_scores[i] >= 0.01
                    )
                )
            elif intent == "rekomendasi":
                threshold = 0.8  # nilai ambang kemiripan

                if st.session_state.max_similarity > threshold:
                    judul_user = st.session_state.judul_user
                    # Judul terlalu mirip, minta Gemini kasih saran judul baru
                    prompt = f"""Judul skripsi berikut ini terlalu mirip dengan yang sudah ada:
                    "{judul_user}"

                    Berikan beberapa alternatif judul skripsi baru yang berbeda dari judul tersebut, tapi masih di bidang yang sama (atau bisa juga bidang yang relevan). Sertakan alasan mengapa topik baru tersebut layak diteliti."""
                
                else:
                    judul_user = st.session_state.judul_user
                    # Judul cukup unik, minta Gemini bantu mengembangkan idenya
                    prompt = f"""Judul skripsi ini cukup unik:
                    "{judul_user}"

                    Berikan saran langkah-langkah atau arahan pengembangan ide skripsi ini. Misalnya: pendekatan metode penelitian, dataset yang bisa dipakai, studi kasus potensial, atau referensi awal."""

                # Kirim prompt ke Gemini dan dapatkan balasannya
                bot_msg = ask_gemini_api(prompt)
            elif intent == "help":
                bot_msg = (
                    "🧠 Aku bisa bantu kamu:\n"
                    "- Cek kemiripan judul skripsi\n"
                    "- Tampilkan judul paling mirip\n"
                    "- Tampilkan metode dari skripsi yang mirip\n"
                    "- Beri saran apakah perlu ganti judul atau tidak"
                )

            elif intent == "tahun_terbit":
                bot_msg = (
                    "📅 Tahun terbit dari judul-judul yang mirip:\n" +
                    "\n".join(
                        f"- {st.session_state.similar_titles[i]}: {df[df['Judul Skripsi'] == st.session_state.similar_titles[i]]['Tahun'].values[0]}"
                        for i in range(len(st.session_state.similar_titles))
                        if st.session_state.similarity_scores[i] >= 0.01 and
                        not df[df['Judul Skripsi'] == st.session_state.similar_titles[i]]['Tahun'].empty
                    )
                )

            elif intent == "pembimbing":
                bot_msg = (
                    "👨‍🏫 Dosen pembimbing dari judul-judul yang mirip:\n" +
                    "\n".join(
                        f"- {st.session_state.similar_titles[i]}: {df[df['Judul Skripsi'] == st.session_state.similar_titles[i]]['Pembimbing'].values[0]}"
                        for i in range(len(st.session_state.similar_titles))
                        if st.session_state.similarity_scores[i] >= 0.01 and
                        not df[df['Judul Skripsi'] == st.session_state.similar_titles[i]]['Pembimbing'].empty
                    )
                )

            elif intent == "mahasiswa":
                bot_msg = (
                    "🧑‍🎓 Nama mahasiswa dari judul-judul yang mirip:\n" +
                    "\n".join(
                        f"- {st.session_state.similar_titles[i]}: {df[df['Judul Skripsi'] == st.session_state.similar_titles[i]]['Mahasiswa'].values[0]}"
                        for i in range(len(st.session_state.similar_titles))
                        if st.session_state.similarity_scores[i] >= 0.01 and
                        not df[df['Judul Skripsi'] == st.session_state.similar_titles[i]]['Mahasiswa'].empty
                    )
                )

            elif intent == "greeting":
                bot_msg = "👋 Halo! Siap bantu cek judul skripsi kamu."
            elif intent == "goodbye":
                bot_msg = "🤗 Terima kasih juga! Semangat ngerjain skripsinya yaa!"
            elif intent == "thanks":
                bot_msg = "Sama-sama! Semangat terus yaa ✨"
            else:
                bot_msg = ask_gemini_api(prompt)

        st.markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
