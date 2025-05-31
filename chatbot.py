import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import requests

TOGETHER_API_KEY = "dbca14bfa35585175c337ca19b242e4f5b765a8bbe6aa0d9d555b32750853da7"
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def ask_together_ai(prompt):
    url = "https://api.together.xyz/inference"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": TOGETHER_MODEL,
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        try:
            json_response = response.json()
            return json_response["choices"][0]["text"]
        except (KeyError, IndexError):
            return "Maaf, aku tidak bisa memproses hasil Together AI."
    else:
        return f"⚠ Error dari Together AI: {response.status_code} - {response.text}"


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

# ---------- Intent Deteksi Sederhana ----------
def detect_intent(text):
    text = clean_text(text)
    if any(word in text for word in ["halo", "hai", "pagi", "siang", "malam"]):
        return "greeting"
    elif any(word in text for word in ["bye", "dadah", "sampai jumpa"]):
        return "goodbye"
    elif any(word in text for word in ["terima kasih", "makasih"]):
        return "thanks"
    elif any(word in text for word in ["persentase", "kemiripan", "similaritas", "seberapa", "berapa"]):
        return "cek_kemiripan"
    elif any(word in text for word in ["mirip", "mirip apa", "top mirip"]):
        return "top_mirip"
    elif any(word in text for word in ["metode mirip", "pakai metode apa"]):
        return "metode_mirip"
    elif any(word in text for word in ["saran", "ganti judul", "unik atau tidak"]):
        return "rekomendasi"
    elif any(word in text for word in ["bisa apa", "bantu apa", "fitur"]):
        return "help"
    elif any(word in text for word in ["tahun", "kapan terbit", "tahun terbit", "tahun skripsi"]):
        return "tahun_terbit"
    elif any(word in text for word in ["dosen pembimbing", "siapa pembimbing", "pembimbingnya"]):
        return "pembimbing"
    elif any(word in text for word in ["mahasiswa", "nama mahasiswa", "siapa yang buat", "penulis"]):
        return "mahasiswa"
    else:
        return "unknown"

# ---------- Streamlit Layout ----------
st.title("🎓 Chatbot Skripsi AI")
st.markdown("Masukkan judul skripsi kamu dan ngobrol langsung sama chatbot buat cek kemiripan!")

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
with st.expander("📝 Masukkan Judul Skripsi Kamu Dulu", expanded=not st.session_state.judul_user):
    judul_input = st.text_input("Judul Skripsi:")
    metode_input = st.text_input("Metode Penelitian (opsional):")
    if st.button("Submit Judul"):
        if judul_input.strip():
            st.session_state.judul_user = judul_input
            scores, titles, methods, bidang, max_score, top_title = cek_kemiripan(judul_input, metode_input)
            st.session_state.similarity_scores = scores
            st.session_state.similar_titles = titles
            st.session_state.similar_methods = methods
            st.session_state.bidang_prediksi = bidang
            st.session_state.max_similarity = max_score
            st.session_state.top_title = top_title
            st.success(f"Judul berhasil diproses! Bidang minat terdeteksi: {bidang}. Sekarang kamu bisa tanya-tanya ke chatbot 👇")

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
                if max(st.session_state.similarity_scores) > 0.8:
                    bot_msg = "💡 Judul kamu terlalu mirip dengan yang sudah ada di bidang yang sama. Coba ganti topik atau metode penelitian."
                else:
                    bot_msg = "✅ Judul kamu cukup unik dalam bidang itu. Lanjutkan dan kembangkan idemu!"
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
                bot_msg = ask_together_ai(prompt)

        st.markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})