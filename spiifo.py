# spiinfo_app.py
import streamlit as st
import os
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr

# --- Inställningar ---
model = SentenceTransformer('all-MiniLM-L6-v2')
DB_PATH = "spiinfo_local_db.pkl"

# --- Enkel lösenordsskydd ---
def authenticate():
    if "authenticated" not in st.session_state:
        pwd = st.text_input("Ange lösenord för att öppna SPIINFO:", type="password")
        if st.button("Logga in"):
            if pwd == "spi123":
                st.session_state.authenticated = True
                st.experimental_rerun()
            else:
                st.error("Fel lösenord!")
    return st.session_state.get("authenticated", False)

# --- Databasfunktioner ---
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {"texts": [], "embeddings": [], "tags": []}

def save_db():
    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)

def add_knowledge(text, tags):
    embedding = model.encode(text)
    database["texts"].append(text)
    database["embeddings"].append(embedding)
    database["tags"].append(tags)
    save_db()

def search_knowledge(query, top_k=5):
    if not database["embeddings"]:
        return []
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], database["embeddings"])[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(database["texts"][i], similarities[i], database["tags"][i]) for i in top_indices]

# --- Röstinmatning ---
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Lyssnar... prata nu")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio, language="sv-SE")
    except sr.UnknownValueError:
        st.error("Kunde inte förstå ljudet.")
    except sr.RequestError:
        st.error("Kunde inte kontakta rösttjänsten.")
    return ""

# --- Export / Import ---
def export_to_json():
    with open("spiinfo_export.json", "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)

def import_from_json():
    try:
        with open("spiinfo_export.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            database["texts"].extend(data["texts"])
            database["embeddings"].extend(data["embeddings"])
            database["tags"].extend(data.get("tags", [""] * len(data["texts"])))
            save_db()
    except FileNotFoundError:
        st.error("Ingen exportfil spiinfo_export.json hittades.")

# --- App start ---
if not authenticate():
    st.stop()

database = load_db()

st.set_page_config(page_title="🧠 SPIINFO v3 – Offline", layout="wide")
st.title("🧠 SPIINFO v3 – Offline AI-minne")

menu = st.sidebar.selectbox("Välj funktion", [
    "➕ Lägg till kunskap",
    "🎙️ Prata till SPIINFO",
    "❓ Fråga SPIINFO",
    "📂 Visa minne",
    "📤 Exportera",
    "📥 Importera"
])

if menu == "➕ Lägg till kunskap":
    st.subheader("Lägg till ny kunskap")
    user_input = st.text_area("Skriv eller klistra in text här:")
    tags = st.text_input("Lägg till taggar (t.ex. #historia #vetenskap):")
    if st.button("Minns detta") and user_input.strip():
        add_knowledge(user_input, tags)
        st.success("SPIINFO har lärt sig det här!")

elif menu == "🎙️ Prata till SPIINFO":
    st.subheader("Prata in ny kunskap")
    if st.button("🎤 Spela in"):
        spoken_text = recognize_speech()
        if spoken_text:
            st.text_area("Din inlästa text:", value=spoken_text, height=100)
            tags = st.text_input("Lägg till taggar:", key="voice_tags")
            if st.button("Minns detta", key="voice_save"):
                add_knowledge(spoken_text, tags)
                st.success("SPIINFO har lärt sig vad du sa!")

elif menu == "❓ Fråga SPIINFO":
    st.subheader("Fråga din AI")
    query = st.text_input("Vad vill du veta?")
    if query:
        results = search_knowledge(query)
        if results:
            st.markdown("### 💡 Svar från SPIINFO:")
            for i, (text, score, tags) in enumerate(results):
                st.markdown(f"**{i+1}.** Likhet: {score:.2f} – Taggar: {tags}\n{text}")
        else:
            st.info("SPIINFO har inget relevant minne än.")

elif menu == "📂 Visa minne":
    st.subheader("Alla snippar i minnet")
    tag_filter = st.text_input("Filtrera på tagg (valfritt):").lower()
    for i, text in enumerate(database["texts"]):
        if tag_filter in database["tags"][i].lower():
            st.markdown(f"---\n**{i+1}.** {text}\n_Taggar: {database['tags'][i]}_")

elif menu == "📤 Exportera":
    export_to_json()
    st.success("Minnet har exporterats till spiinfo_export.json")

elif menu == "📥 Importera":
    import_from_json()
    st.success("Minnet har importerats från spiinfo_export.json")
