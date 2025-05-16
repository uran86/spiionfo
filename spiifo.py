import streamlit as st
import requests
from bs4 import BeautifulSoup
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initiera modellen och databasfil
model = SentenceTransformer('all-MiniLM-L6-v2')
DB_PATH = "spiinfo_db.pkl"

# Ladda/spara databas
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {"texts": [], "embeddings": [], "urls": []}

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

def add_knowledge(text, url=""):
    embedding = model.encode(text)
    database["texts"].append(text)
    database["embeddings"].append(embedding)
    database["urls"].append(url)
    save_db(database)

def search_knowledge(query, top_k=5):
    if not database["embeddings"]:
        return []
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], database["embeddings"])[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(database["texts"][i], similarities[i], database["urls"][i]) for i in top_indices]

# Web scraping-funktion
def scrape_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Ta bort script och style
        for script in soup(["script", "style"]):
            script.decompose()

        # Hämta text och rensa whitespace
        text = soup.get_text(separator=' ')
        text = ' '.join(text.split())
        return text[:5000]  # Begränsa max textlängd för prestanda
    except Exception as e:
        st.error(f"Misslyckades att hämta URL: {e}")
        return ""

# --- App start ---
st.title("SPIINFO v4 - Web Scraping och AI-minne")

database = load_db()

menu = st.sidebar.selectbox("Välj funktion", [
    "➕ Lägg till kunskap från URL",
    "❓ Fråga SPIINFO",
    "📂 Visa minnet"
])

if menu == "➕ Lägg till kunskap från URL":
    st.subheader("Lägg till text från webbsida")
    url = st.text_input("Ange URL att skrapa")
    if st.button("Hämta och minns"):
        if url:
            scraped_text = scrape_text_from_url(url)
            if scraped_text:
                st.text_area("Inhämtad text (förhandsvisning):", scraped_text, height=300)
                add_knowledge(scraped_text, url)
                st.success("SPIINFO har lärt sig innehållet från sidan!")
        else:
            st.warning("Ange en giltig URL.")

elif menu == "❓ Fråga SPIINFO":
    st.subheader("Fråga din AI")
    query = st.text_input("Skriv din fråga här")
    if query:
        results = search_knowledge(query)
        if results:
            for i, (text, score, url) in enumerate(results):
                st.markdown(f"### Resultat {i+1} (Likhet: {score:.2f})")
                st.markdown(f"URL: {url}")
                st.write(text[:1000] + "...")
        else:
            st.info("SPIINFO har inget minne än.")

elif menu == "📂 Visa minnet":
    st.subheader("All text sparad i SPIINFO")
    for i, (text, url) in enumerate(zip(database["texts"], database["urls"])):
        st.markdown(f"---\n**{i+1}. URL:** {url}\n{text[:500]}...")

