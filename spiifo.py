import streamlit as st
import requests
from bs4 import BeautifulSoup
import trafilatura
import os, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
model = SentenceTransformer('all-MiniLM-L6-v2')
DB_FILE = "spiinfo_memory.pkl"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {"texts": [], "embeddings": [], "urls": []}

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

db = load_db()

def smart_scrape(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        html = requests.get(url, headers=headers, timeout=10).text

        # Försök extrahera relevant artikeltext
        result = trafilatura.extract(html, include_comments=False, include_tables=False)
        if result:
            return result
        else:
            # Fallback: ta med synlig text från <p> och <h> taggar
            soup = BeautifulSoup(html, "html.parser")
            text = " ".join(tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3']))
            return text[:5000]
    except Exception as e:
        st.error(f"Kunde inte hämta data: {e}")
        return ""

def remember(text, url):
    embedding = model.encode(text)
    db["texts"].append(text)
    db["embeddings"].append(embedding)
    db["urls"].append(url)
    save_db(db)

def semantic_search(query, top_k=5):
    if not db["embeddings"]:
        return []
    query_emb = model.encode(query)
    sims = cosine_similarity([query_emb], db["embeddings"])[0]
    idxs = np.argsort(sims)[-top_k:][::-1]
    return [(db["texts"][i], sims[i], db["urls"][i]) for i in idxs]

# 🧠 GUI
st.title("SPIINFO v5 – Smart Web Scraper AI")
menu = st.sidebar.radio("Välj", ["📥 Hämta från webben", "🔍 Fråga minnet", "🧾 Visa allt minne"])

if menu == "📥 Hämta från webben":
    url = st.text_input("Ange en URL att analysera")
    if st.button("Hämta & lär"):
        if url:
            scraped = smart_scrape(url)
            if scraped:
                st.success("Innehåll hämtat! Förhandsvisning nedan:")
                st.text_area("Utdrag", scraped[:1500])
                remember(scraped, url)
            else:
                st.warning("Kunde inte hämta vettigt innehåll.")
        else:
            st.warning("Ange en giltig URL.")

elif menu == "🔍 Fråga minnet":
    query = st.text_input("Vad vill du veta?")
    if query:
        results = semantic_search(query)
        if results:
            for i, (text, score, url) in enumerate(results):
                st.markdown(f"### {i+1}. Likhet: {score:.2f}")
                st.markdown(f"🔗 {url}")
                st.write(text[:1000] + "...")
        else:
            st.info("Inget sparat minne än.")

elif menu == "🧾 Visa allt minne":
    st.subheader("🧠 SPIINFOs minne")
    for i, (txt, url) in enumerate(zip(db["texts"], db["urls"])):
        st.markdown(f"---\n**{i+1} – {url}**\n{txt[:700]}...")
