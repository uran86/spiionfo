import streamlit as st
import requests, os, pickle
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import trafilatura
from dotenv import load_dotenv
import openai

# Ladda miljövariabler
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

openai.api_key = OPENAI_API_KEY
model = SentenceTransformer('all-MiniLM-L6-v2')
DB_PATH = "spiinfo_v6_db.pkl"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {"texts": [], "embeddings": [], "urls": []}

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

db = load_db()

def search_google(query):
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}&num=5"
    try:
        res = requests.get(url).json()
        return [r["link"] for r in res.get("organic_results", [])]
    except:
        return []

def scrape_and_summarize(url):
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla"}).text
        clean = trafilatura.extract(html)
        if clean and len(clean) > 500:
            return clean[:3000]
    except:
        return None

def embed_text(text): return model.encode(text)

def remember(text, url):
    db["texts"].append(text)
    db["embeddings"].append(embed_text(text))
    db["urls"].append(url)
    save_db(db)

def semantic_search(query, top_k=3):
    if not db["embeddings"]: return []
    query_vec = embed_text(query)
    sims = cosine_similarity([query_vec], db["embeddings"])[0]
    idxs = np.argsort(sims)[-top_k:][::-1]
    return [(db["texts"][i], sims[i], db["urls"][i]) for i in idxs]

def ask_openai(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Du är en smart AI-sökare."},
                  {"role": "user", "content": prompt}],
        temperature=0.6
    )
    return res["choices"][0]["message"]["content"]

# ----------------- STREAMLIT -----------------
st.title("🤖 SPIINFO v6 – AI Web Brain")
menu = st.sidebar.selectbox("Välj funktion", ["🧠 AI-sök", "📅 Skapa minne", "🔍 Fråga SPIINFO"])

if menu == "🧠 AI-sök":
    query = st.text_input("Vad ska SPIINFO söka efter?")
    if st.button("Sök och lär"):
        links = search_google(query)
        for url in links:
            content = scrape_and_summarize(url)
            if content:
                remember(content, url)
                st.markdown(f"✅ Sparade från [{url}]({url})")

elif menu == "📅 Skapa minne":
    url = st.text_input("Ange URL manuellt")
    if st.button("Hämta och spara"):
        text = scrape_and_summarize(url)
        if text:
            remember(text, url)
            st.text_area("Förhandsvisning", text[:1500])

elif menu == "🔍 Fråga SPIINFO":
    question = st.text_input("Din fråga")
    if question:
        docs = semantic_search(question)
        summary = "\n\n".join([f"Från {url}:\n{text}" for text, _, url in docs])
        answer = ask_openai(f"Fråga: {question}\n\nHär är informationen:\n{summary}")
        st.markdown("### 🧠 Svar från SPIINFO:")
        st.write(answer)
