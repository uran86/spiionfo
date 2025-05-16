import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
import sqlite3
import requests
from bs4 import BeautifulSoup
import threading
import datetime
import time

DB_NAME = 'spiinfo.db'

# --- Databasfunktioner ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS snipps (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    summary TEXT,
                    source TEXT,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

def save_snipp(content, summary, source):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO snipps (content, summary, source, timestamp) VALUES (?, ?, ?, ?)",
              (content, summary, source, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_all_snipps():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT content, summary, source, timestamp FROM snipps ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# --- Textbearbetning (enkel sammanfattning) ---
def summarize_text(text):
    lines = text.strip().split('. ')
    return '. '.join(lines[:2]) + ('.' if len(lines) > 2 else '')

# --- Hämta text från URL ---
def fetch_url_text(url):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception as e:
        return f"Fel vid hämtning: {e}"

# --- GUI ---
class SpiInfoApp:
    def __init__(self, root):
        self.root = root
        root.title("SPIINFO v1 - Ditt superminne")

        self.text_input = scrolledtext.ScrolledText(root, width=60, height=10)
        self.text_input.pack(padx=10, pady=10)

        self.url_button = tk.Button(root, text="Hämta från URL", command=self.get_from_url)
        self.url_button.pack(pady=(0,5))

        self.save_button = tk.Button(root, text="Minns detta", command=self.save_current_snipp)
        self.save_button.pack()

        self.show_button = tk.Button(root, text="Visa alla snippar", command=self.show_snipps)
        self.show_button.pack(pady=(5,10))

        self.start_clipboard_monitor()

    def get_from_url(self):
        url = simpledialog.askstring("Ange URL", "Klistra in en webbadress:")
        if url:
            text = fetch_url_text(url)
            self.text_input.delete("1.0", tk.END)
            self.text_input.insert(tk.END, text)

    def save_current_snipp(self):
        content = self.text_input.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Ingen text", "Skriv eller hämta något först.")
            return
        summary = summarize_text(content)
        save_snipp(content, summary, "Manuell/URL")
        messagebox.showinfo("Sparat", "Snipp sparad i ditt minne!")
        self.text_input.delete("1.0", tk.END)

    def show_snipps(self):
        snipps = get_all_snipps()
        new_window = tk.Toplevel(self.root)
        new_window.title("Dina snippar")

        text_area = scrolledtext.ScrolledText(new_window, width=80, height=20)
        text_area.pack(padx=10, pady=10)

        for s in snipps:
            content, summary, source, timestamp = s
            text_area.insert(tk.END, f"[{timestamp}] ({source})\n{summary}\n---\n")

    def start_clipboard_monitor(self):
        def monitor():
            last_text = ""
            while True:
                try:
                    current = self.root.clipboard_get()
                    if current != last_text and len(current) > 20:
                        summary = summarize_text(current)
                        save_snipp(current, summary, "Clipboard")
                        print(f"[SPIINFO] Ny snipp från clipboard sparad.")
                        last_text = current
                except:
                    pass
                time.sleep(2)

        threading.Thread(target=monitor, daemon=True).start()

# --- Starta appen ---
if __name__ == '__main__':
    init_db()
    root = tk.Tk()
    app = SpiInfoApp(root)
    root.mainloop()
