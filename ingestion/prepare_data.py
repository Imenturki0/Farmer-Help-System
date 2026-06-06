import json
from pathlib import Path
import fitz
import re


# -------------------------
# CLEAN PDF TEXT
# -------------------------
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    # -------------------------
    # CLEANING (IMPORTANT FIXES)
    # -------------------------

    # fix broken hyphen words
    text = re.sub(r"-\n", "", text)

    # normalize spacing/newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # remove page headers like "XVIII.No.1B"
    text = re.sub(r"(?m)^\s*[IVXLC]+\..*$", "", text)

    # remove table titles (light filtering)
    text = re.sub(r"Table\s*\d+.*", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------------------------
# SENTENCE SPLITTING (IMPROVED)
# -------------------------
def split_sentences(text):
    # better PDF-safe sentence split
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 20
    ]


# -------------------------
# SMART CHUNKING WITH OVERLAP (IMPORTANT FIX)
# -------------------------
def chunk_sentences(sentences, max_words=180, min_words=60, overlap=2):
    chunks = []
    current = []
    current_len = 0

    for s in sentences:
        w = len(s.split())

        # if chunk too big → finalize
        if current_len + w > max_words:

            if current_len >= min_words:
                chunks.append(" ".join(current))

            # 🔥 OVERLAP: keep last sentences for context continuity
            current = current[-overlap:] if overlap > 0 else []
            current_len = sum(len(x.split()) for x in current)

        current.append(s)
        current_len += w

    # last chunk
    if current and current_len >= min_words:
        chunks.append(" ".join(current))

    return chunks


# -------------------------
# PIPELINE
# -------------------------
pdf_folder = Path("data/raw/pdfs")

docs = []

for pdf_file in pdf_folder.glob("*.pdf"):

    print(f"Processing: {pdf_file.name}")

    text = extract_pdf_text(pdf_file)

    sentences = split_sentences(text)
    chunks = chunk_sentences(sentences)

    for c in chunks:
        docs.append({
            "text": c,
            "source": pdf_file.name,
            "type": "farming_knowledge",
            "clean": True
        })


# -------------------------
# SAVE OUTPUT
# -------------------------
output_path = "data/processed/chunks.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2)

print(f"\nSaved {len(docs)} CLEAN + HIGH QUALITY chunks")