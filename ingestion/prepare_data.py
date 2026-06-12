import json
from pathlib import Path
from collections import defaultdict
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================
# MODEL
# =========================
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# =========================
# 1. PDF EXTRACTION
# =========================
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []
    freq = defaultdict(int)

    for page in doc:
        text = page.get_text("text")

        # remove weird control chars (VERY IMPORTANT FIX)
        text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        pages.append(lines)

        for l in set(lines):
            freq[l] += 1

    return pages, freq, len(doc)


# =========================
# 2. NOISE DETECTION
# =========================
def detect_noise_lines(freq, total_pages, threshold=0.6):
    return {
        line for line, f in freq.items()
        if f / total_pages >= threshold
    }


# =========================
# 3. CLEAN PAGES
# =========================
def clean_pages(pages, noise_lines):
    out = []

    for lines in pages:
        filtered = []

        for line in lines:
            line = line.strip()

            if line in noise_lines:
                continue

            if re.fullmatch(r"\d{1,4}", line):
                continue

            if len(line) < 3:
                continue

            filtered.append(line)

        out.append(" ".join(filtered))

    return "\n".join(out)


# =========================
# 4. NORMALIZE TEXT
# =========================
def normalize_text(text):
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)

    # fix broken unicode artifacts like \u0001 garbage
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", text)

    return text.strip()


# =========================
# 5. SENTENCE SPLIT (IMPROVED)
# =========================
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cleaned = []
    for s in sentences:
        s = s.strip()

        if len(s) < 25:
            continue

        if len(re.findall(r"[a-zA-Z]", s)) < 5:
            continue

        cleaned.append(s)

    return cleaned


# =========================
# 6. EMBEDDING
# =========================
def embed(sentences):
    return model.encode(sentences, normalize_embeddings=True)


# =========================
# 7. SEMANTIC CHUNKING (FIXED STABLE VERSION)
# =========================
def semantic_chunk(sentences, threshold=0.78, max_words=180):
    embeddings = embed(sentences)

    chunks = []

    current_chunk = [sentences[0]]
    current_embs = [embeddings[0]]

    # FIX: use incremental centroid (more stable than np.mean each time)
    centroid = embeddings[0].copy()

    word_count = len(sentences[0].split())

    for i in range(1, len(sentences)):
        sent = sentences[i]
        emb = embeddings[i]

        sim = np.dot(centroid, emb)
        sent_words = len(sent.split())

        # chunk break conditions
        if sim < threshold or word_count + sent_words > max_words:

            chunks.append(" ".join(current_chunk))

            # reset
            current_chunk = [sent]
            current_embs = [emb]
            centroid = emb.copy()
            word_count = sent_words

        else:
            current_chunk.append(sent)
            current_embs.append(emb)

            # incremental centroid update (IMPORTANT FIX)
            centroid = (centroid * (len(current_embs) - 1) + emb) / len(current_embs)
            centroid = centroid / np.linalg.norm(centroid)

            word_count += sent_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# =========================
# 8. DEDUP (FIXED REAL SEMANTIC VERSION)
# =========================
def deduplicate_chunks(chunks, threshold=0.93):
    if not chunks:
        return []

    embeddings = embed(chunks)

    kept = []
    used = set()

    for i in range(len(chunks)):
        if i in used:
            continue

        base = chunks[i]
        used.add(i)

        for j in range(i + 1, len(chunks)):
            if j in used:
                continue

            sim = np.dot(embeddings[i], embeddings[j])

            if sim >= threshold:
                used.add(j)

        kept.append(base)

    return kept


# =========================
# PIPELINE
# =========================
def process_pdf(pdf_path):
    pages, freq, total_pages = extract_pages(pdf_path)

    noise = detect_noise_lines(freq, total_pages)

    raw = clean_pages(pages, noise)
    text = normalize_text(raw)

    sentences = split_sentences(text)

    if len(sentences) == 0:
        return []

    chunks = semantic_chunk(sentences)

    chunks = deduplicate_chunks(chunks)

    return chunks


# =========================
# RUN
# =========================
pdf_folder = Path("data/raw/pdfs")
docs = []

for pdf_file in pdf_folder.glob("*.pdf"):
    print(f"Processing: {pdf_file.name}")

    chunks = process_pdf(pdf_file)

    for idx, c in enumerate(chunks):
        docs.append({
            "text": c,
            "source": pdf_file.name,
            "type": "farming_knowledge",
            "chunk_id": idx,
            "clean": True
        })


# =========================
# SAVE
# =========================
output_path = "data/processed/chunks.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2)

print(f"\nSaved {len(docs)} semantic chunks")