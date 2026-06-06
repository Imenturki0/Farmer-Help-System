from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os


class FAISSRAG:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.docs = []
        self.metadata = []
        self.index = None
        emb = self.model.encode(["hello world"])

        print(emb.shape)

    def load_docs(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.docs = [item["text"] for item in data]
        self.metadata = data

        print(f"Loaded {len(self.docs)} clean chunks")

    def build_index(self, index_path="vector_db/faiss.index"):

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print("Loaded FAISS index from disk")
            return

        print("Creating embeddings...")

        embeddings = self.model.encode(
            self.docs,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        os.makedirs("vector_db", exist_ok=True)
        faiss.write_index(self.index, index_path)

        print("Built and saved FAISS index")

    def search(self, query, k=5, threshold=0.45):  # 🔥 FIXED THRESHOLD

        q_vec = self.model.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec).astype("float32")

        scores, indices = self.index.search(q_vec, k)

        results = []
        best_score = 0.0

        for idx, score in zip(indices[0], scores[0]):

            if idx == -1:
                continue

            score = float(score)
            best_score = max(best_score, score)

            # 🔥 stricter filtering
            if score < threshold:
                continue

            results.append({
                "text": self.docs[idx],
                "source": self.metadata[idx]["source"],
                "score": score
            })

        return results, best_score


rag = FAISSRAG()