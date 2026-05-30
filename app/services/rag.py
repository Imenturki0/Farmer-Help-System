from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os


class FAISSRAG:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs = []
        self.index = None

    def load_docs(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.docs = [line.strip() for line in f.readlines()]

    def build_index(self, index_path="faiss.index"):
    
        # 🟢 IF INDEX EXISTS → LOAD IT
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print("Loaded FAISS index from disk")
            return

        # 🔴 ELSE → BUILD IT
        embeddings = self.model.encode(
            self.docs,
            normalize_embeddings=True
        )

        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # 💾 SAVE IT
        faiss.write_index(self.index, index_path)

        print("Built and saved FAISS index")



    def search(self, query, k=5, threshold=0.3):
        q_vec = self.model.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec).astype("float32")

        scores, indices = self.index.search(q_vec, k)

        results = []
        for i, score in zip(indices[0], scores[0]):
            if i != -1 and score > threshold:
                results.append(self.docs[i])

        return results
    


# GLOBAL INSTANCE (SAFE NOW)
rag = FAISSRAG()