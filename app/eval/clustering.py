from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def group_chunks_by_embedding(chunks, n_clusters=20):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    labels = kmeans.fit_predict(embeddings)

    grouped = {}

    for label, chunk in zip(labels, chunks):
        grouped.setdefault(label, []).append(chunk)

    return list(grouped.values())