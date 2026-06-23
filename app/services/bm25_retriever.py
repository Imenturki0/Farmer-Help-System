from rank_bm25 import BM25Okapi
import json
import re
import numpy as np


class BM25Retriever:

    def __init__(self):
        self.docs = []
        self.metadata = []
        self.bm25 = None

    def load(self, path):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.metadata = data

        self.docs = [x["text"] for x in data]

        tokenized = [
            re.findall(r"\w+", doc.lower())
            for doc in self.docs
        ]

        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=10):

        tokens = re.findall(r"\w+", query.lower())

        scores = np.array(self.bm25.get_scores(tokens))

        top_k = scores.argsort()[-k:][::-1]

        return [
            {
                "text": self.docs[i],

                "chunk_id":
                    self.metadata[i]["chunk_id"],

                "source":
                    self.metadata[i].get("source"),

                "bm25_score":
                    float(scores[i])
            }
            for i in top_k
        ]