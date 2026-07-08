from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)
import uuid


class QdrantVectorDB:

    def __init__(
        self,
        collection="farming_docs"
    ):

        self.client = QdrantClient(
            host="localhost",
            port=6333
        )

        self.collection = collection


    def create_collection(
        self,
        dimension
    ):

        collections = (
            self.client
            .get_collections()
            .collections
        )

        names = [
            c.name
            for c in collections
        ]


        if self.collection not in names:

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )


    def add_documents(
        self,
        embeddings,
        metadata
    ):

        points=[]


        for emb, meta in zip(
            embeddings,
            metadata
        ):

            points.append(
                PointStruct(
                    # id=str(uuid.uuid4()),
                    id=meta["chunk_id"],
                    vector=emb.tolist(),
                    payload=meta
                )
            )


        self.client.upsert(
            collection_name=self.collection,
            points=points
        )


    def search(
            
        self,
        vector,
        k=20
    ):

        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=k
        )


        return [
            {
                "text": r.payload["text"],
                "chunk_id": r.payload["chunk_id"],
                "source": r.payload["source"],
                "faiss_score": float(r.score)
            }
            for r in results
        ]
    
    def collection_exists(self):
        collections = self.client.get_collections().collections
        return self.collection in [c.name for c in collections]
    
    def upsert_chunks(self,embedder, chunks, source):

        points =[]

        for c in chunks:
            vec = embedder.encode(c["text"], normalize_embeddings=True)

            points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": c["text"],
                    "pages": c["pages"],
                    "source": source
                    }
                )
            )

        self.client.upsert(
        collection_name=self.collection,
        points=points
    )
    
    def delete_by_source(self, source):
        self.client.delete(
        collection_name=self.collection,
        points_selector={
            "filter": {
                "must": [
                    {
                        "key": "source",
                        "match": {"value": source}
                    }
                ]
            }
        }
    )
            

    