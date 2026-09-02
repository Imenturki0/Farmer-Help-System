from typing import Any

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config.settings import settings


class QdrantVectorDB:
    """
    Small production-oriented wrapper around Qdrant.

    Responsibilities:
    - Connect to Qdrant
    - Create collection
    - Upsert document chunks
    - Search vectors
    - Delete chunks by source
    """

    def __init__(
        self,
        collection: str = "farming_docs",
        host: str | None = None,
        port: int | None = None,
    ):

        self.collection = collection

        self.host = host or settings.qdrant.host
        self.port = port or settings.qdrant.port

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
        )

    # ========================================================
    # COLLECTION
    # ========================================================

    def collection_exists(self) -> bool:

        collections = (
            self.client
            .get_collections()
            .collections
        )

        return any(
            collection.name == self.collection
            for collection in collections
        )

    def create_collection(
        self,
        dimension: int,
    ) -> None:

        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )

    # ========================================================
    # UPSERT
    # ========================================================

    def upsert_chunks(
        self,
        embedder,
        chunks: list[dict[str, Any]],
        source: str,
    ) -> None:
        """
        Embed and upsert chunks into Qdrant.

        Chunk IDs are stable and come from the
        ingestion pipeline.
        """

        if not chunks:
            return

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.create_collection(
            embeddings.shape[1]
        )

        points = []

        for chunk, vector in zip(
            chunks,
            embeddings,
        ):

            chunk_id = chunk["chunk_id"]

            # Qdrant point IDs must be
            # integers or UUIDs.
            #
            # Therefore we keep the human-readable
            # chunk ID in the payload and create a
            # deterministic UUID from it.
            point_id = self._point_id(
                chunk_id
            )

            payload = {
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "pages": chunk.get(
                    "pages",
                    [],
                ),
                "source": source,
                "title": chunk.get(
                    "title"
                ),
                "section": chunk.get(
                    "section"
                ),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        vector: np.ndarray,
        k: int = 20,
    ) -> list[dict[str, Any]]:

        if not self.collection_exists():
            return []

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=k,
        )

        results = []

        for point in response.points:

            payload = point.payload or {}

            results.append(
                {
                    "text": payload.get(
                        "text",
                        "",
                    ),
                    "chunk_id": payload.get(
                        "chunk_id"
                    ),
                    "source": payload.get(
                        "source"
                    ),
                    "pages": payload.get(
                        "pages",
                        [],
                    ),
                    "title": payload.get(
                        "title"
                    ),
                    "section": payload.get(
                        "section"
                    ),
                    "vector_score": float(
                        point.score
                    ),
                }
            )

        return results

    # ========================================================
    # DELETE
    # ========================================================

    def delete_by_source(
        self,
        source: str,
    ) -> None:

        if not self.collection_exists():
            return

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(
                            value=source
                        ),
                    )
                ]
            ),
        )

    # ========================================================
    # DETERMINISTIC POINT ID
    # ========================================================

    @staticmethod
    def _point_id(
        chunk_id: str,
    ) -> str:
        """
        Convert our stable chunk ID into a
        deterministic UUID.

        Same chunk_id → same Qdrant point ID.
        """

        import uuid

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk_id,
            )
        )