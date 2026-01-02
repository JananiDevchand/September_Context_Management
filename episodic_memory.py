import hnswlib
import numpy as np
import os
from datetime import datetime
from pymongo import MongoClient
from embeddings import EmbeddingModel


class EpisodicMemory:
    def __init__(self, dim=384, max_elements=10000):
        self.dim = dim
        self.index_path = "data/episodic_hnsw.index"

        # ---- MongoDB ----
        self.client = MongoClient("mongodb://localhost:27017")
        self.db = self.client["memory_chatbot"]
        self.collection = self.db["episodic_memory"]

        # ---- Embedder (for rebuild) ----
        self.embedder = EmbeddingModel()

        # ---- HNSW ----
        self.index = hnswlib.Index(space="cosine", dim=dim)

        if os.path.exists(self.index_path):
            try:
                self.index.load_index(self.index_path)
                self.next_id = self.index.get_current_count()
                print(f"✅ Loaded Episodic HNSW ({self.next_id} episodes)")
            except RuntimeError:
                print("⚠️ Corrupted Episodic index detected. Rebuilding...")
                self._rebuild_from_mongo(max_elements)
        else:
            self._rebuild_from_mongo(max_elements)

    # --------------------------------------------------
    # REBUILD INDEX FROM MONGODB
    # --------------------------------------------------
    def _rebuild_from_mongo(self, max_elements):
        print("🔁 Rebuilding Episodic HNSW index from MongoDB...")

        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        self.next_id = 0

        for doc in self.collection.find():
            eid = doc.get("_id")
            user_text = doc.get("user")

            if user_text is None or eid is None:
                continue

            embedding = self.embedder.encode(user_text)

            self.index.add_items(
                np.array([embedding]),
                np.array([eid])
            )

            self.next_id = max(self.next_id, eid + 1)

        os.makedirs("data", exist_ok=True)
        self.index.save_index(self.index_path)

        print(f"✅ Episodic index rebuilt with {self.next_id} episodes")

    # --------------------------------------------------
    # ADD EPISODE
    # --------------------------------------------------
    def add_episode(self, embedding, user_input, assistant_output):
        eid = self.next_id

        self.index.add_items(
            np.array([embedding]),
            np.array([eid])
        )

        self.collection.insert_one({
            "_id": eid,
            "user": user_input,
            "assistant": assistant_output,
            "timestamp": datetime.utcnow()
        })

        self.next_id += 1
        self.index.save_index(self.index_path)

    # --------------------------------------------------
    # SEARCH (RELEVANCE + RECENCY AWARE)
    # --------------------------------------------------
    def search(
        self,
        embedding,
        k=2,
        similarity_threshold=0.35,
        max_age_days=30
    ):
        """
        Returns high-quality episodic memories.
        Filters by:
        - semantic similarity
        - recency
        - hard cap (k)
        """

        if self.index.get_current_count() == 0:
            return []

        labels, distances = self.index.knn_query(
            np.array([embedding]),
            k=min(k * 3, self.index.get_current_count())
        )

        results = []
        now = datetime.utcnow()

        for idx, dist in zip(labels[0], distances[0]):
            similarity = 1 - dist

            if similarity < similarity_threshold:
                continue

            doc = self.collection.find_one({"_id": int(idx)})
            if not doc:
                continue

            age_days = (now - doc["timestamp"]).days
            if age_days > max_age_days:
                continue

            recency_penalty = min(age_days * 0.02, 0.3)
            final_score = similarity - recency_penalty

            if final_score < similarity_threshold:
                continue

            results.append({
                "user": doc["user"],
                "assistant": doc["assistant"],
                "timestamp": doc["timestamp"].isoformat(),
                "similarity": float(round(similarity, 3)),
                "score": float(round(final_score, 3))
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
