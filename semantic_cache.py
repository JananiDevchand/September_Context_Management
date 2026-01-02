import hnswlib
import numpy as np
import os
from datetime import datetime
from pymongo import MongoClient
from embeddings import EmbeddingModel


class SemanticCache:
    def __init__(self, dim=384, max_elements=5000):
        self.index_path = "data/cache_hnsw.index"

        # ---- MongoDB ----
        self.client = MongoClient("mongodb://localhost:27017")
        self.db = self.client["memory_chatbot"]
        self.collection = self.db["semantic_cache"]

        # ---- Embedder (needed for rebuild) ----
        self.embedder = EmbeddingModel()

        # ---- HNSW ----
        self.index = hnswlib.Index(space="cosine", dim=dim)

        if os.path.exists(self.index_path):
            try:
                self.index.load_index(self.index_path)
                self.next_id = self.index.get_current_count()
                print(f"✅ Loaded Semantic Cache ({self.next_id} items)")
            except RuntimeError:
                print("⚠️ Corrupted Cache index detected. Rebuilding...")
                self._rebuild_from_mongo(max_elements)
        else:
            self._rebuild_from_mongo(max_elements)

    # --------------------------------------------------
    # REBUILD CACHE INDEX FROM MONGODB
    # --------------------------------------------------
    def _rebuild_from_mongo(self, max_elements):
        print("🔁 Rebuilding Semantic Cache from MongoDB...")

        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        self.next_id = 0

        for doc in self.collection.find():
            cid = doc.get("embedding_id")
            query = doc.get("query")

            if cid is None or not query:
                continue

            embedding = self.embedder.encode(query)

            self.index.add_items(
                np.array([embedding]),
                np.array([cid])
            )

            self.next_id = max(self.next_id, cid + 1)

        os.makedirs("data", exist_ok=True)
        self.index.save_index(self.index_path)

        print(f"✅ Semantic Cache rebuilt with {self.next_id} items")

    # --------------------------------------------------
    # LOOKUP
    # --------------------------------------------------
    def lookup(self, embedding, user_id, similarity_threshold=0.90):
        if self.index.get_current_count() == 0:
            return None

        labels, distances = self.index.knn_query(
            np.array([embedding]),
            k=min(3, self.index.get_current_count())
        )

        for idx, dist in zip(labels[0], distances[0]):
            similarity = 1 - dist
            if similarity < similarity_threshold:
                continue

            doc = self.collection.find_one({
                "embedding_id": int(idx),
                "user_id": user_id
            })

            if doc:
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {
                        "$inc": {"hit_count": 1},
                        "$set": {"last_used": datetime.utcnow()}
                    }
                )
                return doc["response"]

        return None

    # --------------------------------------------------
    # ADD TO CACHE
    # --------------------------------------------------
    def add(self, embedding, user_id, query, response):
        cid = self.next_id

        self.index.add_items(
            np.array([embedding]),
            np.array([cid])
        )

        self.collection.insert_one({
            "embedding_id": cid,
            "user_id": user_id,
            "query": query,
            "response": response,
            "hit_count": 1,
            "last_used": datetime.utcnow()
        })

        self.next_id += 1
        self.index.save_index(self.index_path)
