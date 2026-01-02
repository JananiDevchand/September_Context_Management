import hnswlib
import numpy as np
import os
import re
from datetime import datetime
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
from embeddings import EmbeddingModel


# -------------------------------
# BM25 Utilities
# -------------------------------
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self):
        self.docs = []
        self.raw_docs = []
        self.bm25 = None

    def add(self, text: str):
        tokens = tokenize(text)
        if not tokens:
            return
        self.docs.append(tokens)
        self.raw_docs.append(text)
        self.bm25 = BM25Okapi(self.docs)

    def search(self, query: str, top_k=5):
        if not self.bm25:
            return []

        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {"content": self.raw_docs[idx], "bm25_score": float(round(score, 3))}
            for idx, score in ranked[:top_k]
            if score > 0
        ]


# ===============================
# Semantic Memory (Hybrid + Rebuild)
# ===============================
class SemanticMemory:
    def __init__(self, dim=384, max_elements=10000):
        self.index_path = "data/semantic_hnsw.index"

        # ---- MongoDB ----
        self.client = MongoClient("mongodb://localhost:27017")
        self.db = self.client["memory_chatbot"]
        self.collection = self.db["semantic_memory"]

        # ---- Embedder (needed for rebuild) ----
        self.embedder = EmbeddingModel()

        # ---- HNSW ----
        self.index = hnswlib.Index(space="cosine", dim=dim)

        # ---- BM25 per memory type ----
        self.bm25_indices = {
            "knowledge": BM25Index(),
            "persona": BM25Index(),
            "process": BM25Index()
        }

        # ---- Load or rebuild index ----
        if os.path.exists(self.index_path):
            try:
                self.index.load_index(self.index_path)
                self.next_id = self.index.get_current_count()
                print(f"✅ Loaded Semantic HNSW ({self.next_id} items)")
            except RuntimeError:
                print("⚠️ Corrupted Semantic index detected. Rebuilding...")
                self._rebuild_from_mongo(max_elements)
        else:
            self._rebuild_from_mongo(max_elements)

    # --------------------------------------------------
    # REBUILD VECTOR + BM25 FROM MONGODB
    # --------------------------------------------------
    def _rebuild_from_mongo(self, max_elements):
        print("🔁 Rebuilding Semantic Memory from MongoDB...")

        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        self.next_id = 0

        # Reset BM25
        self.bm25_indices = {
            "knowledge": BM25Index(),
            "persona": BM25Index(),
            "process": BM25Index()
        }

        for doc in self.collection.find():
            content = doc.get("content")
            mem_type = doc.get("type")
            embedding_id = doc.get("embedding_id")

            if not content or mem_type not in self.bm25_indices:
                continue

            embedding = self.embedder.encode(content)

            self.index.add_items(
                np.array([embedding]),
                np.array([embedding_id])
            )

            self.bm25_indices[mem_type].add(content)
            self.next_id = max(self.next_id, embedding_id + 1)

        os.makedirs("data", exist_ok=True)
        self.index.save_index(self.index_path)

        print(f"✅ Semantic Memory rebuilt with {self.next_id} items")

    # --------------------------------------------------
    # ADD MEMORY (knowledge | persona | process)
    # --------------------------------------------------
    def add_memory(
        self,
        embedding,
        content,
        mem_type="knowledge",
        user_id=None,
        similarity_threshold=0.65
    ):
        if self.index.get_current_count() > 0:
            labels, distances = self.index.knn_query(
                np.array([embedding]),
                k=min(5, self.index.get_current_count())
            )

            for idx, dist in zip(labels[0], distances[0]):
                similarity = 1 - dist
                if similarity >= similarity_threshold:
                    doc = self.collection.find_one(
                        {"embedding_id": int(idx)}
                    )
                    if not doc:
                        continue

                    self.collection.update_one(
                        {"embedding_id": int(idx)},
                        {
                            "$inc": {"support_count": 1},
                            "$set": {
                                "last_seen": datetime.utcnow(),
                                "confidence": min(
                                    1.0,
                                    float(doc.get("confidence", 0.6)) + 0.05
                                )
                            }
                        }
                    )
                    return

        mid = self.next_id

        self.index.add_items(
            np.array([embedding]),
            np.array([mid])
        )

        self.collection.insert_one({
            "embedding_id": mid,
            "type": mem_type,
            "content": content,
            "user_id": user_id,
            "support_count": 1,
            "confidence": 0.6,
            "last_seen": datetime.utcnow()
        })

        self.bm25_indices[mem_type].add(content)

        self.next_id += 1
        self.index.save_index(self.index_path)

    # --------------------------------------------------
    # HYBRID SEARCH (BM25 + VECTOR)
    # --------------------------------------------------
    def search(
        self,
        embedding,
        query_text,
        k=2,
        mem_type=None,
        user_id=None,
        similarity_threshold=0.35,
        max_age_days=60,
        alpha=0.7
    ):
        if self.index.get_current_count() == 0:
            return []

        labels, distances = self.index.knn_query(
            np.array([embedding]),
            k=min(k * 4, self.index.get_current_count())
        )

        now = datetime.utcnow()
        vector_hits = {}

        for idx, dist in zip(labels[0], distances[0]):
            similarity = 1 - dist
            if similarity < similarity_threshold:
                continue

            query = {"embedding_id": int(idx)}
            if mem_type:
                query["type"] = mem_type
            if user_id:
                query["user_id"] = user_id

            doc = self.collection.find_one(query)
            if not doc:
                continue

            age_days = (now - doc["last_seen"]).days
            if age_days > max_age_days:
                continue

            recency_penalty = min(age_days * 0.015, 0.3)
            support_boost = min(doc.get("support_count", 1) * 0.05, 0.25)

            vector_score = similarity - recency_penalty + support_boost

            vector_hits[doc["content"]] = {
                "doc": doc,
                "vector_score": vector_score
            }

        bm25_hits = []
        if mem_type and mem_type in self.bm25_indices:
            bm25_hits = self.bm25_indices[mem_type].search(
                query_text,
                top_k=k * 3
            )

        bm25_map = {
            hit["content"]: hit["bm25_score"]
            for hit in bm25_hits
        }

        results = []
        for content, data in vector_hits.items():
            bm25_score = bm25_map.get(content, 0.0)

            hybrid_score = (
                alpha * data["vector_score"]
                + (1 - alpha) * bm25_score
            )

            if hybrid_score < similarity_threshold:
                continue

            doc = data["doc"]
            results.append({
                "type": doc["type"],
                "content": doc["content"],
                "support_count": int(doc.get("support_count", 1)),
                "confidence": float(doc.get("confidence", 0.6)),
                "score": float(round(hybrid_score, 3)),
                "vector_score": float(round(data["vector_score"], 3)),
                "bm25_score": float(round(bm25_score, 3)),
                "last_seen": doc["last_seen"].isoformat()
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
