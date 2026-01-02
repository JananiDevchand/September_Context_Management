import re
from rank_bm25 import BM25Okapi


# -------------------------------
# Tokenizer (simple + robust)
# -------------------------------
def tokenize(text: str):
    """
    Tokenize text for BM25.
    - Lowercase
    - Alphanumeric tokens
    - Suitable for code, facts, names
    """
    return re.findall(r"\w+", text.lower())


# -------------------------------
# BM25 Index Wrapper
# -------------------------------
class BM25Index:
    """
    Lightweight BM25 wrapper for semantic memory.

    This index:
    - Stores tokenized documents
    - Supports incremental additions
    - Rebuilds BM25 automatically
    """

    def __init__(self):
        self.documents = []    # tokenized documents
        self.raw_docs = []     # original text
        self.bm25 = None

    # ---------------------------
    # Add document
    # ---------------------------
    def add(self, text: str):
        """
        Add a document to the BM25 index.
        """
        if not text or not isinstance(text, str):
            return

        tokens = tokenize(text)
        if not tokens:
            return

        self.documents.append(tokens)
        self.raw_docs.append(text)

        # Rebuild BM25 (cheap for this scale)
        self.bm25 = BM25Okapi(self.documents)

    # ---------------------------
    # Search
    # ---------------------------
    def search(self, query: str, top_k: int = 5):
        """
        Perform BM25 lexical search.

        Returns:
        [
            {
                "content": "...",
                "bm25_score": float
            }
        ]
        """
        if not self.bm25 or not query:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                continue

            results.append({
                "content": self.raw_docs[idx],
                "bm25_score": float(round(score, 3))
            })

        return results

    # ---------------------------
    # Utility
    # ---------------------------
    def size(self):
        """
        Number of documents indexed.
        """
        return len(self.raw_docs)
