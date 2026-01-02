from embeddings import EmbeddingModel
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from semantic_cache import SemanticCache
from prompt import build_prompt
from llm import call_llm, extract_semantic_memory
from short_term_memory import ShortTermMemory
import time


def main():
    print("\n🧠 Episodic + Semantic Memory Chatbot (CLI)\n")

    # --------------------------------------------------
    # Global Components (same as app.py)
    # --------------------------------------------------
    embedder = EmbeddingModel()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    cache = SemanticCache()
    short_term = ShortTermMemory(k=3)

    user_id = "default_user"

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        start_time = time.time()

        # --------------------------------------------------
        # 1️⃣ Embed Query
        # --------------------------------------------------
        query_embedding = embedder.encode(user_input)

        # --------------------------------------------------
        # 2️⃣ Semantic Cache Lookup
        # --------------------------------------------------
        cached_response = cache.lookup(
            query_embedding,
            user_id=user_id
        )

        if cached_response:
            print("\n⚡ Cache hit!")
            print("\nAssistant:", cached_response, "\n")
            short_term.add(user_input, cached_response)
            continue

        # --------------------------------------------------
        # 3️⃣ Episodic Memory Retrieval
        # --------------------------------------------------
        episodic_hits = episodic.search(query_embedding, k=3)

        # --------------------------------------------------
        # 4️⃣ Type-Aware Semantic Retrieval
        # --------------------------------------------------
        persona_hits = semantic.search(
            query_embedding,
            k=2,
            mem_type="persona",
            user_id=user_id,
            similarity_threshold=0.30
        )

        knowledge_hits = semantic.search(
            query_embedding,
            k=4,
            mem_type="knowledge",
            user_id=user_id,
            similarity_threshold=0.30
        )

        process_hits = semantic.search(
            query_embedding,
            k=2,
            mem_type="process",
            user_id=user_id,
            similarity_threshold=0.30
        )

        # --------------------------------------------------
        # 5️⃣ Short-Term Memory
        # --------------------------------------------------
        short_term_context = short_term.load()

        # --------------------------------------------------
        # 6️⃣ Build Prompt (same contract as Flask)
        # --------------------------------------------------
        prompt, _ = build_prompt(
            user_input,
            episodic_hits,
            {
                "persona": persona_hits,
                "knowledge": knowledge_hits,
                "process": process_hits
            },
            short_term_context
        )

        # --------------------------------------------------
        # 7️⃣ LLM Call
        # --------------------------------------------------
        response = call_llm(prompt)
        latency = round((time.time() - start_time) * 1000, 2)

        print("\nAssistant:", response)
        print(f"⏱️ {latency} ms\n")

        # --------------------------------------------------
        # 8️⃣ Store in Semantic Cache
        # --------------------------------------------------
        cache.add(
            query_embedding,
            user_id=user_id,
            query=user_input,
            response=response
        )

        # --------------------------------------------------
        # 9️⃣ Update Memories
        # --------------------------------------------------
        short_term.add(user_input, response)
        episodic.add_episode(query_embedding, user_input, response)

        # --------------------------------------------------
        # 🔟 Extract & Store Semantic Memory
        # --------------------------------------------------
        extracted = extract_semantic_memory(
            f"User: {user_input}\nAssistant: {response}"
        )

        if extracted:
            semantic_embedding = embedder.encode(extracted["content"])
            semantic.add_memory(
                semantic_embedding,
                extracted["content"],
                mem_type=extracted["type"],
                user_id=user_id
            )
            print("🧠 Stored semantic memory:", extracted["type"])
        else:
            print("⚠️ No semantic memory extracted")


if __name__ == "__main__":
    main()
