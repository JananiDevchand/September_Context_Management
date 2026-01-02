from flask import Flask, render_template, request, jsonify
from embeddings import EmbeddingModel
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from semantic_cache import SemanticCache
from prompt import build_prompt
from llm import call_llm, extract_semantic_memory
from short_term_memory import ShortTermMemory
import time

app = Flask(__name__)

# --------------------------------------------------
# Global instances (prototype-level)
# --------------------------------------------------
embedder = EmbeddingModel()
episodic = EpisodicMemory()
semantic = SemanticMemory()
cache = SemanticCache()
short_term = ShortTermMemory(k=3)


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()

    data = request.get_json()
    user_input = data.get('message', '').strip()
    memory_limit = int(data.get('memory_limit', 3))
    user_id = data.get('user_id', 'test_user_1')

    if not user_input:
        return jsonify({'error': 'Empty message'}), 400

    # --------------------------------------------------
    # Embedding
    # --------------------------------------------------
    query_embedding = embedder.encode(user_input)

    # --------------------------------------------------
    # Semantic Cache Lookup (FAST PATH)
    # --------------------------------------------------
    cached_response = cache.lookup(
        query_embedding,
        user_id=user_id
    )

    if cached_response:
        processing_time = time.time() - start_time
        short_term.add(user_input, cached_response)

        return jsonify({
            "response": cached_response,
            "cache_hit": True,
            "episodic_hits": [],
            "semantic_hits": [],
            "processing_time": round(processing_time * 1000, 2),
            "memory_count": 0,
            "context": {
                "note": "Response served from semantic cache"
            },
            "timestamp": time.time()
        })

    # --------------------------------------------------
    # Episodic Memory Retrieval
    # --------------------------------------------------
    episodic_hits = episodic.search(
        query_embedding,
        k=min(memory_limit, 5)
    )

    # --------------------------------------------------
    # 🔑 HYBRID TYPE-AWARE SEMANTIC RETRIEVAL
    # --------------------------------------------------
    persona_hits = semantic.search(
        embedding=query_embedding,
        query_text=user_input,
        k=2,
        mem_type="persona",
        user_id=user_id,
        similarity_threshold=0.10
    )

    knowledge_hits = semantic.search(
        embedding=query_embedding,
        query_text=user_input,
        k=3,
        mem_type="knowledge",
        user_id=user_id,
        similarity_threshold=0.30
    )

    process_hits = semantic.search(
        embedding=query_embedding,
        query_text=user_input,
        k=2,
        mem_type="process",
        user_id=user_id,
        similarity_threshold=0.30
    )

    # --------------------------------------------------
    # Short-term Memory
    # --------------------------------------------------
    short_term_context = short_term.load()

    # --------------------------------------------------
    # Build Prompt + Context (TYPE-AWARE)
    # --------------------------------------------------
    prompt, context_debug = build_prompt(
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
    # LLM Call
    # --------------------------------------------------
    response = call_llm(prompt)
    processing_time = time.time() - start_time

    # --------------------------------------------------
    # Store in Semantic Cache
    # --------------------------------------------------
    cache.add(
        query_embedding,
        user_id=user_id,
        query=user_input,
        response=response
    )

    # --------------------------------------------------
    # Update Memories
    # --------------------------------------------------
    short_term.add(user_input, response)
    episodic.add_episode(query_embedding, user_input, response)

    # --------------------------------------------------
    # Semantic Memory Extraction
    # --------------------------------------------------
    memory = extract_semantic_memory(
        f"User: {user_input}\nAssistant: {response}"
    )

    if memory:
        semantic_embedding = embedder.encode(memory["content"])
        semantic.add_memory(
            semantic_embedding,
            memory["content"],
            mem_type=memory["type"],
            user_id=user_id
        )

    # --------------------------------------------------
    # Response
    # --------------------------------------------------
    return jsonify({
        "response": response,
        "cache_hit": False,
        "episodic_hits": episodic_hits,
        "semantic_hits": {
            "persona": persona_hits,
            "knowledge": knowledge_hits,
            "process": process_hits
        },
        "processing_time": round(processing_time * 1000, 2),
        "memory_count": (
            len(episodic_hits)
            + len(persona_hits)
            + len(knowledge_hits)
            + len(process_hits)
        ),
        "context": context_debug,
        "timestamp": time.time()
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "episodic_memory_count": episodic.index.get_current_count(),
        "semantic_memory_count": semantic.index.get_current_count()
    })


if __name__ == '__main__':
    print("🚀 Starting NeuroMind AI Flask app (Hybrid Memory Enabled)...")
    app.run(debug=True, host='0.0.0.0', port=5000)
