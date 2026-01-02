import requests
import time

BASE_URL = "http://localhost:5000/chat"

PAYLOAD_BASE = {
    "user_id": "test_user_1",
    "memory_limit": 3
}


# 50 NEUTRAL QUESTIONS FOR CONTEXT-AWARE CHATBOT TESTING
# No personal references, no "you/me"
# Ask sequentially using the SAME user_id

#civics and society
SEED_QUERIES1  = [
    "What is democracy?",
    "What is the role of a constitution?",
    "What does the term economy mean?",
    "Why do governments collect taxes?",
    "What is the separation of powers?",

    "What are fundamental rights?",
    "What is the purpose of elections?",
    "How does the judicial system function?",
    "What is public policy?",
    "Why is the rule of law important?"
]

#environment
SEED_QUERIES2 = [
    "What causes climate change?",
    "Why are forests important to ecosystems?",
    "What is renewable energy?",
    "How does the water cycle work?",
    "Why is biodiversity important?",

    "What is the greenhouse effect?",
    "How do fossil fuels impact the environment?",
    "What causes ocean pollution?",
    "How do glaciers affect sea levels?",
    "Why is conservation necessary?"
]
#HISTORY
SEED_QUERIES3= [
    "Who was the first President of the United States?",
    "In which year did World War II end?",
    "What was the ancient Silk Road used for?",
    "Which civilization built the pyramids of Giza?",
    "What was the Industrial Revolution?",

    "What caused the French Revolution?",
    "Who was Mahatma Gandhi?",
    "What was the Cold War?",
    "Why did the Roman Empire decline?",
    "What was the significance of the Renaissance?"
]

#USER
SEED_QUERIES4 = [
    "Preferred explanation style is detailed and step-by-step.",
    "Structured and well-formatted responses are preferred.",
    "Technical depth is expected when discussing complex topics.",
    "Clear reasoning and justification should be included in answers.",
    "Examples are preferred alongside explanations.",

    "Formal and professional language is preferred.",
    "Accuracy is prioritized over response speed.",
    "High-level overview should be provided before deep details.",
    "Proactive suggestions are welcome when relevant.",
    "Concise summaries at the end of explanations are useful."
]
#PROCESS
SEED_QUERIES5 = [
    "How does a machine learning pipeline work?",
    "What are the steps involved in training a model?",
    "How is data prepared before analysis?",
    "How is overfitting handled?",

    "How does information retrieval work?",
    "How does vector search operate?",
    "How does keyword-based search work?",
    "How does hybrid search combine multiple methods?",
    "How does context get built over multiple interactions?"
]


# 50 NEUTRAL QUESTIONS FOR CONTEXT-AWARE CHATBOT TESTING
# No personal references, no "you/me"
# Ask sequentially using the SAME user_id

SEED_QUERIES6 = [
    "My name is Janani. I prefer concise, technical explanations and I am building an AI memory system.",

    "Vector databases store embeddings and perform similarity search using cosine or dot-product distance.",

    "Semantic memory stores long-term facts, while episodic memory stores past conversations with context.",

    "My chatbot workflow is: embed the query, retrieve episodic and semantic memory, build a prompt, call the LLM, and store updated memories.",

    "I am testing whether my context management system retrieves the correct memories.",

    "Short-term memory should only keep the last few interactions.",

    "This message should push older messages out of short-term memory.",

    "Summarize my chatbot workflow.",

    "Summarize my chatbot workflow.",  # cache hit

    "Do you remember my name and how I prefer explanations?"
]
SEED_QUERIES7= [
    "What is machine learning?",
    "What are the main types of machine learning?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is reinforcement learning?",

    "What is overfitting?",
    "What is underfitting?",
    "What is bias and variance tradeoff?",
    "What is cross validation?",
    "How is model performance evaluated?"
]

SEED_QUERIES8= [

    # -------- FOUNDATIONAL KNOWLEDGE --------
    "What is context management in large language models?",
    "Why is context important for conversational AI systems?",
    "What are the limitations of fixed context windows in LLMs?",
    "How do modern chatbots handle long conversations?",

    # -------- MEMORY TYPES --------
    "What is semantic memory in AI systems?",
    "What is episodic memory?",
    "How does short-term memory differ from long-term memory?",
    "Why is short-term memory usually limited in size?",

    # -------- ARCHITECTURE & DESIGN --------
    "What is a memory-centric chatbot architecture?",
    "How does memory retrieval fit into the request pipeline?",
    "Why should memory retrieval happen before prompt construction?",
    "What role does prompt construction play in context management?",

    # -------- SEMANTIC MEMORY --------
    "What kind of information should be stored in semantic memory?",
    "Why is structured semantic memory preferred over raw text?",
    "How can semantic memory be reused across conversations?",
    "What are the risks of storing incorrect semantic memory?",

    # -------- EPISODIC MEMORY --------
    "What information is typically stored in episodic memory?",
    "When is episodic memory useful during a conversation?",
    "How can episodic memory introduce noise into responses?",
    "What strategies help limit irrelevant episodic recall?",

    # -------- RETRIEVAL MECHANISMS --------
    "How does vector similarity search work?",
    "Why is cosine similarity commonly used for embeddings?",
    "What does top-k retrieval mean?",
    "Why can top-k retrieval alone be insufficient?",

    # -------- CONTEXT QUALITY --------
    "What happens when too much context is passed to an LLM?",
    "How does irrelevant context affect response quality?",
    "Why is ordering of context important in prompts?",
    "How can context be summarized without losing meaning?"]

   
    # -------- CACHING --------
SEED_QUERIES=  [
    "What is a semantic cache in AI systems?",
    "How does caching improve system performance?",
    "When can caching be harmful in LLM applications?",
    "What kinds of queries are safe to cache?",

]

def send_query(text):
    payload = PAYLOAD_BASE | {"message": text}
    response = requests.post(BASE_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("🚀 Seeding memory system...\n")

    for i, query in enumerate(SEED_QUERIES, 1):
        print(f"➡️  [{i}] {query}")
        result = send_query(query)

        print(f"   cache_hit: {result.get('cache_hit')}")
        print(f"   memory_count: {result.get('memory_count')}")
        print(f"   processing_time(ms): {result.get('processing_time')}\n")

        time.sleep(1)  # avoid rate limits

    print("✅ Memory seeding complete.")
