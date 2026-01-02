def build_prompt(user_input, episodic, semantic_blocks, short_term):
    # --------------------------------------------------
    # Short-term memory (working memory)
    # --------------------------------------------------
    short_term_block = [
        {
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content
        }
        for msg in short_term
    ]

    # --------------------------------------------------
    # Episodic memory
    # --------------------------------------------------
    episodic_block = [
        {
            "user": ep["user"],
            "assistant": ep["assistant"],
            "timestamp": ep["timestamp"]
        }
        for ep in episodic
    ]

    # --------------------------------------------------
    # 🔑 FIX 3: TYPE-AWARE SEMANTIC MEMORY (DO NOT MIX)
    # --------------------------------------------------
    persona = [m["content"] for m in semantic_blocks.get("persona", [])]
    knowledge = [m["content"] for m in semantic_blocks.get("knowledge", [])]
    process = [m["content"] for m in semantic_blocks.get("process", [])]

    # --------------------------------------------------
    # Prompt assembly
    # --------------------------------------------------
    prompt = f"""
You are an assistant with structured memory.
Only use the memories below if they are directly relevant to the current question.
Ignore unrelated memories completely.

USER PERSONA:
{chr(10).join('- ' + p for p in persona)}

KNOWN FACTS:
{chr(10).join('- ' + k for k in knowledge)}

KNOWN PROCESSES:
{chr(10).join('- ' + p for p in process)}

EPISODIC CONTEXT:
{chr(10).join(
    f'- User: {e["user"]} | Assistant: {e["assistant"]}'
    for e in episodic_block
)}

RECENT CONVERSATION:
{chr(10).join(f'{m["role"]}: {m["content"]}' for m in short_term_block)}

CURRENT QUESTION:
{user_input}
"""

    return prompt.strip(), {
        "persona": persona,
        "knowledge": knowledge,
        "process": process,
        "episodic": episodic_block,
        "short_term": short_term_block,
        "final_prompt": prompt.strip()
    }
