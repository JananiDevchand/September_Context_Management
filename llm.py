import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-safeguard-20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

import json

def extract_semantic_memory(text: str):
    prompt = f"""
From the text below, extract ONE reusable memory.

Classify it as:
- knowledge
- persona
- process

Return a JSON object with keys "type" and "content". No other text.

TEXT:
{text}
"""
    raw = call_llm(prompt).strip()

    try:
        data = json.loads(raw)

        if (
            isinstance(data, dict)
            and "type" in data
            and "content" in data
            and data["type"] in {"knowledge", "persona", "process"}
        ):
            return data

    except json.JSONDecodeError:
        pass

    return None
