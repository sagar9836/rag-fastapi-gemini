from collections import defaultdict
from typing import List

# Simple in-memory storage (session_id → messages)
chat_memory = defaultdict(list)

MAX_TURNS = 5  # keep last 5 Q&A pairs


def add_to_memory(session_id: str, role: str, content: str):
    chat_memory[session_id].append({"role": role, "content": content})

    # Trim memory
    if len(chat_memory[session_id]) > MAX_TURNS * 2:
        chat_memory[session_id] = chat_memory[session_id][-MAX_TURNS * 2:]


def get_memory(session_id: str) -> List[dict]:
    return chat_memory.get(session_id, [])
