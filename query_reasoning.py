# query_reasoning.py

from typing import List, Dict
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from datetime import datetime
from pytz import timezone

# ✅ 定义推理的输出格式
class MemoryReasoning(BaseModel):
    summary: str = Field(..., description="Summary of reasoning based on memory entries")
    image_refs: List[str] = Field(..., description="List of up to 3 real image file paths supporting the reasoning")

def generate_answer(query: str, memories: List[Dict], model_name: str = "llama3.2:3b") -> MemoryReasoning:
    """
    Generate a reasoning summary based on retrieved memories and a user query.

    Args:
        query (str): User's question.
        memories (List[Dict]): Top-k related memories retrieved from ChromaDB.
        model_name (str): Local LLM model name (default "llama3.2:3b").

    Returns:
        MemoryReasoning: A structured reasoning result.
    """

    # ✅ 格式化 memories，同时附上真实图片路径
    context_str = "\n".join([
        f"- {m['timestamp']} ({m['source']}): {m['description']} [Image: {m['image_path']}]" for m in memories
    ])

    timestamp = datetime.now(timezone("America/New_York"))
    prompt = f"""
    You are a memory recovery assistant.

    Given the following memory records, your goal is to **help the user vividly recall** their forgotten moments.

    Memory records:
    {context_str}

    The user is trying to recall something. Here's their question (asked at {timestamp}): "{query}"

    Please:
    - Analyze the most relevant memories based on the question.
    - Summarize the related memories into a short, vivid answer that helps the user **emotionally reconnect** with the past moment.
    - Recommend up to 3 specific images that best support the answer (only from [Image: ...] entries).

    Important Instructions:
    - Only describe what is actually recorded. Be natural, concise, and emotionally vivid.

    Respond strictly in JSON format with this schema:
    - summary: A 2-3 sentence summary that helps the user recall the moment.
    - image_refs: A list of up to 3 image paths
    """

    # ✅ 调用本地 Ollama + instructor function-calling
    client = instructor.patch(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        mode=instructor.Mode.JSON,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_model=MemoryReasoning,
        temperature=0.2,
    )

    return response
