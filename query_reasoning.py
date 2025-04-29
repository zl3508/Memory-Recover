# query_reasoning.py

from typing import List, Dict
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field

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

    prompt = f"""
You are a helpful memory assistant.

Here is the memory log:
{context_str}

User's question: "{query}"

Based on these records, please infer a short summary answering the user's question.
Important Instructions:
- Only select image file paths that are explicitly provided inside [Image: ...] annotations.
- Do NOT invent, guess, or create any new image file paths.
- Choose up to 3 images most relevant to your reasoning.

Respond strictly in JSON format following this schema:
- summary: A brief explanation
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
