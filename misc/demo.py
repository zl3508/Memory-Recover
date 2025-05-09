# ✅ Imports
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import json
import matplotlib.pyplot as plt
from PIL import Image
from pydantic import BaseModel, Field
import instructor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
import numpy as np

# ✅ Step 1: Load Combined Memory Log
base_dir = Path(__file__).parent
with open(base_dir / "memory_combined.json") as f:
    memory_data = json.load(f)

# ✅ Step 2: Embed All Descriptions
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
documents = [TextNode(text=entry["description"], metadata=entry) for entry in memory_data]
embeddings = embed_model.get_text_embedding_batch([d.text for d in documents])

# ✅ Step 3: Embed the Query and Compute Similarities
query = "Where is my phone?"
query_embed = embed_model.get_text_embedding(query)
scores = [np.dot(e, query_embed) for e in embeddings]

# ✅ Step 4: Select Top 3 Model + Top 3 User Entries Separately (by semantic similarity)
# 把 embedding + metadata 打包
scored_docs = list(zip(scores, documents))

# 拆成两组
model_scored = [doc for score, doc in scored_docs if doc.metadata["source"] == "model"]
user_scored = [doc for score, doc in scored_docs if doc.metadata["source"] == "user"]

# 对各组按语义相似度排序
model_top3 = sorted(
    [(np.dot(embed, query_embed), doc) for embed, doc in zip(embeddings, documents) if doc.metadata["source"] == "model"],
    key=lambda x: x[0], reverse=True
)[:3]

user_top3 = sorted(
    [(np.dot(embed, query_embed), doc) for embed, doc in zip(embeddings, documents) if doc.metadata["source"] == "user"],
    key=lambda x: x[0], reverse=True
)[:3]

# 合并并按时间排序
final_context = [item[1].metadata for item in model_top3 + user_top3]
final_context.sort(key=lambda x: x["timestamp"])


# ✅ Step 5: Format Context for Prompt
context_str = "\n".join([
    f"- {item['timestamp']} ({item['source']}): {item['description']}" for item in final_context
])

# ✅ Step 6: Define Function Output Schema
class MemoryReasoning(BaseModel):
    summary: str = Field(..., description="Summary of reasoning about where the phone might be")
    image_refs: list[str] = Field(..., description="Up to 3 image file paths that support the reasoning")

# ✅ Step 7: Construct Prompt
prompt = f"""
You are a helpful memory assistant.

Here is the memory log:
{context_str}

Each memory log entry is linked to an image file path, embedded inside the description.
Your task:

1. Find at most 3 memory entries that are most relevant to the user's query.
2. Return:
    - A short reasoning summary (string).
    - A list of up to 3 image file paths corresponding to the most relevant entries.

Important rules:
- You MUST return BOTH the summary and the image_refs fields.
- If no relevant images exist, return an empty list [] for image_refs (do not omit it).
- Only use the image paths already mentioned in the memory log. DO NOT create or invent any new image paths.
- Keep the image paths exactly as provided (e.g., memory_images/img_20250420_1637.jpg).

Your output must strictly match the required JSON format.
"""

# ✅ Step 8: Run Function Calling Model
client = instructor.patch(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": prompt}],
    response_model=MemoryReasoning,
    temperature=0.2,
)

# ✅ Step 9: Display Output
print("\n📌 Reasoning Summary:")
print(response.summary)

print("\n🖼️ Reference Images:")
for img_path in response.image_refs:
    full_img_path = base_dir / img_path  # ✅ 自动拼接成绝对路径
    if full_img_path.exists():
        img = Image.open(full_img_path)
        plt.imshow(img)
        plt.title(full_img_path.name)
        plt.axis("off")
        plt.show()
    else:
        print(f"⚠️ Image not found: {full_img_path}")
