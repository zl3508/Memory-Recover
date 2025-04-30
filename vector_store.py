# vector_store.py

from pathlib import Path
import json
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict
import hashlib

def make_id(entry):
    raw = f"{entry['timestamp']} - {entry['description']}"
    return "memory-" + hashlib.md5(raw.encode()).hexdigest()
# 初始化Embedding模型
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# RAG
embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-base-v2") # or small-v2?

def initialize_vector_store(persist_dir: str) -> chromadb.Client:
    """
    Initialize a ChromaDB client with local persistence (new architecture).
    """
    client = chromadb.PersistentClient(path=persist_dir)
    return client

def add_memories_to_vector_store(client: chromadb.Client, memory_json_path: Path, collection_name: str = "memories") -> None:
    with open(memory_json_path, "r") as f:
        memories = json.load(f)

    collection = client.get_or_create_collection(name=collection_name)

    # if collection.count() > 0:
    #     all_records = collection.get(include=["ids"])
    #     existing_ids = set(all_records["ids"])
    # 一次性准备 [(entry, id, timed_text)] 列表
    prepared = [
        (entry,
        mem_id := make_id(entry),
        f"passage: {entry['timestamp']} - {entry['description']}")
        for entry in memories
    ]

    candidate_ids = [mem_id for _, mem_id, _ in prepared]
    existing_records = collection.get(ids=candidate_ids)
    existing_ids = set(existing_records["ids"])

    documents, metadatas, ids = [], [], []
    for entry, mem_id, timed_text in prepared:
        if mem_id in existing_ids:
            continue
        documents.append(timed_text)
        metadatas.append(entry)
        ids.append(mem_id)

    if documents:
        embeddings = embed_model.get_text_embedding_batch(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(documents)} new memories to ChromaDB collection '{collection_name}'.")
    else:
        print(f"⚡ No new memories to add. Vector store is already up to date.")


def query_similar_memories(client: chromadb.Client, query_text: str, top_k: int = 5, collection_name: str = "memories") -> List[Dict]:
    """
    Query ChromaDB for top-k most similar memories to a given query text.
    """
    collection = client.get_or_create_collection(name=collection_name)

    query_embedding = embed_model.get_text_embedding("query: " + query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    matched_memories = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        memory = metadata
        memory["similarity"] = 1 - distance  # 1 - distance (越接近1越相似)
        matched_memories.append(memory)

    return matched_memories
