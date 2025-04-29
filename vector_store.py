# vector_store.py

from pathlib import Path
import json
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict

# 初始化Embedding模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

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

    existing_ids = set()
    if collection.count() > 0:
        all_records = collection.get()
        existing_ids = set(all_records["ids"])

    documents = []
    metadatas = []
    ids = []

    for idx, entry in enumerate(memories):
        memory_id = f"memory-{idx}"
        if memory_id in existing_ids:
            continue

        documents.append(entry["description"])
        metadatas.append(entry)
        ids.append(memory_id)

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

    query_embedding = embed_model.get_text_embedding(query_text)

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
