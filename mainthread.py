# mainthread.py

import time
from pathlib import Path
import json
from datetime import datetime
import requests


from image_processing import generate_image_descriptions
from user_note_processing import load_user_notes
from memory_combiner import combine_memories
from vector_store import initialize_vector_store, add_memories_to_vector_store, query_similar_memories
from query_reasoning import generate_answer
from popup_show_images import popup_images
from voice_interface import listen_to_question_with_confirmation, speak_text, record_note_with_confirmation, wait_for_wake_word

from camera_capture import capture_image
from runner_controller import start_runner

import os
from multiprocessing import Process

os.environ["DISPLAY"] = ":0"
# Config
image_folder = Path("memory_images")
model_output_json = Path("memory_text_model.json")
user_json_path = Path("memory_text_user.json")
combined_output_json = Path("memory_combined.json")
chroma_persist_dir = "chroma_db"
collection_name = "memories"

# Vector DB
client = initialize_vector_store(persist_dir=chroma_persist_dir)


def vlm_loop(interval: int = 5):
    while True:
        try:
            print("🌀 Background VLM started.")
            generate_image_descriptions(image_folder, model_output_json)
        except Exception as e:
            print(f"[sync_loop ERROR] {e}")
        time.sleep(interval)

def sync_memories():
    try:
        print("🔄 Syncing memories...")

        user_data = load_user_notes(user_json_path)
        with open(model_output_json, "r") as f:
            model_data = json.load(f)

        combine_memories(user_data, model_data, combined_output_json)
        add_memories_to_vector_store(client, combined_output_json, collection_name=collection_name)

        print("✅ Sync completed.")
    except Exception as e:
        print(f"[ERROR] Manual sync failed: {e}")

def save_user_note(img_path: str, note: str):
    user_json_path.parent.mkdir(parents=True, exist_ok=True)
    if user_json_path.exists():
        with open(user_json_path, "r") as f:
            existing_notes = json.load(f)
    else:
        existing_notes = []

    img_filename = Path(img_path).stem
    parts = img_filename.split("_")
    if len(parts) >= 3:
        timestamp_raw = parts[1] + parts[2]
        try:
            if len(timestamp_raw) == 14:
                dt = datetime.strptime(timestamp_raw, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            elif len(timestamp_raw) == 12:
                dt = datetime.strptime(timestamp_raw, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M")
            elif len(timestamp_raw) == 10:
                dt = datetime.strptime(timestamp_raw, "%Y%m%d%H").strftime("%Y-%m-%d %H")
            else:
                print(f"⚠️ Unexpected timestamp length in filename: {img_path}")
                dt = None
        except Exception as e:
            print(f"❌ Failed to parse timestamp from {img_path}: {e}")
            dt = None
    else:
        print(f"⚠️ Unexpected filename format: {img_path}")
        dt = None

    if dt is None:
        dt = time.strftime("%Y-%m-%d %H:%M")

    new_entry = {
        "timestamp": dt,
        "description": note,
        "image_path": img_path,
        "source": "user"
    }
    existing_notes.append(new_entry)

    with open(user_json_path, "w") as f:
        json.dump(existing_notes, f, indent=2, ensure_ascii=False)

    print(f"✅ User note saved for {img_path} at {dt}")

# vlm_process = Process(target=vlm_loop, args=(10,), daemon=True)
def interactive_loop():
    speak_text("Memory Assistant is ready. Listening for your commands. Please say take photo or hi man.")

    while True:
        # keyword spotting
        label = wait_for_wake_word("menu")
        print(f"🎯 Detected label: {label}")

        if label == "takephoto":
            speak_text("Ready to take a photo.")
            img_path = capture_image()
            user_note = record_note_with_confirmation()
            if not user_note:
                continue
            save_user_note(img_path, user_note)
            speak_text("Photo and note saved successfully.")
            
            speak_text("Do you want me to describe photos? Please say yes or no.")

            while True:
                label_vlm = wait_for_wake_word("yesno")
                if label_vlm == "yes":
                    start_query = time.time()
                    generate_image_descriptions(image_folder, model_output_json)
                    end_query = time.time()
                    print(f"🔍 generate_image_descriptions took {end_query - start_query:.3f} seconds.")
                    speak_text("The photos have been processed.")
                    break
                elif label_vlm == "no":
                    speak_text("Okay, skipping image description.")
                    break
                else:
                    continue
        
            speak_text("Memory Assistant is ready. Listening for your commands.")

        elif label == "himan":
            user_question = listen_to_question_with_confirmation()
            if not user_question:
                continue
            sync_memories()

            start_query = time.time()
            # I think topk =3 or 8 , the speed is the same for LLM prompt
            matched_memories = query_similar_memories(client, user_question, top_k=5, collection_name=collection_name)
            end_query = time.time()
            print(f"🔍 Query similar memories took {end_query - start_query:.3f} seconds.")

            start_answer = time.time()
            answer = generate_answer(query=user_question, memories=matched_memories)
            end_answer = time.time()
            print(f"🧠 Generate answer took {end_answer - start_answer:.3f} seconds.")

            speak_text(answer.summary)

            if answer.image_refs:
                popup_images(answer.image_refs, delay=5)
            else:
                print("⚡ No reference images to display.")

            speak_text("Memory Assistant is ready. Listening for your commands.")

        else:
            # print(f"⚡ Ignoring label: {label}")
            continue


def preload_ollama_models():
    models_to_preload = [
        # {"model": "llava-phi3:3.8b", "prompt": "Describe this image.", "images": []},
        {"model": "llama3.2:3b", "prompt": "Hello!", "images": []}
    ]
    for m in models_to_preload:
        try:
            res = requests.post("http://localhost:11434/api/generate", json={
                "model": m["model"],
                "prompt": m["prompt"],
                "images": m["images"],
                "stream": False,
            })
            print(f"✅ Preloaded model {m['model']}")
        except Exception as e:
            print(f"⚠️ Failed to preload {m['model']}: {e}")

def main():
    # vlm_process.start()
    preload_ollama_models()
    interactive_loop() 

if __name__ == "__main__":
    main()
