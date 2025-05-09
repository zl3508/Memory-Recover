# image_processing.py

from pathlib import Path
import base64
import json
from datetime import datetime
import ollama
import gc
import time
from PIL import Image
import io

def generate_image_descriptions(image_folder: Path, output_json: Path) -> None:
    """
    Process images using vision-language model and save descriptions safely on Raspberry Pi.
    """
    image_folder = Path(image_folder)
    output_json = Path(output_json)
    model_results = []

    if output_json.exists():
        with open(output_json, "r") as f:
            existing_data = json.load(f)
        processed_images = set(entry["image_path"] for entry in existing_data)
        print(f"✅ Loaded {len(processed_images)} already processed images.")
    else:
        existing_data = []
        processed_images = set()
        print("✅ No existing processed images found. Starting fresh.")

    for img_path in image_folder.glob("*.jpg"):
        img_path_str = str(img_path)
        if img_path_str in processed_images:
            # print(f"⚡ Skipping already processed: {img_path.name}")
            continue

        print(f"🚀 Processing new image: {img_path.name}")
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            # extract time
            parts = img_path.stem.split("_")
            if len(parts) >= 3:
                timestamp_raw = parts[1] + parts[2]
                if len(timestamp_raw) == 14:
                    dt = datetime.strptime(timestamp_raw, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                elif len(timestamp_raw) == 12:
                    dt = datetime.strptime(timestamp_raw, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M")
                elif len(timestamp_raw) == 10:
                    dt = datetime.strptime(timestamp_raw, "%Y%m%d%H").strftime("%Y-%m-%d %H")
                else:
                    print(f"⚠️ Unexpected timestamp: {img_path.name}, skipping.")
                    continue
            else:
                print(f"⚠️ Unexpected filename format: {img_path.name}, skipping.")
                continue

            # send to model
            prompt = f"""
You are a memory assistant.

Describe the attached photo to help someone recall the moment it was taken.

Photo timestamp: {dt}

Rules:
- Max 3 sentences, 80 words, no line breaks.
- Mention only what's clearly visible.
- Keep the tone warm, human, and vivid."""
            # stateless tasks
            response = ollama.generate(
                model="llava-phi3:3.8b",
                prompt=prompt,
                images=[base64.b64encode(image_bytes).decode("utf-8")],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 100,
                    "repeat_penalty": 1.1
                }
            )

            description = response["response"].strip()
            print("✅ Model output:", description)

            model_results.append({
                "timestamp": dt,
                "description": description,
                "image_path": img_path_str,
                "source": "model"
            })

            gc.collect()

            time.sleep(1)

        except Exception as e:
            print(f"❌ Failed to process {img_path.name}: {e}")
            continue

    combined_results = existing_data + model_results

    with open(output_json, "w") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    print(f"\n🎯 Finished! Total {len(combined_results)} entries saved to {output_json.name}")