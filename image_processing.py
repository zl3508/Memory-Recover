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
            print(f"⚡ Skipping already processed: {img_path.name}")
            continue

        print(f"🚀 Processing new image: {img_path.name}")
        try:
            with Image.open(img_path) as image:
                image = image.resize((576, 324))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()

            # 提取时间
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

            # 发给模型
            prompt = f"""
            You are a memory recovery assistant.

            Given the attached photo, generate a **natural and vivid** description to help someone recall the moment it was captured, especially for those who might have forgotten.

            Focus on:
            - The key subjects clearly visible (people, buildings, objects, landscapes).
            - The broader environment (city street, village home, countryside, indoor room).
            - Any noticeable time or lighting clues (morning, evening, sunny, rainy).
            - The overall feeling or mood suggested by the scene (peaceful, lively, nostalgic).

            Important:
            - Write as if you are helping a person emotionally reconnect with their memory.
            - Be concise but rich, using **no more than 3-4 sentences**.
            - **Do not invent** objects not seen in the photo.
            - Keep the tone warm, descriptive, and human — **not mechanical**.

            This photo was taken at: {dt}.
            """
            # stateless tasks
            response = ollama.generate(
                model="llava-phi3:3.8b",
                prompt=prompt,
                images=[base64.b64encode(image_bytes).decode("utf-8")],
                options={
                    "temperature": 0.3,
                    "num_ctx": 1024,
                    "num_predict": 300
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

            # 防止内存泄露
            gc.collect()

            # 每张图片处理后休息1秒
            time.sleep(1)

        except Exception as e:
            print(f"❌ Failed to process {img_path.name}: {e}")
            continue

    combined_results = existing_data + model_results

    with open(output_json, "w") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    print(f"\n🎯 Finished! Total {len(combined_results)} entries saved to {output_json.name}")