# image_processing.py

from pathlib import Path
import base64
import json
from datetime import datetime
import ollama

def generate_image_descriptions(image_folder: Path, output_json: Path) -> None:
    """
    Process images using vision-language model and save descriptions.
    Skip images that have already been processed.
    """
    # 初始化
    image_folder = Path(image_folder)
    output_json = Path(output_json)
    model_results = []

    # 如果output_json已经存在，先加载里面已有的记录
    if output_json.exists():
        with open(output_json, "r") as f:
            existing_data = json.load(f)
        processed_images = set(entry["image_path"] for entry in existing_data)
        print(f"✅ Loaded {len(processed_images)} already processed images.")
    else:
        existing_data = []
        processed_images = set()
        print("✅ No existing processed images found. Starting fresh.")

    # 遍历图片文件夹
    for img_path in image_folder.glob("*.jpg"):
        img_path_str = str(img_path)
        if img_path_str in processed_images:
            print(f"⚡ Skipping already processed: {img_path.name}")
            continue

        print(f"🚀 Processing new image: {img_path.name}")
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        # 提取时间戳
        parts = img_path.stem.split("_")
        if len(parts) >= 3:
            timestamp_raw = parts[1] + parts[2]  # e.g., 20250420 + 1637
            dt = datetime.strptime(timestamp_raw, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M")
        else:
            print(f"⚠️ Filename format unexpected: {img_path.name}, skipping.")
            continue

        # 发给模型生成描述
        try:
            response = ollama.chat(
                model="llava-phi3:3.8b",
                messages=[{
                    "role": "user",
                    "content": f"Describe what's in this image taken at {dt}.",
                    "images": [base64.b64encode(image_bytes).decode("utf-8")],
                }],
                options={"temperature": 0.3}
            )
            description = response["message"]["content"].strip()
            print("✅ Model output:", description)

            model_results.append({
                "timestamp": dt,
                "description": description,
                "image_path": img_path_str,
                "source": "model"
            })

        except Exception as e:
            print(f"❌ Failed to process {img_path.name}: {e}")
            continue

    # 合并新数据 + 已有数据
    combined_results = existing_data + model_results

    # 保存到JSON
    with open(output_json, "w") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    print(f"\n🎯 Finished! Total {len(combined_results)} entries saved to {output_json.name}")

