import time
import base64
from pathlib import Path
from PIL import Image
import io
import ollama
import gc

def load_and_encode_image(img_path: Path, resize: bool = False) -> str:
    with Image.open(img_path) as image:
        if resize:
            image = image.resize((576, 324))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")

def run_inference(encoded_image: str, dt: str):
    prompt = f"""
You are a memory assistant.

Describe the attached photo to help someone recall the moment it was taken.

Photo timestamp: {dt}

Rules:
- Max 3 sentences, 80 words, no line breaks.
- Mention only what's clearly visible.
- Keep the tone warm, human, and vivid.
    """
    response = ollama.generate(
        model="llava-phi3:3.8b",
        prompt=prompt,
        images=[encoded_image],
        options={
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 100,
            "repeat_penalty": 1.1
        }
    )
    return response["response"]

def test_resized_vs_original(img_path: str):
    dt = "2025-04-29 13:00"

    gc.collect()

    print("🔍 Testing with resized image...")
    start = time.time()
    resized_img = load_and_encode_image(Path(img_path), resize=True)
    resized_output = run_inference(resized_img, dt)
    end = time.time()
    print(f"✅ Resized done in {end - start:.2f}s\n")
    print("📋 Resized Output:\n", resized_output, "...\n")
    
    gc.collect()
    print("🔍 Testing with original image...")
    start = time.time()
    original_img = load_and_encode_image(Path(img_path), resize=False)
    original_output = run_inference(original_img, dt)
    end = time.time()
    print(f"✅ Original done in {end - start:.2f}s\n")
    print("📋 Original Output:\n", original_output, "...\n")

    gc.collect()
if __name__ == "__main__":
    test_resized_vs_original("memory_images/img_20250429_122353.jpg")
