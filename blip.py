from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 加载模型和处理器
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 加载图片
image = Image.open("./memory_images/img_20250429_133120.jpg").convert("RGB")

# 推理
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("📝 Caption:", caption)