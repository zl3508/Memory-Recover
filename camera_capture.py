import cv2
import time
from datetime import datetime
from pathlib import Path

def capture_image(save_folder="memory_images"):
    """
    Capture an image using the default camera and save it to the specified folder.
    """
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    # 打开摄像头（设备0）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Failed to open camera.")

    # 小等待，防止黑屏
    time.sleep(0.5)

    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("❌ Failed to capture image.")

    # 生成文件名：img_年月日_时分.jpg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"img_{timestamp}.jpg"
    filepath = save_path / filename

    # 保存
    try:
        cv2.imwrite(str(filepath), frame)
        print(f"✅ Captured and saved image: {filepath}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to save image: {e}")

    cap.release()
    return str(filepath)
