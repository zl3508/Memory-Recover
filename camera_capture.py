import subprocess
import time
from datetime import datetime
from pathlib import Path
from pytz import timezone

def capture_image(save_folder="memory_images"):
    """
    Capture an image using libcamera-still and save it to the specified folder.
    """
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    # 生成文件名：img_年月日_时分秒.jpg
    timestamp = datetime.now(timezone("America/New_York")).strftime("%Y%m%d_%H%M%S")
    filename = f"img_{timestamp}.jpg"
    filepath = save_path / filename

    # 使用 libcamera-still 拍照
    try:
        print("📸 Capturing image with libcamera-still...")
        # 先睡一下，避免摄像头没初始化完
        time.sleep(0.5)
        subprocess.run([
            "libcamera-still",
            "-o", str(filepath),
            "--width", "2304",
            "--height", "1296",
            "-t", "2000"  # 拍照延迟时间2秒，保证清晰
        ], check=True)
        print(f"✅ Captured and saved image: {filepath}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Failed to capture image: {e}")

    return str(filepath)

if __name__ == "__main__":
    try:
        print("📷 Testing image capture using libcamera...")
        img_path = capture_image()
        print(f"✅ Test complete. Image saved at: {img_path}")
    except Exception as e:
        print(f"❌ Test failed: {e}")