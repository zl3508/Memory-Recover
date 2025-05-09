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

    # img_ymd_hms.jpg
    timestamp = datetime.now(timezone("America/New_York")).strftime("%Y%m%d_%H%M%S")
    filename = f"img_{timestamp}.jpg"
    filepath = save_path / filename

    # use libcamera-still 
    try:
        print("📸 Capturing image with libcamera-still...")
        # sleep for config
        time.sleep(0.5)
        subprocess.run([
            "libcamera-still",
            "-o", str(filepath),
            "--width", "2304",
            "--height", "1296",
            "-t", "2000"  # make sure to clear
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