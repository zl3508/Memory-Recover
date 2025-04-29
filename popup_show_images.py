# popup_show_images.py

from tkinter import Tk, Label, StringVar
from PIL import Image, ImageTk
import time
from pathlib import Path

def popup_images(image_paths, delay=5):
    """
    Show a sequence of images in fullscreen popup window with small titles.
    
    Args:
        image_paths (List[str]): List of image file paths to display.
        delay (int): Time to show each image (in seconds).
    """

    root = Tk()
    root.title("Memory Reference Images")

    # ✅ 全屏模式
    root.attributes("-fullscreen", True)

    # ✅ 创建标题 + 图片标签
    title_var = StringVar()
    title_label = Label(root, textvariable=title_var, font=("Arial", 24), pady=10)
    title_label.pack()

    image_label = Label(root)
    image_label.pack()

    def show_image(img_path, idx, total):
        img = Image.open(img_path)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        img = img.resize((screen_width, screen_height - 100), Image.LANCZOS)  # 留出标题高度
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # 保存引用防止回收

        # 更新标题
        title_var.set(f"Reference Image {idx+1}/{total}")

    # ✅ 允许按 Esc 退出
    def close(event=None):
        root.destroy()

    root.bind("<Escape>", close)

    # ✅ 播放每张图
    valid_images = [p for p in image_paths if Path(p).exists()]
    total = len(valid_images)

    for idx, img_path in enumerate(valid_images):
        show_image(img_path, idx, total)
        root.update()
        time.sleep(delay)

    root.destroy()
