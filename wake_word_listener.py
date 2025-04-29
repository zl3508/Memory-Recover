# wake_word_listener.py

import time
import re
import sys
import subprocess
from pathlib import Path

# === 配置 ===
LOG_FILE = Path("runner_output.log")
RUNNER_EXECUTABLE = "/usr/bin/edge-impulse-linux-runner"  # 确保 runner 在 memory_project/ 目录下
MODEL_PATH = "model.eim"  # Edge Impulse 导出的模型文件

WAKE_WORDS = ["take photo", "hi man"]
THRESHOLD = 0.7

runner_process = None  # 全局 runner 进程对象
last_read_pos = 0  # 全局日志读取位置

def start_edge_impulse_runner():
    """启动 edge-impulse-linux-runner，并把输出写入日志文件"""
    global runner_process
    if runner_process is not None and runner_process.poll() is None:
        print("⚡ Runner already running, no need to start.")
        return

    print("🚀 Starting Edge Impulse Runner...")
    log_file = open(LOG_FILE, "w")  # 每次启动时清空旧日志
    runner_process = subprocess.Popen(
        [RUNNER_EXECUTABLE, "--model", MODEL_PATH],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    time.sleep(2)  # 给 runner 稳定时间

def stop_edge_impulse_runner():
    """停止 runner 并确保麦克风资源释放"""
    global runner_process
    if runner_process and runner_process.poll() is None:
        print("🛑 Stopping Edge Impulse Runner...")
        runner_process.kill()
        runner_process.wait()
        runner_process = None
        time.sleep(2)  # 关键：等麦克风资源释放
    else:
        print("⚡ Runner not running, nothing to stop.")

def wait_for_wake_word():
    """
    持续监听 runner_output.log，直到检测到指定唤醒词
    返回检测到的 label (小写)
    """
    global last_read_pos

    print("👂 Monitoring runner_output.log for wake words...")

    # 确保 runner 已经启动
    start_edge_impulse_runner()

    # 等待日志文件生成
    while not LOG_FILE.exists():
        print("[INFO] Waiting for runner_output.log to appear...")
        time.sleep(1)

    while True:
        try:
            current_size = LOG_FILE.stat().st_size

            if current_size > last_read_pos:
                with open(LOG_FILE, "r") as f:
                    f.seek(last_read_pos)
                    new_data = f.read()
                    last_read_pos = f.tell()

                if "classifyRes" in new_data:
                    matches = re.findall(r"'(.*?)': '([\d.]+)'", new_data)
                    if matches:
                        predictions = {label: float(score) for label, score in matches}
                        top_label = max(predictions, key=lambda k: predictions[k])
                        confidence = predictions[top_label]

                        print(f"🎧 System detected: '{top_label}' with confidence {confidence:.2f}")

                        if confidence >= THRESHOLD:
                            print(f"✅ Wake word detected: {top_label}")

                            if top_label.lower() in WAKE_WORDS:
                                # 🔥 检测到目标唤醒词，停止 runner
                                stop_edge_impulse_runner()

                                # ✨ 清空日志，准备下一轮
                                LOG_FILE.write_text("")
                                last_read_pos = 0

                                return top_label.lower()
                            else:
                                print(f"⚡ Detected '{top_label}', but not a wake word. Ignoring.")

            time.sleep(0.3)

        except Exception as e:
            print(f"[ERROR] Error while monitoring log: {e}")
            time.sleep(1)

def restart_edge_impulse_runner():
    """执行完一次主逻辑后，重新启动 runner 开始下一轮监听"""
    global last_read_pos
    stop_edge_impulse_runner()
    start_edge_impulse_runner()
    last_read_pos = 0
