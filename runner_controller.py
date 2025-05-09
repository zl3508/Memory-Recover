# runner_controller.py

import subprocess
import os
import signal

RUNNER_COMMAND = "edge-impulse-linux-runner"
RUNNER_ARGS = ["--model", "model.eim", "--quiet"]
LOG_FILE = "runner_output.log"

runner_process = None

def start_runner():
    global runner_process

    print("🚀 Starting Edge Impulse Runner...")

    with open(LOG_FILE, "w") as log_file:
        runner_process = subprocess.Popen(
            [RUNNER_COMMAND] + RUNNER_ARGS,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

def stop_runner():
    global runner_process

    if runner_process and runner_process.poll() is None:
        print("🛑 Stopping Edge Impulse Runner...")
        runner_process.terminate()
        try:
            runner_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            runner_process.kill()
            runner_process.wait()
        runner_process = None
    else:
        print("⚡ Runner not running, nothing to stop.")

def ensure_log_file_clean():
    if os.path.exists(LOG_FILE):
        open(LOG_FILE, "w").close()
