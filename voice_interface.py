# voice_interface.py

import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
import time
from scipy.io.wavfile import write as write_wav
from wake_word_listener import wait_for_wake_word
import re
from TTS.api import TTS

# Configuration
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 100)  # 放慢一点
# 初始化 Coqui TTS 引擎
tts_model = TTS(model_name="tts_models/en/ljspeech/speedy-speech", progress_bar=False, gpu=False)


# Load Whisper model
whisper_model = whisper.load_model("base.en")

import re

def estimate_tts_duration(text: str) -> float:
    """
    更准确地估算 TTS 播放时间（秒）。
    考虑单词数、字符数、标点停顿。
    """
    words = len(text.split())
    chars = len(text)
    
    # 统计标点符号数量（.,?! 都可能引起短暂停顿）
    punctuation_pauses = len(re.findall(r'[.,?!]', text))

    # 获取当前 TTS 设定的语速（words per minute）
    words_per_minute = tts_engine.getProperty('rate')
    words_per_minute = max(words_per_minute, 1)  # 防止除零错误

    # 基本单词朗读时间
    base_duration = (words / words_per_minute) * 60  # 单位是秒

    # 假设每个标点带来大约 0.3 秒停顿
    punctuation_delay = punctuation_pauses * 0.3

    # 假设每 100 个字符带来 1 秒处理时间（模拟文本复杂度）
    char_delay = (chars / 100) * 1.0

    total_duration = base_duration + punctuation_delay + char_delay

    return total_duration


# def speak_text(text: str):
#     """
#     Speak the given text using TTS.
#     """
#     print(f"🗣️ Speaking: {text}")
#     tts_engine.say(text)
#     tts_engine.runAndWait()
#     # avoid mic to capture this speak_text
#     time.sleep(estimate_tts_duration(text))

def speak_text(text: str):
    print(f"🗣️ Speaking: {text}")

    # 如果文本太短（比如小于2个单词），就跳过
    if len(text.strip().split()) < 2:
        print("⚡ Text too short, skipping TTS playback.")
        return

    # 正常用 Coqui TTS 合成+播放
    wav = tts_model.tts(text)
    sd.play(wav, samplerate=tts_model.synthesizer.output_sample_rate)
    sd.wait()
    time.sleep(0.3)

def record_audio(duration: int = RECORD_SECONDS) -> np.ndarray:
    """
    Record audio from microphone with basic error handling.
    """
    print("🔔 Please start speaking after the beep.")
    time.sleep(0.5)
    try:
        print("🎤 Recording...")
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("🎤 Recording complete.")
        return audio
    except Exception as e:
        print(f"[ERROR] Failed to record audio: {e}")
        raise


def recognize_speech(audio: np.ndarray) -> str:
    """
    Use Whisper to transcribe recorded audio.
    """
    temp_filename = "temp_audio.wav"
    write_wav(temp_filename, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print("🧠 Transcribing...")
    try:
        result = whisper_model.transcribe(temp_filename)
        text = result['text'].strip()
        print(f"📝 Recognized: {text}")
        return text
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return ""

def listen_to_question_with_confirmation() -> str:
    while True:
        # 先录问题
        audio = record_audio()
        question = recognize_speech(audio)

        if not question:
            speak_text("I didn't catch that. Could you please repeat your question?")
            continue

        speak_text(f"Did you say: {question}? Please say yes or no.")
        
        # confirm_audio = record_audio()
        # confirm_text = recognize_speech(confirm_audio).lower()
        # 开始监听
        while True:
            label = wait_for_wake_word("yesno")
            print(f"🎯 Detected label: {label}")
            if label == "yes":
                speak_text("Processing your request.")
                return question  # ✅ 确认yes后才return
            elif label == "no":
                speak_text("Let's try again.")
                break  # ❗ 再录一次问题
            else:
                time.sleep(0.1)
                continue


def record_note() -> str:
    """
    Special recording for a photo description, retry if no speech detected.
    """
    while True:
        speak_text("Please describe the photo after the beep.")
        audio = record_audio()
        note_text = recognize_speech(audio)

        if note_text:
            return note_text
        else:
            speak_text("I didn't catch that. Please describe the photo again.")

