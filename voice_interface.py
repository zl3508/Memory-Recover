# voice_interface.py

import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
import time
from scipy.io.wavfile import write as write_wav
from wake_word_listener import wait_for_wake_word

# Configuration
SAMPLE_RATE = 16000
RECORD_SECONDS = 6

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 100)  # 放慢一点

# Load Whisper model
whisper_model = whisper.load_model("base.en")

def speak_text(text: str):
    """
    Speak the given text using TTS.
    """
    print(f"🗣️ Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

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
        label = wait_for_wake_word("yesno")
        print(f"🎯 Detected label: {label}")
        if label == "yes":
            speak_text("Understood. Processing your request.")
            return question  # ✅ 确认yes后才return
        elif label == "no":
            speak_text("Okay, let's try again.")
            continue  # ❗ 再录一次问题
        else:
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

