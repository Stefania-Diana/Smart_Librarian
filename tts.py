import os
import pyttsx3

def synthesize_to_wav(text: str, out_path: str = "data/answer.wav") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    engine = pyttsx3.init()
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path
