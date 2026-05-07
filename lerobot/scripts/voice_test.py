from kokoro_onnx import Kokoro
import soundfile as sf
import subprocess
import tempfile
import os

kokoro = Kokoro('/home/trossen-ai/models/kokoro/kokoro-v1.0.onnx', '/home/trossen-ai/models/kokoro/voices-v1.0.bin')

test_text = "Ready to record robot data. Starting episode three."

voices = [v for v in kokoro.get_voices() if v.startswith('af_') or v.startswith('bf_')]
speeds = [0.85, 1.0]

for voice in voices:
    for speed in speeds:
        print(f"\nVoice: {voice}  Speed: {speed}")
        input("Press Enter to play...")
        try:
            samples, sample_rate = kokoro.create(test_text, voice=voice, speed=speed, lang='en-us')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                sf.write(tmp_file.name, samples, sample_rate)
            subprocess.run(['aplay', '-D', 'pulse', tmp_file.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.unlink(tmp_file.name)
        except Exception as e:
            print(f"  Failed: {e}")

print("\nDone!")
