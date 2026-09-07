import numpy as np
import soundfile as sf
import pyaudio
import asyncio
import time
import re
import json
import os

from kokoro_onnx import Kokoro
from timeit import default_timer as timer


def kokoro_tts(text: str, 
               voice: str = "af_heart", 
               out_path: str = "output.wav",
               model_path: str = "kokoro-v1.0.onnx",
               voices_path: str = "voices-v1.0.bin"):
    kokoro = Kokoro(model_path, voices_path)
    samples, sample_rate = kokoro.create(text, voice=voice)
    sf.write(out_path, samples, sample_rate)
    print(f"Saved: {out_path}")
    return out_path


def kokoro_tts_stream(text: str, 
                      voice: str = "af_heart",
                      model_path: str = "kokoro-v1.0.onnx",
                      voices_path: str = "voices-v1.0.bin",
                      chunk_size = 1024
                      ):
    
    kokoro = Kokoro(model_path, voices_path)
    start = timer()
    samples, sample_rate = kokoro.create(text, voice=voice)    
    end = timer()
    print(f"Execution time TTS: {end - start:.3f} seconds")
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        output=True,
        frames_per_buffer=chunk_size
    )
    total_samples = len(samples)


    for i in range(0, total_samples, chunk_size):
        chunk = samples[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        stream.write(chunk.astype(np.float32).tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()


def kokoro_tts_stream_split(text: str, 
                            voice: str = "af_heart",
                            model_path: str = "kokoro-v1.0.onnx",
                            voices_path: str = "voices-v1.0.bin",
                            chunk_size = 1024
                            ):
    kokoro = Kokoro(model_path, voices_path)
    p = pyaudio.PyAudio()

    # Split text into smaller parts (by '.')
    arr_text = [t.strip() for t in text.split(".") if t.strip()]

    stream = None

    # total_start = timer()
    for sentence in arr_text:
        # start = timer()
        samples, sample_rate = kokoro.create(sentence, voice=voice)
        # end = timer()
        # print(f"Execution time TTS for sentence: {end - start:.3f} seconds")
        if stream is None:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=chunk_size
            )

        total_samples = len(samples)
        for i in range(0, total_samples, chunk_size):
            chunk = samples[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            stream.write(chunk.astype(np.float32).tobytes())


    # total_end = timer()
    # print(f"Total execution time: {total_end - total_start:.3f} seconds")

    if stream is not None:
        stream.stop_stream()
        stream.close()
    p.terminate()


def tts_to_browser(text: str,
                          websocket,
                          loop,
                          file_name,
                          voice: str = "af_heart",
                          model_path: str = "kokoro-v1.0.onnx",
                          voices_path: str = "voices-v1.0.bin",
                        ):
    kokoro = Kokoro(model_path, voices_path)
    
    tts_end_ts = 0
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    
    for sentence in sentences:
        sample, sample_rate = kokoro.create(sentence, voice=voice)
        sample = np.array(sample, dtype=np.float32)
        asyncio.run_coroutine_threadsafe(
            websocket.send_bytes(sample.tobytes()),
            loop
        ).result()
        print("Sent audio bytes")
        if tts_end_ts == 0:
            tts_end_ts = time.time()
            file_path = os.path.join(file_name, "timestamp.json")
            with open(file_path, "r") as f:
                data = json.load(f)
            data["tts_end_ts"] = tts_end_ts
            data["tts_time_ms"] = (
                tts_end_ts - data["tts_start_ts"]
            ) * 1000
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)


if __name__ == "__main__":
    audio_file = kokoro_tts(
        "I am the bone of my sword. Steel is my body and fire is my blood. I have created over a thousand blades. Unknown to death, Nor known to life. Have withstood pain to create many weapons. Yet, those hands will never hold anything. So, as I pray— Unlimited Blade Works.",
        # "Hello? Can you here me?",
        # "One thing that always boggles my mind is how consciousness emerges from mere electrical signals in the brain. Billions of neurons fire in patterns, yet somehow that gives rise to thoughts, emotions, and a sense of self. It's wild to think that everything you experience—colors, music, love, memories—exists entirely as brain activity, and we still can't fully explain how or why that happens.",
        # "Right",
        voice="af_heart",
        out_path="test.wav"
    )
