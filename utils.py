import numpy as np
import soundfile as sf
import simpleaudio as sa
import json
import os
import numpy as np
import threading

from maai import MaaiInput
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional


def fuse_emotions(text_emotions, audio_emotions, text_weight=0.7):
    if text_emotions == None or audio_emotions == []:
        return "neutral", 1, None

    audio_to_text_map = {
        "neu": "neutral",
        "hap": "joy",
        "ang": "anger",
        "sad": "sadness"
    }
    text_emotions = text_emotions[0]
    audio_aligned = {}
    for a_label, score in audio_emotions.items():
        t_label = audio_to_text_map.get(a_label)
        if t_label:
            audio_aligned[t_label] = score
    fused_scores = {}
    audio_weight = 1.0 - text_weight
    for item in text_emotions:
        label = item["label"]
        text_score = item["score"]
        if label in audio_aligned:
            audio_score = audio_aligned[label]
            fused_scores[label] = (text_weight * text_score + audio_weight * audio_score)
        else:
            # fallback: text only
            fused_scores[label] = text_score

    sorted_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    final_label, final_score = sorted_scores[0]
    
    return final_label, final_score, fused_scores


def get_sound_file(soundfile_dir, utterance, emotion):
    folder = os.path.join(soundfile_dir, utterance, emotion)
    if not os.path.exists(folder):
        return None
    files = [os.path.join(folder, f)
             for f in os.listdir(folder) if f.endswith(".wav")]
    if not files:
        return None

    return np.random.choice(files)


def play_sound(file):
    wave = sa.WaveObject.from_wave_file(file)
    wave.play()


def convert_unix_ts(unix_ts):
    dt_object = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    formatted_ts = dt_object.strftime(
        "%Y-%m-%d %H-%M-%S") + f".{dt_object.microsecond // 1000}"
    return formatted_ts


@dataclass
class Timestamps:
    audio_start: float
    audio_end: float
    transcription_start: float
    transcription_end: float
    llm_gen_start: float
    llm_gen_end: float
    tts_start: float
    tts_end: float

    def to_dict(self):
        return {
            "audio_start": convert_unix_ts(self.audio_start),
            "audio_end": convert_unix_ts(self.audio_end),
            "transcription_start": convert_unix_ts(self.transcription_start),
            "transcription_end": convert_unix_ts(self.transcription_end),
            "llm_gen_start": convert_unix_ts(self.llm_gen_start),
            "llm_gen_end": convert_unix_ts(self.llm_gen_end),
            "tts_start": convert_unix_ts(self.tts_start),
            "tts_end": convert_unix_ts(self.tts_end),
            "audio_start_to_end_ms": (self.audio_end - self.audio_start) * 1000,
            "transcription_time_ms": (self.transcription_end - self.transcription_start) * 1000,
            "llm_gen_time_ms": (self.llm_gen_end - self.llm_gen_start) * 1000,
            "tts_time_ms": (self.tts_end - self.tts_start) * 1000,
        }


@dataclass
class EmotionData:
    current_text_emotion: Optional[str] = None
    current_text_emotion_list: Optional[list] = None
    new_text_emotion: Optional[str] = None
    new_text_emotion_list: Optional[list] = None
    current_audio_emotion: Optional[str] = None
    current_audio_emotion_list: Optional[list] = None


@dataclass
class LogEntry:
    message: str
    output: str
    timestamps: Timestamps
    emotions: EmotionData
    bc_prob: Optional[list[float]] = None


def log_data(file_name: str, entry: LogEntry):
    os.makedirs(file_name, exist_ok=True)

    data = {
        "message": entry.message,
        "output": entry.output,
        "bc_prob": entry.bc_prob,
        **entry.emotions.__dict__,
        **entry.timestamps.to_dict(),
        "audio_start_ts": entry.timestamps.audio_start,
        "audio_end_ts": entry.timestamps.audio_end,
        "transcription_start_ts": entry.timestamps.transcription_start,
        "transcription_end_ts": entry.timestamps.transcription_end,
        "llm_gen_start_ts": entry.timestamps.llm_gen_start,
        "llm_gen_end_ts": entry.timestamps.llm_gen_end,
        "tts_start_ts": entry.timestamps.tts_start,
        "tts_end_ts": entry.timestamps.tts_end,
    }

    with open(os.path.join(file_name, "timestamp.json"), "w") as f:
        json.dump(data, f, indent=4)


# 
class WebSocketMic(MaaiInput.Base):
    def __init__(self, q):
        super().__init__()
        self.q = q
        self.buffer = np.zeros(0, dtype=np.float32)

    def start(self):
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        while True:
            data = self.q.get()
            chunk = np.frombuffer(data, dtype=np.float32)

            self.buffer = np.concatenate([self.buffer, chunk])

            while len(self.buffer) >= self.FRAME_SIZE:
                frame = self.buffer[:self.FRAME_SIZE]
                self.buffer = self.buffer[self.FRAME_SIZE:]

                self._put_to_all_queues(frame.tolist())
