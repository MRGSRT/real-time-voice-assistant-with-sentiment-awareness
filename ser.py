import sounddevice as sd
import numpy as np
import torch
import webrtcvad
import soundfile as sf
import librosa
import os

from sklearn.metrics import accuracy_score, classification_report
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


def predict_emotion(model, id2label, ser_feature_extractor, audio, sample_rate=16000):
    inputs = ser_feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    # inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        print(probs)
        # pred_id = torch.argmax(probs, dim=-1).item()
        pred_id = int(torch.argmax(probs, dim=-1).item())
        emotion_scores = {
            id2label[i]: float(probs[0][i])
            for i in range(len(id2label))
        }

    return id2label[pred_id], probs[0][pred_id].item(), emotion_scores


def float_to_pcm16(audio):
    audio = (audio * 32767).astype(np.int16)
    return audio.tobytes()


def listen(model, id2label, ser_feature_extractor):
    print("Listening..")
    buffer = []
    while True:
        audio = sd.rec(frame_size, samplerate=sample_rate,
                       channels=1, dtype='float32')
        sd.wait()

        audio = audio.flatten()
        pcm = float_to_pcm16(audio)

        is_speech = vad.is_speech(pcm, sample_rate)

        if is_speech:
            buffer.extend(audio)

            if len(buffer) >= chunk_size:
                chunk = np.array(buffer[:chunk_size])
                buffer = buffer[chunk_size:]
                emotion, conf, _ = predict_emotion(model, id2label, ser_feature_extractor, chunk)
                print(f"{emotion} | {conf:.2f}")
        else:
            if buffer:
                buffer = []


def predict_emotion_from_file(file_path, model, ser_feature_extractor, id2label, target_sr=16000, chunk_size=16000*3):
    audio, sr = sf.read(file_path)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # mono channel
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    results = []
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        if len(chunk) == 0:
            continue

        emotion, conf, _ = predict_emotion(model, id2label, ser_feature_extractor, chunk)
        results.append((emotion, conf))

    print(results)
    return results



if __name__ == "__main__":
    vad = webrtcvad.Vad(2)

    sample_rate = 16000
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000)
    chunk_size = 16000 * 3
    # ser_model_name = "superb/wav2vec2-base-superb-er"
    ser_model_name = "Dpngtm/wav2vec2-emotion-recognition"
    ser_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ser_model_name)
    ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(ser_model_name)
    id2label = ser_model.config.id2label
    print(ser_model.config.id2label)
    print(ser_model.config)
    # listen(ser_model, id2label, ser_feature_extractor)
    # predict_emotion_from_file("qwentts.wav", ser_model, ser_feature_extractor, id2label)
