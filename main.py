import numpy as np
import torch
import threading
import time
import os
import asyncio

from utils import *
from maai import Maai, MaaiInput
from io import BytesIO
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, BackgroundTasks,  WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, pipeline
from kokoroTTS import *
from ser import *
from contextlib import asynccontextmanager
from weakref import WeakSet
import queue


audio_queue = queue.Queue()

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
MAX_NEW_TOKENS = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("base.en", device=device, compute_type="float16")
# llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# llm_model_name = "google/gemma-2b-it"
# llm_model_name = "google/gemma-2-2b-it"
llm_model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    dtype="auto",
    device_map="auto"
)

ser_model_name = "superb/wav2vec2-base-superb-er"
ser_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    ser_model_name)
ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(ser_model_name)
id2label = ser_model.config.id2label


text_emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


current_audio_emotion = "neu"
current_audio_emotion_list = []
current_text_emotion = "neutral"
soundfiles_dir = "./soundfiles"
current_text_emotion_list = None
new_text_emotion = None
folder_path = os.path.abspath("./log")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
conversation_history = []


global_cd = 2
last_play_time = 0
tts_lock = threading.Lock()

push2talk = False

bc_prob_result = []
bc_lock = threading.Lock()
bc_thread = None
np.random.default_rng()
bc_categories = ["uhhuh", "yeah", "right", "okay"]
bc_probabilities = [0.5382, 0.325, 0.1107, 0.0261]

# backchannel prediction stream
MAX_HISTORY = 50000
MAX_LLM_TURN_HISTORY = 20


clients = WeakSet()
loop = None


def detect_text_emotion(text):
    res = text_emotion_pipeline(text)
    global new_text_emotion
    new_text_emotion = max(res[0], key=lambda x: x["score"])["label"]  # type: ignore
    return res


def run_bc_pred():
    global bc_prob_result
    global last_play_time
    global loop

    mic = WebSocketMic(audio_queue)
    zero = MaaiInput.Zero()


    maai = Maai(
        mode="bc",
        lang="en",
        frame_rate=10,
        audio_ch1=mic,
        audio_ch2=zero,
        device=device
    )

    maai.start()

    while True:
        result = maai.get_result()

        ts = time.time()

        with bc_lock:
            bc_prob_result.append((ts, result["p_bc"]))
            bc_trigger(result)
            if len(bc_prob_result) > MAX_HISTORY:
                bc_prob_result.pop(0)

        if loop:
            asyncio.run_coroutine_threadsafe(broadcast(result["p_bc"]), loop)


def bc_trigger(result):
    global last_play_time
    now = time.time()
    if (result['p_bc'] >= 0.6
        and now - last_play_time >= global_cd 
        and not tts_lock.locked() 
        and push2talk):
        last_play_time = now
        # uhhuh, yeah, okay, right
        bc_utterance = np.random.choice(bc_categories, p=bc_probabilities)

        # weighted current_text_emotion current_audio_emotion
        try:
            label, score, all_scores = fuse_selected_emotions(
                current_text_emotion_list, current_audio_emotion_list[-1])
        except:
            label, score, all_scores = fuse_selected_emotions(
                current_text_emotion_list, [])
            
        soundfile = get_sound_file(soundfiles_dir, bc_utterance, label)
        if soundfile != None:
            threading.Thread(target=play_sound, args=(
                soundfile,), daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bc_thread
    global loop

    loop = asyncio.get_running_loop()
    app.state.tts_ws = None
    app.state.bc_result = None
    bc_thread = threading.Thread(target=run_bc_pred, daemon=True)
    bc_thread.start()

    print("MaAI started")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def index():
    with open("./frontend/index.html") as f:
        return HTMLResponse(f.read())


@app.post("/audio")
async def receive_audio(file: UploadFile, background_tasks: BackgroundTasks):
    global current_text_emotion
    global current_text_emotion_list
    global conversation_history
    # receive Audio
    audio_bytes = await file.read()
    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))
    audio_received_end_ts = time.time()
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

    duration = len(audio_segment) / 1000.0
    audio_received_start_ts = audio_received_end_ts - duration

    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio_segment.array_type).max

    # transcribe
    transcription_start_ts = time.time()
    segments, _ = whisper_model.transcribe(samples, beam_size=10) # type: ignore
    transcription_end_ts = time.time()
    text = "".join(seg.text for seg in segments)
    print(text)

    # generate output
    conversation_history.append({"role": "user", "content": text})
    try:
        prompt = tokenizer.apply_chat_template( # type: ignore
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        # 
        conversation_history.clear()
        prompt = tokenizer.apply_chat_template( # type: ignore
            conversation_history.append({"role": "user", "content": text}),
            tokenize=False,
            add_generation_prompt=True
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device) # type: ignore

    llm_gen_start_ts = time.time()

    output = llm_model.generate( # type: ignore
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS
    )

    llm_gen_end_ts = time.time()

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

    assistant_text = tokenizer.decode( # type: ignore
        generated_tokens,
        skip_special_tokens=True
    ).replace("*", "")

    tts_start_ts = time.time()
    print("Assistant:", assistant_text)
    conversation_history.append({"role": "assistant", "content": assistant_text})

    def safe_tts():
        with tts_lock:
            ws = app.state.tts_ws
            if ws is None:
                return
            tts_to_browser(assistant_text, ws, loop)

    background_tasks.add_task(safe_tts)
    tts_end_ts = time.time()
    formatted_ts = convert_unix_ts(audio_received_start_ts)

    # Text Sentiment/Emotion Analysis
    new_text_emotion_list = detect_text_emotion(text)

    # trim conv history
    if len(conversation_history) > MAX_LLM_TURN_HISTORY:
        conversation_history = conversation_history[-MAX_LLM_TURN_HISTORY:]
    file_name = os.path.join(folder_path, f"MSG_{formatted_ts}.log")

    os.makedirs(file_name)
    with bc_lock:
        bc_prob = [
            value for (ts, value) in bc_prob_result
            if audio_received_start_ts <= ts <= audio_received_end_ts
        ]

    timestamps = Timestamps(
        audio_start=audio_received_start_ts,
        audio_end=audio_received_end_ts,
        transcription_start=transcription_start_ts,
        transcription_end=transcription_end_ts,
        llm_gen_start=llm_gen_start_ts,
        llm_gen_end=llm_gen_end_ts,
        tts_start=tts_start_ts,
        tts_end=tts_end_ts
    )

    emotions = EmotionData(
        current_text_emotion=current_text_emotion,
        current_text_emotion_list=current_text_emotion_list,
        new_text_emotion=new_text_emotion,
        new_text_emotion_list=new_text_emotion_list,
        current_audio_emotion=current_audio_emotion,
        current_audio_emotion_list=current_audio_emotion_list
    )
    
    entry = LogEntry(
        message=text,
        output=assistant_text,
        timestamps=timestamps,
        emotions=emotions,
        bc_prob=bc_prob,
    )
    log_data(file_name, entry)
    
    audio_segment.export(os.path.join(file_name, ".wav"), format="wav")

    # update current emotion
    current_text_emotion = new_text_emotion
    current_text_emotion_list = new_text_emotion_list
    return {"transcript": text, "response": assistant_text}


@app.post("/emotion_recognition")
async def emotion_recognition(file: UploadFile, background_tasks: BackgroundTasks):
    global current_audio_emotion
    global current_audio_emotion_list
    current_audio_emotion_list.clear()

    audio_bytes = await file.read()
    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio_segment.array_type).max
    # sample_rate * seconds
    chunk_size = 16000 * 5
    results = []
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start:start + chunk_size]
        if len(chunk) == 0:
            continue
        emotion, confidence, emotion_scores = predict_emotion(
            ser_model, id2label, ser_feature_extractor, chunk)
        results.append({"emotion": emotion, "confidence": confidence})
        current_audio_emotion_list.append(emotion_scores)
        if confidence >= 0.8:
            current_audio_emotion = emotion

    print(f"Scores: {current_audio_emotion_list}")  # type: ignore

    return {"segments": results}


# websocket
async def broadcast(data):
    disconnected = []
    
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception as e:
            print("WS ERROR:", e)
            disconnected.append(ws)
    for ws in disconnected:
        clients.discard(ws)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global push2talk
    await websocket.accept()

    clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            state = msg.get("state")

            if msg_type == "mic" and state == "start":
                if push2talk:
                    continue
                push2talk = True

            elif msg_type == "mic" and state == "stop":
                if not push2talk:
                    continue
                push2talk = False
    except WebSocketDisconnect:
        clients.discard(websocket)


@app.websocket("/ws_audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_queue.put(data)

    except Exception as e:
        print("ws_audio error:", e)


@app.websocket("/ws_tts")
async def ws_tts(websocket: WebSocket):
    await websocket.accept()
    websocket.app.state.tts_ws = websocket

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket.app.state.tts_ws = None
