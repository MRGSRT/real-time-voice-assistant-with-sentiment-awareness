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

conversation_history = []
tts_lock = threading.Lock()

push2talk = False

# backchannel prediction stream
MAX_HISTORY = 50000
MAX_LLM_TURN_HISTORY = 20


clients = WeakSet()
loop = None




@asynccontextmanager
async def lifespan(app: FastAPI):
    global bc_thread
    global loop

    loop = asyncio.get_running_loop()
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
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio_segment.array_type).max

    # transcribe
    segments, _ = whisper_model.transcribe(samples, beam_size=10) # type: ignore
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
        conversation_history.clear()
        prompt = tokenizer.apply_chat_template( # type: ignore
            conversation_history.append({"role": "user", "content": text}),
            tokenize=False,
            add_generation_prompt=True
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device) # type: ignore

    output = llm_model.generate( # type: ignore
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS
    )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

    assistant_text = tokenizer.decode( # type: ignore
        generated_tokens,
        skip_special_tokens=True
    ).replace("*", "")

    print("Assistant:", assistant_text)
    conversation_history.append({"role": "assistant", "content": assistant_text})

    def safe_tts():
        with tts_lock:
            ws = app.state.tts_ws
            if ws is None:
                return
            tts_to_browser(assistant_text, ws, loop)

    background_tasks.add_task(safe_tts)

    # trim conv history
    if len(conversation_history) > MAX_LLM_TURN_HISTORY:
        conversation_history = conversation_history[-MAX_LLM_TURN_HISTORY:]
    
    return {"transcript": text, "response": assistant_text}


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
