# real-time-voice-assistant-with-sentiment-awareness

Master Thesis

# Installation
This project uses **Python 3.11**, but it may also work with other **Python 3.1x** versions.

## 1. KokoroTTS
Download the required model files to your working directory:

```bash
# Download voice data (bin format is preferred)
wget -O voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

# Download the model
wget -O kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
```

## 2a. Using pip
Create a virtual environment and install dependencies from `requirements.txt`:
```bash
python -m venv venv
pip install -r requirements.txt
```
## 2b. Anaconda (Windows only atm)
Create the Conda environment from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate va
```

# Start Application
Run the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```
By default, the server will be available at: `http://127.0.0.1:8000`
Press **SPACE** to start talking.
