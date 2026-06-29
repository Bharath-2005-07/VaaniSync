# ⚡ VaaniSync Quick Run & Cheat Sheet Guide

This guide provides a quick-reference list of commands to run and test **VaaniSync** without having to look through the main documentation.

---

## 🛠️ Prerequisite Verification

Before running, make sure your virtual environment and Ollama LLM are ready:

```powershell
# 1. Start Ollama Server (Runs in background)
ollama serve

# 2. Verify or pull the translation LLM
ollama pull gemma2:2b
```

---

## 🚀 Execution Options

### Option A: Interactive Drag & Drop (Windows)
1. Locate your source video file (e.g. `video/virat_kohli.mp4`) in Windows File Explorer.
2. Drag and drop the video file directly onto the `run_dubbing.bat` script.
3. The batch script will automatically load the environment and run the pipeline.

---

### Option B: Visual Web UI (Interactive Graph)
To start the visual local development server provided by Google ADK:

```powershell
# Start the web UI server on port 8001
.\.venv\Scripts\adk.exe web video_localizer --port 8001
```
* **Access URL**: Open [http://127.0.0.1:8001](http://127.0.0.1:8001) in your browser.
* **Command Prompt**: Type `Convert the audio of video/virat_kohli.mp4 to Spanish` (or Kannada, French, Hindi, etc.) to trigger execution.

---

### Option C: Command Line Interface (CLI)
Run the pipeline directly from your PowerShell terminal:

```powershell
# Run the pipeline with any target language
.\.venv\Scripts\adk.exe run video_localizer "Convert the audio of video/virat_kohli.mp4 to Spanish"
```

---

## 🧪 Testing and Quality Control

### Running Unit and Integration Tests
Validate the multi-threaded synthesis, translation batching, and environment mocks:

```powershell
# Run all unit tests
.\.venv\Scripts\pytest -v

# Run a specific test with stdout logging
.\.venv\Scripts\pytest tests/test_pipeline.py -k test_synthesise_segments -s
```

---

## ⚙️ Project Outputs

* **Denoised Original Audio WAV**: `audio/original_audio.wav`
* **Dubbed Kannada Segments**: Located under `audio/dubbed_segments/`
* **Translated Transcripts**: `transcripts/translated_segments.json`
* **Final Multiplexed Video**: `output/virat_kohli.mp4`

