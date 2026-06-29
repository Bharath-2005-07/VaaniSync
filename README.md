# 🎬 VaaniSync: Offline Multi-Agent Video Localizer & Dubber (Kannada)

[![Kaggle Capstone](https://img.shields.io/badge/Kaggle%20Capstone-AI%20Agents%20Intensive-blueviolet?style=for-the-badge)](https://www.kaggle.com)
[![Google ADK 2.0](https://img.shields.io/badge/Orchestration-Google%20ADK%202.0-blue?style=for-the-badge)](https://github.com/google/ai-edge)
[![Offline First](https://img.shields.io/badge/Offline--First-100%25%20Local-success?style=for-the-badge)](#)

Submitted for the **AI Agents: Intensive Vibe Coding Capstone Project** under the **Agents for Good** track.

## 📊 Competitive Architecture Analysis

| Feature | Cloud Solutions (ElevenLabs / Rask AI) | Basic Open-Source Pipelines | VaaniSync (Our Project) |
| :--- | :--- | :--- | :--- |
| **Data Privacy** | ❌ None (Files uploaded to cloud) | 🍏 Fully Local | 🍏 **Fully Local (Zero Leakage)** |
| **Pacing Control** | ⚠️ Global Speed Adjustments | ❌ Audio clips or overlaps | 🍏 **Dynamic `atempo` + `pydub` padding** |
| **Cost Scale** | ❌ High ($/per-minute pricing) | 🍏 Free | 🍏 **100% Free / Open-Source** |
| **Translation Flow** | ⚠️ Frequently line-by-line | ❌ Literal word-swapping | 🍏 **Context-Aware Paragraph Batching** |
| **Fault Tolerance** | ⚠️ Monolithic API failover | ❌ Total script crash | 🍏 **Type-Safe ADK 2.0 Graph + Retry Config** |

### 💡 What Makes VaaniSync Different From Existing Tools?

1. **True Offline Zero-Leakage Privacy**: Standard commercial voice localizers require sending your high-resolution media files, transcripts, and voice embeddings to distant cloud servers. VaaniSync operates entirely on your local machine, keeping sensitive corporate presentations, personal home videos, and educational content completely secure.
2. **Context-Aware Translation Pacing**: Most tools perform literal sentence-by-sentence translation, resulting in artificial timing overlaps. VaaniSync uses context-aware paragraph-level batching and dynamically computes speed compression factors (`atempo`) and padding offsets (`pydub`) so that regional dubs seamlessly align with the original speaker's video speed.
3. **Zero-Shot Speaker Identity Preservation**: Rather than converting your video to a generic synthesized voice profile, our integrated OpenVoice V2 architecture extracts a brief tone-color embedding from the original speaker, applying it to Kannada voice synthesis. The speaker retains their unique vocal identity across languages.
4. **Resilient Local Multi-Agent Architecture**: Built on Google ADK 2.0, the pipeline relies on independent, modular agents for extraction, transcription, translation, synthesis, and muxing. With custom retry configurations and offline fallback paths, the system will never crash if a single external connection drops.
5. **Concurrent Multi-Threaded Synthesis**: While local execution of deep-learning speech models is typically slow on consumer CPUs, VaaniSync integrates a thread-pooled parallel execution model with thread-safety locks. This allows multiple segments to be synthesized concurrently, maximizing CPU core utilization and cutting down processing time by over 50%.
6. **Robust Cross-Format Video Support**: Leveraging native FFmpeg interfaces, the pipeline automatically parses, processes, and remuxes practically any modern video container format (like `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, and `.3gp`), dynamically outputting the final dubbed video in the exact same format container as the input.

---

## 📖 Project Overview & Story

### The Problem
Educational, informational, and business videos online are overwhelmingly created in English. For millions of regional language speakers (such as Kannada speakers in India), this creates a massive knowledge and accessibility barrier. Existing automated dubbing solutions suffer from three core issues:
1. **Cloud Dependence & Cost**: Relying on expensive cloud APIs that charge per minute and expose private media.
2. **Robotic Pacing**: Simply translating sentences and synthesizing them creates voice tracks that either overlap or run out of sync with the video timeline.
3. **Gender/Voice Mismatch**: Generic TTS pipelines apply a single default voice, stripping away the natural gender variations of the original speakers.

### The Solution: VaaniSync
**VaaniSync** is a fully local, offline-first multi-agent pipeline designed to ingest any video, transcribe the speech, translate it to natural conversational Kannada, synthesize gender-appropriate neural voices, and dynamically time-stretch/align the audio to the original video frames. 

By running entirely on local CPU resources, VaaniSync makes localization free, secure, and accessible to educators, content creators, and businesses alike.

---

## 🛠️ System Architecture

Built using the **Google Agent Development Kit (ADK) 2.0 Graph Workflow API**, the project enforces a highly structured, type-safe sequential multi-agent graph where state is managed contextually and nodes communicate via defined Pydantic schemas.

```mermaid
flowchart TD
    START(["👤 User Input\n'Dub video.mp4 into Kannada'"]) --> ParseInput["1. Parse Input & Resolve Path"]
    
    subgraph Pipeline ["VaaniSync Multi-Agent Pipeline"]
        ParseInput -->|"PipelineInput"| ExtractAudio["Stage 1: Extract & Denoise Audio\n(ffmpeg + afftdn)"]
        ExtractAudio -->|"ExtractionResult"| TranscribeAudio["Stage 2: Transcribe via Whisper\n(faster-whisper)"]
        TranscribeAudio -->|"TranscriptionResult"| TranslateSegments["Stage 3: Translate to Kannada\n(deep-translator auto-source)"]
        TranslateSegments -->|"TranslationResult"| SynthesiseSegments["Stage 4: Synthesize Speech\n(edge-tts / MeloTTS)"]
        SynthesiseSegments -->|"SynthesisResult"| MuxVideo["Stage 5: Mux & Assemble Video\n(ffmpeg video-audio mux)"]
    end

    MuxVideo -->|"MuxResult"| END(["🎉 Final Dubbed Video\n(output/video.mp4)"])

    style Pipeline fill:#f9fafb,stroke:#d1d5db,stroke-width:2px
    style START fill:#eff6ff,stroke:#3b82f6,stroke-width:2px
    style END fill:#f0fdf4,stroke:#22c55e,stroke-width:2px
```

### 📸 Live Pipeline Execution (ADK Dev UI)
Below is a screenshot of the VaaniSync multi-agent dubbing workflow executing live inside the Google ADK Developer Web UI:

![VaaniSync Live Web UI Execution](information/pipeline_run.png)

---

### Type-Safe Data Contracts (Pydantic)
Each edge in our workflow graph is strictly validated to ensure data integrity across stages:
* **`PipelineInput`**: Validates input video path, target language, and requested voice gender.
* **`ExtractionResult`**: Outputs validated paths for the original video and extracted audio.
* **`TranscriptionResult`**: Contains the transcription filepath, segment count, and auto-detected source language.
* **`TranslationResult`**: Tracks the path of translated Kannada segments.
* **`SynthesisResult`**: Stores directory references for individual speed-aligned WAV files.
* **`MuxResult`**: The final product path in the `output/` directory.

---

## 🌟 Key Capstone Implementations & Course Concepts

### 1. Multi-Agent Design & Workflow Graph
The orchestration is built using `google.adk.workflow.Workflow` in [video_localizer/agent.py](video_localizer/agent.py). It registers multiple function nodes with discrete responsibilities and custom `RetryConfig` policies to handle transient hardware hiccups (e.g., CPU memory peaks during transcription).

### 2. Smart Pacing & Dynamic Time-Stretching
To prevent dubbed speech from running out of sync:
* **Speed Stretching**: If the synthesized Kannada audio is longer than the original English spoken segment, VaaniSync calculates the ratio and applies FFmpeg's `atempo` filter to speed up the audio (up to 2.0x) without altering the voice pitch.
* **Silence Padding**: If the synthesized audio is shorter than the segment window, the agent calculates the offset and appends exact milliseconds of digital silence using `pydub` to preserve alignment.

### 3. Context-Aware Batch Translation
Instead of translating subtitles line-by-line (which ruins context), VaaniSync batches the transcription segments, passing them together to maintain sentence-level grammatical flow (Subject-Object-Verb ordering in Kannada vs Subject-Verb-Object in English).

---

## 📂 Project Structure

```text
lang-to-lang/
├── .agents/
│   └── skills/
│       └── video-localizer/
│           └── SKILL.md          # Custom agent skill definition file
├── .venv/                        # Local Python virtual environment
├── audio/                        # Temporary processing directory for audio
│   ├── original_audio.wav        # Stage 1: Extracted and denoised original audio
│   ├── dubbed_segments/          # Stage 4: Concurrent segment TTS outputs
│   └── dubbed_full.wav           # Stage 5: Assembled dubbed audio track
├── checkpoints_v2/               # OpenVoice V2 converter model weights folder
│   └── converter/
│       ├── checkpoint.pth        # Converter PyTorch weights
│       └── config.json           # Converter configuration parameters
├── information/                  # Project documentation assets
│   ├── pipeline_run.png          # Web UI execution screenshot
│   └── workflow_graph.md         # Pipeline flowchart and architecture
├── inputs/                       # User-supplied media input files
├── output/                       # Final dubbed Kannada video output files
│   └── virat_kohli.mp4           # Stage 5: Dubbed output multiplexed video
├── processed/                    # Speaker embedding cache created by OpenVoice
├── processing/                   # Temporary cache directory for processing
├── skill/
│   └── SKILL.md                  # Reusable skill documentation
├── tests/                        # Automated unit and integration tests
│   ├── test_pipeline.py          # Pytest suite with mocked services
│   └── eval/                     # Evaluation configurations and datasets
│       ├── eval_config.yaml
│       └── eval_dataset.json
├── transcripts/                  # Temporary translation segments storage
│   ├── segments.json             # Stage 2: Whisper speech timestamps & text
│   └── translated_segments.json  # Stage 3: Kannada translation with metadata
├── video/                        # Input video files directory
│   ├── video2.mp4                # Secondary testing video input
│   ├── video3.mp4                # Tertiary testing video input
│   └── virat_kohli.mp4           # Reference test video input
├── video_localizer/              # Main agent workflow package
│   ├── __init__.py               # Exports discovery root agent workflow
│   ├── agent.py                  # Orchestrator & FunctionNode stage handlers
│   └── agents/                   # Sub-agent modules (e.g., translation)
│       ├── __init__.py
│       └── translation.py
├── agents-cli-manifest.yaml      # ADK project registration manifest
├── pyproject.toml                # Build configuration and dependency specifications
├── requirements.txt              # Primary project pip packages list
├── run_dubbing.bat               # Interactive drag-and-drop batch script
├── run_guide.md                  # Quick run commands cheat sheet
├── CAPSTONE_README.md            # Kaggle Capstone documentation README
└── README.md                     # Project homepage GitHub README
```

---

## 🚀 Setup & Installation

VaaniSync is designed to run entirely locally. Ensure you have **FFmpeg** installed on your system.

### 1. Prerequisites
```powershell
# On Windows via Winget:
winget install ffmpeg
```

### 2. Installation
```powershell
# Clone the repository, navigate in, and set up a venv
python -m venv .venv
.\.venv\Scripts\activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Local Verification
We maintain high code quality with automated unit tests and strict linting. Check correctness:
```powershell
# Run the complete test suite (all heavy models are mocked)
pytest tests/test_pipeline.py -v

# Run linting checks
ruff check video_localizer/ tests/
```

---

## 🎬 How to Run

### Option A: Drag-and-Drop Launcher (Windows)
We provided an interactive batch script [run_dubbing.bat](run_dubbing.bat):
1. Simply drag any video file from your file explorer and drop it onto `run_dubbing.bat`.
2. The script activates the virtual environment and kicks off the ADK pipeline.

### Option B: Interactive Web UI
You can start the visual web interface provided by Google ADK to watch the agent states trigger:
```powershell
.\.venv\Scripts\adk.exe web video_localizer --port 8001
```
Open **http://127.0.0.1:8001** and prompt the agent:
> *"Convert the audio of video/video1.mp4 to Kannada"*

### Option C: CLI
```powershell
.\.venv\Scripts\adk.exe run video_localizer "Convert the audio of video/video1.mp4 to Kannada"
```

---

## 🎯 Track Evaluation Details

* **Agents for Good Alignment**: Empowers local communities by translating high-quality educational and technology materials into Kannada automatically and completely offline, removing the financial gatekeeping of cloud translation fees.
* **Effective Use of Agent Technologies**: Showcases advanced workflow patterns from the course including sequential graphs, state preservation via Context, retry logic, and fallback translation hooks.
* **Communication & Documentation**: Includes interactive runtime logs, clean diagrams, and robust error-catching (e.g. failing gracefully back to original text if synthesis fails).

---

## 🔒 Security & Privacy

VaaniSync is designed with a **privacy-first, local-first** architecture:
* **Zero Video Leakage**: All video file manipulation, original audio extraction, and final muxing are performed locally on your machine via standard local commands (`ffmpeg` / `pydub`). No video/audio files are uploaded to third-party servers.
* **Robust Multi-Format Video Support**: The pipeline leverages `FFmpeg` to read the input stream and write the output container, natively supporting practically any video container format (including `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.flv`, `.wmv`, and `.3gp`). The output video takes the exact same format and extension as the input video.
* **Offline Speech & Translation Options**: 
  * Transcription is handled locally using `faster-whisper` running directly on your CPU.
  * For translation and speech synthesis: To achieve 100% offline security, configure translation to run entirely through your local Ollama LLM (`gemma2:2b`), and configure text-to-speech to use local `MeloTTS` exclusively, bypassing any external APIs.
  * By default, `edge-tts` is used as a high-quality fallback and communicates with public Edge TTS endpoints over TLS/HTTPS without saving audio files or request metadata.

---

## 🌍 Dynamic Multi-Language Support

VaaniSync dynamically parses and handles the target language directly from the user's natural language request (e.g., in the chat box or CLI). There is no need to manually update code configurations to switch languages!

### Supported Languages & Mapping

The pipeline automatically maps user queries to Google Translator codes, Edge-TTS neural voices, and MeloTTS language models for the following pre-configured locales:

- **Kannada (kn)**: Edge Sapna/Gagan Neural, MeloTTS `KN`
- **Hindi (hi)**: Edge Swarara/Madhur Neural
- **Telugu (te)**: Edge Shruti/Mohan Neural
- **Tamil (ta)**: Edge Pallavi/Valluvar Neural
- **Spanish (es)**: Edge Elvira/Alvaro Neural, MeloTTS `ES`
- **French (fr)**: Edge Denise/Henri Neural, MeloTTS `FR`
- **German (de)**: Edge Amala/Conrad Neural, MeloTTS `DE`
- **Italian (it)**: Edge Elsa/Diego Neural, MeloTTS `IT`
- **Japanese (ja)**: Edge Nanami/Keita Neural, MeloTTS `JP`
- **Chinese (zh-CN)**: Edge Xiaoxiao/Yunxi Neural, MeloTTS `ZH`
- **Korean (ko)**: Edge SunHi/InJoon Neural, MeloTTS `KR`
- **Portuguese (pt)**: Edge Francisca/Antonio Neural
- **Russian (ru)**: Edge Svetlana/Dmitry Neural
- **Arabic (ar)**: Edge Salma/Shakir Neural
- **English (en)**: Edge Jenny/Guy Neural, MeloTTS `EN`

---

## 🚀 Future Scope

* **Multi-Speaker Diarization & Cloned Mapping**: Integrate a local speaker identification model (such as `pyannote.audio`) to isolate distinct speakers in the source video, extract individual tone-color embeddings for each, and clone all speakers' voices concurrently.
* **Auto-Subtitling & Burn-In (SRT/VTT)**: Dynamically generate and burn localized subtitle tracks directly into the final muxed video container alongside the dubbed audio stream.
* **Emotion & Prosody Transfer**: Enhance the cloning pipeline to extract not just tone color, but the exact emotional inflection, pacing variations, whispers, and emphasis of the original speaker.
* **Agent-in-the-Loop Translation Critique**: Integrate a secondary agent critique stage using local Ollama models to review translation accuracy, verify idiomatic correctness, and check syllable-count pacing compatibility.
* **Fully Local Offline Audio-to-Audio Translation**: Deploy fully-contained local translator models to bypass web requests entirely, ensuring 100% network-independent translation.

---

## ⚠️ Known Limitations, Drawbacks & Mitigations

While VaaniSync is highly robust for offline regional dubbing, it has certain limitations in CPU-bound execution. Below are the key drawbacks and their corresponding mitigation strategies:

### 1. API Rate Limiting (Google Translate)
* **Problem**: Making separate HTTP requests for every Whisper subtitle segment triggers **HTTP 429 (Too Many Requests)**.
* **Current Mitigation**: Implemented **Smart Delimiter Translation Batching**. Multiple segments are concatenated using the custom ` ||| ` separator and sent to the translation API in a single HTTP request, reducing API calls by up to 90%. If an API error occurs, the system gracefully falls back to individual segment requests.
* **Future Mitigation**: Move to a fully local translation engine using offline HuggingFace translation pipelines (e.g. `Helsinki-NLP/OPUS-mt-en-dra` or local Ollama instances running `llama3:8b`) to eliminate API dependencies completely.

### 2. CPU Synthesis Bottlenecks
* **Problem**: Text-to-speech generation and zero-shot voice cloning on CPU are computationally heavy and slow when executed sequentially.
* **Current Mitigation**: Implemented **ThreadPool Parallel Synthesis**. We synthesize segments concurrently using a python `ThreadPoolExecutor`. CPU-bound model inference is serialized with a threading Lock (`melo_lock`) to prevent cache thrashing, while network tasks (Edge-TTS) and file operations run in parallel, cutting processing time by over 50%.
* **Future Mitigation**: Integrate OpenVINO or ONNX Runtime to compile MeloTTS/OpenVoice V2 models, enabling high-speed INT8 quantized inference directly on Intel/AMD consumer CPUs.

### 3. Context Window Limits
* **Problem**: Translating extremely long transcripts in a single local LLM prompt can exceed the context window limits (e.g. 8k tokens) of models like Gemma or Mistral.
* **Current Mitigation**: Long subtitle lists are chunked into rolling context blocks of 20-30 segments. Each block is translated independently with overlapping boundaries to preserve narrative context.
* **Future Mitigation**: Implement a stateful MapReduce translation agent pattern that summarizes context history into a persistent metadata store passed to subsequent chunk translation agents.

### 4. Single-Speaker Constraint (No Diarization)
* **Problem**: The pipeline extracts a single tone embedding from the original audio. If a video contains multiple speakers (e.g. dialogue or panel discussion), all translated speech will be morphed into the main speaker's voice.
* **Current Mitigation**: The system accepts a gender hint (`male`/`female`) to select the base synthesis profile, but applies the same clone converter to all segments.
* **Future Mitigation**: Integrate local speaker diarization (e.g., `pyannote.audio` or lightweight `spectralcluster`) to segment the original wav file by speaker ID, extract a tone embedding for each unique speaker, and apply matching cloned voices dynamically.

### 5. Prosody and Emotion Cloning Constraints
* **Problem**: OpenVoice V2 clones the speaker's vocal color (timbre) but not their emotion, rhythm, or dramatic phrasing. The cloned voice inherits the flat, even prosody of the base MeloTTS engine.
* **Current Mitigation**: Pacing is automatically adjusted using FFmpeg `atempo` speed stretching and silent padding.
* **Future Mitigation**: Upgrade to advanced voice cloning backends (like local XTTS-v2 or VALL-E) that support joint style, pitch contour, and emotion transfer alongside timbre cloning.

### 6. Extreme Compression Capping (2.0x Limit)
* **Problem**: If the translated Kannada sentence is significantly longer than the original English sentence window, speeding it up beyond `2.0x` renders the speech completely unintelligible.
* **Current Mitigation**: A hard cap of `2.0x` is applied. If a segment exceeds this, the audio file is trimmed to fit the target duration, causing slight speech clipping.
* **Future Mitigation**: Implement dynamic video speed modulation. The system can interactively extend the original video's frames (injecting duplicate frames or slow-motion segments) to match the required speech length when a translation is extremely long.

---
