# WhisperDeck

Real-time system audio transcription using Whisper.

## Description

WhisperDeck captures system audio (loopback) and transcribes it in real-time using OpenAI's Whisper model. Perfect for transcribing audio from your computer, including from applications, browser tabs, or any system audio source.

## Prerequisites

- Python 3.8 or higher
- Virtual audio cable software (OS-specific):
  - Windows: VB-CABLE (If using Voicemeeter, route the audio you want to transcribe to Output B2)
  - macOS: BlackHole
  - Linux: PulseAudio

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/WhisperDeck.git
cd WhisperDeck
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and configure your settings:
```env
MODEL_ID=openai/whisper-base
LOG_LEVEL=INFO
CHUNK_DURATION_S=3
SILENCE_THRESHOLD_RMS=0.005
```

## Usage

Run the transcription script:
```bash
python main.py
```

## Configuration

Environment variables in `.env`:
- `MODEL_ID`: Whisper model to use (base, small, medium, large)
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `CHUNK_DURATION_S`: Duration of audio chunks to process
- `SILENCE_THRESHOLD_RMS`: Threshold for silence detection

## Troubleshooting

1. No audio device found:
   - Ensure virtual audio cable is properly installed
   - Check system sound settings

2. Model loading errors:
   - Verify internet connection for first-time model download
   - Check available disk space

3. Performance issues:
   - Try a smaller model (base or small)
   - Increase chunk duration
   - Ensure GPU drivers are up to date if using GPU