# Jarvis — Windows Native Companion

Native 60 FPS PySide6 (Qt 6) desktop screen companion for Windows, compiled launcher `Jarvis.exe`, and `J.A.R.V.I.S` branding.

## Requirements

- Windows 10 / 11 (x64)
- Python 3.11+ with packages:
  - `PySide6`
  - `sounddevice`
  - `numpy`
  - `websocket-client`
  - `requests`
  - `pillow`
  - `mss`
  - `pygame`
  - `python-dotenv`
  - `pywin32`

## Configuration

Create a `.env` file in the repository root or `windows/.env` (see `../.env.example`) containing any chat/vision provider credentials plus voice:

- **Universal / Any API Endpoint**: `CUSTOM_API_KEY` (or `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `TOGETHER_API_KEY`, `MISTRAL_API_KEY`, `XAI_API_KEY`, `AZURE_OPENAI_API_KEY`, etc.), optional `CUSTOM_API_BASE_URL` / `OPENAI_BASE_URL` / `OLLAMA_BASE_URL` (e.g. `http://localhost:11434/v1` for keyless local Ollama), and optional `CUSTOM_MODEL`
- **Google Gemini & River AI**: `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) and/or `RIVER_AI_API`
- **Voice STT & TTS**: `ASSEMBLYAI_API_KEY` for AssemblyAI Streaming STT and Neural Voice TTS

You can also enter or override any key, base URL, and custom model ID directly inside **Settings -> Studio & Keys** in the app. Settings and UI preferences are saved locally to `windows/user_settings.json` (git-ignored).

## Launch & Verification

```powershell
# Launch app via native executable
.\Jarvis.exe

# Run automated 6-suite verification
.\Jarvis.exe --selftest

# Export UI/overlay verification snapshots
python .\jarvis.py --snapshot .\snapshots
```
