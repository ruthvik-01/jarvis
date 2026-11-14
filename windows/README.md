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

Create a `.env` file in the repository root or `windows/.env` (see `../.env.example`) containing:

- `RIVER_AI_API`
- `ASSEMBLYAI_API_KEY`

Settings and UI preferences are saved locally to `windows/user_settings.json` (git-ignored).

## Launch & Verification

```powershell
# Launch app via native executable
.\Jarvis.exe

# Run automated 6-suite verification
.\Jarvis.exe --selftest

# Export UI/overlay verification snapshots
python .\jarvis.py --snapshot .\snapshots
```
