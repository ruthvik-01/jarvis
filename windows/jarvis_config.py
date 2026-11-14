"""
Settings and API-key storage for the Jarvis Windows replica.

- Regular preferences live in `windows/user_settings.json` (git-ignored).
- API keys live in the repo-root `.env` file, which is where the user already keeps them.
  Keys are never written into settings JSON, logs, or source files.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal

WINDOWS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIRECTORY = os.path.dirname(WINDOWS_DIRECTORY)
USER_SETTINGS_FILE_PATH = os.path.join(WINDOWS_DIRECTORY, "user_settings.json")
ENV_FILE_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, ".env")

# Names of the secrets the app understands, as they appear in .env.
SECRET_RIVER_API_KEY = "RIVER_AI_API"
SECRET_ASSEMBLYAI_API_KEY = "ASSEMBLYAI_API_KEY"
SECRET_ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"  # Legacy fallback constant kept only for backward compatibility
ALL_SECRET_NAMES = [SECRET_RIVER_API_KEY, SECRET_ASSEMBLYAI_API_KEY]

# AssemblyAI Neural TTS voices (`AssemblyAI-TTS-1`, 24 kHz mono WAV via https://tts.us.assemblyai.com/v1/speech).
ASSEMBLYAI_TTS_VOICES: List[Dict[str, str]] = [
    {"voice_id": "aai-kyla", "name": "Kyla", "description": "Bright, warm, conversational (recommended)"},
    {"voice_id": "aai-sara", "name": "Sara", "description": "Warm, clear, friendly"},
    {"voice_id": "aai-audrey", "name": "Audrey", "description": "Calm, articulate tutor"},
    {"voice_id": "aai-hailey", "name": "Hailey", "description": "Upbeat, expressive"},
    {"voice_id": "aai-carrie", "name": "Carrie", "description": "Warm British"},
    {"voice_id": "aai-nancy", "name": "Nancy", "description": "Clear British"},
    {"voice_id": "aai-yasmin", "name": "Yasmin", "description": "Smooth British"},
    {"voice_id": "aai-peter", "name": "Peter", "description": "Friendly American male"},
    {"voice_id": "aai-titus", "name": "Titus", "description": "Deep, steady male"},
    {"voice_id": "aai-buford", "name": "Buford", "description": "Warm British storyteller"},
    {"voice_id": "aai-steven", "name": "Steven", "description": "Upbeat Australian"},
    {"voice_id": "aai-troy", "name": "Troy", "description": "Expressive Scottish"},
    {"voice_id": "windows-sapi", "name": "Windows voice", "description": "Offline, built into Windows"},
]
ELEVENLABS_PREMADE_VOICES = ASSEMBLYAI_TTS_VOICES

SCREEN_DRAWING_MODE_CHOICES: List[Dict[str, str]] = [
    {"mode_id": "precise_lines", "name": "Precise Lines Only (Clean text underlines, no boxes)"},
    {"mode_id": "lines_and_card", "name": "Precise Lines + Margin Summary Card"},
    {"mode_id": "full_sketch", "name": "Full Sketch (Lines, Rings & Summary Card)"},
    {"mode_id": "cursor_only", "name": "Cursor Pointing Only (No screen ink)"},
]

DRAWING_SPEED_CHOICES: List[Dict[str, str]] = [
    {"speed_id": "slow_smooth", "name": "Slow & Smooth (0.95s guided stroke)"},
    {"speed_id": "normal", "name": "Standard (0.55s stroke)"},
]

# Vision-capable chat models on the user's River AI account (verified 15-11-2026:
# DeepSeek-V4.1-Flash reads images; Kimi/GLM returned empty replies for images).
CHAT_MODEL_CHOICES: List[Dict[str, str]] = [
    {"model_id": "deepseek-ai/DeepSeek-V4.1-Flash", "name": "DeepSeek V4.1 Flash (fast, sees screen)"},
    {"model_id": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "name": "DeepSeek V4.1 Flash 1017"},
]

DICTATION_LANGUAGE_CHOICES: List[Dict[str, str]] = [
    {"code": "en", "name": "English"},
    {"code": "multi", "name": "Auto-detect (multilingual)"},
]


def _default_display_name() -> str:
    for environment_variable in ("JARVIS_DISPLAY_NAME", "USERNAME"):
        candidate_name = os.environ.get(environment_variable, "").strip()
        if candidate_name:
            return candidate_name.split()[0].upper() if environment_variable == "USERNAME" else candidate_name
    return ""


DEFAULT_SETTINGS: Dict[str, Any] = {
    "user_display_name": _default_display_name(),
    # Cursor & Screen Drawing
    "cursor_color_hex": "#3380FF",
    "hide_floating_cursor": False,
    "shake_to_detach_enabled": False,
    "screen_highlight_enabled": True,
    "screen_drawing_mode": "precise_lines",
    "drawing_speed": "slow_smooth",
    # Voice (AssemblyAI-TTS-1)
    "voice_id": "aai-kyla",
    "speech_speed": 1.0,
    "captions_enabled": False,
    "typed_turns_respond_with_voice": False,
    # Sounds
    "earcons_enabled": True,
    "drawing_sounds_enabled": True,
    # Microphone (None = Windows default input device)
    "microphone_device_index": None,
    # Dictation
    "dictation_language": "en",
    "dictation_custom_words": ["Jarvis", "Jarvis"],
    # Brain & Agent
    "chat_model": "deepseek-ai/DeepSeek-V4.1-Flash",
    "remember_style_notes": [],
    "agent_actions_enabled": True,
    # General
    "launch_at_login": False,
    "show_in_taskbar": False,
    "show_in_screen_recordings": True,
    "has_completed_onboarding": False,
}


def read_env_file(env_file_path: str = ENV_FILE_PATH) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(env_file_path):
        return values
    try:
        with open(env_file_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                stripped_line = raw_line.strip()
                if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
                    continue
                key_name, key_value = stripped_line.split("=", 1)
                values[key_name.strip()] = key_value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def write_env_value(key_name: str, key_value: str, env_file_path: str = ENV_FILE_PATH) -> None:
    """Updates (or appends) one KEY=value line in .env while keeping all other lines intact."""
    existing_lines: List[str] = []
    if os.path.exists(env_file_path):
        with open(env_file_path, "r", encoding="utf-8") as env_file:
            existing_lines = env_file.read().splitlines()
    replaced_existing_line = False
    for line_index, existing_line in enumerate(existing_lines):
        if existing_line.split("=", 1)[0].strip() == key_name:
            existing_lines[line_index] = f"{key_name}={key_value}"
            replaced_existing_line = True
    if not replaced_existing_line:
        existing_lines.append(f"{key_name}={key_value}")
    with open(env_file_path, "w", encoding="utf-8") as env_file:
        env_file.write("\n".join(existing_lines) + "\n")


class SettingsStore(QObject):
    """
    Single source of truth for preferences and API keys.
    UI code calls get()/set(); anything that cares listens to `settingChanged(key, value)`.
    Thread-safe for reads from worker threads.
    """

    settingChanged = Signal(str, object)
    secretChanged = Signal(str)

    def __init__(self, settings_file_path: str = USER_SETTINGS_FILE_PATH, env_file_path: str = ENV_FILE_PATH) -> None:
        super().__init__()
        self._settings_file_path = settings_file_path
        self._env_file_path = env_file_path
        self._lock = threading.RLock()
        self._values: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self._secrets: Dict[str, str] = {}
        self._load()

    # -- preferences -------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self._settings_file_path):
            try:
                with open(self._settings_file_path, "r", encoding="utf-8") as settings_file:
                    stored_values = json.load(settings_file)
                if isinstance(stored_values, dict):
                    for key_name, stored_value in stored_values.items():
                        if key_name in DEFAULT_SETTINGS:
                            self._values[key_name] = stored_value
            except (OSError, ValueError):
                pass
        # Migrate any old non-AssemblyAI voice ID to AssemblyAI default ("aai-kyla").
        known_voice_ids = {voice["voice_id"] for voice in ASSEMBLYAI_TTS_VOICES}
        if self._values.get("voice_id") not in known_voice_ids:
            self._values["voice_id"] = DEFAULT_SETTINGS["voice_id"]
        self._secrets = read_env_file(self._env_file_path)

    def _save(self) -> None:
        temporary_path = self._settings_file_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as settings_file:
            json.dump(self._values, settings_file, indent=2)
        os.replace(temporary_path, self._settings_file_path)

    def get(self, key_name: str, fallback: Any = None) -> Any:
        with self._lock:
            if key_name in self._values:
                return self._values[key_name]
            return DEFAULT_SETTINGS.get(key_name, fallback)

    def set(self, key_name: str, new_value: Any) -> None:
        with self._lock:
            if self._values.get(key_name) == new_value:
                return
            self._values[key_name] = new_value
            try:
                self._save()
            except OSError:
                pass
        self.settingChanged.emit(key_name, new_value)

    def all_values(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._values)

    # -- secrets -------------------------------------------------------------

    def get_secret(self, secret_name: str) -> str:
        with self._lock:
            environment_override = os.environ.get(secret_name, "").strip()
            return self._secrets.get(secret_name, "") or environment_override

    def set_secret(self, secret_name: str, secret_value: str) -> None:
        secret_value = secret_value.strip()
        with self._lock:
            if self._secrets.get(secret_name, "") == secret_value:
                return
            self._secrets[secret_name] = secret_value
            write_env_value(secret_name, secret_value, self._env_file_path)
        self.secretChanged.emit(secret_name)

    def has_secret(self, secret_name: str) -> bool:
        return bool(self.get_secret(secret_name))
