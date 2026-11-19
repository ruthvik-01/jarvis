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
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

WINDOWS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIRECTORY = os.path.dirname(WINDOWS_DIRECTORY)
USER_SETTINGS_FILE_PATH = os.path.join(WINDOWS_DIRECTORY, "user_settings.json")
ENV_FILE_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, ".env")

# Names of the secrets the app understands, as they appear in .env.
SECRET_CUSTOM_API_KEY = "CUSTOM_API_KEY"
CUSTOM_API_KEY_ALIASES = (
    "CUSTOM_API_KEY",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY",
    "MISTRAL_API_KEY",
    "XAI_API_KEY",
    "FIREWORKS_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "PERPLEXITY_API_KEY",
    "NVIDIA_API_KEY",
    "OPENCODE_API_KEY",
    "API_KEY",
)
ENV_ENDPOINT_URL_ALIASES = (
    "CUSTOM_API_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "LLM_BASE_URL",
    "LLM_API_BASE",
    "API_BASE_URL",
    "API_ENDPOINT",
    "OPENROUTER_BASE_URL",
    "OLLAMA_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "GEMINI_BASE_URL",
    "BASE_URL",
)
ENV_CUSTOM_MODEL_ALIASES = (
    "CUSTOM_MODEL",
    "LLM_MODEL",
    "OPENAI_MODEL",
    "MODEL_NAME",
    "MODEL_ID",
)
SECRET_GOOGLE_API_KEY = "GOOGLE_API_KEY"
GOOGLE_API_KEY_ALIASES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_AI_API_KEY")
SECRET_RIVER_API_KEY = "RIVER_AI_API"
SECRET_ASSEMBLYAI_API_KEY = "ASSEMBLYAI_API_KEY"
SECRET_FISH_API_KEY = "FISH_API_KEY"
SECRET_OPENCODE_API_KEY = "OPENCODE_API_KEY"
SECRET_ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"  # Legacy fallback constant kept only for backward compatibility
ALL_SECRET_NAMES = [
    SECRET_CUSTOM_API_KEY,
    SECRET_GOOGLE_API_KEY,
    SECRET_RIVER_API_KEY,
    SECRET_ASSEMBLYAI_API_KEY,
    SECRET_FISH_API_KEY,
]
DEFAULT_GOOGLE_CHAT_MODEL = "gemini-3.8-flash"

KNOWN_PROVIDER_ENDPOINTS: Dict[str, str] = {
    "OPENAI_API_KEY": "https://api.openai.com/v1/chat/completions",
    "OPENROUTER_API_KEY": "https://openrouter.ai/api/v1/chat/completions",
    "GROQ_API_KEY": "https://api.groq.com/openai/v1/chat/completions",
    "DEEPSEEK_API_KEY": "https://api.deepseek.com/v1/chat/completions",
    "ANTHROPIC_API_KEY": "https://api.anthropic.com/v1/messages",
    "TOGETHER_API_KEY": "https://api.together.xyz/v1/chat/completions",
    "MISTRAL_API_KEY": "https://api.mistral.ai/v1/chat/completions",
    "XAI_API_KEY": "https://api.x.ai/v1/chat/completions",
    "FIREWORKS_API_KEY": "https://api.fireworks.ai/inference/v1/chat/completions",
    "PERPLEXITY_API_KEY": "https://api.perplexity.ai/chat/completions",
    "NVIDIA_API_KEY": "https://integrate.api.nvidia.com/v1/chat/completions",
    "OPENCODE_API_KEY": "https://opencode.ai/zen/v1/chat/completions",
    "RIVER_AI_API": "https://api.river.ai/v1/chat/completions",
    "GOOGLE_API_KEY": "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:streamGenerateContent?alt=sse",
    "GEMINI_API_KEY": "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:streamGenerateContent?alt=sse",
    "GOOGLE_AI_API_KEY": "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:streamGenerateContent?alt=sse",
}

DEFAULT_MODEL_FOR_KEY_ALIAS: Dict[str, str] = {
    "OPENAI_API_KEY": "gpt-4o",
    "OPENROUTER_API_KEY": "openai/gpt-4o",
    "GROQ_API_KEY": "meta-llama/llama-4-scout-17b-16e-instruct",
    "DEEPSEEK_API_KEY": "deepseek-chat",
    "ANTHROPIC_API_KEY": "claude-sonnet-4-20250514",
    "TOGETHER_API_KEY": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "MISTRAL_API_KEY": "pixtral-large-latest",
    "XAI_API_KEY": "grok-2-vision-1212",
    "FIREWORKS_API_KEY": "accounts/fireworks/models/llama4-maverick-instruct-basic",
    "PERPLEXITY_API_KEY": "sonar",
    "NVIDIA_API_KEY": "meta/llama-3.2-90b-vision-instruct",
}

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

# Vision-capable chat models across Google Gemini, OpenAI, Anthropic, River AI, and custom endpoints.
CHAT_MODEL_CHOICES: List[Dict[str, str]] = [
    {"model_id": "gemini-4-hydrogen", "name": "Google Gemini 4 Hydrogen (vision & reasoning)", "provider": "google"},
    {"model_id": "gemini-3.8-flash", "name": "Google Gemini 3.8 Flash (fast, sees screen)", "provider": "google"},
    {"model_id": "gemini-2.5-flash", "name": "Google Gemini 2.5 Flash (sees screen)", "provider": "google"},
    {"model_id": "gemini-2.5-pro", "name": "Google Gemini 2.5 Pro (reasoning + vision)", "provider": "google"},
    {"model_id": "gpt-4o", "name": "OpenAI GPT-4o (vision & agent)", "provider": "openai"},
    {"model_id": "gpt-4.1-mini", "name": "OpenAI GPT-4.1 Mini (fast vision)", "provider": "openai"},
    {"model_id": "claude-sonnet-4-20250514", "name": "Anthropic Claude Sonnet 4 (vision)", "provider": "anthropic"},
    {"model_id": "claude-3-7-sonnet-latest", "name": "Anthropic Claude 3.7 Sonnet (vision)", "provider": "anthropic"},
    {"model_id": "deepseek-chat", "name": "DeepSeek Chat (universal / Direct API)", "provider": "deepseek"},
    {"model_id": "deepseek-ai/DeepSeek-V4.1-Flash", "name": "DeepSeek V4.1 Flash — River AI (sees screen)", "provider": "river"},
    {"model_id": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "name": "DeepSeek V4.1 Flash 1017 — River AI", "provider": "river"},
]


def is_google_gemini_model(model_id: str) -> bool:
    """Returns True when `model_id` targets Google Gemini directly."""
    normalized_model = (model_id or "").strip().lower()
    return normalized_model.startswith(("gemini", "google/", "models/gemini"))


def is_anthropic_model(model_id: str) -> bool:
    """Returns True when `model_id` targets Anthropic Claude Messages API."""
    normalized_model = (model_id or "").strip().lower()
    return normalized_model.startswith(("claude", "anthropic/")) or ("claude-" in normalized_model)


def detect_wire_protocol(
    endpoint_url: str,
    model_id: str = "",
    key_source_name: str = "",
) -> str:
    """Infers wire protocol ('google', 'anthropic', or 'openai') from URL, key alias, and model ID."""
    url_lower = (endpoint_url or "").strip().lower()
    model_lower = (model_id or "").strip().lower()
    alias_upper = (key_source_name or "").strip().upper()

    openai_proxy_hosts = (
        "openrouter.ai",
        "api.river.ai",
        "together.xyz",
        "fireworks.ai",
        "groq.com",
        "api.openai.com",
        "openai.azure.com",
        "api.deepseek.com",
        "api.mistral.ai",
        "api.x.ai",
        "api.perplexity.ai",
        "integrate.api.nvidia.com",
        "/v1beta/openai",
    )
    if any(proxy_host in url_lower for proxy_host in openai_proxy_hosts):
        return "openai"

    if (
        ("generativelanguage.googleapis.com" in url_lower)
        or (":generatecontent" in url_lower)
        or (":streamgeneratecontent" in url_lower)
    ):
        return "google"

    if ("api.anthropic.com" in url_lower) or url_lower.endswith("/v1/messages") or ("/messages" in url_lower):
        return "anthropic"

    if url_lower.endswith("/chat/completions") or ("/chat/completions" in url_lower):
        return "openai"

    if alias_upper == "ANTHROPIC_API_KEY":
        return "anthropic"
    if alias_upper in GOOGLE_API_KEY_ALIASES:
        return "google"

    if not url_lower:
        if is_anthropic_model(model_lower):
            return "anthropic"
        if is_google_gemini_model(model_lower):
            return "google"
    else:
        if is_anthropic_model(model_lower):
            return "anthropic"
        if "gemini" in url_lower and is_google_gemini_model(model_lower):
            return "google"

    return "openai"


def infer_provider_name(
    endpoint_url: str = "",
    model_id: str = "",
    key_source_name: str = "",
) -> str:
    """Returns a friendly provider label inferred from URL, key alias, or model ID."""
    url_lower = (endpoint_url or "").strip().lower()
    alias_upper = (key_source_name or "").strip().upper()
    if "generativelanguage.googleapis.com" in url_lower or alias_upper in GOOGLE_API_KEY_ALIASES:
        return "Google Gemini"
    if "api.anthropic.com" in url_lower or alias_upper == "ANTHROPIC_API_KEY":
        return "Anthropic"
    if "api.river.ai" in url_lower or alias_upper == "RIVER_AI_API":
        return "River AI"
    if "openrouter.ai" in url_lower or alias_upper == "OPENROUTER_API_KEY":
        return "OpenRouter"
    if "api.openai.com" in url_lower or alias_upper == "OPENAI_API_KEY":
        return "OpenAI"
    if "openai.azure.com" in url_lower or alias_upper == "AZURE_OPENAI_API_KEY":
        return "Azure OpenAI"
    if "groq.com" in url_lower or alias_upper == "GROQ_API_KEY":
        return "Groq"
    if "api.deepseek.com" in url_lower or alias_upper == "DEEPSEEK_API_KEY":
        return "DeepSeek"
    if "together.xyz" in url_lower or alias_upper == "TOGETHER_API_KEY":
        return "Together AI"
    if "mistral.ai" in url_lower or alias_upper == "MISTRAL_API_KEY":
        return "Mistral"
    if "api.x.ai" in url_lower or alias_upper == "XAI_API_KEY":
        return "xAI"
    if "fireworks.ai" in url_lower or alias_upper == "FIREWORKS_API_KEY":
        return "Fireworks"
    if "perplexity.ai" in url_lower or alias_upper == "PERPLEXITY_API_KEY":
        return "Perplexity"
    if "nvidia.com" in url_lower or alias_upper == "NVIDIA_API_KEY":
        return "NVIDIA NIM"
    if ("11434" in url_lower) or ("ollama" in url_lower):
        return "Ollama"
    if "1234" in url_lower:
        return "LM Studio"
    if url_lower:
        return "Custom API"
    if is_google_gemini_model(model_id):
        return "Google Gemini"
    if is_anthropic_model(model_id):
        return "Anthropic"
    return "Custom API"


def normalize_endpoint_url(
    raw_url: str,
    wire_protocol: str = "openai",
    model_id: str = "",
) -> str:
    """Normalizes a base URL or full endpoint URL into an executable chat/vision endpoint."""
    cleaned_url = (raw_url or "").strip().rstrip("/")
    if not cleaned_url:
        return ""
    clean_model = (model_id or "gpt-4o").strip()
    if clean_model.startswith("models/"):
        clean_model = clean_model[len("models/"):]
    url_lower = cleaned_url.lower()

    if wire_protocol == "google":
        if "{model_id}" in cleaned_url:
            cleaned_url = cleaned_url.replace("{model_id}", clean_model)
            url_lower = cleaned_url.lower()
        if (":streamgeneratecontent" in url_lower) or (":generatecontent" in url_lower):
            if (":streamgeneratecontent" in url_lower) and ("alt=" not in url_lower):
                separator = "&" if ("?" in cleaned_url) else "?"
                return f"{cleaned_url}{separator}alt=sse"
            return cleaned_url
        if url_lower.endswith("/models"):
            return f"{cleaned_url}/{clean_model}:streamGenerateContent?alt=sse"
        if url_lower.endswith(("/v1beta", "/v1")):
            return f"{cleaned_url}/models/{clean_model}:streamGenerateContent?alt=sse"
        return f"{cleaned_url}/v1beta/models/{clean_model}:streamGenerateContent?alt=sse"

    if wire_protocol == "anthropic":
        if url_lower.endswith("/messages"):
            return cleaned_url
        if url_lower.endswith("/v1"):
            return f"{cleaned_url}/messages"
        return f"{cleaned_url}/v1/messages"

    # OpenAI-compatible wire protocol
    if ("openai.azure.com" in url_lower) or ("api-version=" in url_lower):
        if "/chat/completions" in url_lower:
            return cleaned_url
        if "/deployments/" not in url_lower:
            return f"{cleaned_url}/openai/deployments/{clean_model}/chat/completions?api-version=2024-10-21"
        return f"{cleaned_url}/chat/completions?api-version=2024-10-21"

    if url_lower.endswith(("/chat/completions", "/completions", "/responses")):
        return cleaned_url

    if ("api.groq.com" in url_lower) and url_lower.endswith("/openai"):
        return f"{cleaned_url}/v1/chat/completions"

    if url_lower.endswith(("/v1", "/v2", "/v3", "/v4", "/api/v1", "/inference/v1", "/openai", "/v1beta/openai", "/compatible-mode/v1")):
        return f"{cleaned_url}/chat/completions"

    scheme_split = cleaned_url.split("://", 1)
    host_and_path = scheme_split[1] if len(scheme_split) == 2 else cleaned_url
    has_non_empty_path = ("/" in host_and_path) and bool(host_and_path.split("/", 1)[1].strip("/"))
    if not has_non_empty_path:
        if "api.perplexity.ai" in url_lower:
            return f"{cleaned_url}/chat/completions"
        return f"{cleaned_url}/v1/chat/completions"

    return cleaned_url

DICTATION_LANGUAGE_CHOICES: List[Dict[str, str]] = [
    {"code": "en", "name": "English"},
    {"code": "ar", "name": "Arabic (العربية)"},
    {"code": "he", "name": "Hebrew (עברית)"},
    {"code": "es", "name": "Spanish"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "hi", "name": "Hindi"},
    {"code": "ja", "name": "Japanese"},
    {"code": "ko", "name": "Korean"},
    {"code": "zh", "name": "Chinese"},
    {"code": "multi", "name": "Auto-detect (multilingual)"},
]

TTS_ENGINE_CHOICES: List[Dict[str, str]] = [
    {"engine_id": "assemblyai", "name": "AssemblyAI Neural TTS (default)"},
    {"engine_id": "gemini", "name": "Google Gemini TTS (30 voices)"},
    {"engine_id": "fish", "name": "Fish Audio Live TTS (emotion cues)"},
    {"engine_id": "voicestudio", "name": "VoiceStudio Local TTS"},
    {"engine_id": "windows-sapi", "name": "Windows Built-in Voice (offline)"},
]

GEMINI_TTS_MODELS: List[Dict[str, str]] = [
    {"model_id": "gemini-3.8-flash-lite-tts", "name": "Gemini 3.8 Flash Lite TTS ($0.0015 / 10s)"},
    {"model_id": "gemini-3.8-flash-tts", "name": "Gemini 3.8 Flash TTS ($0.00225 / 10s)"},
]

FISH_TTS_MODELS: List[Dict[str, str]] = [
    {"model_id": "drama-3", "name": "Fish Drama 3 (expressive emotion cues)"},
    {"model_id": "s2.1-pro", "name": "Fish S2.1 Pro (fast studio quality)"},
]

GEMINI_TTS_VOICES: List[Dict[str, str]] = [
    {"voice_id": "Zephyr", "style": "Bright"},
    {"voice_id": "Puck", "style": "Upbeat"},
    {"voice_id": "Charon", "style": "Informative"},
    {"voice_id": "Kore", "style": "Firm"},
    {"voice_id": "Fenrir", "style": "Excitable"},
    {"voice_id": "Leda", "style": "Youthful"},
    {"voice_id": "Orus", "style": "Firm"},
    {"voice_id": "Aoede", "style": "Breezy"},
    {"voice_id": "Callirrhoe", "style": "Easy-going"},
    {"voice_id": "Autonoe", "style": "Bright"},
    {"voice_id": "Enceladus", "style": "Breathy"},
    {"voice_id": "Iapetus", "style": "Clear"},
    {"voice_id": "Umbriel", "style": "Easy-going"},
    {"voice_id": "Algieba", "style": "Smooth"},
    {"voice_id": "Despina", "style": "Smooth"},
    {"voice_id": "Erinome", "style": "Clear"},
    {"voice_id": "Algenib", "style": "Gravelly"},
    {"voice_id": "Rasalgethi", "style": "Informative"},
    {"voice_id": "Laomedeia", "style": "Upbeat"},
    {"voice_id": "Achernar", "style": "Soft"},
    {"voice_id": "Alnilam", "style": "Firm"},
    {"voice_id": "Schedar", "style": "Even"},
    {"voice_id": "Gacrux", "style": "Mature"},
    {"voice_id": "Pulcherrima", "style": "Forward"},
    {"voice_id": "Achird", "style": "Friendly"},
    {"voice_id": "Zubenelgenubi", "style": "Casual"},
    {"voice_id": "Vindemiatrix", "style": "Gentle"},
    {"voice_id": "Sadachbia", "style": "Lively"},
    {"voice_id": "Sadaltager", "style": "Knowledgeable"},
    {"voice_id": "Sulafat", "style": "Warm"},
]

GEMINI_LOOK_VOICES: Dict[str, str] = {
    "jarvis": "Charon",
    "triangle": "Charon",
    "classic": "Puck",
    "glayos": "Erinome",
    "ultren": "Algenib",
    "anime": "Leda",
    "sarah": "Sulafat",
    "adrian": "Achird",
    "rita": "Despina",
}

COMPANION_LOOK_CHOICES: List[Dict[str, str]] = [
    {"look_id": "jarvis", "name": "JARVIS Sphere (3D LED globe)"},
    {"look_id": "triangle", "name": "Classic Blue Cursor (Arrow emblem)"},
    {"look_id": "classic", "name": "Roamie (Glowing pixel creature)"},
    {"look_id": "glayos", "name": "GlayOS (One-eyed lab AI)"},
    {"look_id": "ultren", "name": "Ultren (Chrome twin-optic machine)"},
    {"look_id": "anime", "name": "Anime Girl (Pixel companion)"},
    {"look_id": "sarah", "name": "Sarah (Pixel companion)"},
    {"look_id": "adrian", "name": "Adrian (Pixel companion)"},
    {"look_id": "rita", "name": "Rita X (Pixel secretary)"},
]

COMPANION_SIZE_CHOICES: List[Dict[str, str]] = [
    {"size_id": "default", "name": "Default"},
    {"size_id": "small", "name": "Small"},
    {"size_id": "cursor", "name": "Cursor (Compact reticle / pointer)"},
]

WHERE_I_APPEAR_CHOICES: List[Dict[str, str]] = [
    {"place_id": "cursor", "name": "Follow Cursor + Notch"},
    {"place_id": "capsule", "name": "Top Capsule Everywhere"},
    {"place_id": "notch", "name": "Top Notch Only"},
]

REASONING_EFFORT_CHOICES: List[Dict[str, str]] = [
    {"effort_id": "auto", "name": "Automatic"},
    {"effort_id": "off", "name": "Off"},
    {"effort_id": "low", "name": "Low"},
    {"effort_id": "medium", "name": "Medium"},
    {"effort_id": "high", "name": "High"},
    {"effort_id": "xhigh", "name": "Extra High"},
    {"effort_id": "max", "name": "Max"},
]


def _default_display_name() -> str:
    for environment_variable in ("JARVIS_DISPLAY_NAME", "USERNAME"):
        candidate_name = os.environ.get(environment_variable, "").strip()
        if candidate_name:
            return candidate_name.split()[0].upper() if environment_variable == "USERNAME" else candidate_name
    return ""


DEFAULT_SETTINGS: Dict[str, Any] = {
    "user_display_name": _default_display_name(),
    # Cursor & Screen Drawing & Appearance
    "cursor_color_hex": "#3380FF",
    "hide_floating_cursor": False,
    "shake_to_detach_enabled": False,
    "screen_highlight_enabled": True,
    "screen_drawing_mode": "lines_and_card",
    "drawing_speed": "slow_smooth",
    "companion_look": "jarvis",
    "companion_size": "default",
    "where_i_appear": "cursor",
    "quiet_when_hidden": True,
    # Voice (Multi-Engine TTS: AssemblyAI / Gemini / Fish / VoiceStudio / Windows SAPI)
    "tts_engine": "assemblyai",
    "voice_id": "aai-kyla",
    "gemini_tts_model": "gemini-3.8-flash-lite-tts",
    "gemini_tts_voice": "Charon",
    "fish_tts_model": "drama-3",
    "fish_tts_voice": "e13fa398a7f445a685316a3de6089ce7",
    "voicestudio_url": "http://127.0.0.1:8765",
    "voicestudio_voice": "",
    "speech_speed": 1.0,
    "captions_enabled": False,
    "typed_turns_respond_with_voice": False,
    "voice_cues_enabled": True,
    "math_speech_normalization": True,
    # Sounds
    "earcons_enabled": True,
    "drawing_sounds_enabled": True,
    # Microphone & Listening
    "microphone_device_index": None,
    "wake_word_enabled": False,
    "wake_word_name": "Jarvis",
    "keep_listening_enabled": False,
    "keep_listening_seconds": 8,
    "speak_language": "en",
    # Dictation
    "dictation_language": "en",
    "dictation_custom_words": ["Jarvis", "Jarvis"],
    # Shortcuts & Interactions
    "ask_bar_tap_enabled": True,
    "circle_while_talking_enabled": True,
    "auto_clear_drawings_on_app_switch": True,
    # Brain, Local Models, Hybrid & Lite Mode
    "chat_model": "deepseek-ai/DeepSeek-V4.1-Flash",
    "custom_api_base_url": "",
    "remember_style_notes": [],
    "local_servers": [],
    "lite_mode_enabled": False,
    "reasoning_effort": "auto",
    "hybrid_models": {},
    "backup_model_1": "",
    "backup_model_2": "",
    "effort_per_slot": {},
    # Agent & Safety Controls
    "agent_actions_enabled": True,
    "allow_file_edits": True,
    "allow_shell_commands": True,
    "allow_app_control": True,
    "skip_allow_prompts": False,
    "projects_root_dir": r"D:\Projects",
    "autonomous_agent_max_steps": 6,
    # Subagents, Morning Briefing, Auto-Learn Skills & Encrypted Sync
    "subagent_models": {},
    "subagent_delegate_mode": "ask",
    "morning_briefing_enabled": True,
    "morning_briefing_todoist": False,
    "last_briefing_day": "",
    "auto_learn_skills_enabled": True,
    "sync_folder_path": "",
    "sync_device_id": "",
    "sync_enabled": False,
    # Roamies Companion & Productivity Integrations
    "voice_beats_enabled": True,
    "focus_distraction_sites": [
        "youtube",
        "reddit",
        "twitter",
        "x.com",
        "netflix",
        "instagram",
        "twitch",
        "tiktok",
        "discord",
    ],
    "game_mode_auto_pause": True,
    "roamies_hands_path": r"D:\Projects\Roamies\roamies-hands.exe",
    "roamies_beats_dir": r"D:\Projects\Roamies\sidecar\beats",
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

    def __init__(
        self,
        settings_file_path: str = USER_SETTINGS_FILE_PATH,
        env_file_path: str = ENV_FILE_PATH,
        use_process_env: bool = True,
    ) -> None:
        super().__init__()
        self._settings_file_path = settings_file_path
        self._env_file_path = env_file_path
        self._use_process_env = use_process_env
        self._lock = threading.RLock()
        self._values: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self._secrets: Dict[str, str] = {}
        self._load()

    # -- preferences -------------------------------------------------------

    def _load(self) -> None:
        explicit_model_saved = False
        if os.path.exists(self._settings_file_path):
            try:
                with open(self._settings_file_path, "r", encoding="utf-8") as settings_file:
                    stored_values = json.load(settings_file)
                if isinstance(stored_values, dict):
                    if stored_values.get("chat_model"):
                        explicit_model_saved = True
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
        env_model_override = self.get_env_custom_model()
        if (not explicit_model_saved) and env_model_override:
            self._values["chat_model"] = env_model_override
        elif not explicit_model_saved:
            custom_key_value, custom_key_alias = self.get_custom_secret_with_name()
            if (
                self.has_secret(SECRET_GOOGLE_API_KEY)
                and (not self.has_secret(SECRET_RIVER_API_KEY))
                and (not custom_key_value)
            ):
                self._values["chat_model"] = DEFAULT_GOOGLE_CHAT_MODEL
            elif custom_key_value and (not self.has_secret(SECRET_RIVER_API_KEY)) and (not self.has_secret(SECRET_GOOGLE_API_KEY)):
                self._values["chat_model"] = DEFAULT_MODEL_FOR_KEY_ALIAS.get(custom_key_alias, "gpt-4o")

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

    def _has_dotenv_vision_config_locked(self) -> bool:
        if str(self._values.get("custom_api_base_url", "") or "").strip():
            return True
        all_dotenv_vision_keys = (
            CUSTOM_API_KEY_ALIASES
            + GOOGLE_API_KEY_ALIASES
            + ENV_ENDPOINT_URL_ALIASES
            + (SECRET_RIVER_API_KEY,)
        )
        return any(bool(self._secrets.get(alias_name, "").strip()) for alias_name in all_dotenv_vision_keys)

    def get_secret(self, secret_name: str) -> str:
        with self._lock:
            allow_env_vision_fallback = self._use_process_env and (not self._has_dotenv_vision_config_locked())
            if secret_name == SECRET_GOOGLE_API_KEY:
                for alias_name in GOOGLE_API_KEY_ALIASES:
                    stored_secret = self._secrets.get(alias_name, "").strip()
                    if stored_secret:
                        return stored_secret
                if allow_env_vision_fallback:
                    for alias_name in GOOGLE_API_KEY_ALIASES:
                        environment_override = os.environ.get(alias_name, "").strip()
                        if environment_override:
                            return environment_override
                return ""
            if secret_name == SECRET_CUSTOM_API_KEY:
                custom_secret, _matched_alias = self._lookup_custom_secret_locked()
                return custom_secret
            stored_direct = self._secrets.get(secret_name, "").strip()
            if stored_direct:
                return stored_direct
            if secret_name == SECRET_RIVER_API_KEY and not allow_env_vision_fallback:
                return ""
            if self._use_process_env:
                return os.environ.get(secret_name, "").strip()
            return ""

    def _lookup_custom_secret_locked(self) -> Tuple[str, str]:
        for alias_name in CUSTOM_API_KEY_ALIASES:
            stored_secret = self._secrets.get(alias_name, "").strip()
            if stored_secret:
                return stored_secret, alias_name
        if self._use_process_env and (not self._has_dotenv_vision_config_locked()):
            for alias_name in CUSTOM_API_KEY_ALIASES:
                environment_override = os.environ.get(alias_name, "").strip()
                if environment_override:
                    return environment_override, alias_name
        return "", ""

    def get_custom_secret_with_name(self) -> Tuple[str, str]:
        with self._lock:
            return self._lookup_custom_secret_locked()

    def get_effective_endpoint_url(self) -> str:
        with self._lock:
            configured_url = str(self._values.get("custom_api_base_url", "") or "").strip()
            if configured_url:
                return configured_url
            for alias_name in ENV_ENDPOINT_URL_ALIASES:
                stored_url = self._secrets.get(alias_name, "").strip()
                if stored_url:
                    return stored_url
            if self._use_process_env and (not self._has_dotenv_vision_config_locked()):
                for alias_name in ENV_ENDPOINT_URL_ALIASES:
                    environment_override = os.environ.get(alias_name, "").strip()
                    if environment_override:
                        return environment_override
            return ""

    def get_env_custom_model(self) -> str:
        with self._lock:
            for alias_name in ENV_CUSTOM_MODEL_ALIASES:
                stored_model = self._secrets.get(alias_name, "").strip()
                if stored_model:
                    return stored_model
            if self._use_process_env and (not self._has_dotenv_vision_config_locked()):
                for alias_name in ENV_CUSTOM_MODEL_ALIASES:
                    environment_override = os.environ.get(alias_name, "").strip()
                    if environment_override:
                        return environment_override
            return ""

    def has_any_vision_credentials(self) -> bool:
        return bool(
            self.get_secret(SECRET_CUSTOM_API_KEY)
            or self.get_secret(SECRET_GOOGLE_API_KEY)
            or self.get_secret(SECRET_RIVER_API_KEY)
            or self.get_effective_endpoint_url()
        )

    def get_projects_root_dir(self) -> str:
        """Returns the configured projects workspace directory (defaulting to D:\\Projects)."""
        with self._lock:
            configured_root = str(self._values.get("projects_root_dir", r"D:\Projects") or r"D:\Projects").strip()
            return os.path.realpath(configured_root)

    def get_roamies_hands_path(self) -> str:
        """Returns configured path to roamies-hands.exe for native Win32/UIA/OCR assistance."""
        with self._lock:
            return str(self._values.get("roamies_hands_path", r"D:\Projects\Roamies\roamies-hands.exe") or "").strip()

    def get_roamies_beats_dir(self) -> str:
        """Returns configured path to pre-recorded 24 kHz PCM voice beats directory."""
        with self._lock:
            return str(self._values.get("roamies_beats_dir", r"D:\Projects\Roamies\sidecar\beats") or "").strip()

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
