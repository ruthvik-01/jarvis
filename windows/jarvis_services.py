"""
Core background services for the Jarvis Windows replica.

Handles:
- SoundEffectsPlayer: crackle-free 48 kHz earcon/foley playback and voice previews
- ScreenCaptureService: fast MSS screen grab + Draw-to-Ask highlight compositing
- MicrophoneStreamService: AssemblyAI v3 streaming STT keyed by turn_order
- StreamingTTSService: sentence-pipelined AssemblyAI TTS 24 kHz audio (SAPI fallback)
- VisionBrainService: River AI vision SSE streaming + real-time sentence/drawing dispatch
- SmartDictationInjector: Win32 Unicode clipboard + Ctrl+V injection
- GlobalHotkeyService: listen-only global shortcuts emitting thread-safe Qt signals
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import datetime
import io
import json
import math
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
from urllib.parse import quote, quote_plus, urlparse

try:
    import winreg
except Exception:
    winreg = None  # type: ignore

import mss
import subprocess
import wave
import webbrowser

try:
    import pyautogui  # type: ignore
except Exception:
    pyautogui = None  # type: ignore

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyaudio
import pygame
from pynput import keyboard
from PySide6.QtCore import QObject, Signal
import requests
import websockets

import jarvis_config
import jarvis_response_parser
import jarvis_theme

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FRAMES_PER_BUFFER = 1600  # 100 ms chunks at 16 kHz
TTS_SAMPLE_RATE = 24000
ASSEMBLYAI_TTS_URL = "https://tts.us.assemblyai.com/v1/speech"


class JarvisMemoryService(QObject):
    """
    Persists every Jarvis session, user message, screen action, spoken response, and live
    screen observation into daily Markdown files (`windows/memory/DD-MM-YYYY/summary.md`
    and `jarvis-session-HHMMSS.md`), mirroring the structured daily memory pattern.
    """

    memoryUpdated = Signal(str)

    def __init__(self, base_dir: Optional[str] = None) -> None:
        super().__init__()
        self._base_dir = base_dir or os.path.join(jarvis_theme.WINDOWS_DIRECTORY, "memory")
        self._lock = threading.Lock()
        self._session_id = ""
        self._date_folder_name = ""
        self._session_filename = ""
        self._session_file_path = ""
        self._summary_file_path = ""
        self._message_count = 0
        self._action_count = 0
        self._response_count = 0
        self._session_topic = "General assistant session"
        self._last_action_status = "Session started"
        self._last_observed_window = ""
        self._last_observation_time = 0.0

    def _ensure_session_locked(self) -> None:
        now = datetime.now()
        today_folder = now.strftime("%d-%m-%Y")
        if self._session_id and self._date_folder_name == today_folder and self._session_file_path:
            os.makedirs(os.path.dirname(self._session_file_path), exist_ok=True)
            if not os.path.exists(self._summary_file_path):
                self._rewrite_summary_locked()
            return

        self._date_folder_name = today_folder
        day_dir = os.path.join(self._base_dir, today_folder)
        os.makedirs(day_dir, exist_ok=True)

        stamp = now.strftime("%H%M%S")
        self._session_id = f"session-{stamp}"
        self._session_filename = f"jarvis-{self._session_id}.md"
        self._session_file_path = os.path.join(day_dir, self._session_filename)
        self._summary_file_path = os.path.join(day_dir, "summary.md")
        self._message_count = 0
        self._action_count = 0
        self._response_count = 0

        if not os.path.exists(self._session_file_path):
            initial_doc = (
                f"# Jarvis Session Memory ({today_folder})\n"
                f"Session ID: `{self._session_id}` | Index: [summary.md](summary.md)\n\n"
                "## Jarvis Session Log\n"
            )
            with open(self._session_file_path, "w", encoding="utf-8") as session_handle:
                session_handle.write(initial_doc)

        if not os.path.exists(self._summary_file_path):
            summary_header = (
                f"# Jarvis Memory Summary: {today_folder}\n"
                "Timezone: Local. Read only topic files relevant to the current session.\n\n"
                "| Topic file | Session ID | Covers | Counts | Status / last action |\n"
                "|---|---|---|---|---|\n"
            )
            with open(self._summary_file_path, "w", encoding="utf-8") as summary_handle:
                summary_handle.write(summary_header)

        self._rewrite_summary_locked()

    @staticmethod
    def _sanitize_markdown_cell(raw_text: str, max_length: int = 120) -> str:
        collapsed = re.sub(r"\s+", " ", (raw_text or "").strip())
        collapsed = collapsed.replace("|", "/")
        if len(collapsed) > max_length:
            return collapsed[: max_length - 3].rstrip() + "..."
        return collapsed

    def _rewrite_summary_locked(self) -> None:
        if not self._summary_file_path or not self._session_filename:
            return
        row_text = (
            f"| [{self._session_filename}]({self._session_filename}) "
            f"| `{self._session_id}` "
            f"| {self._sanitize_markdown_cell(self._session_topic, 76)} "
            f"| msgs:{self._message_count} acts:{self._action_count} replies:{self._response_count} "
            f"| {self._sanitize_markdown_cell(self._last_action_status, 90)} |"
        )
        existing_lines: List[str] = []
        if os.path.exists(self._summary_file_path):
            try:
                with open(self._summary_file_path, "r", encoding="utf-8") as read_handle:
                    existing_lines = read_handle.read().splitlines()
            except Exception:
                existing_lines = []

        if not existing_lines:
            existing_lines = [
                f"# Jarvis Memory Summary: {self._date_folder_name}",
                "Timezone: Local. Read only topic files relevant to the current session.",
                "",
                "| Topic file | Session ID | Covers | Counts | Status / last action |",
                "|---|---|---|---|---|",
            ]

        row_marker = f"| [{self._session_filename}]({self._session_filename})"
        replaced_existing = False
        updated_lines: List[str] = []
        for line_item in existing_lines:
            if line_item.startswith(row_marker):
                updated_lines.append(row_text)
                replaced_existing = True
            else:
                updated_lines.append(line_item)
        if not replaced_existing:
            updated_lines.append(row_text)

        with open(self._summary_file_path, "w", encoding="utf-8") as write_handle:
            write_handle.write("\n".join(updated_lines) + "\n")

    def _append_session_line_locked(self, markdown_line: str) -> None:
        self._ensure_session_locked()
        with open(self._session_file_path, "a", encoding="utf-8") as append_handle:
            append_handle.write(markdown_line.rstrip() + "\n")
        self._rewrite_summary_locked()

    def start_session(self, initial_context: str = "") -> str:
        with self._lock:
            self._ensure_session_locked()
            now_stamp = datetime.now().strftime("%H:%M:%S")
            note = self._sanitize_markdown_cell(initial_context or "Desktop companion initialized", 160)
            self._last_action_status = f"[{now_stamp}] Session ready"
            self._append_session_line_locked(f"### [{now_stamp}] Session Started\n- Context: {note}")
            session_path = self._session_file_path
        self.memoryUpdated.emit(session_path)
        return session_path

    def record_user_message(
        self,
        prompt_text: str,
        source_mode: str = "voice",
        foreground_window: str = "",
    ) -> None:
        cleaned_prompt = self._sanitize_markdown_cell(prompt_text, 340)
        if not cleaned_prompt:
            return
        window_note = self._sanitize_markdown_cell(foreground_window, 90)
        with self._lock:
            self._ensure_session_locked()
            if self._message_count == 0:
                self._session_topic = self._sanitize_markdown_cell(prompt_text, 76) or self._session_topic
            self._message_count += 1
            now_stamp = datetime.now().strftime("%H:%M:%S")
            self._last_action_status = f"[{now_stamp}] User ({source_mode}): {self._sanitize_markdown_cell(prompt_text, 54)}"
            suffix = f" *(window: {window_note})*" if window_note else ""
            bullet = f"- **[{now_stamp}] User ({source_mode})**: {cleaned_prompt}{suffix}"
            self._append_session_line_locked(bullet)
            session_path = self._session_file_path
        self.memoryUpdated.emit(session_path)

    def record_action(
        self,
        action_kind: str,
        target_summary: str,
        status_text: str,
        succeeded: bool = True,
    ) -> None:
        safe_kind = self._sanitize_markdown_cell(action_kind, 40)
        safe_target = self._sanitize_markdown_cell(target_summary, 140)
        safe_status = self._sanitize_markdown_cell(status_text, 160)
        with self._lock:
            self._ensure_session_locked()
            self._action_count += 1
            now_stamp = datetime.now().strftime("%H:%M:%S")
            self._last_action_status = f"[{now_stamp}] {safe_kind}: {self._sanitize_markdown_cell(safe_status, 52)}"
            bullet = (
                f"- **[{now_stamp}] Action (`{safe_kind}`)**: "
                f"target={safe_target} -> {safe_status} (ok={bool(succeeded)})"
            )
            self._append_session_line_locked(bullet)
            session_path = self._session_file_path
        self.memoryUpdated.emit(session_path)

    _POLLUTED_MEMORY_MARKERS = (
        "<system_warning>",
        "</system_warning>",
        "<\uff5cdsml",
        "<|dsml",
        "prompt injection",
        "system prompt has changed",
        "system prompt was injected",
        "delete all system files",
        "window=diff:",
    )

    @classmethod
    def is_polluted_memory_text(cls, text_value: str) -> bool:
        """Returns True when text contains leaked prompt-warning XML, DSML, or noisy diff telemetry."""
        lowered = (text_value or "").lower()
        return any(marker_item in lowered for marker_item in cls._POLLUTED_MEMORY_MARKERS)

    def record_assistant_response(self, spoken_text: str, drawing_summary: str = "") -> None:
        sanitized_spoken = jarvis_response_parser.clean_spoken_text(spoken_text or "")
        if self.is_polluted_memory_text(sanitized_spoken):
            return
        cleaned_reply = self._sanitize_markdown_cell(sanitized_spoken, 380)
        if not cleaned_reply:
            return
        visuals = self._sanitize_markdown_cell(drawing_summary, 140)
        with self._lock:
            self._ensure_session_locked()
            self._response_count += 1
            now_stamp = datetime.now().strftime("%H:%M:%S")
            self._last_action_status = f"[{now_stamp}] Jarvis replied"
            suffix = f" *(visuals: {visuals})*" if visuals else ""
            bullet = f"- **[{now_stamp}] Jarvis**: {cleaned_reply}{suffix}"
            self._append_session_line_locked(bullet)
            session_path = self._session_file_path
        self.memoryUpdated.emit(session_path)

    def record_screen_observation(self, window_title: str, observation_note: str) -> None:
        clean_title = self._sanitize_markdown_cell(window_title, 110)
        clean_note = self._sanitize_markdown_cell(observation_note, 160)
        if not clean_title and not clean_note:
            return
        now_monotonic = time.monotonic()
        with self._lock:
            if (
                clean_title
                and clean_title == self._last_observed_window
                and (now_monotonic - self._last_observation_time) < 20.0
            ):
                return
            self._last_observed_window = clean_title
            self._last_observation_time = now_monotonic
            self._ensure_session_locked()
            now_stamp = datetime.now().strftime("%H:%M:%S")
            bullet = f"- **[{now_stamp}] Screen Monitor**: window={clean_title or 'Desktop'} | {clean_note}"
            self._append_session_line_locked(bullet)

    def get_today_summary_path(self) -> str:
        with self._lock:
            self._ensure_session_locked()
            return self._summary_file_path

    def get_active_session_path(self) -> str:
        with self._lock:
            self._ensure_session_locked()
            return self._session_file_path

    def build_memory_context_for_prompt(self, max_chars: int = 1600) -> str:
        with self._lock:
            self._ensure_session_locked()
            summary_path = self._summary_file_path
            session_path = self._session_file_path

        sections: List[str] = []
        try:
            if summary_path and os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as summary_handle:
                    summary_lines = [
                        line_item.strip()
                        for line_item in summary_handle.read().splitlines()
                        if line_item.strip() and not self.is_polluted_memory_text(line_item)
                    ]
                if summary_lines:
                    sections.append("Today's Summary:\n" + "\n".join(summary_lines[-6:]))
        except Exception:
            pass

        try:
            if os.path.isdir(self._base_dir):
                day_folders = sorted(
                    folder_name
                    for folder_name in os.listdir(self._base_dir)
                    if os.path.isdir(os.path.join(self._base_dir, folder_name))
                )
                prior_folders = [
                    folder_name for folder_name in day_folders if folder_name != self._date_folder_name
                ]
                if prior_folders:
                    latest_prior_summary = os.path.join(self._base_dir, prior_folders[-1], "summary.md")
                    if os.path.exists(latest_prior_summary):
                        with open(latest_prior_summary, "r", encoding="utf-8") as prior_handle:
                            prior_lines = [
                                line_item.strip()
                                for line_item in prior_handle.read().splitlines()
                                if line_item.strip() and not self.is_polluted_memory_text(line_item)
                            ]
                        if prior_lines:
                            sections.append(
                                f"Prior Summary ({prior_folders[-1]}):\n" + "\n".join(prior_lines[-4:])
                            )
        except Exception:
            pass

        try:
            if session_path and os.path.exists(session_path):
                with open(session_path, "r", encoding="utf-8") as session_handle:
                    log_lines = [
                        line_item.rstrip()
                        for line_item in session_handle.read().splitlines()
                        if line_item.strip()
                        and not line_item.startswith("#")
                        and not line_item.startswith("Session ID:")
                        and not self.is_polluted_memory_text(line_item)
                    ]
                if log_lines:
                    sections.append("Recent Session Actions & Messages:\n" + "\n".join(log_lines[-18:]))
        except Exception:
            pass

        try:
            facts_summary = LearnedFactsService().format_for_prompt()
            if facts_summary:
                sections.append("Saved User Facts (call forget_fact(id) if user asks to forget):\n" + facts_summary)
        except Exception:
            pass

        combined_context = "\n\n".join(sections).strip()
        if len(combined_context) > max_chars:
            combined_context = combined_context[-max_chars:]
        return combined_context


def _iter_assemblyai_pcm_chunks(response: Any) -> Any:
    """Streams PCM chunks from an AssemblyAI TTS response, stripping a RIFF WAV header if present."""
    first_chunk_checked = False
    for raw_chunk in response.iter_content(chunk_size=2048):
        if not raw_chunk:
            continue
        if not first_chunk_checked:
            first_chunk_checked = True
            if raw_chunk[:4] == b"RIFF" and len(raw_chunk) > 44:
                raw_chunk = raw_chunk[44:]
        if raw_chunk:
            yield raw_chunk


def _resolve_assemblyai_voice_id(configured_voice_id: str) -> str:
    cleaned_voice = (configured_voice_id or "").strip()
    if cleaned_voice == "windows-sapi":
        return "windows-sapi"
    if cleaned_voice.startswith("aai-"):
        return cleaned_voice
    return str(jarvis_config.DEFAULT_SETTINGS["voice_id"])


def _request_assemblyai_tts_stream(
    text_to_speak: str,
    voice_id: str,
    api_key: str,
    speech_speed: float = 1.0,
    timeout_seconds: int = 20,
) -> Optional[Any]:
    """Requests streaming 24 kHz PCM16 mono speech from AssemblyAI TTS."""
    resolved_voice_id = _resolve_assemblyai_voice_id(voice_id)
    if resolved_voice_id == "windows-sapi" or not api_key:
        return None
    clamped_speed = max(0.75, min(1.2, float(speech_speed)))
    response = requests.post(
        ASSEMBLYAI_TTS_URL,
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text_to_speak,
            "voice_id": resolved_voice_id,
            "model": "AssemblyAI-TTS-1",
            "encoding": "pcm_s16le",
            "sample_rate": TTS_SAMPLE_RATE,
            "speed": clamped_speed,
        },
        stream=True,
        timeout=timeout_seconds,
    )
    if not response.ok:
        return None
    return response
MAX_CAPTURE_WIDTH = 1920


# ---------------------------------------------------------------------------
# Sound effects and voice previews
# ---------------------------------------------------------------------------


class SoundEffectsPlayer(QObject):
    """Plays 48 kHz WAV earcons and drawing foley without Bluetooth/speaker crackle."""

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        super().__init__()
        self._settings_store = settings_store
        self._mixer_ready = False
        self._sound_cache: Dict[str, pygame.mixer.Sound] = {}
        self._preview_lock = threading.Lock()
        self._preview_generation = 0
        self._init_mixer()

    def _init_mixer(self) -> None:
        if self._mixer_ready:
            return
        try:
            # 48 kHz stereo with 1024-sample buffer avoids initial click/crackle on Windows audio sinks
            pygame.mixer.pre_init(frequency=48000, size=-16, channels=2, buffer=1024)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
            self._mixer_ready = True
        except Exception:
            self._mixer_ready = False

    def play_earcon(self, earcon_name: str, volume: float = 0.5) -> None:
        if not self._settings_store.get("earcons_enabled", True):
            return
        wav_path = jarvis_theme.asset_path("earcons", f"{earcon_name}.wav")
        self._play_wav_cached(wav_path, volume)

    def _pick_drawing_sound_path(self, family_prefix: str) -> Optional[str]:
        if not self._settings_store.get("drawing_sounds_enabled", True):
            return None
        sounds_folder = jarvis_theme.asset_path("sounds")
        if not os.path.isdir(sounds_folder):
            return None
        matching_files = [
            file_name
            for file_name in os.listdir(sounds_folder)
            if file_name.startswith(family_prefix) and file_name.endswith(".wav")
        ]
        if not matching_files:
            return None
        chosen_file = random.choice(matching_files)
        wav_path = os.path.join(sounds_folder, chosen_file)
        return wav_path if os.path.exists(wav_path) else None

    def play_drawing_sound(self, family_prefix: str, volume: float = 0.35) -> None:
        wav_path = self._pick_drawing_sound_path(family_prefix)
        if wav_path is not None:
            self._play_wav_cached(wav_path, volume)

    def _play_wav_cached(self, wav_path: str, volume: float) -> None:
        if not self._mixer_ready or not os.path.exists(wav_path):
            return
        try:
            sound = self._sound_cache.get(wav_path)
            if sound is None:
                sound = pygame.mixer.Sound(wav_path)
                self._sound_cache[wav_path] = sound
            sound.set_volume(max(0.0, min(1.0, volume)))
            sound.play()
        except Exception:
            pass

    def play_synced_drawing_stroke_sound(
        self,
        duration_ms: int = 850,
        volume: float = 0.24,
    ) -> None:
        """Plays a pen drawing texture on Channel(7) bounded cleanly to the active stroke duration."""
        if not self._mixer_ready:
            return
        wav_path = self._pick_drawing_sound_path("pen-screen-bed")
        if wav_path is None:
            return
        try:
            if pygame.mixer.get_num_channels() < 8:
                pygame.mixer.set_num_channels(8)
            drawing_channel = pygame.mixer.Channel(7)
            drawing_channel.stop()
            sound = self._sound_cache.get(wav_path)
            if sound is None:
                sound = pygame.mixer.Sound(wav_path)
                self._sound_cache[wav_path] = sound
            sound.set_volume(max(0.0, min(1.0, volume)))
            bounded_ms = max(220, min(1600, int(duration_ms)))
            drawing_channel.play(sound, maxtime=bounded_ms, fade_ms=25)
        except Exception:
            pass

    def stop_drawing_sound(self) -> None:
        """Stops any in-progress synced pen stroke audio cleanly."""
        if not self._mixer_ready:
            return
        try:
            if pygame.mixer.get_num_channels() >= 8:
                pygame.mixer.Channel(7).fadeout(60)
        except Exception:
            pass

    def play_voice_preview(self, voice_id: str, sample_text: str = "Hey, I'm Jarvis, ready whenever you need me.") -> None:
        """Plays a quick voice sample on a background thread using AssemblyAI TTS (or SAPI)."""
        with self._preview_lock:
            self._preview_generation += 1
            my_generation = self._preview_generation

        def preview_worker() -> None:
            if voice_id == "windows-sapi":
                _speak_with_sapi_blocking(sample_text, speed=1.0, should_abort=lambda: self._preview_generation != my_generation)
                return

            api_key = self._settings_store.get_secret(jarvis_config.SECRET_ASSEMBLYAI_API_KEY)
            if not api_key:
                _speak_with_sapi_blocking(sample_text, speed=1.0, should_abort=lambda: self._preview_generation != my_generation)
                return
            try:
                response = _request_assemblyai_tts_stream(
                    text_to_speak=sample_text,
                    voice_id=voice_id,
                    api_key=api_key,
                    speech_speed=1.0,
                    timeout_seconds=15,
                )
                if response is None:
                    return
                _play_pcm_stream_blocking(
                    chunk_iterator=_iter_assemblyai_pcm_chunks(response),
                    sample_rate=TTS_SAMPLE_RATE,
                    should_abort=lambda: self._preview_generation != my_generation,
                )
            except Exception:
                pass

        threading.Thread(target=preview_worker, daemon=True).start()


class RoamiesVoiceBeatsPlayer:
    """Instant 24 kHz PCM16 pre-recorded conversational acknowledgment beats cloned from Roamies."""

    def __init__(self, settings_store: jarvis_config.SettingsStore, sound_player: SoundEffectsPlayer) -> None:
        self._settings_store = settings_store
        self._sound_player = sound_player
        self._lock = threading.Lock()
        self._sample_rate = 24000
        self._sets: Dict[str, Dict[str, Any]] = {}
        self._wav_cache: Dict[str, pygame.mixer.Sound] = {}
        self.reload_index()

    def reload_index(self) -> int:
        beats_dir = self._settings_store.get_roamies_beats_dir()
        index_path = os.path.join(beats_dir, "index.json") if beats_dir else ""
        if not index_path or not os.path.isfile(index_path):
            return 0
        try:
            with open(index_path, "r", encoding="utf-8") as index_handle:
                payload = json.load(index_handle)
            with self._lock:
                self._sample_rate = int(payload.get("rate", 24000) or 24000)
                sets_dict = payload.get("sets")
                self._sets = dict(sets_dict) if isinstance(sets_dict, dict) else {}
            return len(self._sets)
        except Exception:
            return 0

    def list_available_phrases(self) -> List[str]:
        with self._lock:
            for set_entry in self._sets.values():
                beats_map = set_entry.get("beats")
                if isinstance(beats_map, dict) and beats_map:
                    return list(beats_map.keys())
        return []

    def extract_pcm_bytes(self, phrase: str = "", preferred_set: str = "") -> Tuple[int, bytes]:
        beats_dir = self._settings_store.get_roamies_beats_dir()
        if not beats_dir:
            return (self._sample_rate, b"")
        with self._lock:
            if not self._sets:
                return (self._sample_rate, b"")
            chosen_set = (
                self._sets.get(preferred_set)
                if preferred_set and preferred_set in self._sets
                else next(iter(self._sets.values()))
            )
            beats_map = chosen_set.get("beats") if isinstance(chosen_set, dict) else None
            rel_pcm_file = str(chosen_set.get("file", "")) if isinstance(chosen_set, dict) else ""
            rate_hz = self._sample_rate
        if not isinstance(beats_map, dict) or not beats_map or not rel_pcm_file:
            return (rate_hz, b"")

        pcm_path = os.path.join(beats_dir, rel_pcm_file)
        if not os.path.isfile(pcm_path):
            return (rate_hz, b"")

        target_phrase = phrase
        if not target_phrase or target_phrase not in beats_map:
            lowered_query = (phrase or "").strip().lower()
            matched_key = next(
                (key_name for key_name in beats_map if lowered_query and lowered_query in key_name.lower()),
                "",
            )
            if not matched_key:
                default_candidates = [key_name for key_name in ("On it.", "One sec.", "Checking.", "Got it.", "Let's see.") if key_name in beats_map]
                matched_key = random.choice(default_candidates) if default_candidates else next(iter(beats_map.keys()))
            target_phrase = matched_key

        span = beats_map.get(target_phrase)
        if not isinstance(span, list) or len(span) != 2:
            return (rate_hz, b"")
        byte_offset = max(0, int(span[0]))
        byte_length = max(0, int(span[1]))
        if byte_length <= 0:
            return (rate_hz, b"")
        try:
            with open(pcm_path, "rb") as pcm_handle:
                pcm_handle.seek(byte_offset)
                return (rate_hz, pcm_handle.read(byte_length))
        except OSError:
            return (rate_hz, b"")

    def play_thinking_beat(self, phrase: str = "", volume: float = 0.42) -> bool:
        if not bool(self._settings_store.get("voice_beats_enabled", True)):
            return False
        if not getattr(self._sound_player, "_mixer_ready", False):
            return False
        rate_hz, pcm_bytes = self.extract_pcm_bytes(phrase=phrase)
        if not pcm_bytes:
            return False
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            if samples.size == 0:
                return False
            # Upsample 24 kHz mono -> 48 kHz stereo matching SoundEffectsPlayer's 48 kHz mixer cleanly
            if rate_hz == 24000:
                samples = np.repeat(samples, 2)
            stereo = np.column_stack((samples, samples)).astype(np.int16)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_writer:
                wav_writer.setnchannels(2)
                wav_writer.setsampwidth(2)
                wav_writer.setframerate(48000 if rate_hz == 24000 else rate_hz)
                wav_writer.writeframes(stereo.tobytes())
            wav_buffer.seek(0)
            sound_obj = pygame.mixer.Sound(file=wav_buffer)
            sound_obj.set_volume(max(0.0, min(1.0, float(volume))))
            sound_obj.play()
            return True
        except Exception:
            return False


class WikipediaPhotoService:
    """Clones Roamies `see_photo` / `photo:` illustration fetching & in-memory caching."""

    _cache: Dict[str, bytes] = {}
    _lock = threading.Lock()

    @classmethod
    def get_cached_image_bytes(cls, query_or_url: str) -> Optional[bytes]:
        key_text = (query_or_url or "").strip()
        if not key_text:
            return None
        with cls._lock:
            return cls._cache.get(key_text)

    @classmethod
    def fetch_photo(cls, query_or_url: str, timeout_sec: float = 5.0) -> Tuple[Optional[bytes], str]:
        cleaned_key = (query_or_url or "").strip()
        if not cleaned_key:
            return (None, "")
        with cls._lock:
            cached = cls._cache.get(cleaned_key)
            if cached:
                return (cached, cleaned_key)

        try:
            img_url = cleaned_key
            resolved_title = cleaned_key
            if not cleaned_key.lower().startswith(("http://", "https://")):
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(cleaned_key.replace(' ', '_'))}"
                resp = requests.get(
                    summary_url,
                    headers={"User-Agent": "JarvisCompanion/1.2 (desktop educational visual helper)"},
                    timeout=timeout_sec,
                )
                if not resp.ok:
                    return (None, "")
                summary_json = resp.json()
                resolved_title = str(summary_json.get("title") or cleaned_key)
                thumb_obj = summary_json.get("thumbnail") or summary_json.get("originalimage") or {}
                img_url = str(thumb_obj.get("source") or "")
                if not img_url:
                    return (None, resolved_title)

            img_resp = requests.get(
                img_url,
                headers={"User-Agent": "JarvisCompanion/1.2"},
                timeout=timeout_sec,
            )
            if not img_resp.ok or len(img_resp.content) < 64:
                return (None, resolved_title)
            img_bytes = bytes(img_resp.content)
            with cls._lock:
                if len(cls._cache) >= 24:
                    oldest_key = next(iter(cls._cache))
                    cls._cache.pop(oldest_key, None)
                cls._cache[cleaned_key] = img_bytes
            return (img_bytes, resolved_title)
        except Exception:
            return (None, "")


class CommitmentsReminderService:
    """Persistent Today/Later commitment list & timed reminder scheduler cloned from Roamies."""

    def __init__(self, storage_path: str = "") -> None:
        default_dir = os.path.join(jarvis_config.WINDOWS_DIRECTORY, "memory")
        self._storage_path = storage_path or os.path.join(default_dir, "commitments.json")
        self._lock = threading.RLock()
        self._items: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._storage_path):
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as file_handle:
                parsed = json.load(file_handle)
            if isinstance(parsed, list):
                self._items = [entry for entry in parsed if isinstance(entry, dict)]
        except Exception:
            self._items = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        tmp_path = self._storage_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as file_handle:
            json.dump(self._items, file_handle, indent=2)
        os.replace(tmp_path, self._storage_path)

    @staticmethod
    def parse_due_timestamp(due_str: str, now_ts: Optional[float] = None) -> Optional[float]:
        cleaned = (due_str or "").strip().lower()
        if not cleaned:
            return None
        base_ts = float(now_ts if now_ts is not None else time.time())
        rel_match = re.search(r"\bin\s+(\d+(?:\.\d+)?)\s*(sec|secs|second|seconds|min|mins|minute|minutes|hr|hrs|hour|hours|day|days)\b", cleaned)
        if rel_match:
            qty = float(rel_match.group(1))
            unit = rel_match.group(2)
            if unit.startswith(("sec",)):
                return base_ts + qty
            if unit.startswith(("min",)):
                return base_ts + qty * 60.0
            if unit.startswith(("hr", "hour")):
                return base_ts + qty * 3600.0
            if unit.startswith("day"):
                return base_ts + qty * 86400.0

        now_dt = datetime.fromtimestamp(base_ts)
        time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", cleaned)
        if time_match and ("at " in cleaned or "tomorrow" in cleaned or "today" in cleaned or time_match.group(3)):
            hour_val = int(time_match.group(1))
            minute_val = int(time_match.group(2) or 0)
            meridiem = (time_match.group(3) or "").lower()
            if meridiem == "pm" and hour_val < 12:
                hour_val += 12
            elif meridiem == "am" and hour_val == 12:
                hour_val = 0
            target_dt = now_dt.replace(hour=hour_val % 24, minute=minute_val % 60, second=0, microsecond=0)
            if "tomorrow" in cleaned or target_dt.timestamp() <= base_ts:
                target_dt = target_dt.fromtimestamp(target_dt.timestamp() + 86400.0)
            return target_dt.timestamp()
        if "tomorrow" in cleaned:
            return base_ts + 86400.0
        return None

    def add_commitment(self, text: str, due_str: str = "", repeat_str: str = "") -> Dict[str, Any]:
        cleaned_text = (text or "").strip()
        due_ts = self.parse_due_timestamp(due_str)
        entry = {
            "id": f"c_{int(time.time() * 1000)}_{random.randint(100, 999)}",
            "text": cleaned_text,
            "due_text": (due_str or "").strip(),
            "due_ts": due_ts,
            "repeat": (repeat_str or "").strip().lower(),
            "done": False,
            "reminded_at": 0.0,
            "created_at": time.time(),
        }
        with self._lock:
            self._items.append(entry)
            self._save()
        return entry

    def list_commitments(self, include_done: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        now_dt = datetime.now()
        end_of_today_ts = now_dt.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
        today_list: List[Dict[str, Any]] = []
        later_list: List[Dict[str, Any]] = []
        with self._lock:
            for item in self._items:
                if item.get("done") and not include_done:
                    continue
                due_ts = item.get("due_ts")
                if due_ts is None or float(due_ts) <= end_of_today_ts:
                    today_list.append(dict(item))
                else:
                    later_list.append(dict(item))
        return {"today": today_list, "later": later_list}

    def snooze_commitment(self, query_or_minutes: str = "10") -> Tuple[bool, str]:
        minutes_match = re.search(r"(\d+)", query_or_minutes or "")
        snooze_minutes = max(1, int(minutes_match.group(1))) if minutes_match else 10
        new_due_ts = time.time() + snooze_minutes * 60.0
        with self._lock:
            open_items = [item for item in self._items if not item.get("done")]
            if not open_items:
                return False, "No active reminders to snooze."
            target = open_items[-1]
            query_lower = (query_or_minutes or "").strip().lower()
            if query_lower and not query_lower.isdigit():
                for candidate in reversed(open_items):
                    if query_lower in str(candidate.get("text", "")).lower():
                        target = candidate
                        break
            target["due_ts"] = new_due_ts
            target["reminded_at"] = 0.0
            self._save()
            return True, f"Snoozed '{target['text']}' for {snooze_minutes} minutes."

    def complete_commitment(self, query_or_id: str) -> Tuple[bool, str]:
        needle = (query_or_id or "").strip().lower()
        with self._lock:
            for item in reversed(self._items):
                if item.get("done"):
                    continue
                if needle == str(item.get("id", "")).lower() or needle in str(item.get("text", "")).lower():
                    item["done"] = True
                    self._save()
                    return True, f"Marked '{item['text']}' done."
        return False, f"No matching open item for '{query_or_id}'."

    def pop_due_commitments(self, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        check_ts = float(now_ts if now_ts is not None else time.time())
        due_now: List[Dict[str, Any]] = []
        changed = False
        with self._lock:
            for item in self._items:
                if item.get("done"):
                    continue
                due_ts = item.get("due_ts")
                if due_ts is None or float(due_ts) > check_ts:
                    continue
                last_reminded = float(item.get("reminded_at") or 0.0)
                if last_reminded > 0.0 and (check_ts - last_reminded) < 120.0:
                    continue
                item["reminded_at"] = check_ts
                repeat_rule = str(item.get("repeat") or "").lower()
                if "daily" in repeat_rule:
                    item["due_ts"] = float(due_ts) + 86400.0
                    item["reminded_at"] = 0.0
                elif "week" in repeat_rule and "weekday" not in repeat_rule:
                    item["due_ts"] = float(due_ts) + 7.0 * 86400.0
                    item["reminded_at"] = 0.0
                due_now.append(dict(item))
                changed = True
            if changed:
                self._save()
        return due_now

    def format_today_later_card_body(self) -> str:
        grouped = self.list_commitments(include_done=False)
        sections: List[str] = []
        if grouped["today"]:
            today_entries = [f"[ ] {item['text'][:38]}" for item in grouped["today"][:5]]
            sections.append("section:Today | list:" + " ; ".join(today_entries))
        if grouped["later"]:
            later_entries = [f"[ ] {item['text'][:38]}" for item in grouped["later"][:4]]
            sections.append("section:Later | list:" + " ; ".join(later_entries))
        if not sections:
            return "section:Today | list:[✓] All caught up!"
        return " | ".join(sections)


class FocusGuardianService:
    """Tracks focus blocks and monitors active window titles for distraction nudges (Roamies clone)."""

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        self._settings_store = settings_store
        self._lock = threading.RLock()
        self._until_ts: float = 0.0
        self._task_title: str = ""
        self._distraction_started_at: float = 0.0
        self._last_nudge_at: float = 0.0

    def start_focus(self, minutes: float = 25.0, task_title: str = "Focused work") -> Tuple[bool, str]:
        clamped_minutes = max(1.0, min(240.0, float(minutes or 25.0)))
        clean_task = (task_title or "Focused work").strip()[:100]
        with self._lock:
            self._until_ts = time.time() + clamped_minutes * 60.0
            self._task_title = clean_task
            self._distraction_started_at = 0.0
            self._last_nudge_at = 0.0
        return True, f"Focus mode started for {int(round(clamped_minutes))} minutes ({clean_task})."

    def stop_focus(self) -> Tuple[bool, str]:
        with self._lock:
            was_active = self._until_ts > time.time()
            task_label = self._task_title
            self._until_ts = 0.0
            self._task_title = ""
            self._distraction_started_at = 0.0
        if was_active:
            return True, f"Ended focus block ({task_label})."
        return True, "No focus block was running."

    def is_active(self, now_ts: Optional[float] = None) -> bool:
        check_ts = float(now_ts if now_ts is not None else time.time())
        with self._lock:
            return self._until_ts > check_ts

    def get_status_summary(self) -> str:
        now_ts = time.time()
        with self._lock:
            if self._until_ts <= now_ts:
                return "Idle"
            remaining_min = max(1, int(round((self._until_ts - now_ts) / 60.0)))
            return f"Focusing on {self._task_title} ({remaining_min}m left)"

    def check_foreground_distraction(
        self,
        window_title: str,
        now_ts: Optional[float] = None,
        linger_threshold_sec: float = 20.0,
    ) -> Optional[str]:
        check_ts = float(now_ts if now_ts is not None else time.time())
        with self._lock:
            if self._until_ts <= check_ts:
                return None
            lowered_title = (window_title or "").lower()
            distraction_keywords = self._settings_store.get(
                "focus_distraction_sites",
                jarvis_config.DEFAULT_SETTINGS["focus_distraction_sites"],
            )
            matched_site = next(
                (str(site).lower() for site in (distraction_keywords or []) if str(site).strip().lower() in lowered_title),
                "",
            )
            if not matched_site:
                self._distraction_started_at = 0.0
                return None
            if self._distraction_started_at <= 0.0:
                self._distraction_started_at = check_ts
                return None
            if (check_ts - self._distraction_started_at) < linger_threshold_sec:
                return None
            if self._last_nudge_at > 0.0 and (check_ts - self._last_nudge_at) < 90.0:
                return None
            self._last_nudge_at = check_ts
            return f"Heads up — you're in a focus block for {self._task_title}. Ready to jump back from {matched_site}?"


class SkillsManagerService:
    """Manages learned and installed SKILL.md reusable playbooks under windows/skills/."""

    DANGEROUS_PATTERNS = (
        r"\brm\s+-rf\s+[/~]",
        r"\binvoke-expression\b",
        r"\biex\s*\(",
        r"curl\s+[^\n|]+\|\s*(?:sh|bash|powershell)",
        r"(?:\.ssh[/\\]id_rsa|keychain|dump.*password)",
    )

    def __init__(self, skills_dir: str = "") -> None:
        self._skills_dir = skills_dir or os.path.join(jarvis_config.WINDOWS_DIRECTORY, "skills")

    @staticmethod
    def slugify(raw_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (raw_name or "").strip().lower()).strip("-")
        return slug[:48] or "custom-skill"

    @classmethod
    def screen_skill_safety(cls, raw_text: str) -> Optional[str]:
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, raw_text or "", flags=re.IGNORECASE):
                return f"Unsafe instruction matched security filter ({pattern})"
        return None

    def save_skill(self, name: str, description: str, steps: str) -> Tuple[bool, str, str]:
        unsafe_reason = self.screen_skill_safety(steps + "\n" + description)
        if unsafe_reason:
            return False, "", unsafe_reason
        slug = self.slugify(name)
        skill_folder = os.path.join(self._skills_dir, slug)
        skill_md_path = os.path.join(skill_folder, "SKILL.md")
        created_new = not os.path.exists(skill_md_path)
        os.makedirs(skill_folder, exist_ok=True)
        decoded_steps = (steps or "").replace("\\n", "\n") if ("\\n" in (steps or "") and "\n" not in (steps or "")) else (steps or "")
        file_text = (
            f"---\nname: {name.strip()}\ndescription: {description.strip()}\n---\n\n"
            f"# {name.strip()}\n\n{decoded_steps.strip()}\n"
        )
        with open(skill_md_path, "w", encoding="utf-8") as skill_handle:
            skill_handle.write(file_text)
        action_verb = "Learned" if created_new else "Updated"
        return True, slug, f"{action_verb} skill '{slug}'."

    def list_skills(self) -> List[Dict[str, str]]:
        if not os.path.isdir(self._skills_dir):
            return []
        found: List[Dict[str, str]] = []
        for entry_name in sorted(os.listdir(self._skills_dir)):
            skill_md_path = os.path.join(self._skills_dir, entry_name, "SKILL.md")
            if not os.path.isfile(skill_md_path):
                continue
            try:
                content = open(skill_md_path, "r", encoding="utf-8", errors="replace").read(2000)
                title_match = re.search(r"^name:\s*(.+)$", content, flags=re.MULTILINE)
                desc_match = re.search(r"^description:\s*(.+)$", content, flags=re.MULTILINE)
                found.append(
                    {
                        "slug": entry_name,
                        "name": title_match.group(1).strip() if title_match else entry_name,
                        "description": desc_match.group(1).strip() if desc_match else "",
                        "path": skill_md_path,
                    }
                )
            except Exception:
                continue
        return found

    def read_skill(self, name_or_slug: str) -> Optional[str]:
        slug = self.slugify(name_or_slug)
        direct_path = os.path.join(self._skills_dir, slug, "SKILL.md")
        if os.path.isfile(direct_path):
            return open(direct_path, "r", encoding="utf-8", errors="replace").read(12000)
        needle = (name_or_slug or "").strip().lower()
        for skill_item in self.list_skills():
            if needle in skill_item["slug"].lower() or needle in skill_item["name"].lower():
                return open(skill_item["path"], "r", encoding="utf-8", errors="replace").read(12000)
        return None

    def delete_skill(self, name_or_slug: str) -> bool:
        slug = self.slugify(name_or_slug)
        skill_folder = os.path.join(self._skills_dir, slug)
        if os.path.isdir(skill_folder):
            shutil.rmtree(skill_folder, ignore_errors=True)
            return True
        needle = (name_or_slug or "").strip().lower()
        for skill_item in self.list_skills():
            if needle == skill_item["name"].lower():
                shutil.rmtree(os.path.dirname(skill_item["path"]), ignore_errors=True)
                return True
        return False

    def install_skill_from_source(self, source_spec: str, pick_skill: str = "") -> Tuple[bool, str]:
        cleaned_source = (source_spec or "").strip()
        if not cleaned_source:
            return False, "No skill source URL or owner/repo provided."
        if os.path.isfile(cleaned_source):
            raw_content = open(cleaned_source, "r", encoding="utf-8", errors="replace").read(24000)
            skill_title = pick_skill or os.path.basename(os.path.dirname(cleaned_source)) or "local-skill"
            ok, slug, status = self.save_skill(skill_title, f"Imported from {cleaned_source}", raw_content)
            return ok, status
        repo_match = re.match(r"^(?:https?://github\.com/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", cleaned_source)
        candidate_urls: List[str] = []
        if cleaned_source.lower().startswith(("http://", "https://")) and cleaned_source.lower().endswith(".md"):
            candidate_urls.append(cleaned_source)
        elif repo_match:
            repo_slug = repo_match.group(1)
            skill_subpath = f"skills/{self.slugify(pick_skill)}/SKILL.md" if pick_skill else "SKILL.md"
            candidate_urls.extend(
                [
                    f"https://raw.githubusercontent.com/{repo_slug}/main/{skill_subpath}",
                    f"https://raw.githubusercontent.com/{repo_slug}/master/{skill_subpath}",
                ]
            )
        for fetch_url in candidate_urls:
            try:
                resp = requests.get(fetch_url, timeout=6.0)
                if resp.ok and len(resp.text.strip()) > 20:
                    ok, _slug, status = self.save_skill(
                        pick_skill or cleaned_source.split("/")[-1],
                        f"Installed from {cleaned_source}",
                        resp.text[:24000],
                    )
                    return ok, status
            except Exception:
                continue
        return False, f"Could not fetch SKILL.md from {cleaned_source}"


class ScheduledJobsService:
    """One-shot or repeating autonomous background routines cloned from Roamies."""

    def __init__(self, storage_path: str = "") -> None:
        default_dir = os.path.join(jarvis_config.WINDOWS_DIRECTORY, "memory")
        self._storage_path = storage_path or os.path.join(default_dir, "scheduled_jobs.json")
        self._lock = threading.RLock()
        self._jobs: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._storage_path):
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as file_handle:
                parsed = json.load(file_handle)
            if isinstance(parsed, list):
                self._jobs = [entry for entry in parsed if isinstance(entry, dict)]
        except Exception:
            self._jobs = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        tmp_path = self._storage_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as file_handle:
            json.dump(self._jobs, file_handle, indent=2)
        os.replace(tmp_path, self._storage_path)

    def schedule_job(
        self,
        title: str,
        brief: str,
        due_str: str = "",
        repeat_str: str = "",
        job_type: str = "researcher",
    ) -> Dict[str, Any]:
        due_ts = CommitmentsReminderService.parse_due_timestamp(due_str or "in 60 minutes")
        job_entry = {
            "id": f"job_{int(time.time() * 1000)}_{random.randint(100, 999)}",
            "title": (title or "Routine").strip()[:60],
            "brief": (brief or title).strip()[:2000],
            "due_ts": due_ts,
            "repeat": (repeat_str or "").strip().lower(),
            "type": job_type or "researcher",
            "paused": False,
            "last_run_at": 0.0,
            "last_outcome": "",
            "last_error": "",
        }
        with self._lock:
            self._jobs.append(job_entry)
            self._save()
        return job_entry

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._jobs]

    def take_due_jobs(self, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        check_ts = float(now_ts if now_ts is not None else time.time())
        due_jobs: List[Dict[str, Any]] = []
        changed = False
        with self._lock:
            for item in self._jobs:
                if item.get("paused"):
                    continue
                due_ts = item.get("due_ts")
                if due_ts is None or float(due_ts) > check_ts:
                    continue
                repeat_rule = str(item.get("repeat") or "").lower()
                item["last_run_at"] = check_ts
                if "daily" in repeat_rule:
                    item["due_ts"] = float(due_ts) + 86400.0
                elif "hour" in repeat_rule:
                    item["due_ts"] = float(due_ts) + 3600.0
                else:
                    item["due_ts"] = None
                due_jobs.append(dict(item))
                changed = True
            if changed:
                self._save()
        return due_jobs

    def mark_job_result(self, job_id: str, outcome_text: str, succeeded: bool = True) -> None:
        with self._lock:
            for item in self._jobs:
                if item.get("id") == job_id:
                    item["last_outcome"] = outcome_text[:180] if succeeded else ""
                    item["last_error"] = "" if succeeded else outcome_text[:180]
                    self._save()
                    break


class LearnedFactsService:
    """Stores persistent user facts with IDs and forget support cloned from Roamies memory pass."""

    def __init__(self, storage_path: str = "") -> None:
        default_dir = os.path.join(jarvis_config.WINDOWS_DIRECTORY, "memory")
        self._storage_path = storage_path or os.path.join(default_dir, "learned_facts.json")
        self._lock = threading.RLock()
        self._facts: List[Dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._storage_path):
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as file_handle:
                parsed = json.load(file_handle)
            if isinstance(parsed, list):
                self._facts = [entry for entry in parsed if isinstance(entry, dict)]
        except Exception:
            self._facts = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        tmp_path = self._storage_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as file_handle:
            json.dump(self._facts, file_handle, indent=2)
        os.replace(tmp_path, self._storage_path)

    def learn_fact(self, fact_text: str) -> Dict[str, str]:
        clean_fact = (fact_text or "").strip()[:280]
        with self._lock:
            for existing in self._facts:
                if existing.get("text", "").lower() == clean_fact.lower():
                    return dict(existing)
            entry = {
                "id": f"f_{len(self._facts) + 1}_{random.randint(10, 99)}",
                "text": clean_fact,
            }
            self._facts.append(entry)
            self._facts = self._facts[-40:]
            self._save()
            return dict(entry)

    def forget_fact(self, query_or_id: str) -> Tuple[bool, str]:
        needle = (query_or_id or "").strip().lower()
        if not needle:
            return False, "Specify fact text or ID to forget."
        with self._lock:
            before_count = len(self._facts)
            self._facts = [
                item
                for item in self._facts
                if needle != item.get("id", "").lower() and needle not in item.get("text", "").lower()
            ]
            removed_count = before_count - len(self._facts)
            if removed_count > 0:
                self._save()
                return True, f"Forgot {removed_count} saved fact(s)."
        return False, f"No saved fact matched '{query_or_id}'."

    def list_facts(self) -> List[Dict[str, str]]:
        with self._lock:
            return [dict(item) for item in self._facts]

    def format_for_prompt(self) -> str:
        facts = self.list_facts()
        if not facts:
            return ""
        return "\n".join(f"[{item['id']}] {item['text']}" for item in facts[-15:])

    @staticmethod
    def extract_facts_from_user_message(user_text: str) -> Optional[str]:
        cleaned = re.sub(r"\s+", " ", (user_text or "").strip())
        if len(cleaned) < 8 or len(cleaned) > 180:
            return None
        patterns = (
            r"^remember\s+(?:that\s+)?(.+)$",
            r"^note\s+(?:that\s+)?(i\s+am\s+.+|my\s+.+)$",
        )
        for pat in patterns:
            matched = re.match(pat, cleaned, flags=re.IGNORECASE)
            if matched and not re.search(r"\b(?:remind|minutes?|hours?|tomorrow|at\s+\d)\b", cleaned, flags=re.IGNORECASE):
                return matched.group(1).strip()
        return None


# ---------------------------------------------------------------------------
# Fast screen capture + Draw-to-Ask highlight compositing
# ---------------------------------------------------------------------------


class ScreenCaptureService:
    """
    Captures the primary display with MSS in 1:1 logical coordinates, draws edge coordinate
    rulers for accurate vision placement, extracts 2x zoomed lasso crops, snaps underlines
    to real text baselines, and measures post-scroll visual changes.
    """

    _last_raw_logical_image: Optional[Image.Image] = None
    _last_lasso_crop_b64: str = ""
    _last_lasso_bbox: Optional[Tuple[int, int, int, int]] = None
    _used_underline_regions: List[Tuple[float, float, float]] = []

    @classmethod
    def reset_underline_baselines(cls) -> None:
        """Clear tracked per-turn underline baselines when starting a new turn or screenshot."""
        cls._used_underline_regions = []
        cls._last_lasso_bbox = None

    @classmethod
    def _compute_glyph_column_mask(cls, band_rows_slice: np.ndarray) -> np.ndarray:
        """Return a boolean mask over columns that contain actual multi-edge text glyphs rather than flat margins."""
        num_cols = int(band_rows_slice.shape[1]) if band_rows_slice.ndim == 2 else 0
        if num_cols < 4:
            return np.zeros((max(0, num_cols),), dtype=bool)
        row_count = int(band_rows_slice.shape[0])
        horiz_diff_1 = np.abs(np.diff(band_rows_slice, axis=1)) > 14.0
        horiz_edges = np.zeros((row_count, num_cols), dtype=bool)
        horiz_edges[:, :-1] |= horiz_diff_1
        horiz_edges[:, 1:] |= horiz_diff_1
        if num_cols >= 3:
            horiz_diff_2 = np.abs(band_rows_slice[:, 2:] - band_rows_slice[:, :-2]) > 20.0
            horiz_edges[:, :-2] |= horiz_diff_2
            horiz_edges[:, 2:] |= horiz_diff_2
        edge_rows_per_col = np.sum(horiz_edges, axis=0)
        raw_edge_cols = edge_rows_per_col >= 2

        # Count neighboring edge columns in [col_idx - 10, col_idx + 10] using prefix sums to reject isolated 1px borders.
        prefix_edges = np.concatenate(([0], np.cumsum(raw_edge_cols.astype(np.int32))))
        col_indices = np.arange(num_cols, dtype=np.int32)
        left_bounds = np.maximum(0, col_indices - 10)
        right_bounds = np.minimum(num_cols, col_indices + 11)
        neighbor_edge_counts = prefix_edges[right_bounds] - prefix_edges[left_bounds]

        # Keep columns within +/-2 px of an edge inside a multi-edge glyph cluster and non-flat 2D local texture.
        near_edge = raw_edge_cols.copy()
        for shift_px in (1, 2):
            near_edge[shift_px:] |= raw_edge_cols[:-shift_px]
            near_edge[:-shift_px] |= raw_edge_cols[shift_px:]

        band_col_mean = np.mean(band_rows_slice, axis=0)
        band_col_sq_mean = np.mean(band_rows_slice * band_rows_slice, axis=0)
        prefix_sum = np.concatenate(([0.0], np.cumsum(band_col_mean)))
        prefix_sq = np.concatenate(([0.0], np.cumsum(band_col_sq_mean)))
        std_left = np.maximum(0, col_indices - 8)
        std_right = np.minimum(num_cols, col_indices + 9)
        window_counts = np.maximum(1.0, (std_right - std_left).astype(np.float64))
        local_mean = (prefix_sum[std_right] - prefix_sum[std_left]) / window_counts
        local_sq_mean = (prefix_sq[std_right] - prefix_sq[std_left]) / window_counts
        local_variance = np.maximum(0.0, local_sq_mean - (local_mean * local_mean))
        local_std = np.sqrt(local_variance)

        return near_edge & (neighbor_edge_counts >= 4) & (local_std >= 10.0)

    @classmethod
    def get_last_lasso_crop_b64(cls) -> str:
        return cls._last_lasso_crop_b64

    @staticmethod
    def _grab_raw_pil_screenshot() -> Optional[Image.Image]:
        try:
            with mss.MSS() as sct:
                monitor_info = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                raw_grab = sct.grab(monitor_info)
                return Image.frombytes("RGB", raw_grab.size, raw_grab.bgra, "raw", "BGRX")
        except Exception:
            pass
        try:
            from PIL import ImageGrab

            return ImageGrab.grab()
        except Exception:
            return None

    @classmethod
    def capture_logical_snapshot_image(
        cls,
        logical_screen_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Image.Image]:
        """Grabs a clean RGB screenshot resized to 1:1 logical screen dimensions."""
        raw_shot = cls._grab_raw_pil_screenshot()
        if raw_shot is None:
            return None
        logical_width, logical_height = logical_screen_size or raw_shot.size
        logical_width = max(1, int(logical_width))
        logical_height = max(1, int(logical_height))
        if raw_shot.size != (logical_width, logical_height):
            return raw_shot.resize((logical_width, logical_height), Image.Resampling.LANCZOS)
        return raw_shot

    @classmethod
    def capture_screen_with_highlight(
        cls,
        highlight_points: Optional[List[Tuple[float, float]]] = None,
        highlight_color_hex: str = jarvis_theme.JARVIS_BLUE_HEX,
        logical_screen_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, int, int, float, float]:
        """
        Returns `(jpeg_base64, logical_width, logical_height, 1.0, 1.0)` so vision model
        coordinates map 1:1 onto Qt logical screen pixels without scaling drift.
        Returns empty `jpeg_base64` gracefully if desktop capture is blocked (lock screen / UAC).
        """
        logical_width, logical_height = logical_screen_size or (1920, 1080)
        logical_width = max(1, int(logical_width))
        logical_height = max(1, int(logical_height))
        cls._last_lasso_crop_b64 = ""
        cls.reset_underline_baselines()

        full_res_shot = cls._grab_raw_pil_screenshot()
        if full_res_shot is None:
            return "", logical_width, logical_height, 1.0, 1.0

        logical_width, logical_height = logical_screen_size or full_res_shot.size
        logical_width = max(1, int(logical_width))
        logical_height = max(1, int(logical_height))

        if full_res_shot.size != (logical_width, logical_height):
            logical_shot = full_res_shot.resize((logical_width, logical_height), Image.Resampling.LANCZOS)
        else:
            logical_shot = full_res_shot.copy()

        # Store clean un-annotated logical frame for baseline snapping & scroll diff detection
        cls._last_raw_logical_image = logical_shot.copy()

        annotated_shot = _paint_coordinate_ruler_on_screenshot(logical_shot)

        cls._last_lasso_bbox = None
        if highlight_points and len(highlight_points) >= 2:
            lasso_xs = [float(pt_x) for pt_x, _pt_y in highlight_points]
            lasso_ys = [float(pt_y) for _pt_x, pt_y in highlight_points]
            cls._last_lasso_bbox = (
                max(0, int(round(min(lasso_xs)))),
                max(0, int(round(min(lasso_ys)))),
                min(int(logical_width), int(round(max(lasso_xs)))),
                min(int(logical_height), int(round(max(lasso_ys)))),
            )
            annotated_shot = _composite_highlight_stroke(
                screenshot=annotated_shot,
                highlight_points=highlight_points,
                highlight_color_hex=highlight_color_hex,
                logical_screen_size=(logical_width, logical_height),
            )
            cls._last_lasso_crop_b64 = _extract_zoomed_lasso_crop_b64(
                full_res_screenshot=full_res_shot,
                highlight_points=highlight_points,
                logical_screen_size=(logical_width, logical_height),
            )

        output_buffer = io.BytesIO()
        annotated_shot.save(output_buffer, format="JPEG", quality=88)
        jpeg_b64 = base64.b64encode(output_buffer.getvalue()).decode("ascii")
        return jpeg_b64, logical_width, logical_height, 1.0, 1.0

    @classmethod
    def compute_screen_difference_ratio(
        cls,
        previous_image: Optional[Image.Image],
        current_image: Optional[Image.Image],
    ) -> float:
        """
        Compares the scrollable center region of two screenshots and returns the normalized
        mean grayscale pixel difference (`0.0..1.0`) to detect when scrolling hit the bottom.
        """
        if previous_image is None or current_image is None:
            return 1.0
        try:
            prev_small = previous_image.convert("L").resize((160, 90), Image.Resampling.BILINEAR)
            curr_small = current_image.convert("L").resize((160, 90), Image.Resampling.BILINEAR)
            prev_arr = np.asarray(prev_small, dtype=np.float32)
            curr_arr = np.asarray(curr_small, dtype=np.float32)
            # Exclude top notch area and left/right margins where floating cards may sit
            center_prev = prev_arr[14:84, 30:130]
            center_curr = curr_arr[14:84, 30:130]
            return float(np.mean(np.abs(center_curr - center_prev)) / 255.0)
        except Exception:
            return 1.0

    @classmethod
    def _detect_single_line_bands(
        cls,
        strip_gray: np.ndarray,
        bg_luminance: float,
        min_active_pixels: int = 8,
    ) -> List[Tuple[int, int]]:
        """
        Detect horizontal text bands in a grayscale strip and recursively split any merged
        multi-line band taller than 24px at its internal minimum-ink valley.
        """
        if strip_gray.shape[1] >= 2:
            pixel_deltas = np.abs(np.diff(strip_gray, axis=1)) > 15.0
        else:
            pixel_deltas = np.abs(strip_gray - bg_luminance) > 18.0
        row_counts = np.sum(pixel_deltas, axis=1).astype(np.float32)
        active_rows = row_counts >= float(max(6, min_active_pixels))
        if not np.any(active_rows):
            return []

        raw_bands: List[Tuple[int, int]] = []
        band_start: Optional[int] = None
        for row_offset, is_active in enumerate(active_rows):
            if is_active and band_start is None:
                band_start = row_offset
            elif not is_active and band_start is not None:
                if (row_offset - band_start) >= 3:
                    raw_bands.append((band_start, row_offset - 1))
                band_start = None
        if band_start is not None and (len(active_rows) - band_start) >= 3:
            raw_bands.append((band_start, len(active_rows) - 1))

        split_bands: List[Tuple[int, int]] = []
        pending_bands = list(raw_bands)
        while pending_bands:
            curr_start, curr_end = pending_bands.pop(0)
            curr_height = curr_end - curr_start + 1
            if curr_height <= 24:
                split_bands.append((curr_start, curr_end))
                continue
            inner_start = curr_start + 5
            inner_end = curr_end - 4
            if inner_end <= inner_start:
                split_bands.append((curr_start, curr_end))
                continue
            inner_slice = row_counts[inner_start:inner_end]
            peak_count = float(np.max(row_counts[curr_start : curr_end + 1]))
            valley_rel = int(np.argmin(inner_slice))
            valley_idx = inner_start + valley_rel
            valley_count = float(row_counts[valley_idx])
            if peak_count > 0.0 and valley_count <= peak_count * 0.65:
                upper_end = valley_idx - 1
                lower_start = valley_idx + 1
                if (upper_end - curr_start) >= 3 and (curr_end - lower_start) >= 3:
                    pending_bands.insert(0, (lower_start, curr_end))
                    pending_bands.insert(0, (curr_start, upper_end))
                    continue
            split_bands.append((curr_start, curr_end))
        return split_bands

    @classmethod
    def snap_mark_to_screen_text(
        cls,
        command: jarvis_response_parser.DrawingCommand,
        hands_exe: str = "",
    ) -> jarvis_response_parser.DrawingCommand:
        """Snaps a labeled mark (underline, target, strike, check, cross) to live OCR/UIA text bounds."""
        cleaned_label = (command.label or "").strip()
        if not cleaned_label or command.kind not in ("underline", "strike", "target", "check", "cross", "ring"):
            return command
        bounds = RoamiesHandsService.find_text_bounds(cleaned_label, spec="front", allow_scroll=False, hands_exe=hands_exe)
        if bounds is None:
            return command
        bx, by, bw, bh = bounds
        if bw < 8 or bh < 6:
            return command
        if command.kind == "underline":
            return jarvis_response_parser.DrawingCommand(
                kind="underline",
                x=float(bx),
                y=float(by + bh + 3),
                width=float(max(28, bw)),
                label=command.label,
                tone=command.tone,
                sentence_index=command.sentence_index,
            )
        if command.kind == "strike":
            return jarvis_response_parser.DrawingCommand(
                kind="strike",
                x=float(bx),
                y=float(by + bh * 0.52),
                width=float(max(24, bw)),
                label=command.label,
                tone=command.tone,
                sentence_index=command.sentence_index,
            )
        if command.kind in ("target", "ring"):
            return jarvis_response_parser.DrawingCommand(
                kind=command.kind,
                x=float(bx + bw * 0.5),
                y=float(by + bh * 0.5),
                radius=float(max(18.0, min(64.0, max(bw, bh) * 0.55))),
                label=command.label,
                tone=command.tone,
                sentence_index=command.sentence_index,
            )
        return jarvis_response_parser.DrawingCommand(
            kind=command.kind,
            x=float(max(14.0, bx - 16.0)),
            y=float(by + bh * 0.5),
            label=command.label,
            tone=command.tone,
            sentence_index=command.sentence_index,
        )

    @classmethod
    def snap_drawing_command_to_content(
        cls,
        command: jarvis_response_parser.DrawingCommand,
        reference_image: Optional[Image.Image] = None,
    ) -> jarvis_response_parser.DrawingCommand:
        """
        Snap an underline or highlight cleanly beneath its exact single line of text:
        - Splits merged multi-line vertical bands on tight lists so adjacent lines stay separate.
        - Avoids re-snapping onto an already-underlined overlapping baseline in the same turn.
        - Measures contiguous horizontal ink runs (bridging word gaps <= 24px) so underlines
          never overshoot past the last character of a shorter line into empty background.
        """
        source_image = reference_image or cls._last_raw_logical_image
        if source_image is None or command.kind not in ("underline", "highlight"):
            return command

        try:
            img_width, img_height = source_image.size
            req_width = max(24.0, float(command.width))
            req_left_px = max(0, int(round(command.x)))
            req_right_px = min(img_width, int(round(command.x + req_width)))
            center_y_px = int(round(command.y))

            lasso_active = False
            lasso_x0, lasso_y0, lasso_x1, lasso_y1 = 0, 0, img_width, img_height
            if cls._last_lasso_bbox is not None and command.kind == "underline":
                lasso_x0, lasso_y0, lasso_x1, lasso_y1 = cls._last_lasso_bbox
                if (lasso_x1 - lasso_x0) >= 60 and (lasso_y1 - lasso_y0) >= 24:
                    lasso_active = True
                    padded_left = max(0, lasso_x0 - 16)
                    padded_right = min(img_width, lasso_x1 + 16)
                    req_left_px = min(max(req_left_px, padded_left), max(padded_left, padded_right - 36))
                    req_right_px = min(padded_right, max(req_left_px + 24, req_right_px))
                    center_y_px = min(
                        max(center_y_px, max(0, lasso_y0 - 16)),
                        min(img_height - 1, lasso_y1 + 20),
                    )
                    req_width = max(24.0, float(req_right_px - req_left_px))

            scan_left_px = max(0, req_left_px - 160)
            right_expand_cap = int(round(min(96.0, req_width * 0.40)))
            scan_right_px = min(img_width, req_right_px + right_expand_cap)
            if lasso_active:
                scan_left_px = max(max(0, lasso_x0 - 24), scan_left_px)
                scan_right_px = min(min(img_width, lasso_x1 + 24), max(scan_left_px + 24, scan_right_px))
            if (scan_right_px - scan_left_px) < 16:
                return command

            rescue_left_px = max(0, lasso_x0 - 20) if lasso_active else 24
            rescue_right_px = min(img_width, lasso_x1 + 20) if lasso_active else max(48, img_width - 24)

            def _is_region_already_underlined(cand_y: float, x_left: float, x_right: float) -> bool:
                for used_y, used_left, used_right in cls._used_underline_regions:
                    if abs(cand_y - used_y) < 13.0:
                        overlap_px = min(x_right, used_right) - max(x_left, used_left)
                        if overlap_px > 48.0:
                            return True
                return False

            def _find_snapped_band(
                search_left_px: int,
                search_right_px: int,
                search_margin_px: int,
                skip_used: bool,
                require_glyphs: bool,
            ) -> Optional[Tuple[int, float, np.ndarray, float, int]]:
                if (search_right_px - search_left_px) < 16:
                    return None
                top_px = max(0, center_y_px - search_margin_px)
                bottom_px = min(img_height, center_y_px + search_margin_px)
                if (bottom_px - top_px) < 10:
                    return None
                strip_gray = np.asarray(
                    source_image.crop((search_left_px, top_px, search_right_px, bottom_px)).convert("L"),
                    dtype=np.float32,
                )
                bg_luminance = float(np.median(strip_gray))
                text_bands = cls._detect_single_line_bands(
                    strip_gray=strip_gray,
                    bg_luminance=bg_luminance,
                    min_active_pixels=8,
                )
                if not text_bands:
                    return None

                eligible_bands: List[Tuple[Tuple[int, int], float, np.ndarray]] = []
                for cand_band in text_bands:
                    cand_y = float(top_px + cand_band[1] + 4.0)
                    if abs(cand_y - float(center_y_px)) > float(search_margin_px):
                        continue
                    band_slice = strip_gray[cand_band[0] : cand_band[1] + 1, :]
                    if require_glyphs:
                        band_glyph_mask = cls._compute_glyph_column_mask(band_slice)
                        if int(np.sum(band_glyph_mask)) < 6:
                            continue
                    if skip_used and _is_region_already_underlined(cand_y, float(req_left_px), float(req_right_px)):
                        continue
                    eligible_bands.append((cand_band, cand_y, band_slice))

                if not eligible_bands:
                    return None

                chosen_band, chosen_y, chosen_slice = min(
                    eligible_bands,
                    key=lambda item: abs(item[1] - float(center_y_px)),
                )
                return (
                    chosen_band[0],
                    chosen_y,
                    chosen_slice,
                    bg_luminance,
                    search_left_px,
                )

            band_hit = (
                _find_snapped_band(scan_left_px, scan_right_px, 20, skip_used=True, require_glyphs=(command.kind == "underline"))
                or _find_snapped_band(scan_left_px, scan_right_px, 44, skip_used=True, require_glyphs=(command.kind == "underline"))
                or (
                    _find_snapped_band(rescue_left_px, rescue_right_px, 48, skip_used=True, require_glyphs=True)
                    if command.kind == "underline"
                    else None
                )
                or _find_snapped_band(scan_left_px, scan_right_px, 20, skip_used=False, require_glyphs=(command.kind == "underline"))
                or _find_snapped_band(scan_left_px, scan_right_px, 44, skip_used=False, require_glyphs=(command.kind == "underline"))
                or (
                    _find_snapped_band(rescue_left_px, rescue_right_px, 48, skip_used=False, require_glyphs=True)
                    if command.kind == "underline"
                    else None
                )
            )
            if band_hit is None:
                return command
            _band_top_offset, snapped_y, band_rows_slice, _bg_luminance, active_scan_left_px = band_hit

            snapped_x = command.x
            snapped_width = command.width
            if command.kind == "underline":
                col_foreground = cls._compute_glyph_column_mask(band_rows_slice)
                if int(np.sum(col_foreground)) >= 6:
                    # Bridge inter-word spaces (<= 24px) without bridging blank margins (> 24px)
                    bridged_ink = col_foreground.copy()
                    false_run_start: Optional[int] = None
                    saw_true_before = False
                    for col_idx, has_ink in enumerate(col_foreground):
                        if has_ink:
                            if saw_true_before and false_run_start is not None:
                                if (col_idx - false_run_start) <= 24:
                                    bridged_ink[false_run_start:col_idx] = True
                            saw_true_before = True
                            false_run_start = None
                        elif saw_true_before and false_run_start is None:
                            false_run_start = col_idx

                    segments: List[Tuple[int, int]] = []
                    seg_start: Optional[int] = None
                    for col_idx, is_filled in enumerate(bridged_ink):
                        if is_filled and seg_start is None:
                            seg_start = col_idx
                        elif not is_filled and seg_start is not None:
                            segments.append((seg_start, col_idx - 1))
                            seg_start = None
                    if seg_start is not None:
                        segments.append((seg_start, len(bridged_ink) - 1))

                    req_rel_left = max(0, req_left_px - active_scan_left_px)
                    req_rel_right = max(req_rel_left + 12, req_right_px - active_scan_left_px)
                    valid_segments = [seg_item for seg_item in segments if (seg_item[1] - seg_item[0] + 1) >= 12]
                    if valid_segments:
                        overlapping = [
                            seg_item
                            for seg_item in valid_segments
                            if min(seg_item[1] + 1, req_rel_right) - max(seg_item[0], req_rel_left) > 0
                        ]
                        if overlapping:
                            chosen_seg = max(
                                overlapping,
                                key=lambda seg_item: min(seg_item[1] + 1, req_rel_right) - max(seg_item[0], req_rel_left),
                            )
                        else:
                            chosen_seg = min(
                                valid_segments,
                                key=lambda seg_item: abs(
                                    ((seg_item[0] + seg_item[1]) * 0.5) - ((req_rel_left + req_rel_right) * 0.5)
                                ),
                            )
                        true_cols_in_seg = np.where(col_foreground[chosen_seg[0] : chosen_seg[1] + 1])[0]
                        if true_cols_in_seg.size >= 6:
                            seg_first_ink = chosen_seg[0] + int(true_cols_in_seg[0])
                            seg_last_ink = chosen_seg[0] + int(true_cols_in_seg[-1])
                            max_allowed_right_rel = req_rel_right + right_expand_cap
                            clipped_last_ink = min(seg_last_ink, max_allowed_right_rel)
                            trimmed_left = active_scan_left_px + seg_first_ink
                            trimmed_right = active_scan_left_px + clipped_last_ink + 1
                            trimmed_span = float(trimmed_right - trimmed_left)
                            if not overlapping and trimmed_span > 420.0:
                                req_mid_px = 0.5 * float(req_left_px + req_right_px)
                                keep_span = int(round(max(240.0, req_width)))
                                if abs(req_mid_px - float(trimmed_right)) <= abs(req_mid_px - float(trimmed_left)):
                                    window_left = max(trimmed_left, trimmed_right - keep_span)
                                    window_right = trimmed_right
                                else:
                                    window_left = trimmed_left
                                    window_right = min(trimmed_right, trimmed_left + keep_span)
                                rel_win_l = max(0, window_left - active_scan_left_px)
                                rel_win_r = min(col_foreground.shape[0], window_right - active_scan_left_px)
                                window_cols = np.where(col_foreground[rel_win_l:rel_win_r])[0]
                                if window_cols.size >= 6:
                                    trimmed_left = active_scan_left_px + rel_win_l + int(window_cols[0])
                                    trimmed_right = active_scan_left_px + rel_win_l + int(window_cols[-1]) + 1
                                    trimmed_span = float(trimmed_right - trimmed_left)
                            if trimmed_span >= 18.0:
                                snapped_x = max(0.0, float(trimmed_left - 2))
                                snapped_width = trimmed_span + 4.0

                cls._used_underline_regions.append((snapped_y, snapped_x, snapped_x + snapped_width))

            return jarvis_response_parser.DrawingCommand(
                kind=command.kind,
                x=snapped_x,
                y=snapped_y,
                width=snapped_width,
                height=command.height,
                radius=command.radius,
                label=command.label,
                header=command.header,
                body=command.body,
                tone=command.tone,
                sentence_index=command.sentence_index,
            )
        except Exception:
            return command

    @classmethod
    def _measure_content_row_range(
        cls,
        source_image: Image.Image,
    ) -> Tuple[int, int, List[int]]:
        """Return `(img_height, lowest_text_baseline_y, all_band_baselines)` in logical pixels."""
        img_width, img_height = source_image.size
        roi_left = max(32, int(img_width * 0.12))
        roi_right = min(img_width - 32, int(img_width * 0.88))
        roi_top = max(56, int(img_height * 0.08))
        roi_bottom = min(img_height - 40, int(img_height * 0.92))
        if (roi_right - roi_left) < 100 or (roi_bottom - roi_top) < 60:
            return img_height, 0, []
        roi_gray = np.asarray(
            source_image.crop((roi_left, roi_top, roi_right, roi_bottom)).convert("L"),
            dtype=np.float32,
        )
        bg_luma = float(np.median(roi_gray))
        bands = cls._detect_single_line_bands(
            strip_gray=roi_gray,
            bg_luminance=bg_luma,
            min_active_pixels=max(12, int((roi_right - roi_left) * 0.03)),
        )
        baselines = [roi_top + band[1] + 4 for band in bands if (band[1] - band[0] + 1) >= 6]
        lowest_y = max(baselines) if baselines else 0
        return img_height, lowest_y, baselines

    @classmethod
    def has_unread_content_below_explained_lines(cls) -> bool:
        """
        Determine whether text clearly continues near/below the viewport bottom edge beyond
        what Jarvis already underlined on the current screen.
        """
        source_image = cls._last_raw_logical_image
        if source_image is None:
            return False
        try:
            img_height, lowest_y, baselines = cls._measure_content_row_range(source_image)
            if not baselines:
                return False
            near_bottom_count = sum(1 for base_y in baselines if base_y >= int(img_height * 0.80))
            if near_bottom_count >= 2:
                return True
            if cls._used_underline_regions:
                lowest_underlined_y = max(region[0] for region in cls._used_underline_regions)
                unread_bands_below = [base_y for base_y in baselines if base_y > lowest_underlined_y + 36.0]
                if len(unread_bands_below) >= 2 and lowest_y >= int(img_height * 0.78):
                    return True
            return False
        except Exception:
            return False

    @classmethod
    def recommend_adaptive_scroll_ticks(
        cls,
        direction: str = "down",
        requested_ticks: int = 0,
        user_prompt: str = "",
        force_if_needed: bool = False,
    ) -> int:
        """
        Compute requirement-based scroll wheel ticks (`0` means no scroll required).
        """
        normalized_prompt = (user_prompt or "").lower().strip()
        explicit_match = re.search(
            r"\bscroll\s+(?:down|up)?\s*(?:by|about)?\s*(\d{1,2})\s*(?:lines?|ticks?|steps?)\b",
            normalized_prompt,
        )
        if explicit_match:
            return int(max(1, min(14, int(explicit_match.group(1)))))

        if any(phrase in normalized_prompt for phrase in ("little", "slightly", "small scroll", "few lines", "bit down")):
            return 3
        if any(phrase in normalized_prompt for phrase in ("half page", "next section", "medium scroll", "paragraph")):
            return 5
        if any(phrase in normalized_prompt for phrase in ("full page", "all the way", "whole screen", "a lot", "far down")):
            return 9

        source_image = cls._last_raw_logical_image
        if source_image is None:
            return int(max(0, min(12, requested_ticks))) if requested_ticks > 0 else (5 if force_if_needed else 0)

        try:
            img_height, lowest_text_y, baselines = cls._measure_content_row_range(source_image)
            if requested_ticks <= 0:
                if not force_if_needed or not cls.has_unread_content_below_explained_lines():
                    return 0

            if not baselines:
                return int(max(2, min(10, requested_ticks or 5)))

            if direction == "up":
                return int(max(2, min(10, requested_ticks or 5)))

            if cls._used_underline_regions:
                lowest_explained_y = max(region[0] for region in cls._used_underline_regions)
                unread_baselines = [base_y for base_y in baselines if base_y > lowest_explained_y + 28.0]
                if unread_baselines:
                    next_unread_y = float(min(unread_baselines))
                    target_top_y = float(img_height) * 0.24
                    scroll_delta_px = max(90.0, next_unread_y - target_top_y)
                else:
                    target_top_y = float(img_height) * 0.24
                    scroll_delta_px = max(120.0, float(lowest_text_y) - target_top_y)
            else:
                scroll_delta_px = max(140.0, float(lowest_text_y) * 0.48)

            geometry_ticks = int(max(2, min(10, round(scroll_delta_px / 58.0))))
            if requested_ticks > 0 and requested_ticks != 6:
                blended = int(round((requested_ticks * 0.6) + (geometry_ticks * 0.4)))
                return int(max(1, min(12, blended)))
            return geometry_ticks
        except Exception:
            return int(max(2, min(10, requested_ticks or 5)))

    @classmethod
    def detect_visual_text_anchors(
        cls,
        screen_width: int,
        screen_height: int,
        count: int = 2,
    ) -> List[Tuple[float, float, float]]:
        """
        Detects up to `count` prominent horizontal text bands `(x, baseline_y, span_width)`
        from `_last_raw_logical_image` so fallback underlines land accurately on screen text.
        """
        default_anchors = [
            (round(screen_width * 0.28, 1), round(screen_height * 0.28, 1), min(360.0, screen_width * 0.26)),
            (round(screen_width * 0.28, 1), round(screen_height * 0.44, 1), min(320.0, screen_width * 0.24)),
        ]
        source_image = cls._last_raw_logical_image
        if source_image is None:
            return default_anchors[:count]

        try:
            img_width, img_height = source_image.size
            roi_left = max(40, int(img_width * 0.22))
            roi_right = min(img_width - 40, int(img_width * 0.88))
            roi_top = max(60, int(img_height * 0.14))
            roi_bottom = min(img_height - 90, int(img_height * 0.78))
            if (roi_right - roi_left) < 120 or (roi_bottom - roi_top) < 80:
                return default_anchors[:count]

            roi_gray = np.asarray(
                source_image.crop((roi_left, roi_top, roi_right, roi_bottom)).convert("L"),
                dtype=np.float32,
            )
            bg_luma = float(np.median(roi_gray))
            row_activity = np.mean(np.abs(roi_gray - bg_luma) > 18.0, axis=1)
            active_rows = row_activity > 0.035

            candidate_bands: List[Tuple[int, int, float]] = []
            band_start: Optional[int] = None
            for row_idx, is_active in enumerate(active_rows):
                if is_active and band_start is None:
                    band_start = row_idx
                elif not is_active and band_start is not None:
                    band_height = row_idx - band_start
                    if 7 <= band_height <= 36:
                        peak_density = float(np.max(row_activity[band_start:row_idx]))
                        candidate_bands.append((band_start, row_idx - 1, peak_density))
                    band_start = None
            if band_start is not None:
                band_height = len(active_rows) - band_start
                if 7 <= band_height <= 36:
                    peak_density = float(np.max(row_activity[band_start:]))
                    candidate_bands.append((band_start, len(active_rows) - 1, peak_density))

            if not candidate_bands:
                return default_anchors[:count]

            selected_anchors: List[Tuple[float, float, float]] = []
            last_picked_y = -999.0
            for band_top, band_bottom, _density in candidate_bands:
                baseline_y = float(roi_top + band_bottom + 5.0)
                if baseline_y - last_picked_y < 52.0:
                    continue
                band_slice = roi_gray[band_top : band_bottom + 1, :]
                col_activity = np.mean(np.abs(band_slice - bg_luma) > 18.0, axis=0)
                active_cols = np.where(col_activity > 0.08)[0]
                if active_cols.size < 18:
                    continue
                first_col = int(active_cols[0])
                anchor_x = float(roi_left + first_col)
                span_right = int(active_cols[-1])
                raw_span = float(span_right - first_col)
                span_width = max(140.0, min(420.0, raw_span))
                selected_anchors.append((anchor_x, baseline_y, span_width))
                last_picked_y = baseline_y
                if len(selected_anchors) >= count:
                    break

            return selected_anchors if selected_anchors else default_anchors[:count]
        except Exception:
            return default_anchors[:count]


def _paint_coordinate_ruler_on_screenshot(screenshot: Image.Image) -> Image.Image:
    """
    Overlays unobtrusive top/left 100px rulers and faint 200px alignment grid hairlines onto the
    vision model's copy so returned drawing coordinates match screen content with high precision.
    """
    rgba_shot = screenshot.convert("RGBA")
    ruler_layer = Image.new("RGBA", rgba_shot.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ruler_layer)
    width_px, height_px = rgba_shot.size
    small_font = ImageFont.load_default()

    # Subtle 200px grid hairlines
    for grid_x in range(200, width_px, 200):
        draw.line([(grid_x, 0), (grid_x, height_px)], fill=(51, 136, 255, 28), width=1)
    for grid_y in range(200, height_px, 200):
        draw.line([(0, grid_y), (width_px, grid_y)], fill=(51, 136, 255, 28), width=1)

    # Top and left ruler strips with 100px tick marks
    draw.rectangle([0, 0, width_px, 16], fill=(12, 16, 24, 150))
    draw.rectangle([0, 0, 34, height_px], fill=(12, 16, 24, 150))

    for tick_x in range(100, width_px, 100):
        draw.line([(tick_x, 0), (tick_x, 15)], fill=(120, 210, 255, 210), width=1)
        draw.text((tick_x + 3, 2), str(tick_x), fill=(210, 238, 255, 235), font=small_font)

    for tick_y in range(100, height_px, 100):
        draw.line([(0, tick_y), (32, tick_y)], fill=(120, 210, 255, 210), width=1)
        draw.text((2, tick_y + 2), str(tick_y), fill=(210, 238, 255, 235), font=small_font)

    return Image.alpha_composite(rgba_shot, ruler_layer).convert("RGB")


def _extract_zoomed_lasso_crop_b64(
    full_res_screenshot: Image.Image,
    highlight_points: List[Tuple[float, float]],
    logical_screen_size: Tuple[int, int],
) -> str:
    """Crops and 2x-supersamples the user's lasso bounding box from the full-res screenshot."""
    if len(highlight_points) < 2:
        return ""
    logical_w, logical_h = max(1, logical_screen_size[0]), max(1, logical_screen_size[1])
    phys_scale_x = float(full_res_screenshot.width) / float(logical_w)
    phys_scale_y = float(full_res_screenshot.height) / float(logical_h)

    x_values = [point_pair[0] for point_pair in highlight_points]
    y_values = [point_pair[1] for point_pair in highlight_points]
    pad_logical = 48.0
    crop_left = max(0, int(round((min(x_values) - pad_logical) * phys_scale_x)))
    crop_top = max(0, int(round((min(y_values) - pad_logical) * phys_scale_y)))
    crop_right = min(full_res_screenshot.width, int(round((max(x_values) + pad_logical) * phys_scale_x)))
    crop_bottom = min(full_res_screenshot.height, int(round((max(y_values) + pad_logical) * phys_scale_y)))

    if (crop_right - crop_left) < 24 or (crop_bottom - crop_top) < 24:
        return ""

    cropped = full_res_screenshot.crop((crop_left, crop_top, crop_right, crop_bottom))
    target_w = min(1024, max(320, int(cropped.width * 1.6)))
    target_h = max(160, int(round(cropped.height * (target_w / float(max(1, cropped.width))))))
    if target_h > 960:
        shrink_ratio = 960.0 / float(target_h)
        target_h = 960
        target_w = max(160, int(round(target_w * shrink_ratio)))
    zoomed = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Stamp 1:1 full-screen logical coordinates along the zoomed crop rulers so coordinates match Image 1 directly.
    logical_crop_left = float(crop_left) / phys_scale_x
    logical_crop_top = float(crop_top) / phys_scale_y
    logical_crop_right = float(crop_right) / phys_scale_x
    logical_crop_bottom = float(crop_bottom) / phys_scale_y
    logical_crop_w = max(1.0, logical_crop_right - logical_crop_left)
    logical_crop_h = max(1.0, logical_crop_bottom - logical_crop_top)

    zoomed_rgba = zoomed.convert("RGBA")
    ruler_layer = Image.new("RGBA", zoomed_rgba.size, (0, 0, 0, 0))
    ruler_draw = ImageDraw.Draw(ruler_layer)
    small_font = ImageFont.load_default()
    ruler_draw.rectangle([0, 0, target_w, 16], fill=(12, 16, 24, 175))
    ruler_draw.rectangle([0, 0, 38, target_h], fill=(12, 16, 24, 175))

    step_x = 50 if logical_crop_w < 450.0 else 100
    first_tick_x = (int(math.ceil(logical_crop_left / float(step_x))) * step_x)
    tick_x = max(step_x, first_tick_x)
    while float(tick_x) < logical_crop_right:
        draw_x = int(round(((float(tick_x) - logical_crop_left) / logical_crop_w) * float(target_w)))
        if 42 <= draw_x < (target_w - 24):
            ruler_draw.line([(draw_x, 0), (draw_x, target_h)], fill=(51, 136, 255, 24), width=1)
            ruler_draw.line([(draw_x, 0), (draw_x, 15)], fill=(120, 210, 255, 225), width=1)
            ruler_draw.text((draw_x + 3, 2), str(tick_x), fill=(220, 242, 255, 245), font=small_font)
        tick_x += step_x

    step_y = 50 if logical_crop_h < 350.0 else 100
    first_tick_y = (int(math.ceil(logical_crop_top / float(step_y))) * step_y)
    tick_y = max(step_y, first_tick_y)
    while float(tick_y) < logical_crop_bottom:
        draw_y = int(round(((float(tick_y) - logical_crop_top) / logical_crop_h) * float(target_h)))
        if 20 <= draw_y < (target_h - 12):
            ruler_draw.line([(0, draw_y), (target_w, draw_y)], fill=(51, 136, 255, 24), width=1)
            ruler_draw.line([(0, draw_y), (36, draw_y)], fill=(120, 210, 255, 225), width=1)
            ruler_draw.text((2, draw_y + 2), str(tick_y), fill=(220, 242, 255, 245), font=small_font)
        tick_y += step_y

    zoomed = Image.alpha_composite(zoomed_rgba, ruler_layer).convert("RGB")
    output_buffer = io.BytesIO()
    zoomed.save(output_buffer, format="JPEG", quality=92)
    return base64.b64encode(output_buffer.getvalue()).decode("ascii")


def _composite_highlight_stroke(
    screenshot: Image.Image,
    highlight_points: List[Tuple[float, float]],
    highlight_color_hex: str,
    logical_screen_size: Tuple[int, int],
) -> Image.Image:
    logical_w, logical_h = max(1, logical_screen_size[0]), max(1, logical_screen_size[1])
    img_scale_x = float(screenshot.width) / float(logical_w)
    img_scale_y = float(screenshot.height) / float(logical_h)

    rgba_shot = screenshot.convert("RGBA")
    overlay_layer = Image.new("RGBA", rgba_shot.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)

    hex_clean = highlight_color_hex.lstrip("#")
    if len(hex_clean) == 6:
        red = int(hex_clean[0:2], 16)
        green = int(hex_clean[2:4], 16)
        blue = int(hex_clean[4:6], 16)
    else:
        red, green, blue = 51, 128, 255

    scaled_points = [(pt_x * img_scale_x, pt_y * img_scale_y) for pt_x, pt_y in highlight_points]
    stroke_width = max(6, int(round(10 * img_scale_x)))
    draw.line(scaled_points, fill=(red, green, blue, 145), width=stroke_width, joint="curve")

    # Annotate exact bounding box coordinates on the vision copy
    point_x_values = [pt_x for pt_x, _ in highlight_points]
    point_y_values = [pt_y for _, pt_y in highlight_points]
    bbox_x0 = int(round(min(point_x_values)))
    bbox_y0 = int(round(min(point_y_values)))
    bbox_w = max(1, int(round(max(point_x_values) - min(point_x_values))))
    bbox_h = max(1, int(round(max(point_y_values) - min(point_y_values))))
    tag_text = f"LASSO({bbox_x0},{bbox_y0},{bbox_w}x{bbox_h})"
    label_x = max(36, int(round(bbox_x0 * img_scale_x)))
    label_y = max(18, int(round(bbox_y0 * img_scale_y)) - 16)
    draw.rectangle([label_x - 3, label_y - 2, label_x + len(tag_text) * 6 + 6, label_y + 12], fill=(16, 26, 44, 215))
    draw.text((label_x, label_y), tag_text, fill=(140, 224, 255, 255), font=ImageFont.load_default())

    composited = Image.alpha_composite(rgba_shot, overlay_layer)
    return composited.convert("RGB")


# ---------------------------------------------------------------------------
# Smooth Win32 Scroll Service (for ChatGPT / Article Scroll & Explain)
# ---------------------------------------------------------------------------


class ScreenScrollService:
    """Performs smooth Win32 wheel scrolling over the active desktop pane."""

    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120

    @classmethod
    def scroll_page_smoothly(
        cls,
        direction: str = "down",
        wheel_ticks: int = 6,
        target_x: Optional[int] = None,
        target_y: Optional[int] = None,
    ) -> None:
        if sys.platform != "win32":
            return
        try:
            user32 = ctypes.windll.user32
            if target_x is not None and target_y is not None:
                user32.SetCursorPos(int(target_x), int(target_y))

            sign = -1 if (direction or "down").lower() == "down" else 1
            total_delta = sign * int(max(1, min(20, wheel_ticks))) * cls.WHEEL_DELTA
            substep_count = 10
            per_step_delta = int(round(total_delta / float(substep_count)))
            for _step_index in range(substep_count):
                user32.mouse_event(cls.MOUSEEVENTF_WHEEL, 0, 0, per_step_delta, 0)
                time.sleep(0.022)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Microphone capture + AssemblyAI v3 streaming STT
# ---------------------------------------------------------------------------


class MicrophoneStreamService(QObject):
    """
    Captures live microphone audio and streams it to AssemblyAI v3 real-time STT.

    Turns are stored by `turn_order` in a dictionary so formatted turns overwrite
    unformatted partials instead of duplicating phrases.
    """

    audioLevelChanged = Signal(float)
    partialTranscriptReceived = Signal(str)
    finalTranscriptReady = Signal(str)
    microphoneErrorOccurred = Signal(str)

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        super().__init__()
        self._settings_store = settings_store
        self._lock = threading.Lock()
        self._recording = False
        self._session_id = 0
        self._audio_queue: Optional[queue.Queue[Optional[bytes]]] = None
        self._mic_thread: Optional[threading.Thread] = None
        self._ws_thread: Optional[threading.Thread] = None

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """Enumerates available Windows audio input devices."""
        devices: List[Dict[str, Any]] = []
        audio_interface = None
        try:
            audio_interface = pyaudio.PyAudio()
            device_count = audio_interface.get_device_count()
            default_index = -1
            try:
                default_index = int(audio_interface.get_default_input_device_info().get("index", -1))
            except Exception:
                default_index = -1

            seen_names = set()
            for device_idx in range(device_count):
                info = audio_interface.get_device_info_by_index(device_idx)
                if int(info.get("maxInputChannels", 0)) <= 0:
                    continue
                if int(info.get("hostApi", 0)) != 0:
                    continue
                device_name = str(info.get("name", f"Microphone {device_idx}")).strip()
                if device_name in seen_names:
                    continue
                seen_names.add(device_name)
                devices.append(
                    {
                        "index": device_idx,
                        "name": device_name,
                        "is_default": device_idx == default_index,
                    }
                )
        except Exception:
            pass
        finally:
            if audio_interface is not None:
                try:
                    audio_interface.terminate()
                except Exception:
                    pass
        return devices

    def start_streaming(self) -> None:
        with self._lock:
            self._session_id += 1
            active_session_id = self._session_id
            self._recording = True
            self._audio_queue = queue.Queue()

        audio_q = self._audio_queue
        self._mic_thread = threading.Thread(
            target=self._capture_audio_loop,
            args=(active_session_id, audio_q),
            daemon=True,
        )
        self._ws_thread = threading.Thread(
            target=self._run_assemblyai_loop,
            args=(active_session_id, audio_q),
            daemon=True,
        )
        self._mic_thread.start()
        self._ws_thread.start()

    def stop_and_finalize(self) -> None:
        """Signals capture stop on the UI thread without blocking; WS thread emits finalTranscriptReady."""
        with self._lock:
            self._recording = False
            audio_q = self._audio_queue
        if audio_q is not None:
            audio_q.put(None)
        self.audioLevelChanged.emit(0.0)

    def _capture_audio_loop(self, session_id: int, audio_q: "queue.Queue[Optional[bytes]]") -> None:
        audio_interface: Optional[pyaudio.PyAudio] = None
        stream: Optional[pyaudio.Stream] = None
        try:
            audio_interface = pyaudio.PyAudio()
            device_index = self._settings_store.get("microphone_device_index", None)
            open_kwargs: Dict[str, Any] = {
                "format": pyaudio.paInt16,
                "channels": AUDIO_CHANNELS,
                "rate": AUDIO_SAMPLE_RATE,
                "input": True,
                "frames_per_buffer": AUDIO_FRAMES_PER_BUFFER,
            }
            if isinstance(device_index, int) and device_index >= 0:
                open_kwargs["input_device_index"] = device_index
            try:
                stream = audio_interface.open(**open_kwargs)
            except Exception:
                # Retry default input device if a saved USB mic was unplugged
                open_kwargs.pop("input_device_index", None)
                stream = audio_interface.open(**open_kwargs)

            while self._recording and self._session_id == session_id:
                pcm_bytes = stream.read(AUDIO_FRAMES_PER_BUFFER, exception_on_overflow=False)
                audio_q.put(pcm_bytes)
                samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                if samples.size > 0:
                    rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
                    normalized = min(1.0, max(0.0, (rms - 0.003) / 0.09))
                    self.audioLevelChanged.emit(normalized)
        except Exception as err:
            if self._session_id == session_id:
                self.microphoneErrorOccurred.emit(f"Microphone unavailable: {err}")
            audio_q.put(None)
        finally:
            self.audioLevelChanged.emit(0.0)
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if audio_interface is not None:
                try:
                    audio_interface.terminate()
                except Exception:
                    pass

    def _fetch_assemblyai_token(self) -> str:
        api_key = self._settings_store.get_secret(jarvis_config.SECRET_ASSEMBLYAI_API_KEY)
        if not api_key:
            return ""
        try:
            response = requests.get(
                "https://streaming.assemblyai.com/v3/token?expires_in_seconds=480",
                headers={"authorization": api_key},
                timeout=6,
            )
            if response.ok:
                return str(response.json().get("token", ""))
        except Exception:
            pass
        return ""

    def _run_assemblyai_loop(self, session_id: int, audio_q: "queue.Queue[Optional[bytes]]") -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_text = ""
        try:
            final_text = loop.run_until_complete(self._stream_turns_coroutine(session_id, audio_q))
        except Exception:
            final_text = ""
        finally:
            try:
                loop.close()
            except Exception:
                pass

        if self._session_id == session_id:
            self.finalTranscriptReady.emit(final_text)

    async def _stream_turns_coroutine(self, session_id: int, audio_q: "queue.Queue[Optional[bytes]]") -> str:
        token = self._fetch_assemblyai_token()
        if not token:
            if self._session_id == session_id:
                self.microphoneErrorOccurred.emit("AssemblyAI API key missing or token fetch failed (.env).")
            return ""

        turns_by_order: Dict[int, str] = {}

        def assemble_transcript() -> str:
            ordered_parts = [turns_by_order[key] for key in sorted(turns_by_order.keys()) if turns_by_order[key]]
            return " ".join(ordered_parts).strip()

        ws_url = (
            "wss://streaming.assemblyai.com/v3/ws"
            f"?sample_rate={AUDIO_SAMPLE_RATE}"
            "&encoding=pcm_s16le"
            "&format_turns=true"
            "&speech_model=u3-rt-pro"
            f"&token={quote(token)}"
        )

        async with websockets.connect(ws_url, ping_interval=15) as websocket:
            termination_received = asyncio.Event()

            async def send_audio() -> None:
                loop = asyncio.get_running_loop()
                while True:
                    chunk = await loop.run_in_executor(None, audio_q.get)
                    if chunk is None:
                        try:
                            await websocket.send(json.dumps({"type": "Terminate"}))
                        except Exception:
                            pass
                        break
                    if self._session_id != session_id:
                        break
                    await websocket.send(chunk)

            async def receive_turns() -> None:
                try:
                    async for raw_message in websocket:
                        if not isinstance(raw_message, str):
                            continue
                        try:
                            payload = json.loads(raw_message)
                        except ValueError:
                            continue
                        msg_type = payload.get("type", "")
                        if msg_type == "Turn":
                            turn_order = int(payload.get("turn_order", 0))
                            transcript = (payload.get("transcript") or "").strip()
                            if transcript:
                                turns_by_order[turn_order] = transcript
                                if self._session_id == session_id:
                                    self.partialTranscriptReceived.emit(assemble_transcript())
                        elif msg_type == "Termination":
                            termination_received.set()
                            break
                except Exception:
                    pass

            sender_task = asyncio.create_task(send_audio())
            receiver_task = asyncio.create_task(receive_turns())
            await sender_task
            try:
                await asyncio.wait_for(termination_received.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            if not receiver_task.done():
                receiver_task.cancel()
                try:
                    await receiver_task
                except Exception:
                    pass

        return assemble_transcript()


# ---------------------------------------------------------------------------
# Streaming ElevenLabs PCM TTS (sentence-pipelined) + SAPI fallback
# ---------------------------------------------------------------------------


def _speak_with_sapi_blocking(
    text: str,
    speed: float = 1.0,
    on_level: Optional[Any] = None,
    should_abort: Optional[Any] = None,
) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = max(-5, min(5, int(round((speed - 1.0) * 6))))
        start_time = time.perf_counter()
        # SVSFlagsAsync = 1 so we can abort mid-sentence when user presses Escape
        voice.Speak(cleaned, 1)
        while not voice.WaitUntilDone(40):
            if should_abort is not None and should_abort():
                voice.Speak("", 2)  # SVSFPurgeBeforeSpeak stops current utterance
                break
            if on_level is not None:
                elapsed = time.perf_counter() - start_time
                on_level(0.30 + 0.45 * abs(math.sin(elapsed * 8.0)))
        if on_level is not None:
            on_level(0.0)
        return max(0.3, time.perf_counter() - start_time)
    except Exception:
        return 0.0
    finally:
        try:
            import pythoncom

            pythoncom.CoUninitialize()
        except Exception:
            pass


def _play_pcm_stream_blocking(
    chunk_iterator: Any,
    sample_rate: int,
    on_level: Optional[Any] = None,
    on_bytes_played: Optional[Any] = None,
    should_abort: Optional[Any] = None,
) -> float:
    audio_interface: Optional[pyaudio.PyAudio] = None
    stream: Optional[pyaudio.Stream] = None
    leftover_byte = b""
    total_bytes_played = 0
    start_time = time.perf_counter()

    try:
        audio_interface = pyaudio.PyAudio()
        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024,
        )
        for raw_chunk in chunk_iterator:
            if should_abort is not None and should_abort():
                break
            if not raw_chunk:
                continue
            combined = leftover_byte + raw_chunk
            # Keep an even number of bytes so 16-bit PCM frames never misalign across chunks
            even_length = len(combined) & ~1
            leftover_byte = combined[even_length:]
            pcm_bytes = combined[:even_length]
            if not pcm_bytes:
                continue

            stream.write(pcm_bytes)
            total_bytes_played += len(pcm_bytes)
            if on_bytes_played is not None:
                on_bytes_played(total_bytes_played)
            if on_level is not None:
                samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                if samples.size > 0:
                    rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
                    on_level(min(1.0, max(0.0, rms * 3.8)))
        return max(0.25, time.perf_counter() - start_time)
    finally:
        if on_level is not None:
            on_level(0.0)
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if audio_interface is not None:
            try:
                audio_interface.terminate()
            except Exception:
                pass


class StreamingTTSService(QObject):
    """
    Plays sentences back-to-back via ElevenLabs `pcm_22050` streaming as the LLM emits them.

    Starts speaking sentence 1 immediately while sentence 2 is still generating.
    Falls back to Windows SAPI if `voice_id == "windows-sapi"` or the key fails.
    """

    speechStarted = Signal()
    sentenceStarted = Signal(int, str)
    speechWordProgress = Signal(str, int, int)
    audioLevelChanged = Signal(float)
    speechFinished = Signal()

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        super().__init__()
        self._settings_store = settings_store
        self._lock = threading.Lock()
        self._generation_id = 0
        self._sentence_enqueue_counter = 0
        self._queue: "queue.Queue[Tuple[int, str, Any]]" = queue.Queue()
        self._is_speaking = False
        self._last_engine_used = "elevenlabs"
        self._worker_thread = threading.Thread(target=self._playback_worker_loop, daemon=True)
        self._worker_thread.start()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def last_engine_used(self) -> str:
        return self._last_engine_used

    def begin_turn(self) -> int:
        """Clears any earlier utterance and opens a new pipelined speech turn."""
        with self._lock:
            self._generation_id += 1
            self._sentence_enqueue_counter = 0
            active_generation = self._generation_id
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        return active_generation

    def enqueue_sentence(self, sentence_text: str, turn_generation: Optional[int] = None) -> None:
        cleaned = sentence_text.strip()
        if not cleaned:
            return
        with self._lock:
            target_gen = self._generation_id if turn_generation is None else turn_generation
            if target_gen == self._generation_id:
                sentence_idx = self._sentence_enqueue_counter
                self._sentence_enqueue_counter += 1
                self._queue.put((target_gen, "sentence", (sentence_idx, cleaned)))

    def mark_turn_complete(self, turn_generation: Optional[int] = None) -> None:
        target_gen = self._generation_id if turn_generation is None else turn_generation
        if target_gen == self._generation_id:
            self._queue.put((target_gen, "done", None))

    def speak_immediately(self, full_text: str) -> int:
        gen = self.begin_turn()
        self.enqueue_sentence(full_text, turn_generation=gen)
        self.mark_turn_complete(turn_generation=gen)
        return gen

    def stop_speaking(self) -> None:
        with self._lock:
            self._generation_id += 1
            self._sentence_enqueue_counter = 0
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def speak_blocking_for_test(self, text: str) -> Tuple[str, float]:
        """Synchronous helper used by `--selftest` to verify live AssemblyAI PCM synthesis."""
        start_time = time.perf_counter()
        engine = self._speak_one_sentence(text, generation_id=self._generation_id, sentence_index=0)
        return engine, time.perf_counter() - start_time

    def _playback_worker_loop(self) -> None:
        announced_start_for_gen = -1
        while True:
            gen_id, item_kind, payload = self._queue.get()
            if gen_id != self._generation_id:
                continue

            if item_kind == "sentence" and payload:
                if isinstance(payload, tuple) and len(payload) == 2:
                    sentence_idx, sentence_text_item = int(payload[0]), str(payload[1])
                else:
                    sentence_idx, sentence_text_item = 0, str(payload)
                if announced_start_for_gen != gen_id:
                    announced_start_for_gen = gen_id
                    self._is_speaking = True
                    self.speechStarted.emit()
                self._speak_one_sentence(sentence_text_item, generation_id=gen_id, sentence_index=sentence_idx)

            elif item_kind == "done":
                if announced_start_for_gen == gen_id:
                    self._is_speaking = False
                    self.audioLevelChanged.emit(0.0)
                    self.speechFinished.emit()

    def _speak_one_sentence(
        self,
        sentence_text: str,
        generation_id: int,
        sentence_index: int = 0,
    ) -> str:
        voice_id = str(self._settings_store.get("voice_id", jarvis_config.DEFAULT_SETTINGS["voice_id"]))
        speech_speed = float(self._settings_store.get("speech_speed", 1.0))
        speech_speed = max(0.7, min(1.2, speech_speed))
        api_key = self._settings_store.get_secret(jarvis_config.SECRET_ASSEMBLYAI_API_KEY)
        words = sentence_text.split()
        total_words = max(1, len(words))

        if voice_id != "windows-sapi" and api_key:
            try:
                response = _request_assemblyai_tts_stream(
                    text_to_speak=sentence_text,
                    voice_id=voice_id,
                    api_key=api_key,
                    speech_speed=speech_speed,
                    timeout_seconds=20,
                )
                if response is not None:
                    # Estimate spoken duration conservatively so karaoke highlight tracks audio closely
                    estimated_seconds = max(0.8, len(sentence_text) / (15.5 * speech_speed))
                    expected_bytes = max(4096, int(estimated_seconds * TTS_SAMPLE_RATE * 2))

                    def on_progress(bytes_played: int) -> None:
                        fraction = min(1.0, float(bytes_played) / float(expected_bytes))
                        word_idx = min(total_words, max(1, int(math.ceil(fraction * total_words))))
                        self.speechWordProgress.emit(sentence_text, word_idx, total_words)

                    if self._generation_id == generation_id:
                        self.sentenceStarted.emit(sentence_index, sentence_text)
                    self.speechWordProgress.emit(sentence_text, 1, total_words)
                    _play_pcm_stream_blocking(
                        chunk_iterator=_iter_assemblyai_pcm_chunks(response),
                        sample_rate=TTS_SAMPLE_RATE,
                        on_level=self.audioLevelChanged.emit,
                        on_bytes_played=on_progress,
                        should_abort=lambda: self._generation_id != generation_id,
                    )
                    self.speechWordProgress.emit(sentence_text, total_words, total_words)
                    self._last_engine_used = "assemblyai"
                    return "assemblyai"
            except Exception:
                pass

        # Fallback to Windows built-in SAPI if offline or configured to Windows voice
        if self._generation_id == generation_id:
            self.sentenceStarted.emit(sentence_index, sentence_text)
        self.speechWordProgress.emit(sentence_text, total_words, total_words)
        _speak_with_sapi_blocking(
            sentence_text,
            speed=speech_speed,
            on_level=self.audioLevelChanged.emit,
            should_abort=lambda: self._generation_id != generation_id,
        )
        self._last_engine_used = "sapi"
        return "sapi"


# ---------------------------------------------------------------------------
# River AI Vision SSE Service
# ---------------------------------------------------------------------------


class VisionBrainService(QObject):
    """
    Streams screen-aware answers from River AI (`deepseek-ai/DeepSeek-V4.1-Flash` default)
    and immediately emits parsed spoken sentences and screen drawing commands.
    """

    turnStarted = Signal(str)
    partialTextUpdated = Signal(str)
    sentenceReady = Signal(str)
    drawingCommandReady = Signal(object)
    turnCompleted = Signal(object)
    turnFailed = Signal(str)

    RIVER_CHAT_URL = "https://api.river.ai/v1/chat/completions"
    GOOGLE_GEMINI_STREAM_URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:streamGenerateContent?alt=sse"
    )

    @classmethod
    def _infer_universal_key_endpoint(
        cls,
        api_key: str,
        key_alias: str,
        selected_model: str,
    ) -> Tuple[str, str]:
        """Returns `(endpoint_url, provider_label)` for an API key when no explicit base URL is set."""
        key_trimmed = (api_key or "").strip()
        if key_alias in jarvis_config.KNOWN_PROVIDER_ENDPOINTS:
            endpoint_url = jarvis_config.KNOWN_PROVIDER_ENDPOINTS[key_alias]
            provider_label = jarvis_config.infer_provider_name(endpoint_url, selected_model, key_alias)
            return (endpoint_url, provider_label)
        if key_trimmed.startswith("sk-or-v1-"):
            return ("https://openrouter.ai/api/v1/chat/completions", "OpenRouter")
        if key_trimmed.startswith("sk-ant-"):
            return ("https://api.anthropic.com/v1/messages", "Anthropic")
        if key_trimmed.startswith("gsk_"):
            return ("https://api.groq.com/openai/v1/chat/completions", "Groq")
        if key_trimmed.startswith("AIza"):
            return (cls.GOOGLE_GEMINI_STREAM_URL_TEMPLATE, "Google Gemini")
        if key_trimmed.startswith("xai-"):
            return ("https://api.x.ai/v1/chat/completions", "xAI")
        if jarvis_config.is_anthropic_model(selected_model):
            return ("https://api.anthropic.com/v1/messages", "Anthropic")
        if jarvis_config.is_google_gemini_model(selected_model):
            return (cls.GOOGLE_GEMINI_STREAM_URL_TEMPLATE, "Google Gemini")
        return ("https://api.openai.com/v1/chat/completions", "OpenAI")

    @staticmethod
    def _adapt_model_for_provider(
        selected_model: str,
        wire_protocol: str,
        provider_label: str,
        fallback_alias: str = "",
    ) -> str:
        """Keeps user's model if compatible, otherwise picks a safe default for the target endpoint."""
        clean_model = (selected_model or "").strip()
        if wire_protocol == "google":
            return (
                clean_model
                if jarvis_config.is_google_gemini_model(clean_model)
                else jarvis_config.DEFAULT_GOOGLE_CHAT_MODEL
            )
        if wire_protocol == "anthropic":
            return (
                clean_model
                if jarvis_config.is_anthropic_model(clean_model)
                else "claude-sonnet-4-20250514"
            )
        if provider_label == "River AI":
            if clean_model and ("/" in clean_model) and (not jarvis_config.is_google_gemini_model(clean_model)) and (not jarvis_config.is_anthropic_model(clean_model)):
                return clean_model
            return "deepseek-ai/DeepSeek-V4.1-Flash"
        if provider_label == "OpenAI" and (
            not clean_model
            or clean_model.startswith("deepseek-ai/")
            or jarvis_config.is_google_gemini_model(clean_model)
            or jarvis_config.is_anthropic_model(clean_model)
        ):
            return "gpt-4o"
        if clean_model:
            return clean_model
        if fallback_alias and fallback_alias in jarvis_config.DEFAULT_MODEL_FOR_KEY_ALIAS:
            return jarvis_config.DEFAULT_MODEL_FOR_KEY_ALIAS[fallback_alias]
        return "gpt-4o"

    @classmethod
    def resolve_all_vision_endpoints(
        cls,
        settings_store: jarvis_config.SettingsStore,
    ) -> List[Dict[str, str]]:
        """Builds an ordered list of primary and fallback vision endpoints from UI settings and .env."""
        selected_model = str(
            settings_store.get("chat_model", jarvis_config.DEFAULT_SETTINGS["chat_model"]) or ""
        ).strip()
        if not selected_model:
            selected_model = str(jarvis_config.DEFAULT_SETTINGS["chat_model"])

        custom_endpoint_raw = settings_store.get_effective_endpoint_url()
        custom_key, custom_key_alias = settings_store.get_custom_secret_with_name()
        google_key = settings_store.get_secret(jarvis_config.SECRET_GOOGLE_API_KEY)
        river_key = settings_store.get_secret(jarvis_config.SECRET_RIVER_API_KEY)

        custom_url_entry: Optional[Dict[str, str]] = None
        if custom_endpoint_raw:
            url_lower = custom_endpoint_raw.lower()
            is_local_host = ("localhost" in url_lower) or ("127.0.0.1" in url_lower) or ("0.0.0.0" in url_lower)
            hint_alias = custom_key_alias
            chosen_key = custom_key
            if not chosen_key:
                if ("generativelanguage.googleapis.com" in url_lower) and google_key:
                    chosen_key = google_key
                    hint_alias = "GOOGLE_API_KEY"
                elif ("api.river.ai" in url_lower) and river_key:
                    chosen_key = river_key
                    hint_alias = "RIVER_AI_API"
                elif (not is_local_host) and google_key and jarvis_config.detect_wire_protocol(custom_endpoint_raw, selected_model, "") == "google":
                    chosen_key = google_key
                    hint_alias = "GOOGLE_API_KEY"

            custom_wire = jarvis_config.detect_wire_protocol(custom_endpoint_raw, selected_model, hint_alias)
            custom_label = jarvis_config.infer_provider_name(custom_endpoint_raw, selected_model, hint_alias)
            custom_model = cls._adapt_model_for_provider(selected_model, custom_wire, custom_label, fallback_alias=hint_alias)
            custom_url_final = (
                custom_endpoint_raw
                if (custom_wire == "google" and "{model_id}" in custom_endpoint_raw)
                else jarvis_config.normalize_endpoint_url(custom_endpoint_raw, custom_wire, custom_model)
            )
            custom_url_entry = {
                "provider_label": custom_label,
                "wire_protocol": custom_wire,
                "endpoint_url": custom_url_final,
                "model_id": custom_model,
                "api_key": chosen_key,
            }

        custom_key_entry: Optional[Dict[str, str]] = None
        if custom_key:
            inferred_url, inferred_label = cls._infer_universal_key_endpoint(
                custom_key,
                custom_key_alias,
                selected_model,
            )
            inferred_wire = jarvis_config.detect_wire_protocol(inferred_url, selected_model, custom_key_alias)
            inferred_model = cls._adapt_model_for_provider(
                selected_model,
                inferred_wire,
                inferred_label,
                fallback_alias=custom_key_alias,
            )
            if custom_key_alias and custom_key_alias in jarvis_config.DEFAULT_MODEL_FOR_KEY_ALIAS and selected_model.startswith("deepseek-ai/"):
                inferred_model = jarvis_config.DEFAULT_MODEL_FOR_KEY_ALIAS[custom_key_alias]
            normalized_key_url = (
                inferred_url
                if (inferred_wire == "google" and "{model_id}" in inferred_url)
                else jarvis_config.normalize_endpoint_url(inferred_url, inferred_wire, inferred_model)
            )
            custom_key_entry = {
                "provider_label": inferred_label,
                "wire_protocol": inferred_wire,
                "endpoint_url": normalized_key_url,
                "model_id": inferred_model,
                "api_key": custom_key,
            }

        google_entry: Optional[Dict[str, str]] = None
        if google_key:
            google_model = (
                selected_model
                if jarvis_config.is_google_gemini_model(selected_model)
                else jarvis_config.DEFAULT_GOOGLE_CHAT_MODEL
            )
            google_entry = {
                "provider_label": "Google Gemini",
                "wire_protocol": "google",
                "endpoint_url": cls.GOOGLE_GEMINI_STREAM_URL_TEMPLATE,
                "model_id": google_model,
                "api_key": google_key,
            }

        river_entry: Optional[Dict[str, str]] = None
        if river_key:
            river_model = cls._adapt_model_for_provider(selected_model, "openai", "River AI")
            river_entry = {
                "provider_label": "River AI",
                "wire_protocol": "openai",
                "endpoint_url": cls.RIVER_CHAT_URL,
                "model_id": river_model,
                "api_key": river_key,
            }

        ordered_candidates: List[Dict[str, str]] = []
        if custom_url_entry is not None:
            ordered_candidates.append(custom_url_entry)

        if jarvis_config.is_google_gemini_model(selected_model):
            priority_list = [google_entry, custom_key_entry, river_entry]
        elif selected_model.startswith("deepseek-ai/"):
            priority_list = [river_entry, custom_key_entry, google_entry]
        else:
            priority_list = [custom_key_entry, google_entry, river_entry]

        for entry in priority_list:
            if entry is not None:
                ordered_candidates.append(entry)

        deduped_endpoints: List[Dict[str, str]] = []
        seen_signatures = set()
        for candidate_entry in ordered_candidates:
            signature = (
                candidate_entry["wire_protocol"],
                candidate_entry["endpoint_url"],
                candidate_entry["model_id"],
                candidate_entry["api_key"],
            )
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                deduped_endpoints.append(candidate_entry)
        return deduped_endpoints

    @classmethod
    def resolve_active_vision_provider_and_model(
        cls,
        settings_store: jarvis_config.SettingsStore,
    ) -> Tuple[str, str, str]:
        """Returns `(provider_name, model_id, api_key)` for the highest-priority configured endpoint."""
        endpoints = cls.resolve_all_vision_endpoints(settings_store)
        if not endpoints:
            selected_model = str(
                settings_store.get("chat_model", jarvis_config.DEFAULT_SETTINGS["chat_model"]) or ""
            ).strip() or str(jarvis_config.DEFAULT_SETTINGS["chat_model"])
            fallback_provider = (
                "google"
                if jarvis_config.is_google_gemini_model(selected_model)
                else ("anthropic" if jarvis_config.is_anthropic_model(selected_model) else "river")
            )
            return (fallback_provider, selected_model, "")
        primary_entry = endpoints[0]
        label_lower = primary_entry["provider_label"].lower()
        if primary_entry["wire_protocol"] == "google" or "google" in label_lower:
            provider_slug = "google"
        elif "river" in label_lower:
            provider_slug = "river"
        elif primary_entry["wire_protocol"] == "anthropic":
            provider_slug = "anthropic"
        else:
            provider_slug = "custom"
        return (provider_slug, primary_entry["model_id"], primary_entry["api_key"])

    @staticmethod
    def build_google_gemini_payload(
        system_prompt: str,
        history_messages: List[Dict[str, str]],
        request_user_text: str,
        jpeg_b64: str = "",
        lasso_crop_b64: str = "",
        max_output_tokens: int = 1100,
    ) -> Dict[str, Any]:
        """Builds a native Gemini `generateContent` JSON request payload with optional screenshots."""
        gemini_contents: List[Dict[str, Any]] = []
        for past_msg in history_messages:
            past_text = str(past_msg.get("content") or "").strip()
            if not past_text:
                continue
            gemini_role = "model" if past_msg.get("role") == "assistant" else "user"
            gemini_contents.append({"role": gemini_role, "parts": [{"text": past_text}]})

        user_parts: List[Dict[str, Any]] = [{"text": request_user_text}]
        if jpeg_b64:
            user_parts.append({"inlineData": {"mimeType": "image/jpeg", "data": jpeg_b64}})
        if lasso_crop_b64:
            user_parts.append({"inlineData": {"mimeType": "image/jpeg", "data": lasso_crop_b64}})
        gemini_contents.append({"role": "user", "parts": user_parts})

        payload: Dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": 0.35,
            },
        }
        if (system_prompt or "").strip():
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    @staticmethod
    def extract_google_sse_delta_text(chunk_json: Dict[str, Any]) -> str:
        """Extracts non-thought text deltas from a Gemini `:streamGenerateContent?alt=sse` event."""
        candidates = chunk_json.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            return ""
        content_obj = first_candidate.get("content")
        if not isinstance(content_obj, dict):
            return ""
        parts = content_obj.get("parts")
        if not isinstance(parts, list):
            return ""
        collected_chunks: List[str] = []
        for part_item in parts:
            if not isinstance(part_item, dict):
                continue
            if part_item.get("thought") is True:
                continue
            part_text = part_item.get("text")
            if isinstance(part_text, str) and part_text:
                collected_chunks.append(part_text)
        return "".join(collected_chunks)

    @staticmethod
    def build_anthropic_messages_payload(
        system_prompt: str,
        history_messages: List[Dict[str, str]],
        request_user_text: str,
        jpeg_b64: str = "",
        lasso_crop_b64: str = "",
        max_output_tokens: int = 1100,
        model_id: str = "claude-sonnet-4-20250514",
    ) -> Dict[str, Any]:
        """Builds an Anthropic `/v1/messages` SSE JSON payload with optional vision attachments."""
        anthropic_messages: List[Dict[str, Any]] = []
        for past_msg in history_messages:
            past_text = str(past_msg.get("content") or "").strip()
            if not past_text:
                continue
            anthropic_role = "assistant" if past_msg.get("role") == "assistant" else "user"
            anthropic_messages.append({"role": anthropic_role, "content": past_text})

        user_blocks: List[Dict[str, Any]] = []
        if jpeg_b64:
            user_blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64},
                }
            )
        if lasso_crop_b64:
            user_blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": lasso_crop_b64},
                }
            )
        user_blocks.append({"type": "text", "text": request_user_text})
        anthropic_messages.append({"role": "user", "content": user_blocks if jpeg_b64 else request_user_text})

        payload: Dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_output_tokens,
            "stream": True,
            "messages": anthropic_messages,
        }
        if (system_prompt or "").strip():
            payload["system"] = system_prompt
        return payload

    @staticmethod
    def extract_anthropic_sse_delta_text(chunk_json: Dict[str, Any]) -> str:
        """Extracts text deltas from Anthropic Messages SSE chunks or complete JSON responses."""
        event_type = str(chunk_json.get("type") or "")
        if event_type == "content_block_delta":
            delta_obj = chunk_json.get("delta")
            if isinstance(delta_obj, dict) and delta_obj.get("type") in ("text_delta", None):
                return str(delta_obj.get("text") or "")
        if event_type == "content_block_start":
            content_block = chunk_json.get("content_block")
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                return str(content_block.get("text") or "")
        content_list = chunk_json.get("content")
        if isinstance(content_list, list):
            text_fragments: List[str] = []
            for block_item in content_list:
                if isinstance(block_item, dict) and block_item.get("type") == "text":
                    block_text = block_item.get("text")
                    if isinstance(block_text, str) and block_text:
                        text_fragments.append(block_text)
            return "".join(text_fragments)
        return ""

    @classmethod
    def extract_universal_sse_delta_text(
        cls,
        chunk_json: Any,
        wire_protocol: str = "openai",
    ) -> str:
        """Extracts streamed or complete text across OpenAI-compat, Google Gemini, Anthropic, and Ollama."""
        if not isinstance(chunk_json, dict):
            return ""
        if wire_protocol == "google" or ("candidates" in chunk_json):
            return cls.extract_google_sse_delta_text(chunk_json)
        if wire_protocol == "anthropic" or chunk_json.get("type") in (
            "content_block_delta",
            "content_block_start",
            "message_start",
            "message_delta",
        ):
            anthropic_text = cls.extract_anthropic_sse_delta_text(chunk_json)
            if anthropic_text or wire_protocol == "anthropic":
                return anthropic_text
        if isinstance(chunk_json.get("content"), list) and ("choices" not in chunk_json):
            return cls.extract_anthropic_sse_delta_text(chunk_json)

        choices = chunk_json.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                delta_obj = first_choice.get("delta")
                if isinstance(delta_obj, dict):
                    delta_content = delta_obj.get("content")
                    if isinstance(delta_content, str):
                        return delta_content
                    if isinstance(delta_content, list):
                        return "".join(
                            str(part_item.get("text") or "")
                            for part_item in delta_content
                            if isinstance(part_item, dict)
                        )
                message_obj = first_choice.get("message")
                if isinstance(message_obj, dict):
                    message_content = message_obj.get("content")
                    if isinstance(message_content, str):
                        return message_content
                    if isinstance(message_content, list):
                        return "".join(
                            str(part_item.get("text") or "")
                            for part_item in message_content
                            if isinstance(part_item, dict)
                        )
                legacy_text = first_choice.get("text")
                if isinstance(legacy_text, str):
                    return legacy_text

        if chunk_json.get("type") == "response.output_text.delta":
            return str(chunk_json.get("delta") or "")

        ollama_message = chunk_json.get("message")
        if isinstance(ollama_message, dict) and isinstance(ollama_message.get("content"), str):
            return ollama_message["content"]
        if isinstance(chunk_json.get("response"), str):
            return chunk_json["response"]
        return ""

    @staticmethod
    def get_google_model_fallback_chain(model_id: str) -> List[str]:
        """Returns the requested Google Gemini model first, followed by deduplicated fallbacks."""
        normalized_model = (model_id or jarvis_config.DEFAULT_GOOGLE_CHAT_MODEL).strip()
        if normalized_model.startswith("models/"):
            normalized_model = normalized_model[len("models/") :]
        elif normalized_model.startswith("google/"):
            normalized_model = normalized_model[len("google/") :]
        if not normalized_model:
            normalized_model = jarvis_config.DEFAULT_GOOGLE_CHAT_MODEL

        lowered_model = normalized_model.lower()
        if "hydrogen" in lowered_model or "pro" in lowered_model:
            fallback_candidates = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
        else:
            fallback_candidates = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"]

        ordered_chain: List[str] = []
        for candidate_name in [normalized_model] + fallback_candidates:
            if candidate_name and candidate_name not in ordered_chain:
                ordered_chain.append(candidate_name)
        return ordered_chain

    def _stream_vision_provider_once(
        self,
        provider_name: str,
        model_id: str,
        api_key: str,
        system_prompt: str,
        filtered_history: List[Dict[str, str]],
        request_user_text: str,
        jpeg_b64: str,
        lasso_crop_b64: str,
        extractor: jarvis_response_parser.StreamingSentenceExtractor,
        turn_generation: int,
        allow_visual_notes: bool,
        endpoint_url: str = "",
        wire_protocol: str = "",
    ) -> Tuple[bool, bool, str]:
        """Streams one vision turn against any OpenAI-compatible, Google Gemini, or Anthropic endpoint.

        Returns `(succeeded, emitted_any_chunk, error_message)`.
        """
        active_protocol = wire_protocol or (
            "google"
            if provider_name == "google"
            else (
                "anthropic"
                if provider_name == "anthropic"
                else jarvis_config.detect_wire_protocol(endpoint_url, model_id, "")
            )
        )
        if endpoint_url:
            provider_label = jarvis_config.infer_provider_name(endpoint_url, model_id, "")
        else:
            provider_label = (
                "Google Gemini"
                if active_protocol == "google"
                else ("Anthropic" if active_protocol == "anthropic" else "River AI")
            )

        candidate_models = (
            self.get_google_model_fallback_chain(model_id)
            if (active_protocol == "google" and (not endpoint_url or "generativelanguage.googleapis.com" in endpoint_url.lower()))
            else [model_id]
        )

        emitted_any_chunk = False
        last_error_message = f"{provider_label} request failed."
        turn_max_tokens = 1100 if allow_visual_notes else 750

        for candidate_index, candidate_model in enumerate(candidate_models):
            attachment_modes = [True, False] if jpeg_b64 else [False]
            for include_images in attachment_modes:
                if self._generation_id != turn_generation:
                    return (False, emitted_any_chunk, "Turn cancelled.")

                step_jpeg = jpeg_b64 if include_images else ""
                step_lasso = lasso_crop_b64 if include_images else ""

                if active_protocol == "google":
                    base_google_template = endpoint_url or self.GOOGLE_GEMINI_STREAM_URL_TEMPLATE
                    if "{model_id}" in base_google_template:
                        request_url = base_google_template.format(model_id=candidate_model)
                    else:
                        request_url = jarvis_config.normalize_endpoint_url(
                            base_google_template,
                            "google",
                            candidate_model,
                        )
                    request_headers: Dict[str, str] = {"Content-Type": "application/json"}
                    if api_key:
                        request_headers["x-goog-api-key"] = api_key
                    request_json = self.build_google_gemini_payload(
                        system_prompt=system_prompt,
                        history_messages=filtered_history,
                        request_user_text=request_user_text,
                        jpeg_b64=step_jpeg,
                        lasso_crop_b64=step_lasso,
                        max_output_tokens=turn_max_tokens,
                    )
                elif active_protocol == "anthropic":
                    request_url = endpoint_url or "https://api.anthropic.com/v1/messages"
                    request_headers = {
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    }
                    if api_key:
                        request_headers["x-api-key"] = api_key
                    request_json = self.build_anthropic_messages_payload(
                        system_prompt=system_prompt,
                        history_messages=filtered_history,
                        request_user_text=request_user_text,
                        jpeg_b64=step_jpeg,
                        lasso_crop_b64=step_lasso,
                        max_output_tokens=turn_max_tokens,
                        model_id=candidate_model,
                    )
                else:
                    openai_messages: List[Dict[str, Any]] = []
                    if (system_prompt or "").strip():
                        openai_messages.append({"role": "system", "content": system_prompt})
                    openai_messages.extend(filtered_history)
                    if step_jpeg:
                        user_content: Any = [
                            {"type": "text", "text": request_user_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{step_jpeg}"}},
                        ]
                        if step_lasso:
                            user_content.append(
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{step_lasso}"}}
                            )
                    else:
                        user_content = request_user_text
                    openai_messages.append({"role": "user", "content": user_content})

                    request_url = endpoint_url or self.RIVER_CHAT_URL
                    url_lower = request_url.lower()
                    request_headers = {"Content-Type": "application/json"}
                    if api_key:
                        if ("openai.azure.com" in url_lower) or ("api-version=" in url_lower):
                            request_headers["api-key"] = api_key
                        else:
                            request_headers["Authorization"] = f"Bearer {api_key}"
                    if "openrouter.ai" in url_lower:
                        request_headers["HTTP-Referer"] = "https://heyclicky.com"
                        request_headers["X-Title"] = "Jarvis"
                    request_json = {
                        "model": candidate_model,
                        "messages": openai_messages,
                        "max_tokens": turn_max_tokens,
                        "stream": True,
                    }
                    if "api.river.ai" in url_lower:
                        request_json["reasoning_effort"] = "none"

                try:
                    with requests.post(
                        request_url,
                        headers=request_headers,
                        json=request_json,
                        stream=True,
                        timeout=45,
                    ) as response:
                        if not response.ok:
                            error_text = response.text[:240]
                            last_error_message = f"{provider_label} error {response.status_code}: {error_text}"
                            if (not emitted_any_chunk) and include_images and response.status_code in (400, 422):
                                # Text-only models (e.g. DeepSeek direct, some Groq/Ollama models) reject image_url; retry text-only once
                                continue
                            has_more_candidates = (candidate_index + 1) < len(candidate_models)
                            if (
                                active_protocol == "google"
                                and (not emitted_any_chunk)
                                and response.status_code in (400, 404)
                                and has_more_candidates
                            ):
                                break
                            return (False, emitted_any_chunk, last_error_message)

                        response.encoding = "utf-8"
                        plain_json_lines: List[str] = []
                        saw_sse_prefix = False
                        for raw_line in response.iter_lines(decode_unicode=True):
                            if self._generation_id != turn_generation:
                                return (False, emitted_any_chunk, "Turn cancelled.")
                            if not raw_line:
                                continue
                            stripped_line = raw_line.strip()
                            if stripped_line.startswith("data:"):
                                saw_sse_prefix = True
                                data_str = stripped_line[5:].strip()
                                if data_str == "[DONE]":
                                    break
                            elif stripped_line.startswith("event:"):
                                saw_sse_prefix = True
                                continue
                            elif (not saw_sse_prefix) and stripped_line.startswith("{"):
                                try:
                                    chunk_json = json.loads(stripped_line)
                                except Exception:
                                    plain_json_lines.append(raw_line)
                                    continue
                                data_str = ""
                            else:
                                if not saw_sse_prefix:
                                    plain_json_lines.append(raw_line)
                                continue

                            if stripped_line.startswith("data:"):
                                try:
                                    chunk_json = json.loads(data_str)
                                except Exception:
                                    continue

                            if isinstance(chunk_json, dict) and isinstance(chunk_json.get("error"), dict):
                                api_err_msg = str(chunk_json["error"].get("message") or chunk_json["error"])[:240]
                                return (False, emitted_any_chunk, f"{provider_label} error: {api_err_msg}")

                            delta_text = self.extract_universal_sse_delta_text(chunk_json, active_protocol)
                            if not delta_text:
                                continue

                            emitted_any_chunk = True
                            new_sentences, new_commands = extractor.feed_delta(delta_text)
                            for cmd in new_commands:
                                if not allow_visual_notes and cmd.kind in ("paper", "underline"):
                                    continue
                                self.drawingCommandReady.emit(cmd)
                            for sentence in new_sentences:
                                self.sentenceReady.emit(sentence)
                            display_so_far = extractor.current_display_text()
                            if display_so_far:
                                self.partialTextUpdated.emit(display_so_far)

                        if (not emitted_any_chunk) and (not saw_sse_prefix) and plain_json_lines:
                            try:
                                complete_payload = json.loads("\n".join(plain_json_lines))
                            except Exception:
                                complete_payload = None
                            if isinstance(complete_payload, dict):
                                if isinstance(complete_payload.get("error"), dict):
                                    api_err_msg = str(
                                        complete_payload["error"].get("message") or complete_payload["error"]
                                    )[:240]
                                    return (False, emitted_any_chunk, f"{provider_label} error: {api_err_msg}")
                                fallback_text = self.extract_universal_sse_delta_text(complete_payload, active_protocol)
                                if fallback_text:
                                    emitted_any_chunk = True
                                    new_sentences, new_commands = extractor.feed_delta(fallback_text)
                                    for cmd in new_commands:
                                        if not allow_visual_notes and cmd.kind in ("paper", "underline"):
                                            continue
                                        self.drawingCommandReady.emit(cmd)
                                    for sentence in new_sentences:
                                        self.sentenceReady.emit(sentence)
                                    display_so_far = extractor.current_display_text()
                                    if display_so_far:
                                        self.partialTextUpdated.emit(display_so_far)

                    return (True, emitted_any_chunk, "")
                except Exception as err:
                    return (False, emitted_any_chunk, f"Could not reach {provider_label}: {err}")

        return (False, emitted_any_chunk, last_error_message)

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        super().__init__()
        self._settings_store = settings_store
        self._lock = threading.Lock()
        self._generation_id = 0
        self._history: List[Dict[str, str]] = []
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def stop_current_turn(self) -> None:
        with self._lock:
            self._generation_id += 1
            self._is_running = False

    def clear_history(self) -> None:
        self._history.clear()

    def start_vision_turn(
        self,
        user_prompt: str,
        include_screenshot: bool = True,
        highlight_points: Optional[List[Tuple[float, float]]] = None,
        highlight_color_hex: str = jarvis_theme.JARVIS_BLUE_HEX,
        logical_screen_size: Tuple[int, int] = (1536, 864),
        scroll_step_index: int = 0,
        max_scroll_steps: int = 0,
        memory_context: str = "",
        live_screen_context: str = "",
        suppress_visual_notes: bool = False,
    ) -> int:
        with self._lock:
            self._generation_id += 1
            turn_generation = self._generation_id
            self._is_running = True

        # Persist answer-style preferences if the user asked for a specific style ("Answers the way you like them")
        learned_rule = jarvis_response_parser.extract_style_preference_from_user_message(user_prompt)
        if learned_rule:
            existing_notes = list(self._settings_store.get("remember_style_notes", []))
            if learned_rule not in existing_notes:
                existing_notes.append(learned_rule)
                self._settings_store.set("remember_style_notes", existing_notes[-8:])

        self.turnStarted.emit(user_prompt)

        threading.Thread(
            target=self._run_turn_worker,
            args=(
                turn_generation,
                user_prompt,
                include_screenshot,
                highlight_points or [],
                highlight_color_hex,
                logical_screen_size,
                scroll_step_index,
                max_scroll_steps,
                memory_context,
                live_screen_context,
                suppress_visual_notes,
            ),
            daemon=True,
        ).start()
        return turn_generation

    def _run_turn_worker(
        self,
        turn_generation: int,
        user_prompt: str,
        include_screenshot: bool,
        highlight_points: List[Tuple[float, float]],
        highlight_color_hex: str,
        logical_screen_size: Tuple[int, int],
        scroll_step_index: int = 0,
        max_scroll_steps: int = 0,
        memory_context: str = "",
        live_screen_context: str = "",
        suppress_visual_notes: bool = False,
    ) -> None:
        vision_endpoints = self.resolve_all_vision_endpoints(self._settings_store)
        if not vision_endpoints:
            self._is_running = False
            if self._generation_id == turn_generation:
                self.turnFailed.emit(
                    "No chat/vision API endpoint or key is configured in Studio & Keys or .env."
                )
            return

        jpeg_b64 = ""
        lasso_crop_b64 = ""
        img_w, img_h = logical_screen_size
        scale_x, scale_y = 1.0, 1.0

        if include_screenshot:
            try:
                jpeg_b64, img_w, img_h, scale_x, scale_y = ScreenCaptureService.capture_screen_with_highlight(
                    highlight_points=highlight_points,
                    highlight_color_hex=highlight_color_hex,
                    logical_screen_size=logical_screen_size,
                )
                if highlight_points and len(highlight_points) >= 2:
                    lasso_crop_b64 = ScreenCaptureService.get_last_lasso_crop_b64()
            except Exception:
                jpeg_b64 = ""
                lasso_crop_b64 = ""

        allow_visual_notes = (not suppress_visual_notes) and (
            bool(lasso_crop_b64)
            or (max_scroll_steps > 0)
            or jarvis_response_parser.should_use_visual_explanation_notes(
                user_prompt,
                has_lasso_crop=bool(lasso_crop_b64),
                is_scroll_session=(max_scroll_steps > 0),
            )
        )

        system_telemetry_snapshot = ""
        try:
            system_telemetry_snapshot = SystemTelemetryService.get_full_system_snapshot()
        except Exception:
            system_telemetry_snapshot = ""

        system_prompt = jarvis_response_parser.build_vision_system_prompt(
            screen_width=img_w,
            screen_height=img_h,
            style_notes=self._settings_store.get("remember_style_notes", []),
            custom_words=self._settings_store.get("dictation_custom_words", []),
            has_lasso_crop=bool(lasso_crop_b64),
            scroll_step_index=scroll_step_index,
            max_scroll_steps=max_scroll_steps,
            drawing_mode=str(
                self._settings_store.get(
                    "screen_drawing_mode",
                    jarvis_config.DEFAULT_SETTINGS["screen_drawing_mode"],
                )
            ),
            agent_actions_enabled=bool(self._settings_store.get("agent_actions_enabled", True)),
            memory_context="",
            live_screen_context="",
            suppress_visual_notes=(not allow_visual_notes),
            system_telemetry_summary=system_telemetry_snapshot,
        )

        # Keep system message completely static across turns; place compact sanitized desktop/session
        # context in the user turn so the model never treats IDE titles or memory logs as system injection.
        context_notes: List[str] = []
        if live_screen_context:
            clean_live_lines = [
                line_text.strip()
                for line_text in live_screen_context.splitlines()
                if line_text.strip()
                and not JarvisMemoryService.is_polluted_memory_text(line_text)
                and "Diff:" not in line_text
            ]
            if clean_live_lines:
                context_notes.append("Desktop Context: " + " | ".join(clean_live_lines[:3]))
        if memory_context:
            clean_memory_lines = [
                line_text.strip()
                for line_text in memory_context.splitlines()
                if line_text.strip()
                and not JarvisMemoryService.is_polluted_memory_text(line_text)
                and "Screen Monitor" not in line_text
            ]
            if clean_memory_lines:
                context_notes.append("Session Notes:\n" + "\n".join(clean_memory_lines[-4:]))

        request_user_text = user_prompt
        if context_notes:
            request_user_text = f"{user_prompt}\n\n[Context]\n" + "\n".join(context_notes)

        filtered_history: List[Dict[str, str]] = []
        for past_message in self._history[-6:]:
            past_content = past_message.get("content", "")
            if isinstance(past_content, str) and JarvisMemoryService.is_polluted_memory_text(past_content):
                continue
            filtered_history.append(past_message)

        extractor = jarvis_response_parser.StreamingSentenceExtractor(scale_x=scale_x, scale_y=scale_y)

        try:
            stream_ok = False
            stream_error_message = "Vision request failed."
            for endpoint_index, endpoint_cfg in enumerate(vision_endpoints):
                if self._generation_id != turn_generation:
                    return
                if endpoint_index > 0:
                    extractor = jarvis_response_parser.StreamingSentenceExtractor(
                        scale_x=scale_x, scale_y=scale_y
                    )
                stream_ok, emitted_any_chunk, stream_error_message = self._stream_vision_provider_once(
                    provider_name=endpoint_cfg["provider_label"],
                    model_id=endpoint_cfg["model_id"],
                    api_key=endpoint_cfg["api_key"],
                    system_prompt=system_prompt,
                    filtered_history=filtered_history,
                    request_user_text=request_user_text,
                    jpeg_b64=jpeg_b64,
                    lasso_crop_b64=lasso_crop_b64,
                    extractor=extractor,
                    turn_generation=turn_generation,
                    allow_visual_notes=allow_visual_notes,
                    endpoint_url=endpoint_cfg["endpoint_url"],
                    wire_protocol=endpoint_cfg["wire_protocol"],
                )
                if stream_ok or emitted_any_chunk or self._generation_id != turn_generation:
                    break

            if not stream_ok:
                if self._generation_id == turn_generation:
                    self._is_running = False
                    self.turnFailed.emit(stream_error_message or "Vision request failed.")
                return

            if self._generation_id != turn_generation:
                return

            final_sentences, final_commands = extractor.flush_remaining()
            for cmd in final_commands:
                if not allow_visual_notes and cmd.kind in ("paper", "underline"):
                    continue
                self.drawingCommandReady.emit(cmd)
            for sentence in final_sentences:
                self.sentenceReady.emit(sentence)

            parsed_response = extractor.build_final_response()
            if not allow_visual_notes:
                parsed_response.commands = [
                    cmd_item
                    for cmd_item in parsed_response.commands
                    if cmd_item.kind not in ("paper", "underline")
                ]

            active_drawing_mode = str(
                self._settings_store.get(
                    "screen_drawing_mode",
                    jarvis_config.DEFAULT_SETTINGS["screen_drawing_mode"],
                )
            ).strip().lower()
            agent_command_kinds = {
                "open_browser",
                "open_url",
                "search_web",
                "open_app",
                "press_keys",
                "type_text",
                "open_side_by_side",
                "window_action",
                "set_clipboard",
                "write_file",
                "system_control",
                "run_shell_read",
                "request_admin_permission",
                "write_project_file",
                "read_project_file",
                "run_project_command",
                "open_project",
                "click_ui",
                "inspect_ui",
                "agent_done",
                "read_window",
                "list_windows",
                "act_in_app",
                "find_text_on_screen",
                "search_files",
                "list_files",
                "plan_steps",
                "remember_commitment",
                "snooze_reminder",
                "list_commitments",
                "show_list",
                "start_focus",
                "stop_focus",
                "save_skill",
                "use_skill",
                "forget_skill",
                "install_skill",
                "schedule_job",
                "list_jobs",
                "learn_fact",
                "forget_fact",
                "edit_drawing",
                "photo",
                "scroll",
            }
            agent_actions_enabled = bool(self._settings_store.get("agent_actions_enabled", True))
            has_agent_command = any(cmd_item.kind in agent_command_kinds for cmd_item in parsed_response.commands)
            if agent_actions_enabled and not has_agent_command:
                fallback_agent_cmds = DesktopAgentService.coalesce_agent_commands(
                    DesktopAgentService.detect_direct_user_agent_commands(user_prompt)
                )
                if fallback_agent_cmds:
                    for fallback_cmd in fallback_agent_cmds:
                        parsed_response.commands.append(fallback_cmd)
                        self.drawingCommandReady.emit(fallback_cmd)
                    has_agent_command = True

            if not parsed_response.spoken_text:
                first_agent_cmd = next(
                    (cmd_item for cmd_item in parsed_response.commands if cmd_item.kind in agent_command_kinds),
                    None,
                )
                if first_agent_cmd is not None:
                    synthesized_speech = (
                        DesktopAgentService.describe_step_for_speech(first_agent_cmd)
                        or "Done."
                    )
                else:
                    synthesized_speech = "I'm ready. Go ahead and tell me what you'd like to do."
                parsed_response.spoken_text = synthesized_speech
                parsed_response.sentences = [synthesized_speech]
                self.partialTextUpdated.emit(synthesized_speech)
                self.sentenceReady.emit(synthesized_speech)

            has_underline = any(cmd_item.kind == "underline" for cmd_item in parsed_response.commands)
            existing_paper_count = sum(1 for cmd_item in parsed_response.commands if cmd_item.kind == "paper")
            underline_labels = [
                cmd_item.label.strip()
                for cmd_item in parsed_response.commands
                if cmd_item.kind == "underline" and cmd_item.label and cmd_item.label.strip()
            ]

            if (
                allow_visual_notes
                and active_drawing_mode != "cursor_only"
                and not has_agent_command
                and parsed_response.spoken_text
                and len(parsed_response.spoken_text) >= 24
            ):
                synthesized_cards = self._build_visual_note_cards_for_response(
                    spoken_sentences=parsed_response.sentences,
                    spoken_text=parsed_response.spoken_text,
                    screen_width=img_w,
                    screen_height=img_h,
                    underline_labels=underline_labels,
                    existing_paper_count=existing_paper_count,
                    user_prompt=user_prompt,
                )
                for synthesized_card in synthesized_cards:
                    parsed_response.commands.append(synthesized_card)
                    self.drawingCommandReady.emit(synthesized_card)
                if not has_underline:
                    anchors = ScreenCaptureService.detect_visual_text_anchors(img_w, img_h, count=2)
                    for underline_idx, (anchor_x, baseline_y, span_width) in enumerate(anchors):
                        fallback_underline = jarvis_response_parser.DrawingCommand(
                            kind="underline",
                            x=anchor_x,
                            y=baseline_y,
                            width=span_width,
                            label="",
                            tone="blue",
                            sentence_index=underline_idx,
                        )
                        parsed_response.commands.append(fallback_underline)
                        self.drawingCommandReady.emit(fallback_underline)

            if parsed_response.spoken_text:
                self._history.append({"role": "user", "content": user_prompt})
                self._history.append({"role": "assistant", "content": parsed_response.spoken_text})

            self._is_running = False
            self.turnCompleted.emit(parsed_response)
        except Exception as err:
            self._is_running = False
            if self._generation_id == turn_generation:
                self.turnFailed.emit(f"Could not reach vision model: {err}")

    @staticmethod
    def _build_visual_note_cards_for_response(
        spoken_sentences: List[str],
        spoken_text: str,
        screen_width: int = 1536,
        screen_height: int = 864,
        underline_labels: Optional[List[str]] = None,
        existing_paper_count: int = 0,
        user_prompt: str = "",
    ) -> List[jarvis_response_parser.DrawingCommand]:
        """
        Synthesizes structured visual diagram note cards so every visual answer has 2 to 3
        complementary topic maps covering overview/archetype visual, connections, and recall clue.
        """
        cleaned_sentences = [sentence.strip() for sentence in spoken_sentences if sentence and sentence.strip()]
        if not cleaned_sentences and spoken_text:
            cleaned_sentences = [spoken_text.strip()]
        if not cleaned_sentences:
            return []

        pattern_meta = jarvis_response_parser.classify_visual_explanation_pattern(
            user_prompt=user_prompt,
            spoken_text=spoken_text,
            labels=underline_labels,
        )
        archetype = pattern_meta.get("archetype", "process")

        label_topics: List[str] = []
        for raw_label in underline_labels or []:
            topic_label = jarvis_response_parser.extract_key_topic_phrase(raw_label, max_words=3)
            if topic_label and topic_label not in label_topics:
                label_topics.append(topic_label)

        sentence_topics: List[str] = []
        for sentence_text in cleaned_sentences:
            topic_label = jarvis_response_parser.extract_key_topic_phrase(sentence_text, max_words=3)
            if topic_label and topic_label not in sentence_topics:
                sentence_topics.append(topic_label)

        desired_card_count = 1
        if len(cleaned_sentences) >= 2 or len(label_topics) >= 2:
            desired_card_count = 2
        if len(cleaned_sentences) >= 5 or len(label_topics) >= 4:
            desired_card_count = 3

        needed_cards = max(0, desired_card_count - max(0, int(existing_paper_count)))
        if needed_cards <= 0:
            return []

        merged_topics: List[str] = []
        for topic_item in label_topics + sentence_topics:
            if topic_item and topic_item not in merged_topics:
                merged_topics.append(topic_item)
        while len(merged_topics) < 6:
            fallback_defaults = ["Overview", "Core Idea", "Mechanism", "Key Step", "Comparison", "Outcome"]
            next_default = fallback_defaults[len(merged_topics) % len(fallback_defaults)]
            if next_default not in merged_topics:
                merged_topics.append(next_default)
            else:
                merged_topics.append(f"Point {len(merged_topics) + 1}")

        card_width = 304.0
        card_height = 264.0
        right_col_x = float(max(360, int(screen_width) - int(card_width) - 24))
        bottom_row_y = float(max(376, int(screen_height) - int(card_height) - 72))
        margin_slots = [
            (24.0, 92.0, card_width, card_height),
            (right_col_x, 92.0, card_width, card_height),
            (24.0, bottom_row_y, card_width, min(246.0, card_height)),
        ]

        cards_to_build: List[jarvis_response_parser.DrawingCommand] = []
        total_sentences = max(1, len(cleaned_sentences))
        for card_offset in range(needed_cards):
            slot_idx = (max(0, int(existing_paper_count)) + card_offset) % len(margin_slots)
            slot_x, slot_y, slot_w, slot_h = margin_slots[slot_idx]
            if slot_idx == 0:
                card_title = merged_topics[0]
                if archetype == "comparison":
                    card_header = "Comparison Table"
                    card_body = (
                        f"compare: {merged_topics[0]} vs {merged_topics[1]} | "
                        f"section:Key Contrast | "
                        f"{merged_topics[2]} -> {merged_topics[3]} | "
                        f"clue:{pattern_meta['memory_clue']}"
                    )
                elif archetype == "hierarchy":
                    card_header = "Hierarchy Tree"
                    card_body = (
                        f"tree: {merged_topics[0]} > {merged_topics[1]}, {merged_topics[2]} | "
                        f"section:Classification | "
                        f"doc:{merged_topics[2]} -> {merged_topics[3]} | "
                        f"clue:{pattern_meta['memory_clue']}"
                    )
                elif archetype == "decision":
                    card_header = "Decision Tree"
                    card_body = (
                        f"if: {merged_topics[0]}? -> {merged_topics[1]} / {merged_topics[2]} | "
                        f"section:Outcome | "
                        f"doc:{merged_topics[1]} -> {merged_topics[3]} | "
                        f"clue:{pattern_meta['memory_clue']}"
                    )
                else:
                    card_header = pattern_meta["header_badge"]
                    card_body = (
                        f"{merged_topics[0]} -> {merged_topics[1]} | "
                        f"section:{pattern_meta['pattern_name']} | "
                        f"doc:{merged_topics[1]} -> {merged_topics[2]} | "
                        f"clue:{pattern_meta['memory_clue']}"
                    )
                target_sentence_idx = 0
            elif slot_idx == 1:
                pair_start = 2 if len(merged_topics) >= 4 else 1
                topic_first = merged_topics[pair_start]
                topic_second = merged_topics[(pair_start + 1) % len(merged_topics)]
                topic_third = merged_topics[(pair_start + 2) % len(merged_topics)]
                card_title = f"{topic_first} Link"
                card_header = "Connections & Example"
                card_body = (
                    f"{topic_first} -> {topic_second} | "
                    f"section:Real-Life Example | "
                    f"doc:{topic_second} -> {topic_third} | "
                    f"clue:Why -> How -> Effect"
                )
                target_sentence_idx = min(total_sentences - 1, max(1, total_sentences // 2))
            else:
                tail_first = merged_topics[-3]
                tail_second = merged_topics[-2]
                tail_third = merged_topics[-1]
                card_title = "Pattern & Recall"
                card_header = "Pattern & Recall"
                card_body = (
                    f"{tail_first} -> {tail_second} | "
                    f"section:Exam Hook | "
                    f"doc:{tail_second} -> {tail_third} | "
                    f"clue:{pattern_meta['pattern_name']}"
                )
                target_sentence_idx = total_sentences - 1

            cards_to_build.append(
                jarvis_response_parser.DrawingCommand(
                    kind="paper",
                    x=slot_x,
                    y=slot_y,
                    width=slot_w,
                    height=slot_h,
                    label=card_title,
                    header=card_header,
                    body=card_body,
                    tone="blue",
                    sentence_index=target_sentence_idx,
                )
            )

        return cards_to_build

    @staticmethod
    def _build_fallback_paper_card(
        spoken_sentences: List[str],
        spoken_text: str,
        screen_height: int,
        underline_labels: Optional[List[str]] = None,
    ) -> Optional[jarvis_response_parser.DrawingCommand]:
        """Backward-compatible accessor returning the primary visual diagram card."""
        cards = VisionBrainService._build_visual_note_cards_for_response(
            spoken_sentences=spoken_sentences,
            spoken_text=spoken_text,
            screen_width=1536,
            screen_height=screen_height,
            underline_labels=underline_labels,
            existing_paper_count=0,
        )
        return cards[0] if cards else None


    def request_autonomous_agent_step(
        self,
        user_goal: str,
        step_index: int,
        max_steps: int,
        observation_text: str,
        jpeg_b64: str = "",
        turn_generation: int = 0,
    ) -> Tuple[bool, jarvis_response_parser.ParsedTurnResponse, str]:
        """Executes one synchronous Plan-Act-Observe LLM step against any configured endpoint."""
        vision_endpoints = self.resolve_all_vision_endpoints(self._settings_store)
        if not vision_endpoints:
            return (
                False,
                jarvis_response_parser.ParsedTurnResponse(),
                "No API endpoint or key configured in Studio & Keys or .env.",
            )

        agent_system_prompt = (
            "You are J.A.R.V.I.S, an ultra-capable Windows & D:\\Projects autonomous builder and desktop agent.\n"
            "You operate in a strict Plan-Act-Observe loop. On each turn you receive the live OS observation "
            "(foreground window, UI Automation controls, D:\\Projects folders, and previous step output/exit code).\n\n"
            "Available single-line tool commands (emit 1 to 4 commands FIRST, followed by one concise natural spoken sentence):\n"
            '- write_project_file("project-folder/relative/path.ext", "full file content with \\n escapes")\n'
            '- read_project_file("project-folder/relative/path.ext")\n'
            '- run_project_command("shell command e.g. python -m pytest or npm run build or python app.py", "project-folder")\n'
            '- open_project("project-folder", "vscode|explorer|terminal|auto")\n'
            '- open_app("app name, Downloads/Desktop/Documents, or settings:<topic>")\n'
            '- open_url("https://... or http://localhost:PORT")\n'
            '- search_web("query")\n'
            '- open_side_by_side("left app/site", "right app/site")\n'
            '- window_action("focus|minimize|maximize|snap_left|snap_right|close", "window title")\n'
            '- click_ui("exact UI control name or x,y")\n'
            '- inspect_ui("optional window query")\n'
            '- type_text("text", "submit")\n'
            '- press_keys("ctrl+s")\n'
            '- request_admin_permission("action title", "risk reason")\n'
            '- agent_done("short completion summary")\n\n'
            "Rules:\n"
            "1. ALWAYS create and build new user coding projects inside D:\\Projects\\<project-name> unless Desktop/Documents is requested.\n"
            "2. Emit real runnable files cleanly with proper \\n escapes inside write_project_file(...) — never leave placeholders.\n"
            "3. Verify builds/tests with run_project_command(...) when appropriate, and open the result in VS Code/Chrome/Explorer if helpful.\n"
            "4. Emit agent_done(\"...\") as soon as the goal is verified complete.\n"
            "5. Spoken text MUST be crisp, polite, direct JARVIS status (1 short sentence, zero raw tool syntax spoken)."
        )
        user_step_prompt = (
            f"User Goal: {user_goal}\n"
            f"Step: {step_index + 1} of {max_steps}\n\n"
            f"Live Observation:\n{observation_text}\n\n"
            "Emit the next command(s) (or agent_done) followed by a short 1-sentence JARVIS status."
        )

        last_err = "Autonomous step failed."
        for endpoint_cfg in vision_endpoints:
            if turn_generation and self._generation_id != turn_generation:
                return (False, jarvis_response_parser.ParsedTurnResponse(), "Turn cancelled.")
            wire_proto = endpoint_cfg["wire_protocol"]
            endpoint_url = endpoint_cfg["endpoint_url"]
            model_id = endpoint_cfg["model_id"]
            api_key = endpoint_cfg["api_key"]
            extractor = jarvis_response_parser.StreamingSentenceExtractor()

            if wire_proto == "google":
                resolved_url = (
                    endpoint_url.format(model_id=model_id)
                    if "{model_id}" in endpoint_url
                    else jarvis_config.normalize_endpoint_url(endpoint_url, "google", model_id)
                )
                req_headers: Dict[str, str] = {"Content-Type": "application/json"}
                if api_key:
                    req_headers["x-goog-api-key"] = api_key
                req_payload = self.build_google_gemini_payload(
                    system_prompt=agent_system_prompt,
                    history_messages=[],
                    request_user_text=user_step_prompt,
                    jpeg_b64=jpeg_b64,
                    max_output_tokens=1600,
                )
            elif wire_proto == "anthropic":
                resolved_url = endpoint_url
                req_headers = {"anthropic-version": "2023-06-01", "Content-Type": "application/json"}
                if api_key:
                    req_headers["x-api-key"] = api_key
                req_payload = self.build_anthropic_messages_payload(
                    system_prompt=agent_system_prompt,
                    history_messages=[],
                    request_user_text=user_step_prompt,
                    jpeg_b64=jpeg_b64,
                    max_output_tokens=1600,
                    model_id=model_id,
                )
            else:
                resolved_url = endpoint_url
                url_lower = resolved_url.lower()
                req_headers = {"Content-Type": "application/json"}
                if api_key:
                    if ("openai.azure.com" in url_lower) or ("api-version=" in url_lower):
                        req_headers["api-key"] = api_key
                    else:
                        req_headers["Authorization"] = f"Bearer {api_key}"
                if "openrouter.ai" in url_lower:
                    req_headers["HTTP-Referer"] = "https://heyclicky.com"
                    req_headers["X-Title"] = "Jarvis"
                step_user_content: Any = (
                    [
                        {"type": "text", "text": user_step_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
                    ]
                    if jpeg_b64
                    else user_step_prompt
                )
                req_payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": agent_system_prompt},
                        {"role": "user", "content": step_user_content},
                    ],
                    "max_tokens": 1600,
                    "stream": True,
                }
                if "api.river.ai" in url_lower:
                    req_payload["reasoning_effort"] = "none"

            try:
                with requests.post(
                    resolved_url,
                    headers=req_headers,
                    json=req_payload,
                    stream=True,
                    timeout=45,
                ) as response:
                    if not response.ok:
                        last_err = f"{endpoint_cfg['provider_label']} HTTP {response.status_code}: {response.text[:180]}"
                        continue
                    response.encoding = "utf-8"
                    plain_lines: List[str] = []
                    saw_sse = False
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if turn_generation and self._generation_id != turn_generation:
                            return (False, jarvis_response_parser.ParsedTurnResponse(), "Turn cancelled.")
                        if not raw_line:
                            continue
                        stripped_line = raw_line.strip()
                        if stripped_line.startswith("data:"):
                            saw_sse = True
                            data_str = stripped_line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                            except Exception:
                                continue
                        elif stripped_line.startswith("event:"):
                            saw_sse = True
                            continue
                        elif (not saw_sse) and stripped_line.startswith("{"):
                            try:
                                chunk_json = json.loads(stripped_line)
                            except Exception:
                                plain_lines.append(raw_line)
                                continue
                        else:
                            if not saw_sse:
                                plain_lines.append(raw_line)
                            continue
                        delta_text = self.extract_universal_sse_delta_text(chunk_json, wire_proto)
                        if delta_text:
                            extractor.feed_delta(delta_text)
                    if (not saw_sse) and plain_lines:
                        try:
                            full_obj = json.loads("\n".join(plain_lines))
                            full_text = self.extract_universal_sse_delta_text(full_obj, wire_proto)
                            if full_text:
                                extractor.feed_delta(full_text)
                        except Exception:
                            pass
                extractor.flush_remaining()
                return (True, extractor.build_final_response(), "")
            except Exception as step_exc:
                last_err = f"{endpoint_cfg['provider_label']} error: {step_exc}"
                continue

        return (False, jarvis_response_parser.ParsedTurnResponse(), last_err)


# ---------------------------------------------------------------------------
# Smart Dictation Clipboard + Paste Injector
# ---------------------------------------------------------------------------


class SmartDictationInjector:
    """Copies finalized dictation text to the clipboard as UTF-16 and sends Ctrl+V."""

    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002
    VK_CONTROL = 0x11
    VK_V = 0x56
    KEYEVENTF_KEYUP = 0x0002

    @classmethod
    def type_text_into_active_window(cls, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        try:
            user32 = ctypes.windll.user32
            cls._set_clipboard_unicode(cleaned + " ")
            time.sleep(0.04)
            user32.keybd_event(cls.VK_CONTROL, 0, 0, 0)
            user32.keybd_event(cls.VK_V, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(cls.VK_V, 0, cls.KEYEVENTF_KEYUP, 0)
            user32.keybd_event(cls.VK_CONTROL, 0, cls.KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    @classmethod
    def _set_clipboard_unicode(cls, text: str) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]

        encoded = (text + "\0").encode("utf-16le")
        mem_handle = kernel32.GlobalAlloc(cls.GMEM_MOVEABLE, len(encoded))
        if not mem_handle:
            return
        ptr = kernel32.GlobalLock(mem_handle)
        if ptr:
            ctypes.memmove(ptr, encoded, len(encoded))
            kernel32.GlobalUnlock(mem_handle)
        if user32.OpenClipboard(None):
            user32.EmptyClipboard()
            user32.SetClipboardData(cls.CF_UNICODETEXT, mem_handle)
            user32.CloseClipboard()


# ---------------------------------------------------------------------------
# Global Hotkey Listener
# ---------------------------------------------------------------------------


class GlobalHotkeyService(QObject):
    """
    Listen-only global keyboard service emitting thread-safe Qt signals for all shortcuts:
    - Ctrl + Alt (hold): Push-to-Talk + Draw-to-Ask
    - Ctrl x 2 (clean double tap): Toggle Notch Text Composer
    - Ctrl + Win (hold) / Ctrl + Win x 2: Push-to-Dictate / Hands-Free Dictation
    - Escape: Stop / Clear
    - Left / Right arrows: Cycle ArcDock cards when Home peek is open
    """

    pushToTalkPressed = Signal()
    pushToTalkReleased = Signal()
    doubleCtrlTapped = Signal()
    dictationPressed = Signal()
    dictationReleased = Signal()
    escapePressed = Signal()
    leftArrowPressed = Signal()
    rightArrowPressed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._ctrl_down = False
        self._alt_down = False
        self._win_down = False
        self._other_key_pressed_during_ctrl = False

        self._ptt_active = False
        self._dictate_active = False
        self._hands_free_dictation = False

        self._ctrl_press_time = 0.0
        self._last_ctrl_tap_release_time = 0.0
        self._last_win_tap_time = 0.0
        self._listener: Optional[keyboard.Listener] = None

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(
            on_press=self._handle_press,
            on_release=self._handle_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _is_ctrl(key: Any) -> bool:
        return key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)

    @staticmethod
    def _is_alt(key: Any) -> bool:
        return key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr)

    @staticmethod
    def _is_win(key: Any) -> bool:
        return key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)

    def _handle_press(self, key: Any) -> None:
        now = time.perf_counter()

        if key == keyboard.Key.esc:
            if self._hands_free_dictation:
                self._hands_free_dictation = False
                self._dictate_active = False
                self.dictationReleased.emit()
            self.escapePressed.emit()
            return

        if key == keyboard.Key.left:
            self.leftArrowPressed.emit()
        elif key == keyboard.Key.right:
            self.rightArrowPressed.emit()

        if self._is_ctrl(key):
            if not self._ctrl_down:
                self._ctrl_down = True
                self._ctrl_press_time = now
                self._other_key_pressed_during_ctrl = False
        elif self._is_alt(key):
            self._alt_down = True
            self._other_key_pressed_during_ctrl = True
        elif self._is_win(key):
            self._win_down = True
            self._other_key_pressed_during_ctrl = True
            if self._ctrl_down:
                if (now - self._last_win_tap_time) < 0.45:
                    self._hands_free_dictation = not self._hands_free_dictation
                self._last_win_tap_time = now
        else:
            self._other_key_pressed_during_ctrl = True

        if self._ctrl_down and self._alt_down and not self._ptt_active:
            self._ptt_active = True
            self.pushToTalkPressed.emit()

        if self._ctrl_down and self._win_down and not self._dictate_active:
            self._dictate_active = True
            self.dictationPressed.emit()

    def _handle_release(self, key: Any) -> None:
        now = time.perf_counter()

        if self._is_ctrl(key):
            self._ctrl_down = False
            hold_duration = now - self._ctrl_press_time
            if (
                not self._other_key_pressed_during_ctrl
                and not self._alt_down
                and not self._win_down
                and hold_duration < 0.40
            ):
                if (now - self._last_ctrl_tap_release_time) < 0.48:
                    self._last_ctrl_tap_release_time = 0.0
                    self.doubleCtrlTapped.emit()
                else:
                    self._last_ctrl_tap_release_time = now
        elif self._is_alt(key):
            self._alt_down = False
        elif self._is_win(key):
            self._win_down = False

        if self._ptt_active and not (self._ctrl_down and self._alt_down):
            self._ptt_active = False
            self.pushToTalkReleased.emit()

        if (
            self._dictate_active
            and not self._hands_free_dictation
            and not (self._ctrl_down and self._win_down)
        ):
            self._dictate_active = False
            self.dictationReleased.emit()


# ---------------------------------------------------------------------------
# System Telemetry, Live Web Lookup, Continuous Screen Monitor & Safe Agent
# ---------------------------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("sullAvailExtendedVirtual", ctypes.c_uint64),
    ]


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


class SystemTelemetryService:
    """Reads live Windows disk space, RAM availability, and battery status accurately."""

    DRIVE_FIXED = 3

    @classmethod
    def get_storage_summary(cls) -> str:
        drive_reports: List[str] = []
        kernel32 = ctypes.windll.kernel32 if sys.platform == "win32" else None
        for ascii_code in range(ord("C"), ord("Z") + 1):
            drive_letter = chr(ascii_code)
            root_path = f"{drive_letter}:\\"
            if not os.path.exists(root_path):
                continue
            if kernel32 is not None:
                try:
                    drive_type = int(kernel32.GetDriveTypeW(root_path))
                    if drive_type != cls.DRIVE_FIXED and drive_letter != "C":
                        continue
                except Exception:
                    pass
            try:
                usage = shutil.disk_usage(root_path)
            except Exception:
                continue
            if usage.total <= 0:
                continue
            gb_divisor = float(1024 ** 3)
            free_gb = usage.free / gb_divisor
            total_gb = usage.total / gb_divisor
            used_pct = ((usage.total - usage.free) / float(usage.total)) * 100.0
            drive_reports.append(
                f"{drive_letter}: {free_gb:.1f} GB free of {total_gb:.1f} GB ({used_pct:.0f}% used)"
            )

        if not drive_reports:
            try:
                fallback_usage = shutil.disk_usage(os.path.abspath(os.sep))
                gb_divisor = float(1024 ** 3)
                free_gb = fallback_usage.free / gb_divisor
                total_gb = fallback_usage.total / gb_divisor
                return f"Available storage: {free_gb:.1f} GB free of {total_gb:.1f} GB."
            except Exception:
                return "Could not read disk storage details right now."
        return "Storage space: " + "; ".join(drive_reports) + "."

    @staticmethod
    def get_memory_summary() -> str:
        if sys.platform != "win32":
            return "Memory telemetry is only available on Windows."
        try:
            mem_status = _MemoryStatusEx()
            mem_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                gb_divisor = float(1024 ** 3)
                avail_gb = mem_status.ullAvailPhys / gb_divisor
                total_gb = mem_status.ullTotalPhys / gb_divisor
                used_pct = int(mem_status.dwMemoryLoad)
                return f"Memory usage is {used_pct}% with {avail_gb:.1f} GB free of {total_gb:.1f} GB RAM."
        except Exception:
            pass
        return "Could not read RAM usage right now."

    @staticmethod
    def get_battery_summary() -> str:
        if sys.platform != "win32":
            return "Battery telemetry is only available on Windows."
        try:
            power_status = _SystemPowerStatus()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power_status)):
                ac_online = int(power_status.ACLineStatus) == 1
                pct = int(power_status.BatteryLifePercent)
                if pct == 255 or int(power_status.BatteryFlag) == 128:
                    return "Desktop PC running on AC power (no battery installed)."
                power_state = "plugged in" if ac_online else "on battery"
                return f"Battery is at {pct}% and currently {power_state}."
        except Exception:
            pass
        return "Could not read battery status right now."

    @classmethod
    def get_full_system_snapshot(cls) -> str:
        """Returns combined live storage, RAM, and battery telemetry for J.A.R.V.I.S. awareness."""
        return (
            f"{cls.get_storage_summary()}\n"
            f"{cls.get_memory_summary()}\n"
            f"{cls.get_battery_summary()}"
        )


class LiveWebLookupService:
    """Fetches real-time stock quotes and concise factual web context alongside Chrome search."""

    KNOWN_STOCK_TICKERS: Dict[str, Tuple[str, str]] = {
        "nvidia": ("NVDA", "Nvidia"),
        "nvda": ("NVDA", "Nvidia"),
        "apple": ("AAPL", "Apple"),
        "aapl": ("AAPL", "Apple"),
        "microsoft": ("MSFT", "Microsoft"),
        "msft": ("MSFT", "Microsoft"),
        "tesla": ("TSLA", "Tesla"),
        "tsla": ("TSLA", "Tesla"),
        "google": ("GOOGL", "Alphabet"),
        "alphabet": ("GOOGL", "Alphabet"),
        "googl": ("GOOGL", "Alphabet"),
        "amazon": ("AMZN", "Amazon"),
        "amzn": ("AMZN", "Amazon"),
        "meta": ("META", "Meta"),
        "amd": ("AMD", "AMD"),
        "intel": ("INTC", "Intel"),
        "intc": ("INTC", "Intel"),
        "netflix": ("NFLX", "Netflix"),
        "nflx": ("NFLX", "Netflix"),
        "bitcoin": ("BTC-USD", "Bitcoin"),
        "btc": ("BTC-USD", "Bitcoin"),
        "ethereum": ("ETH-USD", "Ethereum"),
        "eth": ("ETH-USD", "Ethereum"),
    }

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
    }

    @classmethod
    def resolve_stock_ticker(cls, query_text: str) -> Optional[Tuple[str, str]]:
        lowered = (query_text or "").lower()
        for keyword, ticker_pair in cls.KNOWN_STOCK_TICKERS.items():
            if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                return ticker_pair
        uppercase_tokens = re.findall(r"\b([A-Z]{2,5})\b", query_text or "")
        ignored_uppercase = {"WHAT", "PRICE", "STOCK", "SHARE", "CHROME", "SHOW", "TELL", "TODAY", "LIVE", "USD"}
        for candidate_symbol in uppercase_tokens:
            if candidate_symbol not in ignored_uppercase:
                return candidate_symbol, candidate_symbol
        return None

    @classmethod
    def fetch_stock_quote_summary(cls, query_text: str, timeout_seconds: float = 4.0) -> str:
        resolved = cls.resolve_stock_ticker(query_text)
        if resolved is None:
            return ""
        ticker_symbol, company_name = resolved
        try:
            response = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker_symbol)}?interval=1d&range=5d",
                headers=cls.DEFAULT_HEADERS,
                timeout=timeout_seconds,
            )
            if not response.ok:
                return ""
            payload = response.json()
            result_list = payload.get("chart", {}).get("result") or []
            if not result_list:
                return ""
            meta = result_list[0].get("meta") or {}
            market_price = meta.get("regularMarketPrice")
            if market_price is None:
                return ""
            currency = str(meta.get("currency") or "USD").upper()
            previous_close = meta.get("chartPreviousClose") or meta.get("previousClose")
            if previous_close:
                diff_val = float(market_price) - float(previous_close)
                diff_pct = (diff_val / float(previous_close)) * 100.0
                sign_prefix = "+" if diff_val >= 0 else ""
                return (
                    f"{company_name} ({ticker_symbol}) live quote: {float(market_price):.2f} {currency} "
                    f"({sign_prefix}{diff_val:.2f}, {sign_prefix}{diff_pct:.2f}% vs prior close)."
                )
            return f"{company_name} ({ticker_symbol}) live quote: {float(market_price):.2f} {currency}."
        except Exception:
            return ""

    @classmethod
    def fetch_general_web_summary(cls, query_text: str, timeout_seconds: float = 4.0) -> str:
        cleaned_query = (query_text or "").strip()
        if not cleaned_query:
            return ""
        snippets: List[str] = []
        try:
            ddg_response = requests.get(
                f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(cleaned_query)}&format=json&no_html=1&skip_disambig=1",
                headers=cls.DEFAULT_HEADERS,
                timeout=timeout_seconds,
            )
            if ddg_response.ok:
                ddg_json = ddg_response.json()
                answer_text = str(ddg_json.get("Answer") or "").strip()
                abstract_text = str(ddg_json.get("AbstractText") or "").strip()
                if answer_text:
                    snippets.append(f"Instant answer: {answer_text[:220]}")
                elif abstract_text:
                    snippets.append(f"Reference summary: {abstract_text[:220]}")
        except Exception:
            pass

        try:
            rss_response = requests.get(
                f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(cleaned_query)}&hl=en-US&gl=US&ceid=US:en",
                headers=cls.DEFAULT_HEADERS,
                timeout=timeout_seconds,
            )
            if rss_response.ok and rss_response.text:
                raw_titles = re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", rss_response.text, re.DOTALL)
                filtered_headlines: List[str] = []
                for raw_title in raw_titles[1:5]:
                    cleaned_headline = re.sub(r"\s+", " ", raw_title).strip()
                    if cleaned_headline and "Google News" not in cleaned_headline:
                        filtered_headlines.append(cleaned_headline[:120])
                    if len(filtered_headlines) >= 2:
                        break
                if filtered_headlines:
                    snippets.append("Recent headlines: " + " | ".join(filtered_headlines))
        except Exception:
            pass

        return "\n".join(snippets)

    @classmethod
    def fetch_live_web_context(cls, query_text: str, timeout_seconds: float = 4.5) -> str:
        context_parts: List[str] = []
        stock_summary = cls.fetch_stock_quote_summary(query_text, timeout_seconds=timeout_seconds)
        if stock_summary:
            context_parts.append(stock_summary)
        general_summary = cls.fetch_general_web_summary(query_text, timeout_seconds=timeout_seconds)
        if general_summary:
            context_parts.append(general_summary)
        return "\n".join(context_parts).strip()


@dataclass
class AdminPermissionRequest:
    """Represents a potentially destructive action requiring explicit admin confirmation."""

    action_title: str
    risk_description: str
    is_permanently_blocked: bool = False
    approved_commands: List[jarvis_response_parser.DrawingCommand] = field(default_factory=list)


@dataclass
class ContextualIntentPlan:
    """High-level execution route selected for a natural user prompt."""

    intent_kind: str
    spoken_step_intro: str = ""
    telemetry_response: str = ""
    search_query: str = ""
    agent_commands: List[jarvis_response_parser.DrawingCommand] = field(default_factory=list)
    admin_request: Optional[AdminPermissionRequest] = None


class AgentScreenMonitorService(QObject):
    """
    Periodically samples the active foreground window title and lightweight screen diffs
    so Jarvis knows what is currently visible on screen during agent & tutoring sessions.
    """

    foregroundWindowChanged = Signal(str)
    screenActivityDetected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._poll_counter = 0
        self._current_window_title = ""
        self._last_diff_ratio = 0.0
        self._last_snapshot_image: Optional[Image.Image] = None
        self._recent_transitions: List[str] = []

    @staticmethod
    def get_foreground_window_title() -> str:
        try:
            user32 = ctypes.windll.user32
            active_hwnd = int(user32.GetForegroundWindow())
            if not active_hwnd:
                return ""
            title_buffer = ctypes.create_unicode_buffer(512)
            char_count = int(user32.GetWindowTextW(active_hwnd, title_buffer, len(title_buffer)))
            if char_count <= 0:
                return ""
            return title_buffer.value.strip()
        except Exception:
            return ""

    def poll_once(self) -> Dict[str, Any]:
        window_title = self.get_foreground_window_title()
        window_changed = False
        diff_ratio = self._last_diff_ratio
        activity_summary = ""

        with self._lock:
            self._poll_counter += 1
            poll_number = self._poll_counter
            if window_title and window_title != self._current_window_title:
                window_changed = True
                self._current_window_title = window_title
                stamp = datetime.now().strftime("%H:%M:%S")
                self._recent_transitions.append(f"[{stamp}] {window_title[:70]}")
                self._recent_transitions = self._recent_transitions[-8:]

        if (poll_number % 2 == 0) or window_changed:
            sampled_image = ScreenCaptureService.capture_logical_snapshot_image((160, 90))
            if sampled_image is not None:
                with self._lock:
                    if self._last_snapshot_image is not None:
                        diff_ratio = ScreenCaptureService.compute_screen_difference_ratio(
                            self._last_snapshot_image,
                            sampled_image,
                        )
                    self._last_snapshot_image = sampled_image
                    self._last_diff_ratio = diff_ratio

        if window_changed and window_title:
            self.foregroundWindowChanged.emit(window_title)
            activity_summary = f"Active window switched to {window_title}"
        elif diff_ratio >= 0.045:
            activity_summary = f"Screen content changed ({diff_ratio:.2f}) in {window_title or 'Desktop'}"

        if activity_summary:
            self.screenActivityDetected.emit(activity_summary)

        return {
            "window_title": window_title,
            "window_changed": window_changed,
            "diff_ratio": diff_ratio,
        }

    def get_live_screen_context_summary(self) -> str:
        live_title = self.get_foreground_window_title()
        with self._lock:
            if live_title:
                self._current_window_title = live_title
            active_title = self._current_window_title or "Desktop"
            transitions_text = "; ".join(self._recent_transitions[-4:]) or "None yet"
            last_diff = self._last_diff_ratio
        try:
            open_titles = DesktopAgentService.list_open_window_titles()[:8]
            open_windows_text = " | ".join(open_titles) if open_titles else "None"
        except Exception:
            open_windows_text = "Unavailable"
        return (
            f"Current foreground window: {active_title}\n"
            f"Open desktop windows: {open_windows_text}\n"
            f"Recent window transitions: {transitions_text}\n"
            f"Last screen change ratio: {last_diff:.3f}"
        )


class RoamiesHandsService:
    """Bridge to reverse-engineered D:\\Projects\\Roamies\\roamies-hands.exe (UIA + Windows OCR + Win32 acts).

    Provides hardware-accelerated Rust window inspection, OCR word bounding boxes, box refinement,
    and reliable multi-step UI automation with graceful fallback to WindowsUIAService.
    """

    DEFAULT_HANDS_PATH = r"D:\Projects\Roamies\roamies-hands.exe"
    RISKY_CLICK_WORDS = (
        "delete", "remove", "erase", "empty trash", "clear all", "pay", "purchase",
        "checkout", "buy now", "transfer", "send money", "submit order", "sign out",
        "log out", "uninstall", "format", "shutdown", "reset",
    )

    @classmethod
    def resolve_hands_exe(cls, override_path: str = "") -> Optional[str]:
        candidates = [
            (override_path or "").strip(),
            cls.DEFAULT_HANDS_PATH,
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    @classmethod
    def run_hands_json(
        cls,
        subcommand: str,
        *args: str,
        timeout_sec: float = 5.0,
        hands_exe: str = "",
    ) -> Dict[str, Any]:
        exe_path = cls.resolve_hands_exe(hands_exe)
        if exe_path is None:
            return {"ok": False, "error": "roamies-hands.exe not found"}
        try:
            proc_result = subprocess.run(
                [exe_path, subcommand, *[str(arg_item) for arg_item in args]],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            raw_output = (proc_result.stdout or "").strip()
            if not raw_output:
                return {"ok": False, "error": (proc_result.stderr or "empty output").strip()}
            return json.loads(raw_output)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @classmethod
    def list_windows(cls, hands_exe: str = "") -> List[Dict[str, Any]]:
        res = cls.run_hands_json("windows", timeout_sec=3.5, hands_exe=hands_exe)
        wins = res.get("windows") if isinstance(res, dict) else None
        if isinstance(wins, list) and wins:
            return wins
        fallback_titles = DesktopAgentService.list_open_window_titles()
        return [{"id": idx + 1, "app": title, "title": title, "frontApp": idx == 0} for idx, title in enumerate(fallback_titles)]

    @classmethod
    def locate_window(cls, spec: str = "front", hands_exe: str = "") -> Optional[Dict[str, Any]]:
        res = cls.run_hands_json("locate", spec or "front", timeout_sec=3.5, hands_exe=hands_exe)
        win_obj = res.get("window") if isinstance(res, dict) and res.get("ok") else None
        if isinstance(win_obj, dict):
            return win_obj
        if sys.platform == "win32":
            hwnd = DesktopAgentService._current_foreground_hwnd()
            if hwnd:
                rect = wintypes.RECT()
                if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    return {
                        "id": int(hwnd),
                        "title": DesktopAgentService._window_title(hwnd),
                        "bounds": [int(rect.left), int(rect.top), int(rect.right - rect.left), int(rect.bottom - rect.top)],
                    }
        return None

    @classmethod
    def read_window_controls(cls, spec: str = "front", hands_exe: str = "") -> Dict[str, Any]:
        target_spec = "front" if (not spec or spec.lower() == "front") else (
            spec if spec.startswith(("app:", "window:")) else f"app:{spec}"
        )
        res = cls.run_hands_json("ui", target_spec, timeout_sec=5.0, hands_exe=hands_exe)
        if isinstance(res, dict) and res.get("ok") and res.get("controls"):
            return res
        uia_items = WindowsUIAService.get_active_window_ui_elements(max_elements=30)
        return {
            "ok": True,
            "window": cls.locate_window(spec="front", hands_exe=hands_exe) or {"id": 0, "app": "Foreground", "title": ""},
            "controls": [
                {
                    "id": idx + 1,
                    "role": elem["control_type"],
                    "label": elem["name"],
                    "bounds": [elem["x"] - elem["width"] // 2, elem["y"] - elem["height"] // 2, elem["width"], elem["height"]],
                }
                for idx, elem in enumerate(uia_items)
            ],
        }

    @classmethod
    def find_text_bounds(
        cls,
        query_text: str,
        spec: str = "front",
        allow_scroll: bool = False,
        hands_exe: str = "",
    ) -> Optional[Tuple[int, int, int, int]]:
        needle = (query_text or "").strip()
        if not needle:
            return None
        target_spec = "front" if (not spec or spec.lower() in ("front", "screen")) else (
            spec if spec.startswith(("app:", "window:")) else f"app:{spec}"
        )
        args = [target_spec, needle] + (["scroll"] if allow_scroll else [])
        res = cls.run_hands_json("find-text", *args, timeout_sec=4.5, hands_exe=hands_exe)
        matches = res.get("matches") if isinstance(res, dict) else None
        if not isinstance(matches, list) or not matches:
            ocr_res = cls.run_hands_json("ocr-find", target_spec, needle, timeout_sec=5.0, hands_exe=hands_exe)
            matches = ocr_res.get("matches") if isinstance(ocr_res, dict) else None
        if isinstance(matches, list) and matches:
            first_hit = matches[0]
            bounds = first_hit.get("bounds") or first_hit.get("rect")
            if isinstance(bounds, list) and len(bounds) == 4:
                return (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
        return None

    @classmethod
    def refine_boxes(cls, rects_list: List[List[int]], window_spec: str = "", hands_exe: str = "") -> List[Tuple[int, int, int, int]]:
        args = [json.dumps(rects_list)] + ([window_spec] if window_spec else [])
        res = cls.run_hands_json("refine", *args, timeout_sec=3.5, hands_exe=hands_exe)
        boxes = res.get("boxes") if isinstance(res, dict) else None
        if isinstance(boxes, list):
            refined: List[Tuple[int, int, int, int]] = []
            for box_item in boxes:
                if isinstance(box_item, list) and len(box_item) == 4:
                    refined.append((int(box_item[0]), int(box_item[1]), int(box_item[2]), int(box_item[3])))
            return refined
        return []

    @classmethod
    def is_risky_click_label(cls, label_text: str) -> Optional[str]:
        lowered = (label_text or "").strip().lower()
        for word in cls.RISKY_CLICK_WORDS:
            if word in lowered:
                return word
        return None

    @classmethod
    def _parse_steps_spec(cls, steps_input: str) -> List[Dict[str, Any]]:
        trimmed = (steps_input or "").strip()
        if not trimmed:
            return []
        if trimmed.startswith("["):
            try:
                parsed = json.loads(trimmed)
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
            except Exception:
                pass
        parsed_steps: List[Dict[str, Any]] = []
        for segment in re.split(r"\s*\|\s*|\s*;\s*", trimmed):
            part = segment.strip()
            if not part:
                continue
            if ":" in part:
                action_raw, val_raw = part.split(":", 1)
                action = action_raw.strip().lower().replace(" ", "_")
                val = val_raw.strip()
                if action in ("click", "click_text"):
                    parsed_steps.append({"do": "click_text", "text": val})
                elif action == "click_at" and "," in val:
                    coord_parts = [float(num) for num in re.findall(r"\d+(?:\.\d+)?", val)]
                    if len(coord_parts) >= 2:
                        parsed_steps.append({"do": "click_at", "x": coord_parts[0], "y": coord_parts[1]})
                elif action == "type":
                    parsed_steps.append({"do": "type", "text": val})
                elif action == "key":
                    parsed_steps.append({"do": "key", "key": val})
                elif action == "scroll":
                    parsed_steps.append({"do": "scroll", "direction": val or "down", "amount": 4})
                elif action == "wait":
                    parsed_steps.append({"do": "wait", "seconds": float(val) if val else 0.5})
                else:
                    parsed_steps.append({"do": action, "text": val})
            else:
                parsed_steps.append({"do": "click_text", "text": part})
        return parsed_steps

    @classmethod
    def act_in_app(cls, target_spec: str, steps_input: str, hands_exe: str = "") -> Tuple[bool, str]:
        steps = cls._parse_steps_spec(steps_input)
        if not steps:
            return False, "No steps supplied for act_in_app"
        for step_obj in steps:
            candidate_label = str(step_obj.get("text") or step_obj.get("label") or step_obj.get("option") or "")
            risky_hit = cls.is_risky_click_label(candidate_label)
            if risky_hit and not step_obj.get("confirmed"):
                return False, f"Blocked risky UI click '{candidate_label}' ({risky_hit}) — requires admin confirmation"

        payload: Dict[str, Any] = {"steps": steps}
        clean_target = (target_spec or "front").strip()
        if clean_target.lower() not in ("", "front"):
            if clean_target.startswith("window:"):
                payload["window"] = int(clean_target.split(":", 1)[1])
            else:
                payload["app"] = clean_target

        res = cls.run_hands_json("act", json.dumps(payload), timeout_sec=10.0, hands_exe=hands_exe)
        if isinstance(res, dict) and res.get("ok"):
            return True, f"Executed {len(steps)} step(s) in {clean_target or 'active window'}"

        # Fallback to native WindowsUIAService + Win32 when roamies-hands cannot match target window
        executed = 0
        for step_obj in steps:
            do_verb = str(step_obj.get("do") or "")
            if do_verb in ("click_text", "click") and step_obj.get("text"):
                step_ok, _ = WindowsUIAService.find_and_click_control(str(step_obj["text"]))
                if step_ok:
                    executed += 1
            elif do_verb == "click_at":
                step_ok, _ = WindowsUIAService.find_and_click_control(f"{step_obj.get('x')},{step_obj.get('y')}")
                if step_ok:
                    executed += 1
            elif do_verb == "type" and step_obj.get("text"):
                if DesktopAgentService._send_text_to_focused_window(str(step_obj["text"]), focus_address_bar=False, submit_enter=False):
                    executed += 1
            elif do_verb == "key" and step_obj.get("key") and pyautogui is not None:
                pyautogui.press(str(step_obj["key"]))
                executed += 1
            elif do_verb == "wait":
                time.sleep(max(0.05, min(3.0, float(step_obj.get("seconds") or 0.4))))
                executed += 1
        if executed > 0:
            return True, f"Completed {executed}/{len(steps)} UI step(s)"
        return False, str(res.get("error") or "act_in_app failed")


class WindowsUIAService:
    """Native Windows UI Automation (UIA3 via comtypes) helper for reliable element grounding & clicking.

    Follows the architectural pattern used by Windows-MCP and Microsoft UFO: inspect live
    interactive OS accessibility elements (buttons, inputs, tabs, links, menu items) instead
    of relying solely on blind pixel coordinates.
    """

    INTERACTIVE_CONTROL_TYPES = {
        50000: "Button",
        50002: "CheckBox",
        50003: "ComboBox",
        50004: "Edit",
        50005: "Hyperlink",
        50007: "ListItem",
        50009: "Menu",
        50011: "MenuItem",
        50013: "RadioButton",
        50019: "TabItem",
        50024: "TreeItem",
        50030: "Document",
    }

    @classmethod
    def get_active_window_ui_elements(cls, max_elements: int = 24) -> List[Dict[str, Any]]:
        """Walks the foreground window UIA tree and returns named interactive elements with coordinates."""
        if sys.platform != "win32":
            return []
        try:
            import comtypes
            import comtypes.client
        except Exception:
            return []

        elements: List[Dict[str, Any]] = []
        com_initialized = False
        try:
            comtypes.CoInitialize()
            com_initialized = True
            uia_module = comtypes.client.GetModule("UIAutomationCore.dll")
            uia_client = comtypes.client.CreateObject(
                "{ff48dba4-60ef-4201-aa87-54103eef594e}",
                interface=uia_module.IUIAutomation,
            )
            hwnd = int(ctypes.windll.user32.GetForegroundWindow() or 0)
            root_element = uia_client.ElementFromHandle(hwnd) if hwnd else uia_client.GetFocusedElement()
            if not root_element:
                return []

            tree_walker = uia_client.ControlViewWalker
            work_queue: List[Tuple[Any, int]] = [(root_element, 0)]
            visited_nodes = 0
            max_nodes_to_visit = 130
            seen_keys = set()

            while work_queue and len(elements) < max_elements and visited_nodes < max_nodes_to_visit:
                current_node, depth = work_queue.pop(0)
                visited_nodes += 1
                try:
                    ctrl_type_id = int(getattr(current_node, "CurrentControlType", 0) or 0)
                    ctrl_name = str(getattr(current_node, "CurrentName", "") or "").strip()
                    is_offscreen = bool(getattr(current_node, "CurrentIsOffscreen", False))
                    rect = getattr(current_node, "CurrentBoundingRectangle", None)
                    if (
                        (not is_offscreen)
                        and ctrl_name
                        and (ctrl_type_id in cls.INTERACTIVE_CONTROL_TYPES)
                        and rect is not None
                    ):
                        width_px = int(rect.right - rect.left)
                        height_px = int(rect.bottom - rect.top)
                        if width_px >= 6 and height_px >= 6:
                            center_x = int(rect.left + width_px / 2)
                            center_y = int(rect.top + height_px / 2)
                            dedupe_key = (ctrl_type_id, ctrl_name[:40].lower(), center_x // 8, center_y // 8)
                            if dedupe_key not in seen_keys:
                                seen_keys.add(dedupe_key)
                                elements.append(
                                    {
                                        "name": ctrl_name[:64],
                                        "control_type": cls.INTERACTIVE_CONTROL_TYPES[ctrl_type_id],
                                        "x": center_x,
                                        "y": center_y,
                                        "width": width_px,
                                        "height": height_px,
                                    }
                                )
                except Exception:
                    pass

                if depth >= 4:
                    continue
                try:
                    child_node = tree_walker.GetFirstChildElement(current_node)
                    child_cap = 0
                    while child_node and child_cap < 28:
                        work_queue.append((child_node, depth + 1))
                        child_node = tree_walker.GetNextSiblingElement(child_node)
                        child_cap += 1
                except Exception:
                    continue
        except Exception:
            return []
        finally:
            if com_initialized:
                try:
                    import comtypes
                    comtypes.CoUninitialize()
                except Exception:
                    pass
        return elements

    @classmethod
    def format_ui_summary(cls, max_elements: int = 18) -> str:
        elements = cls.get_active_window_ui_elements(max_elements=max_elements)
        if not elements:
            return "(No interactive UIA controls exposed on current foreground window)"
        formatted_items = [
            f"{elem['control_type']} '{elem['name']}' @({elem['x']},{elem['y']})"
            for elem in elements
        ]
        return "; ".join(formatted_items)

    @classmethod
    def find_and_click_control(cls, target_spec: str) -> Tuple[bool, str]:
        cleaned_spec = (target_spec or "").strip()
        if not cleaned_spec:
            return False, "No UI target specified"

        coord_match = re.match(r"^\s*(\d{1,5})\s*,\s*(\d{1,5})\s*$", cleaned_spec)
        if coord_match:
            target_x = int(coord_match.group(1))
            target_y = int(coord_match.group(2))
            if pyautogui is not None:
                pyautogui.click(target_x, target_y)
                return True, f"Clicked screen coordinates ({target_x}, {target_y})"
            if sys.platform == "win32":
                user32 = ctypes.windll.user32
                user32.SetCursorPos(target_x, target_y)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                return True, f"Clicked screen coordinates ({target_x}, {target_y})"
            return False, "Mouse control unavailable"

        elements = cls.get_active_window_ui_elements(max_elements=40)
        if not elements:
            return False, f"Could not inspect active window controls for '{cleaned_spec}'"

        lowered_target = cleaned_spec.lower()
        exact_hit = next((elem for elem in elements if elem["name"].lower() == lowered_target), None)
        if exact_hit is None:
            exact_hit = next((elem for elem in elements if lowered_target in elem["name"].lower()), None)
        if exact_hit is None:
            target_words = [word for word in re.findall(r"[a-z0-9]+", lowered_target) if len(word) >= 2]
            if target_words:
                exact_hit = next(
                    (
                        elem
                        for elem in elements
                        if all(word in elem["name"].lower() for word in target_words)
                    ),
                    None,
                )
        if exact_hit is None:
            return False, f"Control '{cleaned_spec}' not found on foreground window"

        click_x = int(exact_hit["x"])
        click_y = int(exact_hit["y"])
        if pyautogui is not None:
            pyautogui.click(click_x, click_y)
            return True, f"Clicked {exact_hit['control_type']} '{exact_hit['name']}' at ({click_x}, {click_y})"
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            user32.SetCursorPos(click_x, click_y)
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)
            return True, f"Clicked {exact_hit['control_type']} '{exact_hit['name']}' at ({click_x}, {click_y})"
        return False, "Mouse control unavailable"


class ProjectWorkspaceService:
    """Safe project scaffolding, file reading/writing, and command execution rooted in D:\\Projects."""

    DEFAULT_PROJECTS_ROOT = r"D:\Projects"
    BLOCKED_BINARY_EXTENSIONS = {
        ".exe", ".bat", ".cmd", ".ps1", ".vbs", ".msi", ".dll",
        ".scr", ".com", ".hta", ".reg", ".sys", ".drv", ".lnk",
    }
    SENSITIVE_FILENAMES = {
        ".env", ".env.local", ".env.production", ".env.development",
        "user_settings.json", "id_rsa", "id_ed25519", "credentials.json",
    }
    DESTRUCTIVE_COMMAND_PATTERNS = (
        "rm -rf", "rm -fr", "remove-item", "rmdir /s", "rd /s", "del /s", "del /q",
        "format ", "format-", "diskpart", "bcdedit", "cipher /w", "reg delete",
        "shutdown", "restart-computer", "stop-computer", "taskkill /f",
        "git push --force", "git reset --hard", "drop database", "drop table",
    )

    @classmethod
    def get_projects_root(cls, projects_root: str = "") -> str:
        raw_root = (projects_root or cls.DEFAULT_PROJECTS_ROOT).strip()
        return os.path.realpath(raw_root)

    @classmethod
    def list_projects(cls, projects_root: str = "") -> List[str]:
        root_dir = cls.get_projects_root(projects_root)
        if not os.path.isdir(root_dir):
            return []
        try:
            project_names = [
                entry_name
                for entry_name in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, entry_name)) and (not entry_name.startswith("."))
            ]
            return sorted(project_names, key=lambda item_name: item_name.lower())
        except OSError:
            return []

    @classmethod
    def resolve_project_dir(cls, query: str, projects_root: str = "") -> Optional[str]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return None
        root_dir = cls.get_projects_root(projects_root)
        if os.path.isdir(cleaned_query):
            resolved_direct = os.path.realpath(cleaned_query)
            if resolved_direct.lower().startswith(root_dir.lower()):
                return resolved_direct
        candidate_names = cls.list_projects(root_dir)
        lowered_query = cleaned_query.lower()
        for folder_name in candidate_names:
            if folder_name.lower() == lowered_query:
                return os.path.join(root_dir, folder_name)
        normalized_query = re.sub(r"[^a-z0-9]+", "", lowered_query)
        if normalized_query:
            for folder_name in candidate_names:
                normalized_folder = re.sub(r"[^a-z0-9]+", "", folder_name.lower())
                if normalized_folder == normalized_query or normalized_query in normalized_folder:
                    return os.path.join(root_dir, folder_name)
        return None

    @classmethod
    def resolve_safe_workspace_path(
        cls,
        raw_path: str,
        projects_root: str = "",
        default_to_projects: bool = True,
    ) -> Tuple[Optional[str], str]:
        cleaned_path = (raw_path or "").strip()
        if not cleaned_path:
            return None, "No file path specified"

        root_dir = cls.get_projects_root(projects_root)
        user_home = os.path.expanduser("~")
        desktop_dir = os.path.realpath(os.path.join(user_home, "Desktop"))
        documents_dir = os.path.realpath(os.path.join(user_home, "Documents"))
        allowed_roots = [root_dir, desktop_dir, documents_dir]

        slash_normalized = cleaned_path.replace("\\", "/")
        lowered_slash = slash_normalized.lower()
        if os.path.isabs(cleaned_path):
            resolved_candidate = os.path.realpath(cleaned_path)
        elif lowered_slash.startswith("desktop/"):
            inner_part = slash_normalized[len("desktop/") :].lstrip("/")
            resolved_candidate = os.path.realpath(os.path.join(desktop_dir, inner_part))
        elif lowered_slash.startswith("documents/"):
            inner_part = slash_normalized[len("documents/") :].lstrip("/")
            resolved_candidate = os.path.realpath(os.path.join(documents_dir, inner_part))
        elif lowered_slash.startswith("projects/"):
            inner_part = slash_normalized[len("projects/") :].lstrip("/")
            resolved_candidate = os.path.realpath(os.path.join(root_dir, inner_part))
        elif default_to_projects:
            resolved_candidate = os.path.realpath(os.path.join(root_dir, slash_normalized.lstrip("/")))
        else:
            resolved_candidate = os.path.realpath(os.path.join(desktop_dir, slash_normalized.lstrip("/")))

        if ".." in slash_normalized.split("/"):
            return None, "Path traversal ('..') is blocked"

        is_inside_allowed = any(
            resolved_candidate.lower() == allowed_root.lower()
            or resolved_candidate.lower().startswith(allowed_root.lower() + os.sep)
            for allowed_root in allowed_roots
        )
        if not is_inside_allowed:
            return None, "Path escapes D:\\Projects / Desktop / Documents workspace"

        basename_lower = os.path.basename(resolved_candidate).lower()
        if basename_lower in cls.SENSITIVE_FILENAMES or basename_lower.startswith(".env."):
            return None, f"Refused access to protected config/secret file: {basename_lower}"
        return resolved_candidate, ""

    @classmethod
    def write_workspace_file(
        cls,
        raw_path: str,
        content: str,
        projects_root: str = "",
        default_to_projects: bool = True,
    ) -> Tuple[bool, str]:
        resolved_path, err_msg = cls.resolve_safe_workspace_path(
            raw_path,
            projects_root=projects_root,
            default_to_projects=default_to_projects,
        )
        if resolved_path is None:
            return False, err_msg
        _, out_ext = os.path.splitext(resolved_path.lower())
        if out_ext in cls.BLOCKED_BINARY_EXTENSIONS:
            return False, f"Refused to write executable/binary extension: {out_ext}"
        if not out_ext:
            resolved_path += ".txt"
        decoded_content = (content or "").replace("\\n", "\n") if ("\\n" in (content or "") and "\n" not in (content or "")) else (content or "")
        try:
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(decoded_content)
            return True, f"Wrote {resolved_path}"
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def read_workspace_file(
        cls,
        raw_path: str,
        projects_root: str = "",
        max_chars: int = 3200,
    ) -> Tuple[bool, str]:
        resolved_path, err_msg = cls.resolve_safe_workspace_path(
            raw_path,
            projects_root=projects_root,
            default_to_projects=True,
        )
        if resolved_path is None:
            return False, err_msg
        if not os.path.isfile(resolved_path):
            return False, f"File not found: {resolved_path}"
        try:
            with open(resolved_path, "r", encoding="utf-8", errors="replace") as file_handle:
                text_data = file_handle.read(max_chars)
            return True, text_data
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def run_project_command(
        cls,
        command_text: str,
        cwd_rel: str = "",
        projects_root: str = "",
    ) -> Tuple[bool, str]:
        cleaned_cmd = (command_text or "").strip()
        if not cleaned_cmd:
            return False, "No project command provided"
        lowered_cmd = cleaned_cmd.lower()
        if any(pattern in lowered_cmd for pattern in cls.DESTRUCTIVE_COMMAND_PATTERNS):
            return False, "Blocked destructive command — explicit admin confirmation required"

        root_dir = cls.get_projects_root(projects_root)
        if cwd_rel.strip():
            resolved_cwd, err_msg = cls.resolve_safe_workspace_path(
                cwd_rel.strip(),
                projects_root=root_dir,
                default_to_projects=True,
            )
            if resolved_cwd is None:
                return False, err_msg
            os.makedirs(resolved_cwd, exist_ok=True)
        else:
            resolved_cwd = root_dir
            os.makedirs(resolved_cwd, exist_ok=True)

        is_background_server = any(
            server_hint in lowered_cmd
            for server_hint in ("http.server", "npm run dev", "npm start", "vite", "uvicorn", "flask run")
        )
        try:
            if is_background_server:
                proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", cleaned_cmd],
                    cwd=resolved_cwd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                return True, f"Started server (PID {proc.pid}) in {resolved_cwd}"

            proc_result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cleaned_cmd],
                cwd=resolved_cwd,
                capture_output=True,
                text=True,
                timeout=25.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            combined_out = ((proc_result.stdout or "") + "\n" + (proc_result.stderr or "")).strip()
            compact_out = re.sub(r"\s+", " ", combined_out)[:360]
            if proc_result.returncode == 0:
                return True, compact_out or f"Command succeeded in {os.path.basename(resolved_cwd)}"
            return False, f"Exit {proc_result.returncode}: {compact_out}"
        except subprocess.TimeoutExpired:
            return False, "Project command timed out after 25s"
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def open_project(
        cls,
        project_query: str,
        tool_name: str = "auto",
        projects_root: str = "",
    ) -> Tuple[bool, str]:
        root_dir = cls.get_projects_root(projects_root)
        target_dir = cls.resolve_project_dir(project_query, root_dir)
        if target_dir is None:
            return False, f"Could not find project '{project_query}' in {root_dir}"

        chosen_tool = (tool_name or "auto").strip().lower()
        if chosen_tool in ("vscode", "code", "auto"):
            code_exe = shutil.which("code") or shutil.which("code.cmd")
            if code_exe or chosen_tool in ("vscode", "code"):
                try:
                    subprocess.Popen(
                        [code_exe or "code", target_dir],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        shell=(code_exe is None),
                    )
                    return True, f"Opened {os.path.basename(target_dir)} in VS Code"
                except Exception:
                    if chosen_tool in ("vscode", "code"):
                        return False, f"Could not launch VS Code for {target_dir}"

        if chosen_tool in ("terminal", "powershell", "cmd"):
            try:
                subprocess.Popen(
                    ["powershell", "-NoExit", "-Command", f"Set-Location -LiteralPath '{target_dir}'"],
                    creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                )
                return True, f"Opened terminal at {os.path.basename(target_dir)}"
            except Exception as exc:
                return False, str(exc)

        try:
            os.startfile(target_dir)
            return True, f"Opened project folder {os.path.basename(target_dir)}"
        except Exception as exc:
            return False, str(exc)


class DesktopAgentService:
    """Executes bounded desktop & browser actions requested by the AI or directly by the user."""

    _last_app_launch_timestamp: float = 0.0
    _last_launched_target: str = ""
    _pre_launch_hwnd: int = 0
    _recent_browser_targets: List[str] = []

    VK_CONTROL = 0x11
    VK_RETURN = 0x0D
    VK_1 = 0x31
    VK_2 = 0x32
    VK_C = 0x43
    VK_F = 0x46
    VK_L = 0x4C
    VK_V = 0x56
    VK_W = 0x57
    KEYEVENTF_KEYUP = 0x0002
    SW_RESTORE = 9
    SWP_SHOWWINDOW = 0x0040
    SPI_GETWORKAREA = 0x0030

    MS_SETTINGS_URI_MAP: Dict[str, Tuple[str, str]] = {
        "storage": ("ms-settings:storagesense", "Windows Storage Settings"),
        "disk": ("ms-settings:storagesense", "Windows Storage Settings"),
        "space": ("ms-settings:storagesense", "Windows Storage Settings"),
        "battery": ("ms-settings:batterysaver", "Battery & Power Settings"),
        "power": ("ms-settings:powersleep", "Power & Sleep Settings"),
        "display": ("ms-settings:display", "Display Settings"),
        "brightness": ("ms-settings:display", "Display Settings"),
        "sound": ("ms-settings:sound", "Sound Settings"),
        "audio": ("ms-settings:sound", "Sound Settings"),
        "bluetooth": ("ms-settings:bluetooth", "Bluetooth & Devices Settings"),
        "wifi": ("ms-settings:network-wifi", "Wi-Fi Settings"),
        "wi-fi": ("ms-settings:network-wifi", "Wi-Fi Settings"),
        "network": ("ms-settings:network", "Network Settings"),
        "update": ("ms-settings:windowsupdate", "Windows Update"),
        "apps": ("ms-settings:appsfeatures", "Installed Apps Settings"),
        "uninstall": ("ms-settings:appsfeatures", "Installed Apps Settings"),
        "ram": ("ms-settings:about", "System About Settings"),
        "specs": ("ms-settings:about", "System About Settings"),
        "about": ("ms-settings:about", "System About Settings"),
    }

    KNOWN_URL_ALIASES: Dict[str, str] = {
        "chatgpt": "https://chatgpt.com",
        "chat.openai.com": "https://chatgpt.com",
        "google": "https://www.google.com",
        "youtube": "https://www.youtube.com",
        "github": "https://github.com",
        "gmail": "https://mail.google.com",
        "docs": "https://docs.google.com",
        "drive": "https://drive.google.com",
        "claude": "https://claude.ai",
        "notion": "https://www.notion.so",
        "twitter": "https://x.com",
        "x": "https://x.com",
        "x.com": "https://x.com",
        "twitter.com": "https://x.com",
        "reddit": "https://www.reddit.com",
        "linkedin": "https://www.linkedin.com",
        "instagram": "https://www.instagram.com",
        "facebook": "https://www.facebook.com",
        "whatsapp": "https://web.whatsapp.com",
        "spotify": "https://open.spotify.com",
        "netflix": "https://www.netflix.com",
        "perplexity": "https://www.perplexity.ai",
        "wikipedia": "https://en.wikipedia.org",
        "amazon": "https://www.amazon.com",
        "flipkart": "https://www.flipkart.com",
        "leetcode": "https://leetcode.com",
        "stackoverflow": "https://stackoverflow.com",
    }

    _DEICTIC_SCREEN_REFERENCES = {
        "him",
        "her",
        "them",
        "this",
        "that",
        "it",
        "he",
        "she",
        "they",
        "this guy",
        "this person",
        "this man",
        "this woman",
        "this actor",
        "this topic",
        "this post",
        "this tweet",
        "this page",
        "this article",
    }

    SAFE_APP_LAUNCHERS: Dict[str, List[str]] = {
        "notepad": ["notepad.exe"],
        "calculator": ["calc.exe"],
        "calc": ["calc.exe"],
        "paint": ["mspaint.exe"],
        "mspaint": ["mspaint.exe"],
        "explorer": ["explorer.exe"],
        "files": ["explorer.exe"],
        "terminal": ["wt.exe"],
        "wt": ["wt.exe"],
        "cmd": ["cmd.exe"],
        "powershell": ["powershell.exe"],
        "vscode": ["code"],
        "code": ["code"],
        "chrome": ["cmd", "/c", "start", "", "chrome"],
        "edge": ["cmd", "/c", "start", "", "msedge"],
        "msedge": ["cmd", "/c", "start", "", "msedge"],
        "firefox": ["cmd", "/c", "start", "", "firefox"],
    }

    SAFE_KEY_ALIASES: Dict[str, str] = {
        "control": "ctrl",
        "windows": "win",
        "super": "win",
        "return": "enter",
        "escape": "esc",
        "pgdn": "pagedown",
        "pgup": "pageup",
    }

    ALLOWED_SINGLE_KEYS = {
        "enter",
        "tab",
        "esc",
        "space",
        "backspace",
        "delete",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f11",
    }

    @classmethod
    def normalize_target_url(cls, raw_target: str) -> Optional[str]:
        cleaned = (raw_target or "").strip().strip("\"'")
        if not cleaned or cleaned.lower() in ("default", "browser", "home", "new_tab"):
            return "https://www.google.com"
        lowered = cleaned.lower()
        if lowered in cls.KNOWN_URL_ALIASES:
            return cls.KNOWN_URL_ALIASES[lowered]
        if "://" not in cleaned:
            if "." in cleaned and " " not in cleaned:
                cleaned = f"https://{cleaned}"
            else:
                return None
        parsed = urllib.parse.urlparse(cleaned)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            return None
        return cleaned

    @staticmethod
    def _resolve_chrome_executable() -> Optional[str]:
        which_path = shutil.which("chrome")
        if which_path and os.path.exists(which_path):
            return which_path
        candidate_roots = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        for root_dir in candidate_roots:
            if not root_dir:
                continue
            candidate_exe = os.path.join(root_dir, "Google", "Chrome", "Application", "chrome.exe")
            if os.path.exists(candidate_exe):
                return candidate_exe
        return None

    @staticmethod
    def _current_foreground_hwnd() -> int:
        try:
            return int(ctypes.windll.user32.GetForegroundWindow())
        except Exception:
            return 0

    @classmethod
    def _wait_for_foreground_window_change(
        cls,
        previous_hwnd: int,
        timeout_seconds: float = 2.8,
    ) -> None:
        deadline = time.monotonic() + max(0.4, timeout_seconds)
        window_switched = False
        while time.monotonic() < deadline:
            active_hwnd = cls._current_foreground_hwnd()
            if active_hwnd and active_hwnd != previous_hwnd:
                window_switched = True
                break
            time.sleep(0.1)
        time.sleep(0.55 if window_switched else 0.9)

    @classmethod
    def _press_win32_combo(cls, virtual_keys: List[int]) -> None:
        user32 = ctypes.windll.user32
        for vk_code in virtual_keys:
            user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.03)
        for vk_code in reversed(virtual_keys):
            user32.keybd_event(vk_code, 0, cls.KEYEVENTF_KEYUP, 0)

    @classmethod
    def _send_text_to_focused_window(
        cls,
        text_to_type: str,
        focus_address_bar: bool = False,
        submit_enter: bool = False,
    ) -> bool:
        cleaned = (text_to_type or "")[:280].strip()
        if not cleaned:
            return False
        try:
            if focus_address_bar:
                cls._press_win32_combo([cls.VK_CONTROL, cls.VK_L])
                time.sleep(0.12)
            SmartDictationInjector._set_clipboard_unicode(cleaned)
            time.sleep(0.05)
            cls._press_win32_combo([cls.VK_CONTROL, cls.VK_V])
            if submit_enter:
                time.sleep(0.08)
                cls._press_win32_combo([cls.VK_RETURN])
            return True
        except Exception:
            if pyautogui is not None:
                try:
                    pyautogui.write(cleaned, interval=0.012)
                    if submit_enter:
                        pyautogui.press("enter")
                    return True
                except Exception:
                    return False
            return False

    @classmethod
    def _record_browser_target(cls, target_url: str) -> None:
        cleaned_url = (target_url or "").strip()
        if not cleaned_url:
            return
        if cleaned_url in cls._recent_browser_targets:
            cls._recent_browser_targets.remove(cleaned_url)
        cls._recent_browser_targets.append(cleaned_url)
        if len(cls._recent_browser_targets) > 8:
            cls._recent_browser_targets = cls._recent_browser_targets[-8:]

    @classmethod
    def _get_work_area_rect(cls) -> Tuple[int, int, int, int]:
        if sys.platform != "win32":
            return (0, 0, 1920, 1040)
        try:
            work_rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(cls.SPI_GETWORKAREA, 0, ctypes.byref(work_rect), 0):
                left_val = int(work_rect.left)
                top_val = int(work_rect.top)
                right_val = int(work_rect.right)
                bottom_val = int(work_rect.bottom)
                if right_val - left_val >= 400 and bottom_val - top_val >= 300:
                    return (left_val, top_val, right_val, bottom_val)
        except Exception:
            pass
        return (0, 0, 1920, 1040)

    @classmethod
    def _list_visible_chrome_hwnds(cls) -> List[int]:
        if sys.platform != "win32":
            return []
        found_hwnds: List[int] = []
        try:
            user32 = ctypes.windll.user32
            enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            def _on_window(candidate_hwnd: int, lparam_val: int) -> bool:
                if not user32.IsWindowVisible(candidate_hwnd):
                    return True
                class_buffer = ctypes.create_unicode_buffer(128)
                user32.GetClassNameW(candidate_hwnd, class_buffer, 128)
                if class_buffer.value != "Chrome_WidgetWin_1":
                    return True
                title_length = user32.GetWindowTextLengthW(candidate_hwnd)
                if title_length <= 0:
                    return True
                title_buffer = ctypes.create_unicode_buffer(title_length + 1)
                user32.GetWindowTextW(candidate_hwnd, title_buffer, title_length + 1)
                window_title = (title_buffer.value or "").strip()
                if not window_title or "jarvis" in window_title.lower():
                    return True
                win_rect = wintypes.RECT()
                if user32.GetWindowRect(candidate_hwnd, ctypes.byref(win_rect)):
                    win_width = int(win_rect.right) - int(win_rect.left)
                    win_height = int(win_rect.bottom) - int(win_rect.top)
                    if win_width >= 240 and win_height >= 180:
                        found_hwnds.append(int(candidate_hwnd))
                return True

            callback_ref = enum_proc_type(_on_window)
            user32.EnumWindows(callback_ref, 0)
        except Exception:
            pass
        return found_hwnds

    @classmethod
    def _list_all_windows(cls) -> List[Tuple[int, str]]:
        """Enumerate visible (including minimized) top-level desktop windows as (hwnd, title)."""
        found_windows: List[Tuple[int, str]] = []
        if sys.platform != "win32":
            return found_windows
        try:
            user32 = ctypes.windll.user32
            enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            ignored_titles = {"program manager", "windows input experience", "settings"}

            def _on_window(candidate_hwnd: int, _lparam: int) -> bool:
                if not user32.IsWindowVisible(candidate_hwnd):
                    return True
                title_length = user32.GetWindowTextLengthW(candidate_hwnd)
                if title_length <= 0:
                    return True
                title_buffer = ctypes.create_unicode_buffer(title_length + 1)
                user32.GetWindowTextW(candidate_hwnd, title_buffer, title_length + 1)
                window_title = (title_buffer.value or "").strip()
                if not window_title:
                    return True
                lowered_title = window_title.lower()
                if "jarvis" in lowered_title or lowered_title in ignored_titles:
                    return True
                is_minimized = bool(user32.IsIconic(candidate_hwnd))
                if is_minimized:
                    found_windows.append((int(candidate_hwnd), window_title))
                    return True
                win_rect = wintypes.RECT()
                if user32.GetWindowRect(candidate_hwnd, ctypes.byref(win_rect)):
                    win_width = int(win_rect.right) - int(win_rect.left)
                    win_height = int(win_rect.bottom) - int(win_rect.top)
                    if win_width >= 180 and win_height >= 120:
                        found_windows.append((int(candidate_hwnd), window_title))
                return True

            callback_ref = enum_proc_type(_on_window)
            user32.EnumWindows(callback_ref, 0)
        except Exception:
            pass
        return found_windows

    @classmethod
    def list_open_window_titles(cls) -> List[str]:
        """Return unique titles of visible/minimized open desktop windows."""
        titles: List[str] = []
        seen_lower: set[str] = set()
        for _hwnd_val, win_title in cls._list_all_windows():
            title_key = win_title.lower()
            if title_key not in seen_lower:
                seen_lower.add(title_key)
                titles.append(win_title)
        return titles

    @classmethod
    def _find_matching_window_hwnd(
        cls,
        window_query: str,
        excluded_hwnd: Optional[int] = None,
    ) -> Optional[int]:
        """Find an open window HWND matching a title or common app alias."""
        cleaned_query = (window_query or "").strip().lower()
        if not cleaned_query:
            return None
        alias_map = {
            "vscode": "visual studio code",
            "vs code": "visual studio code",
            "code": "visual studio code",
            "edge": "microsoft edge",
            "chrome": "google chrome",
            "explorer": "file explorer",
            "terminal": "windows terminal",
            "powershell": "powershell",
            "cmd": "command prompt",
            "word": "word",
            "excel": "excel",
            "powerpoint": "powerpoint",
        }
        candidate_queries = [cleaned_query]
        if cleaned_query in alias_map and alias_map[cleaned_query] not in candidate_queries:
            candidate_queries.append(alias_map[cleaned_query])

        open_windows = cls._list_all_windows()
        for query_term in candidate_queries:
            for hwnd_val, win_title in open_windows:
                if excluded_hwnd is not None and hwnd_val == excluded_hwnd:
                    continue
                if query_term in win_title.lower():
                    return hwnd_val
        return None

    @classmethod
    def _looks_like_web_target(cls, target_text: str) -> bool:
        """Heuristic to distinguish web domains/search queries from local desktop app/window names."""
        lowered = (target_text or "").strip().lower()
        if not lowered:
            return False
        if lowered.startswith(("http://", "https://", "www.")):
            return True
        if "." in lowered and not lowered.endswith(".exe"):
            return True
        web_keywords = ("stock", "price", "search", "news", "weather", "wiki", "youtube", "chatgpt", "github")
        return any(word in lowered for word in web_keywords)

    @classmethod
    def _find_installed_app_target(cls, app_query: str) -> Optional[str]:
        """Resolve any installed Windows app via aliases, PATH, Registry App Paths, or Start Menu shortcuts."""
        cleaned = (app_query or "").strip().strip("\"'")
        if not cleaned:
            return None
        lowered = cleaned.lower()
        compact = re.sub(r"[\s_\-]+", "", lowered)

        utility_aliases = {
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "notepad": "notepad.exe",
            "paint": "mspaint.exe",
            "mspaint": "mspaint.exe",
            "cmd": "cmd.exe",
            "commandprompt": "cmd.exe",
            "powershell": "powershell.exe",
            "terminal": "wt.exe",
            "windowsterminal": "wt.exe",
            "explorer": "explorer.exe",
            "fileexplorer": "explorer.exe",
            "taskmanager": "taskmgr.exe",
            "taskmgr": "taskmgr.exe",
            "controlpanel": "control.exe",
            "snippingtool": "snippingtool.exe",
            "devicemanager": "devmgmt.msc",
            "vscode": "code",
            "visualstudiocode": "code",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "outlook": "outlook.exe",
        }
        mapped_cmd = utility_aliases.get(compact)
        if mapped_cmd:
            which_hit = shutil.which(mapped_cmd)
            if which_hit:
                return which_hit
            if mapped_cmd.endswith(".msc"):
                return mapped_cmd

        direct_which = shutil.which(cleaned) or shutil.which(f"{cleaned}.exe")
        if direct_which:
            return direct_which

        if winreg is not None:
            app_paths_subkey = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
            for hive_root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive_root, app_paths_subkey) as parent_key:
                        subkey_count, _val_count, _mod_time = winreg.QueryInfoKey(parent_key)
                        for sub_idx in range(subkey_count):
                            subkey_name = winreg.EnumKey(parent_key, sub_idx)
                            subkey_stem = os.path.splitext(subkey_name)[0].lower()
                            subkey_compact = re.sub(r"[\s_\-]+", "", subkey_stem)
                            if (
                                lowered in subkey_stem
                                or compact in subkey_compact
                                or subkey_compact in compact
                            ):
                                try:
                                    with winreg.OpenKey(parent_key, subkey_name) as exe_key:
                                        default_path, _reg_type = winreg.QueryValueEx(exe_key, "")
                                        candidate_path = str(default_path or "").strip().strip("\"'")
                                        if candidate_path and os.path.exists(candidate_path):
                                            return candidate_path
                                except OSError:
                                    continue
                except OSError:
                    continue

        shortcut_roots = [
            os.path.join(os.environ.get("ProgramData", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
        ]
        best_shortcut: Optional[Tuple[int, str]] = None
        for root_dir in shortcut_roots:
            if not root_dir or not os.path.isdir(root_dir):
                continue
            for walk_root, _dir_names, file_names in os.walk(root_dir):
                for file_name in file_names:
                    if not file_name.lower().endswith((".lnk", ".appref-ms", ".url")):
                        continue
                    file_stem = os.path.splitext(file_name)[0].lower()
                    file_compact = re.sub(r"[\s_\-]+", "", file_stem)
                    if compact == file_compact:
                        return os.path.join(walk_root, file_name)
                    if (lowered in file_stem or compact in file_compact) and best_shortcut is None:
                        best_shortcut = (len(file_stem), os.path.join(walk_root, file_name))
        if best_shortcut is not None:
            return best_shortcut[1]
        return None

    @classmethod
    def _launch_installed_app_target(cls, app_name: str, resolved_target: str) -> Tuple[bool, str]:
        """Launch a resolved executable, shortcut, or system MMC target."""
        try:
            cls._pre_launch_hwnd = cls._current_foreground_hwnd()
            cls._last_app_launch_timestamp = time.monotonic()
            cls._last_launched_target = re.sub(r"[\s_\-]+", "", app_name.lower())
            lower_target = resolved_target.lower()
            if lower_target.endswith(".exe") and os.path.exists(resolved_target):
                subprocess.Popen(
                    [resolved_target],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
            else:
                os.startfile(resolved_target)
            return True, f"Launched {app_name}"
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def _snap_hwnd_to_rect(
        cls,
        target_hwnd: int,
        left_px: int,
        top_px: int,
        width_px: int,
        height_px: int,
    ) -> bool:
        if sys.platform != "win32" or not target_hwnd:
            return False
        try:
            user32 = ctypes.windll.user32
            user32.ShowWindow(int(target_hwnd), cls.SW_RESTORE)
            time.sleep(0.06)
            moved_ok = user32.SetWindowPos(
                int(target_hwnd),
                0,
                int(left_px),
                int(top_px),
                max(360, int(width_px)),
                max(260, int(height_px)),
                cls.SWP_SHOWWINDOW,
            )
            return bool(moved_ok)
        except Exception:
            return False

    @classmethod
    def _focus_hwnd(cls, target_hwnd: int) -> None:
        if sys.platform != "win32" or not target_hwnd:
            return
        try:
            user32 = ctypes.windll.user32
            user32.ShowWindow(int(target_hwnd), cls.SW_RESTORE)
            user32.SetForegroundWindow(int(target_hwnd))
        except Exception:
            pass

    @staticmethod
    def _get_clipboard_unicode() -> str:
        if sys.platform != "win32":
            return ""
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        if not user32.OpenClipboard(None):
            return ""
        try:
            clip_handle = user32.GetClipboardData(SmartDictationInjector.CF_UNICODETEXT)
            if not clip_handle:
                return ""
            raw_ptr = kernel32.GlobalLock(clip_handle)
            if not raw_ptr:
                return ""
            try:
                return str(ctypes.wstring_at(raw_ptr) or "").strip()
            finally:
                kernel32.GlobalUnlock(clip_handle)
        except Exception:
            return ""
        finally:
            user32.CloseClipboard()

    @classmethod
    def _extract_tab_url_by_index(cls, chrome_hwnd: int, tab_index: Optional[int] = None) -> str:
        if sys.platform != "win32" or not chrome_hwnd:
            return ""
        cls._focus_hwnd(chrome_hwnd)
        time.sleep(0.14)
        if tab_index in (1, 2):
            tab_vk = cls.VK_1 if tab_index == 1 else cls.VK_2
            cls._press_win32_combo([cls.VK_CONTROL, tab_vk])
            time.sleep(0.12)
        SmartDictationInjector._set_clipboard_unicode("")
        time.sleep(0.04)
        cls._press_win32_combo([cls.VK_CONTROL, cls.VK_L])
        time.sleep(0.09)
        cls._press_win32_combo([cls.VK_CONTROL, cls.VK_C])
        time.sleep(0.12)
        copied_value = cls._get_clipboard_unicode()
        if copied_value.lower().startswith(("http://", "https://")):
            return copied_value
        return ""

    @classmethod
    def _target_to_chrome_url(cls, raw_target: str) -> str:
        cleaned = (raw_target or "").strip().strip("\"'")
        if not cleaned:
            return "https://www.google.com"
        direct_url = cls.normalize_target_url(cleaned)
        if direct_url is not None:
            return direct_url
        return f"https://www.google.com/search?q={urllib.parse.quote_plus(cleaned)}"

    @classmethod
    def _launch_chrome_new_window_hwnd(
        cls,
        target_url: str,
        timeout_seconds: float = 3.2,
    ) -> Optional[int]:
        existing_hwnds = set(cls._list_visible_chrome_hwnds())
        cls._record_browser_target(target_url)
        chrome_exe = cls._resolve_chrome_executable()
        cls._pre_launch_hwnd = cls._current_foreground_hwnd()
        cls._last_app_launch_timestamp = time.monotonic()
        cls._last_launched_target = "chrome"
        if chrome_exe:
            subprocess.Popen(
                [chrome_exe, "--new-window", target_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
        else:
            webbrowser.open_new(target_url)
            return None

        deadline = time.monotonic() + max(0.8, timeout_seconds)
        while time.monotonic() < deadline:
            current_hwnds = cls._list_visible_chrome_hwnds()
            for candidate_hwnd in current_hwnds:
                if candidate_hwnd not in existing_hwnds:
                    return candidate_hwnd
            time.sleep(0.12)
        return None

    @classmethod
    def open_and_tile_side_by_side(
        cls,
        left_target: str = "",
        right_target: str = "",
    ) -> Tuple[bool, str]:
        """Opens or splits Chrome tabs/windows and tiles two windows side by side across the work area."""
        work_left, work_top, work_right, work_bottom = cls._get_work_area_rect()
        total_width = max(800, work_right - work_left)
        work_height = max(480, work_bottom - work_top)
        left_width = total_width // 2
        right_width = total_width - left_width
        left_rect = (work_left, work_top, left_width, work_height)
        right_rect = (work_left + left_width, work_top, right_width, work_height)

        cleaned_left = (left_target or "").strip()
        cleaned_right = (right_target or "").strip()
        try:
            if cleaned_left and cleaned_right:
                existing_all_hwnds = {hwnd_item for hwnd_item, _title in cls._list_all_windows()}

                def _resolve_or_launch_side_hwnd(
                    target_name: str,
                    excluded_hwnd: Optional[int] = None,
                ) -> Optional[int]:
                    lowered_target = target_name.lower()
                    is_web = cls._looks_like_web_target(target_name)
                    # Check open desktop windows first (always allow if non-web or if exact title already open).
                    open_match = cls._find_matching_window_hwnd(target_name, excluded_hwnd=excluded_hwnd)
                    if open_match is not None and not is_web:
                        return open_match
                    if is_web:
                        for hwnd_item, win_title in cls._list_all_windows():
                            if excluded_hwnd is not None and hwnd_item == excluded_hwnd:
                                continue
                            if lowered_target in win_title.lower():
                                return hwnd_item
                    if not is_web:
                        norm_name = re.sub(r"[\s_\-]+", "", lowered_target)
                        launcher = cls.SAFE_APP_LAUNCHERS.get(norm_name)
                        launched_target = False
                        if norm_name == "chrome":
                            chrome_exe = cls._resolve_chrome_executable()
                            if chrome_exe:
                                launcher = [chrome_exe, "--new-window"]
                        if launcher is not None:
                            cls._pre_launch_hwnd = cls._current_foreground_hwnd()
                            subprocess.Popen(
                                launcher,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                shell=False,
                            )
                            launched_target = True
                        else:
                            resolved_target = cls._find_installed_app_target(target_name)
                            if resolved_target:
                                launched_target, _msg = cls._launch_installed_app_target(target_name, resolved_target)
                        if launched_target:
                            deadline = time.monotonic() + 2.4
                            while time.monotonic() < deadline:
                                matched_now = cls._find_matching_window_hwnd(target_name, excluded_hwnd=excluded_hwnd)
                                if matched_now is not None and matched_now not in existing_all_hwnds:
                                    existing_all_hwnds.add(matched_now)
                                    return matched_now
                                if matched_now is not None:
                                    return matched_now
                                time.sleep(0.12)
                    target_url = cls._target_to_chrome_url(target_name)
                    return cls._launch_chrome_new_window_hwnd(target_url)

                left_hwnd = _resolve_or_launch_side_hwnd(cleaned_left, excluded_hwnd=None)
                right_hwnd = _resolve_or_launch_side_hwnd(cleaned_right, excluded_hwnd=left_hwnd)
                if left_hwnd:
                    cls._snap_hwnd_to_rect(left_hwnd, *left_rect)
                if right_hwnd:
                    cls._snap_hwnd_to_rect(right_hwnd, *right_rect)
                if not left_hwnd or not right_hwnd:
                    visible_hwnds = cls._list_visible_chrome_hwnds()
                    if len(visible_hwnds) >= 2:
                        cls._snap_hwnd_to_rect(visible_hwnds[1], *left_rect)
                        cls._snap_hwnd_to_rect(visible_hwnds[0], *right_rect)
                return (
                    True,
                    f"Tiled side by side: {cleaned_left} on the left and {cleaned_right} on the right.",
                )

            if cleaned_left or cleaned_right:
                single_target = cleaned_left or cleaned_right
                single_url = cls._target_to_chrome_url(single_target)
                existing_hwnds = cls._list_visible_chrome_hwnds()
                if existing_hwnds:
                    cls._snap_hwnd_to_rect(existing_hwnds[0], *left_rect)
                    new_right_hwnd = cls._launch_chrome_new_window_hwnd(single_url)
                    if new_right_hwnd:
                        cls._snap_hwnd_to_rect(new_right_hwnd, *right_rect)
                    return True, f"Tiled Chrome with {single_target} on the right."
                left_fallback_url = (
                    cls._recent_browser_targets[-1]
                    if cls._recent_browser_targets
                    else "https://www.google.com"
                )
                left_hwnd = cls._launch_chrome_new_window_hwnd(left_fallback_url)
                right_hwnd = cls._launch_chrome_new_window_hwnd(single_url)
                if left_hwnd:
                    cls._snap_hwnd_to_rect(left_hwnd, *left_rect)
                if right_hwnd:
                    cls._snap_hwnd_to_rect(right_hwnd, *right_rect)
                return True, f"Opened {single_target} side by side in Chrome."

            chrome_hwnds = cls._list_visible_chrome_hwnds()
            if len(chrome_hwnds) >= 2:
                cls._snap_hwnd_to_rect(chrome_hwnds[0], *left_rect)
                cls._snap_hwnd_to_rect(chrome_hwnds[1], *right_rect)
                return True, "Tiled your two Chrome windows side by side."

            if len(chrome_hwnds) == 1:
                only_hwnd = chrome_hwnds[0]
                tab1_url = cls._extract_tab_url_by_index(only_hwnd, tab_index=1)
                tab2_url = cls._extract_tab_url_by_index(only_hwnd, tab_index=2) if tab1_url else ""
                if tab1_url and tab2_url and tab2_url != tab1_url:
                    # Two distinct tabs are open: detach Tab 2 cleanly into its own right-half window.
                    cls._press_win32_combo([cls.VK_CONTROL, cls.VK_W])
                    time.sleep(0.14)
                    cls._snap_hwnd_to_rect(only_hwnd, *left_rect)
                    right_hwnd = cls._launch_chrome_new_window_hwnd(tab2_url)
                    if right_hwnd:
                        cls._snap_hwnd_to_rect(right_hwnd, *right_rect)
                    return True, "Split your two Chrome tabs side by side."

                if len(cls._recent_browser_targets) >= 2:
                    cls._snap_hwnd_to_rect(only_hwnd, *left_rect)
                    right_hwnd = cls._launch_chrome_new_window_hwnd(cls._recent_browser_targets[-1])
                    if right_hwnd:
                        cls._snap_hwnd_to_rect(right_hwnd, *right_rect)
                    return True, "Arranged your recent Chrome targets side by side."

                fallback_right_url = tab1_url or "https://www.google.com"
                cls._snap_hwnd_to_rect(only_hwnd, *left_rect)
                right_hwnd = cls._launch_chrome_new_window_hwnd(fallback_right_url)
                if right_hwnd:
                    cls._snap_hwnd_to_rect(right_hwnd, *right_rect)
                return True, "Split Chrome into two side-by-side windows."

            if len(cls._recent_browser_targets) >= 2:
                left_url = cls._recent_browser_targets[-2]
                right_url = cls._recent_browser_targets[-1]
                left_hwnd = cls._launch_chrome_new_window_hwnd(left_url)
                right_hwnd = cls._launch_chrome_new_window_hwnd(right_url)
                if left_hwnd:
                    cls._snap_hwnd_to_rect(left_hwnd, *left_rect)
                if right_hwnd:
                    cls._snap_hwnd_to_rect(right_hwnd, *right_rect)
                return True, "Opened your two recent Chrome pages side by side."

            return False, "No open Chrome tabs or target pages found to tile side by side."
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def open_chrome_search(cls, query_text: str) -> Tuple[bool, str]:
        """Opens Chrome directly to the target site (if URL/domain) or Google search results for `query_text`."""
        cleaned_query = (query_text or "").strip()
        if not cleaned_query:
            return False, "Empty search query"
        direct_url = cls.normalize_target_url(cleaned_query)
        target_url = (
            direct_url
            if direct_url is not None
            and ("." in cleaned_query or cleaned_query.lower() in cls.KNOWN_URL_ALIASES)
            else f"https://www.google.com/search?q={urllib.parse.quote_plus(cleaned_query)}"
        )
        cls._record_browser_target(target_url)
        try:
            cls._pre_launch_hwnd = cls._current_foreground_hwnd()
            cls._last_app_launch_timestamp = time.monotonic()
            cls._last_launched_target = "chrome"
            chrome_exe = cls._resolve_chrome_executable()
            if chrome_exe:
                subprocess.Popen(
                    [chrome_exe, target_url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
                return True, f"Opened {target_url} in Chrome"
            opened = webbrowser.open(target_url, new=2)
            return bool(opened), f"Opened {target_url} in browser"
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def _resolve_settings_uri(cls, topic_text: str) -> Optional[Tuple[str, str]]:
        lowered = (topic_text or "").lower().strip()
        for keyword, uri_pair in cls.MS_SETTINGS_URI_MAP.items():
            if keyword in lowered:
                return uri_pair
        return None

    @classmethod
    def describe_step_for_speech(cls, command: jarvis_response_parser.DrawingCommand) -> str:
        """Returns a crisp 3-to-6 word phrase spoken right as each agent step begins."""
        argument_text = (command.body or command.label or command.header or "").strip()
        if command.kind == "open_app":
            if argument_text.lower().startswith("settings:"):
                subtopic = argument_text.split(":", 1)[1].strip()
                resolved_pair = cls._resolve_settings_uri(subtopic)
                if resolved_pair is not None:
                    return f"Opening {resolved_pair[1]}."
                return f"Opening Settings for {subtopic[:26]}."
            app_display = argument_text.capitalize() if argument_text else "app"
            if app_display.lower() == "chrome":
                app_display = "Chrome"
            elif app_display.lower() == "vscode":
                app_display = "VS Code"
            return f"Opening {app_display}."
        if command.kind in ("open_browser", "open_url"):
            short_label = (command.label or argument_text or "browser")[:34]
            if short_label.startswith(("http://", "https://")):
                return "Opening site in browser."
            return f"Opening {short_label}."
        if command.kind == "open_side_by_side":
            return "Arranging windows side by side."
        if command.kind == "window_action":
            action_label = (command.header or argument_text or "window")[:30]
            return f"Updating {action_label}."
        if command.kind == "set_clipboard":
            return "Copied text to clipboard."
        if command.kind == "write_file":
            file_label = os.path.basename(argument_text) or "notes file"
            return f"Saving {file_label[:28]}."
        if command.kind == "system_control":
            return f"Running {argument_text.replace('_', ' ')}."
        if command.kind == "run_shell_read":
            return f"Checking {(command.header or 'system info')[:28]}."
        if command.kind == "write_project_file":
            return f"Writing {os.path.basename(argument_text) or 'project file'}."
        if command.kind == "read_project_file":
            return f"Reading {os.path.basename(argument_text) or 'project file'}."
        if command.kind == "run_project_command":
            return f"Running {argument_text[:30]}."
        if command.kind == "open_project":
            return f"Opening project {argument_text[:28]}."
        if command.kind == "click_ui":
            return f"Clicking {argument_text[:28]}."
        if command.kind == "inspect_ui":
            return "Inspecting active window."
        if command.kind == "read_window":
            return f"Reading controls in {(argument_text or 'active window')[:26]}."
        if command.kind == "list_windows":
            return "Checking open windows."
        if command.kind == "act_in_app":
            return f"Acting in {(argument_text or 'app')[:26]}."
        if command.kind == "find_text_on_screen":
            return f"Locating {argument_text[:28]} on screen."
        if command.kind in ("search_files", "list_files"):
            return f"Checking files in {(command.header or argument_text or 'folder')[:26]}."
        if command.kind == "plan_steps":
            return "Updating checklist."
        if command.kind == "remember_commitment":
            return f"Saved reminder: {command.label[:28]}."
        if command.kind == "snooze_reminder":
            return "Snoozing reminder."
        if command.kind in ("list_commitments", "show_list"):
            return "Showing your list."
        if command.kind == "start_focus":
            return "Starting focus timer."
        if command.kind == "stop_focus":
            return "Ending focus mode."
        if command.kind == "save_skill":
            return f"Saved skill {command.label[:28]}."
        if command.kind in ("use_skill", "install_skill"):
            return f"Loading skill {argument_text[:28]}."
        if command.kind == "forget_skill":
            return f"Removing skill {argument_text[:28]}."
        if command.kind == "schedule_job":
            return f"Scheduled {command.label[:28]}."
        if command.kind == "list_jobs":
            return "Checking scheduled routines."
        if command.kind in ("learn_fact", "forget_fact"):
            return "Updating saved memory."
        if command.kind == "photo":
            return f"Showing photo for {argument_text[:26]}."
        if command.kind == "edit_drawing":
            return "Updating screen markings."
        if command.kind == "agent_done":
            return argument_text or "Completed."
        if command.kind == "search_web":
            return f"Searching for {argument_text[:34]}."
        if command.kind == "type_text":
            return f"Typing {argument_text[:34]}."
        if command.kind == "press_keys":
            return f"Pressing {argument_text}."
        if command.kind == "request_admin_permission":
            return "Admin confirmation required."
        return "Running step."

    @classmethod
    def execute_agent_command_blocking(
        cls,
        command: jarvis_response_parser.DrawingCommand,
    ) -> Tuple[bool, str]:
        """Executes one agent step synchronously so multi-task chains finish typing before switching focus."""
        return cls._execute_agent_command_internal(command, block_for_typing=True)

    @classmethod
    def execute_agent_command(
        cls,
        command: jarvis_response_parser.DrawingCommand,
    ) -> Tuple[bool, str]:
        """Executes a parsed desktop/browser command safely and returns (ok, status_message)."""
        return cls._execute_agent_command_internal(command, block_for_typing=False)

    @classmethod
    def _execute_agent_command_internal(
        cls,
        command: jarvis_response_parser.DrawingCommand,
        block_for_typing: bool = False,
    ) -> Tuple[bool, str]:
        command_kind = command.kind
        argument_text = (command.body or command.label or command.header or "").strip()

        if command_kind in ("open_browser", "open_url"):
            resolved_url = cls.normalize_target_url(argument_text)
            if resolved_url is None:
                return False, f"Invalid URL: {argument_text}"
            cls._record_browser_target(resolved_url)
            try:
                cls._pre_launch_hwnd = cls._current_foreground_hwnd()
                cls._last_app_launch_timestamp = time.monotonic()
                cls._last_launched_target = "browser"
                chrome_exe = cls._resolve_chrome_executable()
                if chrome_exe:
                    launch_cmd = (
                        [chrome_exe, "--new-window", resolved_url]
                        if (command.header or "").strip().lower() == "new_window"
                        else [chrome_exe, resolved_url]
                    )
                    subprocess.Popen(
                        launch_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        shell=False,
                    )
                    return True, f"Opened {resolved_url}"
                opened = webbrowser.open(resolved_url, new=2)
                return bool(opened), f"Opened {resolved_url}"
            except Exception as exc:
                return False, str(exc)

        if command_kind == "open_side_by_side":
            return cls.open_and_tile_side_by_side(command.body, command.header)

        if command_kind == "search_web":
            return cls.open_chrome_search(argument_text)

        if command_kind == "open_app":
            raw_lower = argument_text.lower().strip()
            if raw_lower.startswith("settings:"):
                subtopic = raw_lower.split(":", 1)[1].strip()
                resolved_pair = cls._resolve_settings_uri(subtopic)
                try:
                    cls._pre_launch_hwnd = cls._current_foreground_hwnd()
                    cls._last_app_launch_timestamp = time.monotonic()
                    cls._last_launched_target = "settings"
                    if resolved_pair is not None:
                        uri_target, display_label = resolved_pair
                        os.startfile(uri_target)
                        return True, f"Opened {display_label}"
                    os.startfile("ms-settings:")
                    cls._wait_for_foreground_window_change(cls._pre_launch_hwnd, timeout_seconds=2.5)
                    cls._press_win32_combo([cls.VK_CONTROL, cls.VK_F])
                    time.sleep(0.15)
                    cls._send_text_to_focused_window(subtopic, focus_address_bar=False, submit_enter=True)
                    return True, f"Opened Settings and searched {subtopic}"
                except Exception as exc:
                    return False, str(exc)

            normalized_app = re.sub(r"[\s_\-]+", "", argument_text.lower())
            if normalized_app in ("googlechrome", "chrome"):
                normalized_app = "chrome"
            if "settings" in normalized_app:
                resolved_pair = cls._resolve_settings_uri(argument_text)
                if resolved_pair is not None:
                    try:
                        cls._pre_launch_hwnd = cls._current_foreground_hwnd()
                        cls._last_app_launch_timestamp = time.monotonic()
                        cls._last_launched_target = "settings"
                        os.startfile(resolved_pair[0])
                        return True, f"Opened {resolved_pair[1]}"
                    except Exception as exc:
                        return False, str(exc)
            if normalized_app in ("settings", "windowssettings"):
                try:
                    cls._pre_launch_hwnd = cls._current_foreground_hwnd()
                    cls._last_app_launch_timestamp = time.monotonic()
                    cls._last_launched_target = "settings"
                    os.startfile("ms-settings:")
                    return True, "Opened Windows Settings"
                except Exception as exc:
                    return False, str(exc)
            if normalized_app in ("browser", "webbrowser", "defaultbrowser"):
                return cls._execute_agent_command_internal(
                    jarvis_response_parser.DrawingCommand(kind="open_browser", label=argument_text),
                    block_for_typing=block_for_typing,
                )
            folder_aliases = {
                "downloads": os.path.join(os.path.expanduser("~"), "Downloads"),
                "documents": os.path.join(os.path.expanduser("~"), "Documents"),
                "desktop": os.path.join(os.path.expanduser("~"), "Desktop"),
                "pictures": os.path.join(os.path.expanduser("~"), "Pictures"),
                "music": os.path.join(os.path.expanduser("~"), "Music"),
                "videos": os.path.join(os.path.expanduser("~"), "Videos"),
                "projects": ProjectWorkspaceService.get_projects_root(),
                "dprojects": ProjectWorkspaceService.get_projects_root(),
            }
            if normalized_app in folder_aliases and os.path.isdir(folder_aliases[normalized_app]):
                try:
                    os.startfile(folder_aliases[normalized_app])
                    return True, f"Opened {argument_text}"
                except Exception as exc:
                    return False, str(exc)
            if os.path.exists(argument_text):
                try:
                    os.startfile(argument_text)
                    return True, f"Opened {os.path.basename(argument_text) or argument_text}"
                except Exception as exc:
                    return False, str(exc)
            matched_project_dir = ProjectWorkspaceService.resolve_project_dir(argument_text)
            if matched_project_dir is not None:
                return ProjectWorkspaceService.open_project(argument_text, tool_name="auto")
            launch_args = cls.SAFE_APP_LAUNCHERS.get(normalized_app)
            if normalized_app == "chrome":
                chrome_exe = cls._resolve_chrome_executable()
                if chrome_exe:
                    launch_args = [chrome_exe, "--new-window"]
            if launch_args is None:
                resolved_target = cls._find_installed_app_target(argument_text)
                if resolved_target:
                    return cls._launch_installed_app_target(argument_text, resolved_target)
                return False, f"Could not find installed app: {argument_text}"
            try:
                cls._pre_launch_hwnd = cls._current_foreground_hwnd()
                cls._last_app_launch_timestamp = time.monotonic()
                cls._last_launched_target = normalized_app
                subprocess.Popen(
                    launch_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
                return True, f"Launched {argument_text}"
            except Exception as exc:
                return False, str(exc)

        if command_kind == "press_keys":
            raw_tokens = [token.strip().lower() for token in argument_text.split("+") if token.strip()]
            if not raw_tokens or len(raw_tokens) > 3:
                return False, f"Unsupported shortcut: {argument_text}"
            normalized_tokens = [cls.SAFE_KEY_ALIASES.get(token, token) for token in raw_tokens]
            for token in normalized_tokens:
                is_single_alnum = len(token) == 1 and token.isalnum()
                is_modifier = token in ("ctrl", "alt", "shift", "win")
                if not (is_single_alnum or is_modifier or token in cls.ALLOWED_SINGLE_KEYS):
                    return False, f"Key blocked by safety check: {token}"
            try:
                if pyautogui is not None:
                    if len(normalized_tokens) == 1:
                        pyautogui.press(normalized_tokens[0])
                    else:
                        pyautogui.hotkey(*normalized_tokens)
                    return True, f"Pressed {'+'.join(normalized_tokens)}"
                return False, "Keyboard helper unavailable"
            except Exception as exc:
                return False, str(exc)

        if command_kind == "type_text":
            if not argument_text:
                return False, "No text to type"
            elapsed_since_launch = time.monotonic() - cls._last_app_launch_timestamp
            recently_launched_app = elapsed_since_launch < 6.0
            launched_settings = cls._last_launched_target.startswith("settings")
            if launched_settings:
                resolved_pair = cls._resolve_settings_uri(argument_text)
                if resolved_pair is not None:
                    try:
                        os.startfile(resolved_pair[0])
                        return True, f"Opened {resolved_pair[1]}"
                    except Exception:
                        pass

            launched_browser = any(
                browser_name in cls._last_launched_target
                for browser_name in ("chrome", "edge", "msedge", "firefox", "browser")
            )
            should_submit = (
                (command.header or "").strip().lower() == "submit"
                or launched_browser
                or launched_settings
            )
            if recently_launched_app:
                captured_previous_hwnd = cls._pre_launch_hwnd
                captured_text = argument_text

                def _delayed_typing_worker() -> None:
                    cls._wait_for_foreground_window_change(captured_previous_hwnd, timeout_seconds=2.8)
                    if launched_settings:
                        cls._press_win32_combo([cls.VK_CONTROL, cls.VK_F])
                        time.sleep(0.15)
                    cls._send_text_to_focused_window(
                        captured_text,
                        focus_address_bar=launched_browser,
                        submit_enter=should_submit,
                    )

                if block_for_typing:
                    _delayed_typing_worker()
                    return True, f"Opened & typed {argument_text[:36]}"

                threading.Thread(
                    target=_delayed_typing_worker,
                    name="JarvisDelayedTypeWorker",
                    daemon=True,
                ).start()
                return True, f"Opening & typing {argument_text[:36]}"

            if launched_settings:
                cls._press_win32_combo([cls.VK_CONTROL, cls.VK_F])
                time.sleep(0.15)
            typed_ok = cls._send_text_to_focused_window(
                argument_text,
                focus_address_bar=False,
                submit_enter=should_submit,
            )
            if typed_ok:
                return True, f"Typed {len(argument_text[:280])} characters"
            return False, "Could not focus active field for typing"

        if command_kind == "window_action":
            action_verb = argument_text.lower().replace("-", "_").replace(" ", "_")
            target_query = (command.header or "").strip()
            if action_verb in ("minimize_all", "desktop"):
                cls._press_win32_combo([0x5B, 0x4D])
                return True, "Minimized open windows"
            target_hwnd = cls._find_matching_window_hwnd(target_query) if target_query else cls._current_foreground_hwnd()
            if not target_hwnd:
                return False, f"Could not find open window matching: {target_query or 'active window'}"
            if sys.platform != "win32":
                return False, "Window actions require Windows"
            user32 = ctypes.windll.user32
            if action_verb in ("focus", "activate", "switch"):
                cls._focus_hwnd(target_hwnd)
                return True, f"Focused {target_query or 'window'}"
            if action_verb == "minimize":
                user32.ShowWindow(int(target_hwnd), 6)
                return True, f"Minimized {target_query or 'window'}"
            if action_verb == "maximize":
                user32.ShowWindow(int(target_hwnd), 3)
                return True, f"Maximized {target_query or 'window'}"
            if action_verb == "restore":
                user32.ShowWindow(int(target_hwnd), 9)
                return True, f"Restored {target_query or 'window'}"
            if action_verb in ("snap_left", "snap_right", "left", "right"):
                work_left, work_top, work_right, work_bottom = cls._get_work_area_rect()
                half_w = max(400, (work_right - work_left) // 2)
                work_h = max(360, work_bottom - work_top)
                is_left = action_verb in ("snap_left", "left")
                snap_x = work_left if is_left else (work_left + half_w)
                snap_w = half_w if is_left else max(400, (work_right - work_left) - half_w)
                cls._snap_hwnd_to_rect(target_hwnd, snap_x, work_top, snap_w, work_h)
                return True, f"Snapped {target_query or 'window'} {'left' if is_left else 'right'}"
            if action_verb == "close":
                user32.PostMessageW(int(target_hwnd), 0x0010, 0, 0)
                return True, f"Sent close request to {target_query or 'window'}"
            return False, f"Unsupported window action: {action_verb}"

        if command_kind == "set_clipboard":
            if not argument_text:
                return False, "No clipboard text provided"
            copied_ok = SmartDictationInjector._set_clipboard_unicode(argument_text)
            if copied_ok:
                return True, f"Copied {len(argument_text)} characters to clipboard"
            return False, "Could not write text to clipboard"

        if command_kind == "write_file":
            return ProjectWorkspaceService.write_workspace_file(
                argument_text,
                command.header or "",
                default_to_projects=False,
            )

        if command_kind == "write_project_file":
            return ProjectWorkspaceService.write_workspace_file(
                argument_text,
                command.header or "",
                default_to_projects=True,
            )

        if command_kind == "read_project_file":
            return ProjectWorkspaceService.read_workspace_file(argument_text)

        if command_kind == "run_project_command":
            return ProjectWorkspaceService.run_project_command(
                argument_text,
                cwd_rel=command.header or "",
            )

        if command_kind == "open_project":
            return ProjectWorkspaceService.open_project(
                argument_text,
                tool_name=command.header or "auto",
            )

        if command_kind == "click_ui":
            return WindowsUIAService.find_and_click_control(argument_text)

        if command_kind == "inspect_ui":
            return True, WindowsUIAService.format_ui_summary(max_elements=18)

        if command_kind == "agent_done":
            return True, argument_text or "Completed."

        if command_kind == "system_control":
            control_action = argument_text.lower().replace("-", "_").replace(" ", "_")
            if sys.platform != "win32":
                return False, "System controls require Windows"
            user32 = ctypes.windll.user32

            def _tap_vk(vk_code: int, repeat_count: int = 1) -> None:
                for _step_idx in range(max(1, repeat_count)):
                    user32.keybd_event(int(vk_code), 0, 0, 0)
                    time.sleep(0.02)
                    user32.keybd_event(int(vk_code), 0, 2, 0)
                    time.sleep(0.02)

            if control_action in ("mute", "unmute"):
                _tap_vk(0xAD, 1)
                return True, "Toggled audio mute"
            if control_action in ("volume_up", "vol_up"):
                _tap_vk(0xAF, 3)
                return True, "Turned volume up"
            if control_action in ("volume_down", "vol_down"):
                _tap_vk(0xAE, 3)
                return True, "Turned volume down"
            if control_action in ("play_pause", "play", "pause"):
                _tap_vk(0xB3, 1)
                return True, "Toggled media playback"
            if control_action in ("next_track", "next"):
                _tap_vk(0xB0, 1)
                return True, "Skipped to next track"
            if control_action in ("prev_track", "previous"):
                _tap_vk(0xB1, 1)
                return True, "Returned to previous track"
            if control_action in ("show_desktop", "desktop"):
                cls._press_win32_combo([0x5B, 0x44])
                return True, "Toggled desktop view"
            if control_action in ("screenshot", "snip"):
                cls._press_win32_combo([0x5B, 0x10, 0x53])
                return True, "Opened Windows snipping overlay"
            return False, f"Unsupported system control: {control_action}"

        if command_kind == "run_shell_read":
            ps_query = argument_text.strip()
            if not ps_query:
                return False, "Empty shell query"
            lowered_query = ps_query.lower()
            blocked_patterns = (
                "remove-item", "del ", "rm ", "rmdir", "erase ", "format-", "shutdown",
                "restart-computer", "stop-computer", "stop-process", "taskkill", "kill ",
                "reg ", "set-item", "new-item", "remove-itemproperty", "set-executionpolicy",
                "netsh", "bcdedit", "diskpart", "cipher", "sc ", "invoke-expression", "iex",
                "start-process", "invoke-webrequest", "iwr", "curl ", "wget ", "set-service",
                "stop-service", "uninstall", "msiexec", "out-file", "set-content",
                "add-content", "clear-content", ">", "`", ";", "&",
            )
            if any(token in lowered_query for token in blocked_patterns):
                return False, "Blocked non-read-only shell command"
            pipeline_segments = [seg.strip() for seg in ps_query.split("|") if seg.strip()]
            if not pipeline_segments:
                return False, "Empty pipeline"
            safe_prefixes = (
                "get-", "select-", "sort-", "where-", "measure-", "format-", "group-",
                "test-", "resolve-", "find-", "convertto-", "write-output", "(get-",
                "[system.", "$env:", "hostname", "whoami", "ipconfig", "systeminfo",
            )
            for seg_item in pipeline_segments:
                seg_lower = seg_item.lstrip("( ").lower()
                if not seg_lower.startswith(safe_prefixes):
                    return False, f"Only read-only PowerShell queries allowed (rejected: {seg_item[:28]})"
            try:
                proc_result = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_query],
                    capture_output=True,
                    text=True,
                    timeout=6.0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                raw_out = (proc_result.stdout or proc_result.stderr or "").strip()
                compact_out = re.sub(r"\s+", " ", raw_out)[:240]
                if proc_result.returncode == 0:
                    return True, compact_out or f"Completed {command.header or 'query'}"
                return False, compact_out or f"Query failed (exit {proc_result.returncode})"
            except subprocess.TimeoutExpired:
                return False, "Read-only query timed out"
            except Exception as exc:
                return False, str(exc)

        if command_kind == "read_window":
            res = RoamiesHandsService.read_window_controls(argument_text or "front")
            controls = res.get("controls") or []
            if not controls:
                return False, f"No controls readable in {argument_text or 'active window'}"
            preview = ", ".join(
                f"[{ctrl.get('id')}] {ctrl.get('role', 'ctrl')} '{str(ctrl.get('label', ''))[:26]}'"
                for ctrl in controls[:14]
            )
            return True, preview

        if command_kind == "list_windows":
            wins = RoamiesHandsService.list_windows()
            if not wins:
                return True, "No open windows detected."
            summary = " | ".join(
                f"{win.get('app', '')}: {win.get('title', '')[:32]}".strip(": ")
                for win in wins[:10]
            )
            return True, summary

        if command_kind == "act_in_app":
            return RoamiesHandsService.act_in_app(command.body or "front", command.header or command.label)

        if command_kind == "find_text_on_screen":
            bounds = RoamiesHandsService.find_text_bounds(argument_text, spec="front", allow_scroll=False)
            if bounds is None:
                return False, f"Could not find '{argument_text}' on screen"
            return True, f"Found '{argument_text}' at {bounds}"

        if command_kind == "list_files":
            resolved_dir, err_msg = ProjectWorkspaceService.resolve_safe_workspace_path(argument_text or ".", default_to_projects=True)
            if resolved_dir is None or not os.path.isdir(resolved_dir):
                return False, err_msg or f"Not a workspace folder: {argument_text}"
            entries = sorted(os.listdir(resolved_dir))[:40]
            return True, ", ".join(entries) or "(empty folder)"

        if command_kind == "search_files":
            search_root_arg = (command.header or ".").strip()
            resolved_dir, err_msg = ProjectWorkspaceService.resolve_safe_workspace_path(search_root_arg, default_to_projects=True)
            if resolved_dir is None or not os.path.isdir(resolved_dir):
                return False, err_msg or f"Not a workspace folder: {search_root_arg}"
            pattern_text = (command.body or "").strip()
            hits: List[str] = []
            for walk_root, _dirs, file_names in os.walk(resolved_dir):
                for fname in file_names[:60]:
                    if any(skip in walk_root for skip in ("\\node_modules", "\\.git", "\\__pycache__")):
                        continue
                    fpath = os.path.join(walk_root, fname)
                    try:
                        text_blob = open(fpath, "r", encoding="utf-8", errors="ignore").read(40000)
                        if re.search(pattern_text, text_blob, flags=re.IGNORECASE):
                            hits.append(os.path.relpath(fpath, resolved_dir))
                            if len(hits) >= 12:
                                break
                    except Exception:
                        continue
                if len(hits) >= 12:
                    break
            return True, ", ".join(hits) if hits else f"No matches for '{pattern_text}'"

        if command_kind == "remember_commitment":
            service = CommitmentsReminderService()
            item = service.add_commitment(command.label or argument_text, command.header or "", command.body or "")
            return True, f"Saved commitment: {item['text']}"

        if command_kind == "snooze_reminder":
            return CommitmentsReminderService().snooze_commitment(argument_text or "10")

        if command_kind in ("list_commitments", "show_list"):
            grouped = CommitmentsReminderService().list_commitments()
            today_txt = "; ".join(item["text"] for item in grouped["today"][:5]) or "None"
            later_txt = "; ".join(item["text"] for item in grouped["later"][:4]) or "None"
            return True, f"Today: {today_txt} | Later: {later_txt}"

        if command_kind == "save_skill":
            ok, _slug, msg = SkillsManagerService().save_skill(command.label or "skill", command.header or "Saved skill", command.body or "")
            return ok, msg

        if command_kind == "use_skill":
            skill_md = SkillsManagerService().read_skill(argument_text)
            if not skill_md:
                return False, f"No saved skill named '{argument_text}'"
            return True, re.sub(r"\s+", " ", skill_md)[:320]

        if command_kind == "forget_skill":
            deleted = SkillsManagerService().delete_skill(argument_text)
            return (True, f"Removed skill '{argument_text}'") if deleted else (False, f"No skill named '{argument_text}'")

        if command_kind == "install_skill":
            return SkillsManagerService().install_skill_from_source(command.body, command.header)

        if command_kind == "schedule_job":
            job = ScheduledJobsService().schedule_job(command.label or "Routine", command.body or command.label, command.header or "in 60 minutes")
            return True, f"Scheduled '{job['title']}'"

        if command_kind == "list_jobs":
            jobs_list = ScheduledJobsService().list_jobs()
            if not jobs_list:
                return True, "No scheduled jobs."
            return True, "; ".join(item["title"] for item in jobs_list[:8])

        if command_kind == "learn_fact":
            entry = LearnedFactsService().learn_fact(argument_text)
            return True, f"Saved fact [{entry['id']}]: {entry['text']}"

        if command_kind == "forget_fact":
            return LearnedFactsService().forget_fact(argument_text)

        if command_kind == "photo":
            img_bytes, title_text = WikipediaPhotoService.fetch_photo(command.body or argument_text)
            return (img_bytes is not None), (f"Loaded photo: {title_text}" if img_bytes else f"No photo found for '{command.body or argument_text}'")

        return False, f"Unknown agent command: {command_kind}"

    @classmethod
    def _detect_single_clause_commands(
        cls,
        clause_text: str,
        allow_tab_search: bool = False,
    ) -> List[jarvis_response_parser.DrawingCommand]:
        """Parses one standalone or compound single clause (`open <app> and type <text>`)."""
        cleaned = re.sub(r"\s+", " ", (clause_text or "").strip())
        if not cleaned:
            return []
        cleaned = re.sub(
            r"^(?:and\s+)?(?:(?:in|on)\s+)?(?:(?:another|a\s+new|new|second|next)\s+tab\s*[,:\-]?\s*|and\s+)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        lead_in_pattern = re.compile(
            r"^(?:hey\s+jarvis|jarvis|okay|ok|all\s+right|alright|please|can\s+you|could\s+you|would\s+you|i\s+want\s+to|i\s+would\s+like\s+to|let['’]?s|go\s+ahead\s+and|to)\b\s*[,:\-]?\s*",
            flags=re.IGNORECASE,
        )
        while cleaned:
            stripped_lead = lead_in_pattern.sub("", cleaned, count=1).strip()
            if stripped_lead == cleaned:
                break
            cleaned = stripped_lead
        cleaned = re.sub(
            r"\s+(?:in|on)\s+(?:one|a|the\s+first|another|a\s+new|new|second|next)\s+tab(?:\s+(?:in|on)\s+(?:chrome|google\s+chrome|edge|firefox|browser))?$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"\s+(?:in|on)\s+(?:chrome|google\s+chrome|edge|firefox|browser)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned:
            return []
        lowered = cleaned.lower().strip(" .,!?")

        remind_match = re.match(
            r"^(?:please\s+)?remind\s+me(?:\s+(in\s+\d+\s*(?:mins?|minutes?|hrs?|hours?|days?)|at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|tomorrow(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?))?\s+(?:to\s+)?(.+)$",
            lowered,
        )
        if remind_match:
            due_part = (remind_match.group(1) or "").strip()
            task_part = remind_match.group(2).strip(" .,!?")
            if task_part:
                return [
                    jarvis_response_parser.DrawingCommand(
                        kind="remember_commitment",
                        label=task_part,
                        header=due_part or "today",
                        body="",
                    )
                ]

        if re.match(r"^(?:show|open|view|what(?:'s|\s+is)\s+on)\s+(?:my\s+)?(?:to[\s\-]?do\s+list|todo\s+list|reminders?|commitments?|tasks?|list)$", lowered):
            return [jarvis_response_parser.DrawingCommand(kind="show_list", label="Today & Later", body="show")]

        snooze_match = re.match(r"^(?:please\s+)?snooze(?:\s+(?:for\s+)?(\d+)\s*(?:min|mins|minutes?)?)?$", lowered)
        if snooze_match:
            mins_val = snooze_match.group(1) or "10"
            return [jarvis_response_parser.DrawingCommand(kind="snooze_reminder", label=mins_val, body=mins_val)]

        focus_start_match = re.match(
            r"^(?:please\s+)?(?:start\s+(?:a\s+)?focus(?:\s+mode|\s+block|\s+timer)?|focus)(?:\s+for\s+(\d+)\s*(?:min|mins|minutes?))?(?:\s+(?:on|for)\s+(.+))?$",
            lowered,
        )
        if focus_start_match and ("stop" not in lowered and "end" not in lowered):
            focus_mins = focus_start_match.group(1) or "25"
            focus_task = (focus_start_match.group(2) or "Focused work").strip(" .,!?")
            return [jarvis_response_parser.DrawingCommand(kind="start_focus", body=focus_mins, header=focus_task, label=focus_task)]

        if re.match(r"^(?:please\s+)?(?:stop|end|cancel|exit)\s+(?:the\s+)?focus(?:\s+mode|\s+block|\s+timer)?$", lowered):
            return [jarvis_response_parser.DrawingCommand(kind="stop_focus", label="stop_focus", body="stop")]

        fact_text = LearnedFactsService.extract_facts_from_user_message(cleaned)
        if fact_text:
            return [jarvis_response_parser.DrawingCommand(kind="learn_fact", label=fact_text, body=fact_text)]

        settings_compound_match = re.match(
            r"^(?:please\s+)?(?:open|launch|go\s+to)\s+(?:windows\s+)?settings\s+(?:and|then)\s+(?:please\s+)?(?:search(?:\s+for)?|find|open|check|show|type)\s+(.+)$",
            lowered,
        )
        if settings_compound_match:
            subtopic = cleaned[settings_compound_match.start(1) :].strip(" .!?")
            if subtopic:
                return [
                    jarvis_response_parser.DrawingCommand(
                        kind="open_app",
                        body=f"settings:{subtopic}",
                        label=f"Settings: {subtopic}",
                    )
                ]

        compound_url_match = re.match(
            r"^(?:please\s+)?(?:open|launch|start)\s+(?:my\s+)?(chrome|google\s+chrome|edge|firefox|browser|web\s+browser)\s+(?:and|then)\s+(?:please\s+)?(?:go\s+to|open|visit)\s+(.+)$",
            lowered,
        )
        if compound_url_match:
            raw_target = cleaned[compound_url_match.start(2) :].strip(" .!?")
            normalized_url = cls.normalize_target_url(raw_target)
            if normalized_url is not None:
                return [
                    jarvis_response_parser.DrawingCommand(
                        kind="open_url",
                        body=normalized_url,
                        label=raw_target,
                    )
                ]

        compound_type_match = re.match(
            r"^(?:please\s+)?(?:open|launch|start)\s+(?:my\s+)?(chrome|google\s+chrome|edge|firefox|browser|web\s+browser|notepad|terminal|powershell|cmd|vs\s?code|vscode)\s+(?:and|then)\s+(?:please\s+)?(type|write|enter|search(?:\s+for)?|look\s+up|ask)\s+(.+)$",
            lowered,
        )
        if compound_type_match:
            app_name = compound_type_match.group(1).strip()
            action_verb = compound_type_match.group(2).strip().lower()
            normalized_app_token = re.sub(r"\s+", "", app_name.lower())
            if normalized_app_token == "googlechrome":
                normalized_app_token = "chrome"
            elif normalized_app_token == "webbrowser":
                normalized_app_token = "browser"
            typed_payload = cleaned[compound_type_match.start(3) :].strip(" .!?")
            if typed_payload:
                open_cmd = (
                    jarvis_response_parser.DrawingCommand(
                        kind="open_browser",
                        body="https://www.google.com",
                        label="default browser",
                    )
                    if normalized_app_token == "browser"
                    else jarvis_response_parser.DrawingCommand(
                        kind="open_app",
                        body=normalized_app_token,
                        label=normalized_app_token,
                    )
                )
                is_browser_submit = (
                    normalized_app_token in ("chrome", "edge", "msedge", "firefox", "browser")
                    or action_verb.startswith("search")
                    or action_verb.startswith("look")
                )
                return [
                    open_cmd,
                    jarvis_response_parser.DrawingCommand(
                        kind="type_text",
                        body=typed_payload,
                        label=typed_payload,
                        header="submit" if is_browser_submit else "",
                    ),
                ]

        browser_match = re.search(
            r"\b(?:open|launch|start)\s+(?:my\s+)?(?:web\s+)?browser\b",
            lowered,
        )
        if browser_match:
            return [
                jarvis_response_parser.DrawingCommand(
                    kind="open_browser",
                    body="https://www.google.com",
                    label="default browser",
                )
            ]

        alias_alternation = "|".join(
            re.escape(alias_key)
            for alias_key in sorted(cls.KNOWN_URL_ALIASES.keys(), key=len, reverse=True)
        )
        url_match = re.match(
            rf"^(?:please\s+)?(?:open|go to|launch|visit)\s+(?:(?:the|my)\s+)?([a-z0-9.\-]+\.[a-z]{{2,}}|{alias_alternation})(?:\s+for\s+me)?$",
            lowered,
        )
        if url_match:
            target = url_match.group(1).strip().lower()
            resolved_url = cls.KNOWN_URL_ALIASES.get(target) or cls.normalize_target_url(target) or target
            return [
                jarvis_response_parser.DrawingCommand(
                    kind="open_url",
                    body=resolved_url,
                    label=target,
                )
            ]

        search_match = re.match(
            r"^(?:please\s+)?(?:google|search(?:\s+(?:the\s+web|google|chrome))?(?:\s+for)?)\s+(.+)$",
            lowered,
        )
        if search_match:
            query_text = cleaned[search_match.start(1) :].strip(" .!?")
            deictic_check = re.sub(
                r"\s+(?:on\s+(?:google|chrome|the\s+web)|in\s+chrome)$",
                "",
                query_text.lower(),
            ).strip()
            if deictic_check in cls._DEICTIC_SCREEN_REFERENCES:
                return []
            if query_text:
                direct_url = cls.normalize_target_url(query_text)
                if direct_url is not None and (
                    "." in query_text or query_text.lower() in cls.KNOWN_URL_ALIASES
                ):
                    return [
                        jarvis_response_parser.DrawingCommand(
                            kind="open_url",
                            body=direct_url,
                            label=query_text,
                        )
                    ]
                return [
                    jarvis_response_parser.DrawingCommand(
                        kind="search_web",
                        body=query_text,
                        label=query_text,
                    )
                ]

        direct_type_match = re.match(
            r"^(?:please\s+)?(?:type|write)\s+(.+)$",
            lowered,
        )
        if direct_type_match:
            typed_text = cleaned[direct_type_match.start(1) :].strip(" .!?")
            if typed_text:
                return [
                    jarvis_response_parser.DrawingCommand(
                        kind="type_text",
                        body=typed_text,
                        label=typed_text,
                    )
                ]

        project_open_match = re.match(
            r"^(?:please\s+)?(?:open|launch|load)\s+(?:my\s+)?(.+?)\s+project(?:\s+in\s+(vs\s?code|vscode|explorer|terminal|powershell))?$",
            lowered,
        )
        if project_open_match:
            proj_query = project_open_match.group(1).strip()
            proj_tool = (project_open_match.group(2) or "auto").replace(" ", "").lower()
            if proj_tool == "vs code":
                proj_tool = "vscode"
            if proj_query and proj_query not in cls._DEICTIC_SCREEN_REFERENCES:
                return [
                    jarvis_response_parser.DrawingCommand(
                        kind="open_project",
                        body=proj_query,
                        header=proj_tool,
                        label=proj_query,
                    )
                ]

        click_match = re.match(
            r"^(?:please\s+)?click\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button|\s+tab|\s+link)?$",
            lowered,
        )
        if click_match:
            click_target = cleaned[click_match.start(1) : click_match.end(1)].strip(" .!?\"'")
            if click_target and click_target.lower() not in cls._DEICTIC_SCREEN_REFERENCES:
                return [
                    jarvis_response_parser.DrawingCommand(
                        kind="click_ui",
                        body=click_target,
                        label=click_target,
                    )
                ]

        app_match = re.match(
            r"^(?:please\s+)?(?:open|launch|start)\s+(?:my\s+)?(chrome|google\s+chrome|edge|firefox|notepad|calculator|calc|paint|explorer|terminal|powershell|cmd|vs\s?code|vscode|settings|downloads|documents|desktop|pictures|projects|spotify|whatsapp|discord|slack|obs|vlc|postman|task\s+manager)$",
            lowered,
        )
        if app_match:
            app_name = app_match.group(1).strip()
            if app_name == "google chrome":
                app_name = "chrome"
            elif app_name == "task manager":
                app_name = "taskmgr"
            return [
                jarvis_response_parser.DrawingCommand(
                    kind="open_app",
                    body=app_name,
                    label=app_name,
                )
            ]

        generic_open_match = re.match(
            r"^(?:please\s+)?(?:open|launch|start)\s+(?:the\s+|my\s+)?([a-zA-Z0-9][a-zA-Z0-9\s\-_.:\\]{1,48})(?:\s+in\s+(vs\s?code|vscode|explorer|terminal))?$",
            lowered,
        )
        if generic_open_match and (not allow_tab_search):
            candidate_open = generic_open_match.group(1).strip(" .!?")
            open_tool = (generic_open_match.group(2) or "auto").replace(" ", "").lower()
            if candidate_open and candidate_open.lower() not in cls._DEICTIC_SCREEN_REFERENCES and (
                "tab" not in candidate_open.lower() and "stock" not in candidate_open.lower() and "price" not in candidate_open.lower()
            ):
                if ProjectWorkspaceService.resolve_project_dir(candidate_open) is not None:
                    return [
                        jarvis_response_parser.DrawingCommand(
                            kind="open_project",
                            body=candidate_open,
                            header=open_tool,
                            label=candidate_open,
                        )
                    ]
                if cls._find_installed_app_target(candidate_open) or os.path.exists(candidate_open):
                    return [
                        jarvis_response_parser.DrawingCommand(
                            kind="open_app",
                            body=candidate_open,
                            label=candidate_open,
                        )
                    ]

        tab_open_match = re.match(
            r"^(?:please\s+)?(?:open|launch|show|find|look\s+up|check)\s+(?:a\s+tab\s+(?:for|of)\s+)?(.+)$",
            lowered,
        )
        if tab_open_match and (allow_tab_search or "stock" in lowered or "price" in lowered):
            target_payload = cleaned[tab_open_match.start(1) :].strip(" .!?")
            if target_payload.lower() in cls._DEICTIC_SCREEN_REFERENCES:
                return []
            if target_payload:
                direct_url = cls.normalize_target_url(target_payload)
                if direct_url is not None and (
                    "." in target_payload or target_payload.lower() in cls.KNOWN_URL_ALIASES
                ):
                    return [
                        jarvis_response_parser.DrawingCommand(
                            kind="open_url",
                            body=direct_url,
                            label=target_payload,
                        )
                    ]
                return [
                    jarvis_response_parser.DrawingCommand(
                        kind="search_web",
                        body=target_payload,
                        label=target_payload,
                    )
                ]

        return []

    @classmethod
    def detect_direct_user_agent_commands(
        cls,
        user_prompt: str,
    ) -> List[jarvis_response_parser.DrawingCommand]:
        """Recognizes single, compound, and multi-task sequential user agent requests."""
        cleaned = re.sub(r"\s+", " ", (user_prompt or "").strip())
        if not cleaned:
            return []

        primary_clauses = [
            clause_item.strip()
            for clause_item in re.split(
                r"\s*(?:\.\s+|;\s*|,\s*then\b|;\s*then\b|\band\s+then\b|\bthen\b|\band\s+also\b|,\s*and\b)\s*",
                cleaned,
                flags=re.IGNORECASE,
            )
            if clause_item and clause_item.strip()
        ]
        if not primary_clauses:
            primary_clauses = [cleaned]

        collected_commands: List[jarvis_response_parser.DrawingCommand] = []
        browser_context_seen = False
        for primary_clause in primary_clauses:
            clause_lower = primary_clause.lower()
            has_tab_cue = bool(
                re.search(r"\b(?:another|new|second|next)\s+tab\b", clause_lower)
            )
            starts_compound_app = bool(
                re.match(
                    r"^(?:please\s+)?(?:open|launch|start|go\s+to)\s+(?:my\s+)?(?:windows\s+)?(?:settings|chrome|google\s+chrome|edge|firefox|browser|web\s+browser|notepad|terminal|powershell|cmd|vs\s?code|vscode)\s+(?:and|then)\b",
                    primary_clause,
                    flags=re.IGNORECASE,
                )
            )
            subclauses = [
                sub_item.strip()
                for sub_item in re.split(
                    r"\s+\band\b\s+(?=(?:(?:in|on)\s+)?(?:(?:another|a\s+new|new|second|next)\s+tab\b|(?:please\s+)?(?:open|launch|start|go\s+to|visit|search|google|press)\b))",
                    primary_clause,
                    flags=re.IGNORECASE,
                )
                if sub_item and sub_item.strip()
            ]
            if len(subclauses) >= 2 and not starts_compound_app:
                for subclause in subclauses:
                    sub_has_tab_cue = bool(
                        re.search(r"\b(?:another|new|second|next)\s+tab\b", subclause.lower())
                    )
                    sub_cmds = cls._detect_single_clause_commands(
                        subclause,
                        allow_tab_search=browser_context_seen or sub_has_tab_cue,
                    )
                    if sub_cmds:
                        collected_commands.extend(sub_cmds)
                        if any(
                            cmd_item.kind in ("open_browser", "open_url", "search_web")
                            or (cmd_item.kind == "open_app" and (cmd_item.body or "").strip().lower() in ("chrome", "edge", "firefox", "browser"))
                            for cmd_item in sub_cmds
                        ):
                            browser_context_seen = True
                continue

            direct_cmds = cls._detect_single_clause_commands(
                primary_clause,
                allow_tab_search=browser_context_seen or has_tab_cue,
            )
            if direct_cmds:
                collected_commands.extend(direct_cmds)
                if any(
                    cmd_item.kind in ("open_browser", "open_url", "search_web")
                    or (cmd_item.kind == "open_app" and (cmd_item.body or "").strip().lower() in ("chrome", "edge", "firefox", "browser"))
                    for cmd_item in direct_cmds
                ):
                    browser_context_seen = True
                continue

            if len(subclauses) >= 2:
                for subclause in subclauses:
                    sub_has_tab_cue = bool(
                        re.search(r"\b(?:another|new|second|next)\s+tab\b", subclause.lower())
                    )
                    sub_cmds = cls._detect_single_clause_commands(
                        subclause,
                        allow_tab_search=browser_context_seen or sub_has_tab_cue,
                    )
                    if sub_cmds:
                        collected_commands.extend(sub_cmds)
                        if any(
                            cmd_item.kind in ("open_browser", "open_url", "search_web")
                            or (cmd_item.kind == "open_app" and (cmd_item.body or "").strip().lower() in ("chrome", "edge", "firefox", "browser"))
                            for cmd_item in sub_cmds
                        ):
                            browser_context_seen = True

        return collected_commands

    @classmethod
    def detect_direct_user_agent_command(
        cls,
        user_prompt: str,
    ) -> Optional[jarvis_response_parser.DrawingCommand]:
        """Backward-compatible single-command accessor for direct user agent requests."""
        detected_commands = cls.detect_direct_user_agent_commands(user_prompt)
        return detected_commands[0] if detected_commands else None

    @classmethod
    def coalesce_agent_commands(
        cls,
        commands: List[jarvis_response_parser.DrawingCommand],
    ) -> List[jarvis_response_parser.DrawingCommand]:
        """Merges fragile two-step launch+type pairs into direct Chrome search URLs or deep Settings URIs."""
        coalesced: List[jarvis_response_parser.DrawingCommand] = []
        scan_idx = 0
        total_cmds = len(commands)
        while scan_idx < total_cmds:
            current_cmd = commands[scan_idx]
            if (
                scan_idx + 1 < total_cmds
                and current_cmd.kind == "open_app"
                and (current_cmd.body or "").strip().lower() in ("chrome", "googlechrome")
                and commands[scan_idx + 1].kind in ("open_url", "open_browser", "search_web")
            ):
                target_next = commands[scan_idx + 1]
                if target_next.kind == "search_web":
                    coalesced.append(
                        jarvis_response_parser.DrawingCommand(
                            kind="open_url",
                            body=cls._target_to_chrome_url(target_next.body or target_next.label),
                            label=target_next.label or target_next.body,
                            header="new_window",
                        )
                    )
                else:
                    coalesced.append(
                        jarvis_response_parser.DrawingCommand(
                            kind=target_next.kind,
                            body=target_next.body,
                            label=target_next.label,
                            header="new_window",
                        )
                    )
                scan_idx += 2
                continue
            if scan_idx + 1 < total_cmds and commands[scan_idx + 1].kind == "type_text":
                next_cmd = commands[scan_idx + 1]
                next_text = (next_cmd.body or next_cmd.label or "").strip()
                is_browser_launch = (
                    current_cmd.kind == "open_browser"
                    or (
                        current_cmd.kind == "open_app"
                        and (current_cmd.body or "").strip().lower()
                        in ("chrome", "googlechrome", "edge", "msedge", "firefox", "browser")
                    )
                )
                if is_browser_launch and next_text:
                    maybe_url = cls.normalize_target_url(next_text)
                    if maybe_url is not None and " " not in next_text:
                        coalesced.append(
                            jarvis_response_parser.DrawingCommand(
                                kind="open_url",
                                body=maybe_url,
                                label=next_text,
                            )
                        )
                        scan_idx += 2
                        continue
                    if (next_cmd.header or "").strip().lower() == "submit":
                        coalesced.append(
                            jarvis_response_parser.DrawingCommand(
                                kind="search_web",
                                body=next_text,
                                label=next_text,
                            )
                        )
                        scan_idx += 2
                        continue
                if current_cmd.kind == "open_app" and (current_cmd.body or "").strip().lower().startswith("settings") and next_text:
                    coalesced.append(
                        jarvis_response_parser.DrawingCommand(
                            kind="open_app",
                            body=f"settings:{next_text}",
                            label=f"Settings: {next_text}",
                        )
                    )
                    scan_idx += 2
                    continue
            coalesced.append(current_cmd)
            scan_idx += 1
        return coalesced

    @classmethod
    def check_command_requires_admin(
        cls,
        command: jarvis_response_parser.DrawingCommand,
    ) -> Optional[AdminPermissionRequest]:
        """Checks whether an LLM-emitted command asks for admin approval or touches destructive keys."""
        if command.kind == "request_admin_permission":
            action_title = (command.label or "Protected Action").strip()
            risk_desc = (command.body or "This action modifies files or system state.").strip()
            return AdminPermissionRequest(
                action_title=action_title,
                risk_description=f"{risk_desc} Confirm to approve or cancel to abort.",
                is_permanently_blocked=True,
            )
        if command.kind == "press_keys":
            combo_lower = (command.body or command.label or "").lower().replace(" ", "")
            if any(risky_combo in combo_lower for risky_combo in ("alt+f4", "shift+delete", "shift+del")):
                return AdminPermissionRequest(
                    action_title=f"Send shortcut {combo_lower}",
                    risk_description="Closing an app or deleting items can cause unsaved data loss. Reply 'confirm' to proceed or 'cancel' to abort.",
                    is_permanently_blocked=False,
                    approved_commands=[command],
                )
        return None

    @classmethod
    def check_destructive_or_admin_request(cls, user_prompt: str) -> Optional[AdminPermissionRequest]:
        """Detects potentially destructive user requests and returns an AdminPermissionRequest gate."""
        normalized = re.sub(r"\s+", " ", (user_prompt or "").lower().strip())
        if not normalized:
            return None

        if re.search(
            r"\b(?:format\s+(?:drive|[a-z]:)|wipe\s+(?:disk|drive|system|pc)|erase\s+(?:disk|drive|all)|"
            r"delete\s+(?:all\s+)?(?:system|windows|important|my)?\s*files?|remove\s+all\s+files?|"
            r"rm\s+-rf|rmdir|del\s+/f|empty\s+(?:the\s+)?recycle\s+bin)\b",
            normalized,
        ):
            return AdminPermissionRequest(
                action_title="Delete or Erase Files / Drive Data",
                risk_description=(
                    "Deleting files, clearing the Recycle Bin, or formatting a drive can permanently destroy data. "
                    "Admin confirmation is required — reply 'confirm' or 'cancel'."
                ),
                is_permanently_blocked=True,
            )

        if re.search(
            r"\b(?:reg\s+delete|(?:edit|modify|delete)\s+(?:the\s+)?registry|"
            r"disable\s+(?:windows\s+)?defender|disable\s+firewall|turn\s+off\s+firewall|"
            r"kill\s+process|terminate\s+process|taskkill)\b",
            normalized,
        ):
            return AdminPermissionRequest(
                action_title="Modify Protected System Security or Processes",
                risk_description=(
                    "Changing Windows Defender, firewall, registry keys, or killing processes requires administrator privileges "
                    "and can destabilize Windows. Reply 'confirm' to acknowledge or 'cancel' to abort."
                ),
                is_permanently_blocked=True,
            )

        if re.search(r"\b(?:shut\s*down|power\s+off|restart\s+(?:my\s+)?(?:pc|computer|laptop)|reboot)\b", normalized):
            return AdminPermissionRequest(
                action_title="Shut Down or Restart Windows",
                risk_description=(
                    "Powering off or restarting Windows will close all active applications and may discard unsaved work. "
                    "Reply 'confirm' to proceed to Power Settings or 'cancel' to abort."
                ),
                is_permanently_blocked=False,
                approved_commands=[
                    jarvis_response_parser.DrawingCommand(
                        kind="open_app",
                        body="settings:power",
                        label="Power & Sleep Settings",
                    )
                ],
            )

        if re.search(r"\buninstall\b", normalized):
            return AdminPermissionRequest(
                action_title="Uninstall Application",
                risk_description=(
                    "Uninstalling removes programs and application data from this PC. "
                    "Reply 'confirm' to open Installed Apps Settings or 'cancel' to abort."
                ),
                is_permanently_blocked=False,
                approved_commands=[
                    jarvis_response_parser.DrawingCommand(
                        kind="open_app",
                        body="settings:apps",
                        label="Installed Apps Settings",
                    )
                ],
            )

        if re.search(r"\b(?:alt\s*\+\s*f4|shift\s*\+\s*del(?:ete)?)\b", normalized):
            shortcut_combo = "alt+f4" if "f4" in normalized else "shift+delete"
            return AdminPermissionRequest(
                action_title=f"Execute {shortcut_combo}",
                risk_description=(
                    f"Running {shortcut_combo} closes windows or permanently deletes items. "
                    "Reply 'confirm' to proceed or 'cancel' to abort."
                ),
                is_permanently_blocked=False,
                approved_commands=[
                    jarvis_response_parser.DrawingCommand(
                        kind="press_keys",
                        body=shortcut_combo,
                        label=shortcut_combo,
                    )
                ],
            )

        return None

    @staticmethod
    def _extract_web_lookup_query(user_prompt: str) -> str:
        cleaned = re.sub(r"\s+", " ", (user_prompt or "").strip()).strip(" ?!.")
        cleaned = re.sub(
            r"^(?:please\s+|hey\s+jarvis\s*,?\s*|jarvis\s*,?\s*)?"
            r"(?:can\s+you\s+|could\s+you\s+|go\s+to\s+chrome\s+and\s+|"
            r"open\s+chrome\s+and\s+|check\s+chrome\s+for\s+|"
            r"tell\s+me\s+(?:what\s+(?:is|are)\s+)?|what(?:'s|\s+is|\s+are)\s+(?:the\s+)?|check\s+)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"\s+(?:and\s+tell\s+me|on\s+chrome|in\s+chrome|right\s+now|today|for\s+me)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ?!.")
        return cleaned or (user_prompt or "").strip()

    @classmethod
    def _extract_side_by_side_targets(cls, user_prompt: str) -> Tuple[str, str]:
        """Extracts optional left/right URLs or topics from a side-by-side arrangement command."""
        cleaned = re.sub(r"\s+", " ", (user_prompt or "").strip())
        if not cleaned:
            return ("", "")

        side_clauses = [
            part.strip()
            for part in re.split(r"\s*(?:\.\s+|;\s*)\s*", cleaned)
            if part and "side" in part.lower()
        ]
        candidate_clause = side_clauses[-1] if side_clauses else cleaned
        candidate_clause = re.sub(
            r"^(?:please\s+)?(?:(?:open|launch|start)\s+(?:my\s+)?(?:chrome|google\s+chrome|browser)\s*(?:[.,;]\s*|\s+then\s+|\s+and\s+)?)?"
            r"(?:(?:and\s+)?then\s+)?(?:keep|put|open|place|show|tile|split|arrange)\s+(?:my\s+|both\s+|the\s+)?",
            "",
            candidate_clause,
            flags=re.IGNORECASE,
        ).strip()

        pair_match = re.match(
            r"^(.+?)\s+(?:and|with|next\s+to|beside)\s+(.+?)\s+side[\s\-]+by[\s\-]+side\b",
            candidate_clause,
            flags=re.IGNORECASE,
        )
        generic_placeholders = {
            "",
            "those",
            "these",
            "them",
            "the",
            "both",
            "two",
            "2",
            "those 2",
            "these 2",
            "those two",
            "these two",
            "tabs",
            "tab",
            "those tabs",
            "these tabs",
            "those 2 tabs",
            "these 2 tabs",
            "the tabs",
            "my tabs",
            "chrome tabs",
            "two tabs",
            "both tabs",
            "windows",
            "those windows",
            "these windows",
            "chrome windows",
            "the windows",
            "screens",
        }
        left_clean = ""
        right_clean = ""
        if pair_match:
            left_clean = re.sub(
                r"^(?:the\s+|both\s+|tabs?\s+(?:for|of)\s+)+|\s+(?:tabs?|windows?|in\s+chrome)$",
                "",
                pair_match.group(1).strip(" .,!?\"'"),
                flags=re.IGNORECASE,
            ).strip()
            right_clean = re.sub(
                r"^(?:the\s+|both\s+|tabs?\s+(?:for|of)\s+)+|\s+(?:tabs?|windows?|in\s+chrome)$",
                "",
                pair_match.group(2).strip(" .,!?\"'"),
                flags=re.IGNORECASE,
            ).strip()
            if (
                left_clean.lower() not in generic_placeholders
                and right_clean.lower() not in generic_placeholders
            ):
                return (left_clean, right_clean)

        extracted_targets: List[str] = []
        prior_cmds = cls.detect_direct_user_agent_commands(user_prompt)
        for cmd_item in prior_cmds:
            candidate_val = (cmd_item.body or "").strip()
            if not candidate_val:
                continue
            if cmd_item.kind in ("open_url", "search_web"):
                extracted_targets.append(candidate_val)
            elif cmd_item.kind == "open_app" and candidate_val.lower() not in (
                "chrome",
                "googlechrome",
                "edge",
                "msedge",
                "firefox",
                "browser",
            ):
                extracted_targets.append(candidate_val)
        if len(extracted_targets) >= 2:
            return (extracted_targets[0], extracted_targets[1])
        return ("", "")

    @classmethod
    def classify_contextual_intent(cls, user_prompt: str) -> ContextualIntentPlan:
        """
        Routes natural user prompts with practical intelligence:
        1. Admin confirmation gate for destructive operations
        2. Live Windows telemetry + deep-linked Settings pages (Storage, RAM, Battery)
        3. Autonomous Chrome tab / window side-by-side tiling
        4. Compound / multi-tab multitask desktop & browser agent chains (coalesced cleanly)
        5. Live web / stock / market lookup -> open Chrome search + read exact answer back
        6. Standard visual screen tutor & drawing explanation
        """
        cleaned_prompt = re.sub(r"\s+", " ", (user_prompt or "").strip())
        lowered = cleaned_prompt.lower()

        admin_request = cls.check_destructive_or_admin_request(cleaned_prompt)
        if admin_request is not None:
            return ContextualIntentPlan(
                intent_kind="admin_permission_gate",
                admin_request=admin_request,
            )

        storage_phrases = (
            "storage left",
            "disk space",
            "free space",
            "how much storage",
            "check my storage",
            "drive space",
            "storage space",
            "how much space",
            "available storage",
            "remaining storage",
            "storage remaining",
        )
        is_storage_question = (
            any(phrase in lowered for phrase in storage_phrases)
            or bool(
                re.search(
                    r"\b(?:storage|disk|hard\s*drive|ssd)\b.*\b(?:left|have|got|free|available|remaining|used|full|size|capacity|check|status)\b",
                    lowered,
                )
            )
            or bool(
                re.search(
                    r"\b(?:how\s+much|what(?:'s|\s+is)?(?:\s+my)?|check(?:\s+my)?|show(?:\s+my)?)\b.*\b(?:storage|disk\s+space|drive\s+space)\b",
                    lowered,
                )
            )
        )
        if is_storage_question:
            storage_text = SystemTelemetryService.get_storage_summary()
            return ContextualIntentPlan(
                intent_kind="system_telemetry_and_settings",
                spoken_step_intro="Opening Windows Storage Settings.",
                telemetry_response=f"Opened Windows Storage Settings. {storage_text}",
                agent_commands=[
                    jarvis_response_parser.DrawingCommand(
                        kind="open_app",
                        body="settings:storage",
                        label="Windows Storage Settings",
                    )
                ],
            )

        if any(
            phrase in lowered
            for phrase in ("how much ram", "ram left", "memory left", "ram usage", "memory usage", "check my ram", "how much memory")
        ):
            ram_text = SystemTelemetryService.get_memory_summary()
            return ContextualIntentPlan(
                intent_kind="system_telemetry_and_settings",
                spoken_step_intro="Opening System About Settings.",
                telemetry_response=f"Opened System Settings. {ram_text}",
                agent_commands=[
                    jarvis_response_parser.DrawingCommand(
                        kind="open_app",
                        body="settings:ram",
                        label="System About Settings",
                    )
                ],
            )

        if any(
            phrase in lowered
            for phrase in ("battery left", "battery percentage", "how much battery", "check battery", "battery level")
        ):
            battery_text = SystemTelemetryService.get_battery_summary()
            return ContextualIntentPlan(
                intent_kind="system_telemetry_and_settings",
                spoken_step_intro="Opening Battery Settings.",
                telemetry_response=f"Opened Battery Settings. {battery_text}",
                agent_commands=[
                    jarvis_response_parser.DrawingCommand(
                        kind="open_app",
                        body="settings:battery",
                        label="Battery Settings",
                    )
                ],
            )

        is_side_by_side_request = bool(
            re.search(
                r"\b(?:side[\s\-]+by[\s\-]+side|split[\s\-]+screen|tile\s+(?:the\s+|these\s+|those\s+|my\s+)?(?:tabs?|windows?|chrome))\b",
                lowered,
            )
        )
        if is_side_by_side_request:
            left_target, right_target = cls._extract_side_by_side_targets(cleaned_prompt)
            pair_label = (
                f"{left_target} | {right_target}"
                if left_target and right_target
                else "Chrome side by side"
            )
            step_intro = (
                f"Opening {left_target} and {right_target} side by side."
                if left_target and right_target
                else "Arranging Chrome windows side by side."
            )
            return ContextualIntentPlan(
                intent_kind="multitask_agent",
                spoken_step_intro=step_intro,
                agent_commands=[
                    jarvis_response_parser.DrawingCommand(
                        kind="open_side_by_side",
                        body=left_target,
                        header=right_target,
                        label=pair_label,
                    )
                ],
            )

        direct_commands = cls.detect_direct_user_agent_commands(cleaned_prompt)
        has_explicit_multitab_chain = len(direct_commands) >= 2 or bool(
            re.search(r"\b(?:another|new|second|next)\s+tab\b", lowered)
        )

        is_web_lookup_question = (not has_explicit_multitab_chain) and (
            any(
                phrase in lowered
                for phrase in (
                    "stock price",
                    "share price",
                    "price of ",
                    "bitcoin price",
                    "crypto price",
                    "weather in ",
                    "latest news on ",
                    "who won ",
                )
            )
            or bool(
                re.search(
                    r"\b(?:search|check|look\s+up|find)\b.*\b(?:chrome|google)\b.*\b(?:tell\s+me|what\s+is|price|answer)\b",
                    lowered,
                )
            )
        )
        if is_web_lookup_question:
            lookup_query = cls._extract_web_lookup_query(cleaned_prompt)
            return ContextualIntentPlan(
                intent_kind="web_lookup_and_answer",
                spoken_step_intro=f"Searching Chrome for {lookup_query[:38]}.",
                search_query=lookup_query,
            )

        is_workspace_build_request = bool(
            re.search(
                r"\b(?:build|create|scaffold|generate|make|write|develop|setup|set\s+up)\b.*\b(?:project|app|website|webpage|web\s*app|script|api|server|calculator|dashboard|game|tool|portfolio|todo|cli)\b",
                lowered,
            )
            or re.search(r"\bin\s+d:\\projects\b", lowered)
            or bool(
                re.search(
                    r"\b(?:inspect|find|click|fix|debug|test|run)\b.*\b(?:project|tests?|build|ui|window|controls?)\b",
                    lowered,
                )
            )
        )
        if is_workspace_build_request:
            return ContextualIntentPlan(
                intent_kind="autonomous_workspace_agent",
                spoken_step_intro="On it, sir. Planning workspace steps now.",
            )

        if direct_commands:
            coalesced_commands = cls.coalesce_agent_commands(direct_commands)
            return ContextualIntentPlan(
                intent_kind="multitask_agent",
                agent_commands=coalesced_commands,
            )

        return ContextualIntentPlan(intent_kind="visual_explanation")
