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
import io
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, quote_plus, urlparse

import mss
import subprocess
import wave
import webbrowser

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

        if highlight_points and len(highlight_points) >= 2:
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
    def snap_drawing_command_to_content(
        cls,
        command: jarvis_response_parser.DrawingCommand,
        reference_image: Optional[Image.Image] = None,
    ) -> jarvis_response_parser.DrawingCommand:
        """
        Refines vertical underline placement using luminance contrast rows in the captured
        screenshot so underlines rest cleanly beneath actual text instead of crossing letters.
        Adjustment is capped to +-18 logical pixels to guard against background gradients.
        """
        source_image = reference_image or cls._last_raw_logical_image
        if source_image is None or command.kind not in ("underline", "highlight"):
            return command

        try:
            img_width, img_height = source_image.size
            left_px = max(0, int(round(command.x)))
            right_px = min(img_width, int(round(command.x + max(24.0, command.width))))
            center_y_px = int(round(command.y))
            top_px = max(0, center_y_px - 22)
            bottom_px = min(img_height, center_y_px + 22)
            if (right_px - left_px) < 16 or (bottom_px - top_px) < 10:
                return command

            strip_gray = np.asarray(
                source_image.crop((left_px, top_px, right_px, bottom_px)).convert("L"),
                dtype=np.float32,
            )
            bg_luminance = float(np.median(strip_gray))
            row_activity = np.mean(np.abs(strip_gray - bg_luminance) > 16.0, axis=1)
            active_rows = row_activity > 0.025
            if not np.any(active_rows):
                return command

            # Find contiguous vertical text glyph bands inside the strip
            text_bands: List[Tuple[int, int]] = []
            band_start: Optional[int] = None
            for row_offset, is_active in enumerate(active_rows):
                if is_active and band_start is None:
                    band_start = row_offset
                elif not is_active and band_start is not None:
                    if (row_offset - band_start) >= 3:
                        text_bands.append((band_start, row_offset - 1))
                    band_start = None
            if band_start is not None and (len(active_rows) - band_start) >= 3:
                text_bands.append((band_start, len(active_rows) - 1))

            if not text_bands:
                return command

            best_band = min(
                text_bands,
                key=lambda band: abs((top_px + band[1] + 5.0) - command.y),
            )
            snapped_y = float(top_px + best_band[1] + 5.0)
            if abs(snapped_y - command.y) > 22.0:
                return command

            snapped_x = command.x
            snapped_width = command.width
            if command.kind == "underline":
                band_rows_slice = strip_gray[best_band[0] : best_band[1] + 1, :]
                col_activity = np.mean(np.abs(band_rows_slice - bg_luminance) > 16.0, axis=0)
                active_cols = np.where(col_activity > 0.06)[0]
                if active_cols.size >= 8:
                    trimmed_left = left_px + int(active_cols[0])
                    trimmed_right = left_px + int(active_cols[-1]) + 1
                    trimmed_span = float(trimmed_right - trimmed_left)
                    if trimmed_span >= 18.0:
                        snapped_x = max(0.0, float(trimmed_left - 2))
                        snapped_width = trimmed_span + 4.0

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
            )
        except Exception:
            return command


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
    speechWordProgress = Signal(str, int, int)
    audioLevelChanged = Signal(float)
    speechFinished = Signal()

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        super().__init__()
        self._settings_store = settings_store
        self._lock = threading.Lock()
        self._generation_id = 0
        self._queue: "queue.Queue[Tuple[int, str, Optional[str]]]" = queue.Queue()
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
        target_gen = self._generation_id if turn_generation is None else turn_generation
        if target_gen == self._generation_id:
            self._queue.put((target_gen, "sentence", cleaned))

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
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def speak_blocking_for_test(self, text: str) -> Tuple[str, float]:
        """Synchronous helper used by `--selftest` to verify live ElevenLabs PCM synthesis."""
        start_time = time.perf_counter()
        engine = self._speak_one_sentence(text, generation_id=self._generation_id)
        return engine, time.perf_counter() - start_time

    def _playback_worker_loop(self) -> None:
        announced_start_for_gen = -1
        while True:
            gen_id, item_kind, payload = self._queue.get()
            if gen_id != self._generation_id:
                continue

            if item_kind == "sentence" and payload:
                if announced_start_for_gen != gen_id:
                    announced_start_for_gen = gen_id
                    self._is_speaking = True
                    self.speechStarted.emit()
                self._speak_one_sentence(payload, generation_id=gen_id)

            elif item_kind == "done":
                if announced_start_for_gen == gen_id:
                    self._is_speaking = False
                    self.audioLevelChanged.emit(0.0)
                    self.speechFinished.emit()

    def _speak_one_sentence(self, sentence_text: str, generation_id: int) -> str:
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
    ) -> None:
        api_key = self._settings_store.get_secret(jarvis_config.SECRET_RIVER_API_KEY)
        if not api_key:
            self._is_running = False
            if self._generation_id == turn_generation:
                self.turnFailed.emit("River AI API key is missing in .env.")
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
        )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for past_message in self._history[-6:]:
            messages.append(past_message)

        user_content: Any
        if jpeg_b64:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
            ]
            if lasso_crop_b64:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{lasso_crop_b64}"}}
                )
        else:
            user_content = user_prompt
        messages.append({"role": "user", "content": user_content})

        model_id = str(self._settings_store.get("chat_model", "deepseek-ai/DeepSeek-V4.1-Flash"))
        extractor = jarvis_response_parser.StreamingSentenceExtractor(scale_x=scale_x, scale_y=scale_y)

        try:
            with requests.post(
                self.RIVER_CHAT_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": 700,
                    "stream": True,
                    "reasoning_effort": "none",
                },
                stream=True,
                timeout=45,
            ) as response:
                if not response.ok:
                    error_text = response.text[:240]
                    if self._generation_id == turn_generation:
                        self._is_running = False
                        self.turnFailed.emit(f"River AI error {response.status_code}: {error_text}")
                    return

                response.encoding = "utf-8"
                for raw_line in response.iter_lines(decode_unicode=True):
                    if self._generation_id != turn_generation:
                        return
                    if not raw_line or not raw_line.startswith("data: "):
                        continue
                    data_str = raw_line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        delta_text = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    except Exception:
                        continue
                    if not delta_text:
                        continue

                    new_sentences, new_commands = extractor.feed_delta(delta_text)
                    for cmd in new_commands:
                        self.drawingCommandReady.emit(cmd)
                    for sentence in new_sentences:
                        self.sentenceReady.emit(sentence)
                    display_so_far = extractor.current_display_text()
                    if display_so_far:
                        self.partialTextUpdated.emit(display_so_far)

            if self._generation_id != turn_generation:
                return

            final_sentences, final_commands = extractor.flush_remaining()
            for cmd in final_commands:
                self.drawingCommandReady.emit(cmd)
            for sentence in final_sentences:
                self.sentenceReady.emit(sentence)

            parsed_response = extractor.build_final_response()
            if parsed_response.spoken_text:
                self._history.append({"role": "user", "content": user_prompt})
                self._history.append({"role": "assistant", "content": parsed_response.spoken_text})

            self._is_running = False
            self.turnCompleted.emit(parsed_response)
        except Exception as err:
            self._is_running = False
            if self._generation_id == turn_generation:
                self.turnFailed.emit(f"Could not reach River AI: {err}")


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
# Safe Desktop & Browser Agent Actions
# ---------------------------------------------------------------------------


class DesktopAgentService:
    """Executes bounded desktop & browser actions requested by the AI or directly by the user."""

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

    @classmethod
    def execute_agent_command(
        cls,
        command: jarvis_response_parser.DrawingCommand,
    ) -> Tuple[bool, str]:
        """Executes a parsed desktop/browser command safely and returns (ok, status_message)."""
        command_kind = command.kind
        argument_text = (command.body or command.label or command.header or "").strip()

        if command_kind in ("open_browser", "open_url"):
            resolved_url = cls.normalize_target_url(argument_text)
            if resolved_url is None:
                return False, f"Invalid URL: {argument_text}"
            try:
                opened = webbrowser.open(resolved_url, new=2)
                return bool(opened), f"Opened {resolved_url}"
            except Exception as exc:
                return False, str(exc)

        if command_kind == "search_web":
            query_text = argument_text
            if not query_text:
                return False, "Empty search query"
            search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query_text)}"
            try:
                opened = webbrowser.open(search_url, new=2)
                return bool(opened), f"Searched Google for {query_text}"
            except Exception as exc:
                return False, str(exc)

        if command_kind == "open_app":
            normalized_app = re.sub(r"[\s_\-]+", "", argument_text.lower())
            if normalized_app in ("settings", "windowssettings"):
                try:
                    os.startfile("ms-settings:")
                    return True, "Opened Windows Settings"
                except Exception as exc:
                    return False, str(exc)
            if normalized_app in ("browser", "webbrowser", "defaultbrowser"):
                return cls.execute_agent_command(
                    jarvis_response_parser.DrawingCommand(kind="open_browser", label=argument_text)
                )
            launch_args = cls.SAFE_APP_LAUNCHERS.get(normalized_app)
            if launch_args is None:
                return False, f"App not in safe launcher allowlist: {argument_text}"
            try:
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
                if len(normalized_tokens) == 1:
                    pyautogui.press(normalized_tokens[0])
                else:
                    pyautogui.hotkey(*normalized_tokens)
                return True, f"Pressed {'+'.join(normalized_tokens)}"
            except Exception as exc:
                return False, str(exc)

        if command_kind == "type_text":
            if not argument_text:
                return False, "No text to type"
            try:
                pyautogui.write(argument_text[:280], interval=0.012)
                return True, f"Typed {len(argument_text[:280])} characters"
            except Exception as exc:
                return False, str(exc)

        return False, f"Unknown agent command: {command_kind}"

    @classmethod
    def detect_direct_user_agent_command(
        cls,
        user_prompt: str,
    ) -> Optional[jarvis_response_parser.DrawingCommand]:
        """Fast local fallback recognizing direct open/search user requests even before/without AI tags."""
        cleaned = re.sub(r"\s+", " ", (user_prompt or "").strip())
        if not cleaned:
            return None
        lowered = cleaned.lower().strip(" .,!?")

        browser_match = re.search(
            r"\b(?:open|launch|start)\s+(?:my\s+)?(?:web\s+)?browser\b",
            lowered,
        )
        if browser_match:
            return jarvis_response_parser.DrawingCommand(
                kind="open_browser",
                body="https://www.google.com",
                label="default browser",
            )

        url_match = re.match(
            r"^(?:please\s+)?(?:open|go to|launch|visit)\s+([a-z0-9.\-]+(?:\.[a-z]{2,}|chatgpt|youtube|github|gmail|claude|notion))(?:\s+for\s+me)?$",
            lowered,
        )
        if url_match:
            target = url_match.group(1)
            return jarvis_response_parser.DrawingCommand(
                kind="open_url",
                body=target,
                label=target,
            )

        search_match = re.match(
            r"^(?:please\s+)?(?:google|search(?:\s+the\s+web)?\s+for)\s+(.+)$",
            lowered,
        )
        if search_match:
            query_text = cleaned[search_match.start(1) :].strip(" .!?")
            if query_text:
                return jarvis_response_parser.DrawingCommand(
                    kind="search_web",
                    body=query_text,
                    label=query_text,
                )

        app_match = re.match(
            r"^(?:please\s+)?(?:open|launch|start)\s+(?:my\s+)?(chrome|edge|firefox|notepad|calculator|calc|paint|explorer|terminal|powershell|cmd|vs\s?code|vscode|settings)$",
            lowered,
        )
        if app_match:
            app_name = app_match.group(1).strip()
            return jarvis_response_parser.DrawingCommand(
                kind="open_app",
                body=app_name,
                label=app_name,
            )

        return None
