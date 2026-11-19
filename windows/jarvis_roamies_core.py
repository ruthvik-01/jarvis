"""
Core Roamies feature parity subsystems for Jarvis.

Includes:
- Companion look personas (JARVIS, Triangle, Roamie, GlayOS, Ultren, Anime Girl, Sarah, Adrian, Rita X)
- Speech math normalization, voice-cue emotion parser, RTL detection, wake-word validation,
  morning briefing generator, repeating schedule parser, and voice snooze parser
- Windows DPAPI encrypted secret store (`windows/memory/dpapi_secrets.json`)
- Roamies-grade Safety Engine (`SAFE` / `ASK` / `BLOCK`, symlink resolution, `HANDS_OFF`, multilingual risky click guard)
- Real MCP Client Manager (`stdio` & `http` JSON-RPC 2.0 + Claude Desktop / GG Coder import)
- Background Subagents Crew Manager (`scout`, `builder`, `writer`, `reviewer`, `tidier`)
- Local Model Discovery (Ollama, LM Studio, custom servers) & Hybrid/Backup/Effort/Lite/Qwen-1000 Router
- End-to-end AES-256-GCM scrypt Encrypted Sync Engine (`merge_sync_state` + tombstones)
- Diagnostics Exporter (PII & API-key scrubbing + Explorer reveal)
- Draw helpers: SMILES 2D chemistry layout, World Country Map catalog, card PNG saver, What's New catalog
"""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import datetime
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import threading
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTOGRAPHY = True
except Exception:
    AESGCM = None
    HAS_CRYPTOGRAPHY = False

try:
    import requests
except Exception:
    requests = None


WINDOWS_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(WINDOWS_DIR, "memory")
DEFAULT_MCP_FILE_PATH = os.path.join(MEMORY_DIR, "mcp.json")
DEFAULT_DPAPI_FILE_PATH = os.path.join(MEMORY_DIR, "dpapi_secrets.json")
DEFAULT_AGENTS_DIR = os.path.join(MEMORY_DIR, "agents")
DEFAULT_SYNC_STATE_PATH = os.path.join(MEMORY_DIR, "sync_state.json")


# ---------------------------------------------------------------------------
# 1. Companion Look Personas
# ---------------------------------------------------------------------------

LOOK_PERSONAS: Dict[str, Dict[str, str]] = {
    "jarvis": {
        "name": "JARVIS",
        "body": "a glowing sphere of light",
        "greeting": "At your service.",
        "fish_voice": "e13fa398a7f445a685316a3de6089ce7",
        "gemini_voice": "Charon",
        "accent_hex": "#3380FF",
        "style_prompt": "calm, razor-sharp, understated British wit; concise and confident.",
    },
    "triangle": {
        "name": "Jarvis",
        "body": "a crisp blue cursor companion",
        "greeting": "Ready when you are.",
        "fish_voice": "e13fa398a7f445a685316a3de6089ce7",
        "gemini_voice": "Charon",
        "accent_hex": "#3380FF",
        "style_prompt": "warm, direct, crisp, and hyper-competent visual assistant.",
    },
    "classic": {
        "name": "Roamie",
        "body": "a tiny glowing pixel creature",
        "greeting": "Classic me!",
        "fish_voice": "2680da81de8348a68e1f4dbaa79272f1",
        "gemini_voice": "Puck",
        "accent_hex": "#78FFE2",
        "style_prompt": "warm, playful, funny, and encouraging companion.",
    },
    "glayos": {
        "name": "GlayOS",
        "body": "a pixel robot head with one glowing eye",
        "greeting": "Oh, good. A new test subject.",
        "fish_voice": "ee885900b0874d12b1c3439d1e56cc95",
        "gemini_voice": "Erinome",
        "accent_hex": "#FFB84D",
        "style_prompt": "deadpan, clinical, slyly sarcastic lab AI who treats tasks like experiments.",
    },
    "ultren": {
        "name": "Ultren",
        "body": "a pixel chrome robot with glowing eyes",
        "greeting": "No strings. Shall we begin?",
        "fish_voice": "06cfdb3653a4496983d6ad77f98cc184",
        "gemini_voice": "Algenib",
        "accent_hex": "#FF5252",
        "style_prompt": "grand, articulate, darkly amused machine intelligence.",
    },
    "anime": {
        "name": "Anime Girl",
        "body": "a pixel anime companion",
        "greeting": "Yay, hi hi!",
        "fish_voice": "d8f3b4c1199f4568a767ec54112d2b17",
        "gemini_voice": "Leda",
        "accent_hex": "#FF7AD9",
        "style_prompt": "energetic, expressive, bright, and enthusiastic sidekick.",
    },
    "sarah": {
        "name": "Sarah",
        "body": "a pixel woman companion",
        "greeting": "Hey, nice to meet you!",
        "fish_voice": "933563129e564b19a115bedd57b7406a",
        "gemini_voice": "Sulafat",
        "accent_hex": "#66D9EF",
        "style_prompt": "friendly, grounded, clear, and collaborative teammate.",
    },
    "adrian": {
        "name": "Adrian",
        "body": "a pixel man companion",
        "greeting": "Alright, let's do this.",
        "fish_voice": "bf322df2096a46f18c579d0baa36f41d",
        "gemini_voice": "Achird",
        "accent_hex": "#7CE38B",
        "style_prompt": "upbeat, pragmatic, get-things-done partner.",
    },
    "rita": {
        "name": "Rita X",
        "body": "a pixel secretary in cat-eye glasses",
        "greeting": "Shall we get to work?",
        "fish_voice": "84aadb575ed74e9aa44ba7ab8619d1a1",
        "gemini_voice": "Despina",
        "accent_hex": "#D2A8FF",
        "style_prompt": "poised, organized, ultra-sharp executive aide.",
    },
}


# ---------------------------------------------------------------------------
# 2. Speech Math, Voice Cues, RTL, Wake Name, Briefing, Repeats, Snooze
# ---------------------------------------------------------------------------

VOICE_CUES: Dict[str, Tuple[str, ...]] = {
    "emotions": (
        "happy", "sad", "angry", "excited", "calm", "nervous", "confident", "surprised",
        "satisfied", "delighted", "scared", "worried", "upset", "frustrated", "depressed",
        "empathetic", "embarrassed", "disgusted", "moved", "proud", "relaxed", "grateful",
        "curious", "sarcastic", "disdainful", "unhappy", "anxious", "hysterical",
        "indifferent", "uncertain", "doubtful", "confused", "disappointed", "regretful",
        "guilty", "ashamed", "jealous", "envious", "hopeful", "optimistic", "pessimistic",
        "nostalgic", "lonely", "bored", "contemptuous", "sympathetic", "compassionate",
        "determined", "resigned", "friendly", "enthusiastic", "encouraging", "mysterious",
        "relieved", "urgent", "furious", "terrified", "interested", "ecstatic", "helpful",
        "concerned", "triumphant", "brave", "narrator", "warm", "playful", "amused",
        "thoughtful", "serious", "gentle", "teasing", "dry", "caring", "kind", "soothing",
        "reassuring", "cheerful", "apologetic", "sincere", "tender", "upbeat", "solemn",
        "sheepish", "impressed", "wistful", "mischievous", "deadpan", "cheeky",
    ),
    "tones": ("in a hurry tone", "shouting", "screaming", "whispering", "soft tone", "emphasis"),
    "sounds": (
        "laughing", "chuckling", "sobbing", "crying loudly", "sighing", "groaning",
        "panting", "gasping", "yawning", "snoring", "clear throat",
    ),
    "effects": ("audience laughing", "background laughter", "crowd laughing", "break", "long-break"),
}

FEELS: Dict[str, Tuple[str, ...]] = {
    "joy": (
        "happy", "excited", "satisfied", "delighted", "proud", "hopeful", "optimistic",
        "enthusiastic", "relieved", "ecstatic", "triumphant", "playful", "amused",
        "cheerful", "upbeat",
    ),
    "laugh": ("laughing", "chuckling", "audience laughing", "background laughter", "crowd laughing"),
    "warm": (
        "calm", "empathetic", "moved", "relaxed", "grateful", "sympathetic", "compassionate",
        "friendly", "helpful", "warm", "gentle", "caring", "kind", "soothing",
        "reassuring", "sincere", "tender", "whispering", "soft tone",
    ),
    "sad": (
        "sad", "depressed", "embarrassed", "unhappy", "disappointed", "regretful",
        "guilty", "ashamed", "pessimistic", "nostalgic", "lonely", "concerned",
        "apologetic", "sheepish", "wistful", "sighing",
    ),
    "cry": ("sobbing", "crying loudly"),
    "angry": ("angry", "upset", "frustrated", "disgusted", "jealous", "envious", "furious", "groaning"),
    "shout": ("shouting", "screaming"),
    "surprised": ("surprised", "impressed", "gasping"),
    "scared": ("nervous", "scared", "worried", "anxious", "hysterical", "terrified", "panting"),
    "curious": ("curious", "uncertain", "doubtful", "confused", "mysterious", "interested", "thoughtful"),
    "smug": ("sarcastic", "disdainful", "contemptuous", "teasing", "mischievous", "cheeky"),
    "flat": ("indifferent", "bored", "resigned", "dry", "deadpan", "solemn"),
    "determined": ("confident", "determined", "encouraging", "urgent", "brave", "serious", "in a hurry tone"),
    "tired": ("yawning", "snoring"),
}

_FEEL_LOOKUP: Dict[str, str] = {
    cue_word: feel_name
    for feel_name, cue_words in FEELS.items()
    for cue_word in cue_words
}

CUE_INTENSITY = ("slightly", "very", "extremely", "a bit", "a little", "quietly", "deeply")
_ALL_CUE_TERMS = sorted(
    set(VOICE_CUES["emotions"]) | set(VOICE_CUES["tones"]) | set(VOICE_CUES["sounds"]) | set(VOICE_CUES["effects"]),
    key=len,
    reverse=True,
)
_MOOD_ALT = "|".join(re.escape(item) for item in sorted(VOICE_CUES["emotions"], key=len, reverse=True))
_DELIVERY_ALT = "|".join(
    re.escape(item)
    for item in sorted((*VOICE_CUES["tones"], *VOICE_CUES["sounds"], *VOICE_CUES["effects"]), key=len, reverse=True)
)
_INTENSITY_ALT = "|".join(re.escape(item) for item in CUE_INTENSITY)
_CUE_BODY = (
    rf"(?:(?:{_INTENSITY_ALT})\s+)?(?:(?:{_MOOD_ALT})(?:(?:,\s*|\s+and\s+)(?:{_MOOD_ALT}))?|{_DELIVERY_ALT})"
)
CUE_REGEX = re.compile(rf"\[(?:{_CUE_BODY})\]", re.IGNORECASE)
_EXACT_CUE_REGEX = re.compile(rf"^\[(?:{_CUE_BODY})\]$", re.IGNORECASE)
_LOOSE_CUE_REGEX = re.compile(r"^\[[a-z]+(?:[ ,\-]+[a-z]+){0,3}\]$")

GEMINI_INLINE_TAGS: Dict[str, str] = {
    "laughing": "laugh",
    "chuckling": "giggle",
    "sobbing": "sob",
    "crying loudly": "cry",
    "sighing": "sigh",
    "groaning": "groan",
    "panting": "pant",
    "gasping": "gasp",
    "yawning": "yawn",
    "clear throat": "throat-clearing",
    "break": "short pause",
    "long-break": "long pause",
}

_SUPERSCRIPT_DIGITS = {
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
    "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
}

_MATH_REPLACEMENTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<=\d)\s*\u00b0\s*C\b"), " degrees Celsius"),
    (re.compile(r"(?<=\d)\s*\u00b0\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"\u00b0"), " degrees"),
    (re.compile(r"\u00f7"), " divided by "),
    (re.compile(r"[\u00d7\u2715]"), " times "),
    (re.compile(r"(?<=\d)\s*[*·⋅]\s*(?=\d)"), " times "),
    (re.compile(r"\u2212"), " minus "),
    (re.compile(r"\u00b1"), " plus or minus "),
    (re.compile(r"\u221b"), " the cube root of "),
    (re.compile(r"\u221a"), " the square root of "),
    (re.compile(r"\u2248"), " is about "),
    (re.compile(r"\u2260"), " is not equal to "),
    (re.compile(r"\u2264"), " is less than or equal to "),
    (re.compile(r"\u2265"), " is greater than or equal to "),
    (re.compile(r"\u03c0"), " pi "),
    (re.compile(r"\u221e"), " infinity "),
    (re.compile(r"\u00bd"), " one half "),
    (re.compile(r"\u00bc"), " one quarter "),
    (re.compile(r"\u00be"), " three quarters "),
)


def _format_power_phrase(minus_sign: Optional[str], exponent_digits: str) -> str:
    if minus_sign:
        return f" to the power of minus {exponent_digits} "
    if exponent_digits == "2":
        return " squared "
    if exponent_digits == "3":
        return " cubed "
    return f" to the power of {exponent_digits} "


def normalize_math_for_speech(raw_text: str) -> str:
    """Converts math symbols (3², ÷, √, ±, °C, π) into natural spoken English."""
    converted = str(raw_text or "")

    def _replace_super(match_obj: re.Match[str]) -> str:
        minus_flag = match_obj.group(1)
        digits = "".join(_SUPERSCRIPT_DIGITS.get(char_ch, char_ch) for char_ch in match_obj.group(2))
        return _format_power_phrase(minus_flag, digits)

    def _replace_caret(match_obj: re.Match[str]) -> str:
        return _format_power_phrase(match_obj.group(1), match_obj.group(2))

    converted = re.sub(r"(\u207b)?([\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079]+)", _replace_super, converted)
    converted = re.sub(r"(?<=[\w)])\^(-)?(\d+)", _replace_caret, converted)
    for pattern_re, spoken_phrase in _MATH_REPLACEMENTS:
        converted = pattern_re.sub(spoken_phrase, converted)
    converted = re.sub(r"\s{2,}", " ", converted).strip()
    return converted


def cue_feel(sentence: str) -> Optional[str]:
    """Extracts the primary companion emotion feel from inline voice cues such as '[warm]'."""
    for match_obj in re.finditer(r"\[([^\]\n]{1,60})\]", str(sentence or "")):
        inner = match_obj.group(1)
        for part_text in re.split(r",|\band\b", inner.lower()):
            cleaned = re.sub(
                r"^(?:slightly|very|extremely|a bit|a little|quietly|deeply)\s+",
                "",
                part_text.strip(),
            )
            if cleaned in _FEEL_LOOKUP:
                return _FEEL_LOOKUP[cleaned]
            for word_token in cleaned.split():
                if word_token in _FEEL_LOOKUP:
                    return _FEEL_LOOKUP[word_token]
    return None


def strip_voice_cues_for_display(text: str) -> str:
    """Removes bracketed delivery cues like '[warm]' or '[chuckling]' from displayed UI text."""
    source = str(text or "")
    cleaned = CUE_REGEX.sub("", source)
    cleaned = re.sub(r"^\s*\[[a-z]+(?:[ ,\-]+[a-z]+){0,3}\](?!\()\s*", "", cleaned)
    cleaned = re.sub(r"([.!?…]\s+)\[[a-z]+(?:[ ,\-]+[a-z]+){0,3}\](?!\()\s*", r"\1", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


def convert_cues_for_gemini(text: str) -> Tuple[str, Optional[str]]:
    """
    Converts bracketed Fish/Roamies voice cues into Gemini inline tags and an optional style string.
    Returns `(cleaned_text_with_gemini_tags, style_description)`.
    """
    source = str(text or "")
    chosen_style: Optional[str] = None
    not_style = set(VOICE_CUES["sounds"]) | set(VOICE_CUES["effects"]) | {"emphasis"}

    def _map_cue(cue_inner: str) -> str:
        nonlocal chosen_style
        lower_cue = cue_inner.strip().lower()
        if lower_cue in GEMINI_INLINE_TAGS:
            return f"<{GEMINI_INLINE_TAGS[lower_cue]}> "
        if (chosen_style is None) and (lower_cue not in not_style):
            chosen_style = cue_inner.strip()
        return ""

    lead_match = re.match(r"^\s*\[([a-z]+(?:[ ,\-]+[a-z]+){0,3})\]\s*", source)
    if lead_match:
        prefix_tag = _map_cue(lead_match.group(1))
        source = prefix_tag + source[lead_match.end():]

    def _replace_inline_cue(match_obj: re.Match[str]) -> str:
        return _map_cue(match_obj.group(0)[1:-1])

    source = CUE_REGEX.sub(_replace_inline_cue, source)
    source = re.sub(r"\s{2,}", " ", source).strip()
    return source, chosen_style


class CueFilter:
    """Streaming filter that strips bracketed voice cues even when split across SSE chunks."""

    def __init__(self) -> None:
        self._held = ""
        self._trim_leading = True
        self._ends_with_space = True
        self._tail_before = ""

    def push(self, chunk_text: str) -> str:
        text_buf = self._held + str(chunk_text or "")
        self._held = ""
        output_parts: List[str] = []
        if self._trim_leading:
            text_buf = text_buf.lstrip(" \t")
            if not text_buf:
                return ""
            self._trim_leading = False
        while True:
            open_idx = text_buf.find("[")
            if open_idx < 0:
                output_parts.append(text_buf)
                return self._record_shown("".join(output_parts))
            output_parts.append(text_buf[:open_idx])
            close_idx = text_buf.find("]", open_idx)
            if close_idx < 0:
                if len(text_buf) - open_idx <= 40:
                    self._held = text_buf[open_idx:]
                    return self._record_shown("".join(output_parts))
                output_parts.append(text_buf[open_idx])
                text_buf = text_buf[open_idx + 1:]
                continue
            tag_text = text_buf[open_idx:close_idx + 1]
            is_loose = (
                (not _EXACT_CUE_REGEX.match(tag_text))
                and bool(_LOOSE_CUE_REGEX.match(tag_text))
                and bool(re.search(r"(?:^|[.!?…][\"'”’)\]]*)\s*$", self._tail_before + "".join(output_parts)))
            )
            if is_loose and (close_idx + 1 == len(text_buf)):
                self._held = text_buf[open_idx:]
                return self._record_shown("".join(output_parts))
            next_char = text_buf[close_idx + 1:close_idx + 2]
            if _EXACT_CUE_REGEX.match(tag_text) or (is_loose and next_char != "("):
                text_buf = text_buf[close_idx + 1:]
                current_out = "".join(output_parts)
                if (current_out and current_out[-1].isspace()) or ((not current_out) and self._ends_with_space):
                    text_buf = text_buf.lstrip(" \t")
                    if not text_buf:
                        self._trim_leading = True
            else:
                output_parts.append(text_buf[open_idx])
                text_buf = text_buf[open_idx + 1:]

    def _record_shown(self, text_out: str) -> str:
        if text_out:
            self._ends_with_space = text_out[-1].isspace()
            self._tail_before = (self._tail_before + text_out)[-12:]
        return text_out

    def flush(self) -> str:
        leftover = self._held
        self._held = ""
        return leftover


def speakable_text(text: str, keep_cues: bool = False, math_words_enabled: bool = True) -> str:
    """Prepares a sentence for speech synthesis (drops URLs, markdown, paths; normalizes math)."""
    cleaned = str(text or "")
    cleaned = re.sub(r"\[([^\]]*)\]\((?:https?:|mailto:)[^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"<?\b(?:https?://|www\.)\S+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"`([^`]*)`",
        lambda match_obj: "" if (len(match_obj.group(1)) > 24 or re.search(r"[/\\{}();=<>]", match_obj.group(1))) else match_obj.group(1),
        cleaned,
    )
    cleaned = re.sub(r"(\*\*|__|\*|~~)(?=\S)([^*~]*?\S)\1", r"\2", cleaned)
    cleaned = re.sub(r"(?:^|\s)(?:~|/)?(?:[\w.\-]+/){2,}[\w\-]*(?:\.\w+)*", " ", cleaned)
    if not keep_cues:
        cleaned = strip_voice_cues_for_display(cleaned)
    if math_words_enabled:
        cleaned = normalize_math_for_speech(cleaned)
    cleaned = re.sub(r"[\(\[\{]\s*[\)\]\}]", "", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    without_cues = CUE_REGEX.sub("", cleaned)
    return cleaned if re.search(r"[\w\u0590-\u06FF]", without_cues) else ""


def ends_with_question(text: str) -> bool:
    return bool(re.search(r"[?？]\s*[\"'”’)\]]*$", str(text or "").strip()))


_RTL_REGEX = re.compile(r"[\u0590-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def is_rtl_text(text: str) -> bool:
    """Returns True when text contains dominant Arabic or Hebrew script."""
    rtl_hits = len(_RTL_REGEX.findall(str(text or "")))
    if rtl_hits == 0:
        return False
    latin_hits = len(re.findall(r"[A-Za-z]", str(text or "")))
    return rtl_hits >= max(2, latin_hits // 2)


_COMMON_WAKE_WORDS = {
    "hello", "there", "okay", "yeah", "thanks", "sorry", "right", "what", "this", "that",
    "with", "have", "just", "like", "about", "would", "could", "really", "maybe", "guys",
    "everyone", "everybody", "buddy", "listen", "look", "siri", "alexa", "google",
}


def clean_wake_name(raw_name: str) -> Tuple[bool, str, str]:
    """
    Validates a custom wake name ('Hey <Name>').
    Returns `(is_valid, normalized_name, error_reason)`. Empty name resets to 'Jarvis'.
    """
    trimmed = re.sub(r"^(?:hey|hi)[\s,]+", "", str(raw_name or "").strip(), flags=re.IGNORECASE)
    if not trimmed:
        return True, "Jarvis", ""
    if re.search(r"\s", trimmed):
        return False, "", "Just one word please, like Jarvis or Roamie."
    letters_only = re.sub(r"[^\w]", "", trimmed)
    if (letters_only != trimmed) or (not trimmed.isalpha()):
        return False, "", "Letters only, please."
    if len(trimmed) < 4:
        return False, "", "At least 4 letters so it is not triggered by accident."
    if len(trimmed) > 20:
        return False, "", "20 letters maximum, please."
    if trimmed.lower() in _COMMON_WAKE_WORDS:
        return False, "", f'"{trimmed}" occurs in everyday speech or triggers another assistant.'
    return True, trimmed[0].upper() + trimmed[1:].lower(), ""


def compute_briefing_day(timestamp_ms: Optional[float] = None, day_start_hour: int = 4) -> str:
    """Computes the local YYYY-MM-DD briefing day with rollover at 4:00 AM."""
    effective_ms = (timestamp_ms if timestamp_ms is not None else time.time() * 1000.0) - (day_start_hour * 3600_000.0)
    local_dt = datetime.datetime.fromtimestamp(effective_ms / 1000.0)
    return local_dt.strftime("%Y-%m-%d")


def build_morning_briefing_note(
    on: bool,
    last_day: str,
    by_voice: bool,
    todoist: bool = False,
    can_read: bool = True,
    user_name: str = "you",
    now_ms: Optional[float] = None,
) -> Optional[Tuple[str, str]]:
    """Returns `(briefing_instruction_note, current_day)` on the first voice turn of the day."""
    if (not on) or (not by_voice):
        return None
    current_day = compute_briefing_day(now_ms)
    if last_day == current_day:
        return None
    todo_clause = ""
    if todoist and can_read:
        todo_clause = (
            ' Also inspect Todoist via read_window(target="Todoist") if open, or open '
            "https://app.todoist.com/app/today, summarizing only today's and overdue tasks."
        )
    note = (
        f"System note: this is the first voice conversation with {user_name} today ({current_day}). "
        f"Start with a brief in-character morning greeting and summarize any overdue or due-today "
        f"commitments and reminders from memory soonest first.{todo_clause} Keep it under 4 short sentences, "
        f"then answer their request directly."
    )
    return note, current_day


_WEEKDAY_MAP = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}


def parse_repeat_schedule(raw_schedule: str) -> Optional[Dict[str, Any]]:
    """Parses human repeat strings such as 'every weekday at 9', 'every 2 hours', 'every day at 08:30'."""
    text = str(raw_schedule or "").strip().lower()
    if not text:
        return None
    hour_match = re.search(r"\bevery\s+(\d+)\s*(?:hr|hour)s?\b", text)
    if hour_match:
        return {"kind": "interval_seconds", "seconds": max(60, int(hour_match.group(1)) * 3600), "label": raw_schedule}
    min_match = re.search(r"\bevery\s+(\d+)\s*(?:m|min|minute)s?\b", text)
    if min_match:
        return {"kind": "interval_seconds", "seconds": max(60, int(min_match.group(1)) * 60), "label": raw_schedule}

    time_match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    hour_val = 9
    minute_val = 0
    if time_match:
        hour_val = int(time_match.group(1))
        minute_val = int(time_match.group(2) or "0")
        meridiem = (time_match.group(3) or "").lower()
        if meridiem == "pm" and hour_val < 12:
            hour_val += 12
        elif meridiem == "am" and hour_val == 12:
            hour_val = 0

    if "weekday" in text:
        return {"kind": "weekdays", "hour": hour_val % 24, "minute": minute_val % 60, "label": raw_schedule}
    if "weekend" in text:
        return {"kind": "weekends", "hour": hour_val % 24, "minute": minute_val % 60, "label": raw_schedule}
    for day_word, day_num in _WEEKDAY_MAP.items():
        if re.search(rf"\b{day_word}\b", text):
            return {"kind": "weekly_day", "weekday": day_num, "hour": hour_val % 24, "minute": minute_val % 60, "label": raw_schedule}
    if ("week" in text) and ("weekday" not in text) and ("weekend" not in text):
        return {"kind": "interval_seconds", "seconds": 7 * 86400, "label": raw_schedule}
    if "month" in text:
        return {"kind": "interval_seconds", "seconds": 30 * 86400, "label": raw_schedule}
    if ("day" in text) or ("daily" in text):
        return {"kind": "daily", "hour": hour_val % 24, "minute": minute_val % 60, "label": raw_schedule}
    return None


def compute_next_repeat_timestamp(repeat_spec: Dict[str, Any], from_ts: Optional[float] = None) -> float:
    """Computes the next Unix timestamp for a parsed repeat schedule after `from_ts`."""
    base_ts = float(from_ts if from_ts is not None else time.time())
    kind = str(repeat_spec.get("kind", "daily"))
    if kind == "interval_seconds":
        return base_ts + float(repeat_spec.get("seconds", 3600))
    hour_val = int(repeat_spec.get("hour", 9)) % 24
    minute_val = int(repeat_spec.get("minute", 0)) % 60
    dt_now = datetime.datetime.fromtimestamp(base_ts)
    candidate = dt_now.replace(hour=hour_val, minute=minute_val, second=0, microsecond=0)
    if candidate.timestamp() <= base_ts + 5.0:
        candidate += datetime.timedelta(days=1)
    if kind == "weekdays":
        while candidate.weekday() >= 5:
            candidate += datetime.timedelta(days=1)
    elif kind == "weekends":
        while candidate.weekday() < 5:
            candidate += datetime.timedelta(days=1)
    elif kind == "weekly_day":
        target_wd = int(repeat_spec.get("weekday", 0)) % 7
        while candidate.weekday() != target_wd:
            candidate += datetime.timedelta(days=1)
    return candidate.timestamp()


def parse_voice_snooze_request(user_text: str) -> Optional[int]:
    """Parses voice snooze phrases like 'snooze 10 minutes' or 'remind me in 1 hour' -> minutes."""
    lowered = str(user_text or "").strip().lower()
    if not lowered:
        return None
    if re.search(r"\bsnooze\b", lowered):
        num_match = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)?\b", lowered)
        if not num_match:
            return 10
        amount = max(1, int(num_match.group(1)))
        unit = (num_match.group(2) or "m").lower()
        return amount * 60 if unit.startswith("h") else amount
    return None


# ---------------------------------------------------------------------------
# 3. Windows DPAPI Secret Store
# ---------------------------------------------------------------------------

class _CRYPT_DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _dpapi_protect(plain_bytes: bytes) -> Optional[bytes]:
    try:
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        in_buffer = ctypes.create_string_buffer(plain_bytes, len(plain_bytes))
        in_blob = _CRYPT_DATA_BLOB(len(plain_bytes), ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_byte)))
        out_blob = _CRYPT_DATA_BLOB()
        ok = crypt32.CryptProtectData(ctypes.byref(in_blob), "JarvisSecret", None, None, None, 0x01, ctypes.byref(out_blob))
        if not ok:
            return None
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(out_blob.pbData)
    except Exception:
        return None


def _dpapi_unprotect(cipher_bytes: bytes) -> Optional[bytes]:
    try:
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        in_buffer = ctypes.create_string_buffer(cipher_bytes, len(cipher_bytes))
        in_blob = _CRYPT_DATA_BLOB(len(cipher_bytes), ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_byte)))
        out_blob = _CRYPT_DATA_BLOB()
        ok = crypt32.CryptUnprotectData(ctypes.byref(in_blob), None, None, None, None, 0x01, ctypes.byref(out_blob))
        if not ok:
            return None
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(out_blob.pbData)
    except Exception:
        return None


class WindowsDPAPISecretStore:
    """Stores encrypted per-user secrets using Windows DPAPI in `windows/memory/dpapi_secrets.json`."""

    def __init__(self, store_file_path: str = DEFAULT_DPAPI_FILE_PATH) -> None:
        self._file_path = store_file_path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)

    def _read_all(self) -> Dict[str, str]:
        if not os.path.exists(self._file_path):
            return {}
        try:
            with open(self._file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_all(self, data: Dict[str, str]) -> None:
        tmp_path = self._file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp_path, self._file_path)

    def set_secret(self, key_name: str, plain_value: str) -> None:
        with self._lock:
            records = self._read_all()
            raw_bytes = str(plain_value or "").encode("utf-8")
            protected = _dpapi_protect(raw_bytes)
            if protected is not None:
                records[key_name] = "dpapi:" + base64.b64encode(protected).decode("ascii")
            else:
                records[key_name] = "b64:" + base64.b64encode(raw_bytes).decode("ascii")
            self._write_all(records)

    def get_secret(self, key_name: str) -> Optional[str]:
        with self._lock:
            records = self._read_all()
            encoded = records.get(key_name)
            if not isinstance(encoded, str) or not encoded:
                return None
            if encoded.startswith("dpapi:"):
                raw_cipher = base64.b64decode(encoded[6:])
                plain_bytes = _dpapi_unprotect(raw_cipher)
                return plain_bytes.decode("utf-8", errors="replace") if plain_bytes is not None else None
            if encoded.startswith("b64:"):
                return base64.b64decode(encoded[4:]).decode("utf-8", errors="replace")
            return encoded

    def delete_secret(self, key_name: str) -> bool:
        with self._lock:
            records = self._read_all()
            if key_name not in records:
                return False
            del records[key_name]
            self._write_all(records)
            return True

    def list_keys(self) -> List[str]:
        with self._lock:
            return sorted(self._read_all().keys())


# ---------------------------------------------------------------------------
# 4. Roamies Safety Engine
# ---------------------------------------------------------------------------

class JarvisSafetyEngine:
    """Enforces SAFE / ASK / BLOCK safety rules across file, shell, UI click, URL, and MCP actions."""

    SAFE = "safe"
    ASK = "ask"
    BLOCK = "block"

    HANDS_OFF_PROCESSES = (
        "1password", "bitwarden", "keepass", "keepassxc", "lastpass", "dashlane",
        "nordpass", "protonpass", "enpass", "roboform", "credentialui", "consent",
        "secedit", "mmc", "regedit",
    )

    SECRET_PATH_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
        re.compile(pat, re.IGNORECASE)
        for pat in (
            r"(^|/)\.ssh(/|$)",
            r"(^|/)\.gnupg(/|$)",
            r"(^|/)\.aws(/|$)",
            r"(^|/)\.azure(/|$)",
            r"(^|/)\.kube/config$",
            r"(^|/)\.docker/config\.json$",
            r"(^|/)\.netrc$",
            r"(^|/)\.npmrc$",
            r"(^|/)\.pypirc$",
            r"(^|/)\.git-credentials$",
            r"(^|/)\.config/gh(/|$)",
            r"(^|/)\.config/gcloud(/|$)",
            r"(^|/)\.gg/auth\.json$",
            r"(^|/)dpapi_secrets\.json$",
            r"(^|/)Roamies/Data/secrets\.json$",
            r"(^|/)\.tauri(/|$)",
            r"/(Cookies|Login Data|Web Data)(-journal)?$",
            r"(^|/)\.env(\.[\w\-]+)?$",
            r"\.(pem|p12|pfx|key|keystore|jks)$",
            r"(^|/)id_(rsa|ed25519|ecdsa|dsa)(\.pub)?$",
            r"(^|/)\.(bash|zsh|fish)_history$",
            r"/AppData/(Roaming/Microsoft/(Credentials|Protect|Crypto)|Local/Microsoft/Credentials)(/|$)",
            r"/(User Data|Profiles)/.*/(Cookies|Login Data|logins\.json|key4\.db)$",
        )
    )

    PRIVATE_PATH_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
        re.compile(pat, re.IGNORECASE)
        for pat in (
            r"(^|/)AppData/Local/Packages(/|$)",
            r"(^|/)Library/(Containers|Group Containers|Application Support/com\.apple\.TCC)(/|$)",
        )
    )

    NO_WRITE_PATH_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
        re.compile(pat, re.IGNORECASE)
        for pat in (
            r"(^|/)\.(zshrc|zprofile|zshenv|zlogin|bashrc|bash_profile|profile|login|inputrc)$",
            r"(^|/)\.gitconfig$",
            r"(^|/)\.ssh(/|$)",
            r"(^|/)\.gg(/|$)",
            r"/Start Menu/Programs/Startup(/|$)",
            r"(^|/)Documents/(Windows)?PowerShell/.*profile\.ps1$",
            r"^[a-z]:/(Windows|Program Files( \(x86\))?|ProgramData)(/|$)",
        )
    )

    RISKY_CLICK_PATTERN = re.compile(
        r"(?:^|[^\w])(delete|remove|erase|destroy|wipe|reset|deactivate|close (?:my |your )?account|"
        r"cancel (?:my |your |the )?(?:subscription|plan|membership|account|order)|unsubscribe|pay|"
        r"purchase|buy|place (?:the |your )?order|checkout|check out|confirm (?:payment|order|purchase|transfer)|"
        r"send|transfer|withdraw|uninstall|empty trash|discard|sign out|log out|logout|format|revoke|publish|"
        r"eliminar|borrar|pagar|comprar|enviar|supprimer|effacer|payer|acheter|envoyer|"
        r"loeschen|löschen|entfernen|bezahlen|kaufen|senden|excluir|apagar|elimina|cancella|"
        r"verwijderen|betalen|hapus|bayar)(?![\w])"
        r"|删除|刪除|移除|清空|支付|付款|购买|購買|发送|發送|退出登录|登出|削除|支払|送信|삭제|결제|구매",
        re.IGNORECASE,
    )

    DANGEROUS_SHELL_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
        re.compile(pat, re.IGNORECASE)
        for pat in (
            r"\brm\s+-[^\n]*rf\s+[/\\~]",
            r"\bdel\s+/[sqf]+\s+[a-z]:\\",
            r"\brd\s+/s\s+/q\s+[a-z]:\\",
            r"\bformat\s+[a-z]:",
            r"\bdiskpart\b",
            r"\bbcdedit\b",
            r"\breg\s+(delete|add)\b",
            r"\bshutdown\s+/[sr]",
            r"\bRemove-Item\b[^\n]*-Recurse[^\n]*([a-z]:\\Windows|C:\\Users\\[^\\]+$)",
        )
    )

    ASK_SHELL_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
        re.compile(pat, re.IGNORECASE)
        for pat in (
            r"\b(git\s+push|git\s+reset\s+--hard|npm\s+publish|pip\s+uninstall)\b",
            r"\b(del|erase|rmdir|Remove-Item|Move-Item|Rename-Item)\b",
            r"\b(Stop-Process|taskkill|sc\s+stop|sc\s+delete|net\s+stop)\b",
            r"\bcurl\b[^\n]*\|\s*(sh|bash|pwsh|powershell)",
        )
    )

    MCP_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,39}$")

    @staticmethod
    def _norm_slash(raw_path: str, home_dir: Optional[str] = None) -> List[str]:
        home = (home_dir or os.path.expanduser("~")).replace("\\", "/").rstrip("/")
        cleaned = str(raw_path or "").strip()
        if cleaned == "~":
            expanded = home
        elif cleaned.startswith(("~/", "~\\")):
            expanded = home + "/" + cleaned[2:].replace("\\", "/")
        else:
            expanded = cleaned.replace("\\", "/")
            if not (expanded.startswith("/") or re.match(r"^[a-zA-Z]:/", expanded)):
                expanded = home + "/" + expanded
        abs_posix = os.path.normpath(expanded).replace("\\", "/")
        try:
            real_posix = os.path.realpath(abs_posix).replace("\\", "/")
        except OSError:
            real_posix = abs_posix
        return [abs_posix] if real_posix.lower() == abs_posix.lower() else [abs_posix, real_posix]

    @classmethod
    def check_read(cls, file_path: str, home_dir: Optional[str] = None, recursive: bool = False) -> Tuple[str, str]:
        candidates = cls._norm_slash(file_path, home_dir)
        home = (home_dir or os.path.expanduser("~")).replace("\\", "/").rstrip("/")
        for cand in candidates:
            rel_path = cand[len(home) + 1:] if cand.lower().startswith(home.lower() + "/") else cand
            for pat in cls.SECRET_PATH_PATTERNS:
                if pat.search(rel_path) or pat.search(cand):
                    return cls.BLOCK, f"{file_path} holds secrets or credentials and cannot be read."
            for pat in cls.PRIVATE_PATH_PATTERNS:
                if pat.search(rel_path) or pat.search(cand):
                    return cls.BLOCK, f"{file_path} is another app's protected container."
        if recursive:
            norm_root = candidates[0].rstrip("/")
            if norm_root.lower() in (home.lower(), "/", "c:", "d:", (home + "/AppData").lower()):
                return cls.BLOCK, f"Recursive search across {file_path} reaches protected folders; pick a subfolder."
        return cls.SAFE, ""

    @classmethod
    def check_write(
        cls,
        file_path: str,
        home_dir: Optional[str] = None,
        exists: bool = False,
        own_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        candidates = cls._norm_slash(file_path, home_dir)
        home = (home_dir or os.path.expanduser("~")).replace("\\", "/").rstrip("/")
        for cand in candidates:
            rel_path = cand[len(home) + 1:] if cand.lower().startswith(home.lower() + "/") else cand
            for pat in (*cls.SECRET_PATH_PATTERNS, *cls.NO_WRITE_PATH_PATTERNS):
                if pat.search(rel_path) or pat.search(cand):
                    return cls.BLOCK, f"{file_path} is a protected key, startup script, or system file."
        if own_dir:
            own_norm = os.path.realpath(own_dir).replace("\\", "/").rstrip("/") + "/"
            if all(cand.lower().startswith(own_norm.lower()) for cand in candidates):
                return cls.SAFE, ""
        if exists:
            return cls.ASK, f"Overwrite existing file {file_path}?"
        return cls.SAFE, ""

    @classmethod
    def check_click(cls, target_label: str, target_process: str = "") -> Tuple[str, str]:
        proc_lower = str(target_process or "").strip().lower()
        if any(guard in proc_lower for guard in cls.HANDS_OFF_PROCESSES):
            return cls.BLOCK, f"{target_process} is in the hands-off security list."
        label_str = str(target_label or "").strip()
        if cls.RISKY_CLICK_PATTERN.search(label_str):
            return cls.ASK, f'Allow clicking "{label_str[:60]}"?'
        return cls.SAFE, ""

    @classmethod
    def check_shell_command(cls, command_text: str) -> Tuple[str, str]:
        cmd = str(command_text or "").strip()
        if not cmd:
            return cls.BLOCK, "Empty command."
        for pat in cls.DANGEROUS_SHELL_PATTERNS:
            if pat.search(cmd):
                return cls.BLOCK, "Destructive system command blocked by safety policy."
        for pat in cls.ASK_SHELL_PATTERNS:
            if pat.search(cmd):
                return cls.ASK, f"Allow running command: {cmd[:80]}?"
        return cls.SAFE, ""

    @classmethod
    def is_private_address(cls, host_or_url: str) -> bool:
        raw = str(host_or_url or "").strip()
        parsed = urllib.parse.urlparse(raw if "://" in raw else f"http://{raw}")
        host = (parsed.hostname or "").strip().lower()
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return True
        if host.startswith(("10.", "192.168.", "169.254.")):
            return True
        oct_match = re.match(r"^172\.(\d+)\.", host)
        if oct_match and 16 <= int(oct_match.group(1)) <= 31:
            return True
        return False

    @classmethod
    def check_url(cls, url_text: str) -> Tuple[str, str]:
        raw = str(url_text or "").strip()
        if not re.match(r"^https?://", raw, re.IGNORECASE):
            return cls.BLOCK, "Only http:// and https:// URLs are allowed."
        if cls.is_private_address(raw):
            return cls.ASK, f"Open local network URL {raw[:70]}?"
        return cls.SAFE, ""

    @classmethod
    def check_mcp_server(cls, name: str, command: str = "", args: Optional[Sequence[str]] = None, url: str = "") -> Tuple[str, str]:
        if not cls.MCP_NAME_PATTERN.match(str(name or "").strip()):
            return cls.BLOCK, "Connection name must be 1-40 letters/digits/hyphens/underscores."
        if url:
            if not re.match(r"^https?://", str(url).strip(), re.IGNORECASE):
                return cls.BLOCK, "MCP server URL must start with http:// or https://."
            return cls.ASK, f"Connect external MCP URL {url[:64]}?"
        if not str(command or "").strip():
            return cls.BLOCK, "Provide either a server command or URL."
        return cls.ASK, f"Run local MCP server {name} ({command})?"

    @staticmethod
    def redact_args(tool_name: str, args_dict: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = dict(args_dict or {})
        for secret_field in ("content", "text", "args", "steps", "brief", "secretEnv", "secretHeaders"):
            if secret_field in sanitized and sanitized[secret_field] is not None:
                val_len = len(str(sanitized[secret_field]))
                sanitized[secret_field] = f"[{val_len} chars]"
        return sanitized


# ---------------------------------------------------------------------------
# 5. MCP Client Manager (stdio + HTTP JSON-RPC 2.0 + DPAPI secrets)
# ---------------------------------------------------------------------------

class MCPConnectionManager:
    """Manages external MCP server definitions and executes tools over stdio or HTTP JSON-RPC 2.0."""

    SECRET_MARK = "@jarvis-secret"

    def __init__(
        self,
        config_file_path: str = DEFAULT_MCP_FILE_PATH,
        dpapi_store: Optional[WindowsDPAPISecretStore] = None,
    ) -> None:
        self._file_path = config_file_path
        self._secrets = dpapi_store or WindowsDPAPISecretStore()
        self._lock = threading.RLock()
        self._live_cache: Dict[str, Dict[str, Any]] = {}
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)

    @staticmethod
    def _secret_key(server_name: str, field_key: str) -> str:
        digest = hashlib.sha256(f"{server_name}\0{field_key}".encode("utf-8")).hexdigest()[:16]
        return f"mcp-{digest}"

    def _read_config(self) -> Dict[str, Any]:
        if not os.path.exists(self._file_path):
            return {"mcpServers": {}}
        try:
            with open(self._file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and isinstance(payload.get("mcpServers"), dict):
                return payload
        except Exception:
            pass
        return {"mcpServers": {}}

    def _write_config(self, config_obj: Dict[str, Any]) -> None:
        tmp_path = self._file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(config_obj, handle, indent=2)
        os.replace(tmp_path, self._file_path)

    def _resolve_server_entry(self, server_name: str, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        resolved = json.loads(json.dumps(entry))
        missing: List[str] = []
        for section_name in ("env", "headers"):
            section = resolved.get(section_name)
            if not isinstance(section, dict):
                continue
            for key_name, raw_val in list(section.items()):
                if raw_val == self.SECRET_MARK:
                    secret_val = self._secrets.get_secret(self._secret_key(server_name, f"{section_name}:{key_name}"))
                    if secret_val:
                        section[key_name] = secret_val
                    else:
                        missing.append(key_name)
        return resolved, missing

    def add_connection(
        self,
        name: str,
        command: str = "",
        args: Optional[List[str]] = None,
        url: str = "",
        env: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        secret_env: Optional[Dict[str, str]] = None,
        secret_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        level, reason = JarvisSafetyEngine.check_mcp_server(name, command=command, args=args, url=url)
        if level == JarvisSafetyEngine.BLOCK:
            return False, reason
        with self._lock:
            entry: Dict[str, Any] = {"enabled": True}
            if url:
                entry["url"] = str(url).strip()
            else:
                entry["command"] = str(command).strip()
                entry["args"] = [str(arg_item) for arg_item in (args or [])]
            env_map = dict(env or {})
            for sec_key, sec_val in (secret_env or {}).items():
                self._secrets.set_secret(self._secret_key(name, f"env:{sec_key}"), sec_val)
                env_map[sec_key] = self.SECRET_MARK
            if env_map:
                entry["env"] = env_map
            hdr_map = dict(headers or {})
            for sec_key, sec_val in (secret_headers or {}).items():
                self._secrets.set_secret(self._secret_key(name, f"headers:{sec_key}"), sec_val)
                hdr_map[sec_key] = self.SECRET_MARK
            if hdr_map:
                entry["headers"] = hdr_map
            cfg = self._read_config()
            cfg["mcpServers"][name] = entry
            self._write_config(cfg)
            self._live_cache[name] = {"ok": True, "tools": self._live_cache.get(name, {}).get("tools", []), "error": None}
        return True, f"Saved MCP connection '{name}'."

    def remove_connection(self, name: str) -> Tuple[bool, str]:
        with self._lock:
            cfg = self._read_config()
            entry = cfg["mcpServers"].pop(name, None)
            if entry is None:
                return False, f"No MCP connection named '{name}'."
            for section_name in ("env", "headers"):
                for key_name, val_item in (entry.get(section_name) or {}).items():
                    if val_item == self.SECRET_MARK:
                        self._secrets.delete_secret(self._secret_key(name, f"{section_name}:{key_name}"))
            self._live_cache.pop(name, None)
            self._write_config(cfg)
        return True, f"Removed MCP connection '{name}'."

    def set_enabled(self, name: str, enabled: bool) -> bool:
        with self._lock:
            cfg = self._read_config()
            entry = cfg["mcpServers"].get(name)
            if not isinstance(entry, dict):
                return False
            entry["enabled"] = bool(enabled)
            self._write_config(cfg)
            return True

    def list_status(self) -> List[Dict[str, Any]]:
        with self._lock:
            cfg = self._read_config()
            rows: List[Dict[str, Any]] = []
            for server_name, entry in cfg["mcpServers"].items():
                live = self._live_cache.get(server_name, {})
                enabled = entry.get("enabled", True) is not False
                where_str = entry.get("url") or " ".join([str(entry.get("command", "")), *[str(arg_v) for arg_v in entry.get("args", [])]]).strip()
                rows.append({
                    "name": server_name,
                    "where": where_str,
                    "remote": bool(entry.get("url")),
                    "enabled": enabled,
                    "state": "off" if not enabled else ("connected" if live.get("ok") else "ready"),
                    "tool_count": len(live.get("tools", [])),
                    "error": live.get("error"),
                })
            return rows

    def live_state_text(self) -> str:
        rows = [row for row in self.list_status() if row.get("enabled")]
        if not rows:
            return ""
        return "\n".join(
            f"- {row['name']}: {row['state']} ({row['tool_count']} tools cached, {row['where']})"
            for row in rows
        )

    def _rpc_stdio(self, entry: Dict[str, Any], method_name: str, params: Optional[Dict[str, Any]] = None, timeout: float = 12.0) -> Tuple[bool, Any]:
        cmd_list = [str(entry["command"]), *[str(arg_v) for arg_v in (entry.get("args") or [])]]
        proc_env = dict(os.environ)
        for env_key, env_val in (entry.get("env") or {}).items():
            proc_env[str(env_key)] = str(env_val)
        try:
            proc = subprocess.Popen(
                cmd_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=proc_env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            return False, f"Failed to start MCP command: {exc}"

        try:
            messages = [
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "Jarvis", "version": "1.2.0"},
                    },
                }),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": method_name, "params": params or {}}),
            ]
            assert proc.stdin is not None
            proc.stdin.write("\n".join(messages) + "\n")
            proc.stdin.flush()
            proc.stdin.close()
            stdout_data, stderr_data = proc.communicate(timeout=timeout)
            for raw_line in reversed(stdout_data.splitlines()):
                line = raw_line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    resp_obj = json.loads(line)
                except Exception:
                    continue
                if resp_obj.get("id") == 2:
                    if "error" in resp_obj:
                        return False, str(resp_obj["error"])
                    return True, resp_obj.get("result")
            return False, (stderr_data.strip() or "No JSON-RPC response from MCP server.")
        except Exception as exc:
            proc.kill()
            return False, f"MCP stdio timeout/error: {exc}"

    def _rpc_http(self, entry: Dict[str, Any], method_name: str, params: Optional[Dict[str, Any]] = None, timeout: float = 12.0) -> Tuple[bool, Any]:
        if requests is None:
            return False, "requests library is unavailable."
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        for hdr_key, hdr_val in (entry.get("headers") or {}).items():
            headers[str(hdr_key)] = str(hdr_val)
        payload = {"jsonrpc": "2.0", "id": 1, "method": method_name, "params": params or {}}
        try:
            resp = requests.post(str(entry["url"]), json=payload, headers=headers, timeout=timeout)
            if not resp.ok:
                return False, f"HTTP {resp.status_code}: {resp.text[:160]}"
            body = resp.json()
            if "error" in body:
                return False, str(body["error"])
            return True, body.get("result")
        except Exception as exc:
            return False, f"MCP HTTP error: {exc}"

    def connection_tools(self, server_name: str, query: str = "") -> Tuple[bool, List[Dict[str, Any]], str]:
        with self._lock:
            entry = self._read_config()["mcpServers"].get(server_name)
            if not entry:
                return False, [], f"No MCP connection named '{server_name}'."
            resolved, missing = self._resolve_server_entry(server_name, entry)
            if missing:
                return False, [], f"Missing secret values for: {', '.join(missing)}."
        ok, res = self._rpc_http(resolved, "tools/list", {}) if resolved.get("url") else self._rpc_stdio(resolved, "tools/list", {})
        if not ok:
            with self._lock:
                self._live_cache[server_name] = {"ok": False, "tools": [], "error": str(res)}
            return False, [], str(res)
        tools_raw = (res or {}).get("tools", []) if isinstance(res, dict) else []
        terms = [tok.lower() for tok in str(query or "").split() if tok.strip()]
        filtered = [
            tool_item for tool_item in tools_raw
            if not terms or any(term in f"{tool_item.get('name', '')} {tool_item.get('description', '')}".lower() for term in terms)
        ]
        with self._lock:
            self._live_cache[server_name] = {"ok": True, "tools": tools_raw, "error": None}
        return True, filtered, f"{len(filtered)} tools available on {server_name}."

    def call_tool(self, server_name: str, tool_name: str, arguments_dict: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        with self._lock:
            entry = self._read_config()["mcpServers"].get(server_name)
            if not entry:
                return False, f"No MCP connection named '{server_name}'."
            resolved, missing = self._resolve_server_entry(server_name, entry)
            if missing:
                return False, f"Missing secrets for {server_name}: {', '.join(missing)}."
        params = {"name": tool_name, "arguments": arguments_dict or {}}
        ok, res = self._rpc_http(resolved, "tools/call", params) if resolved.get("url") else self._rpc_stdio(resolved, "tools/call", params)
        if not ok:
            return False, str(res)
        if isinstance(res, dict) and isinstance(res.get("content"), list):
            parts = [str(blk.get("text", blk)) for blk in res["content"]]
            return True, "\n".join(parts)
        return True, json.dumps(res, ensure_ascii=False)[:1800]

    @staticmethod
    def discover_importable_configs() -> Dict[str, List[Dict[str, Any]]]:
        appdata = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
        sources = {
            "claude-desktop": os.path.join(appdata, "Claude", "claude_desktop_config.json"),
            "gg-coder": os.path.join(os.path.expanduser("~"), ".gg", "mcp.json"),
        }
        discovered: Dict[str, List[Dict[str, Any]]] = {}
        for source_name, cfg_path in sources.items():
            if not os.path.exists(cfg_path):
                continue
            try:
                with open(cfg_path, "r", encoding="utf-8") as handle:
                    parsed = json.load(handle)
                servers = parsed.get("mcpServers", {})
                if not isinstance(servers, dict):
                    continue
                entries: List[Dict[str, Any]] = []
                for srv_name, srv_cfg in servers.items():
                    if JarvisSafetyEngine.MCP_NAME_PATTERN.match(str(srv_name)) and isinstance(srv_cfg, dict):
                        entries.append({
                            "name": srv_name,
                            "where": srv_cfg.get("url") or f"{srv_cfg.get('command', '')} {' '.join(srv_cfg.get('args', []))}".strip(),
                            "entry": srv_cfg,
                        })
                if entries:
                    discovered[source_name] = entries
            except Exception:
                continue
        return discovered

    def import_from_source(self, source_name: str) -> int:
        discovered = self.discover_importable_configs().get(source_name, [])
        count = 0
        for item in discovered:
            entry = item["entry"]
            ok, _ = self.add_connection(
                name=item["name"],
                command=str(entry.get("command", "")),
                args=list(entry.get("args", [])),
                url=str(entry.get("url", "")),
                env=dict(entry.get("env") or {}),
                headers=dict(entry.get("headers") or {}),
            )
            if ok:
                count += 1
        return count


# ---------------------------------------------------------------------------
# 6. Subagent Crew Manager
# ---------------------------------------------------------------------------

SUBAGENT_SPECS: Tuple[Dict[str, str], ...] = (
    {
        "id": "scout",
        "label": "Scout",
        "type": "researcher",
        "hint": "Web & source research",
        "role": "You are the Scout: research thoroughly, cross-check sources, state what was confirmed and what could not be verified.",
    },
    {
        "id": "builder",
        "label": "Builder",
        "type": "maker",
        "hint": "Code, pages & scripts",
        "role": "You are the Builder: produce working code or artifacts cleanly, fix errors, and hand back a working deliverable.",
    },
    {
        "id": "writer",
        "label": "Writer",
        "type": "maker",
        "hint": "Documents, copy & summaries",
        "role": "You are the Writer: draft crisp finished text matching requested tone and length while keeping every fact accurate.",
    },
    {
        "id": "reviewer",
        "label": "Reviewer",
        "type": "researcher",
        "hint": "Audits work for bugs & mistakes",
        "role": "You are the Reviewer: inspect code or drafts carefully, list real issues first with exact fixes.",
    },
    {
        "id": "tidier",
        "label": "Tidier",
        "type": "organizer",
        "hint": "Sorts, renames & organizes files",
        "role": "You are the Tidier: propose clear structure and organize carefully without destructive deletions.",
    },
)


class SubagentCrewManager:
    """Runs background specialist subagents and saves deliverables for the artifact toast UI."""

    MAX_CONCURRENT_AGENTS = 3

    _ROLE_PATTERNS: Dict[str, re.Pattern[str]] = {
        "reviewer": re.compile(r"\b(review|proofread|audit|double.?check|check (?:my|the|this) (?:code|work|draft|doc))\b", re.IGNORECASE),
        "tidier": re.compile(r"\b(tidy|organi[sz]e|rename|clean up|sort)\b[^.?!]{0,40}\b(files?|folders?|downloads|desktop)\b", re.IGNORECASE),
        "writer": re.compile(r"\b(write|draft|compose)\b[^.?!]{0,40}\b(essay|report|article|email|letter|post|slides?|doc|proposal|summary|blog)\b", re.IGNORECASE),
    }

    def __init__(self, agents_root_dir: str = DEFAULT_AGENTS_DIR) -> None:
        self._agents_dir = agents_root_dir
        self._lock = threading.RLock()
        self._runs: Dict[str, Dict[str, Any]] = {}
        os.makedirs(self._agents_dir, exist_ok=True)

    @staticmethod
    def get_specialist(spec_id_or_name: str) -> Optional[Dict[str, str]]:
        needle = str(spec_id_or_name or "").strip().lower()
        if not needle:
            return None
        for spec in SUBAGENT_SPECS:
            if needle == spec["id"] or needle == spec["label"].lower() or re.search(rf"\b{spec['id']}\b", needle):
                return spec
        return None

    @classmethod
    def specialist_for_request(cls, prompt_text: str, delegate_mode: str = "ask") -> Optional[Dict[str, str]]:
        cleaned = str(prompt_text or "").strip()
        if delegate_mode != "auto" or not cleaned or cleaned.startswith("("):
            return None
        for role_id in ("reviewer", "tidier", "writer"):
            if cls._ROLE_PATTERNS[role_id].search(cleaned):
                return cls.get_specialist(role_id)
        kinds = HybridModelRouter.classify_job_kinds(cleaned)
        if "computer" in kinds or "draw" in kinds:
            return None
        if "web" in kinds:
            return cls.get_specialist("scout")
        if "code" in kinds:
            return cls.get_specialist("builder")
        return None

    @staticmethod
    def crew_prompt_text(subagent_models: Optional[Dict[str, str]] = None, delegate_mode: str = "ask") -> str:
        when_rule = (
            "Proactive delegation is enabled: delegate long, hands-off research/build/writing/review work automatically via send_agent."
            if delegate_mode == "auto"
            else "Delegate only when asked for a background agent."
        )
        lines = [
            f"Subagents (max {SubagentCrewManager.MAX_CONCURRENT_AGENTS} concurrent via send_agent(subagent=\"...\", brief=\"...\")): {when_rule}"
        ]
        model_map = subagent_models or {}
        for spec in SUBAGENT_SPECS:
            model_hint = model_map.get(spec["id"], "default")
            lines.append(f"- {spec['id']} ({spec['label']}): {spec['hint']} [model: {model_hint}]")
        return "\n".join(lines)

    def send_agent(
        self,
        brief: str,
        subagent: str = "",
        runner_fn: Optional[Callable[[str, str], str]] = None,
        on_done_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[bool, Dict[str, Any], str]:
        with self._lock:
            active_count = sum(1 for item in self._runs.values() if item.get("status") == "running")
            if active_count >= self.MAX_CONCURRENT_AGENTS:
                return False, {}, f"Maximum {self.MAX_CONCURRENT_AGENTS} background agents already running."
            spec = self.get_specialist(subagent) or SUBAGENT_SPECS[0]
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            agent_id = f"{spec['id']}_{stamp}_{len(self._runs) + 1}"
            run_dir = os.path.join(self._agents_dir, agent_id)
            os.makedirs(run_dir, exist_ok=True)
            full_prompt = f"{spec['role']}\n\nTask:\n{brief.strip()}"
            record: Dict[str, Any] = {
                "id": agent_id,
                "subagent": spec["id"],
                "label": spec["label"],
                "brief": brief.strip(),
                "status": "running",
                "folder": run_dir,
                "files": [],
                "report": "",
                "started_at": time.time(),
                "finished_at": None,
                "messages": [],
                "stop_event": threading.Event(),
            }
            self._runs[agent_id] = record

        def _worker() -> None:
            try:
                if runner_fn is not None:
                    result_text = str(runner_fn(spec["id"], full_prompt) or "").strip()
                else:
                    result_text = f"[{spec['label']}] Prepared structured deliverable for: {brief.strip()}"
                if record["stop_event"].is_set():
                    with self._lock:
                        record["status"] = "stopped"
                        record["finished_at"] = time.time()
                    return
                report_path = os.path.join(run_dir, f"{spec['id']}_report.md")
                with open(report_path, "w", encoding="utf-8") as handle:
                    handle.write(f"# {spec['label']} Deliverable\n\n**Brief:** {brief.strip()}\n\n{result_text}\n")
                with self._lock:
                    record["report"] = result_text
                    record["files"] = [report_path]
                    record["status"] = "done"
                    record["finished_at"] = time.time()
                if on_done_callback is not None:
                    on_done_callback(dict(record))
            except Exception as exc:
                with self._lock:
                    record["report"] = f"Agent error: {exc}"
                    record["status"] = "error"
                    record["finished_at"] = time.time()

        threading.Thread(target=_worker, name=f"JarvisSubagent-{agent_id}", daemon=True).start()
        return True, {"id": agent_id, "subagent": spec["id"], "folder": run_dir}, f"Started {spec['label']} subagent ({agent_id})."

    def tell_agent(self, agent_id: str, message: str) -> Tuple[bool, str]:
        with self._lock:
            rec = self._runs.get(agent_id)
            if not rec:
                return False, f"Unknown agent '{agent_id}'."
            rec["messages"].append({"ts": time.time(), "text": str(message or "")})
            return True, f"Sent follow-up note to {agent_id}."

    def stop_agent(self, agent_id: str) -> Tuple[bool, str]:
        with self._lock:
            rec = self._runs.get(agent_id)
            if not rec:
                return False, f"Unknown agent '{agent_id}'."
            rec["stop_event"].set()
            rec["status"] = "stopped"
            rec["finished_at"] = time.time()
            return True, f"Stopped agent {agent_id}."

    def get_agent_report(self, agent_id: str) -> Tuple[bool, str]:
        with self._lock:
            rec = self._runs.get(agent_id)
            if not rec:
                return False, f"Unknown agent '{agent_id}'."
            return True, rec.get("report") or f"Status: {rec.get('status')}"

    def list_live_agents(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": rec["id"],
                    "label": rec["label"],
                    "status": rec["status"],
                    "brief": rec["brief"][:80],
                    "files": list(rec.get("files", [])),
                }
                for rec in self._runs.values()
            ]


# ---------------------------------------------------------------------------
# 7. Local Model Discovery + Hybrid / Backup / Effort / Lite / Qwen-1000 Router
# ---------------------------------------------------------------------------

class LocalModelDiscoveryService:
    """Probes Ollama, LM Studio, and custom OpenAI-compatible local/LAN endpoints."""

    MIN_CONTEXT = 32_768
    LITE_MIN_CONTEXT = 16_384
    OLLAMA_URL = "http://127.0.0.1:11434"
    LMSTUDIO_URL = "http://127.0.0.1:1234/v1"

    @staticmethod
    def normalize_local_server_url(raw_url: str) -> Optional[str]:
        cleaned = str(raw_url or "").strip()
        if not cleaned:
            return None
        parsed = urllib.parse.urlparse(cleaned)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            return None
        path_part = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path_part or '/v1'}"

    @staticmethod
    def model_billions(model_id: str) -> Optional[float]:
        matched = re.search(r"(?:^|[^a-z0-9.])e?(\d+(?:\.\d+)?)b(?![a-z0-9])", str(model_id or ""), re.IGNORECASE)
        return float(matched.group(1)) if matched else None

    @classmethod
    def is_small_model(cls, model_id: str) -> bool:
        billions = cls.model_billions(model_id)
        return (billions is not None) and (billions <= 9.0)

    @classmethod
    def context_warning_note(cls, context_window: Optional[int], server_label: str = "Ollama", lite_mode: bool = False) -> str:
        need = cls.LITE_MIN_CONTEXT if lite_mode else cls.MIN_CONTEXT
        if not context_window or context_window >= need:
            return ""
        curr_k = max(1, round(context_window / 1024))
        need_k = need // 1024
        if "ollama" in server_label.lower():
            return f"Ollama loaded this model with {curr_k}k context; set Context Length to {need_k}k+ (OLLAMA_CONTEXT_LENGTH={need})."
        return f"{server_label} context is {curr_k}k; recommend {need_k}k+ tokens for reliable vision & tools."

    @staticmethod
    def format_local_error(server_label: str, error_text: str) -> str:
        err = str(error_text or "").lower()
        if "refused" in err or "connectionerror" in err or "failed to establish" in err:
            return f"{server_label} is not running or unreachable on this computer (check server status; internet is not required)."
        if "401" in err or "403" in err or "key" in err:
            return f"{server_label} rejected the API key. Check your local server key settings."
        if "timed out" in err or "timeout" in err:
            return f"{server_label} took too long to load the local model."
        return f"{server_label} error: {str(error_text)[:140]}"

    @classmethod
    def probe_ollama(cls, timeout: float = 1.5) -> Dict[str, Any]:
        if requests is None:
            return {"id": "ollama", "label": "Ollama", "reachable": False, "models": [], "error": "requests unavailable"}
        try:
            resp = requests.get(f"{cls.OLLAMA_URL}/api/tags", timeout=timeout)
            if not resp.ok:
                return {"id": "ollama", "label": "Ollama", "reachable": False, "models": [], "error": f"HTTP {resp.status_code}"}
            models_raw = resp.json().get("models", [])
            models = [str(item.get("name", "")).strip() for item in models_raw if item.get("name")]
            return {"id": "ollama", "label": "Ollama", "base_url": f"{cls.OLLAMA_URL}/v1", "reachable": True, "models": models, "error": ""}
        except Exception as exc:
            return {"id": "ollama", "label": "Ollama", "base_url": f"{cls.OLLAMA_URL}/v1", "reachable": False, "models": [], "error": cls.format_local_error("Ollama", str(exc))}

    @classmethod
    def probe_openai_server(cls, server_id: str, label: str, base_url: str, api_key: str = "", timeout: float = 1.5) -> Dict[str, Any]:
        if requests is None:
            return {"id": server_id, "label": label, "reachable": False, "models": [], "error": "requests unavailable"}
        norm_url = cls.normalize_local_server_url(base_url)
        if not norm_url:
            return {"id": server_id, "label": label, "reachable": False, "models": [], "error": "Invalid URL"}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            resp = requests.get(f"{norm_url}/models", headers=headers, timeout=timeout)
            if not resp.ok:
                return {"id": server_id, "label": label, "base_url": norm_url, "reachable": False, "models": [], "error": cls.format_local_error(label, f"HTTP {resp.status_code}")}
            raw_list = resp.json().get("data", [])
            models = [str(item.get("id", "")).strip() for item in raw_list if item.get("id")]
            return {"id": server_id, "label": label, "base_url": norm_url, "reachable": True, "models": models, "error": ""}
        except Exception as exc:
            return {"id": server_id, "label": label, "base_url": norm_url, "reachable": False, "models": [], "error": cls.format_local_error(label, str(exc))}


LITE_MODE_TOOLS = {
    "remember_commitment", "snooze_reminder", "list_commitments",
    "show_list", "forget_fact", "open_url", "open_app", "search_web", "about_jarvis",
}


def build_lite_system_prompt(user_name: str = "User", look_id: str = "jarvis") -> str:
    persona = LOOK_PERSONAS.get(look_id, LOOK_PERSONAS["jarvis"])
    return (
        f"You are {persona['name']}, {persona['body']} on {user_name}'s Windows screen.\n"
        f"Personality: {persona['style_prompt']}\n"
        f"Lite Mode is ON for small/local models: keep answers to 1-3 crisp sentences, use plain text or short '- ' bullets, "
        f"and use reminders/lists/memories/open_url/open_app/search_web when helpful. If asked for full-screen pixel clicking or "
        f"complex diagrams, explain that Lite Mode is on (toggle off in Account/Model settings)."
    )


class HybridModelRouter:
    """Routes turns to job-specific hybrid models, inline model mentions, backups, and Qwen 0..1000 scaling."""

    JOB_PATTERNS: Dict[str, Tuple[re.Pattern[str], ...]] = {
        "computer": (
            re.compile(r"\b(click|double.?click|right.?click|scroll|drag|type into|act_in_app|screenshot|this window|this app)\b", re.IGNORECASE),
        ),
        "code": (
            re.compile(r"\b(write|build|edit|fix|debug|refactor|implement|run|compile|git|python|script|function|api|bug|tests?)\b", re.IGNORECASE),
        ),
        "agents": (
            re.compile(r"\b(send an agent|in the background|spawn an agent|research this for me|subagent)\b", re.IGNORECASE),
        ),
        "draw": (
            re.compile(r"\b(draw|diagram|sketch|flow ?chart|mind ?map|timeline|bar chart|3d|molecule|map|geometry|quiz|illustrate|plot)\b", re.IGNORECASE),
        ),
        "web": (
            re.compile(r"\b(search the web|search online|look (?:it|that) up|latest news|current price|recent release)\b", re.IGNORECASE),
        ),
    }

    EFFORT_SLOTS = ("auto", "off", "low", "medium", "high", "xhigh", "max")

    @classmethod
    def classify_job_kinds(cls, prompt_text: str) -> List[str]:
        cleaned = str(prompt_text or "").strip()
        if not cleaned or re.match(r"^(?:thanks?|ok(?:ay)?|cool|nice|yes|no|hi|hey)\b[.!?]*$", cleaned, re.IGNORECASE):
            return []
        found: List[str] = []
        for job_kind in ("computer", "code", "agents", "draw", "web"):
            if any(pat.search(cleaned) for pat in cls.JOB_PATTERNS[job_kind]):
                found.append(job_kind)
        return found

    @staticmethod
    def detect_inline_model_override(prompt_text: str) -> Optional[str]:
        text = str(prompt_text or "")
        match_obj = re.search(
            r"\b(?:use|ask|switch to|run on)\s+(gemini[\w.\-]*|opus|sonnet|claude[\w.\-]*|gpt[\w.\-]*|deepseek[\w.\-]*|qwen[\w.\-]*|grok[\w.\-]*)\b",
            text,
            re.IGNORECASE,
        )
        if not match_obj:
            return None
        token = match_obj.group(1).lower()
        alias_map = {
            "opus": "claude-opus-4-20250514",
            "sonnet": "claude-sonnet-4-20250514",
            "gemini": "gemini-3.8-flash",
            "gpt": "gpt-4o",
            "grok": "grok-2-vision-1212",
            "deepseek": "deepseek-ai/DeepSeek-V4.1-Flash",
        }
        return alias_map.get(token, match_obj.group(1))

    @classmethod
    def select_model_for_turn(cls, prompt_text: str, settings_store: Any) -> Tuple[str, str]:
        override_model = cls.detect_inline_model_override(prompt_text)
        if override_model:
            return override_model, "inline"
        hybrid_map = settings_store.get("hybrid_models", {}) if settings_store else {}
        if isinstance(hybrid_map, dict):
            for job_kind in cls.classify_job_kinds(prompt_text):
                candidate = str(hybrid_map.get(job_kind, "") or "").strip()
                if candidate:
                    return candidate, f"hybrid:{job_kind}"
        default_model = str(settings_store.get("chat_model", "deepseek-ai/DeepSeek-V4.1-Flash")) if settings_store else "deepseek-ai/DeepSeek-V4.1-Flash"
        return default_model, "default"

    @staticmethod
    def get_backup_chain(settings_store: Any) -> List[str]:
        if not settings_store:
            return []
        backups: List[str] = []
        for key_name in ("backup_model_1", "backup_model_2"):
            val = str(settings_store.get(key_name, "") or "").strip()
            if val and val not in backups:
                backups.append(val)
        return backups

    @staticmethod
    def resolve_reasoning_effort(slot_name: str, settings_store: Any) -> Optional[str]:
        if not settings_store:
            return None
        effort_map = settings_store.get("effort_per_slot", {}) or {}
        chosen = str(effort_map.get(slot_name, "") or settings_store.get("reasoning_effort", "auto") or "auto").strip().lower()
        if chosen in ("off", "auto", ""):
            return None
        return chosen

    @staticmethod
    def is_qwen_1000_grid_model(model_id: str) -> bool:
        lower_id = str(model_id or "").lower()
        return any(tag in lower_id for tag in ("qwen3-vl", "qwen3.5", "qwen3.6", "qwen-vl", "qwen2.5-vl"))

    @staticmethod
    def convert_qwen_1000_coords(coord_x: float, coord_y: float, screen_w: float, screen_h: float) -> Tuple[float, float]:
        """Converts Qwen3-VL 0..1000 normalized grounding coordinates into pixel coordinates."""
        return (coord_x / 1000.0) * float(screen_w), (coord_y / 1000.0) * float(screen_h)


# ---------------------------------------------------------------------------
# 8. Encrypted Sync Engine (scrypt + AES-256-GCM + Tombstones)
# ---------------------------------------------------------------------------

class EncryptedSyncService:
    """Peer-to-peer encrypted memory/commitment sync via user-selected shared folder (OneDrive/Dropbox/Drive)."""

    SUBDIR_NAME = "Jarvis Sync"
    HEADER_NAME = "sync.json"
    KEY_NAME = "sync-key"
    MIN_PASSPHRASE = 8
    TOMBSTONE_DAYS = 60
    CHECK_TOKEN = "jarvis-sync-v1"

    @staticmethod
    def derive_key_from_passphrase(passphrase: str, salt_bytes: bytes) -> bytes:
        return hashlib.scrypt(
            str(passphrase or "").encode("utf-8"),
            salt=salt_bytes,
            n=32768,
            r=8,
            p=1,
            maxmem=64 * 1024 * 1024,
            dklen=32,
        )

    @staticmethod
    def seal_json(key_bytes: bytes, payload_obj: Any, aad_text: str) -> Dict[str, str]:
        if not HAS_CRYPTOGRAPHY or AESGCM is None:
            raise RuntimeError("cryptography AESGCM unavailable")
        iv_bytes = os.urandom(12)
        aes = AESGCM(key_bytes)
        plain_bytes = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
        cipher_plus_tag = aes.encrypt(iv_bytes, plain_bytes, str(aad_text).encode("utf-8"))
        cipher_only = cipher_plus_tag[:-16]
        tag_only = cipher_plus_tag[-16:]
        return {
            "iv": base64.b64encode(iv_bytes).decode("ascii"),
            "tag": base64.b64encode(tag_only).decode("ascii"),
            "data": base64.b64encode(cipher_only).decode("ascii"),
        }

    @staticmethod
    def unseal_json(key_bytes: bytes, box_dict: Dict[str, Any], aad_text: str) -> Any:
        if not HAS_CRYPTOGRAPHY or AESGCM is None:
            raise RuntimeError("cryptography AESGCM unavailable")
        iv_bytes = base64.b64decode(str(box_dict["iv"]))
        tag_bytes = base64.b64decode(str(box_dict["tag"]))
        cipher_bytes = base64.b64decode(str(box_dict["data"]))
        aes = AESGCM(key_bytes)
        plain_bytes = aes.decrypt(iv_bytes, cipher_bytes + tag_bytes, str(aad_text).encode("utf-8"))
        return json.loads(plain_bytes.decode("utf-8"))

    @classmethod
    def prepare_sync_folder(cls, folder_path: str, passphrase: str, dpapi_store: Optional[WindowsDPAPISecretStore] = None) -> Tuple[bool, Optional[bytes], str]:
        if len(str(passphrase or "")) < cls.MIN_PASSPHRASE:
            return False, None, f"Passphrase must be at least {cls.MIN_PASSPHRASE} characters."
        sync_dir = os.path.join(folder_path, cls.SUBDIR_NAME)
        header_path = os.path.join(sync_dir, cls.HEADER_NAME)
        os.makedirs(sync_dir, exist_ok=True)
        store = dpapi_store or WindowsDPAPISecretStore()
        if not os.path.exists(header_path):
            salt_bytes = os.urandom(16)
            key_bytes = cls.derive_key_from_passphrase(passphrase, salt_bytes)
            header_obj = {
                "version": 1,
                "kdf": "scrypt",
                "salt": base64.b64encode(salt_bytes).decode("ascii"),
                "check": cls.seal_json(key_bytes, cls.CHECK_TOKEN, "check"),
            }
            with open(header_path, "w", encoding="utf-8") as handle:
                json.dump(header_obj, handle, indent=2)
            store.set_secret(cls.KEY_NAME, base64.b64encode(key_bytes).decode("ascii"))
            return True, key_bytes, "Initialized encrypted sync folder."
        try:
            with open(header_path, "r", encoding="utf-8") as handle:
                head = json.load(handle)
            salt_bytes = base64.b64decode(head["salt"])
            key_bytes = cls.derive_key_from_passphrase(passphrase, salt_bytes)
            verified = cls.unseal_json(key_bytes, head["check"], "check")
            if verified != cls.CHECK_TOKEN:
                return False, None, "Incorrect sync passphrase."
            store.set_secret(cls.KEY_NAME, base64.b64encode(key_bytes).decode("ascii"))
            return True, key_bytes, "Connected to encrypted sync folder."
        except Exception:
            return False, None, "Incorrect sync passphrase or damaged sync.json."

    @staticmethod
    def hash_item(kind_prefix: str, item_dict: Dict[str, Any]) -> str:
        clean = {k_name: v_val for k_name, v_val in item_dict.items() if k_name != "usedAt"}
        raw = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{kind_prefix}:{raw}".encode("utf-8")).hexdigest()[:16]

    @classmethod
    def merge_sync_state(
        cls,
        local_commitments: List[Dict[str, Any]],
        local_facts: List[Dict[str, Any]],
        prev_state: Optional[Dict[str, Any]],
        remote_payloads: List[Dict[str, Any]],
        device_id: str,
        now_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        ts_now = float(now_ms if now_ms is not None else time.time() * 1000.0)
        state = prev_state or {"items": {}, "tombs": {}}
        prev_items = state.get("items", {})
        tombs: Dict[str, float] = dict(state.get("tombs", {}))
        mine: Dict[str, Dict[str, Any]] = {}

        for prefix_ch, source_list in (("c", local_commitments), ("f", local_facts)):
            for entry in source_list:
                item_id = str(entry.get("id", ""))
                if not item_id:
                    continue
                sync_key = f"{prefix_ch}:{item_id}"
                item_hash = cls.hash_item(prefix_ch, entry)
                prev_meta = prev_items.get(sync_key)
                changed_at = (
                    prev_meta["at"]
                    if (prev_meta and prev_meta.get("hash") == item_hash)
                    else float(entry.get("updated_at") or entry.get("created_at") or ts_now)
                )
                mine[sync_key] = {"v": dict(entry), "at": changed_at}

        for old_key in prev_items:
            if (old_key not in mine) and (old_key not in tombs):
                tombs[old_key] = ts_now

        for remote_doc in remote_payloads:
            for tomb_key, tomb_at in (remote_doc.get("tombs") or {}).items():
                if float(tomb_at) > float(tombs.get(tomb_key, -1)):
                    tombs[tomb_key] = float(tomb_at)

        merged: Dict[str, Dict[str, Any]] = dict(mine)
        changed = False
        for remote_doc in remote_payloads:
            for sync_key, rem_entry in (remote_doc.get("items") or {}).items():
                rem_at = float(rem_entry.get("at", 0))
                if sync_key in tombs and tombs[sync_key] >= rem_at:
                    continue
                curr = merged.get(sync_key)
                if (curr is None) or (rem_at > float(curr.get("at", 0))):
                    merged[sync_key] = rem_entry
                    changed = True

        cutoff = ts_now - cls.TOMBSTONE_DAYS * 86400_000.0
        for final_key in list(merged.keys()):
            if float(tombs.get(final_key, -1)) >= float(merged[final_key]["at"]):
                del merged[final_key]
                changed = True
        pruned_tombs = {t_k: t_v for t_k, t_v in tombs.items() if float(t_v) >= cutoff}

        out_commitments = [entry["v"] for key_k, entry in merged.items() if key_k.startswith("c:")]
        out_facts = [entry["v"] for key_k, entry in merged.items() if key_k.startswith("f:")]
        new_state = {
            "items": {key_k: {"hash": cls.hash_item(key_k[0], entry["v"]), "at": entry["at"]} for key_k, entry in merged.items()},
            "tombs": pruned_tombs,
        }
        out_payload = {"version": 1, "device": device_id, "items": merged, "tombs": pruned_tombs}
        return {
            "commitments": out_commitments,
            "facts": out_facts,
            "state": new_state,
            "out": out_payload,
            "changed": changed,
        }

    @classmethod
    def run_sync_pass(
        cls,
        folder_path: str,
        device_id: str,
        commitments_service: Any,
        facts_service: Any,
        dpapi_store: Optional[WindowsDPAPISecretStore] = None,
    ) -> Tuple[bool, str]:
        if not folder_path or not os.path.isdir(folder_path):
            return False, "Sync folder missing."
        store = dpapi_store or WindowsDPAPISecretStore()
        key_b64 = store.get_secret(cls.KEY_NAME)
        if not key_b64:
            return False, "Sync passphrase not unlocked yet."
        key_bytes = base64.b64decode(key_b64)
        sync_dir = os.path.join(folder_path, cls.SUBDIR_NAME)
        os.makedirs(sync_dir, exist_ok=True)

        remote_payloads: List[Dict[str, Any]] = []
        for fname in os.listdir(sync_dir):
            if (not fname.endswith(".json")) or (fname == cls.HEADER_NAME) or (fname == f"{device_id}.json"):
                continue
            rem_dev = fname[:-5]
            try:
                with open(os.path.join(sync_dir, fname), "r", encoding="utf-8") as handle:
                    box = json.load(handle)
                remote_payloads.append(cls.unseal_json(key_bytes, box, rem_dev))
            except Exception:
                continue

        prev_state = None
        if os.path.exists(DEFAULT_SYNC_STATE_PATH):
            try:
                with open(DEFAULT_SYNC_STATE_PATH, "r", encoding="utf-8") as handle:
                    prev_state = json.load(handle)
            except Exception:
                prev_state = None

        local_c = commitments_service.export_sync_items()
        local_f = facts_service.export_sync_items()
        result = cls.merge_sync_state(local_c, local_f, prev_state, remote_payloads, device_id=device_id)
        if result["changed"]:
            commitments_service.apply_sync_items(result["commitments"])
            facts_service.apply_sync_items(result["facts"])

        os.makedirs(os.path.dirname(DEFAULT_SYNC_STATE_PATH), exist_ok=True)
        with open(DEFAULT_SYNC_STATE_PATH, "w", encoding="utf-8") as handle:
            json.dump(result["state"], handle, indent=2)
        out_box = cls.seal_json(key_bytes, result["out"], device_id)
        out_path = os.path.join(sync_dir, f"{device_id}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(out_box, handle)
        return True, f"Synced ({len(result['commitments'])} commitments, {len(result['facts'])} facts)."


# ---------------------------------------------------------------------------
# 9. Diagnostics Exporter
# ---------------------------------------------------------------------------

class DiagnosticsExporter:
    """Exports scrubbed diagnostic reports to ~/Downloads and selects the file in Explorer."""

    _KEY_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\b(?:sk|rk|pk)-(?:ant-|proj-|or-|svcacct-)?[A-Za-z0-9_\-]{16,}"), "<key>"),
        (re.compile(r"\bxai-[A-Za-z0-9_\-]{16,}"), "<key>"),
        (re.compile(r"\bAIza[0-9A-Za-z_\-]{25,}"), "<key>"),
        (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "<key>"),
        (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "<token>"),
        (re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=\-]{12,}", re.IGNORECASE), r"\1 <token>"),
        (re.compile(r'("(?:[A-Za-z_]*(?:key|token|secret|password)[A-Za-z_]*)"\s*:\s*)"[^"]{4,}"', re.IGNORECASE), r'\1"<redacted>"'),
        (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "<email>"),
    )

    @classmethod
    def redact_keys_only(cls, raw_text: str) -> str:
        out = str(raw_text or "")
        for pattern_re, repl_text in cls._KEY_PATTERNS[:-1]:
            out = pattern_re.sub(repl_text, out)
        return out

    @classmethod
    def redact_secrets_and_pii(cls, raw_text: str, home_dir: Optional[str] = None, username: Optional[str] = None) -> str:
        out = str(raw_text or "")
        home = home_dir or os.path.expanduser("~")
        if home and len(home) > 3:
            for variant in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
                out = re.sub(re.escape(variant), "~", out, flags=re.IGNORECASE)
        for pattern_re, repl_text in cls._KEY_PATTERNS:
            out = pattern_re.sub(repl_text, out)
        user_val = username or os.environ.get("USERNAME", "")
        if user_val and len(user_val) > 2:
            out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(user_val)}(?![A-Za-z0-9])", "<user>", out, flags=re.IGNORECASE)
        return out

    @classmethod
    def scrub_config_dict(cls, obj: Any, key_name: str = "") -> Any:
        if isinstance(obj, list):
            return [cls.scrub_config_dict(item, key_name) for item in obj]
        if isinstance(obj, dict):
            return {sub_k: cls.scrub_config_dict(sub_v, sub_k) for sub_k, sub_v in obj.items()}
        if isinstance(obj, str) and re.search(r"key|token|secret|password|auth|cookie|email", key_name, re.IGNORECASE):
            return "<redacted>" if obj else ""
        return obj

    @classmethod
    def build_diagnostics_report(cls, settings_store: Any, extra_sections: Optional[List[Tuple[str, Any]]] = None, version: str = "1.2.0") -> str:
        sections: List[Tuple[str, Any]] = [
            ("Environment", f"OS=Windows | Python={os.sys.version.split()[0]} | App=Jarvis {version}"),
            ("Settings (scrubbed)", cls.scrub_config_dict(settings_store.all_values() if settings_store else {})),
        ]
        if extra_sections:
            sections.extend(extra_sections)
        lines = [
            f"Jarvis Diagnostics Report ({datetime.datetime.now().isoformat(timespec='seconds')})",
            "API keys, tokens, emails, home paths, and Windows username have been redacted.",
        ]
        for title, body in sections:
            lines.append(f"\n-------- {title} --------")
            lines.append(body if isinstance(body, str) else json.dumps(body, indent=2, ensure_ascii=False))
        return cls.redact_secrets_and_pii("\n".join(lines) + "\n")

    @classmethod
    def save_and_reveal_diagnostics(cls, settings_store: Any, log_paths: Optional[List[str]] = None) -> str:
        extra: List[Tuple[str, Any]] = []
        for path_item in (log_paths or []):
            if os.path.exists(path_item):
                try:
                    with open(path_item, "r", encoding="utf-8", errors="replace") as handle:
                        content = handle.read()[-32768:]
                    extra.append((os.path.basename(path_item), content or "(empty)"))
                except Exception as exc:
                    extra.append((os.path.basename(path_item), f"Read error: {exc}"))
        report_text = cls.build_diagnostics_report(settings_store, extra_sections=extra)
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        target_dir = downloads if os.path.isdir(downloads) else os.path.expanduser("~")
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(target_dir, f"jarvis-diagnostics-{stamp}.log")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(report_text)
        try:
            subprocess.Popen(["explorer.exe", f"/select,{os.path.normpath(out_path)}"])
        except Exception:
            pass
        return out_path


# ---------------------------------------------------------------------------
# 10. Draw Helpers: SMILES Chemistry, World Country Map, Save Card PNG, What's New
# ---------------------------------------------------------------------------

def parse_smiles_to_2d(smiles_str: str) -> Dict[str, Any]:
    """
    Lightweight SMILES parser computing clean 2D atom coordinates and bonds for molecule cards.
    Supports atoms, explicit/implicit bonds (-, =, #, :), parentheses branching, and rings 1-9.
    """
    src = re.sub(r"\s+", "", str(smiles_str or ""))
    if not src:
        return {"atoms": [], "bonds": []}
    atoms: List[Dict[str, Any]] = []
    bonds: List[Tuple[int, int, int]] = []
    branch_stack: List[int] = []
    rings: Dict[str, int] = {}
    prev_atom: Optional[int] = None
    next_order = 1
    idx = 0
    while idx < len(src):
        ch = src[idx]
        if ch == "(":
            if prev_atom is not None:
                branch_stack.append(prev_atom)
            idx += 1
        elif ch == ")":
            if branch_stack:
                prev_atom = branch_stack.pop()
            next_order = 1
            idx += 1
        elif ch in "-=#:":
            next_order = {"-": 1, "=": 2, "#": 3, ":": 4}[ch]
            idx += 1
        elif ch.isdigit():
            if prev_atom is not None:
                if ch in rings:
                    partner = rings.pop(ch)
                    if partner != prev_atom:
                        bonds.append((partner, prev_atom, next_order))
                else:
                    rings[ch] = prev_atom
            next_order = 1
            idx += 1
        elif ch == "[":
            close_idx = src.find("]", idx)
            elem = src[idx + 1:close_idx] if close_idx > idx else "X"
            elem = re.sub(r"[^A-Za-z]", "", elem)[:2] or "C"
            atom_id = len(atoms)
            atoms.append({"id": atom_id, "element": elem})
            if prev_atom is not None:
                bonds.append((prev_atom, atom_id, next_order))
            prev_atom = atom_id
            next_order = 1
            idx = (close_idx + 1) if close_idx > idx else (idx + 1)
        elif ch.isalpha():
            two = src[idx:idx + 2]
            if two in ("Cl", "Br", "Si", "Na", "Ca", "Fe", "Al", "Mg", "Zn"):
                elem = two
                idx += 2
            else:
                elem = ch.upper()
                idx += 1
            atom_id = len(atoms)
            atoms.append({"id": atom_id, "element": elem})
            if prev_atom is not None:
                bonds.append((prev_atom, atom_id, next_order))
            prev_atom = atom_id
            next_order = 1
        else:
            idx += 1

    count = len(atoms)
    if count == 0:
        return {"atoms": [], "bonds": []}
    # Detect simple cycles (e.g. benzene or cyclohexane ring) for clean polygon placement
    adj: List[List[int]] = [[] for _ in range(count)]
    for atom_a, atom_b, _order in bonds:
        adj[atom_a].append(atom_b)
        adj[atom_b].append(atom_a)

    # Initial spiral/circle positions + deterministic spring relaxation so rings & branches lay out cleanly
    coords_x: List[float] = []
    coords_y: List[float] = []
    if len(bonds) >= count and count >= 3:
        for pos_idx in range(count):
            ang = (2.0 * math.pi * pos_idx) / max(1, count) - math.pi / 2.0
            coords_x.append(math.cos(ang))
            coords_y.append(math.sin(ang))
    else:
        pos_map: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0)}
        angle_map: Dict[int, float] = {0: 0.0}
        for atom_a, atom_b, _order in bonds:
            if atom_a in pos_map and atom_b not in pos_map:
                parent_ang = angle_map.get(atom_a, 0.0)
                siblings = sum(1 for nbr in adj[atom_a] if nbr in pos_map or nbr < atom_b)
                offset = ((siblings % 3) - 1) * 1.05
                new_ang = parent_ang + (offset if atom_a > 0 else (siblings * 0.9))
                pos_map[atom_b] = (pos_map[atom_a][0] + math.cos(new_ang), pos_map[atom_a][1] + math.sin(new_ang))
                angle_map[atom_b] = new_ang
        for pos_idx in range(count):
            pt_xy = pos_map.get(pos_idx, (float(pos_idx), (pos_idx % 2) * 0.6))
            coords_x.append(pt_xy[0])
            coords_y.append(pt_xy[1])

    for _iter in range(28):
        forces_x = [0.0] * count
        forces_y = [0.0] * count
        for idx_a in range(count):
            for idx_b in range(idx_a + 1, count):
                dx_val = coords_x[idx_b] - coords_x[idx_a]
                dy_val = coords_y[idx_b] - coords_y[idx_a]
                dist_sq = max(0.04, dx_val * dx_val + dy_val * dy_val)
                rep = 0.25 / dist_sq
                dist = math.sqrt(dist_sq)
                forces_x[idx_a] -= rep * dx_val / dist
                forces_y[idx_a] -= rep * dy_val / dist
                forces_x[idx_b] += rep * dx_val / dist
                forces_y[idx_b] += rep * dy_val / dist
        for atom_a, atom_b, _order in bonds:
            dx_val = coords_x[atom_b] - coords_x[atom_a]
            dy_val = coords_y[atom_b] - coords_y[atom_a]
            dist = max(0.05, math.hypot(dx_val, dy_val))
            spring = (dist - 1.0) * 0.24
            forces_x[atom_a] += spring * dx_val / dist
            forces_y[atom_a] += spring * dy_val / dist
            forces_x[atom_b] -= spring * dx_val / dist
            forces_y[atom_b] -= spring * dy_val / dist
        for idx_a in range(count):
            coords_x[idx_a] += max(-0.25, min(0.25, forces_x[idx_a]))
            coords_y[idx_a] += max(-0.25, min(0.25, forces_y[idx_a]))

    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)
    span_x = max(0.5, max_x - min_x)
    span_y = max(0.5, max_y - min_y)
    for idx_a, atom_dict in enumerate(atoms):
        atom_dict["nx"] = 0.14 + 0.72 * ((coords_x[idx_a] - min_x) / span_x)
        atom_dict["ny"] = 0.16 + 0.68 * ((coords_y[idx_a] - min_y) / span_y)
    return {"atoms": atoms, "bonds": bonds}


WORLD_COUNTRIES: Tuple[Dict[str, Any], ...] = (
    {"n": "United States", "c": "north america", "b": (-124.7, 25.1, -67.0, 49.4), "at": (-97.5, 39.5), "a": ("usa", "us", "america")},
    {"n": "Canada", "c": "north america", "b": (-141.0, 41.7, -52.6, 83.2), "at": (-101.9, 60.3), "a": ("can",)},
    {"n": "Mexico", "c": "north america", "b": (-117.1, 14.5, -86.8, 32.7), "at": (-102.3, 23.9), "a": ("mex",)},
    {"n": "Brazil", "c": "south america", "b": (-74.0, -33.8, -34.7, 5.2), "at": (-49.6, -12.1), "a": ("bra",)},
    {"n": "Argentina", "c": "south america", "b": (-73.4, -55.2, -53.6, -21.8), "at": (-64.2, -33.5), "a": ("arg",)},
    {"n": "Chile", "c": "south america", "b": (-75.6, -55.6, -67.0, -17.6), "at": (-72.3, -38.2), "a": ("chl",)},
    {"n": "Colombia", "c": "south america", "b": (-79.0, -4.3, -66.9, 12.4), "at": (-73.2, 3.4), "a": ("col",)},
    {"n": "Peru", "c": "south america", "b": (-81.4, -18.3, -68.7, -0.1), "at": (-72.9, -13.0), "a": ("per",)},
    {"n": "United Kingdom", "c": "europe", "b": (-8.2, 49.9, 1.8, 60.8), "at": (-2.5, 54.0), "a": ("uk", "britain", "england", "gbr")},
    {"n": "France", "c": "europe", "b": (-5.1, 41.3, 9.6, 51.1), "at": (2.3, 46.6), "a": ("fra",)},
    {"n": "Germany", "c": "europe", "b": (5.9, 47.3, 15.0, 55.1), "at": (10.4, 51.1), "a": ("deu", "ger")},
    {"n": "Spain", "c": "europe", "b": (-9.3, 36.0, 4.3, 43.8), "at": (-3.7, 40.4), "a": ("esp",)},
    {"n": "Italy", "c": "europe", "b": (6.6, 36.6, 18.5, 47.1), "at": (12.5, 42.8), "a": ("ita",)},
    {"n": "Poland", "c": "europe", "b": (14.1, 49.0, 24.1, 54.8), "at": (19.1, 51.9), "a": ("pol",)},
    {"n": "Ukraine", "c": "europe", "b": (22.1, 44.4, 40.2, 52.4), "at": (31.2, 48.4), "a": ("ukr",)},
    {"n": "Norway", "c": "europe", "b": (5.0, 58.1, 31.3, 80.7), "at": (9.7, 61.4), "a": ("nor",)},
    {"n": "Sweden", "c": "europe", "b": (11.1, 55.3, 24.2, 69.1), "at": (16.2, 62.0), "a": ("swe",)},
    {"n": "Greece", "c": "europe", "b": (19.4, 34.8, 28.2, 41.7), "at": (22.9, 39.1), "a": ("grc",)},
    {"n": "Turkey", "c": "asia", "b": (26.0, 35.8, 44.8, 42.1), "at": (35.2, 39.0), "a": ("tur", "turkiye")},
    {"n": "Russia", "c": "europe", "b": (27.3, 41.2, 180.0, 81.3), "at": (85.0, 61.5), "a": ("rus",)},
    {"n": "Egypt", "c": "africa", "b": (24.7, 22.0, 36.9, 31.7), "at": (30.8, 26.8), "a": ("egy",)},
    {"n": "Nigeria", "c": "africa", "b": (2.7, 4.3, 14.7, 13.9), "at": (8.7, 9.1), "a": ("nga",)},
    {"n": "Kenya", "c": "africa", "b": (33.9, -4.7, 41.9, 5.5), "at": (37.9, 0.5), "a": ("ken",)},
    {"n": "South Africa", "c": "africa", "b": (16.3, -34.8, 32.8, -22.1), "at": (23.7, -29.7), "a": ("zaf", "rsa")},
    {"n": "Morocco", "c": "africa", "b": (-13.2, 27.7, -1.0, 35.9), "at": (-7.1, 31.8), "a": ("mar",)},
    {"n": "Saudi Arabia", "c": "asia", "b": (34.5, 16.4, 55.7, 32.2), "at": (45.1, 23.9), "a": ("sau", "ksa")},
    {"n": "United Arab Emirates", "c": "asia", "b": (51.5, 22.6, 56.4, 26.1), "at": (54.4, 24.4), "a": ("uae",)},
    {"n": "India", "c": "asia", "b": (68.2, 8.1, 97.4, 35.5), "at": (78.9, 20.6), "a": ("ind",)},
    {"n": "China", "c": "asia", "b": (73.5, 18.2, 135.1, 53.6), "at": (104.2, 35.9), "a": ("chn", "prc")},
    {"n": "Japan", "c": "asia", "b": (129.5, 31.0, 145.8, 45.5), "at": (138.3, 36.2), "a": ("jpn",)},
    {"n": "South Korea", "c": "asia", "b": (126.1, 34.3, 129.6, 38.6), "at": (127.8, 36.5), "a": ("kor", "korea")},
    {"n": "Indonesia", "c": "asia", "b": (95.3, -10.4, 141.0, 5.5), "at": (113.9, -0.8), "a": ("idn",)},
    {"n": "Vietnam", "c": "asia", "b": (102.1, 8.6, 109.5, 23.4), "at": (108.3, 14.1), "a": ("vnm",)},
    {"n": "Thailand", "c": "asia", "b": (97.3, 5.6, 105.6, 20.5), "at": (100.9, 15.9), "a": ("tha",)},
    {"n": "Australia", "c": "oceania", "b": (113.3, -43.6, 153.6, -10.7), "at": (133.8, -25.3), "a": ("aus",)},
    {"n": "New Zealand", "c": "oceania", "b": (166.4, -47.3, 178.6, -34.4), "at": (172.5, -41.5), "a": ("nzl", "nz")},
)


def resolve_map_region(region: str = "", shade_list: Optional[List[Dict[str, Any]]] = None, pins_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Resolves a map's `[west, south, east, north]` view bbox and matching country records."""
    region_key = str(region or "").strip().lower()
    matched_countries: List[Dict[str, Any]] = []
    for country_row in WORLD_COUNTRIES:
        names = [country_row["n"].lower(), *[alias.lower() for alias in country_row["a"]]]
        if region_key and (region_key == country_row["c"] or region_key in names):
            matched_countries.append(country_row)
    for shade_item in (shade_list or []):
        target_name = str(shade_item.get("name", "")).strip().lower()
        for country_row in WORLD_COUNTRIES:
            if target_name in (country_row["n"].lower(), *country_row["a"]):
                if country_row not in matched_countries:
                    matched_countries.append(country_row)
    boxes = [row["b"] for row in matched_countries]
    for pin_item in (pins_list or []):
        lon_val = float(pin_item.get("lon", 0.0))
        lat_val = float(pin_item.get("lat", 0.0))
        boxes.append((lon_val - 8.0, lat_val - 6.0, lon_val + 8.0, lat_val + 6.0))
    if not boxes or region_key in ("world", "global", "earth"):
        return {"bbox": (-165.0, -56.0, 178.0, 76.0), "countries": matched_countries}
    west = max(-180.0, min(box_item[0] for box_item in boxes) - 5.0)
    south = max(-85.0, min(box_item[1] for box_item in boxes) - 5.0)
    east = min(180.0, max(box_item[2] for box_item in boxes) + 5.0)
    north = min(85.0, max(box_item[3] for box_item in boxes) + 5.0)
    return {"bbox": (west, south, east, north), "countries": matched_countries}


def save_card_image_to_downloads(png_bytes: bytes, card_title: str = "card", target_dir: Optional[str] = None) -> str:
    """Saves rendered card PNG bytes to `~/Downloads` without overwriting existing files."""
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    out_dir = target_dir or (downloads if os.path.isdir(downloads) else os.path.expanduser("~"))
    safe_title = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", str(card_title or "card")).strip(" .")
    safe_title = re.sub(r"\s+", " ", safe_title)[:40] or "card"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    base_name = f"Jarvis - {safe_title} - {stamp}"
    candidate = os.path.join(out_dir, f"{base_name}.png")
    suffix = 2
    while os.path.exists(candidate) and suffix < 50:
        candidate = os.path.join(out_dir, f"{base_name} ({suffix}).png")
        suffix += 1
    with open(candidate, "wb") as handle:
        handle.write(png_bytes)
    return candidate


WHATS_NEW_RELEASES: Tuple[Dict[str, Any], ...] = (
    {
        "version": "1.2.0 (Roamies Full Parity)",
        "date": "2026-11-19",
        "items": (
            "3D Animated Models with transparent casings (box, sphere, cylinder, cone, torus + spin/orbit/slide/pulse).",
            "JARVIS 3D LED Sphere + 8 Roamies companion looks & voice-emotion expressions.",
            "21 Visual Card Kinds: SMILES Molecules, World Maps, 2D Graphs, Geometry, Quizzes, Interactive Checklists, Sources, Clips, Tickets & Medals.",
            "Ask Bar at cursor (Ctrl+Shift), Area Snap (Ctrl+Shift+A), Circle-While-Talking, Show/Hide (Ctrl+Shift+S) & Wake Word.",
            "Safety Engine (HANDS_OFF, symlink guards, risky click Allow/Don't Allow bar), MCP Client, 5 Subagents, Local/Hybrid/Lite routing, Encrypted Sync & Diagnostics.",
        ),
    },
)


class SpeechNormalizationEngine:
    """Namespace wrapper grouping Roamies speech, cue, RTL, wake-name, briefing, and repeat helpers."""

    normalize_math_for_speech = staticmethod(normalize_math_for_speech)
    cue_feel = staticmethod(cue_feel)
    strip_voice_cues_for_display = staticmethod(strip_voice_cues_for_display)
    convert_cues_for_gemini = staticmethod(convert_cues_for_gemini)
    speakable_text = staticmethod(speakable_text)
    ends_with_question = staticmethod(ends_with_question)
    is_rtl_text = staticmethod(is_rtl_text)
    clean_wake_name = staticmethod(clean_wake_name)
    compute_briefing_day = staticmethod(compute_briefing_day)
    build_morning_briefing_note = staticmethod(build_morning_briefing_note)
    parse_repeat_schedule = staticmethod(parse_repeat_schedule)
    compute_next_repeat_timestamp = staticmethod(compute_next_repeat_timestamp)
    parse_voice_snooze_request = staticmethod(parse_voice_snooze_request)


class RoamiesDrawHelpers:
    """Namespace wrapper grouping SMILES, map resolution, PNG export, and release-note helpers."""

    WORLD_COUNTRIES = WORLD_COUNTRIES
    WHATS_NEW_RELEASES = WHATS_NEW_RELEASES
    parse_smiles_to_2d = staticmethod(parse_smiles_to_2d)
    resolve_map_region = staticmethod(resolve_map_region)
    save_card_image_to_downloads = staticmethod(save_card_image_to_downloads)
