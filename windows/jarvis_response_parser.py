"""
Vision system prompt, style-preference detector, and streaming response parser.

Separates natural spoken sentences (for real-time TTS streaming) from inline 2D
screen-drawing commands (`underline`, `ring`, `point`, `box`, `paper`, `highlight`,
and `[POINT:x,y:label]`) while scaling coordinates from screenshot space to screen space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple


@dataclass
class DrawingCommand:
    """Single 2D drawing instruction emitted by the vision model."""

    kind: str  # "underline" | "ring" | "point" | "box" | "paper" | "highlight" | "scroll" | agent action
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    radius: float = 32.0
    label: str = ""
    header: str = ""
    body: str = ""
    tone: str = "blue"


@dataclass
class ParsedTurnResponse:
    """Complete parsed response from a single companion turn."""

    spoken_text: str
    sentences: List[str] = field(default_factory=list)
    commands: List[DrawingCommand] = field(default_factory=list)
    primary_point_target: Optional[Tuple[float, float, str]] = None
    scroll_direction: Optional[str] = None
    scroll_amount: int = 0


_POINT_TAG_PATTERN = re.compile(
    r"\[POINT:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*(?::\s*([^:\]]+?))?(?::\s*[^\]]+)?\]",
    re.IGNORECASE,
)

_COMMAND_NAMES_PATTERN = (
    r"(underline|ring|point|box|frame|paper|highlight|scroll|"
    r"open_browser|open_url|search_web|open_app|press_keys|type_text)"
)

_CALL_LINE_PATTERN = re.compile(
    rf"^\s*{_COMMAND_NAMES_PATTERN}\s*\((.*)\)\s*;?\s*$",
    re.IGNORECASE,
)

_COMMAND_PREFIX_PATTERN = re.compile(
    rf"^{_COMMAND_NAMES_PATTERN}\s*\(",
    re.IGNORECASE,
)

_THINK_BLOCK_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def build_vision_system_prompt(
    screen_width: int,
    screen_height: int,
    style_notes: Optional[List[str]] = None,
    custom_words: Optional[List[str]] = None,
    has_lasso_crop: bool = False,
    scroll_step_index: int = 0,
    max_scroll_steps: int = 0,
    drawing_mode: str = "precise_lines",
    agent_actions_enabled: bool = True,
) -> str:
    """
    Constructs the system prompt for Jarvis's single-brain screen companion.

    Screenshots include 1:1 logical coordinate rulers along top and left edges
    (`0..screen_width` x `0..screen_height`) so drawing coordinates align accurately
    with visible text, equations, ChatGPT sections, and buttons.
    """
    normalized_mode = (drawing_mode or "precise_lines").strip().lower()
    prompt_sections: List[str] = [
        "You are Jarvis, a warm, crystal-clear visual tutor and desktop companion on Windows.",
        f"The attached screenshot is {screen_width}x{screen_height} pixels (1:1 logical screen coordinates, top-left is 0,0).",
        "A coordinate ruler runs along the top and left edges (ticks every 100px, faint grid every 200px) — carefully read exact X and Y pixel values from those rulers so every underline lands cleanly under the intended phrase without drifting into whitespace or covering text.",
        "Keep spoken replies warm, direct, and easy to follow (2 to 4 natural sentences).",
        "Always inspect the current screenshot freshly; never repeat steps or sections you already explained earlier.",
        "Never use markdown asterisks (**bold** or *italic*), bullet symbols, emojis, or code blocks in your text.",
        "Never read coordinates, numbers from drawing commands, or command names aloud.",
        "",
        "VISUAL EXPLANATION & SCREEN DRAWING COMMANDS:",
    ]

    if normalized_mode == "cursor_only":
        prompt_sections.extend(
            [
                "Cursor-only mode is enabled: do NOT emit underline, ring, box, highlight, or paper commands.",
                f"- Optionally emit up to 2 point(x, y, \"\") commands (x, y in 0..{screen_width} x 0..{screen_height}) to aim the cursor cleanly.",
            ]
        )
    elif normalized_mode in ("precise_lines", "lines_and_card"):
        prompt_sections.extend(
            [
                "Keep screen annotations minimal, neat, and textbook-clean while explaining:",
                f"- underline(x, y, width, \"\") -> PRIMARY tool. Draw 2 to 4 crisp horizontal underlines directly beneath the exact key words or short phrases you are explaining (y = bottom baseline of that text + 4px, x = left edge of the phrase, width = exact phrase pixel width; x, y in 0..{screen_width} x 0..{screen_height}). Pass empty \"\" for the label.",
                "- point(x, y, \"\") -> OPTIONAL. Gently glides the Jarvis cursor to a single button, icon, or formula.",
                "STRICTLY AVOID: Do NOT emit `box(...)`, `frame(...)`, `highlight(...)`, or `ring(...)`. Never draw giant boxes around whole paragraphs, never cover readable text with yellow highlights, and never invent floating tags.",
            ]
        )
        if normalized_mode == "lines_and_card":
            prompt_sections.extend(
                [
                    "- paper(24, 160, 320, 340, \"Card Title\", \"short header\", \"Step 1 -> Step 2 | section:Key Idea\") -> OPTIONAL summary diagram card in an empty margin when explaining multi-step flows; invent text strictly from visible screen content.",
                ]
            )
    else:
        prompt_sections.extend(
            [
                "While explaining ChatGPT responses, code, formulas, or articles, emit 2 to 4 accurate visual commands on their own lines between sentences:",
                f"- underline(x, y, width, \"\")  -> clean blue underline directly beneath a key heading/phrase (x, y in 0..{screen_width} x 0..{screen_height}).",
                "- ring(x, y, radius, \"label\")  -> neat circle around a compact icon or term (radius <= 55).",
                "- point(x, y, \"label\")         -> flies the blue Jarvis cursor to (x, y) and points cleanly.",
                "- box(x, y, width, height, \"label\") -> small frame around a compact formula or UI button only (never wrap whole paragraphs).",
                "- highlight(x, y, width, height, \"label\") -> subtle wash across one short line only.",
                "- paper(24, 160, 320, 340, \"Card Title\", \"short header\", \"Concept A -> Concept B | section:Key Idea | doc:Input -> doc:Output\")",
            ]
        )

    if agent_actions_enabled:
        prompt_sections.extend(
            [
                "",
                "DESKTOP & BROWSER AGENT COMMANDS:",
                "When the user explicitly asks you to open their browser, visit a website, search the web, launch an app, press a shortcut, or type text, emit the matching command on its own line alongside a brief spoken confirmation:",
                "- open_browser(\"https://www.google.com\") -> opens the default web browser (or \"https://chatgpt.com\", etc.).",
                "- open_url(\"https://example.com\")        -> opens a specific URL in a new browser tab.",
                "- search_web(\"search query\")            -> opens a web search for the query.",
                "- open_app(\"notepad\")                   -> launches a desktop app (chrome, edge, notepad, calculator, paint, explorer, terminal, vscode, settings).",
                "- press_keys(\"ctrl+t\")                  -> sends a safe keyboard shortcut or key (e.g., ctrl+t, ctrl+l, enter, pagedown).",
                "- type_text(\"hello\")                    -> types short text into the focused field.",
            ]
        )

    if has_lasso_crop:
        prompt_sections.extend(
            [
                "",
                "LASSO HIGHLIGHT FOCUS:",
                "The user drew a cyan lasso around a specific target region, and a 2x zoomed close-up crop is attached as the second image.",
                "Focus your explanation directly on the content inside the cyan lasso and align any underline or point to its exact coordinates on the main screenshot.",
            ]
        )

    if max_scroll_steps > 0:
        prompt_sections.extend(
            [
                "",
                f"SCROLL-AND-EXPLAIN MODE (Section {max(1, scroll_step_index)} of {max_scroll_steps}):",
                "You are walking the user visually through a scrollable page (such as a ChatGPT answer or article).",
                "Explain the section currently visible on screen with accurate, neatly placed underlines beneath key phrases.",
                "If more content clearly continues below the visible bottom of this pane AND more scroll steps remain, place `scroll(\"down\", 6)` on its own final line so Jarvis scrolls down to explain the next section automatically.",
                "If the answer has clearly ended or reached the bottom of the conversation, do NOT emit `scroll(...)`.",
            ]
        )
    else:
        prompt_sections.extend(
            [
                "",
                "OPTIONAL SCROLL COMMAND:",
                "- scroll(\"down\", 6) -> only emit this on its own final line if the user explicitly asked you to scroll and keep explaining what appears below.",
            ]
        )

    if style_notes:
        cleaned_notes = [note.strip() for note in style_notes if note and note.strip()]
        if cleaned_notes:
            prompt_sections.append("")
            prompt_sections.append("USER PREFERENCES (FOLLOW THESE STRICTLY):")
            for note_item in cleaned_notes[-6:]:
                prompt_sections.append(f"- {note_item}")

    if custom_words:
        cleaned_words = [word.strip() for word in custom_words if word and word.strip()]
        if cleaned_words:
            prompt_sections.append("")
            prompt_sections.append("Custom vocabulary the user uses: " + ", ".join(cleaned_words[:20]))

    return "\n".join(prompt_sections)


def is_scroll_and_explain_request(user_message: str) -> bool:
    """Detect prompts asking Jarvis to scroll through a page/ChatGPT answer and explain sections."""
    normalized = (user_message or "").lower().strip()
    if not normalized:
        return False
    negation_phrases = (
        "do not scroll",
        "don't scroll",
        "dont scroll",
        "no scroll",
        "without scrolling",
    )
    if any(phrase in normalized for phrase in negation_phrases):
        return False
    scroll_triggers = (
        "scroll",
        "walk me through",
        "walk through",
        "explain the whole",
        "explain this entire",
        "explain the entire",
        "step by step through",
        "read through",
        "keep going down",
    )
    return any(trigger in normalized for trigger in scroll_triggers)


def extract_style_preference_from_user_message(user_message: str) -> Optional[str]:
    """
    Recognizes persistent answer-style requests ("Answers the way you like them" in v1.1.5)
    such as asking for shorter answers, yes-or-no first, or simpler wording.
    """
    normalized = user_message.lower().strip()
    if not normalized:
        return None

    preference_rules: List[Tuple[Tuple[str, ...], str]] = [
        (
            ("shorter answer", "shorter replies", "keep it short", "keep it brief", "be briefer", "too long", "less words"),
            "Keep answers ultra-brief (1 to 2 short sentences maximum).",
        ),
        (
            ("yes or no first", "yes/no first", "start with yes or no", "direct yes or no"),
            "Start answers directly with Yes or No before a brief explanation.",
        ),
        (
            ("simpler words", "simpler language", "plain english", "explain simply", "like i'm 5", "eli5"),
            "Use simple, everyday words and avoid jargon.",
        ),
        (
            ("don't draw", "no drawings", "stop drawing"),
            "Avoid drawing cards or screen annotations unless explicitly asked.",
        ),
        (
            ("more detail", "longer explanation", "explain thoroughly"),
            "Give clear, detailed explanations (3 to 5 sentences).",
        ),
    ]

    for trigger_phrases, rule_text in preference_rules:
        if any(phrase in normalized for phrase in trigger_phrases):
            return rule_text
    return None


def clean_spoken_text(raw_text: str) -> str:
    """Strips markdown noise, thinking tags, and leftover drawing syntax from spoken text."""
    if not raw_text:
        return ""
    cleaned = _THINK_BLOCK_PATTERN.sub(" ", raw_text)
    cleaned = _POINT_TAG_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    # Strip markdown bold/italic/code/headings while keeping plain apostrophes and punctuation
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*•]\s+", "", cleaned)
    # Normalize common curly typography to clean ASCII equivalents
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": " - ",
        "\u2013": "-",
        "\u2026": "...",
    }
    for source_char, target_char in replacements.items():
        cleaned = cleaned.replace(source_char, target_char)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_call_arguments(arguments_text: str) -> List[str]:
    """Splits comma-separated arguments while preserving quoted strings and nested text."""
    arguments: List[str] = []
    current_chars: List[str] = []
    active_quote: Optional[str] = None
    escape_next = False
    paren_depth = 0

    for character in arguments_text:
        if escape_next:
            current_chars.append(character)
            escape_next = False
            continue
        if character == "\\" and active_quote is not None:
            escape_next = True
            continue
        if active_quote is not None:
            if character == active_quote:
                active_quote = None
            else:
                current_chars.append(character)
            continue
        if character in ('"', "'"):
            active_quote = character
            continue
        if character in ("(", "["):
            paren_depth += 1
        elif character in (")", "]") and paren_depth > 0:
            paren_depth -= 1
        if character == "," and paren_depth == 0:
            arguments.append("".join(current_chars).strip())
            current_chars = []
            continue
        current_chars.append(character)

    trailing = "".join(current_chars).strip()
    if trailing:
        arguments.append(trailing)
    return arguments


def _parse_float(value_text: str, default_value: float = 0.0) -> float:
    try:
        return float(value_text.strip().strip('"').strip("'"))
    except (ValueError, TypeError):
        return default_value


def _parse_string(value_text: str, default_value: str = "") -> str:
    stripped = value_text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ('"', "'"):
        return stripped[1:-1].strip()
    return stripped or default_value


def parse_command_line(
    line_text: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Optional[DrawingCommand]:
    """Parses one standalone drawing command line into a scaled DrawingCommand."""
    match = _CALL_LINE_PATTERN.match(line_text or "")
    if not match:
        return None

    command_name = match.group(1).lower()
    if command_name == "frame":
        command_name = "box"

    arguments = _split_call_arguments(match.group(2))
    if command_name == "scroll":
        if not arguments:
            return None
        direction_raw = _parse_string(arguments[0], "").lower()
        if direction_raw in ("down", "up"):
            scroll_ticks = int(max(1, min(20, _parse_float(arguments[1], 6.0) if len(arguments) >= 2 else 6.0)))
            return DrawingCommand(
                kind="scroll",
                x=0.0,
                y=0.0,
                width=float(scroll_ticks),
                label=direction_raw,
            )
        if len(arguments) >= 3:
            coord_x = _parse_float(arguments[0]) * scale_x
            coord_y = _parse_float(arguments[1]) * scale_y
            direction_str = _parse_string(arguments[2], "down").lower()
            if direction_str not in ("down", "up"):
                direction_str = "down"
            scroll_ticks = int(max(1, min(20, _parse_float(arguments[3], 6.0) if len(arguments) >= 4 else 6.0)))
            return DrawingCommand(
                kind="scroll",
                x=coord_x,
                y=coord_y,
                width=float(scroll_ticks),
                label=direction_str,
            )
        return None

    if command_name in ("open_browser", "open_url", "search_web", "open_app", "press_keys", "type_text"):
        primary_value = _parse_string(arguments[0], "").strip() if arguments else ""
        if command_name == "open_browser" and not primary_value:
            primary_value = "https://www.google.com"
        if not primary_value:
            return None
        secondary_label = _parse_string(arguments[1], primary_value) if len(arguments) >= 2 else primary_value
        return DrawingCommand(
            kind=command_name,
            label=secondary_label,
            body=primary_value,
        )

    if len(arguments) < 2:
        return None

    coord_x = _parse_float(arguments[0]) * scale_x
    coord_y = _parse_float(arguments[1]) * scale_y

    if command_name == "underline":
        width = max(24.0, _parse_float(arguments[2], 180.0) * scale_x) if len(arguments) >= 3 else 180.0
        label = _parse_string(arguments[3]) if len(arguments) >= 4 else ""
        tone = _parse_string(arguments[4], "blue").lower() if len(arguments) >= 5 else "blue"
        return DrawingCommand(kind="underline", x=coord_x, y=coord_y, width=width, label=label, tone=tone)

    if command_name == "ring":
        radius = max(14.0, _parse_float(arguments[2], 32.0) * ((scale_x + scale_y) * 0.5)) if len(arguments) >= 3 else 32.0
        label = _parse_string(arguments[3]) if len(arguments) >= 4 else ""
        tone = _parse_string(arguments[4], "blue").lower() if len(arguments) >= 5 else "blue"
        return DrawingCommand(kind="ring", x=coord_x, y=coord_y, radius=radius, label=label, tone=tone)

    if command_name == "point":
        label = _parse_string(arguments[2]) if len(arguments) >= 3 else ""
        tone = _parse_string(arguments[3], "blue").lower() if len(arguments) >= 4 else "blue"
        return DrawingCommand(kind="point", x=coord_x, y=coord_y, label=label, tone=tone)

    if command_name in ("box", "highlight"):
        width = max(24.0, _parse_float(arguments[2], 160.0) * scale_x) if len(arguments) >= 3 else 160.0
        height = max(20.0, _parse_float(arguments[3], 48.0) * scale_y) if len(arguments) >= 4 else 48.0
        label = _parse_string(arguments[4]) if len(arguments) >= 5 else ""
        default_tone = "yellow" if command_name == "highlight" else "blue"
        tone = _parse_string(arguments[5], default_tone).lower() if len(arguments) >= 6 else default_tone
        return DrawingCommand(
            kind=command_name,
            x=coord_x,
            y=coord_y,
            width=width,
            height=height,
            label=label,
            tone=tone,
        )

    if command_name == "paper":
        width = max(200.0, _parse_float(arguments[2], 330.0) * scale_x) if len(arguments) >= 3 else 330.0
        height = max(180.0, _parse_float(arguments[3], 360.0) * scale_y) if len(arguments) >= 4 else 360.0
        title = _parse_string(arguments[4], "Notes") if len(arguments) >= 5 else "Notes"
        header = _parse_string(arguments[5], "") if len(arguments) >= 6 else ""
        body = _parse_string(arguments[6], "") if len(arguments) >= 7 else ""
        tone = _parse_string(arguments[7], "blue").lower() if len(arguments) >= 8 else "blue"
        return DrawingCommand(
            kind="paper",
            x=coord_x,
            y=coord_y,
            width=width,
            height=height,
            label=title,
            header=header,
            body=body,
            tone=tone,
        )

    return None


def _extract_inline_point_tags(
    text: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[str, List[DrawingCommand]]:
    """Extracts `[POINT:x,y:label]` tags from text and returns cleaned text + commands."""
    commands: List[DrawingCommand] = []

    def _collect(tag_match: re.Match) -> str:
        point_x = _parse_float(tag_match.group(1)) * scale_x
        point_y = _parse_float(tag_match.group(2)) * scale_y
        label = (tag_match.group(3) or "").strip()
        commands.append(DrawingCommand(kind="point", x=point_x, y=point_y, label=label, tone="blue"))
        return ""

    stripped_text = _POINT_TAG_PATTERN.sub(_collect, text)
    return stripped_text, commands


def _find_last_sentence_boundary(text: str) -> int:
    """Returns index of the last `.`/`!`/`?` followed by whitespace (ignoring decimals), or -1."""
    length = len(text)
    for index in range(length - 2, -1, -1):
        character = text[index]
        if character in (".", "!", "?") and text[index + 1].isspace():
            previous_char = text[index - 1] if index > 0 else ""
            following_index = index + 1
            while following_index < length and text[following_index].isspace():
                following_index += 1
            if character == "." and previous_char.isdigit() and following_index < length and text[following_index].isdigit():
                continue
            return index
    return -1


def _split_complete_sentences(text: str) -> Tuple[List[str], str]:
    """
    Splits `text` into complete sentences suitable for immediate TTS synthesis,
    returning `(completed_sentences, trailing_remainder)`.
    Avoids breaking on version numbers or single-letter abbreviations (e.g. `1.1.5`).
    """
    completed: List[str] = []
    search_start = 0
    length = len(text)
    index = 0

    while index < length:
        character = text[index]
        if character in (".", "!", "?"):
            next_char = text[index + 1] if index + 1 < length else ""
            previous_char = text[index - 1] if index > 0 else ""
            # Only treat period as sentence end if followed by whitespace and not inside a decimal/version
            if next_char and next_char.isspace():
                following_index = index + 1
                while following_index < length and text[following_index].isspace():
                    following_index += 1
                if character == "." and previous_char.isdigit() and following_index < length and text[following_index].isdigit():
                    index += 1
                    continue
                candidate = text[search_start : index + 1].strip()
                if len(candidate) >= 2:
                    completed.append(candidate)
                    search_start = following_index
                    index = following_index
                    continue
        index += 1

    remainder = text[search_start:].strip()
    return completed, remainder


def _choose_primary_point_target(commands: List[DrawingCommand]) -> Optional[Tuple[float, float, str]]:
    for command in commands:
        if command.kind == "point":
            return (command.x, command.y, command.label)
    for command in commands:
        if command.kind in ("ring", "underline", "box", "highlight"):
            target_x = command.x + (command.width * 0.5 if command.width > 0 else 0.0)
            target_y = command.y + (command.height * 0.5 if command.height > 0 else 0.0)
            return (target_x, target_y, command.label)
    return None


def _extract_scroll_intent(commands: List[DrawingCommand]) -> Tuple[Optional[str], int]:
    """Return `(scroll_direction, scroll_amount)` if the model emitted a `scroll(...)` command."""
    for command in commands:
        if command.kind == "scroll":
            direction = command.label.lower() if command.label else "down"
            if direction not in ("down", "up"):
                direction = "down"
            amount = int(max(1, min(20, round(command.width or 6.0))))
            return direction, amount
    return None, 0


def parse_model_response(
    raw_text: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> ParsedTurnResponse:
    """Parses a full model reply into clean spoken sentences and scaled DrawingCommands."""
    cleaned_raw = _THINK_BLOCK_PATTERN.sub("", raw_text or "")
    commands: List[DrawingCommand] = []
    spoken_lines: List[str] = []

    for raw_line in cleaned_raw.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("```"):
            continue
        parsed_cmd = parse_command_line(stripped_line, scale_x=scale_x, scale_y=scale_y)
        if parsed_cmd is not None:
            commands.append(parsed_cmd)
            continue
        line_without_tags, inline_commands = _extract_inline_point_tags(
            stripped_line, scale_x=scale_x, scale_y=scale_y
        )
        commands.extend(inline_commands)
        cleaned_line = clean_spoken_text(line_without_tags)
        if cleaned_line:
            spoken_lines.append(cleaned_line)

    full_spoken = " ".join(spoken_lines).strip()
    sentences, remainder = _split_complete_sentences(full_spoken)
    if remainder:
        sentences.append(remainder)

    scroll_direction, scroll_amount = _extract_scroll_intent(commands)
    return ParsedTurnResponse(
        spoken_text=full_spoken,
        sentences=sentences,
        commands=commands,
        primary_point_target=_choose_primary_point_target(commands),
        scroll_direction=scroll_direction,
        scroll_amount=scroll_amount,
    )


class StreamingSentenceExtractor:
    """
    Incremental parser fed by SSE token deltas.

    Emits completed spoken sentences and drawing commands as soon as each line or
    sentence finishes so TTS and screen ink can start before the LLM finishes streaming.
    """

    def __init__(self, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
        self._scale_x = scale_x
        self._scale_y = scale_y
        self._line_buffer = ""
        self._prose_buffer = ""
        self._inside_think_block = False
        self._think_tail = ""
        self.all_commands: List[DrawingCommand] = []
        self.all_sentences: List[str] = []
        self.spoken_text = ""

    def feed_delta(self, text_delta: str) -> Tuple[List[str], List[DrawingCommand]]:
        """Processes one streamed text delta and returns `(new_sentences, new_commands)`."""
        if not text_delta:
            return [], []

        new_sentences: List[str] = []
        new_commands: List[DrawingCommand] = []
        working_text = text_delta

        # Strip <think>...</think> blocks cleanly even when tags span chunk boundaries
        while working_text:
            if self._inside_think_block:
                combined_think = self._think_tail + working_text
                close_pos = combined_think.lower().find("</think>")
                if close_pos == -1:
                    self._think_tail = combined_think[-10:]
                    return new_sentences, new_commands
                working_text = combined_think[close_pos + len("</think>") :]
                self._think_tail = ""
                self._inside_think_block = False
                continue

            combined_raw = self._line_buffer + working_text
            open_pos = combined_raw.lower().find("<think>")
            if open_pos == -1:
                self._line_buffer = combined_raw
                working_text = ""
                break

            before_think = combined_raw[:open_pos]
            after_open = combined_raw[open_pos + len("<think>") :]
            close_pos = after_open.lower().find("</think>")
            if close_pos == -1:
                self._line_buffer = before_think
                self._think_tail = after_open[-10:]
                self._inside_think_block = True
                working_text = ""
                break

            # Complete <think>...</think> inside this buffer: remove it and continue scanning
            self._line_buffer = ""
            working_text = before_think + after_open[close_pos + len("</think>") :]

        # Process complete lines first so drawing commands never leak into prose
        while "\n" in self._line_buffer:
            line_text, remaining_after_newline = self._line_buffer.split("\n", 1)
            self._line_buffer = remaining_after_newline
            line_sentences, line_commands = self._consume_complete_line(line_text)
            new_sentences.extend(line_sentences)
            new_commands.extend(line_commands)

        # If the current incomplete line is clearly prose (not a command call or fence),
        # flush completed sentences mid-line so speech starts fast on single-paragraph replies.
        stripped_partial = self._line_buffer.lstrip()
        looks_like_command = bool(
            _COMMAND_PREFIX_PATTERN.match(stripped_partial)
            or stripped_partial.startswith("```")
            or stripped_partial.startswith("[POINT:")
        )
        if not looks_like_command and "[" not in stripped_partial:
            boundary_index = _find_last_sentence_boundary(self._line_buffer)
            if boundary_index != -1:
                ready_prefix = self._line_buffer[: boundary_index + 1]
                self._line_buffer = self._line_buffer[boundary_index + 1 :].lstrip()
                cleaned_prefix, inline_cmds = _extract_inline_point_tags(
                    ready_prefix, scale_x=self._scale_x, scale_y=self._scale_y
                )
                if inline_cmds:
                    new_commands.extend(inline_cmds)
                    self.all_commands.extend(inline_cmds)
                prose_prefix = clean_spoken_text(cleaned_prefix)
                if prose_prefix:
                    combined = f"{self._prose_buffer} {prose_prefix}".strip()
                    completed, remainder = _split_complete_sentences(combined)
                    for sentence in completed:
                        self._record_sentence(sentence, new_sentences)
                    self._prose_buffer = remainder

        return new_sentences, new_commands

    def flush_remaining(self) -> Tuple[List[str], List[DrawingCommand]]:
        """Flushes any final buffered line or trailing sentence when the stream closes."""
        new_sentences: List[str] = []
        new_commands: List[DrawingCommand] = []

        if self._line_buffer.strip():
            line_sentences, line_commands = self._consume_complete_line(self._line_buffer)
            new_sentences.extend(line_sentences)
            new_commands.extend(line_commands)
            self._line_buffer = ""

        if self._prose_buffer.strip():
            completed, remainder = _split_complete_sentences(self._prose_buffer.strip())
            for sentence in completed:
                self._record_sentence(sentence, new_sentences)
            if remainder:
                self._record_sentence(remainder, new_sentences)
            self._prose_buffer = ""

        return new_sentences, new_commands

    def current_display_text(self) -> str:
        """Returns all non-command spoken text received so far (including the unfinished sentence)."""
        parts: List[str] = []
        if self.spoken_text:
            parts.append(self.spoken_text)
        if self._prose_buffer:
            parts.append(self._prose_buffer.strip())
        partial_line = self._line_buffer.lstrip()
        looks_like_command = bool(
            _COMMAND_PREFIX_PATTERN.match(partial_line)
            or partial_line.startswith("```")
            or partial_line.startswith("[POINT:")
        )
        if partial_line and not looks_like_command:
            cleaned_partial = clean_spoken_text(_POINT_TAG_PATTERN.sub("", partial_line))
            if cleaned_partial:
                parts.append(cleaned_partial)
        return " ".join(parts).strip()

    def build_final_response(self) -> ParsedTurnResponse:
        scroll_direction, scroll_amount = _extract_scroll_intent(self.all_commands)
        return ParsedTurnResponse(
            spoken_text=self.spoken_text,
            sentences=list(self.all_sentences),
            commands=list(self.all_commands),
            primary_point_target=_choose_primary_point_target(self.all_commands),
            scroll_direction=scroll_direction,
            scroll_amount=scroll_amount,
        )

    def _consume_complete_line(self, raw_line: str) -> Tuple[List[str], List[DrawingCommand]]:
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("```"):
            return [], []

        parsed_cmd = parse_command_line(stripped_line, scale_x=self._scale_x, scale_y=self._scale_y)
        if parsed_cmd is not None:
            self.all_commands.append(parsed_cmd)
            return [], [parsed_cmd]

        line_without_tags, inline_commands = _extract_inline_point_tags(
            stripped_line, scale_x=self._scale_x, scale_y=self._scale_y
        )
        if inline_commands:
            self.all_commands.extend(inline_commands)

        cleaned_line = clean_spoken_text(line_without_tags)
        new_sentences: List[str] = []
        if cleaned_line:
            combined = f"{self._prose_buffer} {cleaned_line}".strip()
            completed, remainder = _split_complete_sentences(combined)
            for sentence in completed:
                self._record_sentence(sentence, new_sentences)
            self._prose_buffer = remainder
        return new_sentences, inline_commands

    def _record_sentence(self, sentence: str, output_list: List[str]) -> None:
        cleaned = sentence.strip()
        if not cleaned:
            return
        output_list.append(cleaned)
        self.all_sentences.append(cleaned)
        self.spoken_text = f"{self.spoken_text} {cleaned}".strip() if self.spoken_text else cleaned
