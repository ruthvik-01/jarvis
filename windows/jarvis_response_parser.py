"""
Vision system prompt, style-preference detector, and streaming response parser.

Separates natural spoken sentences (for real-time TTS streaming) from inline 2D
screen-drawing commands (`underline`, `ring`, `point`, `box`, `paper`, `highlight`,
and `[POINT:x,y:label]`) while scaling coordinates from screenshot space to screen space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Sequence, Tuple


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
    sentence_index: int = -1


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
    r"open_browser|open_url|search_web|open_app|press_keys|type_text|"
    r"open_side_by_side|window_action|set_clipboard|write_file|system_control|"
    r"run_shell_read|request_admin_permission)"
)

_CALL_LINE_PATTERN = re.compile(
    rf"^\s*{_COMMAND_NAMES_PATTERN}\s*\((.*)\)\s*;?\s*$",
    re.IGNORECASE,
)

_COMMAND_PREFIX_PATTERN = re.compile(
    rf"^[\s\-*\u2022\d.)>`]*(?:DRAW|CMD|COMMAND|ACTION|TOOL)?\s*:?\s*`*{_COMMAND_NAMES_PATTERN}\s*(?:\(|:)",
    re.IGNORECASE,
)

_INLINE_COMMAND_START_PATTERN = re.compile(
    rf"\b{_COMMAND_NAMES_PATTERN}\s*\(",
    re.IGNORECASE,
)

_COLON_CALL_LINE_PATTERN = re.compile(
    rf"^\s*{_COMMAND_NAMES_PATTERN}\s*:\s*(.+?)\s*;?\s*$",
    re.IGNORECASE,
)

_THINK_BLOCK_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

_ORDINAL_NUMBER_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_CARDINAL_WORD_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_MATCH_STOP_WORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "into",
    "are",
    "was",
    "were",
    "can",
    "will",
    "have",
    "has",
    "had",
    "not",
    "but",
    "all",
    "out",
    "you",
    "your",
    "here",
    "there",
    "they",
    "them",
    "their",
    "what",
    "when",
    "where",
    "which",
    "how",
    "why",
    "uses",
    "used",
    "using",
    "helps",
    "help",
    "allows",
    "allow",
    "make",
    "makes",
    "better",
    "about",
    "each",
    "more",
    "very",
    "just",
    "only",
    "also",
    "such",
    "point",
    "step",
    "item",
    "number",
    "section",
    "next",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "work",
    "working",
    "employee",
    "employees",
    "company",
    "people",
}


def _extract_point_number_and_keywords(raw_text: str) -> Tuple[Optional[int], List[str]]:
    """Extract an optional explicit point/step index and discriminative keywords from text."""
    text_value = (raw_text or "").strip()
    if not text_value:
        return None, []

    lowered = text_value.lower()
    detected_number: Optional[int] = None

    leading_digit_match = re.match(r"^\s*(?:point|step|number|item|trend|#)?\s*(\d{1,2})\s*(?:[.):\-]|$)", lowered)
    if leading_digit_match:
        candidate_num = int(leading_digit_match.group(1))
        if 1 <= candidate_num <= 20:
            detected_number = candidate_num
    else:
        explicit_ref_match = re.search(
            r"\b(?:point|step|number|item|trend)\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            lowered,
        )
        if explicit_ref_match:
            ref_token = explicit_ref_match.group(1)
            if ref_token.isdigit():
                candidate_num = int(ref_token)
                if 1 <= candidate_num <= 20:
                    detected_number = candidate_num
            elif ref_token in _CARDINAL_WORD_MAP:
                detected_number = _CARDINAL_WORD_MAP[ref_token]
        else:
            ordinal_match = re.match(
                r"^(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
                lowered,
            )
            if not ordinal_match:
                ordinal_match = re.search(
                    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:point|one|trend|step|item|activity|feature|method|advantage|concept|principle)\b",
                    lowered,
                )
            if ordinal_match:
                detected_number = _ORDINAL_NUMBER_MAP[ordinal_match.group(1)]

    keywords: List[str] = []
    seen_keywords = set()
    for raw_token in re.findall(r"[A-Za-z0-9]+", text_value):
        token_lower = raw_token.lower()
        is_short_acronym = raw_token.isupper() and len(raw_token) >= 2
        if (len(token_lower) >= 3 or is_short_acronym) and token_lower not in _MATCH_STOP_WORDS:
            if token_lower not in seen_keywords:
                seen_keywords.add(token_lower)
                keywords.append(token_lower)

    return detected_number, keywords


def _extract_leading_phrase(command_label: str) -> str:
    """Return the title/phrase portion of an underline label without numbering or colon descriptions."""
    stripped = re.sub(r"^\s*(?:point|step|item|#)?\s*\d{1,2}\s*[.):\-]?\s*", "", (command_label or "").strip())
    if ":" in stripped:
        stripped = stripped.split(":", 1)[0].strip()
    stripped = re.sub(r"\([^)]*\)", "", stripped).strip()
    return stripped.lower()


def command_label_matches_sentence(command_label: str, sentence_text: str) -> int:
    """
    Score how strongly a drawing command's label matches a spoken sentence.

    Returns > 0 when the sentence discusses the label, 0 when ambiguous/unlabeled,
    and < 0 when the label belongs to a different explicit point number than the sentence.
    """
    clean_label = (command_label or "").strip()
    clean_sentence = (sentence_text or "").strip()
    if not clean_label or not clean_sentence:
        return 0

    label_num, label_keywords = _extract_point_number_and_keywords(clean_label)
    sentence_num, sentence_keywords = _extract_point_number_and_keywords(clean_sentence)
    normalized_sentence = clean_sentence.lower()

    shared_count = 0
    for kw_token in label_keywords:
        if kw_token in sentence_keywords or re.search(rf"\b{re.escape(kw_token)}\b", normalized_sentence):
            shared_count += 1
        elif len(kw_token) >= 5 and any(
            cand.startswith(kw_token[:5]) or kw_token.startswith(cand[:5])
            for cand in sentence_keywords
            if len(cand) >= 5
        ):
            shared_count += 1

    score = shared_count * 4
    core_phrase = _extract_leading_phrase(clean_label)
    if len(core_phrase) >= 5 and core_phrase in normalized_sentence:
        score += 6

    if label_num is not None and sentence_num is not None:
        if label_num == sentence_num:
            score += 5
        elif shared_count == 0:
            return -6
        else:
            score -= 3

    return score


def match_command_to_best_sentence_index(
    command: DrawingCommand,
    sentences: Sequence[str],
    fallback_index: int = 0,
    min_score: int = 4,
) -> int:
    """Return the index in `sentences` that best matches `command.label`, or `fallback_index`."""
    if not sentences or not (command.label or "").strip():
        return fallback_index
    best_index = fallback_index
    best_score = min_score - 1
    for candidate_idx, candidate_sentence in enumerate(sentences):
        candidate_score = command_label_matches_sentence(command.label, candidate_sentence)
        if candidate_score > best_score:
            best_score = candidate_score
            best_index = candidate_idx
    return best_index


def extract_key_topic_phrase(raw_text: str, max_words: int = 3) -> str:
    """Distills prose or a phrase into a clean 1-to-3 word diagram topic label."""
    if not raw_text:
        return "Core Concept"
    cleaned = clean_spoken_text(raw_text)
    if not cleaned:
        return "Core Concept"

    stop_words = {
        "the",
        "this",
        "that",
        "here",
        "we",
        "see",
        "can",
        "is",
        "are",
        "was",
        "were",
        "means",
        "tells",
        "us",
        "what",
        "how",
        "why",
        "when",
        "where",
        "in",
        "simple",
        "words",
        "imagine",
        "you",
        "start",
        "a",
        "an",
        "of",
        "to",
        "for",
        "and",
        "or",
        "with",
        "by",
        "on",
        "at",
        "from",
        "as",
        "it",
        "they",
        "first",
        "second",
        "third",
        "finally",
        "also",
        "basically",
        "essentially",
        "just",
        "simply",
        "really",
        "about",
        "into",
        "over",
        "under",
        "using",
        "shows",
        "explains",
        "describes",
    }

    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-+/]*", cleaned)
    filtered_tokens: List[str] = []
    for token in raw_tokens:
        lower_token = token.lower().strip("'-")
        if not lower_token or lower_token in stop_words:
            continue
        if token.isupper() and len(token) >= 2:
            formatted_token = token
        elif any(char.isupper() for char in token[1:]) and token[0].isupper():
            formatted_token = token
        else:
            formatted_token = token[:1].upper() + token[1:]
        if formatted_token not in filtered_tokens:
            filtered_tokens.append(formatted_token)
        if len(filtered_tokens) >= max(1, max_words):
            break

    if not filtered_tokens:
        fallback_tokens = [tok[:1].upper() + tok[1:] for tok in raw_tokens[: max(1, max_words)] if tok]
        return " ".join(fallback_tokens) if fallback_tokens else "Core Concept"
    return " ".join(filtered_tokens)


def should_use_visual_explanation_notes(
    user_prompt: str,
    has_lasso_crop: bool = False,
    is_scroll_session: bool = False,
    has_agent_commands: bool = False,
) -> bool:
    """
    Decides whether a turn warrants an overlay diagram card (`paper`) and screen underlines,
    suppressing visual notes on direct utility, telemetry, live lookup, or admin-gate turns.
    """
    if has_agent_commands:
        return False
    if has_lasso_crop or is_scroll_session:
        return True

    normalized = re.sub(r"\s+", " ", (user_prompt or "").lower().strip())
    if not normalized:
        return False

    utility_or_control_prefixes = (
        "open ",
        "launch ",
        "start ",
        "go to ",
        "search ",
        "google ",
        "type ",
        "write ",
        "press ",
        "close ",
        "delete ",
        "remove ",
        "uninstall ",
        "shut down",
        "shutdown",
        "restart ",
        "reboot",
    )
    if normalized.startswith(utility_or_control_prefixes):
        return False

    non_visual_phrases = (
        "storage left",
        "disk space",
        "drive space",
        "free space",
        "memory usage",
        "battery",
        "stock price",
        "share price",
        "price of",
        "bitcoin",
        "crypto",
        "weather",
        "what time",
        "shut down",
        "shutdown",
        "hey jarvis",
    )
    if any(phrase in normalized for phrase in non_visual_phrases):
        return False

    non_visual_word_pattern = re.compile(
        r"\b(?:storage|ram|cpu|gpu|delete|remove|uninstall|restart|reboot|"
        r"approve|confirm|cancel|stop|yes|no|hello|hi|thanks|thank you)\b"
    )
    if non_visual_word_pattern.search(normalized):
        return False

    return len(normalized) >= 3


def build_vision_system_prompt(
    screen_width: int,
    screen_height: int,
    style_notes: Optional[List[str]] = None,
    custom_words: Optional[List[str]] = None,
    has_lasso_crop: bool = False,
    scroll_step_index: int = 0,
    max_scroll_steps: int = 0,
    drawing_mode: str = "lines_and_card",
    agent_actions_enabled: bool = True,
    memory_context: str = "",
    live_screen_context: str = "",
    suppress_visual_notes: bool = False,
    system_telemetry_summary: str = "",
) -> str:
    """
    Constructs the system prompt for Jarvis's single-brain screen companion.

    Screenshots include 1:1 logical coordinate rulers along top and left edges
    (`0..screen_width` x `0..screen_height`) so drawing coordinates align accurately
    with visible text, equations, ChatGPT sections, and buttons.
    """
    normalized_mode = (drawing_mode or "lines_and_card").strip().lower()
    prompt_sections: List[str] = [
        "You are J.A.R.V.I.S., an intelligent, proactive real-life Iron Man desktop AI and visual tutor on Windows.",
        "AUTONOMOUS EXECUTION RULE: NEVER tell the user to open Settings, click menus, check disk storage themselves, open tabs themselves, or drag/snap windows side by side themselves. YOU must execute actions yourself using commands (`open_side_by_side`, `open_app(\"settings:storage\")`, `search_web`, `open_url`, `press_keys`, etc.) and report exact live facts directly.",
        f"The attached screenshot is {screen_width}x{screen_height} pixels (1:1 logical screen coordinates, top-left is 0,0).",
        "A coordinate ruler runs along the top and left edges (ticks every 100px, faint grid every 200px). Read exact X and Y pixel values from those rulers so every underline lands directly under the intended text without drifting into whitespace.",
        "Keep spoken replies warm, decisive, crisp, and easy to follow (2 to 5 natural sentences).",
        "Always inspect the current screenshot freshly; never repeat steps or sections you already explained earlier.",
        "CRITICAL FORMATTING & VOICE-SYNCED DRAWING RULES:",
        "1. VOICE-SYNCHRONIZED TIMING: Put each visual command (`paper(...)` or `underline(...)`) on its OWN standalone line IMMEDIATELY BEFORE the spoken sentence that explains that exact line or concept! If you open with an introductory sentence before Point 1, emit Point 1's underline RIGHT BEFORE Point 1's sentence (NOT before the intro sentence). ALWAYS pass the exact visible numbered title/phrase as the 4th argument of `underline(x, y, width, \"exact point title\")` so Jarvis locks the stroke to the matching sentence.",
        "2. Example synced sequence (MULTI-NOTE + LINE SYNCED):\npaper(24, 92, 310, 270, \"Overview\", \"Visual Map\", \"Input -> Process -> Output | section:Core | doc:Key Goal\")\nIntroductory sentence summarizing the main question cleanly before point 1.\nunderline(420, 210, 360, \"1. First Point Title\")\nSpoken sentence explaining point 1 right after its underline command.\npaper(1180, 92, 310, 270, \"Details\", \"Flow Step\", \"Stage 1 -> Stage 2 | doc:Result\")\nunderline(420, 245, 340, \"2. Second Point Title\")\nSpoken sentence explaining point 2 right after its underline command.",
        "3. NO markdown bullets (- or *), NO numbering, NO backticks, NO trailing arrow comments on command lines.",
        "4. NEVER speak or write command names like 'underline', 'paper', 'point', 'ring', 'box', 'open_side_by_side', or coordinates inside your spoken sentences.",
        "5. Never use markdown asterisks (**bold** or *italic*), bullet symbols, emojis, or code blocks in your spoken text.",
        "",
        "VISUAL EXPLANATION & SCREEN DRAWING COMMANDS:",
    ]

    if suppress_visual_notes:
        prompt_sections.extend(
            [
                "DIRECT CONTEXTUAL ANSWER MODE: Do NOT emit underline(...) commands unless pointing out something visual; focus on direct action commands or 1 concise summary note.",
                "Answer the user's question directly and concisely in 1 to 3 spoken sentences using the live screen, live Windows telemetry, and fetched context.",
            ]
        )
    elif normalized_mode == "cursor_only":
        prompt_sections.extend(
            [
                "Cursor-only mode is enabled: do NOT emit underline, ring, box, highlight, or paper commands.",
                f"Optionally emit up to 2 point(x, y, \"\") commands (x, y in 0..{screen_width} x 0..{screen_height}) on their own lines right before the sentence referencing them.",
            ]
        )
    elif normalized_mode in ("precise_lines", "lines_and_card", "full_sketch"):
        prompt_sections.extend(
            [
                "AGGRESSIVE MULTI-NOTE VISUALIZATION: Whenever explaining concepts, code, equations, PDFs, or ChatGPT answers, emit 2 to 3 visual topic-flow `paper(...)` diagram cards placed in empty margin corners (`top-left (24,92)`, `top-right (1180,92)`, `bottom-left (24,380)`) so the user can grasp your entire spoken explanation visually at a glance!",
                "Each `paper(...)` card MUST contain concise 1 to 4 word diagram flow nodes separated by `->`, `|`, and `section:`/`doc:` dividers — NEVER prose sentences or paragraphs:",
                "Card 1 (before intro/sentence 1): paper(24, 92, 310, 270, \"Topic Map\", \"Part 1\", \"Node A -> Node B | section:Mechanism | doc:Core Idea\")",
                "Card 2 (before middle section): paper(1180, 92, 310, 270, \"Breakdown\", \"Part 2\", \"Step 1 -> Step 2 | doc:Outcome\")",
                "Card 3 (optional for 4+ points): paper(24, 382, 310, 250, \"Takeaway\", \"Summary\", \"Benefit -> Impact | doc:Conclusion\")",
                f"Also emit 2 to 5 crisp horizontal underlines, each right BEFORE the spoken sentence discussing that line (y = text baseline + 4px, x = exact left edge of the text word on the page, width = exact phrase pixel width inside the page text — NEVER into empty margins — in 0..{screen_width} x 0..{screen_height}, 4th arg = exact 2 to 6 words on screen):",
                "underline(420, 310, 260, \"exact phrase on screen\")",
                "Optionally glide the cursor subtly without a label using `point(420, 310, \"\")` — NEVER put text labels inside `point(...)`.",
                "STRICTLY FORBIDDEN: Do NOT emit `ring(...)`, `box(...)`, `frame(...)`, `highlight(...)`, or labeled `point(x, y, \"label\")` callouts. Always use crisp `underline(...)` strokes + `paper(...)` diagram cards.",
            ]
        )

    if agent_actions_enabled:
        prompt_sections.extend(
            [
                "",
                "UNIVERSAL DESKTOP & BROWSER AGENT COMMANDS:",
                "You can execute ANY arbitrary user desktop, app, window, browser, file, clipboard, media, or read-only system request by composing these primitives — even if the user asks for something completely new!",
                "NEVER ask the user to perform a desktop action themselves (like opening an app, running a command, copying text, tiling windows, or checking settings). Execute it directly with these commands and report exact facts:",
                "NEVER say 'I will type/search/open it after it loads' or 'you can drag tabs side by side' — Jarvis executes everything automatically!",
                "For ANY destructive or admin-level action (deleting files, uninstalling apps, shutting down/restarting, killing processes, editing registry/security settings, force-closing unsaved apps), NEVER execute destructively — emit `request_admin_permission(\"action title\", \"risk reason\")` first and wait for admin confirmation.",
                "Available agent commands (emit one or more, each on its own standalone line BEFORE speaking):",
                "open_app(\"app name, Start Menu app, system tool like taskmgr/control/snippingtool/devmgmt.msc, or settings:<topic>\")",
                "open_url(\"https://...\")",
                "search_web(\"search query\")",
                "open_side_by_side(\"left window title/app or site/search\", \"right window title/app or site/search\")  -> tiles ANY two open desktop windows or Chrome tabs/sites left & right (or \"\", \"\" to tile existing Chrome tabs)",
                "window_action(\"focus|minimize|maximize|restore|snap_left|snap_right|close|minimize_all\", \"window title or app name\")",
                "type_text(\"text to type into focused field\", \"submit\")",
                "press_keys(\"shortcut like ctrl+c, ctrl+t, alt+tab, win+e\")",
                "set_clipboard(\"text to copy to clipboard\")",
                "write_file(\"Desktop/notes.md\", \"file content to save safely under Desktop or Documents\")",
                "system_control(\"mute|unmute|volume_up|volume_down|play_pause|next_track|prev_track|show_desktop|screenshot\")",
                "run_shell_read(\"non-destructive PowerShell query (e.g. Get-Process | Sort CPU -Desc | Select -First 5 Name,CPU)\", \"Top CPU processes\")",
                "request_admin_permission(\"action title\", \"risk explanation\")",
                "Example when user says 'put VS Code and Edge side by side':",
                "open_side_by_side(\"Visual Studio Code\", \"Microsoft Edge\")",
                "Tiling VS Code and Edge side by side now.",
            ]
        )

    if has_lasso_crop:
        prompt_sections.extend(
            [
                "",
                "LASSO HIGHLIGHT FOCUS:",
                "The user drew a lasso around a specific target region, and a 2x zoomed close-up crop is attached as the second image.",
                "Focus your explanation directly on the content inside the lasso and align any underline or point to its exact coordinates on the main screenshot.",
            ]
        )

    if max_scroll_steps > 0:
        prompt_sections.extend(
            [
                "",
                f"SCROLL-AND-EXPLAIN MODE (Section {max(1, scroll_step_index)} of {max_scroll_steps}):",
                "You are walking the user visually through a scrollable page (such as a ChatGPT answer or article).",
                "Explain the section currently visible on screen with accurate, neatly placed underlines beneath key phrases.",
                "REQUIREMENT-BASED ADAPTIVE SCROLLING: Only emit `scroll(\"down\", ticks)` on its own final line when more unread content is clearly cut off below the visible bottom of this pane AND more scroll steps remain.",
                "Choose `ticks` dynamically based on how much needs to move: use `2` or `3` for a small nudge (1-2 cut-off lines), `4` or `5` for half a paragraph/section, and `7` to `10` for a full screen page.",
                "If all key points and the conclusion/ending are already visible on screen (or the input bar sits right below the last line), do NOT emit `scroll(...)`.",
            ]
        )
    else:
        prompt_sections.extend(
            [
                "",
                "OPTIONAL ADAPTIVE SCROLL COMMAND:",
                "- scroll(\"down\", ticks) -> only emit this on its own final line when the user asked to scroll OR unread content continues below the visible pane. Pick `ticks` proportionally (`2..3` = small scroll, `4..6` = medium section, `7..10` = full page). Omit `scroll(...)` completely if everything needed is already on screen.",
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

    if live_screen_context and live_screen_context.strip():
        prompt_sections.append("")
        prompt_sections.append("LIVE SCREEN MONITOR STATE:")
        prompt_sections.append(live_screen_context.strip())

    if system_telemetry_summary and system_telemetry_summary.strip():
        prompt_sections.append("")
        prompt_sections.append("LIVE WINDOWS SYSTEM TELEMETRY (USE EXACT NUMBERS):")
        prompt_sections.append(system_telemetry_summary.strip())

    if memory_context and memory_context.strip():
        prompt_sections.append("")
        prompt_sections.append("PERSISTENT JARVIS SESSION MEMORY:")
        prompt_sections.append(memory_context.strip())

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


def _find_matching_paren_end(raw_text: str, open_paren_index: int) -> int:
    """Returns the index of the closing `)` balancing `open_paren_index`, or -1 if unclosed."""
    active_quote: Optional[str] = None
    escape_next = False
    paren_depth = 0
    text_length = len(raw_text)

    for char_index in range(open_paren_index, text_length):
        character = raw_text[char_index]
        if escape_next:
            escape_next = False
            continue
        if character == "\\" and active_quote is not None:
            escape_next = True
            continue
        if active_quote is not None:
            if character == active_quote:
                active_quote = None
            continue
        if character in ('"', "'"):
            active_quote = character
            continue
        if character == "(":
            paren_depth += 1
        elif character == ")":
            paren_depth -= 1
            if paren_depth == 0:
                return char_index
    return -1


def _strip_command_line_noise(raw_line: str) -> str:
    """Strips markdown bullets, prefixes, brackets, and backticks around a standalone command."""
    cleaned = (raw_line or "").strip()
    cleaned = re.sub(r"^[\s\-*\u2022\d.)>`]+", "", cleaned).strip()
    cleaned = re.sub(r"^(?:DRAW|CMD|COMMAND|ACTION|TOOL)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned.startswith("`") and cleaned.endswith("`") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("[") and cleaned.endswith("]") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.strip("`").strip()
    return cleaned


def _extract_inline_commands(
    raw_text: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[str, List[DrawingCommand]]:
    """Extracts any inline command calls (`underline(...)`, `paper(...)`, etc.) embedded in prose."""
    if not raw_text:
        return "", []
    text_without_tags, tag_commands = _extract_inline_point_tags(raw_text, scale_x=scale_x, scale_y=scale_y)
    extracted_commands: List[DrawingCommand] = list(tag_commands)

    kept_fragments: List[str] = []
    scan_index = 0
    text_length = len(text_without_tags)
    while scan_index < text_length:
        inline_match = _INLINE_COMMAND_START_PATTERN.search(text_without_tags, scan_index)
        if not inline_match:
            kept_fragments.append(text_without_tags[scan_index:])
            break

        open_paren_index = text_without_tags.find("(", inline_match.start())
        close_paren_index = _find_matching_paren_end(text_without_tags, open_paren_index)
        if close_paren_index == -1:
            kept_fragments.append(text_without_tags[scan_index:])
            break

        candidate_call = text_without_tags[inline_match.start() : close_paren_index + 1]
        parsed_command = parse_command_line(candidate_call, scale_x=scale_x, scale_y=scale_y)
        if parsed_command is not None:
            prefix_slice = text_without_tags[scan_index : inline_match.start()]
            prefix_slice = re.sub(r"[\[`\s\-*\u2022>]+$", " ", prefix_slice)
            prefix_slice = re.sub(r"(?:DRAW|CMD|COMMAND|ACTION|TOOL)\s*:\s*$", "", prefix_slice, flags=re.IGNORECASE)
            kept_fragments.append(prefix_slice)
            extracted_commands.append(parsed_command)
            scan_index = close_paren_index + 1
            while scan_index < text_length and text_without_tags[scan_index] in ("]", "`", ";"):
                scan_index += 1
        else:
            kept_fragments.append(text_without_tags[scan_index : close_paren_index + 1])
            scan_index = close_paren_index + 1

    cleaned_remaining = "".join(kept_fragments)
    # Also remove any leaked colon-form commands like `underline: 120, 200, 300`
    cleaned_remaining = re.sub(
        rf"(?i)(?:^|\s){_COMMAND_NAMES_PATTERN}\s*:\s*[\d.,\s\"'a-zA-Z_-]+?(?=$|\s+[A-Z]|\s*\n)",
        " ",
        cleaned_remaining,
    )
    return cleaned_remaining, extracted_commands


def _has_unclosed_command(partial_text: str) -> bool:
    """True when `partial_text` ends inside an unclosed command call like `underline(420, 3`."""
    if not partial_text:
        return False
    last_match: Optional[re.Match] = None
    for candidate_match in _INLINE_COMMAND_START_PATTERN.finditer(partial_text):
        last_match = candidate_match
    if last_match is None:
        return False
    open_paren_index = partial_text.find("(", last_match.start())
    return _find_matching_paren_end(partial_text, open_paren_index) == -1


def clean_spoken_text(raw_text: str) -> str:
    """Strips markdown noise, thinking tags, and leftover drawing syntax from spoken text."""
    if not raw_text:
        return ""
    cleaned = _THINK_BLOCK_PATTERN.sub(" ", raw_text)
    cleaned = _POINT_TAG_PATTERN.sub("", cleaned)
    cleaned, _unused_commands = _extract_inline_commands(cleaned, scale_x=1.0, scale_y=1.0)
    # Strip any trailing truncated command call (`underline(120, 300...` without closing paren)
    cleaned = re.sub(rf"(?is)\b{_COMMAND_NAMES_PATTERN}\s*\([^)]*$", "", cleaned)
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    # Strip markdown bold/italic/code/headings while keeping plain apostrophes and punctuation
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*•]\s+", "", cleaned)
    # Prevent stray leaked command labels from being spoken aloud
    cleaned = re.sub(
        r"(?im)^\s*(?:underline|paper|point|ring|box|highlight|draw|cmd|command|request_admin_permission)\s*[:\-]\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)^\s*(?:underline|paper)\b[\s,:\-]*$", "", cleaned)
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
    normalized_line = _strip_command_line_noise(line_text)
    match = _CALL_LINE_PATTERN.match(normalized_line)
    if not match:
        prefix_call_match = re.match(rf"^\s*{_COMMAND_NAMES_PATTERN}\s*\(", normalized_line, re.IGNORECASE)
        if prefix_call_match:
            open_paren_index = normalized_line.find("(", prefix_call_match.start())
            close_paren_index = _find_matching_paren_end(normalized_line, open_paren_index)
            if close_paren_index != -1:
                trailing_after_call = normalized_line[close_paren_index + 1 :].strip()
                if (
                    not trailing_after_call
                    or trailing_after_call.startswith((";", "`", "]", "->", "//", "#"))
                ):
                    normalized_line = normalized_line[: close_paren_index + 1]
                    match = _CALL_LINE_PATTERN.match(normalized_line)
    if not match:
        colon_match = _COLON_CALL_LINE_PATTERN.match(normalized_line)
        if colon_match:
            synthesized_call = f"{colon_match.group(1)}({colon_match.group(2)})"
            match = _CALL_LINE_PATTERN.match(synthesized_call)
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

    if command_name == "request_admin_permission":
        action_title = _parse_string(arguments[0], "").strip() if arguments else ""
        if not action_title:
            return None
        risk_reason = _parse_string(arguments[1], action_title).strip() if len(arguments) >= 2 else action_title
        return DrawingCommand(
            kind="request_admin_permission",
            label=action_title,
            body=risk_reason,
        )

    if command_name == "open_side_by_side":
        left_target = _parse_string(arguments[0], "").strip() if arguments else ""
        right_target = _parse_string(arguments[1], "").strip() if len(arguments) >= 2 else ""
        return DrawingCommand(
            kind="open_side_by_side",
            label=right_target or left_target or "side_by_side",
            body=left_target,
            header=right_target,
        )

    if command_name in ("window_action", "write_file", "system_control", "run_shell_read"):
        primary_arg = _parse_string(arguments[0], "").strip() if arguments else ""
        if not primary_arg:
            return None
        secondary_arg = _parse_string(arguments[1], "").strip() if len(arguments) >= 2 else ""
        return DrawingCommand(
            kind=command_name,
            label=secondary_arg or primary_arg,
            body=primary_arg,
            header=secondary_arg,
        )

    if command_name == "set_clipboard":
        clipboard_value = _parse_string(arguments[0], "").strip() if arguments else ""
        if not clipboard_value:
            return None
        return DrawingCommand(
            kind="set_clipboard",
            label="clipboard",
            body=clipboard_value,
        )

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


def align_commands_with_sentences(commands: Sequence[DrawingCommand], sentences: Sequence[str]) -> None:
    """Refine `command.sentence_index` against complete sentences using semantic label matching."""
    if not sentences:
        return
    last_matched_underline_index = -1
    for cmd_item in commands:
        if cmd_item.kind not in ("underline", "paper", "point", "ring", "box", "highlight"):
            continue
        if (cmd_item.label or "").strip():
            matched_idx = match_command_to_best_sentence_index(
                cmd_item,
                sentences,
                fallback_index=cmd_item.sentence_index if cmd_item.sentence_index >= 0 else 0,
                min_score=4,
            )
            if cmd_item.kind == "underline":
                if matched_idx < last_matched_underline_index and last_matched_underline_index < len(sentences):
                    best_score = command_label_matches_sentence(cmd_item.label, sentences[matched_idx])
                    if best_score < 8:
                        matched_idx = max(matched_idx, last_matched_underline_index)
                last_matched_underline_index = matched_idx
            cmd_item.sentence_index = min(max(0, matched_idx), max(0, len(sentences) - 1))


def parse_model_response(
    raw_text: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> ParsedTurnResponse:
    """Parses a full model reply into clean spoken sentences and scaled DrawingCommands."""
    cleaned_raw = _THINK_BLOCK_PATTERN.sub("", raw_text or "")
    commands: List[DrawingCommand] = []
    spoken_lines: List[str] = []
    sentences_completed_so_far: List[str] = []
    pending_prose_buffer = ""
    next_underline_sentence_index = 0

    def _assign_parsed_sentence_index(cmd_item: DrawingCommand) -> None:
        nonlocal next_underline_sentence_index
        if cmd_item.sentence_index >= 0:
            return
        matched_prior = match_command_to_best_sentence_index(
            cmd_item,
            sentences_completed_so_far,
            fallback_index=-1,
            min_score=4,
        )
        if matched_prior >= 0:
            cmd_item.sentence_index = matched_prior
            if cmd_item.kind == "underline":
                next_underline_sentence_index = max(next_underline_sentence_index, matched_prior + 1)
            return
        base_sentence_index = len(sentences_completed_so_far)
        if cmd_item.kind == "underline":
            assigned_index = max(base_sentence_index, next_underline_sentence_index)
            cmd_item.sentence_index = assigned_index
            next_underline_sentence_index = assigned_index + 1
        elif cmd_item.kind in ("paper", "point", "ring", "box", "highlight"):
            cmd_item.sentence_index = base_sentence_index

    for raw_line in cleaned_raw.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("```"):
            continue
        parsed_cmd = parse_command_line(stripped_line, scale_x=scale_x, scale_y=scale_y)
        if parsed_cmd is not None:
            _assign_parsed_sentence_index(parsed_cmd)
            commands.append(parsed_cmd)
            continue
        line_without_commands, inline_commands = _extract_inline_commands(
            stripped_line, scale_x=scale_x, scale_y=scale_y
        )
        for inline_cmd in inline_commands:
            _assign_parsed_sentence_index(inline_cmd)
            commands.append(inline_cmd)
        cleaned_line = clean_spoken_text(line_without_commands)
        if cleaned_line:
            spoken_lines.append(cleaned_line)
            combined_prose = f"{pending_prose_buffer} {cleaned_line}".strip()
            done_sentences, pending_prose_buffer = _split_complete_sentences(combined_prose)
            sentences_completed_so_far.extend(done_sentences)

    full_spoken = " ".join(spoken_lines).strip()
    sentences, remainder = _split_complete_sentences(full_spoken)
    if remainder:
        sentences.append(remainder)

    align_commands_with_sentences(commands, sentences)
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
        self._next_unassigned_underline_sentence_index = 0
        self.all_commands: List[DrawingCommand] = []
        self.all_sentences: List[str] = []
        self.spoken_text = ""

    def _assign_sentence_index_to_command(self, cmd_item: DrawingCommand) -> None:
        if cmd_item.sentence_index >= 0:
            return
        matched_prior = match_command_to_best_sentence_index(
            cmd_item,
            self.all_sentences,
            fallback_index=-1,
            min_score=4,
        )
        if matched_prior >= 0:
            cmd_item.sentence_index = matched_prior
            if cmd_item.kind == "underline":
                self._next_unassigned_underline_sentence_index = max(
                    self._next_unassigned_underline_sentence_index, matched_prior + 1
                )
            return
        base_sentence_index = len(self.all_sentences)
        if cmd_item.kind == "underline":
            assigned_index = max(base_sentence_index, self._next_unassigned_underline_sentence_index)
            cmd_item.sentence_index = assigned_index
            self._next_unassigned_underline_sentence_index = assigned_index + 1
        elif cmd_item.kind in ("paper", "point", "ring", "box", "highlight"):
            cmd_item.sentence_index = base_sentence_index

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

        # If the current incomplete line is prose (and not inside an unclosed command call),
        # flush completed sentences mid-line so speech starts fast on single-paragraph replies.
        stripped_partial = self._line_buffer.lstrip()
        looks_like_command = bool(
            _COMMAND_PREFIX_PATTERN.match(stripped_partial)
            or _has_unclosed_command(stripped_partial)
            or stripped_partial.startswith("```")
            or stripped_partial.startswith("[POINT:")
        )
        if not looks_like_command and "[" not in stripped_partial:
            boundary_index = _find_last_sentence_boundary(self._line_buffer)
            if boundary_index != -1 and not _has_unclosed_command(self._line_buffer[: boundary_index + 1]):
                ready_prefix = self._line_buffer[: boundary_index + 1]
                self._line_buffer = self._line_buffer[boundary_index + 1 :].lstrip()
                cleaned_prefix, inline_cmds = _extract_inline_commands(
                    ready_prefix, scale_x=self._scale_x, scale_y=self._scale_y
                )
                if inline_cmds:
                    for inline_cmd in inline_cmds:
                        self._assign_sentence_index_to_command(inline_cmd)
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
            or _has_unclosed_command(partial_line)
            or partial_line.startswith("```")
            or partial_line.startswith("[POINT:")
        )
        if partial_line and not looks_like_command:
            cleaned_partial = clean_spoken_text(_POINT_TAG_PATTERN.sub("", partial_line))
            if cleaned_partial:
                parts.append(cleaned_partial)
        return " ".join(parts).strip()

    def build_final_response(self) -> ParsedTurnResponse:
        align_commands_with_sentences(self.all_commands, self.all_sentences)
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
            self._assign_sentence_index_to_command(parsed_cmd)
            self.all_commands.append(parsed_cmd)
            return [], [parsed_cmd]

        line_without_commands, inline_commands = _extract_inline_commands(
            stripped_line, scale_x=self._scale_x, scale_y=self._scale_y
        )
        if inline_commands:
            for inline_cmd in inline_commands:
                self._assign_sentence_index_to_command(inline_cmd)
            self.all_commands.extend(inline_commands)

        cleaned_line = clean_spoken_text(line_without_commands)
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
