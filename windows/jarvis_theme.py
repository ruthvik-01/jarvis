"""
Shared design tokens and painting helpers for the Qt-based Jarvis Windows replica.

Every UI module (notch, cursor/drawing overlay, Jarvis window) imports from here so
colors, fonts and glow treatments stay identical across windows. Values come from the
real Jarvis 1.1.5 app (reference photos in `reference images/` and the reverse
engineered notes in Jarvis Windows architecture notes).
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
from typing import Dict, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QPainter,
    QPainterPath,
    QRadialGradient,
    QTransform,
)

WINDOWS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIRECTORY = os.path.join(WINDOWS_DIRECTORY, "assets")
APP_DISPLAY_VERSION = "1.1.5"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

JARVIS_BLUE_HEX = "#3380FF"

# The four cursor colors offered in Settings > Cursor (OnboardingCatalog.CursorColors).
CURSOR_COLOR_CHOICES: Dict[str, str] = {
    "Blue": "#3380FF",
    "Red": "#FF4C4C",
    "Gold": "#D4A017",
    "Green": "#26C281",
}

NOTCH_BLACK = QColor(0, 0, 0)
NOTCH_TEXT_PRIMARY = QColor(255, 255, 255)
NOTCH_TEXT_SECONDARY = QColor(255, 255, 255, 150)
NOTCH_HAIRLINE = QColor(255, 255, 255, 28)

# Activity strip glow colors seen in the reference photos: the right end of the
# Listening strip glows cyan, Thinking glows violet, Speaking glows orange.
ACTIVITY_GLOW_COLORS: Dict[str, tuple] = {
    "listening": (QColor("#2BC8FF"), QColor("#7FE6FF")),
    "dictating": (QColor("#2BC8FF"), QColor("#7FE6FF")),
    "thinking": (QColor("#9B5CFF"), QColor("#E28BFF")),
    "speaking": (QColor("#FF7A2F"), QColor("#FFB24A")),
    "status": (QColor("#8E8E93"), QColor("#C7C7CC")),
    "warning": (QColor("#FF9F0A"), QColor("#FFD60A")),
}

# Pure dark obsidian/carbon studio theme + mirrored light theme.
WINDOW_PALETTE_DARK: Dict[str, str] = {
    "background": "#09090B",
    "sidebar": "#0E0E11",
    "card": "#141418",
    "card_hover": "#1C1C22",
    "separator": "#24242C",
    "text_primary": "#F4F4F6",
    "text_secondary": "#A1A1AA",
    "text_tertiary": "#71717A",
    "accent": "#27272A",
    "accent_soft": "#E4E4E7",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "selection": "#1F1F26",
}
WINDOW_PALETTE_LIGHT: Dict[str, str] = {
    "background": "#F5F5F7",
    "sidebar": "#ECECEE",
    "card": "#FFFFFF",
    "card_hover": "#F2F2F4",
    "separator": "#DCDCE0",
    "text_primary": "#1D1D1F",
    "text_secondary": "#6E6E73",
    "text_tertiary": "#A1A1A6",
    "accent": "#3380FF",
    "accent_soft": "#2F6FE0",
    "danger": "#D93025",
    "selection": "#E2E2E6",
}

# Drawing palette (Jarvis draw engine PALETTE): sleek dark notes paper, crisp ink, semantic tones.
DRAWING_PAPER_CREAM = QColor("#121216")
DRAWING_PAPER_WARM = QColor("#121216")
DRAWING_PAPER_COOL = QColor("#121216")
DRAWING_PAPER_HEADER = QColor("#1A1A20")
DRAWING_GRID_LINE = QColor(255, 255, 255, 16)
DRAWING_INK = QColor("#F4F4F6")
DRAWING_TONES: Dict[str, QColor] = {
    "blue": QColor("#1E7BFF"),
    "red": QColor("#FF4D4F"),
    "green": QColor("#22B573"),
    "amber": QColor("#F5A623"),
    "purple": QColor("#8B5CF6"),
    "teal": QColor("#14B8A6"),
    "coral": QColor("#FF7F66"),
    "grey": QColor("#8E8E93"),
    "yellow": QColor("#FFD84D"),
}
SCREEN_MARKER_BLUE = QColor("#1F86FF")  # glowing underline color on the live screen

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

_REGISTERED_FONT_FAMILIES: Dict[str, str] = {}

FONT_UI_FAMILY = "Segoe UI Variable Text"
FONT_UI_FALLBACK_FAMILY = "Segoe UI"


def register_bundled_fonts() -> Dict[str, str]:
    """
    Registers the fonts shipped with the official app (assets/fonts) with Qt and returns
    a mapping of logical role -> family name. Must be called after QApplication exists.
    Roles: "hand" (PatrickHand handwriting), "barlow" (drawing labels).
    """
    if _REGISTERED_FONT_FAMILIES:
        return dict(_REGISTERED_FONT_FAMILIES)
    font_files_by_role = {
        "hand": "Jarvis-PatrickHand.ttf",
        "barlow": "Jarvis-Barlow-Medium.ttf",
        "barlow_semibold": "Jarvis-Barlow-SemiBold.ttf",
        "barlow_regular": "Jarvis-Barlow-Regular.ttf",
    }
    for role, file_name in font_files_by_role.items():
        font_path = os.path.join(ASSETS_DIRECTORY, "fonts", file_name)
        family_name = "Segoe Print" if role == "hand" else FONT_UI_FALLBACK_FAMILY
        if os.path.exists(font_path):
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id >= 0:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    family_name = families[0]
        _REGISTERED_FONT_FAMILIES[role] = family_name
    return dict(_REGISTERED_FONT_FAMILIES)


def ui_font(point_size: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """System UI font (Segoe UI Variable on Windows 11, Segoe UI on Windows 10) with HD antialiasing."""
    font = QFont(FONT_UI_FAMILY)
    font.setFamilies([FONT_UI_FAMILY, FONT_UI_FALLBACK_FAMILY])
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def hand_font(point_size: float) -> QFont:
    """PatrickHand handwriting font used on drawing paper cards and labels."""
    families = register_bundled_fonts()
    font = QFont(families.get("hand", "Segoe Print"))
    font.setPointSizeF(point_size)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def barlow_font(point_size: float, semibold: bool = False) -> QFont:
    families = register_bundled_fonts()
    font = QFont(families.get("barlow_semibold" if semibold else "barlow", FONT_UI_FALLBACK_FAMILY))
    font.setPointSizeF(point_size)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


# ---------------------------------------------------------------------------
# Shapes and glows
# ---------------------------------------------------------------------------

CURSOR_TRIANGLE_SIDE = 16.0
CURSOR_TRIANGLE_CORNER_FRACTION = 0.09
CURSOR_RESTING_ROTATION_DEGREES = -35.0
CURSOR_FOLLOW_OFFSET = QPointF(35.0, 25.0)


def build_rounded_triangle_path(
    side_length: float = CURSOR_TRIANGLE_SIDE,
    corner_fraction: float = CURSOR_TRIANGLE_CORNER_FRACTION,
    rotation_degrees: float = CURSOR_RESTING_ROTATION_DEGREES,
) -> QPainterPath:
    """
    Rounded equilateral triangle centered on (0, 0), apex pointing up before rotation.
    With the default -35 degree tilt the apex points up-and-left like the real cursor.
    """
    corner_radius = side_length * corner_fraction
    triangle_height = side_length * math.sqrt(3.0) / 2.0
    vertices = [
        QPointF(0.0, -triangle_height * 2.0 / 3.0),
        QPointF(side_length / 2.0, triangle_height / 3.0),
        QPointF(-side_length / 2.0, triangle_height / 3.0),
    ]
    path = QPainterPath()
    vertex_count = len(vertices)
    for index in range(vertex_count):
        previous_vertex = vertices[(index - 1) % vertex_count]
        current_vertex = vertices[index]
        next_vertex = vertices[(index + 1) % vertex_count]
        towards_previous = _move_towards(current_vertex, previous_vertex, corner_radius * 1.7)
        towards_next = _move_towards(current_vertex, next_vertex, corner_radius * 1.7)
        if index == 0:
            path.moveTo(towards_previous)
        else:
            path.lineTo(towards_previous)
        path.quadTo(current_vertex, towards_next)
    path.closeSubpath()
    return QTransform().rotate(rotation_degrees).map(path)


def _move_towards(start: QPointF, end: QPointF, distance: float) -> QPointF:
    delta_x = end.x() - start.x()
    delta_y = end.y() - start.y()
    length = math.hypot(delta_x, delta_y) or 1.0
    return QPointF(start.x() + delta_x / length * distance, start.y() + delta_y / length * distance)


def paint_soft_glow(
    painter: QPainter,
    center: QPointF,
    radius: float,
    color: QColor,
    peak_alpha: int = 150,
) -> None:
    """Radial glow that fades to transparent; used behind the cursor and on notch strips."""
    if radius <= 0.5:
        return
    gradient = QRadialGradient(center, radius)
    inner = QColor(color)
    inner.setAlpha(peak_alpha)
    middle = QColor(color)
    middle.setAlpha(int(peak_alpha * 0.35))
    outer = QColor(color)
    outer.setAlpha(0)
    gradient.setColorAt(0.0, inner)
    gradient.setColorAt(0.45, middle)
    gradient.setColorAt(1.0, outer)
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    painter.drawEllipse(center, radius, radius)
    painter.restore()


def with_alpha(color: QColor, alpha: int) -> QColor:
    copied_color = QColor(color)
    copied_color.setAlpha(max(0, min(255, int(alpha))))
    return copied_color


# ---------------------------------------------------------------------------
# Simple critically damped spring used for all UI motion
# ---------------------------------------------------------------------------


class Spring:
    """
    Damped spring toward a target value. `response` is roughly the settle time in seconds,
    `damping_ratio` 1.0 is critically damped (no overshoot), <1 gives a little bounce.
    """

    def __init__(self, value: float = 0.0, response: float = 0.32, damping_ratio: float = 0.86) -> None:
        self.value = float(value)
        self.target = float(value)
        self.velocity = 0.0
        self.response = response
        self.damping_ratio = damping_ratio

    def set_target(self, target: float) -> None:
        self.target = float(target)

    def snap(self, value: float) -> None:
        self.value = float(value)
        self.target = float(value)
        self.velocity = 0.0

    def step(self, delta_seconds: float) -> float:
        delta_seconds = max(0.0, min(delta_seconds, 1.0 / 20.0))
        angular_frequency = 2.0 * math.pi / max(0.05, self.response)
        stiffness = angular_frequency * angular_frequency
        damping = 2.0 * self.damping_ratio * angular_frequency
        # Two half-steps keep the integration stable at low frame rates.
        for _ in range(2):
            half_step = delta_seconds / 2.0
            acceleration = stiffness * (self.target - self.value) - damping * self.velocity
            self.velocity += acceleration * half_step
            self.value += self.velocity * half_step
        return self.value

    def is_settled(self, tolerance: float = 0.05) -> bool:
        return abs(self.target - self.value) < tolerance and abs(self.velocity) < tolerance * 4


# ---------------------------------------------------------------------------
# Win32 window style helpers for overlay windows
# ---------------------------------------------------------------------------

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_EX_LAYERED = 0x00080000
WS_EX_APPWINDOW = 0x00040000
WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11


def _user32():
    if sys.platform != "win32":
        return None
    return ctypes.windll.user32


def set_window_click_through(window_id: int, click_through: bool) -> None:
    """Toggles WS_EX_TRANSPARENT so mouse events pass through to the windows below."""
    user32 = _user32()
    if user32 is None or not window_id:
        return
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    current_style = user32.GetWindowLongPtrW(ctypes.c_void_p(window_id), GWL_EXSTYLE)
    if click_through:
        new_style = current_style | WS_EX_TRANSPARENT | WS_EX_LAYERED
    else:
        new_style = current_style & ~WS_EX_TRANSPARENT
    if new_style != current_style:
        user32.SetWindowLongPtrW(ctypes.c_void_p(window_id), GWL_EXSTYLE, new_style)


def set_window_no_activate(window_id: int, no_activate: bool = True) -> None:
    """WS_EX_NOACTIVATE + WS_EX_TOOLWINDOW: never steals focus, hidden from Alt+Tab/taskbar."""
    user32 = _user32()
    if user32 is None or not window_id:
        return
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    current_style = user32.GetWindowLongPtrW(ctypes.c_void_p(window_id), GWL_EXSTYLE)
    if no_activate:
        new_style = (current_style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
    else:
        new_style = current_style & ~WS_EX_NOACTIVATE
    if new_style != current_style:
        user32.SetWindowLongPtrW(ctypes.c_void_p(window_id), GWL_EXSTYLE, new_style)


def set_window_excluded_from_capture(window_id: int, excluded: bool) -> None:
    """Hides the window from screenshots/recordings (Windows 10 2004+)."""
    user32 = _user32()
    if user32 is None or not window_id:
        return
    try:
        user32.SetWindowDisplayAffinity(ctypes.c_void_p(window_id), WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE)
    except Exception:
        pass


def force_foreground_window(window_id: int) -> None:
    """
    Brings a window to the foreground so it can receive typing. Windows blocks
    SetForegroundWindow from background processes unless we briefly attach to the
    foreground thread's input queue, which is the documented workaround.
    """
    user32 = _user32()
    if user32 is None or not window_id:
        return
    kernel32 = ctypes.windll.kernel32
    foreground_window = user32.GetForegroundWindow()
    foreground_thread_id = user32.GetWindowThreadProcessId(foreground_window, None)
    current_thread_id = kernel32.GetCurrentThreadId()
    attached = False
    if foreground_thread_id and foreground_thread_id != current_thread_id:
        attached = bool(user32.AttachThreadInput(foreground_thread_id, current_thread_id, True))
    try:
        user32.BringWindowToTop(ctypes.c_void_p(window_id))
        user32.SetForegroundWindow(ctypes.c_void_p(window_id))
        user32.SetFocus(ctypes.c_void_p(window_id))
    finally:
        if attached:
            user32.AttachThreadInput(foreground_thread_id, current_thread_id, False)


def enable_per_monitor_dpi_awareness() -> None:
    """Enable Per-Monitor v2 DPI awareness on Windows before QApplication creation for sharp HiDPI output."""
    if sys.platform != "win32":
        return
    try:
        user32 = ctypes.windll.user32
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
    except Exception:
        pass
    try:
        shcore = ctypes.windll.shcore
        if hasattr(shcore, "SetProcessDpiAwareness"):
            # PROCESS_PER_MONITOR_DPI_AWARE == 2
            shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def asset_path(*relative_parts: str) -> str:
    return os.path.join(ASSETS_DIRECTORY, *relative_parts)


def greeting_for_current_time(display_name: str) -> str:
    """'Morning, RUTHVIK.' style greeting used on the notch Home peek."""
    import time as _time

    hour = _time.localtime().tm_hour
    if 5 <= hour < 12:
        part_of_day = "Morning"
    elif 12 <= hour < 17:
        part_of_day = "Afternoon"
    elif 17 <= hour < 22:
        part_of_day = "Evening"
    else:
        part_of_day = "Hey"
    return f"{part_of_day}, {display_name}." if display_name else f"{part_of_day}."
