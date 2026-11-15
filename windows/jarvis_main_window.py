"""
Resizable Jarvis Settings & Chat window for the Windows PySide6 replica.

Matches the dark obsidian sidebar + card layout from the Windows reference photo and
implements every Jarvis 1.1.4 / 1.1.5 improvement:
- Loads `assets/jarvis.ico` so Alt+Tab and the taskbar show Jarvis's logo.
- Custom-painted `ToggleSwitch` that never lights up blue on hover while OFF.
- Custom-painted `ColorDotButton` circles with zero square outline on hover/focus.
- `Account` page with local BYOK `.env` status and masked API key editing.
- `Voice` page where clicking any voice selects it and emits an immediate preview.
- `Dictation` page with Custom Dictionary placed above Language.
- `General` page with Remembered answer style rules and deletion controls.
- `Chat` view with live streaming replies, Stop button, and reliable keyboard focus.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPoint,
    QPointF,
    QRectF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QUrl,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPixmap,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import jarvis_config
import jarvis_theme

DARK_PALETTE: Dict[str, str] = jarvis_theme.WINDOW_PALETTE_DARK

FIVE_CORE_SHORTCUTS: List[Dict[str, Any]] = [
    {
        "shortcut_id": "push_to_talk",
        "title": "Talk & Highlight",
        "keys": ["Ctrl", "Alt"],
        "mode_label": "Hold",
        "description": "Speak to Jarvis with live screen context. Drag while holding to circle or highlight elements.",
    },
    {
        "shortcut_id": "double_ctrl",
        "title": "Notch Text Composer",
        "keys": ["Ctrl", "Ctrl"],
        "mode_label": "Double-tap",
        "description": "Open or close the floating top-notch typed composer and peek drawer.",
    },
    {
        "shortcut_id": "dictate",
        "title": "Dictate",
        "keys": ["Ctrl", "Win"],
        "mode_label": "Hold",
        "description": "Hold to stream voice dictation directly into whichever Windows text field is focused.",
    },
    {
        "shortcut_id": "hands_free_dictate",
        "title": "Hands-free Dictation",
        "keys": ["Ctrl", "Win", "Win"],
        "mode_label": "Double-tap Win",
        "description": "Lock continuous hands-free dictation on until you press Escape or tap again.",
    },
    {
        "shortcut_id": "escape",
        "title": "Clear / Stop",
        "keys": ["Esc"],
        "mode_label": "Tap",
        "description": "Stop speech or active responses, clear screen drawings, or collapse the notch.",
    },
]

SECRET_FIELD_DEFINITIONS: List[Dict[str, str]] = [
    {
        "secret_name": jarvis_config.SECRET_GOOGLE_API_KEY,
        "display_title": "Google Gemini Key",
        "subtitle": "Gemini 4 Hydrogen & 3.8 Flash Vision (GOOGLE_API_KEY)",
        "placeholder": "Enter Google AI Studio key (AIza...)...",
    },
    {
        "secret_name": jarvis_config.SECRET_RIVER_API_KEY,
        "display_title": "River AI Key",
        "subtitle": "Vision chat & screen reasoning (RIVER_AI_API)",
        "placeholder": "Enter River AI key...",
    },
    {
        "secret_name": jarvis_config.SECRET_ASSEMBLYAI_API_KEY,
        "display_title": "AssemblyAI Key",
        "subtitle": "Streaming speech-to-text & neural voice TTS (ASSEMBLYAI_API_KEY)",
        "placeholder": "Enter AssemblyAI key...",
    },
]

SIDEBAR_SETTINGS_SECTIONS: List[Dict[str, Any]] = [
    {
        "section_header": None,
        "items": [
            {"tab_key": "account", "label": "Studio & Keys", "subtitle": "Personal studio quick-launch and local .env API keys."},
            {"tab_key": "general", "label": "General", "subtitle": "Display name, vision model, audio cues, and learned answer style."},
        ],
    },
    {
        "section_header": "JARVIS",
        "items": [
            {"tab_key": "voice", "label": "Voice", "subtitle": "Speech speed, live captions, and voice preview selection."},
            {"tab_key": "microphone", "label": "Microphone", "subtitle": "Choose your input device and verify live microphone levels."},
            {"tab_key": "dictation", "label": "Dictation", "subtitle": "Custom spelling dictionary and dictation language."},
            {"tab_key": "shortcuts", "label": "Shortcuts", "subtitle": "System-wide hotkeys available across every Windows app."},
            {"tab_key": "cursor", "label": "Cursor", "subtitle": "Cursor color, visibility, Draw-to-Ask, and detach behavior."},
        ],
    },
    {
        "section_header": "WORK",
        "items": [
            {"tab_key": "integrations", "label": "Integrations", "subtitle": "Local BYOK provider connections without cloud OAuth proxies."},
            {"tab_key": "agents", "label": "Agents", "subtitle": "How Jarvis executes screen-aware turns locally on Windows."},
            {"tab_key": "routines", "label": "Routines", "subtitle": "On-demand workflows and local Windows usage notes."},
        ],
    },
]


def _enable_dark_title_bar(window_id: int) -> None:
    """Requests Windows 10/11 dark immersive title bar chrome matching #1B1B1D."""
    if sys.platform != "win32" or not window_id:
        return
    try:
        dwmapi = ctypes.windll.dwmapi
        use_dark_mode = ctypes.c_int(1)
        for attribute_id in (20, 19):
            result_code = dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(window_id),
                ctypes.c_int(attribute_id),
                ctypes.byref(use_dark_mode),
                ctypes.sizeof(use_dark_mode),
            )
            if result_code == 0:
                break
    except Exception:
        pass


def _paint_sidebar_icon(
    painter: QPainter,
    icon_key: str,
    icon_rect: QRectF,
    foreground_color: QColor,
    avatar_initial: str = "R",
) -> None:
    """Draws crisp vector icons in the sidebar so zero missing-glyph boxes appear."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    center_x = icon_rect.center().x()
    center_y = icon_rect.center().y()

    if icon_key == "account":
        circle_diameter = min(icon_rect.width(), icon_rect.height())
        avatar_rect = QRectF(
            center_x - circle_diameter / 2.0,
            center_y - circle_diameter / 2.0,
            circle_diameter,
            circle_diameter,
        )
        painter.setPen(QPen(QColor("#52525B"), 1.0))
        painter.setBrush(QColor("#27272A"))
        painter.drawEllipse(avatar_rect)
        painter.setPen(QColor("#F4F4F6"))
        painter.setFont(jarvis_theme.ui_font(8.5, QFont.Weight.Bold))
        painter.drawText(avatar_rect, Qt.AlignmentFlag.AlignCenter, (avatar_initial or "R")[:1].upper())
        painter.restore()
        return

    icon_pen = QPen(foreground_color, 1.55, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(icon_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if icon_key == "chat":
        bubble_rect = QRectF(center_x - 6.5, center_y - 5.5, 13.0, 10.0)
        bubble_path = QPainterPath()
        bubble_path.addRoundedRect(bubble_rect, 3.2, 3.2)
        painter.drawPath(bubble_path)
        painter.drawLine(QPointF(center_x - 3.5, center_y + 4.5), QPointF(center_x - 5.0, center_y + 7.0))
        painter.drawLine(QPointF(center_x - 5.0, center_y + 7.0), QPointF(center_x - 0.5, center_y + 4.5))

    elif icon_key == "general":
        painter.drawEllipse(QPointF(center_x, center_y), 3.0, 3.0)
        for tooth_index in range(6):
            angle_rad = math.radians(tooth_index * 60.0)
            inner_point = QPointF(center_x + math.cos(angle_rad) * 4.5, center_y + math.sin(angle_rad) * 4.5)
            outer_point = QPointF(center_x + math.cos(angle_rad) * 6.7, center_y + math.sin(angle_rad) * 6.7)
            painter.drawLine(inner_point, outer_point)
        painter.drawEllipse(QPointF(center_x, center_y), 5.2, 5.2)

    elif icon_key == "voice":
        bar_offsets = [(-4.8, 3.2), (-1.6, 6.2), (1.6, 4.6), (4.8, 2.6)]
        for offset_x, half_height in bar_offsets:
            painter.drawLine(
                QPointF(center_x + offset_x, center_y - half_height),
                QPointF(center_x + offset_x, center_y + half_height),
            )

    elif icon_key == "microphone":
        capsule_rect = QRectF(center_x - 2.6, center_y - 6.5, 5.2, 8.0)
        painter.drawRoundedRect(capsule_rect, 2.6, 2.6)
        arc_rect = QRectF(center_x - 5.0, center_y - 3.5, 10.0, 7.8)
        painter.drawArc(arc_rect, 180 * 16, 180 * 16)
        painter.drawLine(QPointF(center_x, center_y + 4.3), QPointF(center_x, center_y + 6.8))
        painter.drawLine(QPointF(center_x - 3.0, center_y + 6.8), QPointF(center_x + 3.0, center_y + 6.8))

    elif icon_key == "dictation":
        painter.drawLine(QPointF(center_x, center_y - 5.8), QPointF(center_x, center_y + 5.8))
        painter.drawLine(QPointF(center_x - 3.2, center_y - 5.8), QPointF(center_x + 3.2, center_y - 5.8))
        painter.drawLine(QPointF(center_x - 3.2, center_y + 5.8), QPointF(center_x + 3.2, center_y + 5.8))

    elif icon_key == "shortcuts":
        box_rect = QRectF(center_x - 2.4, center_y - 2.4, 4.8, 4.8)
        painter.drawRect(box_rect)
        loop_offsets = [(-4.8, -4.8), (2.4, -4.8), (-4.8, 2.4), (2.4, 2.4)]
        for offset_x, offset_y in loop_offsets:
            painter.drawEllipse(QRectF(center_x + offset_x, center_y + offset_y, 2.4, 2.4))

    elif icon_key == "cursor":
        triangle_path = jarvis_theme.build_rounded_triangle_path(side_length=11.0, corner_fraction=0.1, rotation_degrees=-35.0)
        painter.translate(center_x, center_y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(foreground_color)
        painter.drawPath(triangle_path)

    elif icon_key == "integrations":
        arc_rect = QRectF(center_x - 5.5, center_y - 5.5, 11.0, 11.0)
        painter.drawArc(arc_rect, 45 * 16, 270 * 16)
        painter.drawLine(QPointF(center_x, center_y), QPointF(center_x + 5.5, center_y))

    elif icon_key == "agents":
        painter.drawEllipse(QPointF(center_x, center_y - 4.0), 1.8, 1.8)
        painter.drawEllipse(QPointF(center_x - 4.2, center_y + 3.5), 1.8, 1.8)
        painter.drawEllipse(QPointF(center_x + 4.2, center_y + 3.5), 1.8, 1.8)
        painter.drawLine(QPointF(center_x - 0.8, center_y - 2.4), QPointF(center_x - 3.4, center_y + 2.0))
        painter.drawLine(QPointF(center_x + 0.8, center_y - 2.4), QPointF(center_x + 3.4, center_y + 2.0))
        painter.drawLine(QPointF(center_x - 2.4, center_y + 3.5), QPointF(center_x + 2.4, center_y + 3.5))

    elif icon_key == "routines":
        arc_rect = QRectF(center_x - 5.5, center_y - 5.5, 11.0, 11.0)
        painter.drawArc(arc_rect, 25 * 16, 290 * 16)
        painter.drawLine(QPointF(center_x + 4.8, center_y - 3.8), QPointF(center_x + 5.6, center_y - 0.8))
        painter.drawLine(QPointF(center_x + 4.8, center_y - 3.8), QPointF(center_x + 2.0, center_y - 3.6))

    painter.restore()


def _paint_eye_icon(painter: QPainter, icon_rect: QRectF, is_revealed: bool, foreground_color: QColor) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    center_x = icon_rect.center().x()
    center_y = icon_rect.center().y()
    pen = QPen(foreground_color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    eye_path = QPainterPath()
    eye_path.moveTo(center_x - 6.5, center_y)
    eye_path.quadTo(center_x, center_y - 5.5, center_x + 6.5, center_y)
    eye_path.quadTo(center_x, center_y + 5.5, center_x - 6.5, center_y)
    painter.drawPath(eye_path)
    painter.drawEllipse(QPointF(center_x, center_y), 2.0, 2.0)

    if not is_revealed:
        painter.drawLine(QPointF(center_x - 5.5, center_y + 5.0), QPointF(center_x + 5.5, center_y - 5.0))
    painter.restore()


def _build_card_frame(object_name: str = "SettingsCard") -> QFrame:
    card_frame = QFrame()
    card_frame.setObjectName(object_name)
    card_frame.setFrameShape(QFrame.Shape.NoFrame)
    return card_frame


def _build_section_label(label_text: str) -> QLabel:
    section_label = QLabel(label_text)
    section_label.setObjectName("SectionHeaderLabel")
    return section_label


def _build_hairline_divider() -> QFrame:
    divider = QFrame()
    divider.setObjectName("CardHairlineDivider")
    divider.setFixedHeight(1)
    return divider


def _build_setting_row(
    title_text: str,
    subtitle_text: Optional[str],
    trailing_widget: QWidget,
) -> QWidget:
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    row_layout.setContentsMargins(0, 4, 0, 4)
    row_layout.setSpacing(14)

    text_column = QVBoxLayout()
    text_column.setContentsMargins(0, 0, 0, 0)
    text_column.setSpacing(2)

    title_label = QLabel(title_text)
    title_label.setObjectName("RowTitleLabel")
    text_column.addWidget(title_label)

    if subtitle_text:
        subtitle_label = QLabel(subtitle_text)
        subtitle_label.setObjectName("RowSubtitleLabel")
        subtitle_label.setWordWrap(True)
        text_column.addWidget(subtitle_label)

    row_layout.addLayout(text_column, 1)
    row_layout.addWidget(trailing_widget, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
    return row_widget


# ---------------------------------------------------------------------------
# Custom Painted Controls
# ---------------------------------------------------------------------------


class ToggleSwitch(QAbstractButton):
    """
    40x22 pill toggle switch matching Jarvis 1.1.5 behavior:
    hovering an OFF switch shows only a neutral grey hover border/tint and NEVER lights up blue.
    """

    def __init__(self, initial_checked: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(initial_checked)
        self.setFixedSize(40, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setStyleSheet("background: transparent; border: none; outline: none;")

        self._thumb_progress = 1.0 if initial_checked else 0.0
        self._thumb_animation = QPropertyAnimation(self, b"thumbProgress", self)
        self._thumb_animation.setDuration(135)
        self._thumb_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate_thumb_to_state)

    def sizeHint(self) -> QSize:
        return QSize(40, 22)

    @Property(float)
    def thumbProgress(self) -> float:
        return self._thumb_progress

    @thumbProgress.setter
    def thumbProgress(self, progress_value: float) -> None:
        self._thumb_progress = max(0.0, min(1.0, float(progress_value)))
        self.update()

    def _animate_thumb_to_state(self, is_checked: bool) -> None:
        target_progress = 1.0 if is_checked else 0.0
        self._thumb_animation.stop()
        self._thumb_animation.setStartValue(self._thumb_progress)
        self._thumb_animation.setEndValue(target_progress)
        self._thumb_animation.start()

    def setCheckedInstant(self, is_checked: bool) -> None:
        """Sets checked state without firing animations when syncing from settings_store."""
        self._thumb_animation.stop()
        self.setChecked(is_checked)
        self._thumb_progress = 1.0 if is_checked else 0.0
        self.update()

    def paintEvent(self, paint_event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        is_hovered = self.underMouse()
        is_pressed = self.isDown()
        is_checked = self.isChecked()
        track_rect = QRectF(1.0, 1.0, self.width() - 2.0, self.height() - 2.0)
        corner_radius = track_rect.height() / 2.0

        if is_checked:
            if is_pressed:
                track_color = QColor("#D4D4D8")
            elif is_hovered:
                track_color = QColor("#F4F4F6")
            else:
                track_color = QColor("#E4E4E7")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(track_color)
        else:
            # OFF switch stays dark neutral zinc.
            if is_pressed:
                track_color = QColor("#383842")
                border_color = QColor("#646470")
            elif is_hovered:
                track_color = QColor("#2E2E36")
                border_color = QColor("#52525B")
            else:
                track_color = QColor("#222229")
                border_color = QColor("#3F3F46")
            painter.setPen(QPen(border_color, 1.0))
            painter.setBrush(track_color)

        painter.drawRoundedRect(track_rect, corner_radius, corner_radius)

        if self.hasFocus():
            focus_pen = QPen(QColor(DARK_PALETTE["accent_soft"]), 1.2)
            painter.setPen(focus_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(track_rect.adjusted(-0.5, -0.5, 0.5, 0.5), corner_radius, corner_radius)

        thumb_diameter = 16.0
        thumb_min_x = 3.0
        thumb_max_x = self.width() - thumb_diameter - 3.0
        thumb_x = thumb_min_x + (thumb_max_x - thumb_min_x) * self._thumb_progress
        thumb_y = (self.height() - thumb_diameter) / 2.0
        thumb_rect = QRectF(thumb_x, thumb_y, thumb_diameter, thumb_diameter)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 55))
        painter.drawEllipse(thumb_rect.translated(0.0, 0.8))

        thumb_color = QColor("#09090B") if is_checked else QColor("#D4D4D8")
        painter.setBrush(thumb_color)
        painter.drawEllipse(thumb_rect)


class ColorDotButton(QAbstractButton):
    """
    Circular 24x24 color swatch with pure circular selection and hover rings.
    Implements Jarvis 1.1.5 fix: zero square/rectangular outline on hover or focus.
    """

    def __init__(self, color_name: str, color_hex: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.color_name = color_name
        self.color_hex = color_hex
        self.setCheckable(True)
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet("background: transparent; border: none; outline: none;")
        self.setToolTip(color_name)

    def sizeHint(self) -> QSize:
        return QSize(32, 32)

    def paintEvent(self, paint_event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        swatch_rect = QRectF(center_x - 12.0, center_y - 12.0, 24.0, 24.0)
        ring_rect = QRectF(center_x - 14.5, center_y - 14.5, 29.0, 29.0)

        if self.isChecked():
            painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(ring_rect)
        elif self.underMouse():
            painter.setPen(QPen(QColor(255, 255, 255, 115), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(ring_rect)

        painter.setPen(QPen(QColor(0, 0, 0, 80), 1.0))
        painter.setBrush(QColor(self.color_hex))
        painter.drawEllipse(swatch_rect)

        if self.isChecked():
            check_pen = QPen(QColor("#FFFFFF"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(check_pen)
            check_path = QPainterPath()
            check_path.moveTo(center_x - 4.2, center_y + 0.3)
            check_path.lineTo(center_x - 1.2, center_y + 3.4)
            check_path.lineTo(center_x + 4.5, center_y - 2.8)
            painter.drawPath(check_path)


class SidebarNavButton(QAbstractButton):
    """Custom-painted sidebar navigation item matching the dark reference photo."""

    def __init__(self, tab_key: str, label_text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.tab_key = tab_key
        self._label_text = label_text
        self._avatar_initial = "R"
        self.setCheckable(True)
        self.setFixedHeight(34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet("background: transparent; border: none; outline: none;")

    def set_avatar_initial(self, avatar_initial: str) -> None:
        cleaned = (avatar_initial or "R").strip()
        self._avatar_initial = cleaned[:1].upper() if cleaned else "R"
        self.update()

    def paintEvent(self, paint_event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pill_rect = QRectF(0.0, 1.0, self.width(), self.height() - 2.0)
        is_selected = self.isChecked()
        is_hovered = self.underMouse()

        if is_selected:
            gradient = QLinearGradient(pill_rect.topLeft(), pill_rect.bottomLeft())
            gradient.setColorAt(0.0, QColor("#3A3A3F"))
            gradient.setColorAt(1.0, QColor("#2D2D31"))
            painter.setPen(QPen(QColor(255, 255, 255, 28), 1.0))
            painter.setBrush(gradient)
            painter.drawRoundedRect(pill_rect.adjusted(0.5, 0.5, -0.5, -0.5), 8.0, 8.0)
        elif is_hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2A2A2E"))
            painter.drawRoundedRect(pill_rect, 8.0, 8.0)

        icon_size = 20.0 if self.tab_key == "account" else 16.0
        icon_left = 10.0 if self.tab_key == "account" else 12.0
        icon_rect = QRectF(icon_left, (self.height() - icon_size) / 2.0, icon_size, icon_size)

        text_color = QColor(DARK_PALETTE["text_primary"]) if is_selected else QColor("#D2D2D7")
        icon_color = QColor(DARK_PALETTE["text_primary"]) if is_selected else QColor("#B4B4BA")
        _paint_sidebar_icon(
            painter=painter,
            icon_key=self.tab_key,
            icon_rect=icon_rect,
            foreground_color=icon_color,
            avatar_initial=self._avatar_initial,
        )

        text_left = 38.0
        text_rect = QRectF(text_left, 0.0, max(10.0, self.width() - text_left - 8.0), self.height())
        painter.setPen(text_color)
        painter.setFont(jarvis_theme.ui_font(10.0, QFont.Weight.DemiBold if is_selected else QFont.Weight.Medium))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._label_text)


class EyeToggleButton(QAbstractButton):
    """Vector eye button to toggle password visibility on API key fields."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet("background: transparent; border: none; outline: none;")
        self.setToolTip("Show or hide API key")

    def paintEvent(self, paint_event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.underMouse():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#38383E"))
            painter.drawRoundedRect(QRectF(1.0, 1.0, self.width() - 2.0, self.height() - 2.0), 6.0, 6.0)
        foreground = QColor(DARK_PALETTE["text_primary"]) if self.isChecked() or self.underMouse() else QColor(DARK_PALETTE["text_secondary"])
        icon_rect = QRectF(6.0, 6.0, self.width() - 12.0, self.height() - 12.0)
        _paint_eye_icon(painter=painter, icon_rect=icon_rect, is_revealed=self.isChecked(), foreground_color=foreground)


class FlowLayout(QLayout):
    """Wrapping horizontal flow layout used for Custom Dictionary word chips."""

    def __init__(self, parent: Optional[QWidget] = None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: List[QLayoutItem] = []
        self._spacing = spacing

    def __del__(self) -> None:
        while self._items:
            self.takeAt(0)

    def addItem(self, layout_item: QLayoutItem) -> None:
        self._items.append(layout_item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, available_width: int) -> int:
        return self._arrange_items(QRect(0, 0, available_width, 0), apply_geometry=False)

    def setGeometry(self, target_rect: QRect) -> None:
        super().setGeometry(target_rect)
        self._arrange_items(target_rect, apply_geometry=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        minimum_size = QSize()
        for layout_item in self._items:
            minimum_size = minimum_size.expandedTo(layout_item.minimumSize())
        margins = self.contentsMargins()
        return minimum_size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _arrange_items(self, target_rect: QRect, apply_geometry: bool) -> int:
        margins = self.contentsMargins()
        effective_rect = target_rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        current_x = effective_rect.x()
        current_y = effective_rect.y()
        row_height = 0

        for layout_item in self._items:
            item_size = layout_item.sizeHint()
            next_x = current_x + item_size.width() + self._spacing
            if next_x - self._spacing > effective_rect.right() and row_height > 0:
                current_x = effective_rect.x()
                current_y += row_height + self._spacing
                next_x = current_x + item_size.width() + self._spacing
                row_height = 0
            if apply_geometry:
                layout_item.setGeometry(QRect(QPoint(current_x, current_y), item_size))
            current_x = next_x
            row_height = max(row_height, item_size.height())

        return current_y + row_height - target_rect.y() + margins.bottom()


class CursorPreviewWidget(QWidget):
    """Live miniature preview of the -35° rounded triangle cursor in the selected color."""

    def __init__(self, initial_color_hex: str = jarvis_theme.JARVIS_BLUE_HEX, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._color_hex = initial_color_hex
        self.setFixedSize(64, 48)

    def set_color_hex(self, color_hex: str) -> None:
        self._color_hex = color_hex
        self.update()

    def paintEvent(self, paint_event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        background_rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(QPen(QColor(DARK_PALETTE["separator"]), 1.0))
        painter.setBrush(QColor("#202023"))
        painter.drawRoundedRect(background_rect, 10.0, 10.0)

        center_point = QPointF(self.width() / 2.0, self.height() / 2.0)
        cursor_color = QColor(self._color_hex)

        triangle_path = jarvis_theme.build_rounded_triangle_path(
            side_length=18.0,
            corner_fraction=jarvis_theme.CURSOR_TRIANGLE_CORNER_FRACTION,
            rotation_degrees=jarvis_theme.CURSOR_RESTING_ROTATION_DEGREES,
        )
        painter.translate(center_point)
        painter.setPen(QPen(QColor(255, 255, 255, 95), 1.0))
        painter.setBrush(cursor_color)
        painter.drawPath(triangle_path)


class MicrophoneLevelMeter(QWidget):
    """Smooth rounded pill audio input meter updated by `set_microphone_level(level)`."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self.setFixedHeight(12)
        self.setMinimumWidth(160)

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, float(level)))
        self.update()

    def paintEvent(self, paint_event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        track_rect = QRectF(0.0, 1.0, self.width(), self.height() - 2.0)
        radius = track_rect.height() / 2.0
        painter.setPen(QPen(QColor(DARK_PALETTE["separator"]), 1.0))
        painter.setBrush(QColor("#1E1E21"))
        painter.drawRoundedRect(track_rect, radius, radius)

        if self._level > 0.005:
            fill_width = max(track_rect.height(), track_rect.width() * self._level)
            fill_rect = QRectF(track_rect.left(), track_rect.top(), fill_width, track_rect.height())
            gradient = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            gradient.setColorAt(0.0, QColor("#A1A1AA"))
            gradient.setColorAt(1.0, QColor("#F4F4F6"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawRoundedRect(fill_rect, radius, radius)


class PillSlider(QSlider):
    """Custom-painted horizontal pill slider with crisp circular handle."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFixedHeight(24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def paintEvent(self, paint_event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        horizontal_padding = 9.0
        track_height = 4.0
        center_y = self.height() / 2.0
        track_left = horizontal_padding
        track_width = max(10.0, self.width() - horizontal_padding * 2.0)
        track_rect = QRectF(track_left, center_y - track_height / 2.0, track_width, track_height)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#3B3B42"))
        painter.drawRoundedRect(track_rect, 2.0, 2.0)

        value_span = max(1, self.maximum() - self.minimum())
        fraction = (self.value() - self.minimum()) / float(value_span)
        fraction = max(0.0, min(1.0, fraction))
        knob_center_x = track_left + track_width * fraction

        fill_rect = QRectF(track_left, track_rect.top(), max(0.0, knob_center_x - track_left), track_height)
        painter.setBrush(QColor("#E4E4E7"))
        painter.drawRoundedRect(fill_rect, 2.0, 2.0)

        knob_radius = 7.5 if self.underMouse() or self.isSliderDown() else 6.8
        if self.underMouse() or self.isSliderDown():
            painter.setBrush(QColor(228, 228, 231, 55))
            painter.drawEllipse(QPointF(knob_center_x, center_y), knob_radius + 3.5, knob_radius + 3.5)

        painter.setPen(QPen(QColor("#3F3F46"), 2.0))
        painter.setBrush(QColor("#F4F4F6"))
        painter.drawEllipse(QPointF(knob_center_x, center_y), knob_radius, knob_radius)


class ChevronComboBox(QComboBox):
    """Dark QComboBox that renders a crisp vector downward chevron arrow."""

    def paintEvent(self, paint_event: Any) -> None:
        super().paintEvent(paint_event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        chevron_pen = QPen(
            QColor(DARK_PALETTE["text_secondary"]),
            1.6,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(chevron_pen)
        center_x = self.width() - 14.0
        center_y = self.height() / 2.0
        chevron_path = QPainterPath()
        chevron_path.moveTo(center_x - 4.0, center_y - 2.0)
        chevron_path.lineTo(center_x, center_y + 2.2)
        chevron_path.lineTo(center_x + 4.0, center_y - 2.0)
        painter.drawPath(chevron_path)


class VoiceChoiceRow(QFrame):
    """Clickable row inside Settings > Voice that selects and immediately previews a voice."""

    clicked = Signal(str)

    def __init__(self, voice_entry: Dict[str, str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.voice_id = voice_entry["voice_id"]
        self._selected = False
        self._hovered = False
        self.setObjectName("VoiceChoiceRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFixedHeight(48)

        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(12, 8, 12, 8)
        row_layout.setSpacing(12)

        self._radio_indicator = QWidget()
        self._radio_indicator.setFixedSize(18, 18)
        self._radio_indicator.paintEvent = self._paint_radio_indicator
        row_layout.addWidget(self._radio_indicator, 0, Qt.AlignmentFlag.AlignVCenter)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(1)

        name_label = QLabel(voice_entry["name"])
        name_label.setObjectName("VoiceNameLabel")
        text_column.addWidget(name_label)

        description_label = QLabel(voice_entry["description"])
        description_label.setObjectName("VoiceDescriptionLabel")
        text_column.addWidget(description_label)

        row_layout.addLayout(text_column, 1)

        self._badge_label = QLabel(self._provider_badge_text())
        self._badge_label.setObjectName("VoiceProviderBadge")
        row_layout.addWidget(self._badge_label, 0, Qt.AlignmentFlag.AlignVCenter)

    def _provider_badge_text(self) -> str:
        return "Offline SAPI" if self.voice_id == "windows-sapi" else "AssemblyAI"

    def set_selected(self, is_selected: bool) -> None:
        self._selected = is_selected
        self._badge_label.setText("Selected" if is_selected else self._provider_badge_text())
        self.setProperty("selected", is_selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self._radio_indicator.update()

    def _paint_radio_indicator(self, paint_event: Any) -> None:
        painter = QPainter(self._radio_indicator)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        indicator_rect = QRectF(1.5, 1.5, 15.0, 15.0)
        if self._selected:
            painter.setPen(QPen(QColor("#E4E4E7"), 2.0))
            painter.setBrush(QColor("#E4E4E7"))
            painter.drawEllipse(indicator_rect)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#09090B"))
            painter.drawEllipse(QPointF(9.0, 9.0), 3.2, 3.2)
        else:
            ring_color = QColor("#8A8A94") if self._hovered else QColor("#585862")
            painter.setPen(QPen(ring_color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(indicator_rect)

    def enterEvent(self, enter_event: Any) -> None:
        self._hovered = True
        self._radio_indicator.update()
        super().enterEvent(enter_event)

    def leaveEvent(self, leave_event: Any) -> None:
        self._hovered = False
        self._radio_indicator.update()
        super().leaveEvent(leave_event)

    def mouseReleaseEvent(self, mouse_event: Any) -> None:
        if mouse_event.button() == Qt.MouseButton.LeftButton and self.rect().contains(mouse_event.position().toPoint()):
            self.clicked.emit(self.voice_id)
        super().mouseReleaseEvent(mouse_event)


class JarvisMarkBadge(QWidget):
    """Circular J.A.R.V.I.S emblem rendered in header, assistant messages, and welcome card."""

    def __init__(self, badge_size: int = 24, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(badge_size, badge_size)
        logo_path = jarvis_theme.asset_path("jarvis-logo.png")
        pixmap = QPixmap(logo_path) if os.path.exists(logo_path) else QPixmap()
        if not pixmap.isNull():
            pixel_ratio = max(1.0, self.devicePixelRatioF())
            target_px = max(16, int(round(badge_size * pixel_ratio * 2)))
            self._logo_pixmap: Optional[QPixmap] = pixmap.scaled(
                target_px,
                target_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            self._logo_pixmap = None

    def paintEvent(self, paint_event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self._logo_pixmap is not None and not self._logo_pixmap.isNull():
            target_rect = QRectF(0.0, 0.0, float(self.width()), float(self.height()))
            painter.drawPixmap(target_rect, self._logo_pixmap, QRectF(self._logo_pixmap.rect()))
            return
        center_point = QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = min(self.width(), self.height()) * 0.42
        painter.setPen(QPen(QColor("#F3F6FA"), max(1.4, self.width() * 0.08)))
        painter.setBrush(QColor("#090B0E"))
        painter.drawEllipse(center_point, radius, radius)


class ChatMessageBubble(QWidget):
    """Single user or assistant bubble inside the scrollable Chat feed."""

    def __init__(self, role: str, text: str, is_generating: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.role = "user" if role == "user" else "assistant"
        self._is_generating = is_generating

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(0, 2, 0, 2)
        outer_layout.setSpacing(10)

        self._bubble_frame = QFrame()
        self._bubble_frame.setObjectName("UserChatBubble" if self.role == "user" else "AssistantChatBubble")
        self._bubble_frame.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self._bubble_frame.setMaximumWidth(510)

        bubble_layout = QVBoxLayout(self._bubble_frame)
        bubble_layout.setContentsMargins(14, 10, 14, 10)
        bubble_layout.setSpacing(5)

        if self.role == "assistant":
            header_row = QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)
            header_row.setSpacing(6)
            header_row.addWidget(JarvisMarkBadge(18))

            sender_label = QLabel("Jarvis")
            sender_label.setObjectName("ChatSenderLabel")
            header_row.addWidget(sender_label)

            self._generating_badge = QLabel("Responding...")
            self._generating_badge.setObjectName("ChatGeneratingBadge")
            self._generating_badge.setVisible(is_generating)
            header_row.addWidget(self._generating_badge)
            header_row.addStretch(1)
            bubble_layout.addLayout(header_row)
        else:
            self._generating_badge = None

        self._body_label = QLabel(text)
        self._body_label.setObjectName("ChatBodyLabel")
        self._body_label.setWordWrap(True)
        self._body_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble_layout.addWidget(self._body_label)
        self._update_bubble_width()

        if self.role == "user":
            outer_layout.addStretch(1)
            outer_layout.addWidget(self._bubble_frame)
        else:
            outer_layout.addWidget(self._bubble_frame)
            outer_layout.addStretch(1)

    def _update_bubble_width(self) -> None:
        metrics = QFontMetrics(jarvis_theme.ui_font(10.0))
        text_lines = self._body_label.text().splitlines() or [""]
        longest_line_width = max(metrics.horizontalAdvance(line_text) for line_text in text_lines)
        header_width = 165 if self.role == "assistant" else 0
        desired_width = min(430, max(header_width, longest_line_width + 36))
        self._bubble_frame.setMinimumWidth(desired_width)
        self._bubble_frame.setMaximumWidth(450)

    def set_content(self, text: str, is_generating: bool) -> None:
        self._is_generating = is_generating
        if text:
            self._body_label.setText(text)
            self._update_bubble_width()
        if self._generating_badge is not None:
            self._generating_badge.setVisible(is_generating)

    def current_text(self) -> str:
        return self._body_label.text()


# ---------------------------------------------------------------------------
# Main Jarvis Settings & Chat Window
# ---------------------------------------------------------------------------


class JarvisMainWindow(QMainWindow):
    """
    Main resizable Settings and Chat window for Jarvis Windows v1.1.5.

    Connects cleanly to `SettingsStore` and emits action signals to the application
    controller (`promptSubmitted`, `stopRequested`, `demoDrawingRequested`,
    `quitRequested`, `voicePreviewRequested`, `microphoneTestRequested`,
    `shortcutTestRequested`).
    """

    promptSubmitted = Signal(str)
    stopRequested = Signal()
    demoDrawingRequested = Signal()
    scrollExplainRequested = Signal()
    lassoModeRequested = Signal()
    clearOverlayRequested = Signal()
    quitRequested = Signal()
    drawerVisibilityChanged = Signal(bool)
    openMemoryRequested = Signal()
    voicePreviewRequested = Signal(str)
    microphoneTestRequested = Signal()
    shortcutTestRequested = Signal(str)
    agentCommandRequested = Signal(str, str)

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        super().__init__()
        self._settings_store = settings_store
        self._updating_from_store = False
        self._active_top_tab = "settings"
        self._active_settings_subtab = "account"
        self._streaming_bubble: Optional[ChatMessageBubble] = None
        self._is_streaming_chat = False
        self._allow_close = False
        self._opening_from_notch = False
        self._drawer_open_animation: Optional[QVariantAnimation] = None

        self._sidebar_buttons: Dict[str, SidebarNavButton] = {}
        self._subtab_metadata: Dict[str, Dict[str, str]] = {}
        self._subtab_pages: Dict[str, QWidget] = {}
        self._secret_inputs: Dict[str, QLineEdit] = {}
        self._secret_status_badges: Dict[str, QLabel] = {}
        self._secret_save_buttons: Dict[str, QPushButton] = {}
        self._voice_rows: Dict[str, VoiceChoiceRow] = {}
        self._cursor_color_buttons: Dict[str, ColorDotButton] = {}

        self._configure_window_identity()
        self._build_ui()
        self._apply_stylesheet()
        self._load_values_from_store()

        self._settings_store.settingChanged.connect(self._on_external_setting_changed)
        self._settings_store.secretChanged.connect(self._on_external_secret_changed)

        self.open_tab("settings:account")

    def _configure_window_identity(self) -> None:
        self.setWindowTitle("Jarvis")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.resize(920, 560)
        self.setMinimumSize(780, 500)
        icon_path = jarvis_theme.asset_path("jarvis.ico")
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
            qt_app = QApplication.instance()
            if qt_app is not None and qt_app.windowIcon().isNull():
                qt_app.setWindowIcon(app_icon)
        self.position_under_notch()

    def _compute_notch_drawer_target_rect(self) -> QRect:
        primary_screen = QApplication.primaryScreen()
        screen_geo = primary_screen.geometry() if primary_screen is not None else QRect(0, 0, 1920, 1080)
        panel_width = min(max(self.width(), 920), max(760, screen_geo.width() - 48))
        panel_height = min(self.height(), max(480, screen_geo.height() - 64))
        dock_x = screen_geo.x() + (screen_geo.width() - panel_width) // 2
        dock_y = screen_geo.y()
        return QRect(dock_x, dock_y, panel_width, panel_height)

    def position_under_notch(self) -> None:
        """Docks the frameless Studio/Chat panel flush against the top hardware notch."""
        target_rect = self._compute_notch_drawer_target_rect()
        self.setGeometry(target_rect)

    def animate_open_from_notch(self, notch_geometry: Optional[QRect] = None) -> None:
        """Expands the drawer smoothly downward out of the top Notch."""
        target_rect = self._compute_notch_drawer_target_rect()
        if self._drawer_open_animation is not None:
            self._drawer_open_animation.stop()
            self._drawer_open_animation = None

        center_x = target_rect.center().x()
        top_edge_y = target_rect.top()
        if notch_geometry is not None and notch_geometry.width() > 40:
            start_width = max(260, min(420, notch_geometry.width()))
            start_height = max(36, min(64, notch_geometry.height()))
            start_rect = QRect(center_x - start_width // 2, top_edge_y, start_width, start_height)
        else:
            start_rect = QRect(center_x - 160, top_edge_y, 320, 42)

        self._opening_from_notch = True
        self.setGeometry(start_rect)
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

        drawer_anim = QVariantAnimation(self)
        drawer_anim.setDuration(170)
        drawer_anim.setStartValue(start_rect)
        drawer_anim.setEndValue(target_rect)
        drawer_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        drawer_anim.valueChanged.connect(lambda val: self.setGeometry(val))

        def _finish_drawer_open() -> None:
            self._opening_from_notch = False
            self._drawer_open_animation = None
            self.setGeometry(target_rect)

        drawer_anim.finished.connect(_finish_drawer_open)
        self._drawer_open_animation = drawer_anim
        drawer_anim.start()

    def keyPressEvent(self, key_event: Any) -> None:
        if key_event.key() == Qt.Key.Key_Escape:
            self.hide()
            key_event.accept()
            return
        super().keyPressEvent(key_event)

    def allow_real_close(self) -> None:
        self._allow_close = True

    def closeEvent(self, close_event: Any) -> None:
        if self._allow_close:
            super().closeEvent(close_event)
            return
        close_event.ignore()
        self.hide()

    def showEvent(self, show_event: Any) -> None:
        super().showEvent(show_event)
        if not self._opening_from_notch:
            self.position_under_notch()
        self.drawerVisibilityChanged.emit(True)
        window_handle = int(self.winId())
        _enable_dark_title_bar(window_handle)
        jarvis_theme.set_window_excluded_from_capture(
            window_handle,
            not bool(self._settings_store.get("show_in_screen_recordings", True)),
        )
        if self._active_top_tab == "chat":
            QTimer.singleShot(0, self._focus_chat_input)

    def hideEvent(self, hide_event: Any) -> None:
        self._opening_from_notch = False
        super().hideEvent(hide_event)
        self.drawerVisibilityChanged.emit(False)

    def changeEvent(self, change_event: Any) -> None:
        super().changeEvent(change_event)
        if (
            change_event.type() == change_event.Type.ActivationChange
            and self.isActiveWindow()
            and self._active_top_tab == "chat"
        ):
            QTimer.singleShot(0, self._focus_chat_input)

    # -- UI Construction -----------------------------------------------------

    def _build_ui(self) -> None:
        root_widget = QWidget()
        root_widget.setObjectName("MainWindowRoot")
        self.setCentralWidget(root_widget)

        outer_layout = QVBoxLayout(root_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Integrated top-center Notch handle strip so the drawer visually opens directly out of the Notch.
        notch_handle_bar = QFrame()
        notch_handle_bar.setObjectName("IntegratedNotchHeaderBar")
        notch_handle_bar.setFixedHeight(30)
        notch_handle_layout = QHBoxLayout(notch_handle_bar)
        notch_handle_layout.setContentsMargins(14, 3, 14, 3)
        notch_handle_layout.setSpacing(8)

        self._notch_memory_pill = QPushButton("Memory: Active (.md)")
        self._notch_memory_pill.setObjectName("NotchMemoryPill")
        self._notch_memory_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self._notch_memory_pill.setToolTip("Open today's Jarvis Markdown memory log")
        self._notch_memory_pill.clicked.connect(self.openMemoryRequested.emit)
        notch_handle_layout.addWidget(self._notch_memory_pill, 0)

        notch_handle_layout.addStretch(1)
        notch_handle_layout.addWidget(JarvisMarkBadge(18), 0, Qt.AlignmentFlag.AlignVCenter)
        notch_brand_label = QLabel("J.A.R.V.I.S")
        notch_brand_label.setStyleSheet(
            "color: #F4F4F6; font-size: 11px; font-weight: 700; letter-spacing: 1.8px;"
        )
        notch_handle_layout.addWidget(notch_brand_label, 0, Qt.AlignmentFlag.AlignVCenter)
        notch_handle_layout.addStretch(1)

        self._notch_collapse_button = QPushButton("Collapse")
        self._notch_collapse_button.setObjectName("NotchCollapsePill")
        self._notch_collapse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._notch_collapse_button.setToolTip("Collapse back into Notch (Esc)")
        self._notch_collapse_button.clicked.connect(self.hide)
        notch_handle_layout.addWidget(self._notch_collapse_button, 0)

        outer_layout.addWidget(notch_handle_bar)

        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        outer_layout.addLayout(root_layout, 1)

        self._sidebar_panel = self._build_sidebar()
        root_layout.addWidget(self._sidebar_panel)

        vertical_divider = QFrame()
        vertical_divider.setObjectName("SidebarVerticalDivider")
        vertical_divider.setFixedWidth(1)
        root_layout.addWidget(vertical_divider)

        content_container = QWidget()
        content_container.setObjectName("ContentPane")
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._header_bar = self._build_top_header_bar()
        content_layout.addWidget(self._header_bar)

        header_divider = QFrame()
        header_divider.setObjectName("HeaderHorizontalDivider")
        header_divider.setFixedHeight(1)
        content_layout.addWidget(header_divider)

        self._main_stack = QStackedWidget()
        self._settings_stack = QStackedWidget()
        self._build_all_settings_pages()
        self._chat_page = self._build_chat_page()

        self._main_stack.addWidget(self._settings_stack)
        self._main_stack.addWidget(self._chat_page)
        content_layout.addWidget(self._main_stack, 1)

        root_layout.addWidget(content_container, 1)

    def _build_sidebar(self) -> QFrame:
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("SidebarPanel")
        sidebar_frame.setFixedWidth(200)

        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(12, 14, 12, 12)
        sidebar_layout.setSpacing(6)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(4, 0, 4, 4)
        brand_row.setSpacing(8)
        brand_row.addWidget(JarvisMarkBadge(24))
        brand_title = QLabel("J.A.R.V.I.S")
        brand_title.setStyleSheet(
            "color: #F4F7FB; font-size: 13px; font-weight: 700; letter-spacing: 1.6px;"
        )
        brand_row.addWidget(brand_title)
        brand_row.addStretch(1)
        sidebar_layout.addLayout(brand_row)

        # Top segmented pill (Chat | Settings) in the sidebar for immediate mode switching.
        segmented_track = QFrame()
        segmented_track.setObjectName("SidebarSegmentedTrack")
        segmented_layout = QHBoxLayout(segmented_track)
        segmented_layout.setContentsMargins(3, 3, 3, 3)
        segmented_layout.setSpacing(3)

        self._sidebar_chat_pill = QPushButton("Chat")
        self._sidebar_chat_pill.setObjectName("SegmentedPillButton")
        self._sidebar_chat_pill.setCheckable(True)
        self._sidebar_chat_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sidebar_chat_pill.clicked.connect(lambda: self.open_tab("chat"))

        self._sidebar_settings_pill = QPushButton("Settings")
        self._sidebar_settings_pill.setObjectName("SegmentedPillButton")
        self._sidebar_settings_pill.setCheckable(True)
        self._sidebar_settings_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sidebar_settings_pill.clicked.connect(lambda: self.open_tab(f"settings:{self._active_settings_subtab}"))

        segmented_layout.addWidget(self._sidebar_chat_pill, 1)
        segmented_layout.addWidget(self._sidebar_settings_pill, 1)
        sidebar_layout.addWidget(segmented_track)
        sidebar_layout.addSpacing(4)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("SidebarScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_contents = QWidget()
        scroll_contents.setObjectName("SidebarScrollContents")
        items_layout = QVBoxLayout(scroll_contents)
        items_layout.setContentsMargins(0, 2, 0, 6)
        items_layout.setSpacing(2)

        for section_entry in SIDEBAR_SETTINGS_SECTIONS:
            section_header = section_entry["section_header"]
            if section_header:
                items_layout.addSpacing(10)
                header_label = QLabel(section_header)
                header_label.setObjectName("SidebarSectionHeader")
                items_layout.addWidget(header_label)
                items_layout.addSpacing(2)

            for item_entry in section_entry["items"]:
                tab_key = item_entry["tab_key"]
                self._subtab_metadata[tab_key] = item_entry
                nav_button = SidebarNavButton(tab_key=tab_key, label_text=item_entry["label"])
                nav_button.clicked.connect(lambda _checked=False, target_tab=tab_key: self.open_tab(f"settings:{target_tab}"))
                self._sidebar_buttons[tab_key] = nav_button
                items_layout.addWidget(nav_button)

        items_layout.addStretch(1)
        scroll_area.setWidget(scroll_contents)
        sidebar_layout.addWidget(scroll_area, 1)

        footer_label = QLabel(f"Jarvis {jarvis_theme.APP_DISPLAY_VERSION}")
        footer_label.setObjectName("SidebarFooterLabel")
        sidebar_layout.addWidget(footer_label)

        return sidebar_frame

    def _build_top_header_bar(self) -> QWidget:
        header_widget = QWidget()
        header_widget.setObjectName("TopHeaderBar")
        header_widget.setMinimumHeight(64)

        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 12, 20, 12)
        header_layout.setSpacing(10)

        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        self._header_title_label = QLabel("Studio & Keys")
        self._header_title_label.setObjectName("HeaderTitleLabel")
        title_row.addWidget(self._header_title_label)

        quick_switch_frame = QFrame()
        quick_switch_frame.setObjectName("HeaderQuickSwitchFrame")
        quick_switch_layout = QHBoxLayout(quick_switch_frame)
        quick_switch_layout.setContentsMargins(3, 3, 3, 3)
        quick_switch_layout.setSpacing(2)

        self._header_chat_pill = QPushButton("Chat")
        self._header_chat_pill.setObjectName("HeaderTogglePill")
        self._header_chat_pill.setCheckable(True)
        self._header_chat_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header_chat_pill.clicked.connect(lambda: self.open_tab("chat"))

        self._header_settings_pill = QPushButton("Settings")
        self._header_settings_pill.setObjectName("HeaderTogglePill")
        self._header_settings_pill.setCheckable(True)
        self._header_settings_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header_settings_pill.clicked.connect(lambda: self.open_tab(f"settings:{self._active_settings_subtab}"))

        quick_switch_layout.addWidget(self._header_chat_pill)
        quick_switch_layout.addWidget(self._header_settings_pill)
        title_row.addWidget(quick_switch_frame, 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch(1)

        self._header_subtitle_label = QLabel("Personal studio tools & local .env API keys.")
        self._header_subtitle_label.setObjectName("HeaderSubtitleLabel")
        self._header_subtitle_label.setWordWrap(True)

        left_column.addLayout(title_row)
        left_column.addWidget(self._header_subtitle_label)
        header_layout.addLayout(left_column, 1)

        right_actions = QHBoxLayout()
        right_actions.setContentsMargins(0, 0, 0, 0)
        right_actions.setSpacing(6)

        self._scroll_explain_button = QPushButton("Scroll && Explain")
        self._scroll_explain_button.setObjectName("HeaderSkyPillButton")
        self._scroll_explain_button.setMinimumWidth(136)
        self._scroll_explain_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._scroll_explain_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scroll_explain_button.clicked.connect(self.scrollExplainRequested.emit)
        right_actions.addWidget(self._scroll_explain_button)

        self._lasso_mode_button = QPushButton("Lasso")
        self._lasso_mode_button.setObjectName("HeaderSecondaryButton")
        self._lasso_mode_button.setMinimumWidth(68)
        self._lasso_mode_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._lasso_mode_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lasso_mode_button.clicked.connect(self.lassoModeRequested.emit)
        right_actions.addWidget(self._lasso_mode_button)

        self._demo_drawing_button = QPushButton("Demo")
        self._demo_drawing_button.setObjectName("HeaderSecondaryButton")
        self._demo_drawing_button.setMinimumWidth(68)
        self._demo_drawing_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._demo_drawing_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._demo_drawing_button.clicked.connect(self.demoDrawingRequested.emit)
        right_actions.addWidget(self._demo_drawing_button)

        self._byok_status_pill = QPushButton("Keys Ready")
        self._byok_status_pill.setObjectName("HeaderSecondaryButton")
        self._byok_status_pill.setMinimumWidth(96)
        self._byok_status_pill.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._byok_status_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self._byok_status_pill.clicked.connect(lambda: self.open_tab("settings:account"))
        right_actions.addWidget(self._byok_status_pill)

        self._drawer_close_button = QPushButton("×")
        self._drawer_close_button.setObjectName("HeaderCloseButton")
        self._drawer_close_button.setFixedSize(28, 28)
        self._drawer_close_button.setToolTip("Collapse back into Notch (Esc)")
        self._drawer_close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._drawer_close_button.clicked.connect(self.hide)
        right_actions.addWidget(self._drawer_close_button)

        header_layout.addLayout(right_actions, 0)
        return header_widget

    def _wrap_scroll_page(self, content_widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setObjectName("PageScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(content_widget)
        return scroll_area

    def _create_page_canvas(self) -> tuple[QWidget, QVBoxLayout]:
        page_widget = QWidget()
        page_widget.setObjectName("PageCanvas")
        page_layout = QVBoxLayout(page_widget)
        page_layout.setContentsMargins(24, 18, 24, 24)
        page_layout.setSpacing(10)
        return page_widget, page_layout

    def _build_all_settings_pages(self) -> None:
        page_builders: Dict[str, Callable[[], QWidget]] = {
            "account": self._build_account_page,
            "general": self._build_general_page,
            "voice": self._build_voice_page,
            "microphone": self._build_microphone_page,
            "dictation": self._build_dictation_page,
            "shortcuts": self._build_shortcuts_page,
            "cursor": self._build_cursor_page,
            "integrations": self._build_integrations_page,
            "agents": self._build_agents_page,
            "routines": self._build_routines_page,
        }
        for tab_key, builder_fn in page_builders.items():
            scroll_page = self._wrap_scroll_page(builder_fn())
            self._subtab_pages[tab_key] = scroll_page
            self._settings_stack.addWidget(scroll_page)

    # -- Settings Pages ------------------------------------------------------

    def _build_account_page(self) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()

        # PERSONAL STUDIO QUICK LAUNCH & PROVIDER HEALTH
        page_layout.addWidget(_build_section_label("PERSONAL STUDIO"))
        plan_card = _build_card_frame()
        plan_layout = QVBoxLayout(plan_card)
        plan_layout.setContentsMargins(14, 11, 14, 12)
        plan_layout.setSpacing(8)

        plan_title = QLabel("Personal Visual Companion (Local .env Keys)")
        plan_title.setObjectName("CardHeadlineLabel")
        plan_layout.addWidget(plan_title)

        plan_subtitle = QLabel(
            "Launch Scroll & Explain on long ChatGPT answers, drag a neon lasso for zoomed close-ups, or test your 2x screen ink."
        )
        plan_subtitle.setObjectName("CardMutedLabel")
        plan_subtitle.setWordWrap(True)
        plan_layout.addWidget(plan_subtitle)

        studio_actions_row = QHBoxLayout()
        studio_actions_row.setSpacing(8)

        btn_scroll = QPushButton("Scroll && Explain")
        btn_scroll.setObjectName("PrimaryActionButton")
        btn_scroll.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_scroll.clicked.connect(self.scrollExplainRequested.emit)

        btn_lasso = QPushButton("Lasso && Explain")
        btn_lasso.setObjectName("SecondaryActionButton")
        btn_lasso.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_lasso.clicked.connect(self.lassoModeRequested.emit)

        btn_clear_ink = QPushButton("Clear Screen Ink")
        btn_clear_ink.setObjectName("SecondaryActionButton")
        btn_clear_ink.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear_ink.clicked.connect(self.clearOverlayRequested.emit)

        studio_actions_row.addWidget(btn_scroll)
        studio_actions_row.addWidget(btn_lasso)
        studio_actions_row.addWidget(btn_clear_ink)
        studio_actions_row.addStretch(1)
        plan_layout.addLayout(studio_actions_row)

        status_row = QHBoxLayout()
        status_row.setSpacing(12)
        self._plan_google_status = self._create_plan_meter_block("Google Gemini (Vision)", "Checking...")
        self._plan_river_status = self._create_plan_meter_block("River AI (Vision)", "Checking...")
        self._plan_assembly_status = self._create_plan_meter_block("AssemblyAI (STT)", "Checking...")
        self._plan_voice_status = self._create_plan_meter_block("AssemblyAI (Voice TTS)", "Checking...")
        status_row.addWidget(self._plan_google_status["container"], 1)
        status_row.addWidget(self._plan_river_status["container"], 1)
        status_row.addWidget(self._plan_assembly_status["container"], 1)
        status_row.addWidget(self._plan_voice_status["container"], 1)
        plan_layout.addLayout(status_row)
        page_layout.addWidget(plan_card)

        # API KEYS (.ENV) card with masked EchoMode.Password fields + eye toggle + Save
        page_layout.addSpacing(2)
        page_layout.addWidget(_build_section_label("API KEYS (.ENV)"))
        keys_card = _build_card_frame()
        keys_layout = QVBoxLayout(keys_card)
        keys_layout.setContentsMargins(14, 10, 14, 10)
        keys_layout.setSpacing(7)

        for index, field_def in enumerate(SECRET_FIELD_DEFINITIONS):
            if index > 0:
                keys_layout.addWidget(_build_hairline_divider())

            secret_name = field_def["secret_name"]
            row_container = QWidget()
            row_vbox = QVBoxLayout(row_container)
            row_vbox.setContentsMargins(0, 1, 0, 1)
            row_vbox.setSpacing(4)

            top_meta_row = QHBoxLayout()
            top_meta_row.setSpacing(8)
            title_col = QVBoxLayout()
            title_col.setSpacing(1)
            key_title = QLabel(field_def["display_title"])
            key_title.setObjectName("RowTitleLabel")
            key_subtitle = QLabel(field_def["subtitle"])
            key_subtitle.setObjectName("RowSubtitleLabel")
            title_col.addWidget(key_title)
            title_col.addWidget(key_subtitle)
            top_meta_row.addLayout(title_col, 1)

            status_badge = QLabel("Missing")
            status_badge.setObjectName("SecretStatusBadge")
            self._secret_status_badges[secret_name] = status_badge
            top_meta_row.addWidget(status_badge, 0, Qt.AlignmentFlag.AlignVCenter)
            row_vbox.addLayout(top_meta_row)

            input_row = QHBoxLayout()
            input_row.setSpacing(6)

            key_input = QLineEdit()
            key_input.setObjectName("SettingsLineEdit")
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            key_input.setPlaceholderText(field_def["placeholder"])
            key_input.returnPressed.connect(lambda target_secret=secret_name: self._save_secret_field(target_secret))
            self._secret_inputs[secret_name] = key_input
            input_row.addWidget(key_input, 1)

            eye_button = EyeToggleButton()
            eye_button.toggled.connect(
                lambda is_revealed, target_input=key_input: target_input.setEchoMode(
                    QLineEdit.EchoMode.Normal if is_revealed else QLineEdit.EchoMode.Password
                )
            )
            input_row.addWidget(eye_button)

            save_button = QPushButton("Save")
            save_button.setObjectName("SecondaryActionButton")
            save_button.setFixedWidth(68)
            save_button.setCursor(Qt.CursorShape.PointingHandCursor)
            save_button.clicked.connect(lambda _checked=False, target_secret=secret_name: self._save_secret_field(target_secret))
            self._secret_save_buttons[secret_name] = save_button
            input_row.addWidget(save_button)

            row_vbox.addLayout(input_row)
            keys_layout.addWidget(row_container)

        page_layout.addWidget(keys_card)

        # ACCOUNT card matching bottom card in 2.15.41 AM.jpeg
        page_layout.addSpacing(4)
        page_layout.addWidget(_build_section_label("ACCOUNT"))
        account_card = _build_card_frame()
        account_layout = QVBoxLayout(account_card)
        account_layout.setContentsMargins(16, 12, 16, 12)
        account_layout.setSpacing(8)

        self._account_display_name_input = QLineEdit()
        self._account_display_name_input.setObjectName("SettingsLineEdit")
        self._account_display_name_input.setFixedWidth(210)
        self._account_display_name_input.setPlaceholderText("Your name")
        self._account_display_name_input.editingFinished.connect(
            lambda: self._update_display_name_setting(self._account_display_name_input.text())
        )
        account_layout.addWidget(
            _build_setting_row(
                "Display name",
                "Used in the notch greeting and your Account badge.",
                self._account_display_name_input,
            )
        )

        account_layout.addWidget(_build_hairline_divider())

        quit_row = QHBoxLayout()
        quit_row.setContentsMargins(0, 4, 0, 2)
        quit_button = QPushButton("Quit Jarvis")
        quit_button.setObjectName("DangerTextButton")
        quit_button.setCursor(Qt.CursorShape.PointingHandCursor)
        quit_button.clicked.connect(self.quitRequested.emit)
        quit_row.addWidget(quit_button, 0, Qt.AlignmentFlag.AlignLeft)
        quit_row.addStretch(1)
        account_layout.addLayout(quit_row)

        page_layout.addWidget(account_card)
        page_layout.addStretch(1)
        return page_widget

    def _create_plan_meter_block(self, label_text: str, status_text: str) -> Dict[str, Any]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        title_label = QLabel(label_text)
        title_label.setObjectName("PlanMeterTitle")
        status_label = QLabel(status_text)
        status_label.setObjectName("PlanMeterValue")

        bar_frame = QFrame()
        bar_frame.setObjectName("PlanMeterTrack")
        bar_frame.setFixedHeight(4)
        bar_layout = QHBoxLayout(bar_frame)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_fill = QFrame()
        bar_fill.setObjectName("PlanMeterFill")
        bar_fill.setFixedHeight(4)
        bar_layout.addWidget(bar_fill)

        layout.addWidget(title_label)
        layout.addWidget(status_label)
        layout.addWidget(bar_frame)
        return {"container": container, "status_label": status_label, "bar_fill": bar_fill}

    def _build_general_page(self) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()

        page_layout.addWidget(_build_section_label("PROFILE & MODEL"))
        profile_card = _build_card_frame()
        profile_layout = QVBoxLayout(profile_card)
        profile_layout.setContentsMargins(16, 12, 16, 12)
        profile_layout.setSpacing(8)

        self._general_display_name_input = QLineEdit()
        self._general_display_name_input.setObjectName("SettingsLineEdit")
        self._general_display_name_input.setFixedWidth(220)
        self._general_display_name_input.setPlaceholderText("Your name")
        self._general_display_name_input.editingFinished.connect(
            lambda: self._update_display_name_setting(self._general_display_name_input.text())
        )
        profile_layout.addWidget(
            _build_setting_row(
                "Display name",
                "Shown in your top-notch welcome greeting.",
                self._general_display_name_input,
            )
        )

        profile_layout.addWidget(_build_hairline_divider())

        self._vision_model_combo = ChevronComboBox()
        self._vision_model_combo.setObjectName("SettingsComboBox")
        self._vision_model_combo.setMinimumWidth(250)
        self._vision_model_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        for model_choice in jarvis_config.CHAT_MODEL_CHOICES:
            self._vision_model_combo.addItem(model_choice["name"], model_choice["model_id"])
        self._vision_model_combo.currentIndexChanged.connect(self._on_vision_model_changed)
        profile_layout.addWidget(
            _build_setting_row(
                "Vision Model",
                "Model used for live screen understanding and drawing commands.",
                self._vision_model_combo,
            )
        )
        page_layout.addWidget(profile_card)

        page_layout.addSpacing(4)
        page_layout.addWidget(_build_section_label("BEHAVIOR & SOUNDS"))
        toggles_card = _build_card_frame()
        toggles_layout = QVBoxLayout(toggles_card)
        toggles_layout.setContentsMargins(16, 12, 16, 12)
        toggles_layout.setSpacing(8)

        self._typed_voice_toggle = ToggleSwitch()
        self._typed_voice_toggle.toggled.connect(
            lambda checked: self._set_setting_from_ui("typed_turns_respond_with_voice", checked)
        )
        toggles_layout.addWidget(
            _build_setting_row(
                "Respond with voice to typed questions",
                "Read answers aloud even when you submit a typed question from the notch or Chat.",
                self._typed_voice_toggle,
            )
        )

        toggles_layout.addWidget(_build_hairline_divider())

        self._screen_recordings_toggle = ToggleSwitch()
        self._screen_recordings_toggle.toggled.connect(
            lambda checked: self._set_setting_from_ui("show_in_screen_recordings", checked)
        )
        toggles_layout.addWidget(
            _build_setting_row(
                "Show in screen recordings",
                "Turn off to exclude Jarvis overlays from screen shares and capture tools.",
                self._screen_recordings_toggle,
            )
        )

        toggles_layout.addWidget(_build_hairline_divider())

        self._earcons_toggle = ToggleSwitch()
        self._earcons_toggle.toggled.connect(lambda checked: self._set_setting_from_ui("earcons_enabled", checked))
        toggles_layout.addWidget(
            _build_setting_row(
                "Earcons & chimes",
                "Play gentle audio chimes when listening starts, completes, or errors.",
                self._earcons_toggle,
            )
        )

        toggles_layout.addWidget(_build_hairline_divider())

        self._drawing_sounds_toggle = ToggleSwitch()
        self._drawing_sounds_toggle.toggled.connect(
            lambda checked: self._set_setting_from_ui("drawing_sounds_enabled", checked)
        )
        toggles_layout.addWidget(
            _build_setting_row(
                "Drawing pen sounds",
                "Play tactile pen, highlighter, and paper placement foley during screen drawings.",
                self._drawing_sounds_toggle,
            )
        )
        page_layout.addWidget(toggles_card)

        page_layout.addSpacing(4)
        page_layout.addWidget(_build_section_label("REMEMBERED ANSWER STYLE"))
        style_card = _build_card_frame()
        style_layout = QVBoxLayout(style_card)
        style_layout.setContentsMargins(16, 12, 16, 14)
        style_layout.setSpacing(8)

        style_hint = QLabel(
            "Ask Jarvis for shorter answers, yes/no first, or simpler words and it remembers across sessions."
        )
        style_hint.setObjectName("CardMutedLabel")
        style_hint.setWordWrap(True)
        style_layout.addWidget(style_hint)

        self._style_rules_container = QVBoxLayout()
        self._style_rules_container.setContentsMargins(0, 2, 0, 2)
        self._style_rules_container.setSpacing(6)
        style_layout.addLayout(self._style_rules_container)

        add_rule_row = QHBoxLayout()
        add_rule_row.setSpacing(8)
        self._new_style_rule_input = QLineEdit()
        self._new_style_rule_input.setObjectName("SettingsLineEdit")
        self._new_style_rule_input.setPlaceholderText("Add a style rule (e.g. Start with a direct yes or no)...")
        self._new_style_rule_input.returnPressed.connect(self._add_style_rule)
        add_rule_button = QPushButton("Add")
        add_rule_button.setObjectName("SecondaryActionButton")
        add_rule_button.setCursor(Qt.CursorShape.PointingHandCursor)
        add_rule_button.clicked.connect(self._add_style_rule)
        add_rule_row.addWidget(self._new_style_rule_input, 1)
        add_rule_row.addWidget(add_rule_button)
        style_layout.addLayout(add_rule_row)

        page_layout.addWidget(style_card)
        page_layout.addStretch(1)
        return page_widget

    def _build_voice_page(self) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()

        page_layout.addWidget(_build_section_label("PLAYBACK"))
        playback_card = _build_card_frame()
        playback_layout = QVBoxLayout(playback_card)
        playback_layout.setContentsMargins(16, 12, 16, 12)
        playback_layout.setSpacing(8)

        slider_container = QWidget()
        slider_row = QHBoxLayout(slider_container)
        slider_row.setContentsMargins(0, 0, 0, 0)
        slider_row.setSpacing(10)

        self._speech_speed_slider = PillSlider()
        self._speech_speed_slider.setObjectName("SettingsSlider")
        self._speech_speed_slider.setRange(70, 120)
        self._speech_speed_slider.setSingleStep(5)
        self._speech_speed_slider.setPageStep(5)
        self._speech_speed_slider.setFixedWidth(160)
        self._speech_speed_slider.valueChanged.connect(self._on_speech_speed_slider_changed)

        self._speech_speed_value_label = QLabel("1.00x")
        self._speech_speed_value_label.setObjectName("ValuePillLabel")
        self._speech_speed_value_label.setFixedWidth(48)
        self._speech_speed_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        slider_row.addWidget(self._speech_speed_slider)
        slider_row.addWidget(self._speech_speed_value_label)

        playback_layout.addWidget(
            _build_setting_row(
                "Speech speed",
                "Adjust how fast Jarvis speaks (0.7x to 1.2x).",
                slider_container,
            )
        )

        playback_layout.addWidget(_build_hairline_divider())

        self._captions_toggle = ToggleSwitch()
        self._captions_toggle.toggled.connect(lambda checked: self._set_setting_from_ui("captions_enabled", checked))
        playback_layout.addWidget(
            _build_setting_row(
                "Captions while Jarvis talks",
                "Show synchronized live subtitles at the bottom of the screen during speech.",
                self._captions_toggle,
            )
        )
        page_layout.addWidget(playback_card)

        page_layout.addSpacing(4)
        page_layout.addWidget(_build_section_label("VOICES"))
        voices_card = _build_card_frame()
        voices_layout = QVBoxLayout(voices_card)
        voices_layout.setContentsMargins(10, 10, 10, 10)
        voices_layout.setSpacing(4)

        voices_hint = QLabel("Click any voice below to select it and hear an immediate voice preview.")
        voices_hint.setObjectName("CardMutedLabel")
        voices_hint.setContentsMargins(6, 0, 6, 4)
        voices_layout.addWidget(voices_hint)

        for voice_entry in jarvis_config.ASSEMBLYAI_TTS_VOICES:
            voice_row = VoiceChoiceRow(voice_entry)
            voice_row.clicked.connect(self._on_voice_row_clicked)
            self._voice_rows[voice_entry["voice_id"]] = voice_row
            voices_layout.addWidget(voice_row)

        page_layout.addWidget(voices_card)
        page_layout.addStretch(1)
        return page_widget

    def _build_microphone_page(self) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()

        page_layout.addWidget(_build_section_label("INPUT DEVICE"))
        mic_card = _build_card_frame()
        mic_layout = QVBoxLayout(mic_card)
        mic_layout.setContentsMargins(16, 14, 16, 14)
        mic_layout.setSpacing(12)

        self._microphone_combo = ChevronComboBox()
        self._microphone_combo.setObjectName("SettingsComboBox")
        self._microphone_combo.setMinimumWidth(260)
        self._microphone_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._microphone_combo.addItem("System Default Microphone", None)
        self._microphone_combo.currentIndexChanged.connect(self._on_microphone_device_changed)

        mic_layout.addWidget(
            _build_setting_row(
                "Input device",
                "Select which Windows microphone Jarvis listens to for push-to-talk and dictation.",
                self._microphone_combo,
            )
        )

        mic_layout.addWidget(_build_hairline_divider())

        meter_section = QVBoxLayout()
        meter_section.setSpacing(8)
        meter_header = QHBoxLayout()
        meter_title = QLabel("Live audio input level")
        meter_title.setObjectName("RowTitleLabel")
        self._mic_level_percent_label = QLabel("0%")
        self._mic_level_percent_label.setObjectName("ValuePillLabel")
        self._mic_level_percent_label.setFixedWidth(48)
        self._mic_level_percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meter_header.addWidget(meter_title, 1)
        meter_header.addWidget(self._mic_level_percent_label)
        meter_section.addLayout(meter_header)

        self._mic_level_meter = MicrophoneLevelMeter()
        meter_section.addWidget(self._mic_level_meter)

        test_button_row = QHBoxLayout()
        self._test_mic_button = QPushButton("Test Microphone")
        self._test_mic_button.setObjectName("PrimaryActionButton")
        self._test_mic_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_mic_button.clicked.connect(self.microphoneTestRequested.emit)
        test_hint = QLabel("Speaks or records briefly to verify live input waveform response.")
        test_hint.setObjectName("CardMutedLabel")
        test_button_row.addWidget(self._test_mic_button)
        test_button_row.addWidget(test_hint, 1)
        meter_section.addLayout(test_button_row)

        mic_layout.addLayout(meter_section)
        page_layout.addWidget(mic_card)
        page_layout.addStretch(1)
        return page_widget

    def _build_dictation_page(self) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()

        # Changelog 1.1.5: Custom Dictionary sits ABOVE Language.
        page_layout.addWidget(_build_section_label("CUSTOM DICTIONARY"))
        dictionary_card = _build_card_frame()
        dictionary_layout = QVBoxLayout(dictionary_card)
        dictionary_layout.setContentsMargins(16, 14, 16, 14)
        dictionary_layout.setSpacing(10)

        dictionary_hint = QLabel(
            "Add names, technical terms, or acronyms so Jarvis recognizes and formats them reliably."
        )
        dictionary_hint.setObjectName("CardMutedLabel")
        dictionary_hint.setWordWrap(True)
        dictionary_layout.addWidget(dictionary_hint)

        add_word_row = QHBoxLayout()
        add_word_row.setSpacing(8)
        self._custom_word_input = QLineEdit()
        self._custom_word_input.setObjectName("SettingsLineEdit")
        self._custom_word_input.setPlaceholderText("Add a custom word or phrase...")
        self._custom_word_input.returnPressed.connect(self._add_custom_dictionary_word)

        add_word_button = QPushButton("Add")
        add_word_button.setObjectName("PrimaryActionButton")
        add_word_button.setCursor(Qt.CursorShape.PointingHandCursor)
        add_word_button.clicked.connect(self._add_custom_dictionary_word)

        add_word_row.addWidget(self._custom_word_input, 1)
        add_word_row.addWidget(add_word_button)
        dictionary_layout.addLayout(add_word_row)

        self._dictionary_chips_host = QWidget()
        self._dictionary_chips_layout = FlowLayout(self._dictionary_chips_host, spacing=8)
        self._dictionary_chips_layout.setContentsMargins(0, 4, 0, 2)
        dictionary_layout.addWidget(self._dictionary_chips_host)
        page_layout.addWidget(dictionary_card)

        page_layout.addSpacing(4)
        page_layout.addWidget(_build_section_label("LANGUAGE"))
        language_card = _build_card_frame()
        language_layout = QVBoxLayout(language_card)
        language_layout.setContentsMargins(16, 12, 16, 12)
        language_layout.setSpacing(8)

        self._dictation_language_combo = ChevronComboBox()
        self._dictation_language_combo.setObjectName("SettingsComboBox")
        self._dictation_language_combo.setMinimumWidth(240)
        self._dictation_language_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        for language_entry in jarvis_config.DICTATION_LANGUAGE_CHOICES:
            self._dictation_language_combo.addItem(language_entry["name"], language_entry["code"])
        self._dictation_language_combo.currentIndexChanged.connect(self._on_dictation_language_changed)

        language_layout.addWidget(
            _build_setting_row(
                "Dictation language",
                "Select English or multilingual auto-detection for streamed speech.",
                self._dictation_language_combo,
            )
        )
        page_layout.addWidget(language_card)
        page_layout.addStretch(1)
        return page_widget

    def _build_shortcuts_page(self) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()

        page_layout.addWidget(_build_section_label("GLOBAL KEYBOARD SHORTCUTS"))
        shortcuts_card = _build_card_frame()
        shortcuts_layout = QVBoxLayout(shortcuts_card)
        shortcuts_layout.setContentsMargins(16, 12, 16, 12)
        shortcuts_layout.setSpacing(8)

        for index, shortcut_entry in enumerate(FIVE_CORE_SHORTCUTS):
            if index > 0:
                shortcuts_layout.addWidget(_build_hairline_divider())

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 4, 0, 4)
            row_layout.setSpacing(12)

            text_column = QVBoxLayout()
            text_column.setSpacing(2)
            title_label = QLabel(shortcut_entry["title"])
            title_label.setObjectName("RowTitleLabel")
            desc_label = QLabel(shortcut_entry["description"])
            desc_label.setObjectName("RowSubtitleLabel")
            desc_label.setWordWrap(True)
            text_column.addWidget(title_label)
            text_column.addWidget(desc_label)
            row_layout.addLayout(text_column, 1)

            keys_row = QHBoxLayout()
            keys_row.setSpacing(4)
            for key_index, key_text in enumerate(shortcut_entry["keys"]):
                if key_index > 0:
                    plus_label = QLabel("+")
                    plus_label.setObjectName("ShortcutPlusLabel")
                    keys_row.addWidget(plus_label)
                keycap_label = QLabel(key_text)
                keycap_label.setObjectName("KeycapBadge")
                keys_row.addWidget(keycap_label)

            mode_label = QLabel(shortcut_entry["mode_label"])
            mode_label.setObjectName("ShortcutModeLabel")
            keys_row.addSpacing(6)
            keys_row.addWidget(mode_label)
            row_layout.addLayout(keys_row)

            test_button = QPushButton("Test")
            test_button.setObjectName("SecondaryActionButton")
            test_button.setFixedWidth(56)
            test_button.setCursor(Qt.CursorShape.PointingHandCursor)
            shortcut_id = shortcut_entry["shortcut_id"]
            test_button.clicked.connect(
                lambda _checked=False, target_shortcut_id=shortcut_id: self.shortcutTestRequested.emit(target_shortcut_id)
            )
            row_layout.addWidget(test_button)
            shortcuts_layout.addWidget(row_widget)

        page_layout.addWidget(shortcuts_card)
        page_layout.addStretch(1)
        return page_widget

    def _build_cursor_page(self) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()

        page_layout.addWidget(_build_section_label("CURSOR COLOR"))
        color_card = _build_card_frame()
        color_layout = QHBoxLayout(color_card)
        color_layout.setContentsMargins(16, 14, 16, 14)
        color_layout.setSpacing(16)

        left_column = QVBoxLayout()
        left_column.setSpacing(8)
        color_title = QLabel("Companion cursor color")
        color_title.setObjectName("RowTitleLabel")
        color_subtitle = QLabel("Colors Jarvis's floating pointer symbol and thinking dots.")
        color_subtitle.setObjectName("RowSubtitleLabel")
        left_column.addWidget(color_title)
        left_column.addWidget(color_subtitle)

        dots_row = QHBoxLayout()
        dots_row.setContentsMargins(0, 4, 0, 0)
        dots_row.setSpacing(14)

        for color_name, color_hex in jarvis_theme.CURSOR_COLOR_CHOICES.items():
            swatch_column = QVBoxLayout()
            swatch_column.setSpacing(4)
            swatch_column.setAlignment(Qt.AlignmentFlag.AlignCenter)

            dot_button = ColorDotButton(color_name=color_name, color_hex=color_hex)
            dot_button.clicked.connect(
                lambda _checked=False, selected_hex=color_hex: self._on_cursor_color_selected(selected_hex)
            )
            self._cursor_color_buttons[color_hex.upper()] = dot_button

            name_label = QLabel(color_name)
            name_label.setObjectName("ColorDotCaptionLabel")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            swatch_column.addWidget(dot_button, 0, Qt.AlignmentFlag.AlignCenter)
            swatch_column.addWidget(name_label, 0, Qt.AlignmentFlag.AlignCenter)
            dots_row.addLayout(swatch_column)

        dots_row.addStretch(1)
        left_column.addLayout(dots_row)
        color_layout.addLayout(left_column, 1)

        self._cursor_preview_widget = CursorPreviewWidget()
        color_layout.addWidget(self._cursor_preview_widget, 0, Qt.AlignmentFlag.AlignVCenter)
        page_layout.addWidget(color_card)

        page_layout.addSpacing(4)
        page_layout.addWidget(_build_section_label("BEHAVIOR"))
        behavior_card = _build_card_frame()
        behavior_layout = QVBoxLayout(behavior_card)
        behavior_layout.setContentsMargins(16, 12, 16, 12)
        behavior_layout.setSpacing(8)

        self._hide_cursor_toggle = ToggleSwitch()
        self._hide_cursor_toggle.toggled.connect(
            lambda checked: self._set_setting_from_ui("hide_floating_cursor", checked)
        )
        behavior_layout.addWidget(
            _build_setting_row(
                "Hide floating cursor",
                "Fades the cursor out cleanly ('okay see you <3') instead of parking in the notch.",
                self._hide_cursor_toggle,
            )
        )

        behavior_layout.addWidget(_build_hairline_divider())

        self._screen_highlight_toggle = ToggleSwitch()
        self._screen_highlight_toggle.toggled.connect(
            lambda checked: self._set_setting_from_ui("screen_highlight_enabled", checked)
        )
        behavior_layout.addWidget(
            _build_setting_row(
                "Draw-to-Ask screen highlighter",
                "Drag the mouse while holding Ctrl + Alt to circle or highlight elements for Jarvis.",
                self._screen_highlight_toggle,
            )
        )

        behavior_layout.addWidget(_build_hairline_divider())

        self._shake_detach_toggle = ToggleSwitch()
        self._shake_detach_toggle.toggled.connect(
            lambda checked: self._set_setting_from_ui("shake_to_detach_enabled", checked)
        )
        behavior_layout.addWidget(
            _build_setting_row(
                "Shake cursor to detach",
                "Shake your mouse rapidly to park the companion cursor in place briefly.",
                self._shake_detach_toggle,
            )
        )
        page_layout.addWidget(behavior_card)

        page_layout.addSpacing(4)
        page_layout.addWidget(_build_section_label("SCREEN DRAWING & EXPLANATION"))
        drawing_card = _build_card_frame()
        drawing_layout = QVBoxLayout(drawing_card)
        drawing_layout.setContentsMargins(16, 12, 16, 12)
        drawing_layout.setSpacing(8)

        self._drawing_mode_combo = ChevronComboBox()
        self._drawing_mode_combo.setObjectName("SettingsComboBox")
        self._drawing_mode_combo.setMinimumWidth(260)
        self._drawing_mode_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        for mode_entry in jarvis_config.SCREEN_DRAWING_MODE_CHOICES:
            self._drawing_mode_combo.addItem(mode_entry["name"], mode_entry["mode_id"])
        self._drawing_mode_combo.currentIndexChanged.connect(self._on_drawing_mode_changed)

        drawing_layout.addWidget(
            _build_setting_row(
                "Explanation drawing style",
                "Choose crisp underlines directly under text or include diagram cards and full sketches.",
                self._drawing_mode_combo,
            )
        )

        drawing_layout.addWidget(_build_hairline_divider())

        self._drawing_speed_combo = ChevronComboBox()
        self._drawing_speed_combo.setObjectName("SettingsComboBox")
        self._drawing_speed_combo.setMinimumWidth(260)
        self._drawing_speed_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        for speed_entry in jarvis_config.DRAWING_SPEED_CHOICES:
            self._drawing_speed_combo.addItem(speed_entry["name"], speed_entry["speed_id"])
        self._drawing_speed_combo.currentIndexChanged.connect(self._on_drawing_speed_changed)

        drawing_layout.addWidget(
            _build_setting_row(
                "Drawing stroke pace",
                "Control how slowly lines trace across the screen in sync with pen sounds.",
                self._drawing_speed_combo,
            )
        )
        page_layout.addWidget(drawing_card)

        page_layout.addStretch(1)
        return page_widget

    def _build_integrations_page(self) -> QWidget:
        return self._build_byok_info_page(
            section_title="LOCAL BYOK INTEGRATIONS",
            headline="Direct .env provider pipeline",
            paragraphs=[
                (
                    "Google Gemini Vision & Chat",
                    "Streams multimodal screen reasoning and annotations directly from your GOOGLE_API_KEY (or GEMINI_API_KEY) in .env.",
                ),
                (
                    "River AI Vision & Chat",
                    "Streams responses and screen annotations directly from your RIVER_AI_API key in .env.",
                ),
                (
                    "AssemblyAI Streaming STT",
                    "Streams real-time 16 kHz microphone audio using your ASSEMBLYAI_API_KEY in .env.",
                ),
                (
                    "AssemblyAI Neural Voice TTS",
                    "Synthesizes natural 24 kHz AssemblyAI-TTS-1 speech using your ASSEMBLYAI_API_KEY in .env, with offline Windows SAPI fallback.",
                ),
            ],
        )

    def _build_agents_page(self) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()

        page_layout.addWidget(_build_section_label("DESKTOP & BROWSER AGENT"))
        agent_card = _build_card_frame()
        agent_layout = QVBoxLayout(agent_card)
        agent_layout.setContentsMargins(16, 12, 16, 14)
        agent_layout.setSpacing(10)

        self._agent_actions_toggle = ToggleSwitch()
        self._agent_actions_toggle.toggled.connect(
            lambda checked: self._set_setting_from_ui("agent_actions_enabled", checked)
        )
        agent_layout.addWidget(
            _build_setting_row(
                "Allow desktop & browser actions",
                "Let Jarvis open your browser, visit sites, search the web, launch apps, or press shortcuts on request.",
                self._agent_actions_toggle,
            )
        )

        agent_layout.addWidget(_build_hairline_divider())

        quick_label = QLabel("Quick Agent Test")
        quick_label.setObjectName("RowTitleLabel")
        agent_layout.addWidget(quick_label)

        test_buttons_row = QHBoxLayout()
        test_buttons_row.setSpacing(8)
        quick_actions = [
            ("Open Browser", "open_browser", "https://www.google.com"),
            ("Open ChatGPT", "open_url", "https://chatgpt.com"),
            ("Open Notepad", "open_app", "notepad"),
        ]
        for button_label, action_kind, action_target in quick_actions:
            action_button = QPushButton(button_label)
            action_button.setObjectName("SecondaryActionButton")
            action_button.setCursor(Qt.CursorShape.PointingHandCursor)
            action_button.clicked.connect(
                lambda _checked=False, kind=action_kind, target=action_target: self.agentCommandRequested.emit(
                    kind, target
                )
            )
            test_buttons_row.addWidget(action_button)
        test_buttons_row.addStretch(1)
        agent_layout.addLayout(test_buttons_row)
        page_layout.addWidget(agent_card)

        page_layout.addSpacing(4)
        page_layout.addWidget(_build_section_label("LOCAL ASSISTANT EXECUTION"))
        info_card = _build_card_frame()
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 14, 16, 14)
        info_layout.setSpacing(10)

        headline_label = QLabel("Screen-aware assistance without cloud proxies")
        headline_label.setObjectName("CardHeadlineLabel")
        info_layout.addWidget(headline_label)

        execution_notes = [
            (
                "Interactive Vision Turns",
                "Press Ctrl + Alt anywhere or ask by voice/chat to open sites, explain ChatGPT steps, or draw lines accurately.",
            ),
            (
                "Persistent Style Memory",
                "Preference rules saved under General > Remembered answer style are automatically respected across conversations.",
            ),
            (
                "Local Privacy",
                "Screenshots are captured only on demand and sent directly to your configured River AI endpoint.",
            ),
        ]
        for item_title, item_body in execution_notes:
            info_layout.addWidget(_build_hairline_divider())
            block_title = QLabel(item_title)
            block_title.setObjectName("RowTitleLabel")
            block_body = QLabel(item_body)
            block_body.setObjectName("RowSubtitleLabel")
            block_body.setWordWrap(True)
            info_layout.addWidget(block_title)
            info_layout.addWidget(block_body)

        page_layout.addWidget(info_card)
        page_layout.addStretch(1)
        return page_widget

    def _build_routines_page(self) -> QWidget:
        return self._build_byok_info_page(
            section_title="LOCAL ROUTINES",
            headline="On-demand Windows workflows",
            paragraphs=[
                (
                    "Quick Notch Composer",
                    "Double-tap Ctrl at any time to type a request from anywhere on your desktop.",
                ),
                (
                    "Hands-Free Dictation",
                    "Double-tap Win while holding Ctrl to dictate continuously into notes, docs, or code editors.",
                ),
                (
                    "Zero Background Cloud Daemons",
                    "Because this Windows build runs strictly with your local BYOK keys, no remote cron servers run in the background.",
                ),
            ],
        )

    def _build_byok_info_page(
        self,
        section_title: str,
        headline: str,
        paragraphs: List[tuple[str, str]],
    ) -> QWidget:
        page_widget, page_layout = self._create_page_canvas()
        page_layout.addWidget(_build_section_label(section_title))
        info_card = _build_card_frame()
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 14, 16, 14)
        info_layout.setSpacing(10)

        headline_label = QLabel(headline)
        headline_label.setObjectName("CardHeadlineLabel")
        info_layout.addWidget(headline_label)

        for item_title, item_body in paragraphs:
            info_layout.addWidget(_build_hairline_divider())
            block_title = QLabel(item_title)
            block_title.setObjectName("RowTitleLabel")
            block_body = QLabel(item_body)
            block_body.setObjectName("RowSubtitleLabel")
            block_body.setWordWrap(True)
            info_layout.addWidget(block_title)
            info_layout.addWidget(block_body)

        page_layout.addWidget(info_card)
        page_layout.addStretch(1)
        return page_widget

    # -- Chat View -----------------------------------------------------------

    def _build_chat_page(self) -> QWidget:
        chat_page = QWidget()
        chat_page.setObjectName("ChatPage")
        chat_layout = QVBoxLayout(chat_page)
        chat_layout.setContentsMargins(24, 14, 24, 18)
        chat_layout.setSpacing(12)

        # Top action bar with Stop & Clear Chat
        action_bar = QHBoxLayout()
        action_bar.setContentsMargins(0, 0, 0, 0)
        action_bar.setSpacing(8)

        self._chat_status_label = QLabel("Ready · Ask a question or press Ctrl + Alt anywhere")
        self._chat_status_label.setObjectName("CardMutedLabel")
        action_bar.addWidget(self._chat_status_label, 1)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setObjectName("StopActionButton")
        self._stop_button.setEnabled(False)
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        action_bar.addWidget(self._stop_button)

        self._clear_chat_button = QPushButton("Clear Chat")
        self._clear_chat_button.setObjectName("SecondaryActionButton")
        self._clear_chat_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_chat_button.clicked.connect(self.clear_chat)
        action_bar.addWidget(self._clear_chat_button)

        chat_layout.addLayout(action_bar)

        # Scrollable message feed
        self._chat_scroll_area = QScrollArea()
        self._chat_scroll_area.setObjectName("ChatScrollArea")
        self._chat_scroll_area.setWidgetResizable(True)
        self._chat_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._chat_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._chat_feed_container = QWidget()
        self._chat_feed_container.setObjectName("ChatFeedContainer")
        self._chat_feed_layout = QVBoxLayout(self._chat_feed_container)
        self._chat_feed_layout.setContentsMargins(2, 6, 6, 10)
        self._chat_feed_layout.setSpacing(10)

        self._chat_empty_state = self._build_chat_empty_state()
        self._chat_feed_layout.addWidget(self._chat_empty_state)
        self._chat_feed_layout.addStretch(1)

        self._chat_scroll_area.setWidget(self._chat_feed_container)
        chat_layout.addWidget(self._chat_scroll_area, 1)

        # Bottom composer bar
        composer_frame = QFrame()
        composer_frame.setObjectName("ChatComposerFrame")
        composer_layout = QHBoxLayout(composer_frame)
        composer_layout.setContentsMargins(14, 8, 8, 8)
        composer_layout.setSpacing(8)

        self._chat_input = QLineEdit()
        self._chat_input.setObjectName("ChatComposerInput")
        self._chat_input.setPlaceholderText("Ask Jarvis anything...")
        self._chat_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._chat_input.returnPressed.connect(self._submit_chat_prompt)
        composer_layout.addWidget(self._chat_input, 1)

        self._send_button = QPushButton("Send")
        self._send_button.setObjectName("PrimaryActionButton")
        self._send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_button.clicked.connect(self._submit_chat_prompt)
        composer_layout.addWidget(self._send_button)

        chat_layout.addWidget(composer_frame)
        return chat_page

    def _build_chat_empty_state(self) -> QFrame:
        empty_card = _build_card_frame("ChatEmptyCard")
        empty_layout = QVBoxLayout(empty_card)
        empty_layout.setContentsMargins(22, 22, 22, 22)
        empty_layout.setSpacing(8)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_layout.addWidget(JarvisMarkBadge(46), 0, Qt.AlignmentFlag.AlignCenter)

        welcome_title = QLabel("Ask Jarvis anything")
        welcome_title.setObjectName("CardHeadlineLabel")
        welcome_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(welcome_title)

        welcome_subtitle = QLabel(
            "Type below to chat with screen vision, or hold Ctrl + Alt anywhere on Windows to talk and draw."
        )
        welcome_subtitle.setObjectName("CardMutedLabel")
        welcome_subtitle.setWordWrap(True)
        welcome_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(welcome_subtitle)
        return empty_card

    # -- Stylesheet ----------------------------------------------------------

    def _apply_stylesheet(self) -> None:
        font_family = f'"{jarvis_theme.FONT_UI_FAMILY}", "{jarvis_theme.FONT_UI_FALLBACK_FAMILY}", sans-serif'
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget#MainWindowRoot, QWidget#ContentPane,
            QWidget#PageCanvas, QWidget#ChatPage, QWidget#ChatFeedContainer,
            QScrollArea#PageScrollArea, QScrollArea#ChatScrollArea {{
                background-color: {DARK_PALETTE["background"]};
                color: {DARK_PALETTE["text_primary"]};
                font-family: {font_family};
            }}
            QWidget#MainWindowRoot {{
                border: 1px solid #2A2A34;
            }}
            QFrame#IntegratedNotchHeaderBar {{
                background-color: #050507;
                border-bottom: 1px solid #1E1E26;
            }}
            QPushButton#NotchMemoryPill, QPushButton#NotchCollapsePill {{
                background-color: #121216;
                color: #C4C4CE;
                border: 1px solid #262630;
                border-radius: 9px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton#NotchMemoryPill:hover, QPushButton#NotchCollapsePill:hover {{
                background-color: #1C1C23;
                color: #F4F4F6;
                border-color: #464652;
            }}
            QFrame#SidebarPanel, QScrollArea#SidebarScrollArea, QWidget#SidebarScrollContents {{
                background-color: {DARK_PALETTE["sidebar"]};
                color: {DARK_PALETTE["text_primary"]};
                font-family: {font_family};
            }}
            QFrame#SidebarVerticalDivider, QFrame#HeaderHorizontalDivider {{
                background-color: {DARK_PALETTE["separator"]};
                border: none;
            }}
            QFrame#SidebarSegmentedTrack {{
                background-color: #161618;
                border: 1px solid #2E2E32;
                border-radius: 9px;
            }}
            QPushButton#SegmentedPillButton {{
                background: transparent;
                color: {DARK_PALETTE["text_secondary"]};
                border: none;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#SegmentedPillButton:hover {{
                color: {DARK_PALETTE["text_primary"]};
            }}
            QPushButton#SegmentedPillButton:checked {{
                background-color: #323237;
                color: {DARK_PALETTE["text_primary"]};
            }}
            QLabel#SidebarSectionHeader {{
                color: {DARK_PALETTE["text_tertiary"]};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.8px;
                padding-left: 10px;
            }}
            QLabel#SidebarFooterLabel {{
                color: {DARK_PALETTE["text_tertiary"]};
                font-size: 11px;
                padding-left: 8px;
                padding-top: 4px;
            }}
            QWidget#TopHeaderBar {{
                background-color: {DARK_PALETTE["background"]};
            }}
            QLabel#HeaderTitleLabel {{
                color: {DARK_PALETTE["text_primary"]};
                font-size: 20px;
                font-weight: 700;
            }}
            QLabel#HeaderSubtitleLabel {{
                color: {DARK_PALETTE["text_secondary"]};
                font-size: 12px;
            }}
            QFrame#HeaderQuickSwitchFrame {{
                background-color: #252528;
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 8px;
            }}
            QPushButton#HeaderTogglePill {{
                background: transparent;
                color: {DARK_PALETTE["text_secondary"]};
                border: none;
                border-radius: 6px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton#HeaderTogglePill:hover {{
                color: {DARK_PALETTE["text_primary"]};
            }}
            QPushButton#HeaderTogglePill:checked {{
                background-color: #27272E;
                color: #F4F4F6;
                border: 1px solid #3F3F46;
            }}
            QPushButton#HeaderSecondaryButton {{
                background-color: {DARK_PALETTE["card"]};
                color: {DARK_PALETTE["text_primary"]};
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 14px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#HeaderSecondaryButton:hover {{
                background-color: {DARK_PALETTE["card_hover"]};
                border-color: #52525A;
            }}
            QPushButton#HeaderSkyPillButton {{
                background-color: #27272E;
                color: #F4F4F6;
                border: 1px solid #52525B;
                border-radius: 14px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#HeaderSkyPillButton:hover {{
                background-color: #34343D;
                border-color: #71717A;
            }}
            QPushButton#HeaderCloseButton {{
                background-color: #18181D;
                color: #A1A1AA;
                border: 1px solid #2E2E36;
                border-radius: 14px;
                font-size: 15px;
                font-weight: 600;
                padding: 0px;
            }}
            QPushButton#HeaderCloseButton:hover {{
                background-color: #27272E;
                color: #F4F4F6;
                border-color: #52525B;
            }}
            QLabel#SectionHeaderLabel {{
                color: {DARK_PALETTE["text_secondary"]};
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.7px;
                padding-left: 2px;
            }}
            QFrame#SettingsCard, QFrame#ChatEmptyCard {{
                background-color: {DARK_PALETTE["card"]};
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 12px;
            }}
            QFrame#CardHairlineDivider {{
                background-color: {DARK_PALETTE["separator"]};
                border: none;
            }}
            QLabel#CardHeadlineLabel {{
                color: {DARK_PALETTE["text_primary"]};
                font-size: 15px;
                font-weight: 700;
            }}
            QLabel#CardMutedLabel, QLabel#RowSubtitleLabel {{
                color: {DARK_PALETTE["text_secondary"]};
                font-size: 12px;
            }}
            QLabel#RowTitleLabel {{
                color: {DARK_PALETTE["text_primary"]};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#PlanMeterTitle {{
                color: {DARK_PALETTE["text_secondary"]};
                font-size: 11px;
                font-weight: 600;
            }}
            QLabel#PlanMeterValue {{
                color: {DARK_PALETTE["text_primary"]};
                font-size: 12px;
                font-weight: 700;
            }}
            QFrame#PlanMeterTrack {{
                background-color: #26262C;
                border-radius: 2px;
            }}
            QFrame#PlanMeterFill[connected="true"] {{
                background-color: #E4E4E7;
                border-radius: 2px;
            }}
            QFrame#PlanMeterFill[connected="false"] {{
                background-color: #F5A623;
                border-radius: 2px;
            }}
            QLabel#SecretStatusBadge {{
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 600;
                background-color: #38383E;
                color: {DARK_PALETTE["text_secondary"]};
            }}
            QLabel#SecretStatusBadge[configured="true"] {{
                background-color: rgba(38, 194, 129, 44);
                color: #4CE0A2;
            }}
            QLabel#SecretStatusBadge[configured="false"] {{
                background-color: rgba(245, 166, 35, 42);
                color: #FFC35A;
            }}
            QLineEdit#SettingsLineEdit {{
                background-color: #16161B;
                color: {DARK_PALETTE["text_primary"]};
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 8px;
                padding: 5px 10px;
                font-size: 13px;
                selection-background-color: #3F3F46;
            }}
            QLineEdit#SettingsLineEdit:focus {{
                border: 1px solid #71717A;
            }}
            QComboBox#SettingsComboBox {{
                background-color: #16161B;
                color: {DARK_PALETTE["text_primary"]};
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 8px;
                padding: 6px 28px 6px 12px;
                font-size: 13px;
                min-height: 20px;
            }}
            QComboBox#SettingsComboBox:hover {{
                border-color: #565660;
            }}
            QComboBox#SettingsComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox#SettingsComboBox QAbstractItemView {{
                background-color: #1A1A20;
                color: {DARK_PALETTE["text_primary"]};
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 8px;
                selection-background-color: #2E2E38;
                selection-color: #F4F4F6;
                outline: 0;
                padding: 4px;
            }}
            QLabel#ValuePillLabel {{
                background-color: #18181D;
                color: {DARK_PALETTE["text_primary"]};
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 7px;
                padding: 3px 6px;
                font-size: 12px;
                font-weight: 600;
            }}
            QFrame#VoiceChoiceRow {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 9px;
            }}
            QFrame#VoiceChoiceRow:hover {{
                background-color: #222229;
            }}
            QFrame#VoiceChoiceRow[selected="true"] {{
                background-color: #1E1E24;
                border: 1px solid #52525B;
            }}
            QLabel#VoiceNameLabel {{
                color: {DARK_PALETTE["text_primary"]};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#VoiceDescriptionLabel {{
                color: {DARK_PALETTE["text_secondary"]};
                font-size: 11px;
            }}
            QLabel#VoiceProviderBadge {{
                color: {DARK_PALETTE["text_secondary"]};
                background-color: #222226;
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 600;
            }}
            QFrame#DictionaryChipFrame, QFrame#StyleRuleRowFrame {{
                background-color: #202024;
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 10px;
            }}
            QLabel#ChipTextLabel {{
                color: {DARK_PALETTE["text_primary"]};
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton#ChipDeleteButton {{
                background: transparent;
                color: {DARK_PALETTE["text_secondary"]};
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 700;
                padding: 0px 4px;
            }}
            QPushButton#ChipDeleteButton:hover {{
                background-color: rgba(255, 105, 97, 45);
                color: {DARK_PALETTE["danger"]};
            }}
            QLabel#KeycapBadge {{
                background-color: #1F1F23;
                color: {DARK_PALETTE["text_primary"]};
                border: 1px solid #4A4A52;
                border-bottom: 2px solid #4A4A52;
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 700;
            }}
            QLabel#ShortcutPlusLabel, QLabel#ShortcutModeLabel {{
                color: {DARK_PALETTE["text_secondary"]};
                font-size: 11px;
            }}
            QLabel#ColorDotCaptionLabel {{
                color: {DARK_PALETTE["text_secondary"]};
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton#PrimaryActionButton {{
                background-color: #27272E;
                color: #F4F4F6;
                border: 1px solid #52525B;
                border-radius: 8px;
                padding: 7px 14px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#PrimaryActionButton:hover {{
                background-color: #34343D;
                border-color: #71717A;
            }}
            QPushButton#SecondaryActionButton {{
                background-color: #343439;
                color: {DARK_PALETTE["text_primary"]};
                border: 1px solid #484850;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#SecondaryActionButton:hover {{
                background-color: #404047;
                border-color: #5C5C66;
            }}
            QPushButton#StopActionButton {{
                background-color: rgba(255, 105, 97, 38);
                color: {DARK_PALETTE["danger"]};
                border: 1px solid rgba(255, 105, 97, 110);
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#StopActionButton:hover:enabled {{
                background-color: rgba(255, 105, 97, 65);
            }}
            QPushButton#StopActionButton:disabled {{
                background-color: #28282C;
                color: {DARK_PALETTE["text_tertiary"]};
                border-color: {DARK_PALETTE["separator"]};
            }}
            QPushButton#DangerTextButton {{
                background: transparent;
                color: {DARK_PALETTE["danger"]};
                border: none;
                padding: 4px 2px;
                font-size: 13px;
                font-weight: 600;
                text-align: left;
            }}
            QPushButton#DangerTextButton:hover {{
                color: #FF8B85;
            }}
            QFrame#UserChatBubble {{
                background-color: #1C1C22;
                border: 1px solid #3F3F46;
                border-radius: 13px;
            }}
            QFrame#AssistantChatBubble {{
                background-color: {DARK_PALETTE["card"]};
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 13px;
            }}
            QLabel#ChatSenderLabel {{
                color: {DARK_PALETTE["text_primary"]};
                font-size: 12px;
                font-weight: 700;
            }}
            QLabel#ChatGeneratingBadge {{
                color: {DARK_PALETTE["accent_soft"]};
                font-size: 11px;
                font-weight: 600;
            }}
            QLabel#ChatBodyLabel {{
                color: {DARK_PALETTE["text_primary"]};
                font-size: 13px;
            }}
            QFrame#ChatComposerFrame {{
                background-color: #252529;
                border: 1px solid {DARK_PALETTE["separator"]};
                border-radius: 12px;
            }}
            QLineEdit#ChatComposerInput {{
                background: transparent;
                border: none;
                color: {DARK_PALETTE["text_primary"]};
                font-size: 13px;
                padding: 4px 2px;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 2px 0;
            }}
            QScrollBar::handle:vertical {{
                background: #44444B;
                border-radius: 4px;
                min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #585862;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
                height: 0px;
            }}
            """
        )

    # -- Public API ----------------------------------------------------------

    def open_tab(self, tab_name: str) -> None:
        """
        Switches to `'chat'`, `'settings'`, or `'settings:<subtab>'`
        (e.g. `'settings:account'`, `'settings:voice'`, `'settings:cursor'`,
        `'settings:dictation'`, `'settings:microphone'`, `'settings:general'`,
        `'settings:shortcuts'`).
        """
        normalized = (tab_name or "settings:account").strip().lower()
        if normalized == "chat":
            self._active_top_tab = "chat"
            self._main_stack.setCurrentWidget(self._chat_page)
            self._header_title_label.setText("Chat")
            self._header_subtitle_label.setText("Ask Jarvis anything with live screen context.")
            QTimer.singleShot(0, self._focus_chat_input)
        else:
            self._active_top_tab = "settings"
            requested_subtab = self._active_settings_subtab
            if normalized.startswith("settings:"):
                requested_subtab = normalized.split(":", 1)[1].strip() or "account"
            elif normalized != "settings" and normalized in self._subtab_pages:
                requested_subtab = normalized
            if requested_subtab not in self._subtab_pages:
                requested_subtab = "account"

            self._active_settings_subtab = requested_subtab
            self._main_stack.setCurrentWidget(self._settings_stack)
            self._settings_stack.setCurrentWidget(self._subtab_pages[requested_subtab])
            meta = self._subtab_metadata.get(requested_subtab, {"label": "Settings", "subtitle": ""})
            self._header_title_label.setText(meta["label"])
            self._header_subtitle_label.setText(meta["subtitle"])

        is_chat = self._active_top_tab == "chat"
        self._sidebar_chat_pill.setChecked(is_chat)
        self._sidebar_settings_pill.setChecked(not is_chat)
        self._header_chat_pill.setChecked(is_chat)
        self._header_settings_pill.setChecked(not is_chat)

        for key_name, nav_button in self._sidebar_buttons.items():
            nav_button.setChecked((not is_chat) and (key_name == self._active_settings_subtab))

    def append_chat_message(self, role: str, text: str) -> None:
        """Appends a completed user or assistant bubble to the Chat feed."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return
        self._chat_empty_state.setVisible(False)
        if role != "user":
            self._streaming_bubble = None
            self._set_chat_streaming_state(False)

        bubble = ChatMessageBubble(role=role, text=cleaned_text, is_generating=False)
        insert_position = max(0, self._chat_feed_layout.count() - 1)
        self._chat_feed_layout.insertWidget(insert_position, bubble)
        self._scroll_chat_to_bottom()

    def update_streaming_assistant_message(self, text: str, is_generating: bool = True) -> None:
        """Creates or updates the active streaming assistant bubble in place without flicker."""
        self._chat_empty_state.setVisible(False)
        self._set_chat_streaming_state(is_generating)

        if self._streaming_bubble is None:
            self._streaming_bubble = ChatMessageBubble(role="assistant", text=text, is_generating=is_generating)
            insert_position = max(0, self._chat_feed_layout.count() - 1)
            self._chat_feed_layout.insertWidget(insert_position, self._streaming_bubble)
        else:
            self._streaming_bubble.set_content(text=text, is_generating=is_generating)
        self._scroll_chat_to_bottom()

    def finish_streaming_assistant_message(self, final_text: str) -> None:
        """Finalizes the active streaming assistant bubble and re-enables idle Stop state."""
        cleaned_final = (final_text or "").strip()
        if self._streaming_bubble is not None:
            resolved_text = cleaned_final or self._streaming_bubble.current_text()
            self._streaming_bubble.set_content(text=resolved_text, is_generating=False)
            self._streaming_bubble = None
        elif cleaned_final:
            self.append_chat_message("assistant", cleaned_final)
        self._set_chat_streaming_state(False)
        self._scroll_chat_to_bottom()

    def clear_chat(self) -> None:
        """Clears all messages from the Chat feed and restores the welcome state."""
        self._streaming_bubble = None
        self._set_chat_streaming_state(False)
        for item_index in reversed(range(self._chat_feed_layout.count())):
            layout_item = self._chat_feed_layout.itemAt(item_index)
            child_widget = layout_item.widget() if layout_item is not None else None
            if isinstance(child_widget, ChatMessageBubble):
                self._chat_feed_layout.removeWidget(child_widget)
                child_widget.deleteLater()
        self._chat_empty_state.setVisible(True)
        self._focus_chat_input()

    def set_microphone_devices(self, devices: List[Dict[str, Any]]) -> None:
        """Populates Settings > Microphone device dropdown while preserving selection."""
        current_index = self._settings_store.get("microphone_device_index", None)
        self._updating_from_store = True
        try:
            self._microphone_combo.clear()
            self._microphone_combo.addItem("System Default Microphone", None)
            selected_combo_index = 0
            for device_entry in devices or []:
                device_index = device_entry.get("index")
                device_name = str(device_entry.get("name", f"Microphone {device_index}"))
                if device_entry.get("is_default"):
                    device_name = f"{device_name} (Default)"
                self._microphone_combo.addItem(device_name, device_index)
                if current_index is not None and device_index == current_index:
                    selected_combo_index = self._microphone_combo.count() - 1
            self._microphone_combo.setCurrentIndex(selected_combo_index)
        finally:
            self._updating_from_store = False

    def set_microphone_level(self, level: float) -> None:
        """Updates the live microphone level meter (`0.0` .. `1.0`)."""
        clamped = max(0.0, min(1.0, float(level)))
        self._mic_level_meter.set_level(clamped)
        self._mic_level_percent_label.setText(f"{int(round(clamped * 100.0))}%")

    def benchmark_frame_ms(self, iterations: int = 30) -> Dict[str, float]:
        """
        Benchmarks offscreen widget rendering across representative views and returns
        timing statistics in milliseconds (`mean_ms`, `p95_ms`, `max_ms`, `min_ms`).
        """
        safe_iterations = max(1, int(iterations))
        previous_top_tab = self._active_top_tab
        previous_subtab = self._active_settings_subtab
        self.open_tab("chat")
        self.ensurePolished()
        if self.layout() is not None:
            self.layout().activate()
        target_widget = self._main_stack.currentWidget() or self
        frame_pixmap = QPixmap(target_widget.size())
        target_widget.render(frame_pixmap)
        target_widget.render(frame_pixmap)
        frame_samples_ms: List[float] = []

        for _iteration_index in range(safe_iterations):
            frame_start = time.perf_counter()
            target_widget.render(frame_pixmap)
            elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
            frame_samples_ms.append(elapsed_ms)

        if previous_top_tab == "chat":
            self.open_tab("chat")
        else:
            self.open_tab(f"settings:{previous_subtab}")

        sorted_samples = sorted(frame_samples_ms)
        p95_index = min(len(sorted_samples) - 1, max(0, int(math.ceil(0.95 * len(sorted_samples))) - 1))
        return {
            "iterations": float(safe_iterations),
            "mean_ms": round(statistics.fmean(frame_samples_ms), 3),
            "p95_ms": round(sorted_samples[p95_index], 3),
            "max_ms": round(max(frame_samples_ms), 3),
            "min_ms": round(min(frame_samples_ms), 3),
        }

    # -- Settings Store Sync & Handlers --------------------------------------

    def _load_values_from_store(self) -> None:
        self._updating_from_store = True
        try:
            display_name = str(self._settings_store.get("user_display_name", "") or "")
            self._general_display_name_input.setText(display_name)
            self._account_display_name_input.setText(display_name)
            self._refresh_account_avatar(display_name)

            model_id = str(self._settings_store.get("chat_model", ""))
            model_index = self._vision_model_combo.findData(model_id)
            if model_index >= 0:
                self._vision_model_combo.setCurrentIndex(model_index)

            self._typed_voice_toggle.setCheckedInstant(
                bool(self._settings_store.get("typed_turns_respond_with_voice", False))
            )
            self._screen_recordings_toggle.setCheckedInstant(
                bool(self._settings_store.get("show_in_screen_recordings", True))
            )
            self._earcons_toggle.setCheckedInstant(bool(self._settings_store.get("earcons_enabled", True)))
            self._drawing_sounds_toggle.setCheckedInstant(bool(self._settings_store.get("drawing_sounds_enabled", True)))

            speed_value = float(self._settings_store.get("speech_speed", 1.0) or 1.0)
            clamped_speed = max(0.70, min(1.20, speed_value))
            self._speech_speed_slider.setValue(int(round(clamped_speed * 100.0)))
            self._speech_speed_value_label.setText(f"{clamped_speed:.2f}x")
            self._captions_toggle.setCheckedInstant(bool(self._settings_store.get("captions_enabled", False)))

            active_voice_id = str(self._settings_store.get("voice_id", ""))
            self._refresh_voice_selection(active_voice_id)

            language_code = str(self._settings_store.get("dictation_language", "en"))
            lang_index = self._dictation_language_combo.findData(language_code)
            if lang_index >= 0:
                self._dictation_language_combo.setCurrentIndex(lang_index)

            self._rebuild_custom_dictionary_chips(list(self._settings_store.get("dictation_custom_words", []) or []))
            self._rebuild_style_rules_list(list(self._settings_store.get("remember_style_notes", []) or []))

            active_color_hex = str(self._settings_store.get("cursor_color_hex", jarvis_theme.JARVIS_BLUE_HEX)).upper()
            self._refresh_cursor_color_selection(active_color_hex)
            self._hide_cursor_toggle.setCheckedInstant(bool(self._settings_store.get("hide_floating_cursor", False)))
            self._screen_highlight_toggle.setCheckedInstant(
                bool(self._settings_store.get("screen_highlight_enabled", True))
            )
            self._shake_detach_toggle.setCheckedInstant(
                bool(self._settings_store.get("shake_to_detach_enabled", False))
            )
            drawing_mode_id = str(
                self._settings_store.get(
                    "screen_drawing_mode",
                    jarvis_config.DEFAULT_SETTINGS["screen_drawing_mode"],
                )
            )
            drawing_mode_index = self._drawing_mode_combo.findData(drawing_mode_id)
            if drawing_mode_index >= 0:
                self._drawing_mode_combo.setCurrentIndex(drawing_mode_index)

            drawing_speed_id = str(
                self._settings_store.get(
                    "drawing_speed",
                    jarvis_config.DEFAULT_SETTINGS["drawing_speed"],
                )
            )
            drawing_speed_index = self._drawing_speed_combo.findData(drawing_speed_id)
            if drawing_speed_index >= 0:
                self._drawing_speed_combo.setCurrentIndex(drawing_speed_index)

            self._agent_actions_toggle.setCheckedInstant(
                bool(self._settings_store.get("agent_actions_enabled", True))
            )

            for field_def in SECRET_FIELD_DEFINITIONS:
                secret_name = field_def["secret_name"]
                self._secret_inputs[secret_name].setText(self._settings_store.get_secret(secret_name))
            self._refresh_secret_statuses()
        finally:
            self._updating_from_store = False

    def _on_external_setting_changed(self, key_name: str, new_value: Any) -> None:
        if self._updating_from_store:
            return
        self._updating_from_store = True
        try:
            if key_name == "user_display_name":
                text_value = str(new_value or "")
                self._general_display_name_input.setText(text_value)
                self._account_display_name_input.setText(text_value)
                self._refresh_account_avatar(text_value)
            elif key_name == "chat_model":
                model_index = self._vision_model_combo.findData(str(new_value))
                if model_index >= 0:
                    self._vision_model_combo.setCurrentIndex(model_index)
            elif key_name == "typed_turns_respond_with_voice":
                self._typed_voice_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "show_in_screen_recordings":
                self._screen_recordings_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "earcons_enabled":
                self._earcons_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "drawing_sounds_enabled":
                self._drawing_sounds_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "remember_style_notes":
                self._rebuild_style_rules_list(list(new_value or []))
            elif key_name == "speech_speed":
                clamped_speed = max(0.70, min(1.20, float(new_value or 1.0)))
                self._speech_speed_slider.setValue(int(round(clamped_speed * 100.0)))
                self._speech_speed_value_label.setText(f"{clamped_speed:.2f}x")
            elif key_name == "captions_enabled":
                self._captions_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "voice_id":
                self._refresh_voice_selection(str(new_value))
            elif key_name == "dictation_custom_words":
                self._rebuild_custom_dictionary_chips(list(new_value or []))
            elif key_name == "dictation_language":
                lang_index = self._dictation_language_combo.findData(str(new_value))
                if lang_index >= 0:
                    self._dictation_language_combo.setCurrentIndex(lang_index)
            elif key_name == "cursor_color_hex":
                self._refresh_cursor_color_selection(str(new_value).upper())
            elif key_name == "hide_floating_cursor":
                self._hide_cursor_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "screen_highlight_enabled":
                self._screen_highlight_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "shake_to_detach_enabled":
                self._shake_detach_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "screen_drawing_mode":
                drawing_mode_index = self._drawing_mode_combo.findData(str(new_value))
                if drawing_mode_index >= 0:
                    self._drawing_mode_combo.setCurrentIndex(drawing_mode_index)
            elif key_name == "drawing_speed":
                drawing_speed_index = self._drawing_speed_combo.findData(str(new_value))
                if drawing_speed_index >= 0:
                    self._drawing_speed_combo.setCurrentIndex(drawing_speed_index)
            elif key_name == "agent_actions_enabled":
                self._agent_actions_toggle.setCheckedInstant(bool(new_value))
            elif key_name == "show_in_screen_recordings":
                self._screen_recordings_toggle.setCheckedInstant(bool(new_value))
                if self.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
                    jarvis_theme.set_window_excluded_from_capture(int(self.winId()), not bool(new_value))
        finally:
            self._updating_from_store = False

    def _on_external_secret_changed(self, secret_name: str) -> None:
        if secret_name in self._secret_inputs and not self._updating_from_store:
            self._secret_inputs[secret_name].setText(self._settings_store.get_secret(secret_name))
        self._refresh_secret_statuses()

    def _set_setting_from_ui(self, key_name: str, new_value: Any) -> None:
        if self._updating_from_store:
            return
        self._settings_store.set(key_name, new_value)

    def _update_display_name_setting(self, raw_name: str) -> None:
        if self._updating_from_store:
            return
        cleaned_name = raw_name.strip()
        self._updating_from_store = True
        try:
            self._general_display_name_input.setText(cleaned_name)
            self._account_display_name_input.setText(cleaned_name)
            self._refresh_account_avatar(cleaned_name)
        finally:
            self._updating_from_store = False
        self._settings_store.set("user_display_name", cleaned_name)

    def _refresh_account_avatar(self, display_name: str) -> None:
        account_button = self._sidebar_buttons.get("account")
        if account_button is not None:
            initial = display_name.strip()[:1].upper() if display_name.strip() else "R"
            account_button.set_avatar_initial(initial)

    def _save_secret_field(self, secret_name: str) -> None:
        line_edit = self._secret_inputs.get(secret_name)
        if line_edit is None:
            return
        saved_secret_value = line_edit.text().strip()
        self._settings_store.set_secret(secret_name, saved_secret_value)
        if secret_name == jarvis_config.SECRET_GOOGLE_API_KEY and saved_secret_value:
            current_model = str(self._settings_store.get("chat_model", ""))
            if not jarvis_config.is_google_gemini_model(current_model):
                self._settings_store.set("chat_model", jarvis_config.DEFAULT_GOOGLE_CHAT_MODEL)
        elif (
            secret_name == jarvis_config.SECRET_RIVER_API_KEY
            and saved_secret_value
            and (not self._settings_store.has_secret(jarvis_config.SECRET_GOOGLE_API_KEY))
        ):
            current_model = str(self._settings_store.get("chat_model", ""))
            if jarvis_config.is_google_gemini_model(current_model):
                self._settings_store.set("chat_model", "deepseek-ai/DeepSeek-V4.1-Flash")
        self._refresh_secret_statuses()
        save_button = self._secret_save_buttons.get(secret_name)
        if save_button is not None:
            save_button.setText("Saved")
            QTimer.singleShot(1400, lambda target_button=save_button: target_button.setText("Save"))

    def _refresh_secret_statuses(self) -> None:
        configured_count = 0
        for field_def in SECRET_FIELD_DEFINITIONS:
            secret_name = field_def["secret_name"]
            is_configured = self._settings_store.has_secret(secret_name)
            if is_configured:
                configured_count += 1
            badge = self._secret_status_badges.get(secret_name)
            if badge is not None:
                badge.setText("Configured" if is_configured else "Not set")
                badge.setProperty("configured", is_configured)
                badge.style().unpolish(badge)
                badge.style().polish(badge)

        has_google = self._settings_store.has_secret(jarvis_config.SECRET_GOOGLE_API_KEY)
        has_river = self._settings_store.has_secret(jarvis_config.SECRET_RIVER_API_KEY)
        has_assembly = self._settings_store.has_secret(jarvis_config.SECRET_ASSEMBLYAI_API_KEY)

        self._update_plan_meter(
            self._plan_google_status,
            has_google,
            "Connected" if has_google else ("Optional" if has_river else "Missing"),
        )
        self._update_plan_meter(
            self._plan_river_status,
            has_river,
            "Connected" if has_river else ("Optional" if has_google else "Missing"),
        )
        self._update_plan_meter(self._plan_assembly_status, has_assembly, "Connected" if has_assembly else "Missing")
        self._update_plan_meter(self._plan_voice_status, has_assembly, "Connected" if has_assembly else "SAPI Fallback")

        total_keys = len(SECRET_FIELD_DEFINITIONS)
        if (has_google or has_river) and has_assembly:
            self._byok_status_pill.setText("Keys Ready")
        else:
            self._byok_status_pill.setText(f"Setup Keys ({configured_count}/{total_keys})")

    @staticmethod
    def _update_plan_meter(meter_dict: Dict[str, Any], is_connected: bool, status_text: str) -> None:
        meter_dict["status_label"].setText(status_text)
        bar_fill = meter_dict["bar_fill"]
        bar_fill.setProperty("connected", is_connected)
        bar_fill.style().unpolish(bar_fill)
        bar_fill.style().polish(bar_fill)

    def _on_vision_model_changed(self, combo_index: int) -> None:
        if self._updating_from_store or combo_index < 0:
            return
        selected_model_id = self._vision_model_combo.itemData(combo_index)
        if selected_model_id:
            self._settings_store.set("chat_model", str(selected_model_id))

    def _on_speech_speed_slider_changed(self, slider_value: int) -> None:
        speed_float = round(slider_value / 100.0, 2)
        self._speech_speed_value_label.setText(f"{speed_float:.2f}x")
        if not self._updating_from_store:
            self._settings_store.set("speech_speed", speed_float)

    def _on_voice_row_clicked(self, voice_id: str) -> None:
        self._refresh_voice_selection(voice_id)
        self._settings_store.set("voice_id", voice_id)
        self.voicePreviewRequested.emit(voice_id)

    def _refresh_voice_selection(self, active_voice_id: str) -> None:
        for row_voice_id, row_widget in self._voice_rows.items():
            row_widget.set_selected(row_voice_id == active_voice_id)

    def _on_microphone_device_changed(self, combo_index: int) -> None:
        if self._updating_from_store or combo_index < 0:
            return
        device_index = self._microphone_combo.itemData(combo_index)
        self._settings_store.set("microphone_device_index", device_index)

    def _on_dictation_language_changed(self, combo_index: int) -> None:
        if self._updating_from_store or combo_index < 0:
            return
        language_code = self._dictation_language_combo.itemData(combo_index)
        if language_code:
            self._settings_store.set("dictation_language", str(language_code))

    def _on_drawing_mode_changed(self, combo_index: int) -> None:
        if self._updating_from_store or combo_index < 0:
            return
        mode_id = self._drawing_mode_combo.itemData(combo_index)
        if mode_id:
            self._settings_store.set("screen_drawing_mode", str(mode_id))

    def _on_drawing_speed_changed(self, combo_index: int) -> None:
        if self._updating_from_store or combo_index < 0:
            return
        speed_id = self._drawing_speed_combo.itemData(combo_index)
        if speed_id:
            self._settings_store.set("drawing_speed", str(speed_id))

    def _add_custom_dictionary_word(self) -> None:
        new_word = self._custom_word_input.text().strip()
        if not new_word:
            return
        current_words = [str(word) for word in (self._settings_store.get("dictation_custom_words", []) or [])]
        if new_word not in current_words:
            current_words.append(new_word)
            self._settings_store.set("dictation_custom_words", current_words)
            self._rebuild_custom_dictionary_chips(current_words)
        self._custom_word_input.clear()

    def _remove_custom_dictionary_word(self, word_to_remove: str) -> None:
        current_words = [str(word) for word in (self._settings_store.get("dictation_custom_words", []) or [])]
        filtered_words = [word for word in current_words if word != word_to_remove]
        self._settings_store.set("dictation_custom_words", filtered_words)
        self._rebuild_custom_dictionary_chips(filtered_words)

    def _rebuild_custom_dictionary_chips(self, words: List[Any]) -> None:
        while self._dictionary_chips_layout.count() > 0:
            layout_item = self._dictionary_chips_layout.takeAt(0)
            widget = layout_item.widget() if layout_item is not None else None
            if widget is not None:
                widget.deleteLater()

        if not words:
            empty_label = QLabel("No custom words added yet.")
            empty_label.setObjectName("CardMutedLabel")
            self._dictionary_chips_layout.addWidget(empty_label)
            return

        for raw_word in words:
            word_text = str(raw_word).strip()
            if not word_text:
                continue
            chip_frame = QFrame()
            chip_frame.setObjectName("DictionaryChipFrame")
            chip_layout = QHBoxLayout(chip_frame)
            chip_layout.setContentsMargins(10, 4, 6, 4)
            chip_layout.setSpacing(6)

            chip_label = QLabel(word_text)
            chip_label.setObjectName("ChipTextLabel")
            chip_layout.addWidget(chip_label)

            remove_button = QPushButton("×")
            remove_button.setObjectName("ChipDeleteButton")
            remove_button.setFixedSize(18, 18)
            remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
            remove_button.clicked.connect(
                lambda _checked=False, target_word=word_text: self._remove_custom_dictionary_word(target_word)
            )
            chip_layout.addWidget(remove_button)
            self._dictionary_chips_layout.addWidget(chip_frame)

    def _add_style_rule(self) -> None:
        rule_text = self._new_style_rule_input.text().strip()
        if not rule_text:
            return
        current_rules = [str(rule) for rule in (self._settings_store.get("remember_style_notes", []) or [])]
        if rule_text not in current_rules:
            current_rules.append(rule_text)
            self._settings_store.set("remember_style_notes", current_rules)
            self._rebuild_style_rules_list(current_rules)
        self._new_style_rule_input.clear()

    def _remove_style_rule(self, rule_index: int) -> None:
        current_rules = [str(rule) for rule in (self._settings_store.get("remember_style_notes", []) or [])]
        if 0 <= rule_index < len(current_rules):
            del current_rules[rule_index]
            self._settings_store.set("remember_style_notes", current_rules)
            self._rebuild_style_rules_list(current_rules)

    def _rebuild_style_rules_list(self, rules: List[Any]) -> None:
        while self._style_rules_container.count() > 0:
            layout_item = self._style_rules_container.takeAt(0)
            widget = layout_item.widget() if layout_item is not None else None
            if widget is not None:
                widget.deleteLater()

        if not rules:
            empty_label = QLabel("No remembered style rules yet.")
            empty_label.setObjectName("RowSubtitleLabel")
            self._style_rules_container.addWidget(empty_label)
            return

        for rule_index, raw_rule in enumerate(rules):
            rule_frame = QFrame()
            rule_frame.setObjectName("StyleRuleRowFrame")
            rule_layout = QHBoxLayout(rule_frame)
            rule_layout.setContentsMargins(10, 6, 8, 6)
            rule_layout.setSpacing(8)

            rule_label = QLabel(str(raw_rule))
            rule_label.setObjectName("ChipTextLabel")
            rule_label.setWordWrap(True)
            rule_layout.addWidget(rule_label, 1)

            delete_button = QPushButton("×")
            delete_button.setObjectName("ChipDeleteButton")
            delete_button.setFixedSize(22, 22)
            delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
            delete_button.clicked.connect(
                lambda _checked=False, target_index=rule_index: self._remove_style_rule(target_index)
            )
            rule_layout.addWidget(delete_button)
            self._style_rules_container.addWidget(rule_frame)

    def _on_cursor_color_selected(self, selected_hex: str) -> None:
        normalized_hex = selected_hex.upper()
        self._refresh_cursor_color_selection(normalized_hex)
        self._settings_store.set("cursor_color_hex", normalized_hex)

    def _refresh_cursor_color_selection(self, active_hex: str) -> None:
        normalized = (active_hex or jarvis_theme.JARVIS_BLUE_HEX).upper()
        matched = False
        for hex_key, dot_button in self._cursor_color_buttons.items():
            is_match = hex_key == normalized
            dot_button.setChecked(is_match)
            if is_match:
                matched = True
        if not matched and jarvis_theme.JARVIS_BLUE_HEX.upper() in self._cursor_color_buttons:
            self._cursor_color_buttons[jarvis_theme.JARVIS_BLUE_HEX.upper()].setChecked(True)
            normalized = jarvis_theme.JARVIS_BLUE_HEX.upper()
        self._cursor_preview_widget.set_color_hex(normalized)

    # -- Chat helpers --------------------------------------------------------

    def _submit_chat_prompt(self) -> None:
        prompt_text = self._chat_input.text().strip()
        if not prompt_text:
            return
        # Controller pushes the message via append_chat_message("user", prompt_text)
        # so typed prompts never appear twice in the feed.
        self._chat_input.clear()
        self.promptSubmitted.emit(prompt_text)
        self._focus_chat_input()

    def _on_stop_clicked(self) -> None:
        self._set_chat_streaming_state(False)
        self.stopRequested.emit()

    def _set_chat_streaming_state(self, is_streaming: bool) -> None:
        self._is_streaming_chat = is_streaming
        self._stop_button.setEnabled(is_streaming)
        if is_streaming:
            self._chat_status_label.setText("Jarvis is responding...")
        else:
            self._chat_status_label.setText("Ready · Ask a question or press Ctrl + Alt anywhere")

    def _focus_chat_input(self) -> None:
        if self._active_top_tab == "chat":
            self._chat_input.setFocus(Qt.FocusReason.OtherFocusReason)

    def _scroll_chat_to_bottom(self) -> None:
        def scroll_now() -> None:
            scroll_bar = self._chat_scroll_area.verticalScrollBar()
            if scroll_bar is not None:
                scroll_bar.setValue(scroll_bar.maximum())

        QTimer.singleShot(0, scroll_now)


# ---------------------------------------------------------------------------
# Offscreen Visual Snapshot Generator & CLI Verification
# ---------------------------------------------------------------------------


def save_visual_snapshots(output_directory: str) -> List[str]:
    """
    Renders offscreen PNG snapshots for visual verification:
    `settings_account.png`, `settings_voice.png`, `settings_cursor.png`,
    `settings_dictation.png`, and `chat_view.png`.
    Uses an isolated temporary user settings JSON while loading `.env` keys masked
    inside `EchoMode.Password` inputs so API secrets are never exposed.
    """
    os.makedirs(output_directory, exist_ok=True)
    qt_app = QApplication.instance()
    owns_application = qt_app is None
    if qt_app is None:
        qt_app = QApplication(sys.argv[:1])

    jarvis_theme.register_bundled_fonts()

    with tempfile.TemporaryDirectory(prefix="jarvis_main_snapshot_") as temp_dir:
        temp_settings_path = os.path.join(temp_dir, "user_settings.json")
        snapshot_store = jarvis_config.SettingsStore(
            settings_file_path=temp_settings_path,
            env_file_path=jarvis_config.ENV_FILE_PATH,
        )
        snapshot_store.set("user_display_name", "RUTHVIK")
        snapshot_store.set("remember_style_notes", ["Answer in 2 concise bullets", "Lead with Yes or No"])
        snapshot_store.set("dictation_custom_words", ["Jarvis", "Jarvis", "PySide6", "AssemblyAI"])

        main_window = JarvisMainWindow(settings_store=snapshot_store)
        main_window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        main_window.resize(860, 580)
        main_window.show()
        qt_app.processEvents()

        main_window.set_microphone_devices(
            [
                {"index": 0, "name": "Realtek Microphone Array", "is_default": True},
                {"index": 1, "name": "External USB Condenser", "is_default": False},
            ]
        )
        main_window.set_microphone_level(0.42)

        main_window.append_chat_message("user", "Can you circle the Settings icon and summarize what I can change here?")
        main_window.append_chat_message(
            "assistant",
            "You can adjust your local .env API keys under Studio & Keys, preview 13 voices under Voice, "
            "customize your cursor color under Cursor, and manage your Custom Dictionary above Language.",
        )

        snapshot_targets = [
            ("settings:account", "settings_account.png"),
            ("settings:voice", "settings_voice.png"),
            ("settings:cursor", "settings_cursor.png"),
            ("settings:dictation", "settings_dictation.png"),
            ("chat", "chat_view.png"),
        ]
        saved_paths: List[str] = []

        for tab_name, file_name in snapshot_targets:
            main_window.open_tab(tab_name)
            qt_app.processEvents()
            hd_scale = 2.0
            hd_pixmap = QPixmap(
                int(main_window.width() * hd_scale),
                int(main_window.height() * hd_scale),
            )
            hd_pixmap.setDevicePixelRatio(hd_scale)
            hd_pixmap.fill(Qt.GlobalColor.transparent)
            main_window.render(hd_pixmap)
            output_path = os.path.join(output_directory, file_name)
            hd_pixmap.save(output_path, "PNG")
            saved_paths.append(output_path)

        main_window.close()
        main_window.deleteLater()
        qt_app.processEvents()

    if owns_application:
        qt_app.quit()
    return saved_paths


def _parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis Main Settings & Chat Window")
    parser.add_argument(
        "--snapshot",
        dest="snapshot_directory",
        default="",
        help="Render offscreen visual snapshots to the given directory and print frame benchmark.",
    )
    parser.add_argument(
        "--tab",
        dest="initial_tab",
        default="settings:account",
        help="Initial tab when launched interactively (e.g. chat, settings:account, settings:voice).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_cli_arguments()
    if arguments.snapshot_directory:
        saved_files = save_visual_snapshots(arguments.snapshot_directory)
        qt_app = QApplication.instance() or QApplication(sys.argv[:1])
        with tempfile.TemporaryDirectory(prefix="jarvis_bench_") as temp_dir:
            bench_store = jarvis_config.SettingsStore(
                settings_file_path=os.path.join(temp_dir, "user_settings.json"),
                env_file_path=jarvis_config.ENV_FILE_PATH,
            )
            bench_window = JarvisMainWindow(settings_store=bench_store)
            bench_window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            bench_window.resize(860, 580)
            bench_window.show()
            qt_app.processEvents()
            benchmark_results = bench_window.benchmark_frame_ms(iterations=30)
            bench_window.close()
        for saved_file in saved_files:
            print(f"Snapshot saved: {saved_file}")
        print(f"benchmark_frame_ms: {benchmark_results}")
        return 0

    qt_app = QApplication(sys.argv)
    jarvis_theme.register_bundled_fonts()
    settings_store = jarvis_config.SettingsStore()
    main_window = JarvisMainWindow(settings_store=settings_store)
    main_window.open_tab(arguments.initial_tab)
    main_window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
