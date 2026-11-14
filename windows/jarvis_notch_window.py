"""
PySide6 hardware top-edge Notch window (NotchWindow) and fanned ArcDock
for Jarvis Windows v1.1.5.

Provides:
- Top-center hardware notch silhouette with smooth cubic concave 10px bezel ears.
- 10 visual modes via NotchVisualMode (IDLE, RESTING, LISTENING, DICTATING, THINKING,
  SPEAKING, WARNING, TEXT_INPUT, TEXT_RESPONSE, HOME_PEEK).
- Compact 64x10 idle sliver when hide_floating_cursor is enabled (96x18 otherwise) so
  browser tabs stay unobstructed, plus 220ms hover dwell with blinking Jarvis eyes.
- Real interactive Qt QLineEdit inputs in TEXT_INPUT and TEXT_RESPONSE that disable
  WS_EX_NOACTIVATE and call force_foreground_window so typing works everywhere.
- Deactivation / outside-click handling during streaming TEXT_RESPONSE ("Close the chat,
  keep the answer") that emits continueInCaptionsRequested and collapses to SPEAKING/IDLE.
- Fanned 3-card Personal Studio ArcDock (HD Lasso & Ask, Scroll & Explain, 60 FPS Ink Studio) inside HOME_PEEK.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QRadialGradient,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QWidget,
)

try:
    import jarvis_config
    import jarvis_theme
except ImportError:  # pragma: no cover - fallback when imported as windows.jarvis_notch_window
    from . import jarvis_config, jarvis_theme

try:
    from pynput import mouse as pynput_mouse
except Exception:  # pragma: no cover - optional global outside-click listener
    pynput_mouse = None


# ---------------------------------------------------------------------------
# Constants & Visual Specifications (Jarvis Windows v1.1.5)
# ---------------------------------------------------------------------------

BEZIER_CIRCLE_KAPPA = 0.5522847498
TOP_EAR_RADIUS = 10.0

IDLE_COMPACT_SIZE = (64.0, 10.0, 5.0)
IDLE_DEFAULT_SIZE = (96.0, 18.0, 9.0)
RESTING_SIZE = (96.0, 18.0, 9.0)
ACTIVITY_STRIP_SIZE = (290.0, 30.0, 15.0)
TEXT_INPUT_SIZE = (520.0, 108.0, 20.0)
TEXT_RESPONSE_SIZE = (520.0, 250.0, 22.0)
HOME_PEEK_SIZE = (540.0, 395.0, 24.0)

EXPANDED_SIDE_MARGIN = 22
EXPANDED_BOTTOM_MARGIN = 20
COMPACT_SIDE_MARGIN = 12
COMPACT_BOTTOM_MARGIN = 4

HOVER_DWELL_MILLISECONDS = 220
ANIMATION_INTERVAL_MILLISECONDS = 16


class NotchVisualMode:
    """All visual presentation modes supported by NotchWindow."""

    IDLE = "idle"
    RESTING = "resting"
    LISTENING = "listening"
    DICTATING = "dictating"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WARNING = "warning"
    TEXT_INPUT = "text_input"
    TEXT_RESPONSE = "text_response"
    HOME_PEEK = "home_peek"


@dataclass
class ArcDockCardData:
    """Data describing a single interactive card inside the HOME_PEEK ArcDock."""

    card_id: str
    title: str
    badge_text: str
    badge_kind: str  # "done" | "new" | "interactive"
    subtitle: str
    action_label: str
    artifact_path: str
    card_background_hex: str
    header_background_hex: str


@dataclass
class ClickHitRegion:
    """Painted interactive region inside NotchWindow with pointer cursor feedback."""

    region_id: str
    bounds: QRectF
    callback: Callable[[], None]
    tooltip_text: str = ""


def _build_default_arc_cards() -> List[ArcDockCardData]:
    """Builds the 3 Personal Studio ArcDock cards shown in the expanded Home Peek."""
    return [
        ArcDockCardData(
            card_id="lasso-explain",
            title="HD Lasso & Ask",
            badge_text="Precision",
            badge_kind="lasso",
            subtitle=(
                "Drag a smooth neon lasso around any formula, code block, or paragraph "
                "for a 2x zoomed visual breakdown."
            ),
            action_label="Draw Lasso",
            artifact_path="",
            card_background_hex="#F4F7FC",
            header_background_hex="#E6EEF9",
        ),
        ArcDockCardData(
            card_id="scroll-explain",
            title="Scroll & Explain",
            badge_text="ChatGPT Mode",
            badge_kind="scroll",
            subtitle=(
                "Scrolls down long ChatGPT answers or docs step by step and draws "
                "underlines, rings, and diagram cards on every section."
            ),
            action_label="Start Scroll",
            artifact_path="",
            card_background_hex="#FFFFFF",
            header_background_hex="#EBF4FF",
        ),
        ArcDockCardData(
            card_id="screen-sketch-demo",
            title="60 FPS Ink Studio",
            badge_text="Interactive",
            badge_kind="interactive",
            subtitle=(
                "Preview 2x Retina Catmull-Rom ink, baseline-snapped underlines, "
                "and gridded diagram paper cards on screen."
            ),
            action_label="Run Demo",
            artifact_path="",
            card_background_hex="#FAF8F3",
            header_background_hex="#F1EDE4",
        ),
    ]


def build_hardware_notch_path(
    center_x: float,
    notch_width: float,
    notch_height: float,
    top_ear_radius: float = TOP_EAR_RADIUS,
    bottom_corner_radius: float = 15.0,
) -> Tuple[QPainterPath, QPainterPath]:
    """
    Constructs the closed fill path and open side/bottom border path for the
    top-edge hardware notch with smooth cubic concave top bezel ears.
    """
    clamped_width = max(24.0, float(notch_width))
    clamped_height = max(6.0, float(notch_height))
    ear_radius = min(float(top_ear_radius), max(2.0, clamped_height * 0.52))
    bottom_radius = min(
        float(bottom_corner_radius),
        max(2.0, clamped_height - ear_radius),
        max(2.0, clamped_width * 0.25),
    )

    left_x = center_x - clamped_width / 2.0
    right_x = center_x + clamped_width / 2.0
    top_y = 0.0
    bottom_y = clamped_height
    ear_handle = ear_radius * BEZIER_CIRCLE_KAPPA
    bottom_handle = bottom_radius * BEZIER_CIRCLE_KAPPA

    border_path = QPainterPath()
    border_path.moveTo(left_x - ear_radius, top_y)
    # Top-left concave ear flaring into the top monitor bezel
    border_path.cubicTo(
        left_x - ear_radius + ear_handle,
        top_y,
        left_x,
        top_y + ear_radius - ear_handle,
        left_x,
        top_y + ear_radius,
    )
    border_path.lineTo(left_x, bottom_y - bottom_radius)
    # Bottom-left convex corner
    border_path.cubicTo(
        left_x,
        bottom_y - bottom_radius + bottom_handle,
        left_x + bottom_radius - bottom_handle,
        bottom_y,
        left_x + bottom_radius,
        bottom_y,
    )
    border_path.lineTo(right_x - bottom_radius, bottom_y)
    # Bottom-right convex corner
    border_path.cubicTo(
        right_x - bottom_radius + bottom_handle,
        bottom_y,
        right_x,
        bottom_y - bottom_radius + bottom_handle,
        right_x,
        bottom_y - bottom_radius,
    )
    border_path.lineTo(right_x, top_y + ear_radius)
    # Top-right concave ear flaring out to y=0
    border_path.cubicTo(
        right_x,
        top_y + ear_radius - ear_handle,
        right_x + ear_radius - ear_handle,
        top_y,
        right_x + ear_radius,
        top_y,
    )

    fill_path = QPainterPath(border_path)
    fill_path.closeSubpath()
    return fill_path, border_path


# ---------------------------------------------------------------------------
# Crisp Custom-Painted Action Buttons Inside Typing Notch Modes
# ---------------------------------------------------------------------------


class _NotchIconButton(QPushButton):
    """Compact circular vector icon button used for Send and Close inside typing modes."""

    def __init__(self, icon_kind: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._icon_kind = icon_kind
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(26, 26)

    def enterEvent(self, event: QEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: QEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bounds = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if self._icon_kind == "send":
            fill_color = QColor("#4A92FF") if self._hovered else QColor("#3380FF")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawEllipse(bounds)
            center = bounds.center()
            arrow_pen = QPen(QColor(255, 255, 255), 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(arrow_pen)
            painter.drawLine(QPointF(center.x(), center.y() + 4.5), QPointF(center.x(), center.y() - 4.2))
            painter.drawLine(QPointF(center.x() - 3.6, center.y() - 0.8), QPointF(center.x(), center.y() - 4.5))
            painter.drawLine(QPointF(center.x() + 3.6, center.y() - 0.8), QPointF(center.x(), center.y() - 4.5))
        else:
            fill_color = QColor(52, 54, 64, 240) if self._hovered else QColor(34, 36, 44, 225)
            painter.setPen(QPen(QColor(255, 255, 255, 34), 1.0))
            painter.setBrush(fill_color)
            painter.drawEllipse(bounds)
            center = bounds.center()
            close_pen = QPen(QColor(215, 220, 232), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(close_pen)
            painter.drawLine(QPointF(center.x() - 3.4, center.y() - 3.4), QPointF(center.x() + 3.4, center.y() + 3.4))
            painter.drawLine(QPointF(center.x() + 3.4, center.y() - 3.4), QPointF(center.x() - 3.4, center.y() + 3.4))


class _NotchPillButton(QPushButton):
    """Crisp pill button used for [Stop] and [Expand] in TEXT_RESPONSE."""

    def __init__(self, label_text: str, button_kind: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(label_text, parent)
        self._label_text = label_text
        self._button_kind = button_kind
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedHeight(24)

    def enterEvent(self, event: QEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: QEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._button_kind == "stop":
            fill_color = QColor(72, 30, 34, 245) if self._hovered else QColor(52, 24, 28, 235)
            border_color = QColor(255, 105, 97, 130)
            text_color = QColor("#FF8079")
        else:
            fill_color = QColor(46, 49, 60, 245) if self._hovered else QColor(32, 34, 42, 235)
            border_color = QColor(255, 255, 255, 42)
            text_color = QColor("#E7EAF2")

        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(bounds, 12.0, 12.0)

        icon_center_x = bounds.left() + 13.0
        icon_center_y = bounds.center().y()
        if self._button_kind == "stop":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#FF6961"))
            painter.drawRoundedRect(QRectF(icon_center_x - 3.5, icon_center_y - 3.5, 7.0, 7.0), 1.5, 1.5)
        else:
            expand_pen = QPen(text_color, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(expand_pen)
            painter.drawLine(QPointF(icon_center_x - 3.0, icon_center_y + 3.0), QPointF(icon_center_x + 3.0, icon_center_y - 3.0))
            painter.drawLine(QPointF(icon_center_x + 0.2, icon_center_y - 3.0), QPointF(icon_center_x + 3.0, icon_center_y - 3.0))
            painter.drawLine(QPointF(icon_center_x + 3.0, icon_center_y - 3.0), QPointF(icon_center_x + 3.0, icon_center_y - 0.2))

        painter.setFont(jarvis_theme.ui_font(8.5, QFont.Weight.DemiBold))
        painter.setPen(text_color)
        text_rect = QRectF(bounds.left() + 21.0, bounds.top(), bounds.width() - 27.0, bounds.height())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._label_text)


# ---------------------------------------------------------------------------
# NotchWindow
# ---------------------------------------------------------------------------


class NotchWindow(QWidget):
    """
    Top-center hardware notch widget with spring physics, live voice/thinking/speaking
    corner-glow strips, real QLineEdit typing modes, and fanned 3-card HOME_PEEK ArcDock.
    """

    promptSubmitted = Signal(str)
    stopRequested = Signal()
    openMainWindowRequested = Signal(str)
    continueInCaptionsRequested = Signal()
    demoDrawingRequested = Signal(str)
    openArtifactRequested = Signal(str)
    scrollExplainRequested = Signal()
    lassoModeRequested = Signal()
    collapsed = Signal()

    _outsideMouseClickDetected = Signal()

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        super().__init__(None)
        self._settings_store = settings_store
        self._mode: str = NotchVisualMode.IDLE
        self._status_text: str = ""
        self._audio_level: float = 0.0
        self._smoothed_audio_level: float = 0.0
        self._live_transcript: str = ""
        self._response_text: str = ""
        self._is_response_generating: bool = False

        self._animation_origin = time.perf_counter()
        self._last_tick_timestamp = self._animation_origin
        self._activation_grace_until: float = 0.0
        self._hover_opened_home_peek: bool = False
        self._is_hovering_notch: bool = False
        self._hovered_region_id: str = ""
        self._hit_regions: List[ClickHitRegion] = []

        self._arc_cards: List[ArcDockCardData] = _build_default_arc_cards()
        self._selected_card_target: float = 1.0  # Center Scroll & Explain by default
        self._selected_card_index: int = 1

        initial_width, initial_height, initial_radius = self._target_geometry_for_mode(NotchVisualMode.IDLE)
        self._width_spring = jarvis_theme.Spring(initial_width, response=0.28, damping_ratio=0.86)
        self._height_spring = jarvis_theme.Spring(initial_height, response=0.28, damping_ratio=0.86)
        self._bottom_radius_spring = jarvis_theme.Spring(initial_radius, response=0.28, damping_ratio=0.88)
        self._card_index_spring = jarvis_theme.Spring(self._selected_card_target, response=0.30, damping_ratio=0.84)

        self._configure_window_attributes()
        self._build_typing_child_widgets()
        self._configure_timers()
        self._mouse_listener: Optional[object] = None
        self._outsideMouseClickDetected.connect(
            self._handle_outside_interaction, Qt.ConnectionType.QueuedConnection
        )
        self._settings_store.settingChanged.connect(self._handle_setting_changed)

        self._sync_window_envelope(force_shrink=True)
        self._layout_typing_children()

    def _configure_window_attributes(self) -> None:
        self.setWindowTitle("Jarvis Notch")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)

    def _configure_timers(self) -> None:
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(ANIMATION_INTERVAL_MILLISECONDS)
        self._animation_timer.timeout.connect(self._on_animation_tick)

        self._hover_dwell_timer = QTimer(self)
        self._hover_dwell_timer.setSingleShot(True)
        self._hover_dwell_timer.setInterval(HOVER_DWELL_MILLISECONDS)
        self._hover_dwell_timer.timeout.connect(self._on_hover_dwell_elapsed)

        self._hover_leave_collapse_timer = QTimer(self)
        self._hover_leave_collapse_timer.setSingleShot(True)
        self._hover_leave_collapse_timer.setInterval(180)
        self._hover_leave_collapse_timer.timeout.connect(self._on_hover_leave_collapse)

    def _build_typing_child_widgets(self) -> None:
        self._text_input_edit = QLineEdit(self)
        self._text_input_edit.setPlaceholderText("Ask Jarvis anything about your screen...")
        self._text_input_edit.setFrame(False)
        self._text_input_edit.setFont(jarvis_theme.ui_font(10.5))
        self._apply_line_edit_style(self._text_input_edit)
        self._text_input_edit.returnPressed.connect(self._submit_text_input)
        self._text_input_edit.hide()

        self._text_input_send_button = _NotchIconButton("send", parent=self)
        self._text_input_send_button.clicked.connect(self._submit_text_input)
        self._text_input_send_button.hide()

        self._stop_button = _NotchPillButton("Stop", button_kind="stop", parent=self)
        self._stop_button.setFixedWidth(64)
        self._stop_button.clicked.connect(self._handle_stop_clicked)
        self._stop_button.hide()

        self._expand_button = _NotchPillButton("Expand", button_kind="expand", parent=self)
        self._expand_button.setFixedWidth(74)
        self._expand_button.clicked.connect(lambda: self.openMainWindowRequested.emit("chat"))
        self._expand_button.hide()

        self._response_close_button = _NotchIconButton("close", parent=self)
        self._response_close_button.clicked.connect(self._handle_response_close_clicked)
        self._response_close_button.hide()

        self._response_text_edit = QTextEdit(self)
        self._response_text_edit.setReadOnly(True)
        self._response_text_edit.setFrameShape(QFrame.Shape.NoFrame)
        self._response_text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._response_text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._response_text_edit.setFont(jarvis_theme.ui_font(10.0))
        self._response_text_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._response_text_edit.setStyleSheet(
            """
            QTextEdit {
                background: transparent;
                color: #EEF1F8;
                border: none;
                selection-background-color: #3380FF;
                selection-color: #FFFFFF;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 2px 0px 2px 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 70);
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 115);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
                height: 0px;
            }
            """
        )
        self._response_text_edit.hide()

        self._reply_input_edit = QLineEdit(self)
        self._reply_input_edit.setPlaceholderText("Reply to Jarvis...")
        self._reply_input_edit.setFrame(False)
        self._reply_input_edit.setFont(jarvis_theme.ui_font(10.0))
        self._apply_line_edit_style(self._reply_input_edit)
        self._reply_input_edit.returnPressed.connect(self._submit_reply_input)
        self._reply_input_edit.hide()

        self._reply_send_button = _NotchIconButton("send", parent=self)
        self._reply_send_button.clicked.connect(self._submit_reply_input)
        self._reply_send_button.hide()

        self._text_input_edit.installEventFilter(self)
        self._response_text_edit.installEventFilter(self)
        self._reply_input_edit.installEventFilter(self)

    @staticmethod
    def _apply_line_edit_style(line_edit: QLineEdit) -> None:
        line_edit.setStyleSheet(
            """
            QLineEdit {
                background: transparent;
                color: #F5F7FC;
                border: none;
                selection-background-color: #3380FF;
                selection-color: #FFFFFF;
                padding: 0px;
            }
            """
        )
        palette = line_edit.palette()
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(146, 152, 168))
        line_edit.setPalette(palette)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def current_mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str, status_text: str = "") -> None:
        normalized_mode = (mode or NotchVisualMode.IDLE).strip().lower()
        valid_modes = {
            NotchVisualMode.IDLE,
            NotchVisualMode.RESTING,
            NotchVisualMode.LISTENING,
            NotchVisualMode.DICTATING,
            NotchVisualMode.THINKING,
            NotchVisualMode.SPEAKING,
            NotchVisualMode.WARNING,
            NotchVisualMode.TEXT_INPUT,
            NotchVisualMode.TEXT_RESPONSE,
            NotchVisualMode.HOME_PEEK,
        }
        if normalized_mode not in valid_modes:
            return

        previous_mode = self._mode
        self._mode = normalized_mode
        self._status_text = status_text or ""

        if normalized_mode != NotchVisualMode.HOME_PEEK:
            self._hover_opened_home_peek = False
        if normalized_mode not in (NotchVisualMode.IDLE, NotchVisualMode.RESTING):
            self._hover_dwell_timer.stop()
        self._hover_leave_collapse_timer.stop()

        target_width, target_height, target_radius = self._target_geometry_for_mode(normalized_mode)
        self._width_spring.set_target(target_width)
        self._height_spring.set_target(target_height)
        self._bottom_radius_spring.set_target(target_radius)

        self._sync_window_envelope(force_shrink=False)
        self._update_win32_activation_mode(previous_mode=previous_mode)
        self._layout_typing_children()
        self._sync_global_outside_mouse_listener()
        self._sync_animation_timer_state()
        self.update()

    def set_audio_level(self, level: float) -> None:
        clamped_level = max(0.0, min(1.0, float(level)))
        self._audio_level = clamped_level
        self._smoothed_audio_level = max(
            clamped_level, self._smoothed_audio_level * 0.72 + clamped_level * 0.28
        )

    def set_live_transcript(self, text: str) -> None:
        self._live_transcript = text or ""

    def set_streaming_response(self, text: str, is_generating: bool = True) -> None:
        self._response_text = text or ""
        self._is_response_generating = bool(is_generating)
        if self._mode != NotchVisualMode.TEXT_RESPONSE:
            self.set_mode(NotchVisualMode.TEXT_RESPONSE)
        self._sync_response_text_widget()
        self._layout_typing_children()
        self.update()

    def toggle_text_input(self) -> None:
        if self._mode == NotchVisualMode.TEXT_INPUT:
            self.collapse_to_idle()
        else:
            self.set_mode(NotchVisualMode.TEXT_INPUT)

    def toggle_home_peek(self) -> None:
        if self._mode == NotchVisualMode.HOME_PEEK:
            self.collapse_to_idle()
        else:
            self._hover_opened_home_peek = False
            self.set_mode(NotchVisualMode.HOME_PEEK)

    def cycle_arc_card(self, delta: int) -> None:
        if not self._arc_cards or delta == 0:
            return
        self._selected_card_target += float(delta)
        self._selected_card_index = int(round(self._selected_card_target)) % len(self._arc_cards)
        self._card_index_spring.set_target(self._selected_card_target)
        self._sync_animation_timer_state()
        self.update()

    def collapse_to_idle(self) -> None:
        was_non_idle = self._mode != NotchVisualMode.IDLE
        self._hover_opened_home_peek = False
        self._is_hovering_notch = False
        self.set_mode(NotchVisualMode.IDLE)
        if was_non_idle:
            self.collapsed.emit()

    def benchmark_frame_ms(self, iterations: int = 60) -> Dict[str, float]:
        """
        Renders offscreen frames directly via QPixmap / QWidget.render() across the core
        notch modes and returns per-mode and overall average frame paint times in ms.
        """
        saved_mode = self._mode
        saved_status = self._status_text
        saved_response = self._response_text
        saved_generating = self._is_response_generating
        saved_audio = self._audio_level
        saved_origin = self._animation_origin

        sample_modes: List[Tuple[str, str]] = [
            (NotchVisualMode.IDLE, ""),
            (NotchVisualMode.LISTENING, "Listening"),
            (NotchVisualMode.THINKING, "Thinking"),
            (NotchVisualMode.SPEAKING, "Speaking"),
            (NotchVisualMode.TEXT_INPUT, ""),
            (NotchVisualMode.TEXT_RESPONSE, ""),
            (NotchVisualMode.HOME_PEEK, ""),
        ]
        safe_iterations = max(1, int(iterations))
        results: Dict[str, float] = {}
        all_timings: List[float] = []

        self._response_text = (
            "Here is a quick summary of what is on your screen. Click Expand for the full chat."
        )
        self._is_response_generating = True
        self._audio_level = 0.62
        self._sync_response_text_widget()

        try:
            for mode_name, status_label in sample_modes:
                self.set_mode(mode_name, status_label)
                self._snap_springs_to_targets()
                self._sync_window_envelope(force_shrink=True)
                self._layout_typing_children()

                target_pixmap = QPixmap(max(1, self.width()), max(1, self.height()))
                target_pixmap.fill(Qt.GlobalColor.transparent)
                # Warm one frame so font metrics / pixmap caches reflect steady-state paint cost
                self.render(target_pixmap)

                mode_total_ms = 0.0
                for frame_index in range(safe_iterations):
                    self._animation_origin = time.perf_counter() - (frame_index * 0.016)
                    target_pixmap.fill(Qt.GlobalColor.transparent)
                    frame_start = time.perf_counter()
                    self.render(target_pixmap)
                    frame_elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
                    mode_total_ms += frame_elapsed_ms
                    all_timings.append(frame_elapsed_ms)

                results[mode_name] = round(mode_total_ms / float(safe_iterations), 4)
        finally:
            self._animation_origin = saved_origin
            self._response_text = saved_response
            self._is_response_generating = saved_generating
            self._audio_level = saved_audio
            self._sync_response_text_widget()
            self.set_mode(saved_mode, saved_status)
            self._snap_springs_to_targets()
            self._sync_window_envelope(force_shrink=True)
            self._layout_typing_children()

        results["average_ms"] = round(sum(all_timings) / max(1, len(all_timings)), 4)
        results["max_ms"] = round(max(all_timings) if all_timings else 0.0, 4)
        return results

    # -----------------------------------------------------------------------
    # Geometry, Springs & Timer Lifecycle
    # -----------------------------------------------------------------------

    def _target_geometry_for_mode(self, mode: str) -> Tuple[float, float, float]:
        if mode == NotchVisualMode.IDLE:
            if bool(self._settings_store.get("hide_floating_cursor", False)):
                return IDLE_COMPACT_SIZE
            return IDLE_DEFAULT_SIZE
        if mode == NotchVisualMode.RESTING:
            return RESTING_SIZE
        if mode in (
            NotchVisualMode.LISTENING,
            NotchVisualMode.DICTATING,
            NotchVisualMode.THINKING,
            NotchVisualMode.SPEAKING,
            NotchVisualMode.WARNING,
        ):
            return ACTIVITY_STRIP_SIZE
        if mode == NotchVisualMode.TEXT_INPUT:
            return TEXT_INPUT_SIZE
        if mode == NotchVisualMode.TEXT_RESPONSE:
            return TEXT_RESPONSE_SIZE
        if mode == NotchVisualMode.HOME_PEEK:
            return HOME_PEEK_SIZE
        return IDLE_DEFAULT_SIZE

    def _envelope_margins_for_mode(self, mode: str) -> Tuple[int, int]:
        if mode in (NotchVisualMode.IDLE, NotchVisualMode.RESTING):
            return COMPACT_SIDE_MARGIN, COMPACT_BOTTOM_MARGIN
        return EXPANDED_SIDE_MARGIN, EXPANDED_BOTTOM_MARGIN

    def _snap_springs_to_targets(self) -> None:
        target_width, target_height, target_radius = self._target_geometry_for_mode(self._mode)
        self._width_spring.snap(target_width)
        self._height_spring.snap(target_height)
        self._bottom_radius_spring.snap(target_radius)
        self._card_index_spring.snap(self._selected_card_target)

    def _sync_window_envelope(self, force_shrink: bool) -> None:
        target_width, target_height, _ = self._target_geometry_for_mode(self._mode)
        side_margin, bottom_margin = self._envelope_margins_for_mode(self._mode)
        needed_width = int(math.ceil(max(target_width, self._width_spring.value) + 2 * side_margin))
        needed_height = int(math.ceil(max(target_height, self._height_spring.value) + bottom_margin))

        springs_settled = self._width_spring.is_settled() and self._height_spring.is_settled()
        if force_shrink or springs_settled:
            desired_width = int(math.ceil(target_width + 2 * side_margin))
            desired_height = int(math.ceil(target_height + bottom_margin))
        else:
            desired_width = max(self.width(), needed_width)
            desired_height = max(self.height(), needed_height)

        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry() if screen is not None else QRect(0, 0, 1920, 1080)
        target_x = screen_geometry.x() + (screen_geometry.width() - desired_width) // 2
        target_y = screen_geometry.y()
        if (
            self.width() != desired_width
            or self.height() != desired_height
            or self.x() != target_x
            or self.y() != target_y
        ):
            self.setGeometry(target_x, target_y, desired_width, desired_height)

    def _needs_animation_timer(self) -> bool:
        springs_moving = not (
            self._width_spring.is_settled()
            and self._height_spring.is_settled()
            and self._bottom_radius_spring.is_settled()
            and self._card_index_spring.is_settled()
        )
        if springs_moving:
            return True
        if self._mode in (
            NotchVisualMode.LISTENING,
            NotchVisualMode.DICTATING,
            NotchVisualMode.THINKING,
            NotchVisualMode.SPEAKING,
            NotchVisualMode.WARNING,
            NotchVisualMode.RESTING,
        ):
            return True
        if self._mode == NotchVisualMode.IDLE and self._is_hovering_notch:
            return True
        return False

    def _sync_animation_timer_state(self) -> None:
        if self._needs_animation_timer():
            if not self._animation_timer.isActive():
                self._last_tick_timestamp = time.perf_counter()
                self._animation_timer.start()
        else:
            if self._animation_timer.isActive():
                self._animation_timer.stop()

    def _on_animation_tick(self) -> None:
        now = time.perf_counter()
        delta_seconds = max(0.001, min(now - self._last_tick_timestamp, 0.05))
        self._last_tick_timestamp = now

        self._width_spring.step(delta_seconds)
        self._height_spring.step(delta_seconds)
        self._bottom_radius_spring.step(delta_seconds)
        self._card_index_spring.step(delta_seconds)

        if self._mode not in (NotchVisualMode.LISTENING, NotchVisualMode.DICTATING):
            self._smoothed_audio_level *= 0.90

        springs_settled = (
            self._width_spring.is_settled()
            and self._height_spring.is_settled()
            and self._bottom_radius_spring.is_settled()
            and self._card_index_spring.is_settled()
        )
        if springs_settled:
            self._snap_springs_to_targets()
            self._sync_window_envelope(force_shrink=True)

        self._layout_typing_children()
        self.update()
        self._sync_animation_timer_state()

    def _current_notch_rect(self) -> QRectF:
        center_x = self.width() / 2.0
        notch_width = max(24.0, self._width_spring.value)
        notch_height = max(6.0, self._height_spring.value)
        return QRectF(center_x - notch_width / 2.0, 0.0, notch_width, notch_height)

    # -----------------------------------------------------------------------
    # Child Widget Visibility & Placement (TEXT_INPUT / TEXT_RESPONSE)
    # -----------------------------------------------------------------------

    def _layout_typing_children(self) -> None:
        notch_rect = self._current_notch_rect()
        show_text_input = (
            self._mode == NotchVisualMode.TEXT_INPUT and notch_rect.height() >= 84.0
        )
        show_text_response = (
            self._mode == NotchVisualMode.TEXT_RESPONSE and notch_rect.height() >= 190.0
        )

        self._text_input_edit.setVisible(show_text_input)
        self._text_input_send_button.setVisible(show_text_input)

        self._stop_button.setVisible(show_text_response and self._is_response_generating)
        self._expand_button.setVisible(show_text_response)
        self._response_close_button.setVisible(show_text_response)
        self._response_text_edit.setVisible(show_text_response)
        self._reply_input_edit.setVisible(show_text_response)
        self._reply_send_button.setVisible(show_text_response)

        if show_text_input:
            input_box_left = int(notch_rect.left() + 18)
            input_box_top = int(notch_rect.top() + 38)
            input_box_width = int(max(120, notch_rect.width() - 36))
            input_box_height = 40
            self._text_input_send_button.setGeometry(
                input_box_left + input_box_width - 33,
                input_box_top + 7,
                26,
                26,
            )
            self._text_input_edit.setGeometry(
                input_box_left + 12,
                input_box_top + 6,
                max(60, input_box_width - 52),
                input_box_height - 12,
            )

        if show_text_response:
            header_y = int(notch_rect.top() + 11)
            right_edge = int(notch_rect.right() - 16)
            self._response_close_button.setGeometry(right_edge - 26, header_y - 1, 26, 26)
            self._expand_button.setGeometry(right_edge - 26 - 6 - 74, header_y, 74, 24)
            if self._is_response_generating:
                self._stop_button.setGeometry(
                    right_edge - 26 - 6 - 74 - 6 - 64, header_y, 64, 24
                )

            body_left = int(notch_rect.left() + 18)
            body_top = int(notch_rect.top() + 44)
            body_width = int(max(120, notch_rect.width() - 36))
            reply_box_top = int(notch_rect.bottom() - 52)
            body_height = max(40, reply_box_top - body_top - 8)
            self._response_text_edit.setGeometry(body_left, body_top, body_width, body_height)

            reply_left = int(notch_rect.left() + 16)
            reply_width = int(max(120, notch_rect.width() - 32))
            self._reply_send_button.setGeometry(
                reply_left + reply_width - 33,
                reply_box_top + 6,
                26,
                26,
            )
            self._reply_input_edit.setGeometry(
                reply_left + 12,
                reply_box_top + 5,
                max(60, reply_width - 52),
                28,
            )

    def _sync_response_text_widget(self) -> None:
        scroll_bar = self._response_text_edit.verticalScrollBar()
        was_at_bottom = scroll_bar.value() >= max(0, scroll_bar.maximum() - 12)
        display_text = self._response_text or (
            "Thinking..." if self._is_response_generating else ""
        )
        if self._response_text_edit.toPlainText() != display_text:
            self._response_text_edit.setPlainText(display_text)
        if was_at_bottom and self._is_response_generating:
            scroll_bar.setValue(scroll_bar.maximum())

    # -----------------------------------------------------------------------
    # Win32 Styles, Typing Focus ("Typing works everywhere") & Deactivation
    # -----------------------------------------------------------------------

    def _apply_capture_exclusion(self) -> None:
        if not self.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
            return
        show_in_recordings = bool(self._settings_store.get("show_in_screen_recordings", True))
        jarvis_theme.set_window_excluded_from_capture(
            int(self.winId()), excluded=not show_in_recordings
        )

    def _activate_for_typing(self, target_line_edit: QLineEdit) -> None:
        self._activation_grace_until = time.perf_counter() + 0.25
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        if self.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
            window_handle = int(self.winId())
            jarvis_theme.set_window_no_activate(window_handle, no_activate=False)
            if self.isVisible():
                jarvis_theme.force_foreground_window(window_handle)
                self.raise_()
                self.activateWindow()
        target_line_edit.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        if self.isVisible():
            QTimer.singleShot(
                25,
                lambda: self._ensure_deferred_typing_focus(target_line_edit),
            )

    def _ensure_deferred_typing_focus(self, target_line_edit: QLineEdit) -> None:
        if not self.isVisible():
            return
        if self._mode in (NotchVisualMode.TEXT_INPUT, NotchVisualMode.TEXT_RESPONSE):
            if target_line_edit.isVisible():
                target_line_edit.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def _update_win32_activation_mode(self, previous_mode: str) -> None:
        if self._mode == NotchVisualMode.TEXT_INPUT:
            self._activate_for_typing(self._text_input_edit)
            return

        if self._mode == NotchVisualMode.TEXT_RESPONSE and previous_mode == NotchVisualMode.TEXT_INPUT:
            # Preserve foreground status when transitioning from TEXT_INPUT so clicking
            # elsewhere immediately triggers QEvent.WindowDeactivate ("Close the chat, keep the answer").
            if self.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
                jarvis_theme.set_window_no_activate(int(self.winId()), no_activate=False)
            return

        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        if self.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
            jarvis_theme.set_window_no_activate(int(self.winId()), no_activate=True)

    def _sync_global_outside_mouse_listener(self) -> None:
        needs_listener = (
            pynput_mouse is not None
            and self.isVisible()
            and self._mode
            in (
                NotchVisualMode.TEXT_INPUT,
                NotchVisualMode.TEXT_RESPONSE,
                NotchVisualMode.HOME_PEEK,
            )
        )
        if needs_listener and self._mouse_listener is None:
            try:
                listener = pynput_mouse.Listener(on_click=self._on_global_mouse_click)
                listener.daemon = True
                listener.start()
                self._mouse_listener = listener
            except Exception:
                self._mouse_listener = None
        elif not needs_listener and self._mouse_listener is not None:
            self._stop_global_outside_mouse_listener()

    def _stop_global_outside_mouse_listener(self) -> None:
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None

    def _on_global_mouse_click(
        self, cursor_x: int, cursor_y: int, button: object, pressed: bool
    ) -> None:
        del cursor_x, cursor_y, button
        if pressed:
            self._outsideMouseClickDetected.emit()

    def _is_global_point_inside_notch(self, global_point: QPoint) -> bool:
        local_point = QPointF(self.mapFromGlobal(global_point))
        return self._current_notch_rect().adjusted(-6.0, 0.0, 6.0, 6.0).contains(local_point)

    def _handle_outside_interaction(self) -> None:
        if not self.isVisible():
            return
        if time.perf_counter() < self._activation_grace_until:
            return
        if self._mode not in (
            NotchVisualMode.TEXT_INPUT,
            NotchVisualMode.TEXT_RESPONSE,
            NotchVisualMode.HOME_PEEK,
        ):
            return
        if self._is_global_point_inside_notch(QCursor.pos()):
            return
        self._dismiss_active_popover()

    def _dismiss_active_popover(self) -> None:
        if self._mode == NotchVisualMode.TEXT_RESPONSE:
            if self._is_response_generating:
                self.continueInCaptionsRequested.emit()
                self.set_mode(NotchVisualMode.SPEAKING)
            else:
                self.collapse_to_idle()
        elif self._mode in (NotchVisualMode.TEXT_INPUT, NotchVisualMode.HOME_PEEK):
            self.collapse_to_idle()

    # -----------------------------------------------------------------------
    # Submissions, Actions & Settings Reactions
    # -----------------------------------------------------------------------

    def _submit_text_input(self) -> None:
        prompt_text = self._text_input_edit.text().strip()
        if not prompt_text:
            return
        self._text_input_edit.clear()
        self.promptSubmitted.emit(prompt_text)

    def _submit_reply_input(self) -> None:
        reply_text = self._reply_input_edit.text().strip()
        if not reply_text:
            return
        self._reply_input_edit.clear()
        self.promptSubmitted.emit(reply_text)

    def _handle_stop_clicked(self) -> None:
        self._is_response_generating = False
        self._layout_typing_children()
        self.update()
        self.stopRequested.emit()

    def _handle_response_close_clicked(self) -> None:
        if self._is_response_generating:
            self.continueInCaptionsRequested.emit()
            self.set_mode(NotchVisualMode.SPEAKING)
        else:
            self.collapse_to_idle()

    def _handle_setting_changed(self, key_name: str, new_value: object) -> None:
        del new_value
        if key_name == "show_in_screen_recordings":
            self._apply_capture_exclusion()
        elif key_name == "hide_floating_cursor" and self._mode == NotchVisualMode.IDLE:
            self.set_mode(NotchVisualMode.IDLE)
        elif key_name in ("user_display_name", "cursor_color_hex", "hide_floating_cursor"):
            self.update()

    def _toggle_hide_floating_cursor(self) -> None:
        current_hidden = bool(self._settings_store.get("hide_floating_cursor", False))
        self._settings_store.set("hide_floating_cursor", not current_hidden)
        self.update()

    def _activate_arc_card(self, card_index: int, card_data: ArcDockCardData) -> None:
        self._hover_opened_home_peek = False
        if card_index != self._selected_card_index:
            shortest_delta = (
                (card_index - self._selected_card_index + len(self._arc_cards) / 2.0)
                % len(self._arc_cards)
            ) - len(self._arc_cards) / 2.0
            self.cycle_arc_card(int(round(shortest_delta)))
            return
        self._trigger_card_primary_action(card_data)

    def _trigger_card_primary_action(self, card_data: ArcDockCardData) -> None:
        self._hover_opened_home_peek = False
        if card_data.card_id == "scroll-explain":
            self.scrollExplainRequested.emit()
        elif card_data.card_id == "lasso-explain":
            self.lassoModeRequested.emit()
        elif card_data.card_id == "screen-sketch-demo":
            self.demoDrawingRequested.emit("full_demo")
        elif card_data.artifact_path:
            self.openArtifactRequested.emit(card_data.artifact_path)
        else:
            self.demoDrawingRequested.emit(card_data.card_id)

    # -----------------------------------------------------------------------
    # Qt Events (Show, Close, Hover Dwell, Click, Wheel, Key, Deactivate)
    # -----------------------------------------------------------------------

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        window_handle = int(self.winId())
        is_typing_mode = self._mode == NotchVisualMode.TEXT_INPUT
        jarvis_theme.set_window_no_activate(window_handle, no_activate=not is_typing_mode)
        self._apply_capture_exclusion()
        self._sync_global_outside_mouse_listener()
        self._sync_animation_timer_state()
        if is_typing_mode:
            self._activate_for_typing(self._text_input_edit)

    def hideEvent(self, event: QEvent) -> None:
        self._stop_global_outside_mouse_listener()
        self._hover_dwell_timer.stop()
        self._hover_leave_collapse_timer.stop()
        if self._animation_timer.isActive():
            self._animation_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event: QEvent) -> None:
        self._stop_global_outside_mouse_listener()
        self._hover_dwell_timer.stop()
        self._hover_leave_collapse_timer.stop()
        if self._animation_timer.isActive():
            self._animation_timer.stop()
        super().closeEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.WindowDeactivate:
            if time.perf_counter() >= self._activation_grace_until:
                if self._mode in (
                    NotchVisualMode.TEXT_INPUT,
                    NotchVisualMode.TEXT_RESPONSE,
                    NotchVisualMode.HOME_PEEK,
                ):
                    self._dismiss_active_popover()
        super().changeEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._reply_input_edit and event.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.FocusIn,
        ):
            self._activate_for_typing(self._reply_input_edit)
        elif watched is self._text_input_edit and event.type() == QEvent.Type.MouseButtonPress:
            self._activate_for_typing(self._text_input_edit)
        elif event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key.Key_Escape:
                self._dismiss_active_popover()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._dismiss_active_popover()
            event.accept()
            return
        if self._mode == NotchVisualMode.HOME_PEEK:
            if event.key() == Qt.Key.Key_Left:
                self.cycle_arc_card(-1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Right:
                self.cycle_arc_card(1)
                event.accept()
                return
        super().keyPressEvent(event)

    def enterEvent(self, event: QEvent) -> None:
        self._hover_leave_collapse_timer.stop()
        local_pos = QPointF(self.mapFromGlobal(QCursor.pos()))
        self._update_hover_state(local_pos)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._is_hovering_notch = False
        self._hovered_region_id = ""
        self._hover_dwell_timer.stop()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if self._mode == NotchVisualMode.HOME_PEEK and self._hover_opened_home_peek:
            self._hover_leave_collapse_timer.start()
        self._sync_animation_timer_state()
        self.update()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._update_hover_state(event.position())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        click_point = event.position()
        notch_rect = self._current_notch_rect()
        if not notch_rect.adjusted(-10.0, 0.0, 10.0, 6.0).contains(click_point):
            if self._mode in (
                NotchVisualMode.TEXT_INPUT,
                NotchVisualMode.TEXT_RESPONSE,
                NotchVisualMode.HOME_PEEK,
            ):
                self._dismiss_active_popover()
                event.accept()
                return

        if self._mode in (
            NotchVisualMode.IDLE,
            NotchVisualMode.RESTING,
            NotchVisualMode.LISTENING,
            NotchVisualMode.DICTATING,
            NotchVisualMode.THINKING,
            NotchVisualMode.SPEAKING,
            NotchVisualMode.WARNING,
        ):
            self._hover_opened_home_peek = False
            self.set_mode(NotchVisualMode.HOME_PEEK)
            event.accept()
            return

        if self._mode == NotchVisualMode.HOME_PEEK:
            # Any explicit click inside HOME_PEEK pins it open beyond hover leave.
            self._hover_opened_home_peek = False
            for hit_region in reversed(self._hit_regions):
                if hit_region.bounds.contains(click_point):
                    hit_region.callback()
                    event.accept()
                    return

        if self._mode == NotchVisualMode.TEXT_RESPONSE:
            self._activation_grace_until = time.perf_counter() + 0.25
            if self.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
                jarvis_theme.set_window_no_activate(int(self.winId()), no_activate=False)
                if self.isVisible():
                    jarvis_theme.force_foreground_window(int(self.winId()))

        super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._mode == NotchVisualMode.HOME_PEEK:
            delta_y = event.angleDelta().y()
            delta_x = event.angleDelta().x()
            dominant_delta = delta_x if abs(delta_x) > abs(delta_y) else delta_y
            if dominant_delta > 0:
                self.cycle_arc_card(-1)
                event.accept()
                return
            if dominant_delta < 0:
                self.cycle_arc_card(1)
                event.accept()
                return
        super().wheelEvent(event)

    def _update_hover_state(self, local_point: QPointF) -> None:
        notch_rect = self._current_notch_rect()
        inside_notch = notch_rect.adjusted(-8.0, 0.0, 8.0, 4.0).contains(local_point)

        if self._mode in (NotchVisualMode.IDLE, NotchVisualMode.RESTING):
            if inside_notch and not self._is_hovering_notch:
                self._is_hovering_notch = True
                self._hover_dwell_timer.start()
                self._sync_animation_timer_state()
                self.update()
            elif not inside_notch and self._is_hovering_notch:
                self._is_hovering_notch = False
                self._hover_dwell_timer.stop()
                self._sync_animation_timer_state()
                self.update()
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if inside_notch else Qt.CursorShape.ArrowCursor
            )
            return

        if self._mode == NotchVisualMode.HOME_PEEK:
            if inside_notch:
                self._hover_leave_collapse_timer.stop()
            elif self._hover_opened_home_peek and not self._hover_leave_collapse_timer.isActive():
                self._hover_leave_collapse_timer.start()

            matched_region_id = ""
            for hit_region in reversed(self._hit_regions):
                if hit_region.bounds.contains(local_point):
                    matched_region_id = hit_region.region_id
                    break
            if matched_region_id != self._hovered_region_id:
                self._hovered_region_id = matched_region_id
                self.update()
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if matched_region_id else Qt.CursorShape.ArrowCursor
            )
            return

        if self._mode in (
            NotchVisualMode.LISTENING,
            NotchVisualMode.DICTATING,
            NotchVisualMode.THINKING,
            NotchVisualMode.SPEAKING,
            NotchVisualMode.WARNING,
        ):
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if inside_notch else Qt.CursorShape.ArrowCursor
            )
            return

        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_hover_dwell_elapsed(self) -> None:
        if not self._is_hovering_notch:
            return
        if self._mode in (NotchVisualMode.IDLE, NotchVisualMode.RESTING):
            self._hover_opened_home_peek = True
            self.set_mode(NotchVisualMode.HOME_PEEK)

    def _on_hover_leave_collapse(self) -> None:
        if self._mode == NotchVisualMode.HOME_PEEK and self._hover_opened_home_peek:
            if not self._is_global_point_inside_notch(QCursor.pos()):
                self.collapse_to_idle()

    # -----------------------------------------------------------------------
    # Master Painting Pipeline (Hardware Silhouette + Glow + Mode Content)
    # -----------------------------------------------------------------------

    def paintEvent(self, event: QEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        now = time.perf_counter()
        elapsed_seconds = now - self._animation_origin
        notch_rect = self._current_notch_rect()
        ear_radius = min(TOP_EAR_RADIUS, max(2.0, notch_rect.height() * 0.52))
        bottom_radius = self._bottom_radius_spring.value

        fill_path, border_path = build_hardware_notch_path(
            center_x=notch_rect.center().x(),
            notch_width=notch_rect.width(),
            notch_height=notch_rect.height(),
            top_ear_radius=ear_radius,
            bottom_corner_radius=bottom_radius,
        )

        # Soft layered drop shadow for expanded states (pure vector, zero slow blur passes)
        if notch_rect.height() >= 24.0:
            self._paint_notch_shadow(
                painter=painter,
                center_x=notch_rect.center().x(),
                notch_width=notch_rect.width(),
                notch_height=notch_rect.height(),
                ear_radius=ear_radius,
                bottom_radius=bottom_radius,
            )

        # Pure pitch-black hardware notch body
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(jarvis_theme.NOTCH_BLACK)
        painter.drawPath(fill_path)

        # Interior clipped to exact notch silhouette
        self._hit_regions.clear()
        painter.save()
        painter.setClipPath(fill_path)

        if self._mode in (NotchVisualMode.IDLE, NotchVisualMode.RESTING) and notch_rect.height() <= 24.0:
            if self._mode == NotchVisualMode.RESTING or self._is_hovering_notch:
                self._paint_blinking_eyes(painter, notch_rect, elapsed_seconds)
        elif self._mode in (
            NotchVisualMode.LISTENING,
            NotchVisualMode.DICTATING,
            NotchVisualMode.THINKING,
            NotchVisualMode.SPEAKING,
            NotchVisualMode.WARNING,
        ):
            self._paint_activity_strip(painter, notch_rect, elapsed_seconds)
        elif self._mode == NotchVisualMode.TEXT_INPUT:
            self._paint_text_input_chrome(painter, notch_rect)
        elif self._mode == NotchVisualMode.TEXT_RESPONSE:
            self._paint_text_response_chrome(painter, notch_rect)
        elif self._mode == NotchVisualMode.HOME_PEEK:
            self._paint_home_peek(painter, notch_rect)

        painter.restore()

        # Subtle crisp rim hairline along sides/bottom (excluding top monitor bezel edge)
        hairline_color = QColor(255, 255, 255, 48 if notch_rect.height() <= 20.0 else 30)
        painter.setPen(QPen(hairline_color, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(border_path)

    def _paint_notch_shadow(
        self,
        painter: QPainter,
        center_x: float,
        notch_width: float,
        notch_height: float,
        ear_radius: float,
        bottom_radius: float,
    ) -> None:
        shadow_rings = (
            (10.0, 5.0, 20),
            (6.0, 3.0, 38),
            (3.0, 1.5, 62),
        )
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for spread_px, drop_y, alpha_value in shadow_rings:
            shadow_path, _ = build_hardware_notch_path(
                center_x=center_x,
                notch_width=notch_width + spread_px,
                notch_height=notch_height + drop_y + spread_px * 0.45,
                top_ear_radius=ear_radius,
                bottom_corner_radius=bottom_radius + spread_px * 0.35,
            )
            painter.setBrush(QColor(0, 0, 0, alpha_value))
            painter.drawPath(shadow_path)
        painter.restore()

    def _paint_blinking_eyes(
        self,
        painter: QPainter,
        notch_rect: QRectF,
        elapsed_seconds: float,
    ) -> None:
        cycle_position = elapsed_seconds % 4.2
        blink_scale = 1.0
        if 3.85 <= cycle_position <= 4.02:
            blink_progress = (cycle_position - 3.85) / 0.17
            blink_scale = max(0.18, abs(math.cos(blink_progress * math.pi)))

        eye_color = QColor(self._settings_store.get("cursor_color_hex", jarvis_theme.JARVIS_BLUE_HEX))
        eye_tint = QColor(
            min(255, int(eye_color.red() * 0.45 + 145)),
            min(255, int(eye_color.green() * 0.45 + 145)),
            min(255, int(eye_color.blue() * 0.45 + 145)),
            235,
        )
        base_eye_height = min(5.4, max(2.8, notch_rect.height() * 0.36))
        eye_height = max(1.2, base_eye_height * blink_scale)
        eye_width = min(3.4, max(2.2, notch_rect.height() * 0.22))
        eye_spacing = min(13.0, notch_rect.width() * 0.16)
        center_x = notch_rect.center().x()
        center_y = notch_rect.center().y() + 0.4

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(eye_tint)
        for sign in (-1.0, 1.0):
            eye_rect = QRectF(
                center_x + sign * (eye_spacing / 2.0) - eye_width / 2.0,
                center_y - eye_height / 2.0,
                eye_width,
                eye_height,
            )
            painter.drawRoundedRect(eye_rect, eye_width / 2.0, eye_width / 2.0)

    def _paint_activity_strip(
        self,
        painter: QPainter,
        notch_rect: QRectF,
        elapsed_seconds: float,
    ) -> None:
        mode_defaults = {
            NotchVisualMode.LISTENING: ("Listening", "listening"),
            NotchVisualMode.DICTATING: ("Dictating", "dictating"),
            NotchVisualMode.THINKING: ("Thinking", "thinking"),
            NotchVisualMode.SPEAKING: ("Speaking", "speaking"),
            NotchVisualMode.WARNING: ("Check microphone", "warning"),
        }
        default_label, glow_key = mode_defaults.get(self._mode, ("Listening", "listening"))
        display_label = self._status_text or default_label
        outer_glow_color, inner_glow_color = jarvis_theme.ACTIVITY_GLOW_COLORS.get(
            glow_key, jarvis_theme.ACTIVITY_GLOW_COLORS["listening"]
        )

        # 1. Right-end liquid corner aura matching reference photos 2.15.02 / 2.16.28 / 2.16.51
        activity_pulse = math.sin(elapsed_seconds * 4.0) * 2.5 + self._smoothed_audio_level * 6.0
        outer_center = QPointF(notch_rect.right() - 10.0, notch_rect.height() * 0.62)
        outer_radius = 48.0 + activity_pulse
        outer_gradient = QRadialGradient(outer_center, outer_radius)
        outer_gradient.setColorAt(0.0, jarvis_theme.with_alpha(outer_glow_color, 215))
        outer_gradient.setColorAt(0.45, jarvis_theme.with_alpha(outer_glow_color, 105))
        outer_gradient.setColorAt(1.0, jarvis_theme.with_alpha(outer_glow_color, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(outer_gradient)
        painter.drawEllipse(outer_center, outer_radius, notch_rect.height() * 1.15)

        core_center = QPointF(notch_rect.right() - 3.0, notch_rect.height() * 0.56)
        core_radius = 28.0 + activity_pulse * 0.65
        core_gradient = QRadialGradient(core_center, core_radius)
        core_gradient.setColorAt(0.0, jarvis_theme.with_alpha(inner_glow_color, 250))
        core_gradient.setColorAt(0.50, jarvis_theme.with_alpha(outer_glow_color, 175))
        core_gradient.setColorAt(1.0, jarvis_theme.with_alpha(outer_glow_color, 0))
        painter.setBrush(core_gradient)
        painter.drawEllipse(core_center, core_radius, notch_rect.height() * 0.92)

        # 2. Crisp white left label
        label_font = jarvis_theme.ui_font(9.8, QFont.Weight.DemiBold)
        painter.setFont(label_font)
        painter.setPen(jarvis_theme.NOTCH_TEXT_PRIMARY)
        available_label_width = max(80.0, notch_rect.width() - 96.0)
        elided_label = QFontMetricsF(label_font).elidedText(
            display_label, Qt.TextElideMode.ElideRight, available_label_width
        )
        painter.drawText(
            QRectF(notch_rect.left() + 18.0, notch_rect.top(), available_label_width, notch_rect.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            elided_label,
        )

        # 3. Right indicator: 5 waveform bars (Listening/Dictating/Speaking),
        #    3 bouncing dots (Thinking), or compact warning badge (Warning)
        mid_y = notch_rect.center().y()
        if self._mode == NotchVisualMode.THINKING:
            dots_start_x = notch_rect.right() - 36.0
            for dot_index in range(3):
                phase_value = elapsed_seconds * 6.2 - dot_index * 0.92
                bounce = math.sin(phase_value)
                lift_offset = max(0.0, bounce) * 3.8 - 1.1
                dot_alpha = int(190 + 65 * max(0.0, bounce))
                dot_center = QPointF(dots_start_x + dot_index * 5.6, mid_y - lift_offset)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(255, 255, 255, dot_alpha))
                painter.drawEllipse(dot_center, 1.85, 1.85)
        elif self._mode == NotchVisualMode.WARNING:
            badge_center = QPointF(notch_rect.right() - 25.0, mid_y)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 240))
            painter.drawEllipse(badge_center, 7.0, 7.0)
            painter.setPen(QColor("#1A1200"))
            painter.setFont(jarvis_theme.ui_font(8.5, QFont.Weight.Bold))
            painter.drawText(
                QRectF(badge_center.x() - 7.0, badge_center.y() - 7.0, 14.0, 14.0),
                Qt.AlignmentFlag.AlignCenter,
                "!",
            )
        else:
            bar_weights = (0.64, 0.92, 1.00, 0.88, 0.60)
            bars_start_x = notch_rect.right() - 40.0
            if self._mode == NotchVisualMode.SPEAKING:
                speaking_cadence = 0.56 + 0.24 * math.sin(elapsed_seconds * 3.1)
                effective_level = max(self._smoothed_audio_level, speaking_cadence)
                speed_multiplier = 8.4
            else:
                effective_level = max(0.22, min(1.0, self._smoothed_audio_level * 1.45 + 0.22))
                speed_multiplier = 7.4

            painter.setPen(Qt.PenStyle.NoPen)
            for bar_index in range(5):
                wave = math.sin(elapsed_seconds * speed_multiplier + bar_index * 0.95) * 0.5 + 0.5
                bar_height = max(
                    3.2,
                    min(15.0, 3.0 + (wave * 0.68 + 0.32) * 12.0 * effective_level * bar_weights[bar_index]),
                )
                bar_x = bars_start_x + bar_index * 4.2
                bar_alpha = int(195 + 60 * wave)
                painter.setBrush(QColor(255, 255, 255, bar_alpha))
                painter.drawRoundedRect(
                    QRectF(bar_x, mid_y - bar_height / 2.0, 2.0, bar_height),
                    1.0,
                    1.0,
                )

    def _paint_text_input_chrome(self, painter: QPainter, notch_rect: QRectF) -> None:
        # Header icon + title
        cursor_color = QColor(self._settings_store.get("cursor_color_hex", jarvis_theme.JARVIS_BLUE_HEX))
        self._paint_mini_cursor_mark(painter, QPointF(notch_rect.left() + 26.0, notch_rect.top() + 21.0), cursor_color)

        painter.setFont(jarvis_theme.ui_font(9.5, QFont.Weight.DemiBold))
        painter.setPen(jarvis_theme.NOTCH_TEXT_PRIMARY)
        painter.drawText(
            QRectF(notch_rect.left() + 41.0, notch_rect.top() + 10.0, 240.0, 22.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "Ask Jarvis",
        )

        painter.setFont(jarvis_theme.ui_font(8.0, QFont.Weight.Normal))
        painter.setPen(jarvis_theme.NOTCH_TEXT_SECONDARY)
        painter.drawText(
            QRectF(notch_rect.right() - 120.0, notch_rect.top() + 10.0, 102.0, 22.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            "Esc to close",
        )

        # Rounded dark composer box
        box_rect = QRectF(notch_rect.left() + 18.0, notch_rect.top() + 38.0, notch_rect.width() - 36.0, 40.0)
        border_color = (
            QColor(51, 128, 255, 165)
            if self._text_input_edit.hasFocus()
            else QColor(255, 255, 255, 38)
        )
        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(QColor("#14161D"))
        painter.drawRoundedRect(box_rect, 12.0, 12.0)

        # Footer hint
        painter.setFont(jarvis_theme.ui_font(7.8, QFont.Weight.Normal))
        painter.setPen(QColor(152, 158, 174, 210))
        painter.drawText(
            QRectF(notch_rect.left() + 22.0, notch_rect.bottom() - 23.0, notch_rect.width() - 44.0, 16.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "Enter to send · Esc to close",
        )

    def _paint_text_response_chrome(self, painter: QPainter, notch_rect: QRectF) -> None:
        cursor_color = QColor(self._settings_store.get("cursor_color_hex", jarvis_theme.JARVIS_BLUE_HEX))
        self._paint_mini_cursor_mark(painter, QPointF(notch_rect.left() + 26.0, notch_rect.top() + 22.0), cursor_color)

        painter.setFont(jarvis_theme.ui_font(9.5, QFont.Weight.DemiBold))
        painter.setPen(jarvis_theme.NOTCH_TEXT_PRIMARY)
        painter.drawText(
            QRectF(notch_rect.left() + 41.0, notch_rect.top() + 11.0, 120.0, 22.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "Jarvis",
        )

        if self._is_response_generating:
            painter.setFont(jarvis_theme.ui_font(8.0, QFont.Weight.Medium))
            painter.setPen(QColor("#78B9FF"))
            painter.drawText(
                QRectF(notch_rect.left() + 108.0, notch_rect.top() + 11.0, 110.0, 22.0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                "Answering...",
            )

        # Subtle separator under header
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1.0))
        painter.drawLine(
            QPointF(notch_rect.left() + 18.0, notch_rect.top() + 39.0),
            QPointF(notch_rect.right() - 18.0, notch_rect.top() + 39.0),
        )

        # Reply input box at bottom
        reply_rect = QRectF(notch_rect.left() + 16.0, notch_rect.bottom() - 52.0, notch_rect.width() - 32.0, 38.0)
        reply_border = (
            QColor(51, 128, 255, 165)
            if self._reply_input_edit.hasFocus()
            else QColor(255, 255, 255, 34)
        )
        painter.setPen(QPen(reply_border, 1.0))
        painter.setBrush(QColor("#14161D"))
        painter.drawRoundedRect(reply_rect, 12.0, 12.0)

    def _paint_mini_cursor_mark(self, painter: QPainter, center: QPointF, color: QColor) -> None:
        painter.save()
        painter.translate(center)
        triangle_path = jarvis_theme.build_rounded_triangle_path(side_length=13.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPath(triangle_path)
        painter.restore()

    # -----------------------------------------------------------------------
    # HOME_PEEK Painting (Matching 2.15.25 AM.jpeg 1:1)
    # -----------------------------------------------------------------------

    def _paint_home_peek(self, painter: QPainter, notch_rect: QRectF) -> None:
        center_x = notch_rect.center().x()
        top_y = notch_rect.top() + 14.0

        # 1. Top Bar: [glasses] [chat] ... [Scroll & Explain] [gear]
        cursor_hidden = bool(self._settings_store.get("hide_floating_cursor", False))
        glasses_rect = QRectF(notch_rect.left() + 18.0, top_y, 28.0, 28.0)
        self._paint_circle_icon_button(
            painter=painter,
            bounds=glasses_rect,
            region_id="top_glasses",
            icon_kind="glasses",
            active=not cursor_hidden,
            slashed=cursor_hidden,
        )
        self._hit_regions.append(
            ClickHitRegion("top_glasses", glasses_rect, self._toggle_hide_floating_cursor)
        )

        chat_rect = QRectF(notch_rect.left() + 52.0, top_y, 28.0, 28.0)
        self._paint_circle_icon_button(
            painter=painter,
            bounds=chat_rect,
            region_id="top_chat",
            icon_kind="chat",
            active=True,
        )
        self._hit_regions.append(
            ClickHitRegion(
                "top_chat",
                chat_rect,
                lambda: self.openMainWindowRequested.emit("chat"),
            )
        )

        scroll_pill_rect = QRectF(notch_rect.right() - 160.0, top_y, 106.0, 28.0)
        self._paint_studio_pill(painter, scroll_pill_rect, "Scroll & Explain")
        self._hit_regions.append(
            ClickHitRegion(
                "top_scroll_explain",
                scroll_pill_rect,
                lambda: self.scrollExplainRequested.emit(),
            )
        )

        gear_rect = QRectF(notch_rect.right() - 46.0, top_y, 28.0, 28.0)
        self._paint_circle_icon_button(
            painter=painter,
            bounds=gear_rect,
            region_id="top_settings",
            icon_kind="gear",
            active=True,
        )
        self._hit_regions.append(
            ClickHitRegion(
                "top_settings",
                gear_rect,
                lambda: self.openMainWindowRequested.emit("settings"),
            )
        )

        # 2. Center Greeting
        display_name = str(self._settings_store.get("user_display_name", "") or "").strip()
        greeting_text = jarvis_theme.greeting_for_current_time(display_name)
        painter.setFont(jarvis_theme.ui_font(15.5, QFont.Weight.Bold))
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(
            QRectF(notch_rect.left() + 24.0, notch_rect.top() + 47.0, notch_rect.width() - 48.0, 28.0),
            Qt.AlignmentFlag.AlignCenter,
            greeting_text,
        )

        painter.setFont(jarvis_theme.ui_font(8.8, QFont.Weight.Normal))
        painter.setPen(QColor(214, 218, 228, 225))
        painter.drawText(
            QRectF(notch_rect.left() + 24.0, notch_rect.top() + 76.0, notch_rect.width() - 48.0, 19.0),
            Qt.AlignmentFlag.AlignCenter,
            "Hold Ctrl + Alt to ask with lasso, or double-tap Ctrl to type.",
        )

        # 3. Fanned ArcDock Cards (sorted back-to-front by distance from active center)
        card_count = len(self._arc_cards)
        dock_center_y = notch_rect.top() + 214.0
        rendered_cards: List[Tuple[float, int, ArcDockCardData]] = []
        for card_index, card_data in enumerate(self._arc_cards):
            raw_offset = (
                (card_index - self._card_index_spring.value + card_count / 2.0) % card_count
            ) - card_count / 2.0
            rendered_cards.append((raw_offset, card_index, card_data))

        rendered_cards.sort(key=lambda entry: abs(entry[0]), reverse=True)
        for offset_value, card_index, card_data in rendered_cards:
            self._paint_arc_card(
                painter=painter,
                dock_center_x=center_x,
                dock_center_y=dock_center_y,
                offset_value=offset_value,
                card_index=card_index,
                card_data=card_data,
            )

        # Subtle left/right cycle buttons on ArcDock edges
        prev_arrow_rect = QRectF(notch_rect.left() + 10.0, dock_center_y - 12.0, 24.0, 24.0)
        next_arrow_rect = QRectF(notch_rect.right() - 34.0, dock_center_y - 12.0, 24.0, 24.0)
        self._paint_circle_icon_button(painter, prev_arrow_rect, "dock_prev", "chevron_left", active=True, subtle=True)
        self._paint_circle_icon_button(painter, next_arrow_rect, "dock_next", "chevron_right", active=True, subtle=True)
        self._hit_regions.append(ClickHitRegion("dock_prev", prev_arrow_rect, lambda: self.cycle_arc_card(-1)))
        self._hit_regions.append(ClickHitRegion("dock_next", next_arrow_rect, lambda: self.cycle_arc_card(1)))

        # 4. Bottom Footer Pills ([Lasso & Ask] on left, [Studio] on right)
        footer_top = notch_rect.bottom() - 48.0
        lasso_rect = QRectF(notch_rect.left() + 20.0, footer_top, 114.0, 30.0)
        self._paint_footer_pill(
            painter=painter,
            bounds=lasso_rect,
            region_id="footer_lasso",
            label_text="Lasso & Ask",
            icon_kind="sparkle",
            is_light_pill=True,
        )
        self._hit_regions.append(
            ClickHitRegion(
                "footer_lasso",
                lasso_rect,
                lambda: self.lassoModeRequested.emit(),
            )
        )

        studio_rect = QRectF(notch_rect.right() - 104.0, footer_top, 84.0, 30.0)
        self._paint_footer_pill(
            painter=painter,
            bounds=studio_rect,
            region_id="footer_studio",
            label_text="Studio",
            icon_kind="menu",
            is_light_pill=False,
        )
        self._hit_regions.append(
            ClickHitRegion(
                "footer_studio",
                studio_rect,
                lambda: self.openMainWindowRequested.emit("settings"),
            )
        )

    def _paint_arc_card(
        self,
        painter: QPainter,
        dock_center_x: float,
        dock_center_y: float,
        offset_value: float,
        card_index: int,
        card_data: ArcDockCardData,
    ) -> None:
        base_width = 272.0
        base_height = 194.0
        abs_offset = abs(offset_value)
        card_scale = max(0.60, 1.0 - abs_offset * 0.32)
        card_center_x = dock_center_x + offset_value * 170.0
        card_center_y = dock_center_y + abs_offset * 6.0
        tilt_degrees = offset_value * 5.5

        painter.save()
        painter.translate(card_center_x, card_center_y)
        painter.rotate(tilt_degrees)
        painter.scale(card_scale, card_scale)

        local_rect = QRectF(-base_width / 2.0, -base_height / 2.0, base_width, base_height)

        # Card drop shadow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 120 if abs_offset < 0.4 else 85))
        painter.drawRoundedRect(local_rect.translated(0.0, 5.0), 15.0, 15.0)

        # Card surface clipped cleanly to rounded corners
        card_clip = QPainterPath()
        card_clip.addRoundedRect(local_rect, 15.0, 15.0)
        painter.save()
        painter.setClipPath(card_clip)

        painter.setBrush(QColor(card_data.card_background_hex))
        painter.drawRect(local_rect)

        # Header band
        header_rect = QRectF(local_rect.left(), local_rect.top(), base_width, 28.0)
        painter.setBrush(QColor(card_data.header_background_hex))
        painter.drawRect(header_rect)

        if card_data.card_id == "scroll-explain":
            self._paint_scroll_explain_card_interior(painter, local_rect, header_rect, card_data)
        elif card_data.card_id == "lasso-explain":
            self._paint_lasso_explain_card_interior(painter, local_rect, header_rect, card_data, is_center=(abs_offset < 0.35))
        else:
            self._paint_sketch_demo_card_interior(painter, local_rect, header_rect, card_data)

        # Subtle depth shade when fanned to the side
        if abs_offset > 0.2:
            shade_alpha = int(min(52, abs_offset * 28))
            painter.setBrush(QColor(0, 0, 0, shade_alpha))
            painter.drawRect(local_rect)

        painter.restore()

        # Crisp card rim
        is_card_hovered = self._hovered_region_id == f"card_{card_data.card_id}"
        rim_color = QColor("#7CC3FF") if is_card_hovered else QColor(232, 236, 244, 240)
        painter.setPen(QPen(rim_color, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(local_rect, 15.0, 15.0)
        painter.restore()

        # Register screen-space hit-test regions using actual transform
        scaled_w = base_width * card_scale
        scaled_h = base_height * card_scale
        screen_card_rect = QRectF(
            card_center_x - scaled_w / 2.0,
            card_center_y - scaled_h / 2.0,
            scaled_w,
            scaled_h,
        )
        self._hit_regions.append(
            ClickHitRegion(
                f"card_{card_data.card_id}",
                screen_card_rect,
                lambda idx=card_index, item=card_data: self._activate_arc_card(idx, item),
            )
        )

        if abs_offset < 0.35:
            pill_local = QRectF(
                local_rect.left() + 16.0,
                local_rect.bottom() - 38.0,
                92.0,
                24.0,
            )
            pill_screen = QRectF(
                card_center_x + pill_local.left() * card_scale,
                card_center_y + pill_local.top() * card_scale,
                pill_local.width() * card_scale,
                pill_local.height() * card_scale,
            )
            self._hit_regions.append(
                ClickHitRegion(
                    f"pill_{card_data.card_id}",
                    pill_screen,
                    lambda item=card_data: self._trigger_card_primary_action(item),
                )
            )

    def _paint_scroll_explain_card_interior(
        self,
        painter: QPainter,
        local_rect: QRectF,
        header_rect: QRectF,
        card_data: ArcDockCardData,
    ) -> None:
        painter.setFont(jarvis_theme.ui_font(8.4, QFont.Weight.DemiBold))
        painter.setPen(QColor("#0A84FF"))
        painter.drawText(header_rect, Qt.AlignmentFlag.AlignCenter, card_data.badge_text)

        painter.setFont(jarvis_theme.ui_font(11.6, QFont.Weight.DemiBold))
        painter.setPen(QColor("#1F232C"))
        painter.drawText(
            QRectF(local_rect.left() + 16.0, local_rect.top() + 38.0, 166.0, 22.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            card_data.title,
        )

        painter.setFont(jarvis_theme.ui_font(7.7, QFont.Weight.Normal))
        painter.setPen(QColor("#626977"))
        painter.drawText(
            QRectF(local_rect.left() + 16.0, local_rect.top() + 63.0, 152.0, 74.0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            card_data.subtitle,
        )

        pill_rect = QRectF(local_rect.left() + 16.0, local_rect.bottom() - 38.0, 96.0, 24.0)
        is_hovered = self._hovered_region_id == f"pill_{card_data.card_id}"
        self._paint_card_action_pill(painter, pill_rect, card_data.action_label, icon_kind="play", hovered=is_hovered)

        # Mini vector illustration of a scrolling ChatGPT page with neon underline + down arrow
        preview_rect = QRectF(local_rect.right() - 84.0, local_rect.top() + 56.0, 68.0, 90.0)
        painter.setPen(QPen(QColor("#D5E2F5"), 1.0))
        painter.setBrush(QColor("#F7FAFF"))
        painter.drawRoundedRect(preview_rect, 8.0, 8.0)

        for row_idx, width_ratio in enumerate((0.72, 0.85, 0.62, 0.78)):
            line_y = preview_rect.top() + 16.0 + row_idx * 13.0
            line_x0 = preview_rect.left() + 10.0
            line_x1 = line_x0 + (preview_rect.width() - 20.0) * width_ratio
            painter.setPen(QPen(QColor("#CBD5E1"), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(line_x0, line_y), QPointF(line_x1, line_y))
            if row_idx == 1:
                painter.setPen(QPen(QColor("#1E88FF"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(line_x0, line_y + 4.0), QPointF(line_x1, line_y + 4.0))

        # Scroll down chevron indicator
        arrow_cx = preview_rect.center().x()
        arrow_cy = preview_rect.bottom() - 14.0
        painter.setPen(QPen(QColor("#0A84FF"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(QPointF(arrow_cx - 6.0, arrow_cy - 3.0), QPointF(arrow_cx, arrow_cy + 3.0))
        painter.drawLine(QPointF(arrow_cx, arrow_cy + 3.0), QPointF(arrow_cx + 6.0, arrow_cy - 3.0))

    def _paint_lasso_explain_card_interior(
        self,
        painter: QPainter,
        local_rect: QRectF,
        header_rect: QRectF,
        card_data: ArcDockCardData,
        is_center: bool,
    ) -> None:
        painter.setFont(jarvis_theme.ui_font(8.4, QFont.Weight.DemiBold))
        painter.setPen(QColor("#1A78E5"))
        painter.drawText(header_rect, Qt.AlignmentFlag.AlignCenter, card_data.badge_text)

        painter.setFont(jarvis_theme.ui_font(12.2 if is_center else 13.0, QFont.Weight.DemiBold))
        painter.setPen(QColor("#222630"))
        painter.drawText(
            QRectF(local_rect.left() + 16.0, local_rect.top() + 42.0, local_rect.width() - 32.0, 26.0),
            Qt.AlignmentFlag.AlignCenter,
            card_data.title,
        )

        # Crisp miniature neon lasso loop
        loop_rect = QRectF(local_rect.center().x() - 34.0, local_rect.top() + 72.0, 68.0, 32.0)
        painter.setPen(QPen(QColor("#26B4FF"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(QColor(38, 180, 255, 28))
        painter.drawRoundedRect(loop_rect, 14.0, 14.0)

        if is_center:
            painter.setFont(jarvis_theme.ui_font(7.8, QFont.Weight.Normal))
            painter.setPen(QColor("#626977"))
            painter.drawText(
                QRectF(local_rect.left() + 20.0, local_rect.top() + 110.0, local_rect.width() - 40.0, 38.0),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                card_data.subtitle,
            )
            pill_rect = QRectF(local_rect.left() + 16.0, local_rect.bottom() - 38.0, 96.0, 24.0)
            is_hovered = self._hovered_region_id == f"pill_{card_data.card_id}"
            self._paint_card_action_pill(
                painter, pill_rect, card_data.action_label, icon_kind="play", hovered=is_hovered
            )

    def _paint_sketch_demo_card_interior(
        self,
        painter: QPainter,
        local_rect: QRectF,
        header_rect: QRectF,
        card_data: ArcDockCardData,
    ) -> None:
        painter.setFont(jarvis_theme.ui_font(8.4, QFont.Weight.DemiBold))
        painter.setPen(QColor("#3380FF"))
        painter.drawText(header_rect, Qt.AlignmentFlag.AlignCenter, card_data.badge_text)

        painter.setFont(jarvis_theme.ui_font(11.6, QFont.Weight.DemiBold))
        painter.setPen(QColor("#262A34"))
        painter.drawText(
            QRectF(local_rect.left() + 16.0, local_rect.top() + 38.0, 164.0, 22.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            card_data.title,
        )

        painter.setFont(jarvis_theme.ui_font(7.8, QFont.Weight.Normal))
        painter.setPen(QColor("#6A707E"))
        painter.drawText(
            QRectF(local_rect.left() + 16.0, local_rect.top() + 64.0, 152.0, 74.0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            card_data.subtitle,
        )

        # Mini live-sketch illustration badge on right
        sketch_rect = QRectF(local_rect.right() - 82.0, local_rect.top() + 58.0, 66.0, 86.0)
        painter.setPen(QPen(QColor("#DFDBD1"), 1.0))
        painter.setBrush(QColor("#FFFDF9"))
        painter.drawRoundedRect(sketch_rect, 8.0, 8.0)
        painter.setPen(QPen(QColor("#1E7BFF"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(sketch_rect.center().x(), sketch_rect.top() + 28.0), 14.0, 11.0)
        arrow_path = QPainterPath()
        arrow_path.moveTo(sketch_rect.left() + 14.0, sketch_rect.bottom() - 22.0)
        arrow_path.quadTo(
            sketch_rect.center().x(),
            sketch_rect.center().y() + 6.0,
            sketch_rect.right() - 16.0,
            sketch_rect.top() + 34.0,
        )
        painter.drawPath(arrow_path)

        pill_rect = QRectF(local_rect.left() + 16.0, local_rect.bottom() - 38.0, 92.0, 24.0)
        is_hovered = self._hovered_region_id == f"pill_{card_data.card_id}"
        self._paint_card_action_pill(painter, pill_rect, card_data.action_label, icon_kind="play", hovered=is_hovered)

    def _paint_card_action_pill(
        self,
        painter: QPainter,
        pill_rect: QRectF,
        label_text: str,
        icon_kind: str,
        hovered: bool,
    ) -> None:
        fill_color = QColor("#D5ECFF") if hovered else QColor("#E4F3FF")
        painter.setPen(QPen(QColor("#78C4FF"), 1.0))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(pill_rect, pill_rect.height() / 2.0, pill_rect.height() / 2.0)

        icon_x = pill_rect.left() + 15.0
        icon_y = pill_rect.center().y()
        accent_color = QColor("#007CD6")
        if icon_kind == "pdf":
            painter.setPen(QPen(accent_color, 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(icon_x - 4.0, icon_y - 5.2, 8.0, 10.4), 1.5, 1.5)
            painter.drawLine(QPointF(icon_x - 2.0, icon_y - 1.2), QPointF(icon_x + 2.0, icon_y - 1.2))
            painter.drawLine(QPointF(icon_x - 2.0, icon_y + 1.8), QPointF(icon_x + 2.0, icon_y + 1.8))
        else:
            play_triangle = QPainterPath()
            play_triangle.moveTo(icon_x - 3.0, icon_y - 4.2)
            play_triangle.lineTo(icon_x + 4.2, icon_y)
            play_triangle.lineTo(icon_x - 3.0, icon_y + 4.2)
            play_triangle.closeSubpath()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent_color)
            painter.drawPath(play_triangle)

        painter.setFont(jarvis_theme.ui_font(8.2, QFont.Weight.DemiBold))
        painter.setPen(accent_color)
        painter.drawText(
            QRectF(pill_rect.left() + 23.0, pill_rect.top(), pill_rect.width() - 28.0, pill_rect.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            label_text,
        )

    def _paint_studio_pill(self, painter: QPainter, bounds: QRectF, label_text: str) -> None:
        hovered = self._hovered_region_id == "top_scroll_explain"
        gradient = QLinearGradient(bounds.topLeft(), bounds.bottomLeft())
        gradient.setColorAt(0.0, QColor("#5AB6FF") if hovered else QColor("#3A9BFF"))
        gradient.setColorAt(1.0, QColor("#2983FF") if hovered else QColor("#1D6FE8"))
        painter.setPen(QPen(QColor(175, 224, 255, 190), 1.0))
        painter.setBrush(gradient)
        painter.drawRoundedRect(bounds, bounds.height() / 2.0, bounds.height() / 2.0)

        painter.setFont(jarvis_theme.ui_font(8.4, QFont.Weight.DemiBold))
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(bounds, Qt.AlignmentFlag.AlignCenter, label_text)

    def _paint_footer_pill(
        self,
        painter: QPainter,
        bounds: QRectF,
        region_id: str,
        label_text: str,
        icon_kind: str,
        is_light_pill: bool,
    ) -> None:
        hovered = self._hovered_region_id == region_id
        if is_light_pill:
            fill_color = QColor("#FFFFFF") if hovered else QColor("#F3F4F8")
            border_color = QColor(255, 255, 255, 200)
            foreground_color = QColor("#1D1F26")
            icon_color = QColor("#2578D4")
        else:
            fill_color = QColor("#2A2D36") if hovered else QColor("#1B1D23")
            border_color = QColor(255, 255, 255, 34)
            foreground_color = QColor("#EEF1F7")
            icon_color = QColor("#DCE1EC")

        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(bounds, bounds.height() / 2.0, bounds.height() / 2.0)

        icon_x = bounds.left() + 17.0
        icon_y = bounds.center().y()
        self._paint_vector_glyph(painter, icon_kind, QPointF(icon_x, icon_y), icon_color)

        painter.setFont(jarvis_theme.ui_font(8.6, QFont.Weight.DemiBold))
        painter.setPen(foreground_color)
        painter.drawText(
            QRectF(bounds.left() + 27.0, bounds.top(), bounds.width() - 34.0, bounds.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            label_text,
        )

    def _paint_circle_icon_button(
        self,
        painter: QPainter,
        bounds: QRectF,
        region_id: str,
        icon_kind: str,
        active: bool,
        slashed: bool = False,
        subtle: bool = False,
    ) -> None:
        hovered = self._hovered_region_id == region_id
        if subtle:
            fill_color = QColor(42, 45, 56, 210) if hovered else QColor(24, 26, 33, 155)
            border_color = QColor(255, 255, 255, 56 if hovered else 24)
            icon_color = QColor(235, 240, 250, 235 if hovered else 175)
        else:
            fill_color = QColor("#2D303A") if hovered else QColor("#1C1E24" if active else "#14161A")
            border_color = QColor(255, 255, 255, 56 if hovered else (38 if active else 20))
            icon_color = QColor(238, 242, 250, 245 if active else 130)

        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(fill_color)
        painter.drawEllipse(bounds)
        self._paint_vector_glyph(painter, icon_kind, bounds.center(), icon_color, slashed=slashed)

    @staticmethod
    def _paint_vector_glyph(
        painter: QPainter,
        icon_kind: str,
        center: QPointF,
        color: QColor,
        slashed: bool = False,
    ) -> None:
        painter.save()
        glyph_pen = QPen(color, 1.45, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(glyph_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if icon_kind == "glasses":
            painter.drawEllipse(QPointF(center.x() - 4.2, center.y() + 0.5), 3.1, 3.1)
            painter.drawEllipse(QPointF(center.x() + 4.2, center.y() + 0.5), 3.1, 3.1)
            painter.drawArc(QRectF(center.x() - 1.6, center.y() - 1.2, 3.2, 3.0), 20 * 16, 140 * 16)
            if slashed:
                painter.drawLine(
                    QPointF(center.x() - 6.0, center.y() + 5.2),
                    QPointF(center.x() + 6.0, center.y() - 5.2),
                )
        elif icon_kind == "chat":
            bubble_path = QPainterPath()
            bubble_path.addRoundedRect(QRectF(center.x() - 6.0, center.y() - 5.0, 12.0, 9.0), 3.2, 3.2)
            painter.drawPath(bubble_path)
            painter.drawLine(QPointF(center.x() - 2.8, center.y() + 4.0), QPointF(center.x() - 4.5, center.y() + 6.0))
            painter.drawLine(QPointF(center.x() - 4.5, center.y() + 6.0), QPointF(center.x() - 0.5, center.y() + 4.0))
        elif icon_kind == "gear":
            painter.drawEllipse(center, 2.3, 2.3)
            painter.drawEllipse(center, 4.7, 4.7)
            for tooth_index in range(8):
                angle = tooth_index * (math.pi / 4.0)
                inner_point = QPointF(center.x() + math.cos(angle) * 4.8, center.y() + math.sin(angle) * 4.8)
                outer_point = QPointF(center.x() + math.cos(angle) * 6.5, center.y() + math.sin(angle) * 6.5)
                painter.drawLine(inner_point, outer_point)
        elif icon_kind == "sparkle":
            lasso_path = QPainterPath()
            lasso_path.addRoundedRect(QRectF(center.x() - 5.2, center.y() - 3.8, 10.4, 7.6), 3.6, 3.6)
            painter.drawPath(lasso_path)
            painter.drawLine(QPointF(center.x() + 2.5, center.y() + 2.2), QPointF(center.x() + 5.4, center.y() + 5.2))
        elif icon_kind == "gift":
            painter.drawRoundedRect(QRectF(center.x() - 5.0, center.y() - 1.5, 10.0, 6.8), 1.4, 1.4)
            painter.drawRoundedRect(QRectF(center.x() - 5.8, center.y() - 3.8, 11.6, 2.6), 1.0, 1.0)
            painter.drawLine(QPointF(center.x(), center.y() - 3.8), QPointF(center.x(), center.y() + 5.2))
            painter.drawEllipse(QPointF(center.x() - 2.0, center.y() - 5.0), 1.9, 1.4)
            painter.drawEllipse(QPointF(center.x() + 2.0, center.y() - 5.0), 1.9, 1.4)
        elif icon_kind == "menu":
            for line_offset in (-3.2, 0.0, 3.2):
                painter.drawLine(
                    QPointF(center.x() - 4.8, center.y() + line_offset),
                    QPointF(center.x() + 4.8, center.y() + line_offset),
                )
        elif icon_kind == "chevron_left":
            painter.drawLine(QPointF(center.x() + 1.8, center.y() - 3.8), QPointF(center.x() - 2.2, center.y()))
            painter.drawLine(QPointF(center.x() - 2.2, center.y()), QPointF(center.x() + 1.8, center.y() + 3.8))
        elif icon_kind == "chevron_right":
            painter.drawLine(QPointF(center.x() - 1.8, center.y() - 3.8), QPointF(center.x() + 2.2, center.y()))
            painter.drawLine(QPointF(center.x() + 2.2, center.y()), QPointF(center.x() - 1.8, center.y() + 3.8))

        painter.restore()


# ---------------------------------------------------------------------------
# Visual Snapshot Exporter & CLI Entry Point
# ---------------------------------------------------------------------------


def save_visual_snapshots(output_directory: str) -> List[str]:
    """
    Renders and saves PNG snapshots of all core NotchWindow visual modes for
    automated visual inspection.
    """
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    jarvis_theme.register_bundled_fonts()

    os.makedirs(output_directory, exist_ok=True)
    settings_store = jarvis_config.SettingsStore()
    notch_window = NotchWindow(settings_store=settings_store)
    notch_window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    notch_window.show()

    snapshot_specs: List[Tuple[str, str, str, float, str, bool]] = [
        ("idle.png", NotchVisualMode.IDLE, "", 0.0, "", False),
        ("listening.png", NotchVisualMode.LISTENING, "Listening", 0.68, "", False),
        ("thinking.png", NotchVisualMode.THINKING, "Thinking", 0.0, "", False),
        ("speaking.png", NotchVisualMode.SPEAKING, "Speaking", 0.56, "", False),
        ("text_input.png", NotchVisualMode.TEXT_INPUT, "", 0.0, "", False),
        (
            "text_response.png",
            NotchVisualMode.TEXT_RESPONSE,
            "",
            0.0,
            (
                "Hey! I looked at your screen — you can click Expand for the full chat, "
                "stop generation at any time, or type a follow-up directly below while this "
                "answer streams cleanly."
            ),
            True,
        ),
        ("home_peek.png", NotchVisualMode.HOME_PEEK, "", 0.0, "", False),
    ]

    saved_paths: List[str] = []
    try:
        for file_name, mode_name, status_label, audio_level, response_text, is_generating in snapshot_specs:
            if response_text:
                notch_window.set_streaming_response(response_text, is_generating=is_generating)
            else:
                notch_window.set_mode(mode_name, status_label)
            notch_window.set_audio_level(audio_level)
            notch_window._snap_springs_to_targets()
            notch_window._sync_window_envelope(force_shrink=True)
            notch_window._layout_typing_children()
            application.processEvents()

            hd_scale = 2.0
            snapshot_pixmap = QPixmap(
                int(notch_window.width() * hd_scale),
                int(notch_window.height() * hd_scale),
            )
            snapshot_pixmap.setDevicePixelRatio(hd_scale)
            snapshot_pixmap.fill(Qt.GlobalColor.transparent)
            notch_window.render(snapshot_pixmap)
            output_path = os.path.join(output_directory, file_name)
            snapshot_pixmap.save(output_path, "PNG")
            saved_paths.append(output_path)
    finally:
        notch_window.close()

    return saved_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis NotchWindow snapshot & benchmark CLI")
    parser.add_argument("--snapshot", metavar="DIR", help="Render PNG snapshots to DIR")
    parser.add_argument("--iterations", type=int, default=60, help="Benchmark frame iterations per mode")
    parsed_args = parser.parse_args()

    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    jarvis_theme.register_bundled_fonts()

    if parsed_args.snapshot:
        saved_paths = save_visual_snapshots(parsed_args.snapshot)
        print(f"Saved {len(saved_paths)} snapshots to {parsed_args.snapshot}")
        for saved_path in saved_paths:
            print(f"  - {saved_path}")

    settings_store = jarvis_config.SettingsStore()
    benchmark_window = NotchWindow(settings_store=settings_store)
    benchmark_window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    benchmark_window.show()
    try:
        timings = benchmark_window.benchmark_frame_ms(iterations=parsed_args.iterations)
        print("benchmark_frame_ms:", timings)
    finally:
        benchmark_window.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
