"""
PySide6 full-screen companion cursor, Draw-to-Ask highlighter, 2D screen drawing
overlay, and live caption window for Jarvis Windows v1.1.5.

Renders directly on a transparent, non-activating, click-through top-level
widget (`OverlayCanvasWindow`) paired with a non-activating clickable floating
`ClearOverlayButtonWindow` (`[ x Clear ]`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEasingCurve, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QApplication, QWidget

import jarvis_config
import jarvis_response_parser
import jarvis_theme

FRAME_INTERVAL_MILLISECONDS = 16
CURSOR_SPRING_RESPONSE_SECONDS = 0.13
CURSOR_SPRING_DAMPING_RATIO = 0.90
CURSOR_OPACITY_SPRING_RESPONSE_SECONDS = 0.22
CURSOR_MORPH_SPRING_RESPONSE_SECONDS = 0.16

CURSOR_HIDE_MESSAGE = "okay see you <3"
CURSOR_HIDE_HOLD_SECONDS = 1.2
CURSOR_HIDE_FADE_DELAY_SECONDS = 0.78
CURSOR_TARGET_RELEASE_MOUSE_DISTANCE = 55.0

HIGHLIGHT_SAMPLE_MIN_DISTANCE = 4.0
HIGHLIGHT_FADE_DURATION_SECONDS = 0.8
HIGHLIGHT_CORE_WIDTH = 8.0
HIGHLIGHT_GLOW_WIDTH = 18.0

DRAWING_ANIMATE_IN_SECONDS = 0.85
DEFAULT_DRAWING_HOLD_SECONDS = 24.0
DRAWING_FADE_OUT_SECONDS = 0.45

PAPER_CORNER_RADIUS = 14.0
PAPER_HEADER_HEIGHT = 28.0
PAPER_GRID_SPACING = 22.0

SHAKE_WINDOW_SECONDS = 0.5
SHAKE_MIN_REVERSALS = 6
SHAKE_MIN_PATH_LENGTH = 260.0
SHAKE_MIN_DIRECTION_SEGMENT = 16.0
SHAKE_PARK_SECONDS = 3.0
SHAKE_COOLDOWN_SECONDS = 1.0

CLEAR_BUTTON_PILL_WIDTH = 92.0
CLEAR_BUTTON_PILL_HEIGHT = 32.0
CLEAR_BUTTON_SHADOW_MARGIN = 10
CLEAR_BUTTON_OFFSET_FROM_RIGHT = 220
CLEAR_BUTTON_TOP_Y = 96

CAPTION_BOTTOM_CENTER_OFFSET = 86
CAPTION_CORNER_RADIUS = 16.0
CAPTION_MAX_WIDTH = 780.0


@dataclass
class _PaperNode:
    """Single box or document node inside a paper flow-diagram row."""

    label: str
    is_document: bool = False


@dataclass
class _PaperRow:
    """Parsed structural row from a `paper` command body."""

    kind: str  # "section" | "flow" | "compare" | "tree" | "decision" | "clue" | "note"
    text: str = ""
    nodes: List[_PaperNode] = field(default_factory=list)


@dataclass
class _ActiveDrawingItem:
    """State wrapper for one active screen-drawing command."""

    command: jarvis_response_parser.DrawingCommand
    created_at: float
    delay_seconds: float
    duration_seconds: float
    expires_at: float
    seed_value: int
    has_started_sound: bool = False
    cached_path: Optional[QPainterPath] = None
    cached_paper_pixmap: Optional[QPixmap] = None
    cached_paper_size: Tuple[int, int] = (0, 0)
    cached_stroke_pixmap: Optional[QPixmap] = None
    cached_stroke_origin: Tuple[float, float] = (0.0, 0.0)


class _ShakeToDetachDetector:
    """
    Detects rapid mouse direction reversals (`>= 6` reversals and `>= 260px` travel
    within `0.5s`) to temporarily park the floating companion cursor.
    """

    def __init__(self) -> None:
        self._samples: List[Tuple[float, float, float]] = []
        self._cooldown_until: float = 0.0

    def reset(self) -> None:
        self._samples.clear()

    def feed_sample(self, cursor_x: float, cursor_y: float, now_seconds: float) -> bool:
        if now_seconds < self._cooldown_until:
            return False

        self._samples.append((now_seconds, cursor_x, cursor_y))
        cutoff_time = now_seconds - SHAKE_WINDOW_SECONDS
        while self._samples and self._samples[0][0] < cutoff_time:
            self._samples.pop(0)

        if len(self._samples) < 8:
            return False

        total_travel = 0.0
        reversal_count = 0
        previous_direction_x = 0.0
        previous_direction_y = 0.0
        accumulated_dx = 0.0
        accumulated_dy = 0.0

        for sample_index in range(1, len(self._samples)):
            _, prev_x, prev_y = self._samples[sample_index - 1]
            _, curr_x, curr_y = self._samples[sample_index]
            delta_x = curr_x - prev_x
            delta_y = curr_y - prev_y
            step_distance = math.hypot(delta_x, delta_y)
            if step_distance <= 0.0:
                continue

            total_travel += step_distance
            accumulated_dx += delta_x
            accumulated_dy += delta_y
            segment_length = math.hypot(accumulated_dx, accumulated_dy)

            if segment_length >= SHAKE_MIN_DIRECTION_SEGMENT:
                dir_x = accumulated_dx / segment_length
                dir_y = accumulated_dy / segment_length
                if previous_direction_x != 0.0 or previous_direction_y != 0.0:
                    dot_product = dir_x * previous_direction_x + dir_y * previous_direction_y
                    if dot_product < -0.35:
                        reversal_count += 1
                previous_direction_x = dir_x
                previous_direction_y = dir_y
                accumulated_dx = 0.0
                accumulated_dy = 0.0

        if reversal_count >= SHAKE_MIN_REVERSALS and total_travel >= SHAKE_MIN_PATH_LENGTH:
            self._samples.clear()
            self._cooldown_until = now_seconds + SHAKE_COOLDOWN_SECONDS
            return True
        return False


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _ease_out_cubic(progress: float) -> float:
    clamped = _clamp_float(progress, 0.0, 1.0)
    inverse = 1.0 - clamped
    return 1.0 - inverse * inverse * inverse


def _ease_in_out_cubic(progress: float) -> float:
    clamped = _clamp_float(progress, 0.0, 1.0)
    if clamped < 0.5:
        return 4.0 * clamped * clamped * clamped
    adjusted = -2.0 * clamped + 2.0
    return 1.0 - (adjusted * adjusted * adjusted) / 2.0


def _resolve_tone_color(tone_name: str, fallback_color: QColor) -> QColor:
    cleaned = (tone_name or "").strip().lower()
    if cleaned in jarvis_theme.DRAWING_TONES:
        return QColor(jarvis_theme.DRAWING_TONES[cleaned])
    if cleaned.startswith("#"):
        candidate = QColor(cleaned)
        if candidate.isValid():
            return candidate
    return QColor(fallback_color)


def _blend_colors(first_color: QColor, second_color: QColor, ratio: float) -> QColor:
    clamped = _clamp_float(ratio, 0.0, 1.0)
    inv_ratio = 1.0 - clamped
    return QColor(
        int(first_color.red() * inv_ratio + second_color.red() * clamped),
        int(first_color.green() * inv_ratio + second_color.green() * clamped),
        int(first_color.blue() * inv_ratio + second_color.blue() * clamped),
        int(first_color.alpha() * inv_ratio + second_color.alpha() * clamped),
    )


def _build_smooth_catmull_rom_path(
    points: List[Tuple[float, float]],
    closed: bool = False,
) -> QPainterPath:
    """Converts a point sequence into a C1-smooth cubic Bezier QPainterPath."""
    path = QPainterPath()
    point_count = len(points)
    if point_count == 0:
        return path
    if point_count == 1:
        path.moveTo(points[0][0], points[0][1])
        return path
    if point_count == 2 and not closed:
        path.moveTo(points[0][0], points[0][1])
        path.lineTo(points[1][0], points[1][1])
        return path

    if closed:
        extended = [points[-1], *points, points[0], points[1]]
        segment_end = point_count + 1
    else:
        extended = [points[0], *points, points[-1]]
        segment_end = point_count

    path.moveTo(extended[1][0], extended[1][1])
    for index in range(1, segment_end):
        p0_x, p0_y = extended[index - 1]
        p1_x, p1_y = extended[index]
        p2_x, p2_y = extended[index + 1]
        p3_x, p3_y = extended[index + 2]

        ctrl1_x = p1_x + (p2_x - p0_x) / 6.0
        ctrl1_y = p1_y + (p2_y - p0_y) / 6.0
        ctrl2_x = p2_x - (p3_x - p1_x) / 6.0
        ctrl2_y = p2_y - (p3_y - p1_y) / 6.0
        path.cubicTo(ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y, p2_x, p2_y)
    return path


def _build_wobbly_circle_points(
    center_x: float,
    center_y: float,
    radius: float,
    seed_value: int,
    step_count: int = 44,
) -> List[Tuple[float, float]]:
    """Generates deterministic organic hand-drawn circle points with slight overshoot."""
    generator = random.Random(seed_value)
    start_angle = generator.uniform(0.0, math.pi * 2.0)
    phase_one = generator.uniform(0.0, math.pi * 2.0)
    phase_two = generator.uniform(0.0, math.pi * 2.0)
    total_sweep = math.pi * 2.0 * 1.07

    points: List[Tuple[float, float]] = []
    for step_index in range(step_count + 1):
        fraction = step_index / float(step_count)
        angle = start_angle + total_sweep * fraction
        wobble_factor = (
            1.0
            + 0.028 * math.sin(angle * 2.0 + phase_one)
            + 0.018 * math.cos(angle * 3.0 + phase_two)
            + 0.022 * fraction
        )
        point_radius = radius * wobble_factor
        points.append((center_x + math.cos(angle) * point_radius, center_y + math.sin(angle) * point_radius))
    return points


def _build_wobbly_line_points(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    seed_value: int,
    bow_amplitude: float = 1.6,
    samples: int = 8,
) -> List[Tuple[float, float]]:
    """Builds slightly bowed hand-sketched segment points between two endpoints."""
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length = math.hypot(delta_x, delta_y) or 1.0
    normal_x = -delta_y / length
    normal_y = delta_x / length

    generator = random.Random(seed_value)
    bow_sign = -1.0 if generator.random() < 0.5 else 1.0
    bow_scale = bow_sign * min(bow_amplitude, max(0.4, length * 0.018)) * generator.uniform(0.65, 1.15)
    ripple_phase = generator.uniform(0.0, math.pi * 2.0)

    points: List[Tuple[float, float]] = []
    for step_index in range(samples + 1):
        fraction = step_index / float(samples)
        offset = math.sin(fraction * math.pi) * bow_scale + math.sin(fraction * math.pi * 3.0 + ripple_phase) * 0.35
        points.append((start_x + delta_x * fraction + normal_x * offset, start_y + delta_y * fraction + normal_y * offset))
    return points


def _build_wobbly_rect_points(
    rect_x: float,
    rect_y: float,
    rect_width: float,
    rect_height: float,
    corner_radius: float,
    seed_value: int,
) -> List[Tuple[float, float]]:
    """Builds a hand-drawn rounded frame polyline with subtle organic edge bow."""
    clamped_radius = min(corner_radius, rect_width * 0.35, rect_height * 0.35)
    corners = [
        (rect_x + clamped_radius, rect_y),
        (rect_x + rect_width - clamped_radius, rect_y),
        (rect_x + rect_width, rect_y + clamped_radius),
        (rect_x + rect_width, rect_y + rect_height - clamped_radius),
        (rect_x + rect_width - clamped_radius, rect_y + rect_height),
        (rect_x + clamped_radius, rect_y + rect_height),
        (rect_x, rect_y + rect_height - clamped_radius),
        (rect_x, rect_y + clamped_radius),
    ]
    points: List[Tuple[float, float]] = []
    edge_count = len(corners)
    for corner_index in range(edge_count):
        origin = corners[corner_index]
        target = corners[(corner_index + 1) % edge_count]
        is_straight_edge = (corner_index % 2 == 0)
        segment_points = _build_wobbly_line_points(
            origin[0],
            origin[1],
            target[0],
            target[1],
            seed_value=seed_value + corner_index * 31,
            bow_amplitude=2.2 if is_straight_edge else 0.7,
            samples=7 if is_straight_edge else 3,
        )
        if corner_index > 0 and segment_points:
            segment_points = segment_points[1:]
        points.extend(segment_points)
    return points


def _build_underline_wave_points(
    start_x: float,
    start_y: float,
    span_width: float,
    seed_value: int,
) -> List[Tuple[float, float]]:
    """Builds steady, high-precision horizontal underline points with minimal micro-variation."""
    step_count = max(10, int(span_width / 20.0))
    phase = (seed_value % 97) * 0.13
    points: List[Tuple[float, float]] = []
    for step_index in range(step_count + 1):
        fraction = step_index / float(step_count)
        point_x = start_x + span_width * fraction
        point_y = start_y + math.sin(fraction * math.pi * 2.1 + phase) * 0.25
        points.append((point_x, point_y))
    return points


def _paint_glow_stroke(
    painter: QPainter,
    path: QPainterPath,
    core_color: QColor,
    core_width: float,
    glow_width: float,
    opacity: float = 1.0,
) -> None:
    """Shared multi-pass neon glow pen renderer for underlines, rings, arrows, and highlights."""
    if path.isEmpty() or opacity <= 0.01:
        return

    painter.save()
    painter.setBrush(Qt.BrushStyle.NoBrush)

    bright_core = _blend_colors(core_color, QColor(200, 242, 255), 0.48)
    outer_glow_color = QColor(core_color.red(), core_color.green(), core_color.blue())

    passes = [
        (glow_width, jarvis_theme.with_alpha(outer_glow_color, int(72 * opacity))),
        ((glow_width + core_width) * 0.54, jarvis_theme.with_alpha(core_color, int(148 * opacity))),
        (core_width, jarvis_theme.with_alpha(core_color, int(238 * opacity))),
        (max(1.4, core_width * 0.28), jarvis_theme.with_alpha(bright_core, int(195 * opacity))),
    ]

    for stroke_width, stroke_color in passes:
        pen = QPen(stroke_color, stroke_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)

    painter.restore()


def _paint_text_with_halo(
    painter: QPainter,
    origin_x: float,
    baseline_y: float,
    text: str,
    font: QFont,
    text_color: QColor,
    halo_color: QColor,
    halo_width: float = 5.0,
) -> None:
    """Draws crisp handwritten heading text surrounded by a soft two-pass outline halo."""
    if not text:
        return
    text_path = QPainterPath()
    text_path.addText(origin_x, baseline_y, font, text)

    painter.save()
    outer_halo = QColor(halo_color.red(), halo_color.green(), halo_color.blue(), int(halo_color.alpha() * 0.58))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(outer_halo, halo_width * 1.45, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(text_path)

    painter.setPen(QPen(halo_color, halo_width * 0.85, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(text_path)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(text_color)
    painter.drawPath(text_path)
    painter.restore()


def _paint_callout_pill(
    painter: QPainter,
    center_x: float,
    top_y: float,
    text: str,
    accent_color: QColor,
    opacity: float = 1.0,
) -> QRectF:
    """Renders a dark crisp callout label pill centered horizontally at `center_x`."""
    cleaned = jarvis_response_parser.clean_spoken_text(text)
    if not cleaned:
        return QRectF()

    font = jarvis_theme.ui_font(10.0, QFont.Weight.DemiBold)
    metrics = QFontMetricsF(font)
    text_width = metrics.horizontalAdvance(cleaned)
    pill_width = max(56.0, text_width + 20.0)
    pill_height = 24.0
    pill_rect = QRectF(center_x - pill_width / 2.0, top_y, pill_width, pill_height)

    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, int(80 * opacity)))
    painter.drawRoundedRect(pill_rect.translated(0.0, 2.0), 8.0, 8.0)

    painter.setBrush(QColor(18, 22, 30, int(234 * opacity)))
    painter.setPen(QPen(jarvis_theme.with_alpha(accent_color, int(205 * opacity)), 1.1))
    painter.drawRoundedRect(pill_rect, 8.0, 8.0)

    painter.setFont(font)
    painter.setPen(QColor(246, 249, 255, int(250 * opacity)))
    painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, cleaned)
    painter.restore()
    return pill_rect


def _paint_clear_button_pill(
    painter: QPainter,
    pill_rect: QRectF,
    hovered: bool = False,
    pressed: bool = False,
) -> None:
    """Shared painter for the floating dark `[ x Clear ]` top-right pill button."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    radius = pill_rect.height() / 2.0

    # Soft multi-layer elevation shadow
    for offset_y, expand_px, shadow_alpha in ((4.0, 2.5, 28), (2.0, 1.0, 44)):
        shadow_rect = pill_rect.adjusted(-expand_px, -expand_px + offset_y, expand_px, expand_px + offset_y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, shadow_alpha))
        painter.drawRoundedRect(shadow_rect, radius + expand_px, radius + expand_px)

    fill_color = QColor(36, 36, 44, 250) if pressed else (QColor(28, 28, 34, 252) if hovered else QColor(18, 18, 22, 245))
    border_color = QColor(82, 82, 91, 250) if hovered else QColor(63, 63, 70, 235)

    draw_rect = pill_rect.translated(0.0, 1.0) if pressed else pill_rect
    painter.setBrush(fill_color)
    painter.setPen(QPen(border_color, 1.0))
    painter.drawRoundedRect(draw_rect, radius, radius)

    # Crisp vector 'x' icon on the left side of the pill
    ink_color = QColor(244, 244, 246, 245)
    icon_center_x = draw_rect.left() + 21.0
    icon_center_y = draw_rect.center().y()
    cross_half = 4.0
    cross_pen = QPen(ink_color, 1.65, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    painter.setPen(cross_pen)
    painter.drawLine(
        QPointF(icon_center_x - cross_half, icon_center_y - cross_half),
        QPointF(icon_center_x + cross_half, icon_center_y + cross_half),
    )
    painter.drawLine(
        QPointF(icon_center_x + cross_half, icon_center_y - cross_half),
        QPointF(icon_center_x - cross_half, icon_center_y + cross_half),
    )

    label_font = jarvis_theme.ui_font(10.5, QFont.Weight.DemiBold)
    painter.setFont(label_font)
    painter.setPen(ink_color)
    text_rect = QRectF(draw_rect.left() + 30.0, draw_rect.top(), draw_rect.width() - 40.0, draw_rect.height())
    painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "Clear")
    painter.restore()


def _compact_diagram_node_label(raw_label: str) -> str:
    """Keeps node labels inside diagram boxes crisp and scannable (1 to 3 words)."""
    cleaned_label = jarvis_response_parser.clean_spoken_text(raw_label or "")
    if not cleaned_label:
        return "Topic"
    word_tokens = cleaned_label.split()
    if len(word_tokens) <= 3 and len(cleaned_label) <= 26:
        return cleaned_label
    return jarvis_response_parser.extract_key_topic_phrase(cleaned_label, max_words=3)


def _parse_paper_body_rows(body_text: str) -> List[_PaperRow]:
    """Parses `|` or newline-delimited paper card rows strictly into visual diagram sections and flow rows."""
    if not body_text:
        return []
    normalized = body_text.replace("\n", "|")
    raw_segments = [segment.strip() for segment in normalized.split("|") if segment.strip()]
    parsed_rows: List[_PaperRow] = []
    pending_topic_nodes: List[_PaperNode] = []

    def _flush_pending_topics() -> None:
        nonlocal pending_topic_nodes
        while len(pending_topic_nodes) >= 2:
            left_node = pending_topic_nodes.pop(0)
            right_node = pending_topic_nodes.pop(0)
            parsed_rows.append(_PaperRow(kind="flow", nodes=[left_node, right_node]))
        if len(pending_topic_nodes) == 1:
            solo_node = pending_topic_nodes.pop(0)
            solo_node.is_document = True
            parsed_rows.append(
                _PaperRow(
                    kind="flow",
                    nodes=[solo_node, _PaperNode(label="Key Point", is_document=False)],
                )
            )

    for segment in raw_segments:
        cleaned_segment = jarvis_response_parser.clean_spoken_text(segment)
        if not cleaned_segment:
            continue
        lowered_segment = cleaned_segment.lower()

        if lowered_segment.startswith(("clue:", "memory:", "pattern:", "tip:")):
            _flush_pending_topics()
            clue_body = cleaned_segment.split(":", 1)[1].strip()
            if clue_body:
                parsed_rows.append(_PaperRow(kind="clue", text=clue_body))
            continue

        if lowered_segment.startswith("section:"):
            _flush_pending_topics()
            section_title = cleaned_segment.split(":", 1)[1].strip()
            if section_title:
                parsed_rows.append(_PaperRow(kind="section", text=section_title))
            continue

        normalized_arrow_segment = cleaned_segment.replace("→", "->").replace("=>", "->")
        has_vs_separator = bool(re.search(r"\b(?:vs\.?|versus)\b", cleaned_segment, flags=re.IGNORECASE))
        if lowered_segment.startswith(("compare:", "vs:")) or has_vs_separator:
            _flush_pending_topics()
            compare_body = (
                cleaned_segment.split(":", 1)[1].strip()
                if lowered_segment.startswith(("compare:", "vs:"))
                else cleaned_segment
            )
            compare_parts = [
                part.strip()
                for part in re.split(r"\s+(?:vs\.?|versus)\s+|/", compare_body, maxsplit=1, flags=re.IGNORECASE)
                if part.strip()
            ]
            if len(compare_parts) < 2 and "," in compare_body:
                compare_parts = [part.strip() for part in compare_body.split(",", 1) if part.strip()]
            if len(compare_parts) >= 2:
                parsed_rows.append(
                    _PaperRow(
                        kind="compare",
                        nodes=[
                            _PaperNode(label=_compact_diagram_node_label(compare_parts[0]), is_document=False),
                            _PaperNode(label=_compact_diagram_node_label(compare_parts[1]), is_document=False),
                        ],
                    )
                )
                continue

        if lowered_segment.startswith(("tree:", "hierarchy:")) or (">" in cleaned_segment and "->" not in normalized_arrow_segment):
            _flush_pending_topics()
            tree_body = (
                cleaned_segment.split(":", 1)[1].strip()
                if lowered_segment.startswith(("tree:", "hierarchy:"))
                else cleaned_segment
            )
            if ">" in tree_body:
                parent_raw, children_raw = tree_body.split(">", 1)
                parent_label = _compact_diagram_node_label(parent_raw)
                raw_children = [item.strip() for item in children_raw.split(",") if item.strip()]
                if len(raw_children) < 2 and "/" in children_raw:
                    raw_children = [item.strip() for item in children_raw.split("/") if item.strip()]
                child_nodes = [
                    _PaperNode(label=_compact_diagram_node_label(child_text), is_document=False)
                    for child_text in raw_children[:3]
                ]
                if not child_nodes:
                    child_nodes = [_PaperNode(label="Type A"), _PaperNode(label="Type B")]
                parsed_rows.append(_PaperRow(kind="tree", text=parent_label, nodes=child_nodes))
                continue

        has_decision_branches = ("?" in normalized_arrow_segment and ("/" in normalized_arrow_segment or "yes:" in lowered_segment or "no:" in lowered_segment))
        if lowered_segment.startswith(("if:", "decision:")) or has_decision_branches:
            _flush_pending_topics()
            decision_body = (
                normalized_arrow_segment.split(":", 1)[1].strip()
                if lowered_segment.startswith(("if:", "decision:"))
                else normalized_arrow_segment
            )
            condition_raw = decision_body
            branches_raw = ""
            if "?" in decision_body:
                condition_raw, branches_raw = decision_body.split("?", 1)
                condition_label = _compact_diagram_node_label(condition_raw) + "?"
            elif "->" in decision_body:
                condition_raw, branches_raw = decision_body.split("->", 1)
                condition_label = _compact_diagram_node_label(condition_raw)
            else:
                condition_label = _compact_diagram_node_label(condition_raw)
            branches_raw = branches_raw.lstrip("- >:").strip()
            branch_tokens = [item.strip() for item in re.split(r"\s*/\s*|\s*,\s*", branches_raw) if item.strip()]
            branch_nodes: List[_PaperNode] = []
            for branch_text in branch_tokens[:2]:
                clean_branch = re.sub(r"^(?:yes|no)\s*:\s*", "", branch_text, flags=re.IGNORECASE).strip()
                branch_nodes.append(_PaperNode(label=_compact_diagram_node_label(clean_branch or branch_text)))
            if not branch_nodes:
                branch_nodes = [_PaperNode(label="Yes Path"), _PaperNode(label="No Path")]
            parsed_rows.append(_PaperRow(kind="decision", text=condition_label, nodes=branch_nodes))
            continue

        if "->" in normalized_arrow_segment:
            _flush_pending_topics()
            raw_parts = [part.strip() for part in normalized_arrow_segment.split("->") if part.strip()]
            row_nodes: List[_PaperNode] = []
            for part in raw_parts[:3]:
                is_doc = part.lower().startswith("doc:")
                raw_node_text = part.split(":", 1)[1].strip() if is_doc else part
                node_label = _compact_diagram_node_label(raw_node_text)
                if node_label:
                    row_nodes.append(_PaperNode(label=node_label, is_document=is_doc))
            if len(row_nodes) == 1:
                row_nodes[0].is_document = True
                row_nodes.append(_PaperNode(label="Outcome", is_document=False))
            if row_nodes:
                parsed_rows.append(_PaperRow(kind="flow", nodes=row_nodes))
            continue

        if ":" in cleaned_segment and not cleaned_segment.lower().startswith("doc:"):
            _flush_pending_topics()
            left_raw, right_raw = cleaned_segment.split(":", 1)
            left_label = _compact_diagram_node_label(left_raw)
            right_label = _compact_diagram_node_label(right_raw)
            parsed_rows.append(
                _PaperRow(
                    kind="flow",
                    nodes=[
                        _PaperNode(label=left_label, is_document=False),
                        _PaperNode(label=right_label, is_document=False),
                    ],
                )
            )
            continue

        is_doc_topic = cleaned_segment.lower().startswith("doc:")
        raw_topic_text = cleaned_segment.split(":", 1)[1].strip() if is_doc_topic else cleaned_segment
        topic_label = _compact_diagram_node_label(raw_topic_text)
        pending_topic_nodes.append(_PaperNode(label=topic_label, is_document=is_doc_topic))
        if len(pending_topic_nodes) >= 2:
            _flush_pending_topics()

    _flush_pending_topics()
    return parsed_rows


def _paint_paper_arrow(
    painter: QPainter,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    color: QColor,
    seed_value: int,
) -> None:
    """Draws a slim hand-sketched horizontal/angled connector arrow inside a paper diagram."""
    if math.hypot(end_x - start_x, end_y - start_y) < 6.0:
        return
    shaft_points = _build_wobbly_line_points(
        start_x, start_y, end_x, end_y, seed_value=seed_value, bow_amplitude=0.75, samples=6
    )
    shaft_path = _build_smooth_catmull_rom_path(shaft_points)

    painter.save()
    pen = QPen(color, 1.35, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(shaft_path)

    angle = math.atan2(end_y - start_y, end_x - start_x)
    head_length = 6.5
    head_spread = 0.48
    wing_one = QPointF(
        end_x - head_length * math.cos(angle - head_spread),
        end_y - head_length * math.sin(angle - head_spread),
    )
    wing_two = QPointF(
        end_x - head_length * math.cos(angle + head_spread),
        end_y - head_length * math.sin(angle + head_spread),
    )
    painter.drawLine(QPointF(end_x, end_y), wing_one)
    painter.drawLine(QPointF(end_x, end_y), wing_two)
    painter.restore()


def _paint_paper_document_icon(
    painter: QPainter,
    center_x: float,
    top_y: float,
    doc_width: float,
    doc_height: float,
    highlighted: bool,
    accent_color: QColor,
    seed_value: int,
) -> None:
    """Renders a miniature hand-sketched paper document icon for `doc:` diagram nodes."""
    left_x = center_x - doc_width / 2.0
    frame_points = _build_wobbly_rect_points(
        left_x, top_y, doc_width, doc_height, corner_radius=2.5, seed_value=seed_value
    )
    frame_path = _build_smooth_catmull_rom_path(frame_points, closed=True)

    fill_color = QColor(39, 39, 46, 240) if highlighted else QColor(26, 26, 32, 228)
    stroke_color = QColor("#E4E4E7") if highlighted else QColor(138, 138, 150, 235)
    line_color = QColor("#D4D4D8") if highlighted else QColor(161, 161, 170, 200)

    painter.save()
    painter.setBrush(fill_color)
    painter.setPen(QPen(stroke_color, 1.7 if highlighted else 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(frame_path)

    # Interior tiny text lines matching reference photo 2.14.00 AM.jpeg
    line_pen = QPen(line_color, 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    painter.setPen(line_pen)
    margin_x = 6.5
    if not highlighted:
        painter.drawLine(QPointF(left_x + margin_x, top_y + 11.0), QPointF(left_x + doc_width - margin_x, top_y + 11.0))
        painter.drawLine(QPointF(left_x + margin_x, top_y + 17.5), QPointF(left_x + doc_width - margin_x - 3.0, top_y + 17.5))
    else:
        painter.drawLine(QPointF(left_x + margin_x, top_y + 12.0), QPointF(left_x + doc_width - margin_x, top_y + 12.0))
        painter.drawLine(QPointF(left_x + margin_x, top_y + 19.0), QPointF(left_x + doc_width - margin_x - 4.0, top_y + 19.0))
    painter.restore()


def _render_paper_card_to_pixmap(
    command: jarvis_response_parser.DrawingCommand,
    seed_value: int,
) -> QPixmap:
    """
    Renders the cool-light gridded paper card (`DRAWING_PAPER_COOL`, `14px` corner
    radius, slate header strip, subtle grid lines, bold haloed `hand_font` title,
    and dynamically fitted flow diagram) into a crisp 2x cached QPixmap.
    """
    card_width = max(200.0, float(command.width))
    card_height = max(180.0, float(command.height))
    shadow_pad = 14
    scale_factor = 2.5

    pixmap_width = int(math.ceil((card_width + shadow_pad * 2) * scale_factor))
    pixmap_height = int(math.ceil((card_height + shadow_pad * 2) * scale_factor))
    pixmap = QPixmap(pixmap_width, pixmap_height)
    pixmap.fill(Qt.GlobalColor.transparent)
    pixmap.setDevicePixelRatio(scale_factor)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.translate(shadow_pad, shadow_pad)

    card_rect = QRectF(0.0, 0.0, card_width, card_height)
    accent_color = _resolve_tone_color(command.tone, jarvis_theme.DRAWING_TONES["blue"])

    # 1. Soft elevation shadow behind the paper card
    for offset_y, expand_px, alpha_value in ((8.0, 4.0, 26), (4.0, 1.5, 42), (1.5, 0.5, 50)):
        shadow_rect = card_rect.adjusted(-expand_px, -expand_px + offset_y, expand_px, expand_px + offset_y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, alpha_value))
        painter.drawRoundedRect(shadow_rect, PAPER_CORNER_RADIUS + expand_px, PAPER_CORNER_RADIUS + expand_px)

    # 2. Clipped cool-light paper body + header strip + graph grid
    clip_path = QPainterPath()
    clip_path.addRoundedRect(card_rect, PAPER_CORNER_RADIUS, PAPER_CORNER_RADIUS)

    painter.save()
    painter.setClipPath(clip_path)
    painter.fillRect(card_rect, jarvis_theme.DRAWING_PAPER_COOL)

    # Top header strip (`DRAWING_PAPER_HEADER`)
    header_rect = QRectF(0.0, 0.0, card_width, PAPER_HEADER_HEIGHT)
    painter.fillRect(header_rect, jarvis_theme.DRAWING_PAPER_HEADER)
    painter.setPen(QPen(QColor(42, 42, 52, 210), 1.0))
    painter.drawLine(QPointF(0.0, PAPER_HEADER_HEIGHT), QPointF(card_width, PAPER_HEADER_HEIGHT))

    header_label = jarvis_response_parser.clean_spoken_text(command.header or "")
    if header_label:
        header_font = jarvis_theme.barlow_font(9.5, semibold=False)
        painter.setFont(header_font)
        painter.setPen(QColor(161, 161, 170, 235))
        painter.drawText(header_rect, Qt.AlignmentFlag.AlignCenter, header_label)

    # Graph-paper grid lines (`DRAWING_GRID_LINE`)
    painter.setPen(QPen(jarvis_theme.DRAWING_GRID_LINE, 1.0))
    grid_x = PAPER_GRID_SPACING
    while grid_x < card_width:
        painter.drawLine(QPointF(grid_x, PAPER_HEADER_HEIGHT), QPointF(grid_x, card_height))
        grid_x += PAPER_GRID_SPACING
    grid_y = PAPER_HEADER_HEIGHT + PAPER_GRID_SPACING
    while grid_y < card_height:
        painter.drawLine(QPointF(0.0, grid_y), QPointF(card_width, grid_y))
        grid_y += PAPER_GRID_SPACING

    # 3. Title (`hand_font(19)` with soft dark outline halo)
    inner_margin_x = 20.0
    inner_width = card_width - inner_margin_x * 2.0
    cursor_y = PAPER_HEADER_HEIGHT + 16.0

    title_text = jarvis_response_parser.clean_spoken_text(command.label or "")
    if title_text:
        title_point_size = 19.0
        title_font = jarvis_theme.hand_font(title_point_size)
        title_font.setWeight(QFont.Weight.Bold)
        title_metrics = QFontMetricsF(title_font)
        while title_point_size > 13.0 and title_metrics.horizontalAdvance(title_text) > inner_width:
            title_point_size -= 1.0
            title_font = jarvis_theme.hand_font(title_point_size)
            title_font.setWeight(QFont.Weight.Bold)
            title_metrics = QFontMetricsF(title_font)

        if title_metrics.horizontalAdvance(title_text) > inner_width:
            title_text = title_metrics.elidedText(title_text, Qt.TextElideMode.ElideRight, inner_width)

        baseline_y = cursor_y + title_metrics.ascent()
        _paint_text_with_halo(
            painter,
            origin_x=inner_margin_x,
            baseline_y=baseline_y,
            text=title_text,
            font=title_font,
            text_color=jarvis_theme.DRAWING_INK,
            halo_color=QColor(18, 18, 22, 232),
            halo_width=5.2,
        )
        cursor_y += title_metrics.height() + 14.0

    # 4. Dynamic flow diagram rows from `command.body`
    parsed_rows = _parse_paper_body_rows(command.body)
    if parsed_rows:
        natural_heights: List[float] = []
        for row in parsed_rows:
            if row.kind == "section":
                natural_heights.append(34.0)
            elif row.kind == "flow":
                has_docs = any(node.is_document for node in row.nodes)
                natural_heights.append(82.0 if has_docs else 64.0)
            elif row.kind == "compare":
                natural_heights.append(58.0)
            elif row.kind == "tree":
                natural_heights.append(84.0)
            elif row.kind == "decision":
                natural_heights.append(78.0)
            elif row.kind == "clue":
                natural_heights.append(36.0)
            else:
                natural_heights.append(34.0)

        available_height = max(48.0, card_height - cursor_y - 14.0)
        total_natural = sum(natural_heights)
        vertical_scale = _clamp_float(available_height / max(1.0, total_natural), 0.62, 1.0)

        for row_index, row in enumerate(parsed_rows):
            row_height = natural_heights[row_index] * vertical_scale
            if cursor_y + row_height * 0.6 > card_height - 8.0:
                break

            if row.kind == "section":
                section_font = jarvis_theme.hand_font(16.0 * max(0.82, vertical_scale))
                section_font.setWeight(QFont.Weight.Bold)
                section_metrics = QFontMetricsF(section_font)
                section_text = section_metrics.elidedText(row.text, Qt.TextElideMode.ElideRight, inner_width)
                _paint_text_with_halo(
                    painter,
                    origin_x=inner_margin_x,
                    baseline_y=cursor_y + section_metrics.ascent() + 2.0,
                    text=section_text,
                    font=section_font,
                    text_color=jarvis_theme.DRAWING_INK,
                    halo_color=QColor(18, 18, 22, 230),
                    halo_width=4.2,
                )
            elif row.kind == "flow":
                _paint_paper_flow_row(
                    painter=painter,
                    nodes=row.nodes,
                    origin_x=inner_margin_x,
                    origin_y=cursor_y,
                    row_width=inner_width,
                    row_height=row_height,
                    accent_color=accent_color,
                    seed_value=seed_value + row_index * 101,
                )
            elif row.kind == "compare":
                _paint_paper_compare_row(
                    painter=painter,
                    nodes=row.nodes,
                    origin_x=inner_margin_x,
                    origin_y=cursor_y,
                    row_width=inner_width,
                    row_height=row_height,
                    accent_color=accent_color,
                    seed_value=seed_value + row_index * 107,
                )
            elif row.kind == "tree":
                _paint_paper_tree_row(
                    painter=painter,
                    parent_label=row.text,
                    child_nodes=row.nodes,
                    origin_x=inner_margin_x,
                    origin_y=cursor_y,
                    row_width=inner_width,
                    row_height=row_height,
                    accent_color=accent_color,
                    seed_value=seed_value + row_index * 113,
                )
            elif row.kind == "decision":
                _paint_paper_decision_row(
                    painter=painter,
                    condition_label=row.text,
                    branch_nodes=row.nodes,
                    origin_x=inner_margin_x,
                    origin_y=cursor_y,
                    row_width=inner_width,
                    row_height=row_height,
                    accent_color=accent_color,
                    seed_value=seed_value + row_index * 127,
                )
            elif row.kind == "clue":
                _paint_paper_clue_row(
                    painter=painter,
                    clue_text=row.text,
                    origin_x=inner_margin_x,
                    origin_y=cursor_y,
                    row_width=inner_width,
                    row_height=row_height,
                    accent_color=accent_color,
                )
            else:
                note_font = jarvis_theme.barlow_font(11.0, semibold=False)
                painter.setFont(note_font)
                painter.setPen(jarvis_theme.DRAWING_INK)
                note_rect = QRectF(inner_margin_x + 4.0, cursor_y, inner_width - 8.0, row_height)
                painter.drawText(
                    note_rect,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                    row.text,
                )

            cursor_y += row_height

    painter.restore()

    # Crisp subtle dark paper rim border
    painter.setPen(QPen(QColor(63, 63, 70, 220), 1.1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(card_rect, PAPER_CORNER_RADIUS, PAPER_CORNER_RADIUS)
    painter.end()
    return pixmap


def _paint_paper_flow_row(
    painter: QPainter,
    nodes: List[_PaperNode],
    origin_x: float,
    origin_y: float,
    row_width: float,
    row_height: float,
    accent_color: QColor,
    seed_value: int,
) -> None:
    """Renders one horizontal box/document flow row inside a `paper` card without overflow."""
    node_count = len(nodes)
    if node_count == 0:
        return

    has_documents = any(node.is_document for node in nodes)
    if has_documents:
        slot_width = row_width / float(node_count)
        doc_width = 30.0
        doc_height = min(38.0, max(26.0, row_height * 0.48))
        doc_top_y = origin_y + 4.0
        caption_font = jarvis_theme.barlow_font(10.0, semibold=False)
        caption_metrics = QFontMetricsF(caption_font)

        centers_x: List[float] = []
        for node_index, node in enumerate(nodes):
            center_x = origin_x + slot_width * (node_index + 0.5)
            centers_x.append(center_x)
            is_target = node_index == node_count - 1 and node_count > 1

            if node.is_document:
                _paint_paper_document_icon(
                    painter=painter,
                    center_x=center_x,
                    top_y=doc_top_y,
                    doc_width=doc_width,
                    doc_height=doc_height,
                    highlighted=is_target,
                    accent_color=accent_color,
                    seed_value=seed_value + node_index * 19,
                )
                label_rect = QRectF(
                    origin_x + slot_width * node_index + 4.0,
                    doc_top_y + doc_height + 5.0,
                    slot_width - 8.0,
                    max(18.0, row_height - doc_height - 10.0),
                )
                painter.save()
                painter.setFont(caption_font)
                painter.setPen(QColor(212, 212, 216, 240))
                elided_label = caption_metrics.elidedText(node.label, Qt.TextElideMode.ElideRight, label_rect.width())
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, elided_label)
                painter.restore()
            else:
                box_width = min(slot_width - 14.0, 118.0)
                box_height = doc_height
                box_x = center_x - box_width / 2.0
                _paint_sketched_node_box(
                    painter=painter,
                    box_rect=QRectF(box_x, doc_top_y, box_width, box_height),
                    label=node.label,
                    highlighted=is_target,
                    accent_color=accent_color,
                    seed_value=seed_value + node_index * 23,
                )

        arrow_y = doc_top_y + doc_height * 0.5
        for arrow_index in range(node_count - 1):
            gap_offset = 22.0 if nodes[arrow_index].is_document else 52.0
            next_gap_offset = 22.0 if nodes[arrow_index + 1].is_document else 52.0
            arrow_start_x = centers_x[arrow_index] + gap_offset
            arrow_end_x = centers_x[arrow_index + 1] - next_gap_offset
            _paint_paper_arrow(
                painter,
                arrow_start_x,
                arrow_y,
                arrow_end_x,
                arrow_y,
                color=QColor(212, 212, 216, 225),
                seed_value=seed_value + arrow_index * 41,
            )
        return

    # Standard box-to-box flow row
    arrow_gap = 38.0 if node_count == 2 else 24.0
    total_gap = arrow_gap * max(0, node_count - 1)
    box_width = max(56.0, (row_width - total_gap) / float(node_count))
    box_height = min(44.0, max(32.0, row_height - 14.0))
    box_top_y = origin_y + (row_height - box_height) * 0.42

    for node_index, node in enumerate(nodes):
        box_x = origin_x + node_index * (box_width + arrow_gap)
        is_target = node_index == node_count - 1 and node_count > 1
        _paint_sketched_node_box(
            painter=painter,
            box_rect=QRectF(box_x, box_top_y, box_width, box_height),
            label=node.label,
            highlighted=is_target,
            accent_color=accent_color,
            seed_value=seed_value + node_index * 29,
        )
        if node_index < node_count - 1:
            arrow_start_x = box_x + box_width + 7.0
            arrow_end_x = box_x + box_width + arrow_gap - 7.0
            arrow_mid_y = box_top_y + box_height * 0.5
            _paint_paper_arrow(
                painter,
                arrow_start_x,
                arrow_mid_y,
                arrow_end_x,
                arrow_mid_y,
                color=QColor(212, 212, 216, 225),
                seed_value=seed_value + node_index * 53,
            )


def _paint_paper_compare_row(
    painter: QPainter,
    nodes: List[_PaperNode],
    origin_x: float,
    origin_y: float,
    row_width: float,
    row_height: float,
    accent_color: QColor,
    seed_value: int,
) -> None:
    """Draws two side-by-side comparison boxes separated by a crisp `VS` badge."""
    if not nodes:
        return
    left_label = nodes[0].label
    right_label = nodes[1].label if len(nodes) > 1 else "Contrast"

    center_gap = 40.0
    box_width = max(56.0, (row_width - center_gap) * 0.5)
    box_height = min(44.0, max(30.0, row_height - 10.0))
    box_top_y = origin_y + (row_height - box_height) * 0.5

    left_rect = QRectF(origin_x, box_top_y, box_width, box_height)
    right_rect = QRectF(origin_x + box_width + center_gap, box_top_y, box_width, box_height)

    _paint_sketched_node_box(
        painter=painter,
        box_rect=left_rect,
        label=left_label,
        highlighted=False,
        accent_color=accent_color,
        seed_value=seed_value + 11,
    )
    _paint_sketched_node_box(
        painter=painter,
        box_rect=right_rect,
        label=right_label,
        highlighted=True,
        accent_color=accent_color,
        seed_value=seed_value + 29,
    )

    pill_width = 28.0
    pill_height = 20.0
    pill_x = origin_x + box_width + (center_gap - pill_width) * 0.5
    pill_y = box_top_y + (box_height - pill_height) * 0.5
    vs_rect = QRectF(pill_x, pill_y, pill_width, pill_height)

    painter.save()
    painter.setBrush(QColor(22, 30, 44, 245))
    painter.setPen(QPen(jarvis_theme.with_alpha(accent_color, 210), 1.2))
    painter.drawRoundedRect(vs_rect, 9.0, 9.0)
    painter.setFont(jarvis_theme.barlow_font(8.5, semibold=True))
    painter.setPen(QColor(235, 244, 255, 245))
    painter.drawText(vs_rect, Qt.AlignmentFlag.AlignCenter, "VS")
    painter.restore()


def _paint_paper_tree_row(
    painter: QPainter,
    parent_label: str,
    child_nodes: List[_PaperNode],
    origin_x: float,
    origin_y: float,
    row_width: float,
    row_height: float,
    accent_color: QColor,
    seed_value: int,
) -> None:
    """Draws a parent concept box branched cleanly to child category boxes below."""
    children = child_nodes[:3] if child_nodes else [_PaperNode(label="Branch A"), _PaperNode(label="Branch B")]
    parent_width = min(176.0, max(92.0, row_width * 0.64))
    parent_height = min(28.0, max(22.0, row_height * 0.34))
    parent_x = origin_x + (row_width - parent_width) * 0.5
    parent_y = origin_y + 2.0
    parent_center_x = parent_x + parent_width * 0.5

    _paint_sketched_node_box(
        painter=painter,
        box_rect=QRectF(parent_x, parent_y, parent_width, parent_height),
        label=parent_label or "Concept",
        highlighted=True,
        accent_color=accent_color,
        seed_value=seed_value + 7,
    )

    branch_y = parent_y + parent_height + max(6.0, row_height * 0.11)
    child_top_y = branch_y + max(6.0, row_height * 0.10)
    child_height = max(24.0, row_height - (child_top_y - origin_y) - 4.0)
    child_count = len(children)
    child_gap = 10.0 if child_count > 1 else 0.0
    child_width = max(50.0, (row_width - child_gap * max(0, child_count - 1)) / float(child_count))

    child_centers_x: List[float] = []
    for child_index in range(child_count):
        child_x = origin_x + child_index * (child_width + child_gap)
        child_centers_x.append(child_x + child_width * 0.5)

    painter.save()
    branch_pen = QPen(QColor(186, 198, 218, 215), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(branch_pen)
    painter.drawLine(QPointF(parent_center_x, parent_y + parent_height), QPointF(parent_center_x, branch_y))
    if len(child_centers_x) >= 2:
        painter.drawLine(QPointF(child_centers_x[0], branch_y), QPointF(child_centers_x[-1], branch_y))
    for drop_x in child_centers_x:
        painter.drawLine(QPointF(drop_x, branch_y), QPointF(drop_x, child_top_y))
    painter.restore()

    for child_index, child_node in enumerate(children):
        child_x = origin_x + child_index * (child_width + child_gap)
        _paint_sketched_node_box(
            painter=painter,
            box_rect=QRectF(child_x, child_top_y, child_width, child_height),
            label=child_node.label,
            highlighted=False,
            accent_color=accent_color,
            seed_value=seed_value + 37 + child_index * 17,
        )


def _paint_paper_decision_row(
    painter: QPainter,
    condition_label: str,
    branch_nodes: List[_PaperNode],
    origin_x: float,
    origin_y: float,
    row_width: float,
    row_height: float,
    accent_color: QColor,
    seed_value: int,
) -> None:
    """Draws a condition header pill branching into YES (emerald) and NO (amber) outcome boxes."""
    cond_width = min(row_width - 12.0, 204.0)
    cond_height = min(26.0, max(20.0, row_height * 0.34))
    cond_x = origin_x + (row_width - cond_width) * 0.5
    cond_y = origin_y + 2.0
    cond_rect = QRectF(cond_x, cond_y, cond_width, cond_height)
    cond_center_x = cond_rect.center().x()

    painter.save()
    painter.setBrush(QColor(22, 34, 54, 242))
    painter.setPen(QPen(jarvis_theme.with_alpha(accent_color, 215), 1.4))
    painter.drawRoundedRect(cond_rect, 11.0, 11.0)
    cond_font = jarvis_theme.barlow_font(9.5, semibold=True)
    cond_metrics = QFontMetricsF(cond_font)
    elided_cond = cond_metrics.elidedText(condition_label or "Condition?", Qt.TextElideMode.ElideRight, cond_width - 14.0)
    painter.setFont(cond_font)
    painter.setPen(QColor(240, 246, 255, 248))
    painter.drawText(cond_rect, Qt.AlignmentFlag.AlignCenter, elided_cond)
    painter.restore()

    outcomes = branch_nodes[:2] if branch_nodes else [_PaperNode(label="Proceed"), _PaperNode(label="Alternative")]
    split_y = cond_y + cond_height + max(6.0, row_height * 0.11)
    out_top_y = split_y + max(6.0, row_height * 0.10)
    out_height = max(24.0, row_height - (out_top_y - origin_y) - 4.0)
    out_count = len(outcomes)
    out_gap = 14.0 if out_count == 2 else 0.0
    out_width = max(58.0, (row_width - out_gap * max(0, out_count - 1)) / float(out_count))

    branch_centers_x = [
        origin_x + idx * (out_width + out_gap) + out_width * 0.5
        for idx in range(out_count)
    ]

    painter.save()
    painter.setPen(QPen(QColor(186, 198, 218, 215), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(QPointF(cond_center_x, cond_y + cond_height), QPointF(cond_center_x, split_y))
    if len(branch_centers_x) == 2:
        painter.drawLine(QPointF(branch_centers_x[0], split_y), QPointF(branch_centers_x[1], split_y))
    for drop_x in branch_centers_x:
        painter.drawLine(QPointF(drop_x, split_y), QPointF(drop_x, out_top_y))
    painter.restore()

    for branch_index, branch_node in enumerate(outcomes):
        box_x = origin_x + branch_index * (out_width + out_gap)
        box_rect = QRectF(box_x, out_top_y, out_width, out_height)
        is_yes = branch_index == 0
        rim_color = QColor("#34D399") if is_yes else QColor("#FB923C")
        fill_tint = QColor(18, 42, 36, 236) if is_yes else QColor(44, 28, 22, 236)
        prefix_tag = "YES: " if (is_yes and out_count == 2) else ("NO: " if out_count == 2 else "")
        display_text = f"{prefix_tag}{branch_node.label}" if prefix_tag else branch_node.label

        outline_points = _build_wobbly_rect_points(
            box_rect.left(),
            box_rect.top(),
            box_rect.width(),
            box_rect.height(),
            corner_radius=4.0,
            seed_value=seed_value + 41 + branch_index * 19,
        )
        outline_path = _build_smooth_catmull_rom_path(outline_points, closed=True)
        painter.save()
        painter.setBrush(fill_tint)
        painter.setPen(QPen(rim_color, 1.55, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(outline_path)
        out_font = jarvis_theme.barlow_font(9.5, semibold=True)
        out_metrics = QFontMetricsF(out_font)
        text_rect = box_rect.adjusted(6.0, 2.0, -6.0, -2.0)
        elided_branch = out_metrics.elidedText(display_text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.setFont(out_font)
        painter.setPen(QColor(244, 248, 255, 245))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, elided_branch)
        painter.restore()


def _paint_paper_clue_row(
    painter: QPainter,
    clue_text: str,
    origin_x: float,
    origin_y: float,
    row_width: float,
    row_height: float,
    accent_color: QColor,
) -> None:
    """Draws a crisp bottom memory-clue or pattern callout strip inside a paper card."""
    strip_height = max(24.0, row_height - 6.0)
    strip_rect = QRectF(origin_x, origin_y + 3.0, row_width, strip_height)

    painter.save()
    painter.setBrush(QColor(22, 32, 50, 232))
    painter.setPen(QPen(jarvis_theme.with_alpha(accent_color, 175), 1.15))
    painter.drawRoundedRect(strip_rect, 7.0, 7.0)

    badge_width = 42.0
    badge_height = min(18.0, strip_height - 6.0)
    badge_rect = QRectF(
        strip_rect.left() + 6.0,
        strip_rect.center().y() - badge_height * 0.5,
        badge_width,
        badge_height,
    )
    painter.setBrush(jarvis_theme.with_alpha(accent_color, 64))
    painter.setPen(QPen(jarvis_theme.with_alpha(accent_color, 215), 1.0))
    painter.drawRoundedRect(badge_rect, 5.0, 5.0)
    painter.setFont(jarvis_theme.barlow_font(8.0, semibold=True))
    painter.setPen(QColor(210, 236, 255, 248))
    painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "CLUE")

    text_rect = QRectF(
        badge_rect.right() + 7.0,
        strip_rect.top(),
        strip_rect.width() - badge_width - 18.0,
        strip_rect.height(),
    )
    clue_font = jarvis_theme.barlow_font(9.5, semibold=True)
    clue_metrics = QFontMetricsF(clue_font)
    elided_clue = clue_metrics.elidedText((clue_text or "").strip(), Qt.TextElideMode.ElideRight, text_rect.width())
    painter.setFont(clue_font)
    painter.setPen(jarvis_theme.DRAWING_INK)
    painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided_clue)
    painter.restore()


def _paint_sketched_node_box(
    painter: QPainter,
    box_rect: QRectF,
    label: str,
    highlighted: bool,
    accent_color: QColor,
    seed_value: int,
) -> None:
    """Draws one slightly wobbly hand-sketched diagram box with adaptive text sizing."""
    outline_points = _build_wobbly_rect_points(
        box_rect.left(),
        box_rect.top(),
        box_rect.width(),
        box_rect.height(),
        corner_radius=3.5,
        seed_value=seed_value,
    )
    outline_path = _build_smooth_catmull_rom_path(outline_points, closed=True)

    fill_color = QColor(26, 40, 64, 244) if highlighted else QColor(24, 24, 30, 232)
    border_color = _blend_colors(accent_color, QColor("#E4E4E7"), 0.35) if highlighted else QColor(82, 82, 91, 235)
    border_width = 1.9 if highlighted else 1.4

    painter.save()
    painter.setBrush(fill_color)
    painter.setPen(QPen(border_color, border_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(outline_path)

    text_area = box_rect.adjusted(7.0, 4.0, -7.0, -4.0)
    font_size = 11.5
    label_font = jarvis_theme.barlow_font(font_size, semibold=highlighted)
    metrics = QFontMetricsF(label_font)
    while font_size > 8.5 and metrics.horizontalAdvance(label) > text_area.width() * 1.02:
        font_size -= 0.75
        label_font = jarvis_theme.barlow_font(font_size, semibold=highlighted)
        metrics = QFontMetricsF(label_font)

    painter.setFont(label_font)
    painter.setPen(jarvis_theme.DRAWING_INK)
    if metrics.horizontalAdvance(label) <= text_area.width() + 2.0:
        painter.drawText(text_area, Qt.AlignmentFlag.AlignCenter, label)
    else:
        painter.drawText(text_area, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, label)
    painter.restore()


# ---------------------------------------------------------------------------
# Floating clickable [ x Clear ] pill window
# ---------------------------------------------------------------------------


class ClearOverlayButtonWindow(QWidget):
    """
    Small floating always-on-top non-activating `[ x Clear ]` pill anchored at
    `(screen_width - 220, 96)`. Not click-through so clicking it clears all
    active drawings immediately without stealing foreground focus.
    """

    clearClicked = Signal()

    def __init__(
        self,
        settings_store: Optional[jarvis_config.SettingsStore] = None,
        screen_geometry: Optional[QRect] = None,
    ) -> None:
        super().__init__(None)
        self._settings_store = settings_store
        self._hovered = False
        self._pressed = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        total_width = int(CLEAR_BUTTON_PILL_WIDTH) + CLEAR_BUTTON_SHADOW_MARGIN * 2
        total_height = int(CLEAR_BUTTON_PILL_HEIGHT) + CLEAR_BUTTON_SHADOW_MARGIN * 2
        self.setFixedSize(total_width, total_height)

        resolved_geometry = screen_geometry or self._resolve_primary_geometry()
        self.reposition_for_screen(resolved_geometry)

    @staticmethod
    def _resolve_primary_geometry() -> QRect:
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            return screen.geometry()
        return QRect(0, 0, 1920, 1080)

    def reposition_for_screen(self, screen_geometry: QRect) -> None:
        anchor_x = screen_geometry.x() + max(24, screen_geometry.width() - CLEAR_BUTTON_OFFSET_FROM_RIGHT)
        anchor_y = screen_geometry.y() + CLEAR_BUTTON_TOP_Y
        self.move(anchor_x - CLEAR_BUTTON_SHADOW_MARGIN, anchor_y - CLEAR_BUTTON_SHADOW_MARGIN)

    def apply_capture_policy(self, excluded_from_capture: bool) -> None:
        try:
            window_handle = int(self.winId())
            jarvis_theme.set_window_excluded_from_capture(window_handle, excluded_from_capture)
        except Exception:
            pass

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        try:
            window_handle = int(self.winId())
            jarvis_theme.set_window_no_activate(window_handle, True)
            jarvis_theme.set_window_click_through(window_handle, False)
            excluded = False
            if self._settings_store is not None:
                excluded = not bool(self._settings_store.get("show_in_screen_recordings", True))
            jarvis_theme.set_window_excluded_from_capture(window_handle, excluded)
        except Exception:
            pass

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = False
        self._pressed = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        was_pressed = self._pressed
        self._pressed = False
        self.update()
        if event.button() == Qt.MouseButton.LeftButton and was_pressed and self.rect().contains(event.pos()):
            self.clearClicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        pill_rect = QRectF(
            float(CLEAR_BUTTON_SHADOW_MARGIN),
            float(CLEAR_BUTTON_SHADOW_MARGIN),
            CLEAR_BUTTON_PILL_WIDTH,
            CLEAR_BUTTON_PILL_HEIGHT,
        )
        _paint_clear_button_pill(painter, pill_rect, hovered=self._hovered, pressed=self._pressed)


# ---------------------------------------------------------------------------
# Full-screen click-through companion overlay canvas
# ---------------------------------------------------------------------------


class OverlayCanvasWindow(QWidget):
    """
    Full-screen transparent click-through overlay hosting:
    - Companion cursor (`-35 deg` triangle, 5 cyan listening bars, 3 thinking dots,
      quadratic Bezier pointing/pen riding, shake-to-detach, `hide_floating_cursor`)
    - Crisp word-by-word cursor message bubble
    - Draw-to-Ask glowing cyan-blue Catmull-Rom highlight stroke
    - 2D screen drawings (`underline`, `ring`, `point`, `box`, `highlight`, `paper`)
    - Bottom `LiveCaption` bar with word-level karaoke contrast
    - Smart 60 Hz QTimer that pauses completely when idle
    """

    clearRequested = Signal()
    interactiveLassoCompleted = Signal(object)
    strokeAnimationStarted = Signal(str, int)
    strokeAnimationFinished = Signal()

    def __init__(self, settings_store: jarvis_config.SettingsStore) -> None:
        super().__init__(None)
        self._settings_store = settings_store
        jarvis_theme.register_bundled_fonts()

        screen = QGuiApplication.primaryScreen()
        self._screen_geometry = screen.geometry() if screen is not None else QRect(0, 0, 1920, 1080)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setGeometry(self._screen_geometry)

        # Settings cache
        self._cursor_color = QColor(str(self._settings_store.get("cursor_color_hex", jarvis_theme.JARVIS_BLUE_HEX)))
        if not self._cursor_color.isValid():
            self._cursor_color = QColor(jarvis_theme.JARVIS_BLUE_HEX)
        self._hide_floating_cursor = bool(self._settings_store.get("hide_floating_cursor", False))
        self._shake_to_detach_enabled = bool(self._settings_store.get("shake_to_detach_enabled", False))
        self._screen_highlight_enabled = bool(self._settings_store.get("screen_highlight_enabled", True))
        self._captions_enabled = bool(self._settings_store.get("captions_enabled", False))
        self._show_in_screen_recordings = bool(self._settings_store.get("show_in_screen_recordings", True))
        self._drawing_mode = str(
            self._settings_store.get(
                "screen_drawing_mode",
                jarvis_config.DEFAULT_SETTINGS["screen_drawing_mode"],
            )
        )
        self._drawing_speed = str(
            self._settings_store.get(
                "drawing_speed",
                jarvis_config.DEFAULT_SETTINGS["drawing_speed"],
            )
        )

        now_seconds = time.monotonic()
        self._last_tick_time = now_seconds
        self._visual_clock_offset = 0.0

        # Cursor motion & morph state
        initial_mouse_x, initial_mouse_y = self._sample_local_mouse_position()
        self._last_mouse_position = (initial_mouse_x, initial_mouse_y)
        start_x = initial_mouse_x + jarvis_theme.CURSOR_FOLLOW_OFFSET.x()
        start_y = initial_mouse_y + jarvis_theme.CURSOR_FOLLOW_OFFSET.y()
        self._cursor_x_spring = jarvis_theme.Spring(
            start_x, response=CURSOR_SPRING_RESPONSE_SECONDS, damping_ratio=CURSOR_SPRING_DAMPING_RATIO
        )
        self._cursor_y_spring = jarvis_theme.Spring(
            start_y, response=CURSOR_SPRING_RESPONSE_SECONDS, damping_ratio=CURSOR_SPRING_DAMPING_RATIO
        )

        initial_opacity = 0.0 if self._hide_floating_cursor else 1.0
        self._cursor_opacity_spring = jarvis_theme.Spring(
            initial_opacity, response=CURSOR_OPACITY_SPRING_RESPONSE_SECONDS, damping_ratio=1.0
        )
        self._pending_hide_fade_at: Optional[float] = None

        self._is_listening = False
        self._is_thinking = False
        self._is_speaking = False
        self._audio_level = 0.0
        self._smoothed_audio_level = 0.0
        self._listening_morph_spring = jarvis_theme.Spring(0.0, response=CURSOR_MORPH_SPRING_RESPONSE_SECONDS, damping_ratio=1.0)
        self._thinking_morph_spring = jarvis_theme.Spring(0.0, response=CURSOR_MORPH_SPRING_RESPONSE_SECONDS, damping_ratio=1.0)

        # Quadratic Bezier cursor flight & pen riding state
        self._navigation_mode = "following"  # "following" | "flying" | "pointing" | "parked"
        self._flight_start = QPointF(start_x, start_y)
        self._flight_control = QPointF(start_x, start_y)
        self._flight_target = QPointF(start_x, start_y)
        self._flight_start_time = 0.0
        self._flight_duration = 0.35
        self._target_hold_until = 0.0
        self._target_lock_mouse_anchor: Optional[Tuple[float, float]] = None

        # Shake-to-detach detector
        self._shake_detector = _ShakeToDetachDetector()
        self._parked_until = 0.0

        # Cursor message bubble state
        self._bubble_words: List[str] = []
        self._bubble_visible_count = 0
        self._bubble_next_word_time = 0.0
        self._bubble_word_by_word = False
        self._bubble_hold_seconds = 4.0
        self._bubble_fade_start = 0.0
        self._bubble_fade_end = 0.0
        self._bubble_opacity = 0.0

        # Draw-to-Ask & interactive lasso highlight state
        self._is_highlighting = False
        self._interactive_lasso_active = False
        self._interactive_lasso_dragging = False
        self._highlight_points: List[Tuple[float, float]] = []
        self._highlight_cached_path: Optional[QPainterPath] = None
        self._highlight_fade_start = 0.0
        self._highlight_fade_end = 0.0

        # Screen drawings state
        self._drawings: List[_ActiveDrawingItem] = []
        self._next_drawing_seed = 1001
        self._next_stroke_schedule_time = 0.0
        self._was_actively_drawing = False

        # Bottom LiveCaption state & sprite cache
        self._caption_text = ""
        self._caption_words: List[str] = []
        self._caption_highlight_word_index = -1
        self._caption_force_visible = False
        self._caption_expire_at = 0.0
        self._cached_caption_key: Optional[Tuple[Any, ...]] = None
        self._cached_caption_pixmap: Optional[QPixmap] = None
        self._cached_caption_origin: Tuple[float, float] = (0.0, 0.0)

        # Pre-built cursor triangle path
        self._cursor_triangle_path = jarvis_theme.build_rounded_triangle_path(
            side_length=jarvis_theme.CURSOR_TRIANGLE_SIDE,
            corner_fraction=jarvis_theme.CURSOR_TRIANGLE_CORNER_FRACTION,
            rotation_degrees=jarvis_theme.CURSOR_RESTING_ROTATION_DEGREES,
        )

        # Floating non-activating clickable Clear pill window
        self.clear_button_window = ClearOverlayButtonWindow(
            settings_store=self._settings_store,
            screen_geometry=self._screen_geometry,
        )
        self.clear_button_window.clearClicked.connect(self._on_clear_button_clicked)

        # Smart 60 Hz frame timer
        self._frame_timer = QTimer(self)
        self._frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._frame_timer.setInterval(FRAME_INTERVAL_MILLISECONDS)
        self._frame_timer.timeout.connect(self._on_animation_tick)

        try:
            self._settings_store.settingChanged.connect(self.on_setting_changed)
        except Exception:
            pass

        if not self._hide_floating_cursor:
            self._ensure_timer_running()

    # ------------------------------------------------------------------
    # Qt lifecycle & Win32 window configuration
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_win32_styles()
        self._sync_clear_button_visibility()
        self._ensure_timer_running()

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self.clear_button_window.hide()
        self._frame_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._frame_timer.stop()
        self.clear_button_window.close()
        super().closeEvent(event)

    def _apply_win32_styles(self) -> None:
        try:
            window_handle = int(self.winId())
            jarvis_theme.set_window_no_activate(window_handle, True)
            jarvis_theme.set_window_click_through(window_handle, not self._interactive_lasso_active)
            jarvis_theme.set_window_excluded_from_capture(window_handle, not self._show_in_screen_recordings)
        except Exception:
            pass

    def start_interactive_lasso_mode(self) -> None:
        """
        Temporarily intercepts mouse drag anywhere on screen so the user can draw a smooth
        neon lasso without holding modifier keys, then automatically restores click-through.
        """
        self._interactive_lasso_active = True
        self._interactive_lasso_dragging = False
        self._is_highlighting = True
        self._highlight_points.clear()
        self._highlight_cached_path = None
        self._highlight_fade_start = 0.0
        self._highlight_fade_end = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._apply_win32_styles()
        self.show()
        self.raise_()
        self.set_bubble_message(
            "Drag a lasso around anything on screen (Esc to cancel)",
            word_by_word=False,
            hold_seconds=6.0,
        )
        self._ensure_timer_running()
        self.update()

    def cancel_interactive_lasso_mode(self) -> None:
        """Restores normal click-through behavior cleanly if lasso mode is cancelled."""
        if not self._interactive_lasso_active:
            return
        self._interactive_lasso_active = False
        self._interactive_lasso_dragging = False
        self._is_highlighting = False
        self._highlight_points.clear()
        self._highlight_cached_path = None
        self.unsetCursor()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._apply_win32_styles()
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._interactive_lasso_active and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._interactive_lasso_dragging = True
            self._is_highlighting = True
            self._highlight_points = [(float(pos.x()), float(pos.y()))]
            self._highlight_cached_path = None
            self.set_bubble_message("")
            self._ensure_timer_running()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._interactive_lasso_active and self._interactive_lasso_dragging:
            pos = event.position()
            self._append_smoothed_highlight_point(float(pos.x()), float(pos.y()))
            self._ensure_timer_running()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._interactive_lasso_active and self._interactive_lasso_dragging and event.button() == Qt.MouseButton.LeftButton:
            self._interactive_lasso_dragging = False
            self._interactive_lasso_active = False
            self.unsetCursor()
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self._apply_win32_styles()
            captured_points = self.end_highlight_stroke()
            if len(captured_points) >= 4:
                self.interactiveLassoCompleted.emit(captured_points)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _append_smoothed_highlight_point(self, raw_x: float, raw_y: float) -> bool:
        """Applies exponential moving average smoothing before appending a highlight point."""
        if not self._highlight_points:
            self._highlight_points.append((raw_x, raw_y))
            self._highlight_cached_path = None
            return True
        prev_x, prev_y = self._highlight_points[-1]
        if math.hypot(raw_x - prev_x, raw_y - prev_y) < HIGHLIGHT_SAMPLE_MIN_DISTANCE:
            return False
        smoothed_x = prev_x * 0.42 + raw_x * 0.58
        smoothed_y = prev_y * 0.42 + raw_y * 0.58
        self._highlight_points.append((smoothed_x, smoothed_y))
        self._highlight_cached_path = None
        return True

    def _sample_local_mouse_position(self) -> Tuple[float, float]:
        global_pos = QCursor.pos()
        local_x = float(global_pos.x() - self._screen_geometry.x())
        local_y = float(global_pos.y() - self._screen_geometry.y())
        clamped_x = _clamp_float(local_x, 0.0, float(max(1, self.width())))
        clamped_y = _clamp_float(local_y, 0.0, float(max(1, self.height())))
        return (clamped_x, clamped_y)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_cursor_visual_state(
        self,
        listening: bool = False,
        thinking: bool = False,
        speaking: bool = False,
    ) -> None:
        self._is_listening = bool(listening)
        self._is_thinking = bool(thinking) and not self._is_listening
        self._is_speaking = bool(speaking)
        self._listening_morph_spring.set_target(1.0 if self._is_listening else 0.0)
        self._thinking_morph_spring.set_target(1.0 if self._is_thinking else 0.0)
        self._ensure_timer_running()
        self.update()

    def set_audio_level(self, level: float) -> None:
        self._audio_level = _clamp_float(float(level), 0.0, 1.0)
        if self._is_listening:
            self._ensure_timer_running()

    def set_bubble_message(
        self,
        text: str,
        word_by_word: bool = False,
        hold_seconds: float = 4.0,
    ) -> None:
        cleaned = jarvis_response_parser.clean_spoken_text(text)
        if not cleaned:
            self._clear_bubble_state()
            self.update()
            return

        now_seconds = time.monotonic()
        self._bubble_words = cleaned.split()
        self._bubble_word_by_word = bool(word_by_word)
        self._bubble_hold_seconds = max(0.4, float(hold_seconds))
        self._bubble_opacity = 1.0

        if self._bubble_word_by_word and len(self._bubble_words) > 1:
            self._bubble_visible_count = 1
            self._bubble_next_word_time = now_seconds + self._word_reveal_delay(self._bubble_words[0])
            self._bubble_fade_start = now_seconds + self._estimate_reveal_duration(self._bubble_words) + self._bubble_hold_seconds
        else:
            self._bubble_visible_count = len(self._bubble_words)
            self._bubble_next_word_time = 0.0
            self._bubble_fade_start = now_seconds + self._bubble_hold_seconds
        self._bubble_fade_end = self._bubble_fade_start + 0.28

        self._ensure_timer_running()
        self.update()

    def set_karaoke_progress(self, sentence_text: str, word_index: int, total_words: int) -> None:
        cleaned = jarvis_response_parser.clean_spoken_text(sentence_text)
        if not cleaned:
            return

        words = cleaned.split()
        if not words:
            return

        clamped_index = max(0, min(int(word_index), len(words) - 1))
        if self._captions_enabled or self._caption_force_visible:
            self._caption_text = cleaned
            self._caption_words = words
            self._caption_highlight_word_index = clamped_index
            self._caption_expire_at = time.monotonic() + 6.0

        if self._bubble_words and self._bubble_word_by_word:
            ratio = (clamped_index + 1) / float(max(1, total_words, len(words)))
            target_count = max(1, min(len(self._bubble_words), math.ceil(ratio * len(self._bubble_words))))
            self._bubble_visible_count = max(self._bubble_visible_count, target_count)
            if self._bubble_visible_count >= len(self._bubble_words):
                now_seconds = time.monotonic()
                self._bubble_fade_start = max(self._bubble_fade_start, now_seconds + self._bubble_hold_seconds)
                self._bubble_fade_end = self._bubble_fade_start + 0.28

        self._ensure_timer_running()
        self.update()

    def set_caption_text(
        self,
        text: str,
        highlight_word_index: int = -1,
        force_visible: bool = False,
    ) -> None:
        cleaned = jarvis_response_parser.clean_spoken_text(text)
        if not cleaned:
            self.clear_caption()
            return

        self._caption_text = cleaned
        self._caption_words = cleaned.split()
        self._caption_highlight_word_index = int(highlight_word_index)
        self._caption_force_visible = bool(force_visible)
        self._caption_expire_at = time.monotonic() + 12.0
        self._ensure_timer_running()
        self.update()

    def clear_caption(self) -> None:
        self._caption_text = ""
        self._caption_words = []
        self._caption_highlight_word_index = -1
        self._caption_force_visible = False
        self._caption_expire_at = 0.0
        self.update()

    def begin_highlight_stroke(self) -> None:
        self._is_highlighting = True
        self._highlight_points.clear()
        self._highlight_cached_path = None
        self._highlight_fade_start = 0.0
        self._highlight_fade_end = 0.0
        if self._screen_highlight_enabled:
            mouse_x, mouse_y = self._sample_local_mouse_position()
            self._highlight_points.append((mouse_x, mouse_y))
        self._ensure_timer_running()

    def end_highlight_stroke(self) -> List[Tuple[float, float]]:
        self._is_highlighting = False
        captured_points = list(self._highlight_points)
        has_genuine_motion = False
        if len(captured_points) >= 6:
            x_coords = [float(point_item[0]) for point_item in captured_points]
            y_coords = [float(point_item[1]) for point_item in captured_points]
            bbox_span_x = max(x_coords) - min(x_coords)
            bbox_span_y = max(y_coords) - min(y_coords)
            total_path_length = 0.0
            for point_idx in range(1, len(captured_points)):
                prev_pt = captured_points[point_idx - 1]
                curr_pt = captured_points[point_idx]
                total_path_length += math.hypot(
                    float(curr_pt[0]) - float(prev_pt[0]),
                    float(curr_pt[1]) - float(prev_pt[1]),
                )
            has_genuine_motion = (total_path_length >= 65.0) and (max(bbox_span_x, bbox_span_y) >= 36.0)

        if has_genuine_motion:
            now_seconds = time.monotonic()
            self._highlight_fade_start = now_seconds
            self._highlight_fade_end = now_seconds + HIGHLIGHT_FADE_DURATION_SECONDS
            self._ensure_timer_running()
            return captured_points

        self._highlight_points.clear()
        self._highlight_cached_path = None
        self._highlight_fade_start = 0.0
        self._highlight_fade_end = 0.0
        self.update()
        return []

    def _normalize_command_for_drawing_mode(
        self,
        command: jarvis_response_parser.DrawingCommand,
    ) -> Optional[jarvis_response_parser.DrawingCommand]:
        """Normalizes incoming commands for clean, precise line explanations and blocks oversized boxes."""
        drawing_mode = (self._drawing_mode or "precise_lines").strip().lower()
        screen_width = float(max(800, self.width()))
        screen_height = float(max(600, self.height()))

        if drawing_mode == "cursor_only":
            if command.kind != "point":
                return None
            return jarvis_response_parser.DrawingCommand(
                kind="point",
                x=command.x,
                y=command.y,
                label="",
                tone=command.tone,
            )

        if command.kind in ("box", "highlight") and (
            command.width > screen_width * 0.65 or command.height > screen_height * 0.35
        ):
            return None

        if command.kind == "paper":
            return command
        if command.kind == "point":
            cleaned_label = (command.label or "").strip()
            if cleaned_label:
                converted = jarvis_response_parser.DrawingCommand(
                    kind="underline",
                    x=max(24.0, command.x - 90.0),
                    y=command.y,
                    width=220.0,
                    label=cleaned_label,
                    tone=command.tone,
                    sentence_index=command.sentence_index,
                )
                try:
                    import jarvis_services

                    converted = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(converted)
                except Exception:
                    pass
                return converted
            return jarvis_response_parser.DrawingCommand(
                kind="point",
                x=command.x,
                y=command.y,
                label="",
                tone=command.tone,
                sentence_index=command.sentence_index,
            )
        if command.kind in ("box", "highlight"):
            baseline_y = command.y + min(command.height, 28.0) + 2.0
            converted = jarvis_response_parser.DrawingCommand(
                kind="underline",
                x=command.x,
                y=baseline_y,
                width=max(36.0, min(command.width, 460.0)),
                label=(command.label or "").strip(),
                tone=command.tone,
                sentence_index=command.sentence_index,
            )
            try:
                import jarvis_services

                converted = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(converted)
            except Exception:
                pass
            return converted
        if command.kind == "ring":
            span_radius = max(24.0, min(command.radius, 120.0))
            converted = jarvis_response_parser.DrawingCommand(
                kind="underline",
                x=max(12.0, command.x - span_radius),
                y=command.y + span_radius * 0.55,
                width=max(48.0, min(span_radius * 2.0, 260.0)),
                label=command.label,
                tone=command.tone,
                sentence_index=command.sentence_index,
            )
            try:
                import jarvis_services

                converted = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(converted)
            except Exception:
                pass
            return converted
        if command.kind == "underline":
            return jarvis_response_parser.DrawingCommand(
                kind="underline",
                x=command.x,
                y=command.y,
                width=command.width,
                label=command.label,
                tone=command.tone,
                sentence_index=command.sentence_index,
            )

        return command

    def _compute_stroke_duration_seconds(
        self,
        command: jarvis_response_parser.DrawingCommand,
    ) -> float:
        if command.kind == "paper":
            return 0.42
        is_slow = (self._drawing_speed or "slow_smooth").strip().lower() == "slow_smooth"
        if command.kind in ("underline", "highlight"):
            if is_slow:
                return _clamp_float(0.72 + command.width / 1200.0, 0.75, 1.15)
            return _clamp_float(0.36 + command.width / 2200.0, 0.38, 0.58)
        return 0.80 if is_slow else 0.42

    def _active_stroke_tip_point(
        self,
        command: jarvis_response_parser.DrawingCommand,
        eased_progress: float,
    ) -> Optional[Tuple[float, float]]:
        if command.kind == "underline":
            tip_x = command.x + max(12.0, command.width) * eased_progress
            return (tip_x, command.y)
        if command.kind == "highlight":
            tip_x = command.x + max(12.0, command.width) * eased_progress
            return (tip_x, command.y + command.height * 0.5)
        if command.kind == "point":
            return (command.x, command.y)
        return None

    def add_drawing_command(self, command: jarvis_response_parser.DrawingCommand) -> None:
        normalized_command = self._normalize_command_for_drawing_mode(command)
        if normalized_command is None:
            return

        now_seconds = time.monotonic()
        self._next_drawing_seed += 37
        prepared_command = self._clamp_drawing_command(normalized_command)

        stroke_duration = self._compute_stroke_duration_seconds(prepared_command)
        if prepared_command.kind == "paper":
            delay_seconds = 0.0
        else:
            leading_edge = self._leading_edge_for_command(prepared_command)
            initial_flight_lead = 0.0
            if self._next_stroke_schedule_time <= now_seconds and leading_edge is not None:
                flight_dist = math.hypot(
                    leading_edge[0] - self._cursor_x_spring.value,
                    leading_edge[1] - self._cursor_y_spring.value,
                )
                if flight_dist > 110.0:
                    initial_flight_lead = 0.06
            elif now_seconds >= (self._next_stroke_schedule_time - 0.14):
                # Start voice-synced underlines crisply when the previous stroke has nearly finished
                self._next_stroke_schedule_time = now_seconds
            scheduled_start = max(now_seconds + initial_flight_lead, self._next_stroke_schedule_time)
            delay_seconds = max(0.0, scheduled_start - now_seconds)
            self._next_stroke_schedule_time = scheduled_start + stroke_duration + 0.08

        drawing_item = _ActiveDrawingItem(
            command=prepared_command,
            created_at=now_seconds,
            delay_seconds=delay_seconds,
            duration_seconds=stroke_duration,
            expires_at=now_seconds + delay_seconds + stroke_duration + DEFAULT_DRAWING_HOLD_SECONDS,
            seed_value=self._next_drawing_seed,
        )
        self._prebuild_drawing_cache(drawing_item)
        self._drawings.append(drawing_item)

        # Fly companion cursor smoothly to the first active marking/pointing primitive;
        # subsequent queued strokes glide smoothly along their active underline tip in _step_state.
        if prepared_command.kind != "paper" and delay_seconds <= 0.22:
            target_position = self._leading_edge_for_command(prepared_command)
            if target_position is not None:
                self.fly_cursor_to(
                    target_position[0],
                    target_position[1],
                    label="",
                    hold_seconds=max(4.0, delay_seconds + stroke_duration + 2.4),
                )

        self._sync_clear_button_visibility()
        self._ensure_timer_running()
        self.update()

    def fly_cursor_to(
        self,
        anchor_x: float,
        anchor_y: float,
        label: str = "",
        hold_seconds: float = 4.5,
    ) -> None:
        now_seconds = time.monotonic()
        screen_width = float(max(200, self.width()))
        screen_height = float(max(200, self.height()))

        # Offset slightly bottom-right so the -35 deg top-left apex points cleanly at (anchor_x, anchor_y)
        target_x = _clamp_float(float(anchor_x) + 8.0, 20.0, screen_width - 20.0)
        target_y = _clamp_float(float(anchor_y) + 10.0, 20.0, screen_height - 20.0)

        current_x = self._cursor_x_spring.value
        current_y = self._cursor_y_spring.value
        delta_x = target_x - current_x
        delta_y = target_y - current_y
        distance = math.hypot(delta_x, delta_y)

        # Quadratic Bezier arc perpendicular lift
        mid_x = (current_x + target_x) * 0.5
        mid_y = (current_y + target_y) * 0.5
        normal_x = -delta_y / (distance or 1.0)
        normal_y = delta_x / (distance or 1.0)
        arc_lift = min(75.0, distance * 0.18) * (-1.0 if normal_y > 0 else 1.0)

        self._flight_start = QPointF(current_x, current_y)
        self._flight_control = QPointF(mid_x + normal_x * arc_lift, mid_y + normal_y * arc_lift)
        self._flight_target = QPointF(target_x, target_y)
        self._flight_start_time = now_seconds
        self._flight_duration = _clamp_float(distance / 1700.0, 0.18, 0.44)
        self._target_hold_until = now_seconds + self._flight_duration + max(0.5, float(hold_seconds))
        self._target_lock_mouse_anchor = None
        self._navigation_mode = "flying"

        if label and not self._bubble_words:
            self.set_bubble_message(label, word_by_word=False, hold_seconds=hold_seconds)

        self._ensure_timer_running()

    def load_demo_showcase(self) -> None:
        self.clear_all_drawings()
        screen_width = float(max(800, self.width()))
        screen_height = float(max(600, self.height()))
        scale_x = screen_width / 1536.0
        scale_y = screen_height / 864.0

        now_seconds = time.monotonic()
        demo_commands: List[Tuple[jarvis_response_parser.DrawingCommand, float]] = [
            (
                jarvis_response_parser.DrawingCommand(
                    kind="paper",
                    x=26.0,
                    y=max(110.0, screen_height * 0.34),
                    width=min(340.0, screen_width * 0.25),
                    height=min(358.0, screen_height * 0.44),
                    label="Advice and records",
                    header="two powers",
                    body="President -> Court | section:Simple examples | doc:Past rulings -> doc:Later cases",
                    tone="blue",
                ),
                0.0,
            ),
            (
                jarvis_response_parser.DrawingCommand(
                    kind="underline",
                    x=435.0 * scale_x,
                    y=210.0 * scale_y,
                    width=360.0 * scale_x,
                    label="",
                    tone="blue",
                ),
                0.08,
            ),
            (
                jarvis_response_parser.DrawingCommand(
                    kind="underline",
                    x=435.0 * scale_x,
                    y=370.0 * scale_y,
                    width=355.0 * scale_x,
                    label="",
                    tone="blue",
                ),
                0.96,
            ),
            (
                jarvis_response_parser.DrawingCommand(
                    kind="underline",
                    x=455.0 * scale_x,
                    y=414.0 * scale_y,
                    width=455.0 * scale_x,
                    label="",
                    tone="blue",
                ),
                1.86,
            ),
            (
                jarvis_response_parser.DrawingCommand(
                    kind="underline",
                    x=455.0 * scale_x,
                    y=502.0 * scale_y,
                    width=435.0 * scale_x,
                    label="",
                    tone="blue",
                ),
                2.80,
            ),
            (
                jarvis_response_parser.DrawingCommand(
                    kind="point",
                    x=892.0 * scale_x,
                    y=502.0 * scale_y,
                    label="",
                    tone="blue",
                ),
                3.65,
            ),
        ]

        for command, delay in demo_commands:
            self._next_drawing_seed += 41
            clamped = self._clamp_drawing_command(command)
            stroke_duration = self._compute_stroke_duration_seconds(clamped)
            item = _ActiveDrawingItem(
                command=clamped,
                created_at=now_seconds,
                delay_seconds=delay,
                duration_seconds=stroke_duration,
                expires_at=now_seconds + delay + stroke_duration + DEFAULT_DRAWING_HOLD_SECONDS,
                seed_value=self._next_drawing_seed,
            )
            self._prebuild_drawing_cache(item)
            self._drawings.append(item)

        self.fly_cursor_to(892.0 * scale_x, 502.0 * scale_y, label="", hold_seconds=5.0)
        self.set_caption_text(
            "Under Article 143 the President can consult the Supreme Court, and past rulings act as binding records.",
            highlight_word_index=7,
            force_visible=True,
        )
        self._sync_clear_button_visibility()
        self._ensure_timer_running()
        self.update()

    def clear_all_drawings(self) -> None:
        self._drawings.clear()
        self._next_stroke_schedule_time = 0.0
        self._was_actively_drawing = False
        self._is_highlighting = False
        self._highlight_points.clear()
        self._highlight_cached_path = None
        self._highlight_fade_start = 0.0
        self._highlight_fade_end = 0.0
        if self._navigation_mode in ("flying", "pointing"):
            self._navigation_mode = "following"
            self._target_lock_mouse_anchor = None
        self._sync_clear_button_visibility()
        self.update()

    def clear_viewport_markings_keep_paper(self) -> None:
        """
        Clears viewport-anchored highlights/rings/underlines before scrolling while preserving
        margin paper cards so the summary board stays visible across scroll sections.
        """
        self._drawings = [item for item in self._drawings if item.command.kind == "paper"]
        self._next_stroke_schedule_time = 0.0
        self._was_actively_drawing = False
        self._is_highlighting = False
        self._highlight_points.clear()
        self._highlight_cached_path = None
        self._sync_clear_button_visibility()
        self.update()

    def has_active_drawings(self) -> bool:
        return bool(self._drawings)

    def on_setting_changed(self, key_name: str, value: Any) -> None:
        if key_name == "cursor_color_hex":
            candidate = QColor(str(value or jarvis_theme.JARVIS_BLUE_HEX))
            if candidate.isValid():
                self._cursor_color = candidate
                self.update()
            return

        if key_name == "hide_floating_cursor":
            new_hidden = bool(value)
            if new_hidden == self._hide_floating_cursor:
                return
            self._hide_floating_cursor = new_hidden
            now_seconds = time.monotonic()
            if new_hidden:
                self._pending_hide_fade_at = now_seconds + CURSOR_HIDE_FADE_DELAY_SECONDS
                self._cursor_opacity_spring.set_target(1.0)
                self.set_bubble_message(CURSOR_HIDE_MESSAGE, word_by_word=False, hold_seconds=CURSOR_HIDE_HOLD_SECONDS)
            else:
                self._pending_hide_fade_at = None
                mouse_x, mouse_y = self._sample_local_mouse_position()
                self._last_mouse_position = (mouse_x, mouse_y)
                self._cursor_x_spring.snap(mouse_x + jarvis_theme.CURSOR_FOLLOW_OFFSET.x())
                self._cursor_y_spring.snap(mouse_y + jarvis_theme.CURSOR_FOLLOW_OFFSET.y())
                self._navigation_mode = "following"
                self._cursor_opacity_spring.set_target(1.0)
            self._ensure_timer_running()
            self.update()
            return

        if key_name == "shake_to_detach_enabled":
            self._shake_to_detach_enabled = bool(value)
            if not self._shake_to_detach_enabled:
                self._shake_detector.reset()
                if self._navigation_mode == "parked":
                    self._navigation_mode = "following"
            return

        if key_name == "screen_highlight_enabled":
            self._screen_highlight_enabled = bool(value)
            if not self._screen_highlight_enabled:
                self._is_highlighting = False
                self._highlight_points.clear()
                self._highlight_cached_path = None
                self.update()
            return

        if key_name == "captions_enabled":
            self._captions_enabled = bool(value)
            if not self._captions_enabled and not self._caption_force_visible:
                self.update()
            return

        if key_name == "show_in_screen_recordings":
            self._show_in_screen_recordings = bool(value)
            if self.isVisible():
                self._apply_win32_styles()
            self.clear_button_window.apply_capture_policy(not self._show_in_screen_recordings)
            return

        if key_name == "screen_drawing_mode":
            self._drawing_mode = str(value or jarvis_config.DEFAULT_SETTINGS["screen_drawing_mode"])
            return

        if key_name == "drawing_speed":
            self._drawing_speed = str(value or jarvis_config.DEFAULT_SETTINGS["drawing_speed"])
            return

    def benchmark_frame_ms(self, iterations: int = 60) -> Dict[str, float]:
        """
        Measures real `QPainter` frame rendering cost over `iterations` frames on a
        screen-sized offscreen ARGB32_Premultiplied surface.
        """
        iteration_count = max(1, int(iterations))
        restore_empty_demo = not self.has_active_drawings()
        if restore_empty_demo:
            self.load_demo_showcase()

        surface_width = max(1280, self.width())
        surface_height = max(720, self.height())
        image = QImage(surface_width, surface_height, QImage.Format.Format_ARGB32_Premultiplied)
        bounds = QRectF(0.0, 0.0, float(surface_width), float(surface_height))

        # Warm one frame so fonts/glyph caches settle before timing
        warmup_painter = QPainter(image)
        image.fill(Qt.GlobalColor.transparent)
        self._render_canvas(warmup_painter, bounds, time.monotonic() + 0.15)
        warmup_painter.end()

        frame_durations_ms: List[float] = []
        base_time = time.monotonic()
        for frame_index in range(iteration_count):
            simulated_now = base_time + frame_index * 0.016667
            start_ns = time.perf_counter_ns()
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            self._render_canvas(painter, bounds, simulated_now)
            painter.end()
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            frame_durations_ms.append(elapsed_ms)

        if restore_empty_demo:
            self.clear_all_drawings()
            self.clear_caption()

        sorted_durations = sorted(frame_durations_ms)
        p95_index = min(len(sorted_durations) - 1, int(0.95 * len(sorted_durations)))
        return {
            "avg_ms": round(sum(frame_durations_ms) / len(frame_durations_ms), 3),
            "min_ms": round(sorted_durations[0], 3),
            "max_ms": round(sorted_durations[-1], 3),
            "p95_ms": round(sorted_durations[p95_index], 3),
            "iterations": float(iteration_count),
        }

    # ------------------------------------------------------------------
    # Internal timing, animation steps & drawing cache helpers
    # ------------------------------------------------------------------

    def _on_clear_button_clicked(self) -> None:
        self.clear_all_drawings()
        self.clearRequested.emit()

    def _sync_clear_button_visibility(self) -> None:
        should_show = self.isVisible() and self.has_active_drawings()
        if should_show:
            self.clear_button_window.reposition_for_screen(self._screen_geometry)
            if not self.clear_button_window.isVisible():
                self.clear_button_window.show()
            self.clear_button_window.raise_()
        elif self.clear_button_window.isVisible():
            self.clear_button_window.hide()

    def _clear_bubble_state(self) -> None:
        self._bubble_words = []
        self._bubble_visible_count = 0
        self._bubble_next_word_time = 0.0
        self._bubble_fade_start = 0.0
        self._bubble_fade_end = 0.0
        self._bubble_opacity = 0.0

    @staticmethod
    def _word_reveal_delay(word_text: str) -> float:
        if word_text.endswith((".", "!", "?")):
            return 0.28
        if word_text.endswith((",", ";", ":")):
            return 0.18
        return 0.095 + min(len(word_text), 10) * 0.008

    def _estimate_reveal_duration(self, words: List[str]) -> float:
        if len(words) <= 1:
            return 0.0
        return sum(self._word_reveal_delay(word_text) for word_text in words[:-1])

    def _clamp_drawing_command(self, command: jarvis_response_parser.DrawingCommand) -> jarvis_response_parser.DrawingCommand:
        screen_width = float(max(640, self.width()))
        screen_height = float(max(480, self.height()))
        if command.kind == "paper":
            clamped_width = min(max(200.0, command.width), screen_width - 40.0)
            clamped_height = min(max(180.0, command.height), screen_height - 130.0)
            clamped_x = _clamp_float(command.x, 18.0, screen_width - clamped_width - 18.0)
            clamped_y = _clamp_float(command.y, 72.0, screen_height - clamped_height - 56.0)

            existing_paper_rects: List[Tuple[float, float, float, float]] = []
            content_rects: List[Tuple[float, float, float, float]] = []
            for active_item in self._drawings:
                item_cmd = active_item.command
                if item_cmd.kind == "paper":
                    paper_right = item_cmd.x + max(200.0, item_cmd.width) + 14.0
                    paper_bottom = item_cmd.y + max(180.0, item_cmd.height) + 14.0
                    existing_paper_rects.append((item_cmd.x - 14.0, item_cmd.y - 14.0, paper_right, paper_bottom))
                else:
                    content_right = item_cmd.x + max(64.0, item_cmd.width)
                    content_bottom = item_cmd.y + max(28.0, item_cmd.height)
                    content_rects.append((item_cmd.x, item_cmd.y, content_right, content_bottom))

            for lasso_x, lasso_y in self._highlight_points:
                content_rects.append((lasso_x - 24.0, lasso_y - 24.0, lasso_x + 24.0, lasso_y + 24.0))

            def _overlap_area(rect_a: Tuple[float, float, float, float], rect_b: Tuple[float, float, float, float]) -> float:
                inter_w = min(rect_a[2], rect_b[2]) - max(rect_a[0], rect_b[0])
                inter_h = min(rect_a[3], rect_b[3]) - max(rect_a[1], rect_b[1])
                if inter_w <= 0.0 or inter_h <= 0.0:
                    return 0.0
                return inter_w * inter_h

            margin_left = 22.0
            margin_right = max(margin_left, screen_width - clamped_width - 22.0)
            margin_top = 78.0
            margin_bottom = max(margin_top, screen_height - clamped_height - 56.0)
            margin_mid_y = max(margin_top, (margin_top + margin_bottom) * 0.5)

            candidate_slots: List[Tuple[float, float]] = [
                (clamped_x, clamped_y),
                (margin_left, margin_top),
                (margin_right, margin_top),
                (margin_left, margin_bottom),
                (margin_right, margin_bottom),
                (margin_left, margin_mid_y),
                (margin_right, margin_mid_y),
            ]
            best_slot = (clamped_x, clamped_y)
            best_score = float("inf")
            for cand_x, cand_y in candidate_slots:
                cand_rect = (cand_x, cand_y, cand_x + clamped_width, cand_y + clamped_height)
                paper_overlap = sum(_overlap_area(cand_rect, other_rect) for other_rect in existing_paper_rects)
                content_overlap = sum(_overlap_area(cand_rect, content_rect) for content_rect in content_rects)
                shift_distance = math.hypot(cand_x - clamped_x, cand_y - clamped_y)
                slot_score = (paper_overlap * 12.0) + (content_overlap * 2.5) + (shift_distance * 0.01)
                if slot_score < best_score:
                    best_score = slot_score
                    best_slot = (cand_x, cand_y)

            clamped_x, clamped_y = best_slot
            return jarvis_response_parser.DrawingCommand(
                kind=command.kind,
                x=clamped_x,
                y=clamped_y,
                width=clamped_width,
                height=clamped_height,
                radius=command.radius,
                label=command.label,
                header=command.header,
                body=command.body,
                tone=command.tone,
                sentence_index=command.sentence_index,
            )
        return command

    @staticmethod
    def _leading_edge_for_command(command: jarvis_response_parser.DrawingCommand) -> Optional[Tuple[float, float]]:
        if command.kind == "point":
            return (command.x, command.y)
        if command.kind == "underline":
            return (command.x + command.width, command.y)
        if command.kind == "ring":
            return (command.x + command.radius, command.y)
        return None

    def _prebuild_drawing_cache(self, item: _ActiveDrawingItem) -> None:
        command = item.command
        if command.kind == "paper":
            item.cached_paper_pixmap = _render_paper_card_to_pixmap(command, item.seed_value)
            item.cached_paper_size = (int(command.width), int(command.height))
        elif command.kind == "underline":
            points = _build_underline_wave_points(command.x, command.y, command.width, item.seed_value)
            item.cached_path = _build_smooth_catmull_rom_path(points)
        elif command.kind == "ring":
            points = _build_wobbly_circle_points(command.x, command.y, command.radius, item.seed_value)
            item.cached_path = _build_smooth_catmull_rom_path(points)
        elif command.kind == "box":
            points = _build_wobbly_rect_points(
                command.x, command.y, command.width, command.height, corner_radius=8.0, seed_value=item.seed_value
            )
            item.cached_path = _build_smooth_catmull_rom_path(points, closed=True)
        elif command.kind == "point":
            item.cached_path = self._build_point_arrow_path(command.x, command.y, item.seed_value)

    def _build_point_arrow_path(self, target_x: float, target_y: float, seed_value: int) -> QPainterPath:
        screen_width = float(max(640, self.width()))
        screen_height = float(max(480, self.height()))
        offset_x = 68.0 if target_x + 120.0 < screen_width else -68.0
        offset_y = -54.0 if target_y > 120.0 else 54.0
        start_x = target_x + offset_x
        start_y = _clamp_float(target_y + offset_y, 28.0, screen_height - 28.0)

        control_x = start_x - offset_x * 0.15
        control_y = target_y + offset_y * 0.12

        path = QPainterPath()
        path.moveTo(start_x, start_y)
        path.quadTo(control_x, control_y, target_x, target_y)
        return path

    def _needs_animation_tick(self, now_seconds: float) -> bool:
        cursor_visible_or_fading = (
            self._cursor_opacity_spring.value > 0.01
            or self._cursor_opacity_spring.target > 0.01
            or self._pending_hide_fade_at is not None
        )
        if cursor_visible_or_fading:
            return True
        if self._is_highlighting or (self._highlight_fade_end > now_seconds and len(self._highlight_points) >= 2):
            return True
        if self._bubble_words and self._bubble_opacity > 0.01:
            return True
        if self._is_caption_visible(now_seconds):
            return True
        if self._drawings:
            return True
        return False

    def _ensure_timer_running(self) -> None:
        now_seconds = time.monotonic()
        if self._needs_animation_tick(now_seconds):
            if not self._frame_timer.isActive():
                self._last_tick_time = now_seconds
                self._frame_timer.start()

    def _on_animation_tick(self) -> None:
        now_seconds = time.monotonic()
        delta_seconds = _clamp_float(now_seconds - self._last_tick_time, 0.001, 0.05)
        self._last_tick_time = now_seconds

        needs_repaint = self._step_state(delta_seconds, now_seconds)
        if needs_repaint:
            self.update()

        if not self._needs_animation_tick(now_seconds):
            self._frame_timer.stop()

    def _step_state(self, delta_seconds: float, now_seconds: float) -> bool:
        changed = False

        # 1. Delayed hide-cursor fade trigger ("okay see you <3")
        if self._pending_hide_fade_at is not None and now_seconds >= self._pending_hide_fade_at:
            self._pending_hide_fade_at = None
            self._cursor_opacity_spring.set_target(0.0)
            changed = True

        # 2. Sample mouse & update cursor position / flight / shake
        mouse_x, mouse_y = self._sample_local_mouse_position()
        mouse_moved = math.hypot(mouse_x - self._last_mouse_position[0], mouse_y - self._last_mouse_position[1]) > 0.5
        self._last_mouse_position = (mouse_x, mouse_y)

        if (
            self._shake_to_detach_enabled
            and not self._hide_floating_cursor
            and self._navigation_mode == "following"
            and self._shake_detector.feed_sample(mouse_x, mouse_y, now_seconds)
        ):
            self._navigation_mode = "parked"
            self._parked_until = now_seconds + SHAKE_PARK_SECONDS
            changed = True

        if self._navigation_mode == "parked":
            if now_seconds >= self._parked_until:
                self._navigation_mode = "following"
                changed = True
        elif self._navigation_mode == "flying":
            elapsed = now_seconds - self._flight_start_time
            progress = _ease_in_out_cubic(elapsed / max(0.05, self._flight_duration))
            inv = 1.0 - progress
            curve_x = (
                inv * inv * self._flight_start.x()
                + 2.0 * inv * progress * self._flight_control.x()
                + progress * progress * self._flight_target.x()
            )
            curve_y = (
                inv * inv * self._flight_start.y()
                + 2.0 * inv * progress * self._flight_control.y()
                + progress * progress * self._flight_target.y()
            )
            self._cursor_x_spring.snap(curve_x)
            self._cursor_y_spring.snap(curve_y)
            changed = True
            if progress >= 1.0:
                self._navigation_mode = "pointing"
                self._target_lock_mouse_anchor = (mouse_x, mouse_y)
        elif self._navigation_mode == "pointing":
            if now_seconds >= self._target_hold_until:
                self._navigation_mode = "following"
                self._target_lock_mouse_anchor = None
                changed = True
            elif self._target_lock_mouse_anchor is not None:
                anchor_x, anchor_y = self._target_lock_mouse_anchor
                if math.hypot(mouse_x - anchor_x, mouse_y - anchor_y) > CURSOR_TARGET_RELEASE_MOUSE_DISTANCE:
                    self._navigation_mode = "following"
                    self._target_lock_mouse_anchor = None
                    changed = True

        if self._navigation_mode == "following" and not self._hide_floating_cursor:
            screen_width = float(max(200, self.width()))
            screen_height = float(max(200, self.height()))
            desired_x = _clamp_float(mouse_x + jarvis_theme.CURSOR_FOLLOW_OFFSET.x(), 16.0, screen_width - 16.0)
            desired_y = _clamp_float(mouse_y + jarvis_theme.CURSOR_FOLLOW_OFFSET.y(), 16.0, screen_height - 16.0)
            self._cursor_x_spring.set_target(desired_x)
            self._cursor_y_spring.set_target(desired_y)

        if not self._cursor_x_spring.is_settled(0.15) or not self._cursor_y_spring.is_settled(0.15) or mouse_moved:
            self._cursor_x_spring.step(delta_seconds)
            self._cursor_y_spring.step(delta_seconds)
            changed = True

        # 3. Opacity & morph springs
        for spring_instance in (self._cursor_opacity_spring, self._listening_morph_spring, self._thinking_morph_spring):
            if not spring_instance.is_settled(0.005):
                spring_instance.step(delta_seconds)
                changed = True

        if self._is_listening or self._is_thinking or self._is_speaking:
            smoothing_rate = min(1.0, delta_seconds * 14.0)
            self._smoothed_audio_level += (self._audio_level - self._smoothed_audio_level) * smoothing_rate
            changed = True

        # 4. Word-by-word bubble reveal & fade
        if self._bubble_words:
            if self._bubble_word_by_word and self._bubble_visible_count < len(self._bubble_words):
                if now_seconds >= self._bubble_next_word_time:
                    revealed_word = self._bubble_words[self._bubble_visible_count]
                    self._bubble_visible_count += 1
                    self._bubble_next_word_time = now_seconds + self._word_reveal_delay(revealed_word)
                    changed = True
            if now_seconds >= self._bubble_fade_start:
                if now_seconds >= self._bubble_fade_end:
                    self._clear_bubble_state()
                else:
                    fade_span = max(0.05, self._bubble_fade_end - self._bubble_fade_start)
                    self._bubble_opacity = _clamp_float((self._bubble_fade_end - now_seconds) / fade_span, 0.0, 1.0)
                changed = True

        # 5. Draw-to-Ask mouse sampling (smoothed via EMA) & fade-out
        if self._is_highlighting and not self._interactive_lasso_active and self._screen_highlight_enabled:
            if self._append_smoothed_highlight_point(mouse_x, mouse_y):
                changed = True
        elif self._highlight_fade_end > 0.0:
            if now_seconds >= self._highlight_fade_end:
                self._highlight_points.clear()
                self._highlight_cached_path = None
                self._highlight_fade_start = 0.0
                self._highlight_fade_end = 0.0
            changed = True

        # 6. Active drawings animation, synced sound trigger & expiration
        if self._drawings:
            before_count = len(self._drawings)
            still_active: List[_ActiveDrawingItem] = []
            active_drawing_now = False
            for item in self._drawings:
                if now_seconds < item.expires_at:
                    still_active.append(item)
                    elapsed_item = now_seconds - item.created_at - item.delay_seconds
                    if 0.0 <= elapsed_item < item.duration_seconds and not item.has_started_sound:
                        item.has_started_sound = True
                        self.strokeAnimationStarted.emit(
                            item.command.kind,
                            int(round(item.duration_seconds * 1000.0)),
                        )
                    if 0.0 <= elapsed_item <= item.duration_seconds:
                        active_drawing_now = True
                        if self._navigation_mode != "flying":
                            raw_progress = _clamp_float(
                                elapsed_item / max(0.05, item.duration_seconds),
                                0.0,
                                1.0,
                            )
                            eased_progress = raw_progress * raw_progress * (3.0 - 2.0 * raw_progress)
                            tip_point = self._active_stroke_tip_point(item.command, eased_progress)
                            if tip_point is not None:
                                self._cursor_x_spring.set_target(tip_point[0] + 8.0)
                                self._cursor_y_spring.set_target(tip_point[1] + 10.0)
                                self._target_hold_until = max(self._target_hold_until, now_seconds + 2.6)
                                self._navigation_mode = "pointing"
                    if elapsed_item <= item.duration_seconds + 0.05 or (item.expires_at - now_seconds) <= DRAWING_FADE_OUT_SECONDS:
                        changed = True
            if self._was_actively_drawing and not active_drawing_now:
                self.strokeAnimationFinished.emit()
            self._was_actively_drawing = active_drawing_now
            if len(still_active) != before_count:
                self._drawings = still_active
                self._sync_clear_button_visibility()
                changed = True
        elif self._was_actively_drawing:
            self._was_actively_drawing = False
            self.strokeAnimationFinished.emit()

        # 7. Caption expiry
        if self._caption_expire_at > 0.0 and now_seconds >= self._caption_expire_at:
            self.clear_caption()
            changed = True

        return changed

    def _is_caption_visible(self, now_seconds: float) -> bool:
        if not self._caption_words:
            return False
        if not self._captions_enabled and not self._caption_force_visible:
            return False
        if self._caption_expire_at > 0.0 and now_seconds >= self._caption_expire_at:
            return False
        return True

    # ------------------------------------------------------------------
    # Painting (shared across live paintEvent, snapshots & benchmark)
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        self._render_canvas(painter, QRectF(self.rect()), time.monotonic())

    def _render_canvas(
        self,
        painter: QPainter,
        bounds: QRectF,
        now_seconds: float,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # 1. Screen drawing primitives (paper cards behind ink marks)
        for item in self._drawings:
            if item.command.kind == "paper":
                self._paint_drawing_item(painter, item, now_seconds)
        for item in self._drawings:
            if item.command.kind != "paper":
                self._paint_drawing_item(painter, item, now_seconds)

        # 2. Draw-to-Ask highlight stroke
        self._paint_highlight_stroke(painter, now_seconds)

        # 3. Companion cursor & cursor message bubble
        cursor_opacity = _clamp_float(self._cursor_opacity_spring.value, 0.0, 1.0)
        if cursor_opacity > 0.01:
            cursor_center = QPointF(self._cursor_x_spring.value, self._cursor_y_spring.value)
            self._paint_companion_cursor(painter, cursor_center, cursor_opacity, now_seconds)
            if self._bubble_words and self._bubble_opacity > 0.01:
                self._paint_cursor_bubble(painter, cursor_center, cursor_opacity * self._bubble_opacity, bounds)

        # 4. Bottom LiveCaption bar
        if self._is_caption_visible(now_seconds):
            self._paint_live_caption(painter, bounds)

    def _paint_drawing_item(
        self,
        painter: QPainter,
        item: _ActiveDrawingItem,
        now_seconds: float,
    ) -> None:
        local_elapsed = now_seconds - item.created_at - item.delay_seconds
        if local_elapsed < 0.0:
            return

        draw_progress = _ease_out_cubic(local_elapsed / max(0.05, item.duration_seconds))
        remaining = item.expires_at - now_seconds
        fade_alpha = 1.0 if remaining >= DRAWING_FADE_OUT_SECONDS else _clamp_float(remaining / DRAWING_FADE_OUT_SECONDS, 0.0, 1.0)
        if fade_alpha <= 0.01:
            return

        command = item.command
        if command.kind == "paper":
            if item.cached_paper_pixmap is None:
                item.cached_paper_pixmap = _render_paper_card_to_pixmap(command, item.seed_value)
            painter.save()
            painter.setOpacity(draw_progress * fade_alpha)
            lift_offset = (1.0 - draw_progress) * 12.0
            painter.drawPixmap(
                QPointF(command.x - 14.0, command.y - 14.0 + lift_offset),
                item.cached_paper_pixmap,
            )
            painter.restore()
            return

        tone_color = _resolve_tone_color(
            command.tone,
            jarvis_theme.DRAWING_TONES["yellow"] if command.kind == "highlight" else jarvis_theme.SCREEN_MARKER_BLUE,
        )

        if command.kind == "underline":
            if draw_progress >= 0.99:
                self._paint_cached_or_live_stroke_item(
                    painter,
                    item,
                    fade_alpha,
                    lambda target_painter: self._paint_underline(target_painter, item, tone_color, 1.0, 1.0),
                )
            else:
                self._paint_underline(painter, item, tone_color, draw_progress, fade_alpha)
        elif command.kind == "ring":
            if draw_progress >= 0.99:
                self._paint_cached_or_live_stroke_item(
                    painter,
                    item,
                    fade_alpha,
                    lambda target_painter: self._paint_ring(target_painter, item, tone_color, 1.0, 1.0),
                )
            else:
                self._paint_ring(painter, item, tone_color, draw_progress, fade_alpha)
        elif command.kind == "point":
            if draw_progress >= 0.99:
                self._paint_cached_or_live_stroke_item(
                    painter,
                    item,
                    fade_alpha,
                    lambda target_painter: self._paint_point_arrow(target_painter, item, tone_color, 1.0, 1.0),
                )
            else:
                self._paint_point_arrow(painter, item, tone_color, draw_progress, fade_alpha)
        elif command.kind == "box":
            if draw_progress >= 0.99:
                self._paint_cached_or_live_stroke_item(
                    painter,
                    item,
                    fade_alpha,
                    lambda target_painter: self._paint_box_frame(target_painter, item, tone_color, 1.0, 1.0),
                )
            else:
                self._paint_box_frame(painter, item, tone_color, draw_progress, fade_alpha)
        elif command.kind == "highlight":
            if draw_progress >= 0.99:
                self._paint_cached_or_live_stroke_item(
                    painter,
                    item,
                    fade_alpha,
                    lambda target_painter: self._paint_highlight_wash(target_painter, command, tone_color, 1.0, 1.0),
                )
            else:
                self._paint_highlight_wash(painter, command, tone_color, draw_progress, fade_alpha)

    def _compute_item_sprite_bounds(self, item: _ActiveDrawingItem) -> QRectF:
        command = item.command
        if command.kind == "underline":
            bounds = QRectF(command.x - 48.0, command.y - 36.0, command.width + 72.0, 72.0)
        elif command.kind == "ring":
            bounds = QRectF(
                command.x - command.radius - 28.0,
                command.y - command.radius - 28.0,
                command.radius * 2.0 + 56.0,
                command.radius * 2.0 + 56.0,
            )
        elif command.kind == "point":
            if item.cached_path is not None:
                bounds = item.cached_path.boundingRect().adjusted(-28.0, -28.0, 28.0, 28.0)
            else:
                bounds = QRectF(command.x - 100.0, command.y - 80.0, 200.0, 160.0)
        else:
            bounds = QRectF(command.x, command.y, max(24.0, command.width), max(24.0, command.height)).adjusted(
                -26.0, -26.0, 26.0, 26.0
            )

        if command.label:
            label_font = jarvis_theme.ui_font(10.0, QFont.Weight.DemiBold)
            label_width = max(80.0, QFontMetricsF(label_font).horizontalAdvance(command.label) + 40.0)
            if command.kind == "ring":
                label_rect = QRectF(command.x - label_width * 0.5, max(0.0, command.y - command.radius - 42.0), label_width, 46.0)
                bounds = bounds.united(label_rect)
            elif command.kind == "point":
                anchor_x = item.cached_path.pointAtPercent(0.0).x() if item.cached_path is not None else command.x
                anchor_y = item.cached_path.pointAtPercent(0.0).y() if item.cached_path is not None else command.y
                label_rect = QRectF(anchor_x - label_width * 0.5, max(0.0, anchor_y - 36.0), label_width, 46.0)
                bounds = bounds.united(label_rect)
            elif command.kind in ("box", "highlight"):
                label_rect = QRectF(
                    command.x + command.width * 0.5 - label_width * 0.5,
                    max(0.0, command.y - 38.0),
                    label_width,
                    46.0,
                )
                bounds = bounds.united(label_rect)
        return bounds

    def _paint_cached_or_live_stroke_item(
        self,
        painter: QPainter,
        item: _ActiveDrawingItem,
        fade_alpha: float,
        draw_fn: Any,
    ) -> None:
        if item.cached_stroke_pixmap is None:
            sprite_bounds = self._compute_item_sprite_bounds(item)
            sprite_width = max(8, int(math.ceil(sprite_bounds.width())))
            sprite_height = max(8, int(math.ceil(sprite_bounds.height())))
            dpr_scale = max(2.0, float(self.devicePixelRatioF()))
            pixmap = QPixmap(
                int(math.ceil(sprite_width * dpr_scale)),
                int(math.ceil(sprite_height * dpr_scale)),
            )
            pixmap.fill(Qt.GlobalColor.transparent)
            pixmap.setDevicePixelRatio(dpr_scale)
            sprite_painter = QPainter(pixmap)
            sprite_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            sprite_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            sprite_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            sprite_painter.translate(-sprite_bounds.left(), -sprite_bounds.top())
            draw_fn(sprite_painter)
            sprite_painter.end()
            item.cached_stroke_pixmap = pixmap
            item.cached_stroke_origin = (sprite_bounds.left(), sprite_bounds.top())

        if fade_alpha < 0.999:
            painter.save()
            painter.setOpacity(fade_alpha)
            painter.drawPixmap(QPointF(item.cached_stroke_origin[0], item.cached_stroke_origin[1]), item.cached_stroke_pixmap)
            painter.restore()
        else:
            painter.drawPixmap(QPointF(item.cached_stroke_origin[0], item.cached_stroke_origin[1]), item.cached_stroke_pixmap)

    def _paint_underline(
        self,
        painter: QPainter,
        item: _ActiveDrawingItem,
        tone_color: QColor,
        progress: float,
        fade_alpha: float,
    ) -> None:
        command = item.command
        eased_progress = 1.0 if progress >= 0.99 else (progress * progress * (3.0 - 2.0 * progress))
        if eased_progress >= 0.99 and item.cached_path is not None:
            stroke_path = item.cached_path
        else:
            span = max(8.0, command.width * eased_progress)
            points = _build_underline_wave_points(command.x, command.y, span, item.seed_value)
            stroke_path = _build_smooth_catmull_rom_path(points)

        if not stroke_path.isEmpty() and fade_alpha > 0.01:
            painter.save()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            shadow_pen = QPen(
                QColor(8, 16, 28, int(88 * fade_alpha)),
                4.4,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(shadow_pen)
            painter.drawPath(stroke_path)

            pen_color = _blend_colors(tone_color, QColor(90, 178, 255), 0.28)
            core_pen = QPen(
                jarvis_theme.with_alpha(pen_color, int(242 * fade_alpha)),
                2.8,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(core_pen)
            painter.drawPath(stroke_path)
            painter.restore()

        if self._drawing_mode == "full_sketch":
            badge_text = jarvis_response_parser.clean_spoken_text(command.label)
            if badge_text and len(badge_text) <= 3:
                badge_center = QPointF(command.x - 24.0, command.y - 12.0)
                badge_radius = 11.0
                badge_opacity = eased_progress * fade_alpha
                painter.save()
                painter.setBrush(jarvis_theme.with_alpha(tone_color, int(235 * badge_opacity)))
                painter.setPen(QPen(QColor(175, 232, 255, int(220 * badge_opacity)), 1.2))
                painter.drawEllipse(badge_center, badge_radius, badge_radius)

                badge_font = jarvis_theme.ui_font(10.0, QFont.Weight.Bold)
                painter.setFont(badge_font)
                painter.setPen(QColor(255, 255, 255, int(252 * badge_opacity)))
                badge_rect = QRectF(
                    badge_center.x() - badge_radius,
                    badge_center.y() - badge_radius,
                    badge_radius * 2.0,
                    badge_radius * 2.0,
                )
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
                painter.restore()

    def _paint_ring(
        self,
        painter: QPainter,
        item: _ActiveDrawingItem,
        tone_color: QColor,
        progress: float,
        fade_alpha: float,
    ) -> None:
        command = item.command
        if progress >= 0.99 and item.cached_path is not None:
            ring_path = item.cached_path
        else:
            all_points = _build_wobbly_circle_points(command.x, command.y, command.radius, item.seed_value)
            visible_count = max(2, int(len(all_points) * progress))
            ring_path = _build_smooth_catmull_rom_path(all_points[:visible_count])

        if progress > 0.25:
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(jarvis_theme.with_alpha(tone_color, int(24 * progress * fade_alpha)))
            painter.drawEllipse(QPointF(command.x, command.y), command.radius, command.radius)
            painter.restore()

        _paint_glow_stroke(
            painter,
            ring_path,
            core_color=tone_color,
            core_width=2.8,
            glow_width=5.4,
            opacity=fade_alpha,
        )

        if self._drawing_mode == "full_sketch" and command.label and progress > 0.45:
            pill_opacity = _clamp_float((progress - 0.45) / 0.55, 0.0, 1.0) * fade_alpha
            pill_top_y = max(12.0, command.y - command.radius - 34.0)
            _paint_callout_pill(painter, command.x, pill_top_y, command.label, tone_color, opacity=pill_opacity)

    def _paint_point_arrow(
        self,
        painter: QPainter,
        item: _ActiveDrawingItem,
        tone_color: QColor,
        progress: float,
        fade_alpha: float,
    ) -> None:
        command = item.command
        arrow_path = item.cached_path or self._build_point_arrow_path(command.x, command.y, item.seed_value)
        if progress < 0.99:
            partial_path = QPainterPath()
            sample_steps = 16
            partial_path.moveTo(arrow_path.pointAtPercent(0.0))
            for step_index in range(1, sample_steps + 1):
                percent = (step_index / float(sample_steps)) * progress
                partial_path.lineTo(arrow_path.pointAtPercent(percent))
            draw_path = partial_path
        else:
            draw_path = arrow_path

        _paint_glow_stroke(
            painter,
            draw_path,
            core_color=tone_color,
            core_width=2.8,
            glow_width=5.2,
            opacity=fade_alpha,
        )

        # Hand-drawn arrowhead at tip
        if progress > 0.7:
            tip_point = arrow_path.pointAtPercent(progress)
            prev_point = arrow_path.pointAtPercent(max(0.0, progress - 0.08))
            angle = math.atan2(tip_point.y() - prev_point.y(), tip_point.x() - prev_point.x())
            head_len = 11.0
            wing_angle = 0.52
            head_path = QPainterPath()
            head_path.moveTo(
                tip_point.x() - head_len * math.cos(angle - wing_angle),
                tip_point.y() - head_len * math.sin(angle - wing_angle),
            )
            head_path.lineTo(tip_point.x(), tip_point.y())
            head_path.lineTo(
                tip_point.x() - head_len * math.cos(angle + wing_angle),
                tip_point.y() - head_len * math.sin(angle + wing_angle),
            )
            _paint_glow_stroke(
                painter,
                head_path,
                core_color=tone_color,
                core_width=2.8,
                glow_width=5.0,
                opacity=fade_alpha,
            )

        if self._drawing_mode == "full_sketch" and command.label and progress > 0.35:
            tail_point = arrow_path.pointAtPercent(0.0)
            pill_opacity = _clamp_float((progress - 0.35) / 0.5, 0.0, 1.0) * fade_alpha
            _paint_callout_pill(painter, tail_point.x(), max(12.0, tail_point.y() - 28.0), command.label, tone_color, opacity=pill_opacity)

    def _paint_box_frame(
        self,
        painter: QPainter,
        item: _ActiveDrawingItem,
        tone_color: QColor,
        progress: float,
        fade_alpha: float,
    ) -> None:
        command = item.command
        if progress >= 0.99 and item.cached_path is not None:
            box_path = item.cached_path
        else:
            all_points = _build_wobbly_rect_points(
                command.x, command.y, command.width, command.height, corner_radius=8.0, seed_value=item.seed_value
            )
            visible_count = max(2, int(len(all_points) * progress))
            box_path = _build_smooth_catmull_rom_path(all_points[:visible_count])

        _paint_glow_stroke(
            painter,
            box_path,
            core_color=tone_color,
            core_width=2.6,
            glow_width=5.2,
            opacity=fade_alpha,
        )

        if self._drawing_mode == "full_sketch" and command.label and progress > 0.45:
            pill_opacity = _clamp_float((progress - 0.45) / 0.55, 0.0, 1.0) * fade_alpha
            _paint_callout_pill(
                painter,
                command.x + command.width * 0.5,
                max(12.0, command.y - 30.0),
                command.label,
                tone_color,
                opacity=pill_opacity,
            )

    def _paint_highlight_wash(
        self,
        painter: QPainter,
        command: jarvis_response_parser.DrawingCommand,
        tone_color: QColor,
        progress: float,
        fade_alpha: float,
    ) -> None:
        reveal_width = max(12.0, command.width * progress)
        wash_rect = QRectF(command.x, command.y, reveal_width, command.height)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(jarvis_theme.with_alpha(tone_color, int(76 * fade_alpha)))
        painter.drawRoundedRect(wash_rect, 8.0, 8.0)

        # Subtle brighter core band like a highlighter marker pass
        inner_rect = wash_rect.adjusted(2.0, command.height * 0.16, -2.0, -command.height * 0.16)
        painter.setBrush(jarvis_theme.with_alpha(tone_color, int(48 * fade_alpha)))
        painter.drawRoundedRect(inner_rect, 6.0, 6.0)
        painter.restore()

        if self._drawing_mode == "full_sketch" and command.label and progress > 0.55:
            _paint_callout_pill(
                painter,
                command.x + reveal_width * 0.5,
                max(12.0, command.y - 28.0),
                command.label,
                tone_color,
                opacity=fade_alpha * progress,
            )

    def _paint_highlight_stroke(self, painter: QPainter, now_seconds: float) -> None:
        if len(self._highlight_points) < 2:
            return
        if not self._is_highlighting and self._highlight_fade_end > 0.0:
            fade_span = max(0.05, self._highlight_fade_end - self._highlight_fade_start)
            fade_alpha = _clamp_float((self._highlight_fade_end - now_seconds) / fade_span, 0.0, 1.0)
        elif self._is_highlighting:
            fade_alpha = 1.0
        else:
            return

        if fade_alpha <= 0.01:
            return

        if self._highlight_cached_path is None:
            self._highlight_cached_path = _build_smooth_catmull_rom_path(self._highlight_points)

        # Clean, simple, non-glowing lasso stroke without corner HUD brackets or floating pill
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        lasso_pen = QPen(
            QColor(228, 232, 240, int(235 * fade_alpha)),
            2.2,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(lasso_pen)
        painter.drawPath(self._highlight_cached_path)
        painter.restore()

    def _paint_companion_cursor(
        self,
        painter: QPainter,
        center: QPointF,
        opacity: float,
        now_seconds: float,
    ) -> None:
        listening_weight = _clamp_float(self._listening_morph_spring.value, 0.0, 1.0)
        thinking_weight = _clamp_float(self._thinking_morph_spring.value, 0.0, 1.0)
        normal_weight = _clamp_float(1.0 - max(listening_weight, thinking_weight), 0.0, 1.0)

        if normal_weight > 0.02:
            self._paint_cursor_triangle(painter, center, opacity * normal_weight, now_seconds)
        if listening_weight > 0.02:
            self._paint_cursor_listening_bars(painter, center, opacity * listening_weight, now_seconds)
        if thinking_weight > 0.02:
            self._paint_cursor_thinking_dots(painter, center, opacity * thinking_weight, now_seconds)

    def _paint_cursor_triangle(
        self,
        painter: QPainter,
        center: QPointF,
        opacity: float,
        now_seconds: float,
    ) -> None:
        del now_seconds
        painter.save()
        painter.translate(center)
        gradient = QLinearGradient(-8.0, -8.0, 8.0, 8.0)
        gradient.setColorAt(0.0, jarvis_theme.with_alpha(_blend_colors(self._cursor_color, QColor(255, 255, 255), 0.25), int(252 * opacity)))
        gradient.setColorAt(1.0, jarvis_theme.with_alpha(self._cursor_color, int(252 * opacity)))

        rim_color = _blend_colors(self._cursor_color, QColor(225, 246, 255), 0.62)
        painter.setBrush(gradient)
        painter.setPen(QPen(jarvis_theme.with_alpha(rim_color, int(225 * opacity)), 1.15))
        painter.drawPath(self._cursor_triangle_path)
        painter.restore()

    def _paint_cursor_listening_bars(
        self,
        painter: QPainter,
        center: QPointF,
        opacity: float,
        now_seconds: float,
    ) -> None:
        cyan_base = QColor("#2ECBFF")
        cyan_core = QColor("#8BEBFF")

        bar_count = 5
        bar_width = 3.0
        bar_gap = 2.6
        total_width = bar_count * bar_width + (bar_count - 1) * bar_gap
        start_x = center.x() - total_width / 2.0
        envelopes = (0.64, 0.90, 1.0, 0.90, 0.64)
        level_factor = max(0.20, self._smoothed_audio_level)

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for bar_index in range(bar_count):
            wave = math.sin(now_seconds * 7.8 + bar_index * 1.15) * 0.5 + 0.5
            bar_height = 4.5 + wave * 14.0 * level_factor * envelopes[bar_index]
            bar_x = start_x + bar_index * (bar_width + bar_gap)
            bar_rect = QRectF(bar_x, center.y() - bar_height / 2.0, bar_width, bar_height)
            bar_alpha = int((200 + 55 * wave) * opacity)

            painter.setBrush(jarvis_theme.with_alpha(_blend_colors(cyan_base, cyan_core, 0.45), bar_alpha))
            painter.drawRoundedRect(bar_rect, bar_width / 2.0, bar_width / 2.0)
        painter.restore()

    def _paint_cursor_thinking_dots(
        self,
        painter: QPainter,
        center: QPointF,
        opacity: float,
        now_seconds: float,
    ) -> None:
        dot_color = _blend_colors(self._cursor_color, QColor("#67D2FF"), 0.52)

        dot_count = 3
        dot_spacing = 7.8
        base_radius = 2.7
        start_x = center.x() - dot_spacing

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for dot_index in range(dot_count):
            phase = now_seconds * 5.2 - dot_index * 0.85
            hop = max(0.0, math.sin(phase)) ** 1.6
            dot_x = start_x + dot_index * dot_spacing
            dot_y = center.y() - hop * 5.4
            dot_radius = base_radius * (1.0 + 0.22 * hop)
            dot_center = QPointF(dot_x, dot_y)

            bright_dot = _blend_colors(dot_color, QColor(235, 249, 255), 0.35 + 0.4 * hop)
            painter.setBrush(jarvis_theme.with_alpha(bright_dot, int((205 + 50 * hop) * opacity)))
            painter.drawEllipse(dot_center, dot_radius, dot_radius)
        painter.restore()

    def _paint_cursor_bubble(
        self,
        painter: QPainter,
        cursor_center: QPointF,
        opacity: float,
        bounds: QRectF,
    ) -> None:
        visible_words = self._bubble_words[: max(1, self._bubble_visible_count)]
        visible_text = " ".join(visible_words)
        if not visible_text:
            return

        bubble_font = jarvis_theme.ui_font(10.5, QFont.Weight.Medium)
        metrics = QFontMetricsF(bubble_font)
        padding_x = 12.0
        padding_y = 7.0
        max_text_width = min(270.0, max(120.0, bounds.width() - padding_x * 2.0 - 32.0))

        text_bounds = metrics.boundingRect(
            QRectF(0.0, 0.0, max_text_width, 400.0),
            Qt.TextFlag.TextWordWrap,
            visible_text,
        )
        pill_width = min(max_text_width + padding_x * 2.0, text_bounds.width() + padding_x * 2.0 + 4.0)
        pill_height = text_bounds.height() + padding_y * 2.0

        origin_x = cursor_center.x() + 16.0
        origin_y = cursor_center.y() + 18.0
        if origin_x + pill_width > bounds.right() - 16.0:
            origin_x = max(bounds.left() + 16.0, cursor_center.x() - pill_width - 14.0)
        if origin_y + pill_height > bounds.bottom() - 16.0:
            origin_y = max(bounds.top() + 16.0, cursor_center.y() - pill_height - 14.0)

        pill_rect = QRectF(origin_x, origin_y, pill_width, pill_height)

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, int(75 * opacity)))
        painter.drawRoundedRect(pill_rect.translated(0.0, 3.0), 10.0, 10.0)

        painter.setBrush(QColor(20, 20, 24, int(215 * opacity)))
        painter.setPen(QPen(QColor(255, 255, 255, int(36 * opacity)), 1.0))
        painter.drawRoundedRect(pill_rect, 10.0, 10.0)

        painter.setFont(bubble_font)
        painter.setPen(QColor(246, 249, 255, int(248 * opacity)))
        inner_rect = pill_rect.adjusted(padding_x, padding_y, -padding_x, -padding_y)
        painter.drawText(inner_rect, Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, visible_text)
        painter.restore()

    def _paint_live_caption(self, painter: QPainter, bounds: QRectF) -> None:
        if not self._caption_words:
            return

        cache_key = (
            tuple(self._caption_words),
            self._caption_highlight_word_index,
            int(bounds.width()),
            int(bounds.height()),
            self._cursor_color.rgba(),
        )
        if self._cached_caption_key == cache_key and self._cached_caption_pixmap is not None:
            painter.drawPixmap(QPointF(self._cached_caption_origin[0], self._cached_caption_origin[1]), self._cached_caption_pixmap)
            return

        caption_font = jarvis_theme.ui_font(13.0, QFont.Weight.Medium)
        metrics = QFontMetricsF(caption_font)
        max_line_width = min(CAPTION_MAX_WIDTH, bounds.width() - 140.0)

        lines: List[List[Tuple[int, str]]] = []
        current_line: List[Tuple[int, str]] = []
        current_width = 0.0
        space_width = metrics.horizontalAdvance(" ")

        for word_index, word_text in enumerate(self._caption_words):
            word_width = metrics.horizontalAdvance(word_text)
            added_width = word_width + (space_width if current_line else 0.0)
            if current_line and current_width + added_width > max_line_width:
                lines.append(current_line)
                current_line = [(word_index, word_text)]
                current_width = word_width
            else:
                current_line.append((word_index, word_text))
                current_width += added_width
        if current_line:
            lines.append(current_line)

        lines = lines[:3]
        line_widths = [
            sum(metrics.horizontalAdvance(word_text) for _, word_text in line)
            + space_width * max(0, len(line) - 1)
            for line in lines
        ]
        content_width = max(line_widths) if line_widths else 120.0
        line_height = metrics.height() + 4.0
        padding_x = 22.0
        padding_y = 11.0

        pill_width = content_width + padding_x * 2.0
        pill_height = len(lines) * line_height + padding_y * 2.0
        pill_x = bounds.center().x() - pill_width / 2.0
        center_y = bounds.height() - CAPTION_BOTTOM_CENTER_OFFSET
        pill_y = _clamp_float(center_y - pill_height / 2.0, 40.0, bounds.height() - pill_height - 22.0)

        shadow_pad = 10.0
        sprite_w = int(math.ceil(pill_width + shadow_pad * 2.0))
        sprite_h = int(math.ceil(pill_height + shadow_pad * 2.0))
        dpr_scale = max(2.0, float(self.devicePixelRatioF()))
        caption_pixmap = QPixmap(
            int(math.ceil(sprite_w * dpr_scale)),
            int(math.ceil(sprite_h * dpr_scale)),
        )
        caption_pixmap.fill(Qt.GlobalColor.transparent)
        caption_pixmap.setDevicePixelRatio(dpr_scale)

        sprite_painter = QPainter(caption_pixmap)
        sprite_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        sprite_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        sprite_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        local_rect = QRectF(shadow_pad, shadow_pad, pill_width, pill_height)
        sprite_painter.setPen(Qt.PenStyle.NoPen)
        sprite_painter.setBrush(QColor(0, 0, 0, 82))
        sprite_painter.drawRoundedRect(local_rect.translated(0.0, 4.0), CAPTION_CORNER_RADIUS, CAPTION_CORNER_RADIUS)

        sprite_painter.setBrush(QColor(16, 18, 22, 225))
        sprite_painter.setPen(QPen(QColor(255, 255, 255, 36), 1.0))
        sprite_painter.drawRoundedRect(local_rect, CAPTION_CORNER_RADIUS, CAPTION_CORNER_RADIUS)

        sprite_painter.setFont(caption_font)
        has_active_karaoke = self._caption_highlight_word_index >= 0

        for line_index, line in enumerate(lines):
            line_width = line_widths[line_index]
            cursor_x = local_rect.center().x() - line_width / 2.0
            baseline_y = local_rect.top() + padding_y + line_index * line_height + metrics.ascent()

            for word_index, word_text in line:
                word_width = metrics.horizontalAdvance(word_text)
                if not has_active_karaoke or word_index < self._caption_highlight_word_index:
                    text_color = QColor(246, 248, 255, 246)
                elif word_index == self._caption_highlight_word_index:
                    text_color = QColor(255, 255, 255, 255)
                else:
                    text_color = QColor(255, 255, 255, 112)

                sprite_painter.setPen(text_color)
                sprite_painter.drawText(QPointF(cursor_x, baseline_y), word_text)

                if has_active_karaoke and word_index == self._caption_highlight_word_index:
                    underline_y = baseline_y + 4.0
                    sprite_painter.setPen(
                        QPen(jarvis_theme.with_alpha(self._cursor_color, 235), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
                    )
                    sprite_painter.drawLine(QPointF(cursor_x, underline_y), QPointF(cursor_x + word_width, underline_y))

                cursor_x += word_width + space_width

        sprite_painter.end()
        self._cached_caption_key = cache_key
        self._cached_caption_pixmap = caption_pixmap
        self._cached_caption_origin = (pill_x - shadow_pad, pill_y - shadow_pad)
        painter.drawPixmap(QPointF(self._cached_caption_origin[0], self._cached_caption_origin[1]), caption_pixmap)


# ---------------------------------------------------------------------------
# Visual snapshots & CLI entry point
# ---------------------------------------------------------------------------


def _paint_dark_snapshot_backdrop(painter: QPainter, width: int, height: int) -> None:
    """Draws a realistic dark reader backdrop matching the reference camera photos."""
    backdrop = QLinearGradient(0.0, 0.0, float(width), float(height))
    backdrop.setColorAt(0.0, QColor("#181A20"))
    backdrop.setColorAt(1.0, QColor("#121419"))
    painter.fillRect(0, 0, width, height, backdrop)

    heading_font = jarvis_theme.ui_font(15.0, QFont.Weight.Bold)
    body_font = jarvis_theme.ui_font(11.5, QFont.Weight.Normal)

    sample_blocks = [
        (435.0, 198.0, "3. Appellate Jurisdiction (Articles 132-136)", True),
        (455.0, 236.0, "The Supreme Court is the highest appellate authority in India.", False),
        (455.0, 266.0, "It hears appeals in constitutional, civil and criminal matters.", False),
        (435.0, 358.0, "4. Advisory Jurisdiction (Article 143)", True),
        (455.0, 402.0, "The opinion is advisory and is not binding on the President.", False),
        (435.0, 490.0, "5. Court of Record (Article 129)", True),
        (455.0, 532.0, "Its judgments serve as important legal precedents.", False),
    ]
    painter.save()
    for text_x, text_y, line_text, is_heading in sample_blocks:
        if text_x > width - 120:
            continue
        painter.setFont(heading_font if is_heading else body_font)
        painter.setPen(QColor(235, 240, 248, 225) if is_heading else QColor(198, 205, 218, 185))
        painter.drawText(QPointF(text_x, text_y), line_text)
    painter.restore()


def save_visual_snapshots(output_directory: str) -> List[str]:
    """
    Renders three high-DPI reference verification snapshots:
    1. `cursor_states.png` (normal triangle + bubble, listening bars, thinking dots)
    2. `draw_to_ask_highlight.png` (Catmull-Rom glowing cyan-blue Draw-to-Ask stroke)
    3. `drawing_demo.png` (paper card, underlines, ring, point, Clear pill, caption)
    """
    os.makedirs(output_directory, exist_ok=True)
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication([])

    settings_store = jarvis_config.SettingsStore()
    overlay = OverlayCanvasWindow(settings_store)
    saved_paths: List[str] = []
    hd_scale = 2.0

    # 1. cursor_states.png (rendered at 2x Retina HD resolution)
    states_width, states_height = 960, 420
    states_image = QImage(
        int(states_width * hd_scale),
        int(states_height * hd_scale),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    states_image.setDevicePixelRatio(hd_scale)
    states_image.fill(Qt.GlobalColor.transparent)
    states_painter = QPainter(states_image)
    states_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    states_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    states_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    states_painter.fillRect(0, 0, states_width, states_height, QColor("#16181E"))

    label_font = jarvis_theme.ui_font(12.0, QFont.Weight.DemiBold)
    sub_font = jarvis_theme.ui_font(9.5, QFont.Weight.Normal)
    panel_titles = [
        ("1. Normal + Bubble", "Clean rounded -35 deg triangle (no glow)"),
        ("2. Listening", "5 crisp cyan bars (audio_level=0.72)"),
        ("3. Thinking", "3 crisp hopping dots"),
    ]
    panel_width = states_width / 3.0
    for panel_index, (title_text, subtitle_text) in enumerate(panel_titles):
        panel_rect = QRectF(panel_index * panel_width + 16.0, 16.0, panel_width - 32.0, states_height - 32.0)
        states_painter.setPen(QPen(QColor(255, 255, 255, 28), 1.0))
        states_painter.setBrush(QColor(26, 29, 36, 235))
        states_painter.drawRoundedRect(panel_rect, 14.0, 14.0)
        states_painter.setFont(label_font)
        states_painter.setPen(QColor(245, 247, 252))
        states_painter.drawText(QPointF(panel_rect.left() + 18.0, panel_rect.top() + 34.0), title_text)
        states_painter.setFont(sub_font)
        states_painter.setPen(QColor(162, 168, 180))
        states_painter.drawText(QPointF(panel_rect.left() + 18.0, panel_rect.top() + 54.0), subtitle_text)

    # Panel 1: Normal triangle + message bubble
    overlay.set_bubble_message("Right here! Article 143 lets the President consult the Court.", word_by_word=False, hold_seconds=5.0)
    overlay._paint_cursor_triangle(states_painter, QPointF(48.0, 175.0), opacity=1.0, now_seconds=1.0)
    overlay._paint_cursor_bubble(
        states_painter,
        QPointF(48.0, 175.0),
        opacity=1.0,
        bounds=QRectF(26.0, 26.0, panel_width - 52.0, states_height - 52.0),
    )

    # Panel 2: Listening bars
    overlay._smoothed_audio_level = 0.72
    overlay._paint_cursor_listening_bars(
        states_painter,
        QPointF(panel_width + panel_width * 0.5, 210.0),
        opacity=1.0,
        now_seconds=1.35,
    )

    # Panel 3: Thinking dots
    overlay._paint_cursor_thinking_dots(
        states_painter,
        QPointF(panel_width * 2.0 + panel_width * 0.5, 210.0),
        opacity=1.0,
        now_seconds=1.42,
    )
    states_painter.end()

    cursor_states_path = os.path.join(output_directory, "cursor_states.png")
    states_image.save(cursor_states_path)
    saved_paths.append(cursor_states_path)

    # 2. draw_to_ask_highlight.png (rendered at 2x Retina HD resolution)
    highlight_width, highlight_height = 1280, 720
    highlight_image = QImage(
        int(highlight_width * hd_scale),
        int(highlight_height * hd_scale),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    highlight_image.setDevicePixelRatio(hd_scale)
    highlight_painter = QPainter(highlight_image)
    highlight_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    highlight_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    highlight_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    _paint_dark_snapshot_backdrop(highlight_painter, highlight_width, highlight_height)

    overlay.clear_all_drawings()
    overlay.clear_caption()
    overlay._clear_bubble_state()
    overlay._is_highlighting = True
    overlay._highlight_points = [
        (560.0, 290.0),
        (680.0, 272.0),
        (825.0, 286.0),
        (890.0, 345.0),
        (875.0, 420.0),
        (740.0, 448.0),
        (590.0, 434.0),
        (535.0, 360.0),
    ]
    overlay._highlight_cached_path = None
    overlay._paint_highlight_stroke(highlight_painter, time.monotonic())
    overlay._paint_cursor_triangle(highlight_painter, QPointF(535.0, 360.0), opacity=1.0, now_seconds=1.0)
    highlight_painter.end()

    highlight_path = os.path.join(output_directory, "draw_to_ask_highlight.png")
    highlight_image.save(highlight_path)
    saved_paths.append(highlight_path)

    # 3. drawing_demo.png (rendered at 2x Retina HD resolution)
    demo_width = max(1280, overlay.width())
    demo_height = max(720, overlay.height())
    demo_image = QImage(
        int(demo_width * hd_scale),
        int(demo_height * hd_scale),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    demo_image.setDevicePixelRatio(hd_scale)
    demo_painter = QPainter(demo_image)
    demo_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    demo_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    demo_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    _paint_dark_snapshot_backdrop(demo_painter, demo_width, demo_height)

    overlay._is_highlighting = False
    overlay._highlight_points.clear()
    overlay.load_demo_showcase()
    # Advance springs and animation clock so showcase renders cleanly settled
    settled_time = time.monotonic() + 4.8
    scale_x = demo_width / 1536.0
    scale_y = demo_height / 864.0
    overlay._cursor_x_spring.snap(892.0 * scale_x + 8.0)
    overlay._cursor_y_spring.snap(502.0 * scale_y + 10.0)
    overlay._cursor_opacity_spring.snap(1.0)
    overlay._listening_morph_spring.snap(0.0)
    overlay._thinking_morph_spring.snap(0.0)
    overlay._clear_bubble_state()

    overlay._render_canvas(demo_painter, QRectF(0.0, 0.0, float(demo_width), float(demo_height)), settled_time)

    clear_pill_rect = QRectF(
        float(demo_width - CLEAR_BUTTON_OFFSET_FROM_RIGHT),
        float(CLEAR_BUTTON_TOP_Y),
        CLEAR_BUTTON_PILL_WIDTH,
        CLEAR_BUTTON_PILL_HEIGHT,
    )
    _paint_clear_button_pill(demo_painter, clear_pill_rect)
    demo_painter.end()

    demo_path = os.path.join(output_directory, "drawing_demo.png")
    demo_image.save(demo_path)
    saved_paths.append(demo_path)

    overlay.close()
    return saved_paths


def _parse_cli_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis PySide6 overlay window & visual snapshot tool.")
    parser.add_argument(
        "--snapshot",
        type=str,
        default="",
        help="Directory where cursor_states.png, draw_to_ask_highlight.png, and drawing_demo.png are saved.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_cli_arguments(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    settings_store = jarvis_config.SettingsStore()
    overlay = OverlayCanvasWindow(settings_store)

    if args.snapshot:
        saved_files = save_visual_snapshots(args.snapshot)
        for file_path in saved_files:
            print(f"SAVED: {file_path}")
        metrics = overlay.benchmark_frame_ms(iterations=60)
        print(f"BENCHMARK: {metrics}")
        overlay.close()
        return 0

    overlay.load_demo_showcase()
    overlay.show()
    metrics = overlay.benchmark_frame_ms(iterations=60)
    print(f"BENCHMARK: {metrics}")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
