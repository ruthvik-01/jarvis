"""
Visual renderers and floating mini-windows for full KenKaiii/Roamies feature parity.

Includes:
- Jarvis3DModelRenderer (NumPy + QPainter 3D engine with glass enclosing parts & spin/orbit/slide/pulse)
- Painters for molecule, graph, map, geometry, sources, clip, legend, ticket, medal & photo pins
- All 8 Roamies companion looks + JARVIS 3D LED sphere + emotion feedback & cursor reticle mode
- CursorAskBarWindow (Ctrl+Shift cursor-adjacent composer)
- RiskyActionApprovalBar (Allow / Don't allow confirmation bar)
- AgentArtifactToastWindow (clickable subagent output notification toast)
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import jarvis_roamies_core


TONE_COLOR_MAP: Dict[str, QColor] = {
    "accent": QColor("#3380FF"),
    "blue": QColor("#3380FF"),
    "teal": QColor("#78FFE2"),
    "amber": QColor("#FFB84D"),
    "coral": QColor("#FF7A59"),
    "lilac": QColor("#C69CFF"),
    "sky": QColor("#66D9EF"),
    "lime": QColor("#7CE38B"),
    "green": QColor("#34D399"),
    "violet": QColor("#C69CFF"),
    "purple": QColor("#C69CFF"),
    "yellow": QColor("#FBBF24"),
    "orange": QColor("#FB923C"),
    "red": QColor("#FF5252"),
    "white": QColor("#EDF4FF"),
}

_ELEMENT_COLOR_MAP: Dict[str, QColor] = {
    "C": QColor("#A5C4FF"),
    "H": QColor("#EDF4FF"),
    "O": QColor("#FF6B6B"),
    "N": QColor("#66D9EF"),
    "S": QColor("#FFD166"),
    "P": QColor("#FFB84D"),
    "F": QColor("#7CE38B"),
    "Cl": QColor("#7CE38B"),
    "Br": QColor("#D97757"),
}


# ---------------------------------------------------------------------------
# 1. 3D Mesh Builder & QPainter Depth-Sorted Renderer (ink3d.js equivalent)
# ---------------------------------------------------------------------------

def _build_mesh_arrays(shape_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Builds `(vertices_Nx3x3, face_normals_Nx3)` for box, sphere, cylinder, cone, torus."""
    kind = str(shape_name or "box").strip().lower()
    tris: List[List[Tuple[float, float, float]]] = []
    if kind == "box":
        for axis_idx, sign_val in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)):
            axis_u = (axis_idx + 1) % 3
            axis_v = (axis_idx + 2) % 3
            corners: List[Tuple[float, float, float]] = []
            for coef_u, coef_v in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
                pt = [0.0, 0.0, 0.0]
                pt[axis_idx] = 0.5 * sign_val
                pt[axis_u] = coef_u
                pt[axis_v] = coef_v * sign_val
                corners.append((pt[0], pt[1], pt[2]))
            tris.append([corners[0], corners[1], corners[2]])
            tris.append([corners[0], corners[2], corners[3]])
    elif kind == "sphere":
        seg_u, seg_v = 14, 9
        for idx_u in range(seg_u):
            for idx_v in range(seg_v):
                quad_pts: List[Tuple[float, float, float]] = []
                for step_u, step_v in ((idx_u, idx_v), (idx_u + 1, idx_v), (idx_u + 1, idx_v + 1), (idx_u, idx_v + 1)):
                    ang_t = (step_u / seg_u) * 2.0 * math.pi
                    ang_p = (step_v / seg_v) * math.pi
                    quad_pts.append((
                        0.5 * math.sin(ang_p) * math.cos(ang_t),
                        0.5 * math.cos(ang_p),
                        0.5 * math.sin(ang_p) * math.sin(ang_t),
                    ))
                tris.append([quad_pts[0], quad_pts[1], quad_pts[2]])
                tris.append([quad_pts[0], quad_pts[2], quad_pts[3]])
    elif kind in ("cylinder", "cone"):
        seg_u = 16
        top_r = 0.0 if kind == "cone" else 0.5
        for idx_u in range(seg_u):
            ang_a = (idx_u / seg_u) * 2.0 * math.pi
            ang_b = ((idx_u + 1) / seg_u) * 2.0 * math.pi
            bot_a = (0.5 * math.cos(ang_a), -0.5, 0.5 * math.sin(ang_a))
            bot_b = (0.5 * math.cos(ang_b), -0.5, 0.5 * math.sin(ang_b))
            top_a = (top_r * math.cos(ang_a), 0.5, top_r * math.sin(ang_a))
            top_b = (top_r * math.cos(ang_b), 0.5, top_r * math.sin(ang_b))
            tris.append([bot_a, bot_b, top_b])
            if top_r > 0.0:
                tris.append([bot_a, top_b, top_a])
                tris.append([(0.0, 0.5, 0.0), top_b, top_a])
            tris.append([(0.0, -0.5, 0.0), bot_a, bot_b])
    elif kind == "torus":
        seg_major, seg_minor = 18, 9
        major_r, minor_r = 0.36, 0.14
        for idx_u in range(seg_major):
            for idx_v in range(seg_minor):
                quad_pts = []
                for step_u, step_v in ((idx_u, idx_v), (idx_u + 1, idx_v), (idx_u + 1, idx_v + 1), (idx_u, idx_v + 1)):
                    ang_t = (step_u / seg_major) * 2.0 * math.pi
                    ang_f = (step_v / seg_minor) * 2.0 * math.pi
                    ring_cx, ring_cz = math.cos(ang_t), math.sin(ang_t)
                    quad_pts.append((
                        ring_cx * (major_r + minor_r * math.cos(ang_f)),
                        minor_r * math.sin(ang_f),
                        ring_cz * (major_r + minor_r * math.cos(ang_f)),
                    ))
                tris.append([quad_pts[0], quad_pts[1], quad_pts[2]])
                tris.append([quad_pts[0], quad_pts[2], quad_pts[3]])
    else:
        return _build_mesh_arrays("box")

    vert_arr = np.asarray(tris, dtype=np.float32)
    edge1 = vert_arr[:, 1, :] - vert_arr[:, 0, :]
    edge2 = vert_arr[:, 2, :] - vert_arr[:, 0, :]
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    return vert_arr, normals


_MESH_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _get_cached_mesh(shape_name: str) -> Tuple[np.ndarray, np.ndarray]:
    key = str(shape_name or "box").strip().lower()
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = _build_mesh_arrays(key)
    return _MESH_CACHE[key]


def resolve_3d_preset_or_parts(spec: Any) -> List[Dict[str, Any]]:
    """Parses JSON parts array or expands built-in named 3D demo presets."""
    if isinstance(spec, list):
        return [item for item in spec if isinstance(item, dict)][:40]
    if isinstance(spec, dict) and isinstance(spec.get("parts"), list):
        return [item for item in spec["parts"] if isinstance(item, dict)][:40]
    raw_text = str(spec or "").strip()
    if raw_text.startswith("[") or raw_text.startswith("{"):
        try:
            parsed = json.loads(raw_text)
            return resolve_3d_preset_or_parts(parsed)
        except Exception:
            pass
    key = raw_text.lower()
    if "piston" in key or "engine" in key:
        return [
            {"shape": "cylinder", "at": [0.0, 0.0, 0.0], "size": [1.15, 1.9, 1.15], "tone": "sky", "label": "Cylinder"},
            {"shape": "cylinder", "at": [0.0, -0.35, 0.0], "size": [0.78, 0.55, 0.78], "tone": "amber", "label": "Piston", "motion": {"do": "slide", "to": [0.0, 0.48, 0.0], "secs": 1.8}},
            {"shape": "box", "at": [0.0, -0.82, 0.0], "size": [0.22, 0.75, 0.22], "tone": "teal", "motion": {"do": "slide", "to": [0.0, -0.05, 0.0], "secs": 1.8}},
        ]
    if "atom" in key or "electron" in key or "nucleus" in key:
        return [
            {"shape": "sphere", "at": [0.0, 0.0, 0.0], "size": 0.65, "tone": "coral", "label": "Nucleus", "motion": {"do": "pulse", "secs": 1.8}},
            {"shape": "torus", "at": [0.0, 0.0, 0.0], "size": [1.8, 1.8, 1.8], "rot": [55, 0, 20], "tone": "teal", "motion": {"do": "spin", "axis": "y", "secs": 5.0}},
            {"shape": "sphere", "at": [0.78, 0.0, 0.0], "size": 0.22, "tone": "accent", "label": "e-", "motion": {"do": "orbit", "around": [0.0, 0.0, 0.0], "axis": "y", "secs": 2.4}},
        ]
    if "black" in key or "hole" in key:
        return [
            {"shape": "sphere", "at": [0.0, 0.0, 0.0], "size": 0.85, "tone": "lilac", "label": "Event Horizon"},
            {"shape": "torus", "at": [0.0, 0.0, 0.0], "size": [2.2, 0.38, 2.2], "rot": [12, 0, 8], "tone": "amber", "label": "Accretion Disk", "motion": {"do": "spin", "axis": "y", "secs": 3.5}},
            {"shape": "cone", "at": [0.0, 0.95, 0.0], "size": [0.3, 0.95, 0.3], "tone": "sky", "label": "Relativistic Jet"},
        ]
    # Default 3D solar/gyro assembly
    return [
        {"shape": "sphere", "at": [0.0, 0.0, 0.0], "size": 0.72, "tone": "amber", "label": "Core", "motion": {"do": "pulse", "secs": 2.4}},
        {"shape": "torus", "at": [0.0, 0.0, 0.0], "size": [1.75, 1.75, 1.75], "rot": [35, 0, 15], "tone": "accent", "motion": {"do": "spin", "axis": "y", "secs": 4.2}},
        {"shape": "sphere", "at": [0.85, 0.0, 0.0], "size": 0.26, "tone": "teal", "label": "Satellite", "motion": {"do": "orbit", "axis": "y", "secs": 3.2}},
    ]


class Jarvis3DModelRenderer:
    """Fast vectorized 3D projection engine rendering animated multi-part models with glass casings."""

    def __init__(self, parts_spec: Any) -> None:
        self._raw_parts = resolve_3d_preset_or_parts(parts_spec)
        self._items: List[Dict[str, Any]] = []
        for part in self._raw_parts:
            shape = str(part.get("shape", "box")).lower()
            verts, norms = _get_cached_mesh(shape)
            at_pos = np.asarray(part.get("at") or [0.0, 0.0, 0.0], dtype=np.float32)
            raw_size = part.get("size", 0.6)
            if isinstance(raw_size, (int, float)):
                size_vec = np.asarray([float(raw_size), float(raw_size), float(raw_size)], dtype=np.float32)
            else:
                size_list = list(raw_size)[:3] + [0.6, 0.6, 0.6]
                size_vec = np.asarray(size_list[:3], dtype=np.float32)
            rot_deg = list(part.get("rot") or [0.0, 0.0, 0.0])[:3] + [0.0, 0.0, 0.0]
            rot_mat = self._euler_mat(
                math.radians(float(rot_deg[0])),
                math.radians(float(rot_deg[1])),
                math.radians(float(rot_deg[2])),
            )
            tone_key = str(part.get("tone", "accent")).lower()
            base_color = TONE_COLOR_MAP.get(tone_key, TONE_COLOR_MAP["accent"])
            self._items.append({
                "verts": verts,
                "norms": norms,
                "at": at_pos,
                "size": size_vec,
                "rot_mat": rot_mat,
                "color": base_color,
                "label": str(part.get("label", "") or "").strip(),
                "motion": part.get("motion") if isinstance(part.get("motion"), dict) else None,
                "radius": float(np.max(size_vec)),
                "glass": False,
            })

        # Enclosing static container around smaller parts -> transparent glass so moving interior stays visible
        for idx_outer, outer in enumerate(self._items):
            if outer["motion"] is not None:
                continue
            for idx_inner, inner in enumerate(self._items):
                if idx_outer == idx_inner or inner["radius"] >= outer["radius"]:
                    continue
                delta = np.abs(inner["at"] - outer["at"])
                if np.all(delta < outer["size"] * 0.55):
                    outer["glass"] = True
                    break

        if self._items:
            centers = np.asarray([item["at"] for item in self._items], dtype=np.float32)
            self._scene_center = centers.mean(axis=0)
            max_reach = max(
                float(np.linalg.norm(item["at"] - self._scene_center) + item["radius"] * 0.72)
                for item in self._items
            )
            self._scene_scale = 1.0 / max(0.35, max_reach)
        else:
            self._scene_center = np.zeros(3, dtype=np.float32)
            self._scene_scale = 1.0

    @staticmethod
    def _rot_x(ang: float) -> np.ndarray:
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        return np.asarray([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]], dtype=np.float32)

    @staticmethod
    def _rot_y(ang: float) -> np.ndarray:
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        return np.asarray([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]], dtype=np.float32)

    @staticmethod
    def _rot_z(ang: float) -> np.ndarray:
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        return np.asarray([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    @classmethod
    def _euler_mat(cls, rx: float, ry: float, rz: float) -> np.ndarray:
        return cls._rot_z(rz) @ cls._rot_y(ry) @ cls._rot_x(rx)

    def paint_into_rect(self, painter: QPainter, rect: QRectF, anim_time: float) -> None:
        if not self._items or rect.width() < 24 or rect.height() < 24:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Background viewport card frame
        painter.setPen(QPen(QColor(115, 165, 255, 80), 1.0))
        painter.setBrush(QColor(8, 16, 30, 195))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        yaw = anim_time * 0.45 + 0.55
        view_rot = self._rot_x(0.36) @ self._rot_y(yaw)
        light_dir = np.asarray([-0.42, -0.70, 0.58], dtype=np.float32)
        light_dir = light_dir / max(1e-6, float(np.linalg.norm(light_dir)))

        cx = rect.center().x()
        cy = rect.center().y()
        proj_scale = min(rect.width(), rect.height()) * 0.40
        draw_tris: List[Tuple[float, QPolygonF, QColor, bool]] = []
        label_points: List[Tuple[float, float, str, QColor]] = []

        default_cycles = {"spin": 4.0, "orbit": 6.0, "slide": 2.0, "pulse": 1.6}
        for item in self._items:
            part_pos = item["at"].copy()
            part_size = item["size"].copy()
            part_rot = item["rot_mat"]
            motion = item["motion"]
            if motion:
                do_mode = str(motion.get("do", "spin")).lower()
                secs = max(0.3, min(60.0, float(motion.get("secs") or default_cycles.get(do_mode, 3.0))))
                phase = (anim_time / secs) * 2.0 * math.pi
                axis_name = str(motion.get("axis", "y")).lower()
                axis_fn = {"x": self._rot_x, "y": self._rot_y, "z": self._rot_z}.get(axis_name, self._rot_y)
                if do_mode == "spin":
                    part_rot = axis_fn(phase) @ part_rot
                elif do_mode == "orbit":
                    around = np.asarray(motion.get("around") or [0.0, 0.0, 0.0], dtype=np.float32)
                    orb_mat = axis_fn(phase)
                    part_pos = around + (orb_mat @ (part_pos - around))
                    part_rot = orb_mat @ part_rot
                elif do_mode == "slide":
                    target_to = np.asarray(motion.get("to") or part_pos, dtype=np.float32)
                    slide_k = 0.5 * (1.0 - math.cos(phase))
                    part_pos = part_pos + (target_to - part_pos) * slide_k
                elif do_mode == "pulse":
                    part_size = part_size * (1.0 + 0.14 * math.sin(phase))

            local_scaled = item["verts"] * part_size[None, None, :]
            world_verts = np.einsum("ij,mnj->mni", part_rot, local_scaled) + part_pos[None, None, :]
            centered = (world_verts - self._scene_center[None, None, :]) * self._scene_scale
            cam_verts = np.einsum("ij,mnj->mni", view_rot, centered)

            world_norms = np.einsum("ij,mj->mi", part_rot, item["norms"])
            cam_norms = np.einsum("ij,mj->mi", view_rot, world_norms)

            mean_z = cam_verts[:, :, 2].mean(axis=1)
            denom = np.maximum(0.6, 3.2 - cam_verts[:, :, 2])
            screen_x = cx + (cam_verts[:, :, 0] / denom) * proj_scale * 2.15
            screen_y = cy - (cam_verts[:, :, 1] / denom) * proj_scale * 2.15

            lambert = np.clip(cam_norms @ light_dir, 0.0, 1.0)
            base_col: QColor = item["color"]
            is_glass = bool(item["glass"])

            for tri_idx in range(cam_verts.shape[0]):
                tri_z = float(mean_z[tri_idx])
                if (not is_glass) and cam_norms[tri_idx, 2] < -0.15:
                    continue
                shade = 0.34 + 0.66 * float(lambert[tri_idx])
                poly = QPolygonF([
                    QPointF(float(screen_x[tri_idx, 0]), float(screen_y[tri_idx, 0])),
                    QPointF(float(screen_x[tri_idx, 1]), float(screen_y[tri_idx, 1])),
                    QPointF(float(screen_x[tri_idx, 2]), float(screen_y[tri_idx, 2])),
                ])
                fill_col = QColor(
                    min(255, int(base_col.red() * shade + 18)),
                    min(255, int(base_col.green() * shade + 20)),
                    min(255, int(base_col.blue() * shade + 24)),
                    62 if is_glass else 225,
                )
                draw_tris.append((tri_z, poly, fill_col, is_glass))

            if item["label"]:
                part_cam = view_rot @ ((part_pos - self._scene_center) * self._scene_scale)
                denom_pt = max(0.6, 3.2 - float(part_cam[2]))
                lbl_x = cx + (float(part_cam[0]) / denom_pt) * proj_scale * 2.15
                lbl_y = cy - (float(part_cam[1]) / denom_pt) * proj_scale * 2.15
                label_points.append((lbl_x, lbl_y, item["label"], base_col))

        draw_tris.sort(key=lambda entry: entry[0])
        for _tri_z, poly, fill_col, is_glass in draw_tris:
            if is_glass:
                painter.setPen(QPen(QColor(fill_col.red(), fill_col.green(), fill_col.blue(), 120), 0.7))
            else:
                painter.setPen(QPen(QColor(15, 25, 42, 70), 0.5))
            painter.setBrush(fill_col)
            painter.drawPolygon(poly)

        lbl_font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(lbl_font)
        placed_tag_rects: List[QRectF] = []
        for lbl_x, lbl_y, lbl_text, lbl_col in label_points:
            tag_rect = QRectF(lbl_x - 36.0, lbl_y - 22.0, 72.0, 17.0)
            for prev_rect in placed_tag_rects:
                if tag_rect.intersects(prev_rect):
                    tag_rect.translate(0.0, 22.0)
            placed_tag_rects.append(tag_rect)
            painter.setPen(QPen(lbl_col, 1.0))
            painter.setBrush(QColor(10, 18, 32, 210))
            painter.drawRoundedRect(tag_rect, 4.0, 4.0)
            painter.setPen(QColor(235, 245, 255))
            painter.drawText(tag_rect, Qt.AlignCenter, lbl_text[:16])

        painter.restore()


# ---------------------------------------------------------------------------
# 2. Card Painters (Molecule, Graph, Map, Geometry, Sources, Clip, Legend, Ticket, Medal, Photo Pins)
# ---------------------------------------------------------------------------

def paint_molecule_row(painter: QPainter, rect: QRectF, smiles_str: str, accent_color: QColor) -> None:
    layout = jarvis_roamies_core.parse_smiles_to_2d(smiles_str)
    atoms = layout.get("atoms", [])
    bonds = layout.get("bonds", [])
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(QColor(115, 165, 255, 75), 1.0))
    painter.setBrush(QColor(10, 20, 36, 180))
    painter.drawRoundedRect(rect, 7.0, 7.0)
    if not atoms:
        painter.setPen(QColor(190, 210, 240))
        painter.drawText(rect, Qt.AlignCenter, smiles_str[:40] or "Molecule")
        painter.restore()
        return

    def _pt(atom_dict: Dict[str, Any]) -> QPointF:
        inner_pad_x = min(28.0, rect.width() * 0.14)
        inner_pad_y = min(22.0, rect.height() * 0.22)
        usable_w = max(24.0, rect.width() - inner_pad_x * 2.0)
        usable_h = max(20.0, rect.height() - inner_pad_y * 2.0)
        return QPointF(
            rect.left() + inner_pad_x + atom_dict["nx"] * usable_w,
            rect.top() + inner_pad_y + atom_dict["ny"] * usable_h,
        )

    for idx_a, idx_b, order in bonds:
        pt_a = _pt(atoms[idx_a])
        pt_b = _pt(atoms[idx_b])
        dx_val = pt_b.x() - pt_a.x()
        dy_val = pt_b.y() - pt_a.y()
        seg_len = max(1.0, math.hypot(dx_val, dy_val))
        perp_x = -dy_val / seg_len * 3.0
        perp_y = dx_val / seg_len * 3.0
        bond_pen = QPen(QColor(195, 220, 255, 215), 2.0, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(bond_pen)
        if order == 2:
            painter.drawLine(QPointF(pt_a.x() + perp_x, pt_a.y() + perp_y), QPointF(pt_b.x() + perp_x, pt_b.y() + perp_y))
            painter.drawLine(QPointF(pt_a.x() - perp_x, pt_a.y() - perp_y), QPointF(pt_b.x() - perp_x, pt_b.y() - perp_y))
        elif order == 3:
            painter.drawLine(pt_a, pt_b)
            painter.drawLine(QPointF(pt_a.x() + perp_x * 1.3, pt_a.y() + perp_y * 1.3), QPointF(pt_b.x() + perp_x * 1.3, pt_b.y() + perp_y * 1.3))
            painter.drawLine(QPointF(pt_a.x() - perp_x * 1.3, pt_a.y() - perp_y * 1.3), QPointF(pt_b.x() - perp_x * 1.3, pt_b.y() - perp_y * 1.3))
        elif order == 4:
            painter.drawLine(pt_a, pt_b)
            painter.setPen(QPen(accent_color, 1.4, Qt.DashLine))
            painter.drawLine(QPointF(pt_a.x() + perp_x, pt_a.y() + perp_y), QPointF(pt_b.x() + perp_x, pt_b.y() + perp_y))
        else:
            painter.drawLine(pt_a, pt_b)

    atom_font = QFont("Segoe UI", 8, QFont.Bold)
    painter.setFont(atom_font)
    for atom_dict in atoms:
        center_pt = _pt(atom_dict)
        elem = str(atom_dict.get("element", "C"))
        col = _ELEMENT_COLOR_MAP.get(elem, accent_color)
        painter.setPen(QPen(col, 1.4))
        painter.setBrush(QColor(12, 22, 40, 240))
        painter.drawEllipse(center_pt, 10.0, 10.0)
        painter.setPen(col)
        painter.drawText(QRectF(center_pt.x() - 10.0, center_pt.y() - 9.0, 20.0, 18.0), Qt.AlignCenter, elem)
    painter.restore()


def parse_graph_spec_from_text(raw_text: str) -> Dict[str, Any]:
    """Parses either JSON graph spec or Roamies shorthand `expr=x**2|x_range=-3,3|caption=Parabola` into sampled points."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("{"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and (parsed.get("plots") or parsed.get("dots")):
                return parsed
        except Exception:
            pass
    expr_list: List[str] = []
    x_min_val = -3.0
    x_max_val = 3.0
    dots_list: List[Dict[str, Any]] = []
    for segment in cleaned.split("|"):
        seg_clean = segment.strip()
        if not seg_clean:
            continue
        seg_lower = seg_clean.lower()
        if seg_lower.startswith(("expr=", "fn=", "formula=", "y=")):
            raw_expr = seg_clean.split("=", 1)[1].strip()
            for sub_expr in raw_expr.split(";"):
                if sub_expr.strip():
                    expr_list.append(sub_expr.strip())
        elif seg_lower.startswith(("x_range=", "xrange=", "domain=")):
            nums = [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", seg_clean.split("=", 1)[1])]
            if len(nums) >= 2 and nums[1] > nums[0]:
                x_min_val, x_max_val = nums[0], nums[1]
        elif seg_lower.startswith("points="):
            pts_raw = seg_clean.split("=", 1)[1].strip()
            sampled_pts: List[List[float]] = []
            for pair_str in re.split(r"[;\s]+", pts_raw):
                pair_nums = [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", pair_str)]
                if len(pair_nums) >= 2:
                    sampled_pts.append([pair_nums[0], pair_nums[1]])
            if len(sampled_pts) >= 2:
                return {"plots": [{"points": sampled_pts, "tone": "accent"}]}
        elif ("=" not in seg_clean) and any(tok in seg_lower for tok in ("x", "sin", "cos", "tan", "exp", "log", "^", "**")):
            expr_list.append(seg_clean)

    if not expr_list:
        expr_list = ["x**2"]

    safe_math_env = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": lambda val: math.sqrt(max(0.0, float(val))),
        "abs": abs,
        "exp": lambda val: math.exp(min(12.0, max(-12.0, float(val)))),
        "log": lambda val: math.log(max(1e-4, float(val))),
        "pi": math.pi,
        "e": math.e,
    }
    tones_cycle = ("accent", "teal", "amber", "violet")
    plots_out: List[Dict[str, Any]] = []
    sample_count = 41
    for expr_idx, expr_str in enumerate(expr_list[:3]):
        py_expr = expr_str.replace("^", "**")
        py_expr = re.sub(r"(\d)([a-zA-Z\(])", r"\1*\2", py_expr)
        if re.search(r"[^0-9xX+\-*/().\s_a-zA-Z]", py_expr):
            continue
        sampled_points: List[List[float]] = []
        for step_idx in range(sample_count):
            frac = step_idx / float(sample_count - 1)
            x_val = x_min_val + frac * (x_max_val - x_min_val)
            try:
                local_vars = dict(safe_math_env)
                local_vars["x"] = x_val
                local_vars["X"] = x_val
                y_val = float(eval(py_expr, {"__builtins__": {}}, local_vars))
                if math.isfinite(y_val):
                    sampled_points.append([round(x_val, 4), round(y_val, 4)])
            except Exception:
                continue
        if len(sampled_points) >= 2:
            plots_out.append({"points": sampled_points, "tone": tones_cycle[expr_idx % len(tones_cycle)]})
            if expr_idx == 0 and any(abs(pt[0]) < 0.08 for pt in sampled_points):
                zero_pt = min(sampled_points, key=lambda pt: abs(pt[0]))
                dots_out_label = f"({zero_pt[0]:.0f},{zero_pt[1]:.1f})".replace(".0)", ")")
                dots_list.append({"x": zero_pt[0], "y": zero_pt[1], "label": dots_out_label})

    if not plots_out:
        plots_out = [{"points": [[-2.0, 4.0], [-1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 4.0]], "tone": "accent"}]
    return {"plots": plots_out, "dots": dots_list, "x_range": [x_min_val, x_max_val]}


def parse_geometry_spec_from_text(raw_text: str) -> Dict[str, Any]:
    """Parses either JSON geometry spec or shorthand `shapes=triangle 0,0 100,0 50,80` into vertices/lines/circles."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("{"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and parsed.get("vertices"):
                return parsed
        except Exception:
            pass
    coord_pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", cleaned)
    if len(coord_pairs) >= 3:
        letters = "ABCDEFGH"
        vertices = [
            {"id": letters[idx % len(letters)], "x": float(pair_xy[0]), "y": float(pair_xy[1])}
            for idx, pair_xy in enumerate(coord_pairs[:6])
        ]
        lines = [
            {
                "from": vertices[idx]["id"],
                "to": vertices[(idx + 1) % len(vertices)]["id"],
                "label": f"{math.hypot(vertices[(idx + 1) % len(vertices)]['x'] - vertices[idx]['x'], vertices[(idx + 1) % len(vertices)]['y'] - vertices[idx]['y']):.0f}",
            }
            for idx in range(len(vertices))
        ]
        return {"vertices": vertices, "lines": lines}
    return {
        "vertices": [{"id": "A", "x": 0.0, "y": 0.0}, {"id": "B", "x": 4.0, "y": 0.0}, {"id": "C", "x": 2.0, "y": 3.0}],
        "lines": [{"from": "A", "to": "B", "label": "4"}, {"from": "B", "to": "C"}, {"from": "C", "to": "A"}],
    }


def paint_graph_row(painter: QPainter, rect: QRectF, graph_spec: Dict[str, Any], accent_color: QColor) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(QColor(115, 165, 255, 75), 1.0))
    painter.setBrush(QColor(10, 20, 36, 185))
    painter.drawRoundedRect(rect, 7.0, 7.0)

    inner = rect.adjusted(26.0, 14.0, -14.0, -22.0)
    plots = graph_spec.get("plots") or []
    dots = graph_spec.get("dots") or []
    all_pts = [pt for pl in plots for pt in (pl.get("points") or [])] + [[dot_entry.get("x", 0.0), dot_entry.get("y", 0.0)] for dot_entry in dots]
    if not all_pts:
        painter.setPen(QColor(190, 210, 240))
        painter.drawText(rect, Qt.AlignCenter, "2D Plot")
        painter.restore()
        return

    x_range = graph_spec.get("x_range")
    y_range = graph_spec.get("y_range")
    min_x = float(x_range[0]) if x_range else min(float(pt[0]) for pt in all_pts)
    max_x = float(x_range[1]) if x_range else max(float(pt[0]) for pt in all_pts)
    min_y = float(y_range[0]) if y_range else min(float(pt[1]) for pt in all_pts)
    max_y = float(y_range[1]) if y_range else max(float(pt[1]) for pt in all_pts)
    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)

    def _to_px(vx: float, vy: float) -> QPointF:
        return QPointF(
            inner.left() + ((vx - min_x) / span_x) * inner.width(),
            inner.bottom() - ((vy - min_y) / span_y) * inner.height(),
        )

    painter.setPen(QPen(QColor(140, 175, 225, 70), 1.0))
    painter.drawLine(inner.topLeft(), inner.bottomLeft())
    painter.drawLine(inner.bottomLeft(), inner.bottomRight())

    for pl_idx, plot_item in enumerate(plots):
        pts_raw = plot_item.get("points") or []
        if len(pts_raw) < 2:
            continue
        col = TONE_COLOR_MAP.get(str(plot_item.get("tone", "")).lower(), accent_color if pl_idx == 0 else TONE_COLOR_MAP["teal"])
        path = QPainterPath()
        first_px = _to_px(float(pts_raw[0][0]), float(pts_raw[0][1]))
        path.moveTo(first_px)
        for pt_raw in pts_raw[1:]:
            path.lineTo(_to_px(float(pt_raw[0]), float(pt_raw[1])))
        painter.setPen(QPen(col, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    lbl_font = QFont("Segoe UI", 7.5, QFont.Bold)
    painter.setFont(lbl_font)
    for dot_item in dots:
        px = _to_px(float(dot_item.get("x", 0.0)), float(dot_item.get("y", 0.0)))
        painter.setPen(Qt.NoPen)
        painter.setBrush(TONE_COLOR_MAP["amber"])
        painter.drawEllipse(px, 4.0, 4.0)
        dot_lbl = str(dot_item.get("label", "") or "")
        if dot_lbl:
            painter.setPen(QColor(235, 245, 255))
            painter.drawText(QRectF(px.x() + 6.0, px.y() - 9.0, 90.0, 18.0), Qt.AlignLeft | Qt.AlignVCenter, dot_lbl[:18])
    painter.restore()


def paint_map_row(painter: QPainter, rect: QRectF, map_spec: Dict[str, Any], accent_color: QColor, anim_time: float = 0.0) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(QColor(115, 165, 255, 80), 1.0))
    painter.setBrush(QColor(8, 20, 38, 205))
    painter.drawRoundedRect(rect, 7.0, 7.0)

    resolved = jarvis_roamies_core.resolve_map_region(
        region=str(map_spec.get("region", "")),
        shade_list=map_spec.get("shade"),
        pins_list=map_spec.get("pins"),
    )
    west, south, east, north = resolved["bbox"]
    lon_span = max(8.0, east - west)
    lat_span = max(8.0, north - south)
    inner = rect.adjusted(10.0, 10.0, -10.0, -10.0)

    def _proj(lon_val: float, lat_val: float) -> QPointF:
        return QPointF(
            inner.left() + ((lon_val - west) / lon_span) * inner.width(),
            inner.bottom() - ((lat_val - south) / lat_span) * inner.height(),
        )

    # Graticule lines
    painter.setPen(QPen(QColor(95, 140, 200, 36), 0.8, Qt.DashLine))
    for step_idx in range(1, 4):
        gx = inner.left() + inner.width() * (step_idx / 4.0)
        gy = inner.top() + inner.height() * (step_idx / 4.0)
        painter.drawLine(QPointF(gx, inner.top()), QPointF(gx, inner.bottom()))
        painter.drawLine(QPointF(inner.left(), gy), QPointF(inner.right(), gy))

    shade_tones = {str(item.get("name", "")).lower(): str(item.get("tone", "teal")).lower() for item in (map_spec.get("shade") or [])}
    lbl_font = QFont("Segoe UI", 7.5, QFont.Bold)
    painter.setFont(lbl_font)
    for country_row in jarvis_roamies_core.WORLD_COUNTRIES:
        c_west, c_south, c_east, c_north = country_row["b"]
        if c_east < west or c_west > east or c_north < south or c_south > north:
            continue
        top_left = _proj(max(west, c_west), min(north, c_north))
        bot_right = _proj(min(east, c_east), max(south, c_south))
        box = QRectF(top_left, bot_right).normalized()
        c_name_lower = country_row["n"].lower()
        is_shaded = c_name_lower in shade_tones or any(al in shade_tones for al in country_row["a"])
        if is_shaded:
            tone_key = shade_tones.get(c_name_lower, "teal")
            col = TONE_COLOR_MAP.get(tone_key, accent_color)
            painter.setPen(QPen(col, 1.3))
            painter.setBrush(QColor(col.red(), col.green(), col.blue(), 110))
        else:
            painter.setPen(QPen(QColor(110, 165, 225, 70), 0.9))
            painter.setBrush(QColor(48, 92, 148, 65))
        painter.drawRoundedRect(box, 3.0, 3.0)
        if box.width() > 36 and box.height() > 16:
            painter.setPen(QColor(225, 240, 255, 200))
            painter.drawText(box, Qt.AlignCenter, country_row["n"][:14])

    pins_by_name: Dict[str, QPointF] = {}
    for pin_item in (map_spec.get("pins") or []):
        pt = _proj(float(pin_item.get("lon", 0.0)), float(pin_item.get("lat", 0.0)))
        pin_name = str(pin_item.get("name", ""))
        pins_by_name[pin_name] = pt
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 184, 77, 85))
        painter.drawEllipse(pt, 8.0, 8.0)
        painter.setBrush(TONE_COLOR_MAP["amber"])
        painter.drawEllipse(pt, 3.8, 3.8)
        painter.setPen(QColor(245, 250, 255))
        painter.drawText(QRectF(pt.x() + 6.0, pt.y() - 9.0, 95.0, 18.0), Qt.AlignLeft | Qt.AlignVCenter, pin_name[:16])

    for route_item in (map_spec.get("routes") or []):
        pt_from = pins_by_name.get(str(route_item.get("from", "")))
        pt_to = pins_by_name.get(str(route_item.get("to", "")))
        if pt_from is None or pt_to is None:
            continue
        r_col = TONE_COLOR_MAP.get(str(route_item.get("tone", "coral")).lower(), TONE_COLOR_MAP["coral"])
        ctrl_pt = QPointF(0.5 * (pt_from.x() + pt_to.x()), min(pt_from.y(), pt_to.y()) - 18.0)
        arc = QPainterPath(pt_from)
        arc.quadTo(ctrl_pt, pt_to)
        painter.setPen(QPen(r_col, 2.0, Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(arc)
        travel_t = (anim_time * 0.45) % 1.0
        dot_pt = arc.pointAtPercent(travel_t)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(dot_pt, 3.2, 3.2)

    painter.restore()


def paint_geometry_row(painter: QPainter, rect: QRectF, geom_spec: Dict[str, Any], accent_color: QColor) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(QColor(115, 165, 255, 75), 1.0))
    painter.setBrush(QColor(10, 20, 36, 185))
    painter.drawRoundedRect(rect, 7.0, 7.0)

    verts = geom_spec.get("vertices") or []
    if len(verts) < 2:
        painter.setPen(QColor(200, 220, 250))
        painter.drawText(rect, Qt.AlignCenter, "Geometry")
        painter.restore()
        return

    inner = rect.adjusted(22.0, 18.0, -22.0, -18.0)
    min_x = min(float(vert_item["x"]) for vert_item in verts)
    max_x = max(float(vert_item["x"]) for vert_item in verts)
    min_y = min(float(vert_item["y"]) for vert_item in verts)
    max_y = max(float(vert_item["y"]) for vert_item in verts)
    span = max(1e-3, max(max_x - min_x, max_y - min_y))
    scale_px = min(inner.width(), inner.height()) / span
    offset_x = inner.left() + (inner.width() - (max_x - min_x) * scale_px) * 0.5
    offset_y = inner.bottom() - (inner.height() - (max_y - min_y) * scale_px) * 0.5

    pt_map: Dict[str, QPointF] = {
        str(vert_item["id"]): QPointF(offset_x + (float(vert_item["x"]) - min_x) * scale_px, offset_y - (float(vert_item["y"]) - min_y) * scale_px)
        for vert_item in verts
    }

    lbl_font = QFont("Segoe UI", 8, QFont.Bold)
    painter.setFont(lbl_font)
    for circle_item in (geom_spec.get("circles") or []):
        center_pt = pt_map.get(str(circle_item.get("at", "")))
        if center_pt is None:
            continue
        rad_px = float(circle_item.get("r", 1.0)) * scale_px
        col = TONE_COLOR_MAP.get(str(circle_item.get("tone", "teal")).lower(), accent_color)
        painter.setPen(QPen(col, 1.8))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center_pt, rad_px, rad_px)

    for line_item in (geom_spec.get("lines") or []):
        pt_a = pt_map.get(str(line_item.get("from", "")))
        pt_b = pt_map.get(str(line_item.get("to", "")))
        if pt_a is None or pt_b is None:
            continue
        col = TONE_COLOR_MAP.get(str(line_item.get("tone", "accent")).lower(), accent_color)
        pen = QPen(col, 2.1, Qt.DashLine if line_item.get("dash") else Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(pt_a, pt_b)
        side_lbl = str(line_item.get("label", "") or "")
        if side_lbl:
            mid_pt = QPointF(0.5 * (pt_a.x() + pt_b.x()), 0.5 * (pt_a.y() + pt_b.y()))
            painter.setPen(QColor(235, 245, 255))
            painter.drawText(QRectF(mid_pt.x() - 28.0, mid_pt.y() - 16.0, 56.0, 16.0), Qt.AlignCenter, side_lbl[:14])

    for v_id, v_pt in pt_map.items():
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(v_pt, 3.6, 3.6)
        painter.setPen(TONE_COLOR_MAP["amber"])
        painter.drawText(QRectF(v_pt.x() - 14.0, v_pt.y() - 18.0, 28.0, 16.0), Qt.AlignCenter, v_id[:6])
    painter.restore()


def paint_photo_labeled_pins(painter: QPainter, img_rect: QRectF, pins_list: Sequence[Dict[str, Any]], accent_color: QColor) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    lbl_font = QFont("Segoe UI", 8, QFont.Bold)
    painter.setFont(lbl_font)
    for pin_item in pins_list[:8]:
        px = img_rect.left() + float(pin_item.get("x", 0.5)) * img_rect.width()
        py = img_rect.top() + float(pin_item.get("y", 0.5)) * img_rect.height()
        text = str(pin_item.get("text", "") or pin_item.get("label", "")).strip()
        painter.setPen(QPen(accent_color, 1.4))
        painter.setBrush(QColor(255, 184, 77))
        painter.drawEllipse(QPointF(px, py), 4.0, 4.0)
        if text:
            pill = QRectF(min(img_rect.right() - 86.0, px + 6.0), max(img_rect.top(), py - 11.0), 82.0, 18.0)
            painter.setBrush(QColor(10, 18, 34, 225))
            painter.drawRoundedRect(pill, 4.0, 4.0)
            painter.setPen(QColor(240, 248, 255))
            painter.drawText(pill.adjusted(4.0, 0.0, -4.0, 0.0), Qt.AlignCenter, text[:14])
    painter.restore()


def paint_sources_row(painter: QPainter, rect: QRectF, sources_list: Sequence[Dict[str, str]], accent_color: QColor) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    font_sm = QFont("Segoe UI", 8.5, QFont.Medium)
    painter.setFont(font_sm)
    cur_x = rect.left()
    cur_y = rect.top() + 2.0
    for src in sources_list[:6]:
        name = str(src.get("name", "Source"))[:22]
        chip_w = min(rect.width(), max(68.0, len(name) * 7.2 + 24.0))
        if cur_x + chip_w > rect.right() and cur_x > rect.left():
            cur_x = rect.left()
            cur_y += 24.0
        chip_rect = QRectF(cur_x, cur_y, chip_w, 20.0)
        painter.setPen(QPen(QColor(102, 217, 239, 130), 1.0))
        painter.setBrush(QColor(18, 36, 58, 220))
        painter.drawRoundedRect(chip_rect, 10.0, 10.0)
        painter.setPen(QColor(150, 232, 255))
        painter.drawText(chip_rect, Qt.AlignCenter, name)
        cur_x += chip_w + 6.0
    painter.restore()


def paint_clip_row(painter: QPainter, rect: QRectF, clip_title: str, clip_text: str, accent_color: QColor) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(QColor(120, 175, 255, 90), 1.0))
    painter.setBrush(QColor(14, 24, 42, 210))
    painter.drawRoundedRect(rect, 7.0, 7.0)
    painter.setPen(Qt.NoPen)
    painter.setBrush(accent_color)
    painter.drawRoundedRect(QRectF(rect.left() + 4.0, rect.top() + 6.0, 3.0, rect.height() - 12.0), 1.5, 1.5)
    painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
    painter.setPen(QColor(235, 245, 255))
    painter.drawText(QRectF(rect.left() + 14.0, rect.top() + 6.0, rect.width() - 22.0, 18.0), Qt.AlignLeft | Qt.AlignVCenter, clip_title[:48])
    painter.setFont(QFont("Consolas", 8.5))
    painter.setPen(QColor(190, 215, 245))
    painter.drawText(QRectF(rect.left() + 14.0, rect.top() + 26.0, rect.width() - 22.0, rect.height() - 32.0), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, clip_text[:220])
    painter.restore()


def paint_legend_row(painter: QPainter, rect: QRectF, keys_list: Sequence[Dict[str, str]]) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setFont(QFont("Segoe UI", 8.5, QFont.Medium))
    row_y = rect.top() + 2.0
    cur_x = rect.left()
    for key_item in keys_list[:8]:
        col = TONE_COLOR_MAP.get(str(key_item.get("tone", "accent")).lower(), TONE_COLOR_MAP["accent"])
        label_txt = str(key_item.get("text", ""))[:24]
        item_w = max(72.0, len(label_txt) * 7.0 + 22.0)
        if cur_x + item_w > rect.right() and cur_x > rect.left():
            cur_x = rect.left()
            row_y += 20.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(col)
        painter.drawEllipse(QPointF(cur_x + 6.0, row_y + 9.0), 4.5, 4.5)
        painter.setPen(QColor(230, 240, 255))
        painter.drawText(QRectF(cur_x + 15.0, row_y, item_w - 15.0, 18.0), Qt.AlignLeft | Qt.AlignVCenter, label_txt)
        cur_x += item_w + 8.0
    painter.restore()


def paint_ticket_row(painter: QPainter, rect: QRectF, title_text: str, task_text: str, accent_color: QColor) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(QColor(120, 255, 226, 130), 1.2, Qt.DashLine))
    painter.setBrush(QColor(12, 32, 42, 220))
    painter.drawRoundedRect(rect, 8.0, 8.0)
    painter.setFont(QFont("Segoe UI", 9.5, QFont.Bold))
    painter.setPen(QColor(120, 255, 226))
    painter.drawText(QRectF(rect.left() + 12.0, rect.top() + 8.0, rect.width() - 24.0, 20.0), Qt.AlignLeft | Qt.AlignVCenter, f"TICKET: {title_text[:40]}")
    if task_text:
        painter.setFont(QFont("Segoe UI", 8.5))
        painter.setPen(QColor(225, 245, 242))
        painter.drawText(QRectF(rect.left() + 12.0, rect.top() + 30.0, rect.width() - 24.0, rect.height() - 36.0), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, task_text[:140])
    painter.restore()


def paint_medal_row(painter: QPainter, rect: QRectF, medal_text: str) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(QColor(255, 198, 90, 180), 1.5))
    painter.setBrush(QColor(38, 28, 12, 220))
    painter.drawRoundedRect(rect, 9.0, 9.0)
    cx = rect.left() + 26.0
    cy = rect.center().y()
    painter.setBrush(QColor(255, 196, 64))
    painter.drawEllipse(QPointF(cx, cy), 12.0, 12.0)
    painter.setFont(QFont("Segoe UI", 9.5, QFont.Bold))
    painter.setPen(QColor(255, 232, 165))
    painter.drawText(QRectF(rect.left() + 48.0, rect.top() + 6.0, rect.width() - 56.0, rect.height() - 12.0), Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap, medal_text[:120])
    painter.restore()


# ---------------------------------------------------------------------------
# 3. Companion Looks (JARVIS 3D LED Sphere + 7 Pixel Looks + HUD Readout)
# ---------------------------------------------------------------------------

def _golden_sphere_points(count: int) -> np.ndarray:
    pts: List[Tuple[float, float, float]] = []
    golden_angle = 2.39996323
    for idx in range(count):
        pt_y = 1.0 - (2.0 * (idx + 0.5)) / count
        rad = math.sqrt(max(0.0, 1.0 - pt_y * pt_y))
        ang = idx * golden_angle
        pts.append((math.cos(ang) * rad, pt_y, math.sin(ang) * rad))
    return np.asarray(pts, dtype=np.float32)


_SPHERE_PTS_DEFAULT = _golden_sphere_points(180)
_SPHERE_PTS_SMALL = _golden_sphere_points(96)

_PIXEL_GRIDS: Dict[str, Tuple[str, ...]] = {
    "classic": (
        "..####..",
        ".######.",
        "########",
        "########",
        ".######.",
        "..####..",
    ),
    "glayos": (
        ".######.",
        "########",
        "###OO###",
        "########",
        "########",
        ".######.",
    ),
    "ultren": (
        ".######.",
        "##.##.##",
        "########",
        "########",
        ".######.",
        "..####..",
    ),
    "anime": (
        ".######.",
        "########",
        "########",
        "#.####.#",
        "..####..",
        ".######.",
    ),
    "sarah": (
        ".######.",
        "########",
        "########",
        ".######.",
        "..####..",
        ".######.",
    ),
    "adrian": (
        ".######.",
        "########",
        "########",
        ".######.",
        "..####..",
        ".######.",
    ),
    "rita": (
        ".######.",
        "########",
        "########",
        ".######.",
        "..####..",
        ".######.",
    ),
}


def paint_companion_avatar(
    painter: QPainter,
    center_x: float,
    center_y: float,
    look_id: str,
    size_mode: str,
    anim_time: float,
    audio_level: float,
    state_mode: str,
    emotion_feel: Optional[str],
    accent_color: QColor,
) -> None:
    """Renders the active companion look (JARVIS 3D LED sphere, Roamie, GlayOS, Ultren, etc.)."""
    look_key = str(look_id or "jarvis").strip().lower()
    size_key = str(size_mode or "default").strip().lower()
    persona = jarvis_roamies_core.LOOK_PERSONAS.get(look_key, jarvis_roamies_core.LOOK_PERSONAS["jarvis"])
    col = accent_color if look_key in ("jarvis", "triangle") else QColor(persona["accent_hex"])
    feel = str(emotion_feel or "").lower()
    is_sad = feel in ("sad", "cry", "tired")
    is_angry = feel in ("angry", "shout")
    is_joy = feel in ("joy", "laugh", "warm")
    shake_x = math.sin(anim_time * 48.0) * 1.4 if is_angry else 0.0

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)

    if size_key == "cursor" and look_key == "jarvis":
        # Compact HUD reticle cursor mode from jarvis.js
        rad = 12.0 + (audio_level * 3.5 if state_mode in ("talk", "listen") else 0.0)
        spin = anim_time * (2.4 if state_mode == "think" else 0.9 if state_mode != "idle" else 0.15)
        painter.setPen(QPen(col, 1.8, Qt.SolidLine, Qt.RoundCap))
        for q_idx in range(4):
            start_deg = math.degrees(spin) + q_idx * 90.0 + 12.0
            painter.drawArc(QRectF(center_x - rad, center_y - rad, rad * 2.0, rad * 2.0), int(start_deg * 16), 66 * 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(235, 248, 255))
        painter.drawEllipse(QPointF(center_x, center_y), 3.0, 3.0)
        painter.restore()
        return

    if look_key == "jarvis":
        rad = 14.0 if size_key == "small" else 20.0
        pts = _SPHERE_PTS_SMALL if size_key == "small" else _SPHERE_PTS_DEFAULT
        spin_rate = 1.6 if state_mode == "think" else (0.9 if state_mode == "talk" else 0.38)
        yaw = anim_time * spin_rate
        pitch = -0.38
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        wavy = max(0.14, audio_level * 0.95) if state_mode in ("talk", "listen") else 0.12

        px_x = pts[:, 0] * cos_y + pts[:, 2] * sin_y
        px_z_rot = -pts[:, 0] * sin_y + pts[:, 2] * cos_y
        px_y = pts[:, 1] * cos_p - px_z_rot * sin_p
        px_z = pts[:, 1] * sin_p + px_z_rot * cos_p
        ripple = 1.0 + wavy * 0.16 * np.sin(pts[:, 1] * 6.5 - anim_time * (8.5 if state_mode == "talk" else 2.8))

        painter.setPen(Qt.NoPen)
        cx_adj = center_x + shake_x
        for dot_idx in range(pts.shape[0]):
            zv = float(px_z[dot_idx])
            if zv < -0.22:
                continue
            rx = cx_adj + float(px_x[dot_idx] * ripple[dot_idx]) * rad
            ry = center_y + float(px_y[dot_idx] * ripple[dot_idx]) * rad
            bright = max(0.28, min(1.0, 0.45 + 0.55 * zv))
            painter.setBrush(QColor(col.red(), col.green(), col.blue(), int(240 * bright)))
            dot_sz = 1.75 if zv > 0.35 else 1.25
            painter.drawEllipse(QPointF(rx, ry), dot_sz, dot_sz)

        # Lit glowing core heart
        heart_rad = rad * (0.20 if is_sad else (0.29 + wavy * 0.08))
        glow = QRadialGradient(cx_adj, center_y, heart_rad * 2.2)
        glow.setColorAt(0.0, QColor(240, 250, 255, 245))
        glow.setColorAt(0.45, QColor(col.red(), col.green(), col.blue(), 190))
        glow.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(cx_adj, center_y), heart_rad * 2.1, heart_rad * 2.1)

        if state_mode == "think":
            scan_deg = (anim_time * 240.0) % 360.0
            painter.setPen(QPen(QColor(235, 248, 255, 220), 2.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(QRectF(cx_adj - rad - 3.5, center_y - rad - 3.5, (rad + 3.5) * 2, (rad + 3.5) * 2), int(scan_deg * 16), 85 * 16)
        painter.restore()
        return

    # Pixel companion sprite look (Roamie, GlayOS, Ultren, Anime Girl, Sarah, Adrian, Rita X)
    cell_sz = 2.6 if size_key in ("small", "cursor") else 3.8
    grid = _PIXEL_GRIDS.get(look_key, _PIXEL_GRIDS["classic"])
    grid_w = len(grid[0]) * cell_sz
    grid_h = len(grid) * cell_sz
    origin_x = center_x - grid_w * 0.5 + shake_x
    origin_y = center_y - grid_h * 0.5

    painter.setPen(Qt.NoPen)
    body_col = QColor("#8A94A6") if look_key == "ultren" else (QColor("#243040") if look_key == "glayos" else col)
    for row_idx, row_str in enumerate(grid):
        for col_idx, cell_ch in enumerate(row_str):
            if cell_ch == ".":
                continue
            cell_rect = QRectF(origin_x + col_idx * cell_sz, origin_y + row_idx * cell_sz, cell_sz - 0.3, cell_sz - 0.3)
            if cell_ch == "O":
                painter.setBrush(QColor("#FFB84D"))
            else:
                painter.setBrush(body_col)
            painter.drawRect(cell_rect)

    # Eyes (blinking every ~4.2s)
    is_blinking = (anim_time % 4.2) > 4.06
    eye_col = QColor("#FF4D4D") if look_key == "ultren" else (QColor("#FFB84D") if look_key == "glayos" else QColor("#FFFFFF"))
    painter.setBrush(eye_col)
    if look_key == "glayos":
        painter.drawEllipse(QPointF(center_x + shake_x, origin_y + cell_sz * 2.4), cell_sz * 0.85, cell_sz * 0.25 if is_blinking else cell_sz * 0.85)
    else:
        for eye_offset in (-cell_sz * 1.4, cell_sz * 1.4):
            painter.drawRect(QRectF(center_x + shake_x + eye_offset - cell_sz * 0.45, origin_y + cell_sz * 2.0, cell_sz * 0.9, cell_sz * (0.25 if is_blinking else 0.85)))
        if look_key == "rita":
            painter.setPen(QPen(QColor("#E8FFFA"), 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(center_x + shake_x - cell_sz * 2.3, origin_y + cell_sz * 1.7, cell_sz * 1.9, cell_sz * 1.3))
            painter.drawRect(QRectF(center_x + shake_x + cell_sz * 0.4, origin_y + cell_sz * 1.7, cell_sz * 1.9, cell_sz * 1.3))

    # Mouth reacting to voice & emotion feel
    mouth_y = origin_y + cell_sz * 4.3
    mouth_h = 1.6 + (audio_level * cell_sz * 0.9 if state_mode == "talk" else 0.0)
    painter.setPen(QPen(QColor(245, 250, 255), 1.5, Qt.SolidLine, Qt.RoundCap))
    if is_joy:
        painter.drawArc(QRectF(center_x + shake_x - cell_sz * 1.1, mouth_y - 2.0, cell_sz * 2.2, 5.0), 195 * 16, 150 * 16)
    elif is_sad:
        painter.drawArc(QRectF(center_x + shake_x - cell_sz * 1.1, mouth_y, cell_sz * 2.2, 5.0), 15 * 16, 150 * 16)
    else:
        painter.drawLine(QPointF(center_x + shake_x - cell_sz, mouth_y), QPointF(center_x + shake_x + cell_sz, mouth_y + (0.0 if mouth_h < 2.2 else 1.2)))
    painter.restore()


# ---------------------------------------------------------------------------
# 4. Cursor Ask Bar Window (Ctrl+Shift beside cursor)
# ---------------------------------------------------------------------------

class CursorAskBarWindow(QWidget):
    """Slim cursor-adjacent typing field toggled by Ctrl+Shift tap (ask-bar.js equivalent)."""

    promptSubmitted = Signal(str)
    closedSignal = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._draft_text = ""

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)
        self._container = QFrame(self)
        self._container.setStyleSheet(
            "QFrame { background-color: rgba(12, 16, 26, 244); border: 1.5px solid #4F92FF; border-radius: 12px; }"
        )
        inner_layout = QHBoxLayout(self._container)
        inner_layout.setContentsMargins(10, 8, 10, 8)
        inner_layout.setSpacing(8)

        dot_lbl = QLabel(self._container)
        dot_lbl.setFixedSize(8, 8)
        dot_lbl.setStyleSheet("background-color: #3380FF; border-radius: 4px; border: none;")
        inner_layout.addWidget(dot_lbl, 0, Qt.AlignTop | Qt.AlignHCenter)

        self._editor = QPlainTextEdit(self._container)
        self._editor.setPlaceholderText("Ask Jarvis...")
        self._editor.setMinimumWidth(290)
        self._editor.setMaximumWidth(340)
        self._editor.setFixedHeight(28)
        self._editor.setStyleSheet(
            "QPlainTextEdit { background: transparent; color: #F2F6FF; border: none; font-family: 'Segoe UI'; font-size: 13px; }"
        )
        self._editor.textChanged.connect(self._on_text_changed)
        self._editor.installEventFilter(self)
        inner_layout.addWidget(self._editor, 1)
        root_layout.addWidget(self._container)

    def _on_text_changed(self) -> None:
        text = self._editor.toPlainText()
        doc_lines = max(1, min(5, int(self._editor.document().size().height())))
        target_h = max(28, min(110, doc_lines * 21 + 6))
        self._editor.setFixedHeight(target_h)
        is_rtl = jarvis_roamies_core.is_rtl_text(text)
        self._editor.setLayoutDirection(Qt.RightToLeft if is_rtl else Qt.LeftToRight)
        self.adjustSize()

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is self._editor and event.type() == QEvent.KeyPress:
            key_event = event  # type: ignore[assignment]
            if key_event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (key_event.modifiers() & Qt.ShiftModifier):
                text = self._editor.toPlainText().strip()
                self._draft_text = ""
                self._editor.clear()
                self.hide()
                if text:
                    self.promptSubmitted.emit(text)
                return True
            if key_event.key() == Qt.Key_Escape:
                self._draft_text = ""
                self._editor.clear()
                self.hide()
                self.closedSignal.emit()
                return True
        return super().eventFilter(watched, event)

    def open_at_cursor(self, cursor_pos: QPointF, screen_bounds: QRectF) -> None:
        if self._draft_text and not self._editor.toPlainText():
            self._editor.setPlainText(self._draft_text)
        self.adjustSize()
        win_w = max(320, self.width())
        win_h = max(48, self.height())
        pos_x = cursor_pos.x() + 18.0
        pos_y = cursor_pos.y() + 8.0
        if pos_x + win_w > screen_bounds.right() - 8.0:
            pos_x = cursor_pos.x() - 8.0 - win_w
        if pos_y + win_h > screen_bounds.bottom() - 8.0:
            pos_y = cursor_pos.y() - 8.0 - win_h
        pos_x = max(screen_bounds.left() + 8.0, min(pos_x, screen_bounds.right() - win_w - 8.0))
        pos_y = max(screen_bounds.top() + 8.0, min(pos_y, screen_bounds.bottom() - win_h - 8.0))
        self.move(int(pos_x), int(pos_y))
        self.show()
        self.raise_()
        self.activateWindow()
        self._editor.setFocus()
        self._editor.selectAll()

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow() and self.isVisible():
            self._draft_text = self._editor.toPlainText()
            self.hide()
            self.closedSignal.emit()
        super().changeEvent(event)


# ---------------------------------------------------------------------------
# 5. Risky Action Approval Bar (Allow / Don't Allow)
# ---------------------------------------------------------------------------

class RiskyActionApprovalBar(QWidget):
    """Floating Allow / Don't Allow bar for risky clicks, overwrites, or commands."""

    decisionMade = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        card = QFrame(self)
        card.setStyleSheet(
            "QFrame { background-color: rgba(14, 18, 30, 246); border: 1.5px solid #FFB84D; border-radius: 12px; }"
        )
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 8, 10, 8)
        row.setSpacing(10)

        self._reason_label = QLabel("Allow Jarvis to perform this action?", card)
        self._reason_label.setStyleSheet("color: #F5F7FF; font-size: 12px; font-weight: 600; border: none; background: transparent;")
        self._reason_label.setWordWrap(True)
        self._reason_label.setMaximumWidth(340)
        row.addWidget(self._reason_label)

        allow_btn = QPushButton("Allow", card)
        allow_btn.setCursor(Qt.PointingHandCursor)
        allow_btn.setStyleSheet(
            "QPushButton { background-color: #2B7BFF; color: #FFFFFF; border: none; border-radius: 7px; padding: 6px 12px; font-weight: 700; font-size: 12px; }"
            "QPushButton:hover { background-color: #4690FF; }"
        )
        allow_btn.clicked.connect(lambda: self._emit_decision(True))
        row.addWidget(allow_btn)

        deny_btn = QPushButton("Don't allow", card)
        deny_btn.setCursor(Qt.PointingHandCursor)
        deny_btn.setStyleSheet(
            "QPushButton { background-color: rgba(255, 82, 82, 36); color: #FFA8A8; border: 1px solid rgba(255, 82, 82, 110); border-radius: 7px; padding: 6px 10px; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background-color: rgba(255, 82, 82, 70); }"
        )
        deny_btn.clicked.connect(lambda: self._emit_decision(False))
        row.addWidget(deny_btn)
        layout.addWidget(card)

    def _emit_decision(self, allowed: bool) -> None:
        self.hide()
        self.decisionMade.emit(bool(allowed))

    def prompt_action(self, reason_text: str, screen_bounds: QRectF) -> None:
        self._reason_label.setText(str(reason_text or "Allow action?")[:140])
        self.adjustSize()
        pos_x = screen_bounds.center().x() - self.width() * 0.5
        pos_y = screen_bounds.top() + 58.0
        self.move(int(pos_x), int(pos_y))
        self.show()
        self.raise_()


# ---------------------------------------------------------------------------
# 6. Agent Artifact Toast Window (Click generated files to open)
# ---------------------------------------------------------------------------

class AgentArtifactToastWindow(QWidget):
    """Bottom-right clickable toast announcing completed background subagent outputs."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._files: List[str] = []
        self._hide_timer = QTimer(self)
        self._hide_timer.setInterval(12000)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        self._frame = QFrame(self)
        self._frame.setStyleSheet(
            "QFrame { background-color: rgba(13, 19, 32, 245); border: 1px solid #3A7FFF; border-radius: 12px; }"
        )
        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(12, 10, 12, 10)
        frame_layout.setSpacing(6)

        hdr_row = QHBoxLayout()
        self._title_lbl = QLabel("Subagent Finished", self._frame)
        self._title_lbl.setStyleSheet("color: #EAF2FF; font-weight: 700; font-size: 12px; border: none; background: transparent;")
        hdr_row.addWidget(self._title_lbl)
        hdr_row.addStretch(1)
        close_btn = QPushButton("×", self._frame)
        close_btn.setFixedSize(20, 20)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("QPushButton { color: #9FB5D9; background: transparent; border: none; font-size: 14px; font-weight: 700; }")
        close_btn.clicked.connect(self.hide)
        hdr_row.addWidget(close_btn)
        frame_layout.addLayout(hdr_row)

        self._body_lbl = QLabel("", self._frame)
        self._body_lbl.setWordWrap(True)
        self._body_lbl.setStyleSheet("color: #B9CBE8; font-size: 11px; border: none; background: transparent;")
        frame_layout.addWidget(self._body_lbl)

        self._open_btn = QPushButton("Open output file", self._frame)
        self._open_btn.setCursor(Qt.PointingHandCursor)
        self._open_btn.setStyleSheet(
            "QPushButton { background-color: rgba(51, 128, 255, 50); color: #9AC6FF; border: 1px solid rgba(51, 128, 255, 130); border-radius: 6px; padding: 5px 10px; font-size: 11px; font-weight: 600; }"
            "QPushButton:hover { background-color: rgba(51, 128, 255, 95); }"
        )
        self._open_btn.clicked.connect(self._open_first_file)
        frame_layout.addWidget(self._open_btn)
        outer.addWidget(self._frame)
        self.setFixedWidth(290)

    def _open_first_file(self) -> None:
        if not self._files:
            return
        target = self._files[0]
        try:
            os.startfile(target)
        except Exception:
            try:
                subprocess.Popen(["explorer.exe", f"/select,{os.path.normpath(target)}"])
            except Exception:
                pass

    def show_artifact(self, agent_label: str, brief_text: str, files: List[str], screen_bounds: QRectF) -> None:
        self._files = list(files or [])
        self._title_lbl.setText(f"{agent_label} finished")
        file_names = ", ".join(os.path.basename(path_item) for path_item in self._files[:2])
        self._body_lbl.setText(f"{brief_text[:80]}\nFiles: {file_names or 'report saved'}")
        self._open_btn.setVisible(bool(self._files))
        self.adjustSize()
        self.move(int(screen_bounds.right() - self.width() - 18.0), int(screen_bounds.bottom() - self.height() - 48.0))
        self.show()
        self.raise_()
        self._hide_timer.start()

    def enterEvent(self, event: QEvent) -> None:
        self._hide_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._hide_timer.start()
        super().leaveEvent(event)
