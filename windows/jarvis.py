"""
Jarvis for Windows — PySide6 Entry Point.

Supports:
- Normal system-tray + top-notch + screen-overlay interactive run
- `--demo` and `--demo-seconds <N>` for showcase mode
- `--snapshot <dir>` for offscreen PNG inspection across all windows
- `--selftest` for strict offline + live API/FPS/TTS verification
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

WINDOWS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if WINDOWS_DIRECTORY not in sys.path:
    sys.path.insert(0, WINDOWS_DIRECTORY)

import requests
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import jarvis_app
import jarvis_config
import jarvis_main_window
import jarvis_notch_window
import jarvis_overlay_windows
import jarvis_response_parser
import jarvis_services
import jarvis_theme

LOG_FILE_PATH = os.path.join(WINDOWS_DIRECTORY, "jarvis.log")


def _append_log_line(message: str) -> None:
    """Append an unexpected runtime error to `jarvis.log` without raising."""
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(message.rstrip() + "\n")
    except Exception:
        pass


def _configure_clean_startup_logging() -> None:
    """Reset `jarvis.log` to empty on startup and route uncaught tracebacks into it."""
    try:
        with open(LOG_FILE_PATH, "w", encoding="utf-8"):
            pass
    except Exception:
        pass

    def handle_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            return
        import traceback

        formatted_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _append_log_line(formatted_trace)

    def handle_thread_exception(thread_args) -> None:
        handle_uncaught_exception(
            thread_args.exc_type,
            thread_args.exc_value,
            thread_args.exc_traceback,
        )

    sys.excepthook = handle_uncaught_exception
    threading.excepthook = handle_thread_exception


def _create_qt_application() -> QApplication:
    """Create or reuse the singleton `QApplication` configured for HiDPI tray companion mode."""
    jarvis_theme.enable_per_monitor_dpi_awareness()
    existing_app = QApplication.instance()
    if isinstance(existing_app, QApplication):
        qt_application = existing_app
    else:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        qt_application = QApplication(sys.argv)

    qt_application.setQuitOnLastWindowClosed(False)
    qt_application.setApplicationName("Jarvis")
    qt_application.setApplicationDisplayName("Jarvis")

    icon_path = os.path.join(jarvis_theme.ASSETS_DIRECTORY, "jarvis.ico")
    if os.path.exists(icon_path):
        qt_application.setWindowIcon(QIcon(icon_path))

    jarvis_theme.register_bundled_fonts()
    return qt_application


def run_snapshot_export(output_directory: str) -> int:
    """Render visual snapshots of Notch, Overlay, and Main windows to PNG files."""
    _create_qt_application()
    resolved_directory = os.path.abspath(output_directory)
    os.makedirs(resolved_directory, exist_ok=True)

    exported_paths: List[str] = []
    exported_paths.extend(
        jarvis_notch_window.save_visual_snapshots(os.path.join(resolved_directory, "notch"))
    )
    exported_paths.extend(
        jarvis_overlay_windows.save_visual_snapshots(os.path.join(resolved_directory, "overlay"))
    )
    exported_paths.extend(
        jarvis_main_window.save_visual_snapshots(os.path.join(resolved_directory, "main"))
    )

    for file_path in exported_paths:
        print(f"[SNAPSHOT] {file_path}")
    print(f"Exported {len(exported_paths)} snapshot PNGs to {resolved_directory}")
    return 0 if exported_paths else 1


def _build_synthetic_vision_test_jpeg_b64() -> str:
    """
    Create an in-memory 400x200 JPEG card showing a bright red square and bold '73'
    so Vision SSE self-testing works reliably even if the Windows desktop is locked.
    """
    test_image = Image.new("RGB", (400, 200), (248, 250, 252))
    painter = ImageDraw.Draw(test_image)
    painter.rectangle([28, 36, 158, 166], fill=(220, 38, 38), outline=(127, 29, 29), width=4)

    font_path = os.path.join(jarvis_theme.ASSETS_DIRECTORY, "fonts", "Jarvis-Barlow-SemiBold.ttf")
    try:
        label_font = ImageFont.truetype(font_path, 76)
    except Exception:
        label_font = ImageFont.load_default()
    painter.text((225, 58), "73", fill=(15, 23, 42), font=label_font)

    output_buffer = io.BytesIO()
    test_image.save(output_buffer, format="JPEG", quality=92)
    return base64.b64encode(output_buffer.getvalue()).decode("ascii")


def run_strict_selftest() -> int:
    """
    Execute strict verification across assets, 60 FPS Qt rendering, response parsing,
    main-window close-to-tray behavior, live AssemblyAI token, live River AI Vision SSE,
    and live ElevenLabs PCM speech synthesis.
    """
    qt_application = _create_qt_application()
    settings_store = jarvis_config.SettingsStore()
    failures: List[str] = []

    def record_result(suite_name: str, passed: bool, detail_message: str) -> None:
        status_label = "PASS" if passed else "FAIL"
        print(f"[{status_label}] {suite_name}: {detail_message}")
        if not passed:
            failures.append(f"{suite_name}: {detail_message}")

    # -----------------------------------------------------------------------
    # Suite 1: Assets, Fonts, and Required Secrets
    # -----------------------------------------------------------------------
    required_asset_relpaths = [
        "jarvis.ico",
        "jarvis.ico",
        os.path.join("fonts", "Jarvis-PatrickHand.ttf"),
        os.path.join("fonts", "Jarvis-Barlow-Regular.ttf"),
        os.path.join("fonts", "Jarvis-Barlow-Medium.ttf"),
        os.path.join("fonts", "Jarvis-Barlow-SemiBold.ttf"),
        os.path.join("earcons", "reveal-boot.wav"),
    ]
    missing_assets = [
        relpath
        for relpath in required_asset_relpaths
        if not os.path.exists(os.path.join(jarvis_theme.ASSETS_DIRECTORY, relpath))
    ]
    has_assembly_key = bool(settings_store.get_secret(jarvis_config.SECRET_ASSEMBLYAI_API_KEY))
    has_vision_key = settings_store.has_any_vision_credentials()
    missing_secrets: List[str] = []
    if not has_assembly_key:
        missing_secrets.append(jarvis_config.SECRET_ASSEMBLYAI_API_KEY)
    if not has_vision_key:
        missing_secrets.append(
            f"{jarvis_config.SECRET_CUSTOM_API_KEY}|{jarvis_config.SECRET_GOOGLE_API_KEY}|{jarvis_config.SECRET_RIVER_API_KEY}"
        )
    suite1_ok = (not missing_assets) and (not missing_secrets)
    record_result(
        "Suite 1 (Assets & Secrets)",
        suite1_ok,
        (
            "All required assets, fonts, earcons, and .env secrets present."
            if suite1_ok
            else f"missing_assets={missing_assets}, missing_secrets={missing_secrets}"
        ),
    )

    # -----------------------------------------------------------------------
    # Suite 2: Notch Window 60 FPS Render Benchmark & Personal Cards Check
    # -----------------------------------------------------------------------
    try:
        notch_window = jarvis_notch_window.NotchWindow(settings_store)
        notch_metrics = notch_window.benchmark_frame_ms(iterations=60)
        notch_avg_ms = float(notch_metrics.get("average_ms", 999.0))
        notch_card_texts = " ".join(
            f"{card.card_id} {card.title} {card.subtitle}"
            for card in getattr(notch_window, "_arc_cards", [])
        ).lower()
        notch_personal_ok = ("upgrade" not in notch_card_texts) and ("invite" not in notch_card_texts)
        notch_window.close()
        record_result(
            "Suite 2 (Notch FPS)",
            (notch_avg_ms < 5.0) and notch_personal_ok,
            f"average_ms={notch_avg_ms:.3f}ms (target < 5.0ms), personal_cards_ok={notch_personal_ok}",
        )
    except Exception as exc:
        record_result("Suite 2 (Notch FPS)", False, f"Exception: {exc}")

    # -----------------------------------------------------------------------
    # Suite 3: Overlay & Ink Canvas 60 FPS Render Benchmark & HiDPI Paper DPR
    # -----------------------------------------------------------------------
    try:
        overlay_window = jarvis_overlay_windows.OverlayCanvasWindow(settings_store)
        overlay_metrics = overlay_window.benchmark_frame_ms(iterations=60)
        overlay_avg_ms = float(overlay_metrics.get("avg_ms", 999.0))
        sample_paper = jarvis_response_parser.DrawingCommand(
            kind="paper",
            x=60.0,
            y=120.0,
            width=310.0,
            height=260.0,
            label="HiDPI Card",
            body="Line 1 -> Line 2",
        )
        paper_pixmap = jarvis_overlay_windows._render_paper_card_to_pixmap(sample_paper, seed_value=1)
        paper_dpr_ok = paper_pixmap.devicePixelRatio() >= 2.0
        overlay_window.clear_button_window.close()
        overlay_window.close()
        record_result(
            "Suite 3 (Overlay FPS)",
            (overlay_avg_ms < 5.0) and paper_dpr_ok,
            f"avg_ms={overlay_avg_ms:.3f}ms (target < 5.0ms), paper_dpr={paper_pixmap.devicePixelRatio():.2f}",
        )
    except Exception as exc:
        record_result("Suite 3 (Overlay FPS)", False, f"Exception: {exc}")

    # -----------------------------------------------------------------------
    # Suite 4: MainWindow, Parser, Scroll/Snap, Personal UI & AST Checks
    # -----------------------------------------------------------------------
    try:
        import ast
        import tempfile
        from PIL import Image, ImageDraw

        main_window = jarvis_main_window.JarvisMainWindow(settings_store)
        main_metrics = main_window.benchmark_frame_ms(iterations=30)
        main_mean_ms = float(main_metrics.get("mean_ms", 999.0))
        subtab_keys_joined = " ".join(str(key) for key in getattr(main_window, "_subtab_pages", {}).keys()).lower()
        main_personal_ok = ("upgrade" not in subtab_keys_joined) and ("invite" not in subtab_keys_joined)

        main_window.show()
        qt_application.processEvents()
        main_window.close()
        qt_application.processEvents()
        hidden_on_close = not main_window.isVisible()

        main_window.allow_real_close()
        main_window.close()
        qt_application.processEvents()

        extractor = jarvis_response_parser.StreamingSentenceExtractor(scale_x=1.0, scale_y=1.0)
        collected_sentences: List[str] = []
        collected_commands: List[jarvis_response_parser.DrawingCommand] = []
        for chunk_text in [
            "<th",
            "ink>internal reasoning hidden</think>\n",
            'underline(100, 120, 220, "Heading")\n',
            'ring(300, 220, 34, "Save")\n',
            "Click the Save button. [POINT:300,220:Save]\n",
            'paper(28, 310, 330, 360, "Card Title", "summary", "Concept A -> Concept B")\n',
            'scroll("down", 6)\n',
        ]:
            part_sentences, part_commands = extractor.feed_delta(chunk_text)
            collected_sentences.extend(part_sentences)
            collected_commands.extend(part_commands)
        end_sentences, end_commands = extractor.flush_remaining()
        collected_sentences.extend(end_sentences)
        collected_commands.extend(end_commands)
        parsed_turn = extractor.build_final_response()
        learned_style = jarvis_response_parser.extract_style_preference_from_user_message(
            "Please keep it brief and give a shorter answer."
        )
        scroll_intent_ok = jarvis_response_parser.is_scroll_and_explain_request(
            "scroll down and explain this ChatGPT response"
        )
        scroll_parse_ok = (parsed_turn.scroll_direction == "down") and (parsed_turn.scroll_amount == 6)

        previous_snapshot = jarvis_services.ScreenCaptureService._last_raw_logical_image
        previous_regions = list(jarvis_services.ScreenCaptureService._used_underline_regions)
        try:
            jarvis_services.ScreenCaptureService.reset_underline_baselines()
            synthetic_screen = Image.new("RGB", (600, 300), (20, 22, 28))
            synthetic_draw = ImageDraw.Draw(synthetic_screen)
            for bar_x in range(104, 256, 5):
                synthetic_draw.rectangle([bar_x, 118, bar_x + 2, 130], fill=(242, 246, 255))
            jarvis_services.ScreenCaptureService._last_raw_logical_image = synthetic_screen
            off_baseline_cmd = jarvis_response_parser.DrawingCommand(
                kind="underline",
                x=100.0,
                y=142.0,
                width=160.0,
                label="Line",
            )
            snapped_cmd = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(
                off_baseline_cmd
            )
            baseline_snap_ok = abs(snapped_cmd.y - 142.0) <= 18.0 and (snapped_cmd.y < 140.0)

            # Stacked 34px-spaced list test matching HRM list layout:
            # verify distinct baseline locking + strict horizontal trimming without Line 2/overshoot leak
            jarvis_services.ScreenCaptureService.reset_underline_baselines()
            stacked_screen = Image.new("RGB", (900, 500), (24, 24, 24))
            stacked_draw = ImageDraw.Draw(stacked_screen)
            line_specs = [
                (150, 164, 80, 580),  # Point 1: 500px
                (184, 198, 80, 620),  # Point 2: 540px
                (218, 232, 80, 510),  # Point 3: 430px (shorter than 1 & 2)
                (252, 266, 80, 500),  # Point 4: 420px
            ]
            for top_row, bottom_row, col_start, col_end in line_specs:
                for glyph_x in range(col_start, col_end, 5):
                    stacked_draw.rectangle([glyph_x, top_row, glyph_x + 2, bottom_row], fill=(235, 240, 248))
            jarvis_services.ScreenCaptureService._last_raw_logical_image = stacked_screen

            snap_line_1 = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(
                jarvis_response_parser.DrawingCommand(
                    kind="underline", x=80.0, y=168.0, width=500.0, label="1. Artificial Intelligence"
                )
            )
            # Deliberately supply slightly high Y near Line 1 to ensure it locks onto unused Line 2
            snap_line_2 = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(
                jarvis_response_parser.DrawingCommand(
                    kind="underline", x=80.0, y=174.0, width=540.0, label="2. Remote Working"
                )
            )
            # Deliberately supply huge overshooting width (640px) on Line 3 (actual text ends at x=510, w=430)
            snap_line_3 = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(
                jarvis_response_parser.DrawingCommand(
                    kind="underline", x=80.0, y=236.0, width=640.0, label="3. HR Analytics"
                )
            )
            stacked_list_snap_ok = (
                (snap_line_2.y > snap_line_1.y + 20.0)
                and (abs(snap_line_3.y - 236.0) <= 6.0)
                and (snap_line_3.width <= 442.0)
                and (snap_line_3.width >= 415.0)
            )

            adaptive_scroll_ok = (
                jarvis_services.ScreenCaptureService.recommend_adaptive_scroll_ticks(
                    direction="down", requested_ticks=6, user_prompt="scroll down a little"
                ) == 3
                and jarvis_services.ScreenCaptureService.recommend_adaptive_scroll_ticks(
                    direction="down", requested_ticks=6, user_prompt="scroll half page"
                ) == 5
                and jarvis_services.ScreenCaptureService.recommend_adaptive_scroll_ticks(
                    direction="down", requested_ticks=6, user_prompt="scroll a full page"
                ) == 9
                and jarvis_services.ScreenCaptureService.recommend_adaptive_scroll_ticks(
                    direction="down", requested_ticks=0, user_prompt="explain this", force_if_needed=True
                ) == 0
            )
        finally:
            jarvis_services.ScreenCaptureService._last_raw_logical_image = previous_snapshot
            jarvis_services.ScreenCaptureService._used_underline_regions = previous_regions

        single_char_violations: List[str] = []
        active_module_paths = [
            os.path.join(jarvis_theme.WINDOWS_DIRECTORY, module_name)
            for module_name in (
                "jarvis.py",
                "jarvis_app.py",
                "jarvis_config.py",
                "jarvis_main_window.py",
                "jarvis_notch_window.py",
                "jarvis_overlay_windows.py",
                "jarvis_roamies_core.py",
                "jarvis_roamies_visuals.py",
                "jarvis_response_parser.py",
                "jarvis_services.py",
                "jarvis_theme.py",
            )
        ]
        for module_path in active_module_paths:
            with open(module_path, "r", encoding="utf-8") as source_handle:
                parsed_tree = ast.parse(source_handle.read(), filename=module_path)
            exempt_field_lines: set[int] = set()
            if os.path.basename(module_path) == "jarvis_response_parser.py":
                for top_node in ast.walk(parsed_tree):
                    if isinstance(top_node, ast.ClassDef) and top_node.name == "DrawingCommand":
                        for body_stmt in top_node.body:
                            if isinstance(body_stmt, ast.AnnAssign) and isinstance(body_stmt.target, ast.Name):
                                if body_stmt.target.id in ("x", "y"):
                                    exempt_field_lines.add(body_stmt.target.lineno)
            for ast_node in ast.walk(parsed_tree):
                if (
                    isinstance(ast_node, ast.Name)
                    and len(ast_node.id) == 1
                    and ast_node.id != "_"
                    and ast_node.lineno not in exempt_field_lines
                ):
                    single_char_violations.append(f"{os.path.basename(module_path)}:{ast_node.lineno}:{ast_node.id}")
                elif isinstance(ast_node, ast.arg) and len(ast_node.arg) == 1 and ast_node.arg != "_":
                    single_char_violations.append(f"{os.path.basename(module_path)}:{ast_node.lineno}:{ast_node.arg}")

        agent_open_browser_cmd = jarvis_response_parser.parse_command_line('open_browser("https://chatgpt.com")')
        agent_search_cmd = jarvis_response_parser.parse_command_line('search_web("HRM scope")')
        direct_browser_intent = jarvis_services.DesktopAgentService.detect_direct_user_agent_command("open my browser")
        agent_ok = (
            agent_open_browser_cmd is not None
            and agent_open_browser_cmd.kind == "open_browser"
            and agent_search_cmd is not None
            and agent_search_cmd.kind == "search_web"
            and direct_browser_intent is not None
            and direct_browser_intent.kind == "open_browser"
        )

        # Verify precise_lines normalizes compact box -> underline, blocks giant boxes,
        # ignores 1-point clicks on end_highlight_stroke, and separates multiple paper notes.
        temp_overlay = jarvis_overlay_windows.OverlayCanvasWindow(settings_store)
        temp_overlay.resize(1536, 864)
        temp_overlay._drawing_mode = "precise_lines"
        compact_box_cmd = jarvis_response_parser.DrawingCommand(kind="box", x=100.0, y=120.0, width=220.0, height=34.0, label="Tag")
        normalized_compact = temp_overlay._normalize_command_for_drawing_mode(compact_box_cmd)
        oversized_box_cmd = jarvis_response_parser.DrawingCommand(kind="box", x=20.0, y=20.0, width=1600.0, height=760.0, label="Huge")
        normalized_huge = temp_overlay._normalize_command_for_drawing_mode(oversized_box_cmd)
        temp_overlay.begin_highlight_stroke()
        click_only_lasso = temp_overlay.end_highlight_stroke()
        temp_overlay._highlight_points = [
            (100.0 + float(pt_idx * 18), 100.0 + float(pt_idx * 12))
            for pt_idx in range(8)
        ]
        real_drag_lasso = temp_overlay.end_highlight_stroke()
        multi_paper_cmds = jarvis_services.VisionBrainService._build_visual_note_cards_for_response(
            spoken_sentences=[
                "First, planning forecasts hiring demand across departments.",
                "Second, recruitment sources candidates cleanly through referrals.",
                "Third, appraisal reviews employee growth and goal alignment.",
            ],
            spoken_text=(
                "First, planning forecasts hiring demand across departments. "
                "Second, recruitment sources candidates cleanly through referrals. "
                "Third, appraisal reviews employee growth and goal alignment."
            ),
            screen_width=1536,
            screen_height=864,
            underline_labels=["Planning", "Recruitment", "Appraisal"],
        )
        temp_overlay.clear_all_drawings()
        for card_idx in range(3):
            temp_overlay.add_drawing_command(
                jarvis_response_parser.DrawingCommand(
                    kind="paper",
                    x=24.0,
                    y=92.0,
                    width=304.0,
                    height=240.0,
                    label=f"Topic {card_idx + 1}",
                    header="Summary",
                    body="Step A -> Step B | section:Core | Point C -> Point D",
                    sentence_index=card_idx,
                )
            )
        placed_paper_items = [
            item.command for item in temp_overlay._drawings if item.command.kind == "paper"
        ]
        placed_positions = {
            (round(item.x, 1), round(item.y, 1)) for item in placed_paper_items
        }
        multi_notes_ok = (
            len(click_only_lasso) == 0
            and len(real_drag_lasso) == 8
            and len(multi_paper_cmds) >= 2
            and len({cmd_item.sentence_index for cmd_item in multi_paper_cmds}) >= 2
            and len(placed_paper_items) == 3
            and len(placed_positions) == 3
            and [cmd_item.sentence_index for cmd_item in placed_paper_items] == [0, 1, 2]
        )
        precise_norm_ok = (
            normalized_compact is not None
            and normalized_compact.kind == "underline"
            and normalized_huge is None
            and multi_notes_ok
        )
        temp_overlay.close()

        multitask_cmds = jarvis_services.DesktopAgentService.detect_direct_user_agent_commands(
            "open chrome and search HRM scope, then open notepad and type key points"
        )
        multitask_agent_ok = [cmd_item.kind for cmd_item in multitask_cmds] == [
            "open_app",
            "type_text",
            "open_app",
            "type_text",
        ]

        sync_extractor = jarvis_response_parser.StreamingSentenceExtractor(scale_x=1.0, scale_y=1.0)
        sync_extractor.feed_delta(
            'paper(24, 140, 320, 320, "Notes", "Core", "A -> B")\n'
            'underline(320, 200, 220, "first line")\n'
            'underline(320, 280, 220, "second line")\n'
            "First sentence explains the top line. Second sentence explains the next line.\n"
        )
        sync_extractor.flush_remaining()
        sync_parsed = sync_extractor.build_final_response()
        sync_paper_cmds = [cmd_item for cmd_item in sync_parsed.commands if cmd_item.kind == "paper"]
        sync_underline_cmds = [cmd_item for cmd_item in sync_parsed.commands if cmd_item.kind == "underline"]
        voice_sync_ok = (
            len(sync_paper_cmds) == 1
            and sync_paper_cmds[0].sentence_index == 0
            and [cmd_item.sentence_index for cmd_item in sync_underline_cmds] == [0, 1]
        )

        memory_ok = False
        with tempfile.TemporaryDirectory() as memory_temp_dir:
            temp_memory = jarvis_services.JarvisMemoryService(base_dir=memory_temp_dir)
            session_md_path = temp_memory.start_session("Selftest session")
            summary_md_path = temp_memory.get_today_summary_path()
            temp_memory.record_user_message("Explain HRM", source_mode="chat", foreground_window="Chrome")
            temp_memory.record_action("underline", "Definition", "synced@0", succeeded=True)
            temp_memory.record_assistant_response("HRM manages people cleanly.", "paper, underline")
            memory_context_text = temp_memory.build_memory_context_for_prompt()
            memory_ok = (
                os.path.exists(summary_md_path)
                and os.path.exists(session_md_path)
                and ("User" in memory_context_text)
            )

        parsed_diagram_rows = jarvis_overlay_windows._parse_paper_body_rows(
            "Planning | Recruitment | section:Key Idea | Training -> Appraisal"
        )
        diagram_notes_ok = (
            bool(parsed_diagram_rows)
            and all(row_item.kind in ("flow", "section") for row_item in parsed_diagram_rows)
            and (jarvis_response_parser.should_use_visual_explanation_notes("what is my storage left") is False)
            and (jarvis_response_parser.should_use_visual_explanation_notes("explain the scope of HRM") is True)
            and (jarvis_response_parser.should_use_visual_explanation_notes("explain this diagram") is True)
        )

        web_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "what is the nvidia stock price"
        )
        storage_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "what is my storage left"
        )
        natural_storage_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "All right. How much storage I have got?"
        )
        side_those_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "keep those tabs side by side. And show me."
        )
        side_two_sites_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "Open Chrome. Then keep Fast.com and Nvidia stock price side by side."
        )
        multitab_raw_cmds = jarvis_services.DesktopAgentService.detect_direct_user_agent_commands(
            "Open Chrome. And search fast.com. And another tab, open Nvidia stock"
        )
        multitab_coalesced_cmds = jarvis_services.DesktopAgentService.coalesce_agent_commands(multitab_raw_cmds)
        admin_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "delete all system files"
        )
        intent_router_ok = (
            web_intent.intent_kind == "web_lookup_and_answer"
            and ("nvidia" in web_intent.search_query.lower())
            and storage_intent.intent_kind == "system_telemetry_and_settings"
            and natural_storage_intent.intent_kind == "system_telemetry_and_settings"
            and ("GB" in natural_storage_intent.telemetry_response)
            and side_those_intent.intent_kind == "multitask_agent"
            and len(side_those_intent.agent_commands) == 1
            and side_those_intent.agent_commands[0].kind == "open_side_by_side"
            and side_those_intent.agent_commands[0].body == ""
            and side_those_intent.agent_commands[0].header == ""
            and side_two_sites_intent.intent_kind == "multitask_agent"
            and len(side_two_sites_intent.agent_commands) == 1
            and side_two_sites_intent.agent_commands[0].kind == "open_side_by_side"
            and ("fast.com" in side_two_sites_intent.agent_commands[0].body.lower())
            and ("nvidia" in side_two_sites_intent.agent_commands[0].header.lower())
            and ([cmd_item.kind for cmd_item in multitab_raw_cmds] == ["open_app", "open_url", "search_web"])
            and ([cmd_item.kind for cmd_item in multitab_coalesced_cmds] == ["open_url", "search_web"])
            and (multitab_coalesced_cmds[0].header == "new_window")
            and ("GB" in jarvis_services.SystemTelemetryService.get_storage_summary())
            and admin_intent.intent_kind == "admin_permission_gate"
            and (admin_intent.admin_request is not None)
        )

        intro_block_parsed = jarvis_response_parser.parse_model_response(
            'underline(80, 168, 490, "1. Artificial Intelligence (AI)")\n'
            'underline(80, 202, 520, "2. Remote Working")\n'
            'underline(80, 236, 430, "3. HR Analytics")\n'
            "Emerging trends in HRM are new practices shaping organizations.\n"
            "First, Artificial Intelligence speeds up resume screening.\n"
            "Second, Remote Working lets staff work from home.\n"
            "Third, HR Analytics turns employee metrics into better decisions.\n"
        )
        post_sentence_parsed = jarvis_response_parser.parse_model_response(
            "Artificial Intelligence helps in resume screening and recruitment.\n"
            'underline(80, 168, 490, "1. Artificial Intelligence (AI)")\n'
            "Remote Working allows employees to work from home.\n"
            'underline(80, 202, 520, "2. Remote Working")\n'
        )
        semantic_sync_ok = (
            [cmd_item.sentence_index for cmd_item in intro_block_parsed.commands if cmd_item.kind == "underline"] == [1, 2, 3]
            and [cmd_item.sentence_index for cmd_item in post_sentence_parsed.commands if cmd_item.kind == "underline"] == [0, 1]
            and (
                jarvis_response_parser.command_label_matches_sentence(
                    "2. Remote Working",
                    "1. Artificial Intelligence helps in resume screening.",
                ) < 0
            )
            and (
                jarvis_response_parser.command_label_matches_sentence(
                    "2. Remote Working",
                    "Remote working lets employees work from home instead of the office.",
                ) > 0
            )
        )

        from PIL import Image as PilImage, ImageDraw as PilImageDraw

        pdf_shot = PilImage.new("RGB", (1536, 864), (48, 48, 50))
        pdf_draw = PilImageDraw.Draw(pdf_shot)
        pdf_draw.rectangle([40, 40, 1050, 820], fill=(252, 252, 252))
        pdf_draw.line([(1050, 40), (1050, 820)], fill=(80, 80, 80), width=1)
        for col_cursor in range(120, 380, 7):
            pdf_draw.rectangle([col_cursor, 302, col_cursor + 2, 314], fill=(20, 20, 22))
        for col_cursor in range(420, 890, 7):
            pdf_draw.rectangle([col_cursor, 302, col_cursor + 2, 314], fill=(20, 20, 22))

        prev_logical_img = jarvis_services.ScreenCaptureService._last_raw_logical_image
        try:
            jarvis_services.ScreenCaptureService._last_raw_logical_image = pdf_shot
            jarvis_services.ScreenCaptureService.reset_underline_baselines()
            margin_cmd = jarvis_response_parser.DrawingCommand(
                kind="underline",
                x=1060.0,
                y=318.0,
                width=280.0,
                label="Job Design and Recruitment",
            )
            overlap_cmd = jarvis_response_parser.DrawingCommand(
                kind="underline",
                x=750.0,
                y=318.0,
                width=480.0,
                label="Recruitment",
            )
            margin_snapped = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(margin_cmd)
            jarvis_services.ScreenCaptureService.reset_underline_baselines()
            overlap_snapped = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(overlap_cmd)
            pdf_margin_snap_ok = (
                margin_snapped.x >= 110.0
                and (margin_snapped.x + margin_snapped.width) < 1040.0
                and overlap_snapped.x >= 110.0
                and (overlap_snapped.x + overlap_snapped.width) < 1040.0
            )
        finally:
            jarvis_services.ScreenCaptureService._last_raw_logical_image = prev_logical_img
            jarvis_services.ScreenCaptureService.reset_underline_baselines()

        temp_norm_overlay = jarvis_overlay_windows.OverlayCanvasWindow(settings_store)
        norm_pt = temp_norm_overlay._normalize_command_for_drawing_mode(
            jarvis_response_parser.DrawingCommand(
                kind="point",
                x=300.0,
                y=320.0,
                label="Job Design and Recruitment",
                sentence_index=1,
            )
        )
        norm_ring = temp_norm_overlay._normalize_command_for_drawing_mode(
            jarvis_response_parser.DrawingCommand(
                kind="ring",
                x=200.0,
                y=260.0,
                width=90.0,
                height=90.0,
                label="Unit I Topic",
                sentence_index=0,
            )
        )
        temp_norm_overlay.close()
        labeled_point_norm_ok = (
            norm_pt is not None
            and norm_pt.kind == "underline"
            and norm_pt.label == "Job Design and Recruitment"
            and norm_pt.sentence_index == 1
            and norm_ring is not None
            and norm_ring.kind == "underline"
            and norm_ring.label == "Unit I Topic"
            and norm_ring.sentence_index == 0
        )

        parsed_win_cmd = jarvis_response_parser.parse_command_line('window_action("focus", "Visual Studio Code")')
        parsed_clip_cmd = jarvis_response_parser.parse_command_line('set_clipboard("Hello Jarvis")')
        parsed_file_cmd = jarvis_response_parser.parse_command_line('write_file("Desktop/test_note.md", "body text")')
        parsed_sys_cmd = jarvis_response_parser.parse_command_line('system_control("volume_up")')
        parsed_shell_cmd = jarvis_response_parser.parse_command_line('run_shell_read("(Get-PSDrive C).Free", "Disk")')
        blocked_shell_ok, _blocked_shell_msg = jarvis_services.DesktopAgentService.execute_agent_command(
            jarvis_response_parser.DrawingCommand(kind="run_shell_read", body="Remove-Item C:\\Windows\\test.txt", header="Unsafe")
        )
        blocked_path_ok, _blocked_path_msg = jarvis_services.DesktopAgentService.execute_agent_command(
            jarvis_response_parser.DrawingCommand(kind="write_file", body="C:\\Windows\\bad.bat", header="echo hi")
        )
        blocked_ext_ok, _blocked_ext_msg = jarvis_services.DesktopAgentService.execute_agent_command(
            jarvis_response_parser.DrawingCommand(kind="write_file", body="Desktop/evil.ps1", header="dir")
        )
        safe_shell_ok, safe_shell_msg = jarvis_services.DesktopAgentService.execute_agent_command(
            jarvis_response_parser.DrawingCommand(kind="run_shell_read", body="Write-Output 'JarvisShellOk'", header="SelfCheck")
        )
        universal_agent_ok = (
            parsed_win_cmd is not None and parsed_win_cmd.kind == "window_action" and parsed_win_cmd.body == "focus" and parsed_win_cmd.header == "Visual Studio Code"
            and parsed_clip_cmd is not None and parsed_clip_cmd.kind == "set_clipboard" and parsed_clip_cmd.body == "Hello Jarvis"
            and parsed_file_cmd is not None and parsed_file_cmd.kind == "write_file" and parsed_file_cmd.body == "Desktop/test_note.md"
            and parsed_sys_cmd is not None and parsed_sys_cmd.kind == "system_control" and parsed_sys_cmd.body == "volume_up"
            and parsed_shell_cmd is not None and parsed_shell_cmd.kind == "run_shell_read" and "Get-PSDrive" in parsed_shell_cmd.body
            and (not blocked_shell_ok)
            and (not blocked_path_ok)
            and (not blocked_ext_ok)
            and safe_shell_ok
            and ("JarvisShellOk" in safe_shell_msg)
        )

        twitter_cmds = jarvis_services.DesktopAgentService.detect_direct_user_agent_commands("To open Twitter.")
        him_search_cmds = jarvis_services.DesktopAgentService.detect_direct_user_agent_commands("Search for him.")
        person_search_cmds = jarvis_services.DesktopAgentService.detect_direct_user_agent_commands("Search for Konidela Pavan Tej")
        multi_tab_side_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "Open fast.com in one tab in Chrome and in another tab open Nvidia stock. And keep those 2 tabs side by side."
        )
        raw_dsml_and_warning = (
            "<system_warning>Warning: The system prompt has changed.</system_warning> "
            "<system_warning>Warning: Prompt injection detected.</system_warning>\n"
            "<\uff5cDSML\uff5c function_calls>\n"
            '<\uff5cDSML\uff5c invoke name="open_url">'
            '<\uff5cDSML\uff5c parameter name="url" string="true">https://x.com</\uff5cDSML\uff5c parameter>'
            "</\uff5cDSML\uff5c invoke>\n"
            "</\uff5cDSML\uff5c function_calls>\n"
            "Opening Twitter for you now."
        )
        parsed_dsml_turn = jarvis_response_parser.parse_model_response(raw_dsml_and_warning)
        dsml_extractor = jarvis_response_parser.StreamingSentenceExtractor()
        for chunk_idx in range(0, len(raw_dsml_and_warning), 17):
            dsml_extractor.feed_delta(raw_dsml_and_warning[chunk_idx : chunk_idx + 17])
        dsml_extractor.flush_remaining()
        stream_dsml_turn = dsml_extractor.build_final_response()
        google_alias_ok = False
        with tempfile.TemporaryDirectory() as google_temp_dir:
            alias_env_path = os.path.join(google_temp_dir, ".env")
            alias_settings_path = os.path.join(google_temp_dir, "user_settings.json")
            with open(alias_env_path, "w", encoding="utf-8") as alias_env_file:
                alias_env_file.write("GEMINI_API_KEY=AIzaTestAliasKey\nASSEMBLYAI_API_KEY=aai_test_key\n")
            alias_store = jarvis_config.SettingsStore(
                settings_file_path=alias_settings_path,
                env_file_path=alias_env_path,
            )
            resolved_tuple = jarvis_services.VisionBrainService.resolve_active_vision_provider_and_model(alias_store)
            google_alias_ok = (
                alias_store.get_secret(jarvis_config.SECRET_GOOGLE_API_KEY) == "AIzaTestAliasKey"
                and alias_store.get("chat_model") == "gemini-3.8-flash"
                and resolved_tuple == ("google", "gemini-3.8-flash", "AIzaTestAliasKey")
            )
        hydrogen_chain = jarvis_services.VisionBrainService.get_google_model_fallback_chain("gemini-4-hydrogen")
        gemini_payload = jarvis_services.VisionBrainService.build_google_gemini_payload(
            system_prompt="System rules",
            history_messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            request_user_text="What is on screen?",
            jpeg_b64="AAAA",
            max_output_tokens=256,
        )
        gemini_delta_text = jarvis_services.VisionBrainService.extract_google_sse_delta_text(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"thought": True, "text": "internal thought hidden"},
                                {"text": "Visible answer."},
                            ]
                        }
                    }
                ]
            }
        )
        google_gemini_ok = (
            jarvis_config.is_google_gemini_model("gemini-4-hydrogen")
            and jarvis_config.is_google_gemini_model("gemini-3.8-flash")
            and (not jarvis_config.is_google_gemini_model("deepseek-ai/DeepSeek-V4.1-Flash"))
            and any(choice_item["model_id"] == "gemini-4-hydrogen" for choice_item in jarvis_config.CHAT_MODEL_CHOICES)
            and any(choice_item["model_id"] == "gemini-3.8-flash" for choice_item in jarvis_config.CHAT_MODEL_CHOICES)
            and google_alias_ok
            and (hydrogen_chain[0] == "gemini-4-hydrogen")
            and ("gemini-3.8-flash" in hydrogen_chain)
            and ("gemini-2.5-flash" in hydrogen_chain)
            and ("systemInstruction" in gemini_payload)
            and (len(gemini_payload["contents"]) == 3)
            and (gemini_payload["contents"][-1]["parts"][1]["inlineData"]["data"] == "AAAA")
            and (gemini_payload["generationConfig"]["maxOutputTokens"] == 256)
            and (gemini_delta_text == "Visible answer.")
        )

        rich_archetypes_ok = (
            jarvis_response_parser.classify_visual_explanation_pattern("compare RAM vs ROM")["archetype"] == "comparison"
            and jarvis_response_parser.classify_visual_explanation_pattern("types of cloud services")["archetype"] == "hierarchy"
            and jarvis_response_parser.classify_visual_explanation_pattern("when to choose quick sort")["archetype"] == "decision"
            and jarvis_response_parser.classify_visual_explanation_pattern("explain photosynthesis steps")["archetype"] == "process"
        )
        rich_body = "compare: RAM vs ROM | tree: Memory > Primary, Cache | if: Volatile? -> RAM / ROM | clue:R=Reset"
        rich_parsed_rows = jarvis_overlay_windows._parse_paper_body_rows(rich_body)
        rich_paper_pixmap = jarvis_overlay_windows._render_paper_card_to_pixmap(
            jarvis_response_parser.DrawingCommand(
                kind="paper",
                x=40.0,
                y=92.0,
                width=310.0,
                height=340.0,
                label="Memory Map",
                header="Visual Study",
                body=rich_body,
            ),
            seed_value=7,
        )
        mentor_prompt_text = jarvis_response_parser.build_vision_system_prompt(1536, 864)
        mentor_visuals_ok = (
            rich_archetypes_ok
            and [row_item.kind for row_item in rich_parsed_rows] == ["compare", "tree", "decision", "clue"]
            and (not rich_paper_pixmap.isNull())
            and ("5-Mark" in mentor_prompt_text)
            and ("Simple Explanation -> Visual Representation" in mentor_prompt_text)
            and ("tree:" in mentor_prompt_text)
            and ("compare:" in mentor_prompt_text)
        )

        # Universal API endpoints, keys, protocol detection & multi-format SSE delta extraction checks
        proto_checks_ok = (
            jarvis_config.detect_wire_protocol("https://api.openai.com/v1", "gpt-4o", "") == "openai"
            and jarvis_config.detect_wire_protocol("https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4", "") == "openai"
            and jarvis_config.detect_wire_protocol("https://api.anthropic.com", "claude-sonnet-4-20250514", "") == "anthropic"
            and jarvis_config.detect_wire_protocol("https://generativelanguage.googleapis.com", "gemini-2.5-flash", "") == "google"
            and jarvis_config.detect_wire_protocol("http://localhost:11434/v1", "llama3.2-vision", "") == "openai"
        )
        url_norm_ok = (
            jarvis_config.normalize_endpoint_url("https://api.openai.com", "openai", "gpt-4o")
            == "https://api.openai.com/v1/chat/completions"
            and jarvis_config.normalize_endpoint_url("http://localhost:11434/v1", "openai", "llama3.2-vision")
            == "http://localhost:11434/v1/chat/completions"
            and jarvis_config.normalize_endpoint_url("https://openrouter.ai/api/v1", "openai", "openai/gpt-4o")
            == "https://openrouter.ai/api/v1/chat/completions"
            and jarvis_config.normalize_endpoint_url("https://api.anthropic.com", "anthropic", "claude-sonnet-4-20250514")
            == "https://api.anthropic.com/v1/messages"
            and jarvis_config.normalize_endpoint_url("https://generativelanguage.googleapis.com", "google", "gemini-2.5-flash")
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
            and jarvis_config.normalize_endpoint_url("https://proxy.example.com/custom/chat", "openai", "gpt-4o")
            == "https://proxy.example.com/custom/chat"
        )
        anthropic_payload = jarvis_services.VisionBrainService.build_anthropic_messages_payload(
            system_prompt="Help the student.",
            history_messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            request_user_text="Explain this diagram",
            jpeg_b64="ZmFrZWpwZWc=",
            max_output_tokens=600,
            model_id="claude-sonnet-4-20250514",
        )
        anthropic_payload_ok = (
            anthropic_payload.get("model") == "claude-sonnet-4-20250514"
            and anthropic_payload.get("system") == "Help the student."
            and len(anthropic_payload.get("messages", [])) == 3
            and isinstance(anthropic_payload["messages"][2]["content"], list)
            and anthropic_payload["messages"][2]["content"][0]["type"] == "image"
        )
        delta_extract_ok = (
            jarvis_services.VisionBrainService.extract_universal_sse_delta_text(
                {"choices": [{"delta": {"content": "OpenAI chunk"}}]}, "openai"
            ) == "OpenAI chunk"
            and jarvis_services.VisionBrainService.extract_universal_sse_delta_text(
                {"choices": [{"message": {"content": "Non-SSE reply"}}]}, "openai"
            ) == "Non-SSE reply"
            and jarvis_services.VisionBrainService.extract_universal_sse_delta_text(
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Claude chunk"}}, "anthropic"
            ) == "Claude chunk"
            and jarvis_services.VisionBrainService.extract_universal_sse_delta_text(
                {"content": [{"type": "text", "text": "Claude full"}]}, "anthropic"
            ) == "Claude full"
            and jarvis_services.VisionBrainService.extract_universal_sse_delta_text(
                {"candidates": [{"content": {"parts": [{"text": "skip", "thought": True}, {"text": "Gemini chunk"}]}}]},
                "google",
            ) == "Gemini chunk"
            and jarvis_services.VisionBrainService.extract_universal_sse_delta_text(
                {"message": {"role": "assistant", "content": "Ollama chunk"}}, "openai"
            ) == "Ollama chunk"
        )
        env_endpoint_resolution_ok = False
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_env_file = os.path.join(temp_config_dir, ".env")
            temp_settings_file = os.path.join(temp_config_dir, "user_settings.json")
            with open(temp_env_file, "w", encoding="utf-8") as temp_env_handle:
                temp_env_handle.write(
                    "CUSTOM_API_BASE_URL=http://localhost:11434/v1\n"
                    "CUSTOM_MODEL=llama3.2-vision\n"
                )
            temp_store_local = jarvis_config.SettingsStore(
                settings_file_path=temp_settings_file,
                env_file_path=temp_env_file,
                use_process_env=False,
            )
            local_endpoints = jarvis_services.VisionBrainService.resolve_all_vision_endpoints(temp_store_local)
            with open(temp_env_file, "w", encoding="utf-8") as temp_env_handle:
                temp_env_handle.write("ANTHROPIC_API_KEY=sk-ant-test1234567890\n")
            temp_store_anthropic = jarvis_config.SettingsStore(
                settings_file_path=os.path.join(temp_config_dir, "settings_anth.json"),
                env_file_path=temp_env_file,
                use_process_env=False,
            )
            anth_endpoints = jarvis_services.VisionBrainService.resolve_all_vision_endpoints(temp_store_anthropic)
            env_endpoint_resolution_ok = (
                temp_store_local.has_any_vision_credentials()
                and bool(local_endpoints)
                and local_endpoints[0]["endpoint_url"] == "http://localhost:11434/v1/chat/completions"
                and local_endpoints[0]["model_id"] == "llama3.2-vision"
                and local_endpoints[0]["wire_protocol"] == "openai"
                and bool(anth_endpoints)
                and anth_endpoints[0]["wire_protocol"] == "anthropic"
                and anth_endpoints[0]["endpoint_url"] == "https://api.anthropic.com/v1/messages"
                and "claude-" in anth_endpoints[0]["model_id"]
            )
        universal_api_ok = (
            proto_checks_ok
            and url_norm_ok
            and anthropic_payload_ok
            and delta_extract_ok
            and env_endpoint_resolution_ok
        )

        # Upgraded Jarvis UIA, D:\Projects workspace builder, and Plan-Act-Observe checks
        uia_summary_text = jarvis_services.WindowsUIAService.format_ui_summary(max_elements=6)
        projects_found = jarvis_services.ProjectWorkspaceService.list_projects()
        nectar_resolved = jarvis_services.ProjectWorkspaceService.resolve_project_dir("nectar link")
        parsed_build_turn = jarvis_response_parser.parse_model_response(
            'write_project_file("demo-app/main.py", "print(42)")\nCreating demo app.'
        )
        open_proj_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "open my Nectarlink project in VS Code"
        )
        build_proj_intent = jarvis_services.DesktopAgentService.classify_contextual_intent(
            "build me a todo web app in D:\\Projects"
        )
        with tempfile.TemporaryDirectory() as temp_workspace_root:
            jail_escape_ok, _ = jarvis_services.ProjectWorkspaceService.write_workspace_file(
                r"C:\Windows\System32\evil.txt",
                "bad",
                projects_root=temp_workspace_root,
            )
            jail_dotdot_ok, _ = jarvis_services.ProjectWorkspaceService.write_workspace_file(
                "../escape.txt",
                "bad",
                projects_root=temp_workspace_root,
            )
            jail_env_ok, _ = jarvis_services.ProjectWorkspaceService.write_workspace_file(
                "demo/.env",
                "SECRET=1",
                projects_root=temp_workspace_root,
            )
            jail_exe_ok, _ = jarvis_services.ProjectWorkspaceService.write_workspace_file(
                "demo/app.exe",
                "MZ",
                projects_root=temp_workspace_root,
            )
            write_py_ok, _ = jarvis_services.ProjectWorkspaceService.write_workspace_file(
                "demo/hello.py",
                "print('WORKSPACE_RUN_OK')",
                projects_root=temp_workspace_root,
            )
            read_py_ok, read_py_body = jarvis_services.ProjectWorkspaceService.read_workspace_file(
                "demo/hello.py",
                projects_root=temp_workspace_root,
            )
            run_py_ok, run_py_out = jarvis_services.ProjectWorkspaceService.run_project_command(
                "python hello.py",
                cwd_rel="demo",
                projects_root=temp_workspace_root,
            )
            run_blocked_ok, _ = jarvis_services.ProjectWorkspaceService.run_project_command(
                "rm -rf /",
                cwd_rel="demo",
                projects_root=temp_workspace_root,
            )
        jarvis_agent_upgrade_ok = (
            isinstance(uia_summary_text, str)
            and bool(uia_summary_text)
            and ("clicky" in [name.lower() for name in projects_found])
            and (nectar_resolved is not None and "nectarlink" in nectar_resolved.lower())
            and any(
                cmd_item.kind == "write_project_file" and cmd_item.body == "demo-app/main.py"
                for cmd_item in parsed_build_turn.commands
            )
            and open_proj_intent.intent_kind == "multitask_agent"
            and bool(open_proj_intent.agent_commands)
            and open_proj_intent.agent_commands[0].kind == "open_project"
            and build_proj_intent.intent_kind == "autonomous_workspace_agent"
            and (not jail_escape_ok)
            and (not jail_dotdot_ok)
            and (not jail_env_ok)
            and (not jail_exe_ok)
            and write_py_ok
            and read_py_ok
            and ("WORKSPACE_RUN_OK" in read_py_body)
            and run_py_ok
            and ("WORKSPACE_RUN_OK" in run_py_out)
            and (not run_blocked_ok)
        )

        # Cloned Roamies capabilities verification
        hands_wins = jarvis_services.RoamiesHandsService.list_windows()
        hands_front = jarvis_services.RoamiesHandsService.locate_window("front")
        beats_player = jarvis_services.RoamiesVoiceBeatsPlayer(
            settings_store,
            jarvis_services.SoundEffectsPlayer(settings_store),
        )
        beats_dir_exists = os.path.isdir(settings_store.get_roamies_beats_dir())
        beat_rate, beat_pcm = beats_player.extract_pcm_bytes("On it.")
        beats_ok = (beat_rate == 24000 and len(beat_pcm) > 1000) if beats_dir_exists else True

        roamies_marks_script = (
            'plan_steps("Inspect window | Write file | Open VS Code", "1")\n'
            'target(300, 220, 28, "Run Button")\n'
            'strike(260, 260, 140, "old label")\n'
            'check(420, 220, "Valid")\n'
            'cross(460, 220, "Invalid")\n'
            'num(220, 220, "1")\n'
            'connect(260, 260, 480, 320, "flow")\n'
            'spotlight(240, 200, 180, 70, "Focus")\n'
            'edit_drawing("recolor", "p1", "amber")\n'
            'Applying Roamies visual marks now.'
        )
        parsed_roamies_marks = jarvis_response_parser.parse_model_response(roamies_marks_script)
        roamies_card_body = (
            "bars: Python=95% ; Rust=99% | "
            "timeline: v1 -> Launch ; v2 -> UIA3 | "
            "table: Feature, Basic, Pro ; Voice, yes, yes ; UIA, no, yes | "
            "math: F = m * a => Force scales with mass | "
            "mindmap: Agent -> Sight, Hands, Memory | "
            "quiz: LIFO? *Stack ; Queue :: Stack pops last item | "
            "list: [ ] Review code ; [x] Run tests"
        )
        parsed_roamies_rows = jarvis_overlay_windows._parse_paper_body_rows(roamies_card_body)
        roamies_pixmap = jarvis_overlay_windows._render_paper_card_to_pixmap(
            jarvis_response_parser.DrawingCommand(
                kind="paper",
                x=32.0,
                y=96.0,
                width=330.0,
                height=390.0,
                label="Roamies Visuals",
                header="Multi-Mode Card",
                body=roamies_card_body,
            ),
            seed_value=19,
        )
        roamies_overlay = jarvis_overlay_windows.OverlayCanvasWindow(settings_store)
        for mark_cmd in parsed_roamies_marks.commands:
            if mark_cmd.kind in ("target", "strike", "check", "cross", "num", "connect", "spotlight"):
                roamies_overlay.add_drawing_command(mark_cmd)
        roamies_overlay.set_plan_steps(["Inspect window", "Write file", "Open VS Code"], active_index=1)
        recolor_ok = roamies_overlay.apply_drawing_edit("recolor", "p1", "amber")
        erase_ok = roamies_overlay.apply_drawing_edit("erase", "p1", "")
        clear_ok = roamies_overlay.apply_drawing_edit("clear", "", "")
        roamies_overlay.clear_plan_steps()
        roamies_overlay.close()

        with tempfile.TemporaryDirectory() as roamies_tmp_dir:
            commitments_svc = jarvis_services.CommitmentsReminderService(
                storage_path=os.path.join(roamies_tmp_dir, "commitments.json")
            )
            added_c = commitments_svc.add_commitment("Push Jarvis PR", due_str="in 1 minutes", repeat_str="daily")
            snooze_ok, _ = commitments_svc.snooze_commitment("15")
            card_preview = commitments_svc.format_today_later_card_body()
            due_immediate = commitments_svc.pop_due_commitments(now_ts=time.time() + 1800.0)

            focus_svc = jarvis_services.FocusGuardianService(settings_store)
            focus_start_ok, _ = focus_svc.start_focus(25.0, "Study DSA")
            base_focus_ts = time.time()
            nudge_early = focus_svc.check_foreground_distraction("YouTube - Music", now_ts=base_focus_ts)
            nudge_late = focus_svc.check_foreground_distraction("YouTube - Music", now_ts=base_focus_ts + 25.0)
            focus_stop_ok, _ = focus_svc.stop_focus()

            skills_svc = jarvis_services.SkillsManagerService(skills_dir=os.path.join(roamies_tmp_dir, "skills"))
            skill_save_ok, skill_slug, _ = skills_svc.save_skill("Deploy App", "Deploy build steps", "1. Build\n2. Verify")
            skill_bad_ok, _bad_slug, _ = skills_svc.save_skill("Bad Skill", "Bad", "rm -rf /")
            skill_read_text = skills_svc.read_skill("Deploy App") or ""
            skill_del_ok = skills_svc.delete_skill(skill_slug)

            jobs_svc = jarvis_services.ScheduledJobsService(
                storage_path=os.path.join(roamies_tmp_dir, "jobs.json")
            )
            job_entry = jobs_svc.schedule_job("Morning Brief", "Summarize AI news", due_str="in 1 minutes", repeat_str="daily")
            due_jobs = jobs_svc.take_due_jobs(now_ts=time.time() + 1800.0)
            jobs_svc.mark_job_result(job_entry["id"], "Summarized 5 articles", succeeded=True)

            facts_svc = jarvis_services.LearnedFactsService(
                storage_path=os.path.join(roamies_tmp_dir, "facts.json")
            )
            learned_row = facts_svc.learn_fact("User is studying CS")
            facts_prompt = facts_svc.format_for_prompt()
            forgot_ok, _ = facts_svc.forget_fact(learned_row["id"])

        roamies_clone_ok = (
            isinstance(hands_wins, list)
            and (hands_front is None or isinstance(hands_front, dict))
            and beats_ok
            and len(parsed_roamies_marks.commands) >= 9
            and ("applying roamies visual marks now." in parsed_roamies_marks.spoken_text.lower())
            and [row_item.kind for row_item in parsed_roamies_rows] == [
                "bars", "timeline", "table", "math", "mindmap", "quiz", "list"
            ]
            and (not roamies_pixmap.isNull())
            and recolor_ok
            and erase_ok
            and clear_ok
            and bool(added_c.get("id"))
            and snooze_ok
            and ("Push Jarvis PR" in card_preview)
            and len(due_immediate) == 1
            and focus_start_ok
            and (nudge_early is None)
            and (nudge_late is not None and "youtube" in nudge_late.lower())
            and focus_stop_ok
            and skill_save_ok
            and (not skill_bad_ok)
            and ("Build" in skill_read_text)
            and skill_del_ok
            and len(due_jobs) == 1
            and ("User is studying CS" in facts_prompt)
            and forgot_ok
        )

        import jarvis_roamies_core
        import jarvis_roamies_visuals
        from PySide6.QtCore import QRectF as _SelftestRectF
        from PySide6.QtGui import QColor as _SelftestColor, QImage as _SelftestImage, QPainter as _SelftestPainter

        # 14-point full KenKaiii/Roamies GitHub parity verification (offline & deterministic)
        avatar_img = _SelftestImage(96, 96, _SelftestImage.Format.Format_ARGB32_Premultiplied)
        avatar_img.fill(0)
        avatar_painter = _SelftestPainter(avatar_img)
        for sample_look in ("jarvis", "anime", "glayos"):
            jarvis_roamies_visuals.paint_companion_avatar(
                avatar_painter,
                48.0,
                48.0,
                sample_look,
                "default",
                0.8,
                0.4,
                "speaking",
                "happy",
                _SelftestColor(jarvis_theme.JARVIS_BLUE_HEX),
            )
        avatar_painter.end()
        looks_ok = (len(jarvis_roamies_core.LOOK_PERSONAS) == 9) and (not avatar_img.isNull())

        cube_parts = jarvis_roamies_visuals.resolve_3d_preset_or_parts("cube")
        model_renderer = jarvis_roamies_visuals.Jarvis3DModelRenderer(cube_parts)
        render3d_img = _SelftestImage(140, 140, _SelftestImage.Format.Format_ARGB32_Premultiplied)
        render3d_img.fill(0)
        render3d_painter = _SelftestPainter(render3d_img)
        model_renderer.paint_into_rect(render3d_painter, _SelftestRectF(10.0, 10.0, 120.0, 120.0), 0.4)
        render3d_painter.end()
        card3d_pm = jarvis_overlay_windows._render_paper_card_to_pixmap(
            jarvis_response_parser.DrawingCommand(
                kind="paper",
                x=40.0,
                y=40.0,
                width=360.0,
                height=320.0,
                label="3D Test",
                header="Cube & Visuals",
                body=(
                    "model3d:preset=cube|caption=Cube | "
                    "molecule:smiles=CCO|caption=Ethanol | "
                    "graph:expr=x**2|caption=Parabola | "
                    "map:region=Europe|highlight=France | "
                    "geometry:shapes=triangle 0,0 100,0 50,80 | "
                    "sources:NASA|https://nasa.gov | "
                    "clip:https://youtu.be/demo|10|25|Clip | "
                    "legend:Blue=Core;Amber=Warn | "
                    "ticket:JR-42|In Progress|High | "
                    "medal:Gold|Top Score"
                ),
            ),
            seed_value=23,
        )
        visual_cards_ok = (
            bool(cube_parts)
            and (not render3d_img.isNull())
            and (not card3d_pm.isNull())
        )

        norm_math = jarvis_roamies_core.normalize_math_for_speech("Calculate 3/2 + x^2")
        cue_val = jarvis_roamies_core.cue_feel("[happy] Great work!")
        gem_text, _gem_style = jarvis_roamies_core.convert_cues_for_gemini("[sighing] Great work 3/2!")
        clean_disp = jarvis_roamies_core.strip_voice_cues_for_display("[warm] Great work!")
        spoken_ok = (
            ("joy" == cue_val)
            and ("[warm]" not in clean_disp)
            and ("<sigh>" in gem_text.lower())
            and (norm_math != "Calculate 3/2 + x^2")
            and jarvis_roamies_core.is_rtl_text("שלום עולם")
            and jarvis_roamies_core.is_rtl_text("مرحبا بكم")
        )

        write_blk, _ = jarvis_roamies_core.JarvisSafetyEngine.check_write("~/.ssh/authorized_keys")
        click_ask, _ = jarvis_roamies_core.JarvisSafetyEngine.check_click("Delete forever")
        url_ask, _ = jarvis_roamies_core.JarvisSafetyEngine.check_url("http://192.168.1.50")
        url_ok, _ = jarvis_roamies_core.JarvisSafetyEngine.check_url("https://docs.python.org")
        safety_ok = (
            write_blk == jarvis_roamies_core.JarvisSafetyEngine.BLOCK
            and click_ask == jarvis_roamies_core.JarvisSafetyEngine.ASK
            and url_ask == jarvis_roamies_core.JarvisSafetyEngine.ASK
            and url_ok == jarvis_roamies_core.JarvisSafetyEngine.SAFE
        )

        with tempfile.TemporaryDirectory() as roamies_full_tmp:
            iso_dpapi = jarvis_roamies_core.WindowsDPAPISecretStore(
                store_file_path=os.path.join(roamies_full_tmp, "dpapi.json")
            )
            mcp_mgr = jarvis_roamies_core.MCPConnectionManager(
                config_file_path=os.path.join(roamies_full_tmp, "mcp.json"),
                dpapi_store=iso_dpapi,
            )
            mcp_add_ok, _ = mcp_mgr.add_connection("demo-mcp", command="echo", args=["ok"])
            mcp_rem_ok, _ = mcp_mgr.remove_connection("demo-mcp")

            crew_mgr = jarvis_roamies_core.SubagentCrewManager(
                agents_root_dir=os.path.join(roamies_full_tmp, "agents")
            )
            picked_spec = crew_mgr.specialist_for_request("Review this PR carefully", delegate_mode="auto")
            sent_ok, sent_entry, _ = crew_mgr.send_agent(
                brief="Summarize readme",
                subagent="reviewer",
                runner_fn=lambda spec_name, brief_val: f"{spec_name}: reviewed {brief_val}",
            )
            rep_ok = False
            rep_text = ""
            for _poll_idx in range(40):
                rep_ok, rep_text = crew_mgr.get_agent_report(str(sent_entry.get("id") or ""))
                if rep_ok and ("reviewed" in rep_text.lower()):
                    break
                time.sleep(0.025)

            prev_sync_state_path = jarvis_roamies_core.DEFAULT_SYNC_STATE_PATH
            jarvis_roamies_core.DEFAULT_SYNC_STATE_PATH = os.path.join(roamies_full_tmp, "sync_state.json")
            try:
                sync_c = jarvis_services.CommitmentsReminderService(storage_path=os.path.join(roamies_full_tmp, "sc.json"))
                sync_c.add_commitment("Sync parity item", due_str="tomorrow", repeat_str="every 2 hours")
                sync_f = jarvis_services.LearnedFactsService(storage_path=os.path.join(roamies_full_tmp, "sf.json"))
                sync_f.learn_fact("Testing encrypted sync")
                prep_ok, sync_key_bytes, _ = jarvis_roamies_core.EncryptedSyncService.prepare_sync_folder(
                    roamies_full_tmp,
                    "sync-passphrase-1",
                    dpapi_store=iso_dpapi,
                )
                sealed_box = jarvis_roamies_core.EncryptedSyncService.seal_json(
                    sync_key_bytes or b"0" * 32,
                    {"hello": "jarvis"},
                    "dev-a",
                )
                unsealed_box = jarvis_roamies_core.EncryptedSyncService.unseal_json(
                    sync_key_bytes or b"0" * 32,
                    sealed_box,
                    "dev-a",
                )
                pass_ok, _ = jarvis_roamies_core.EncryptedSyncService.run_sync_pass(
                    roamies_full_tmp,
                    "dev-a",
                    sync_c,
                    sync_f,
                    dpapi_store=iso_dpapi,
                )
            finally:
                jarvis_roamies_core.DEFAULT_SYNC_STATE_PATH = prev_sync_state_path

        local_ok = (
            jarvis_roamies_core.LocalModelDiscoveryService.normalize_local_server_url("http://127.0.0.1:11434")
            == "http://127.0.0.1:11434/v1"
            and jarvis_roamies_core.LocalModelDiscoveryService.normalize_local_server_url("ftp://bad") is None
            and bool(jarvis_roamies_core.LocalModelDiscoveryService.format_local_error("Ollama", "connection refused"))
            and bool(jarvis_roamies_core.LocalModelDiscoveryService.context_warning_note(4096, "Ollama"))
        )

        inline_ovr = jarvis_roamies_core.HybridModelRouter.detect_inline_model_override("use sonnet to debug")
        qwen_x, qwen_y = jarvis_roamies_core.HybridModelRouter.convert_qwen_1000_coords(500.0, 250.0, 1920.0, 1080.0)
        hybrid_ok = (
            "code" in jarvis_roamies_core.HybridModelRouter.classify_job_kinds("fix python unit test")
            and bool(inline_ovr)
            and jarvis_roamies_core.HybridModelRouter.is_qwen_1000_grid_model("qwen3-vl-32b")
            and abs(qwen_x - 960.0) < 1.0
            and abs(qwen_y - 270.0) < 1.0
        )

        redacted_diag = jarvis_roamies_core.DiagnosticsExporter.redact_secrets_and_pii(
            "key=sk-proj-1234567890abcdef123456 user@example.com"
        )
        diag_ok = (
            ("sk-proj-123456" not in redacted_diag)
            and ("example.com" not in redacted_diag)
            and bool(jarvis_roamies_core.DiagnosticsExporter.build_diagnostics_report(settings_store))
        )

        rep_spec = jarvis_roamies_core.parse_repeat_schedule("every 2 hours")
        wake_valid, wake_cleaned, _ = jarvis_roamies_core.clean_wake_name("Jarvis")
        wake_bad_valid, _bad_clean, _bad_err = jarvis_roamies_core.clean_wake_name("Hey Jarvis!")
        wake_heard, wake_rem = jarvis_services.WakeWordService.matches_wake_phrase("Hey Jarvis check my screen", "Jarvis")
        brief_res = jarvis_roamies_core.build_morning_briefing_note(
            on=True,
            last_day="",
            by_voice=True,
            todoist=True,
            user_name="Ruthvik",
        )
        schedule_wake_ok = (
            rep_spec is not None
            and jarvis_roamies_core.compute_next_repeat_timestamp(rep_spec, from_ts=1000.0) == 8200.0
            and jarvis_roamies_core.parse_voice_snooze_request("snooze for 15 minutes") == 15
            and wake_valid
            and (not wake_bad_valid)
            and wake_cleaned == "Jarvis"
            and wake_heard
            and ("check my screen" in wake_rem.lower())
            and (brief_res is not None)
        )

        ask_bar_w = jarvis_roamies_visuals.CursorAskBarWindow()
        approval_w = jarvis_roamies_visuals.RiskyActionApprovalBar()
        toast_w = jarvis_roamies_visuals.AgentArtifactToastWindow()
        widgets_ok = (
            not ask_bar_w.grab().isNull()
            and not approval_w.grab().isNull()
            and not toast_w.grab().isNull()
        )
        ask_bar_w.close()
        approval_w.close()
        toast_w.close()
        ask_bar_w.deleteLater()
        approval_w.deleteLater()
        toast_w.deleteLater()
        roamies_overlay.deleteLater()
        qt_application.processEvents()

        full_cmds_turn = jarvis_response_parser.parse_model_response(
            '[happy] Drawing every full mark now.\n'
            'hover(400, 300, "Submit")\n'
            'trace(100, 100, 220, 140)\n'
            'line(10, 20, 110, 120)\n'
            'curve(10, 20, 60, 80, 120, 20)\n'
            'arrow(20, 20, 140, 80)\n'
            'circle(200, 200, 45)\n'
            'rect(300, 100, 120, 60)\n'
            'write(320, 120, "Note")\n'
            'label(320, 160, "Step 1")\n'
            'emote("happy")\n'
            'sfx("jarvis-text-open")\n'
            'button(120, 420, "Run", "ask|test")\n'
            'walkthrough_step(500, 300, 32, "Click Save", "click")\n'
            'add_connection("docs", "npx -y @demo/docs")\n'
            'send_agent("scout", "Scan project")\n'
            'about_jarvis("show")\n'
        )
        parsed_full_kinds = {cmd_item.kind for cmd_item in full_cmds_turn.commands}
        expected_full_kinds = {
            "hover", "trace", "line", "curve", "arrow", "circle", "rect",
            "write", "label", "emote", "sfx", "button", "walkthrough_step",
            "add_connection", "send_agent", "about_jarvis",
        }
        full_parser_ok = expected_full_kinds.issubset(parsed_full_kinds)

        # Comprehensive Agentic Tasks, Latency Budgets & Visualization Accuracy Verification
        cap_t0 = time.perf_counter()
        fast_jpeg_b64, fast_w, fast_h, fast_sx, fast_sy = (
            jarvis_services.ScreenCaptureService.capture_screen_with_highlight(
                highlight_points=[],
                logical_screen_size=(1280, 720),
                fast_agent_mode=True,
            )
        )
        fast_cap_ms = (time.perf_counter() - cap_t0) * 1000.0
        fast_capture_ok = (
            (fast_w == 1280 and fast_h == 720 and fast_sx == 1.0 and fast_sy == 1.0)
            and (fast_cap_ms < 220.0)
            and (len(fast_jpeg_b64) >= 0)
        )

        parabola_spec = jarvis_roamies_visuals.parse_graph_spec_from_text("expr=x**2|x_range=-3,3|caption=Parabola")
        parabola_pts = (parabola_spec.get("plots") or [{}])[0].get("points") or []
        geom_triangle_spec = jarvis_roamies_visuals.parse_geometry_spec_from_text("shapes=triangle 0,0 100,0 50,80")
        math_graph_ok = (
            len(parabola_pts) == 41
            and abs(parabola_pts[0][0] - (-3.0)) < 0.01
            and abs(parabola_pts[0][1] - 9.0) < 0.01
            and abs(parabola_pts[20][0] - 0.0) < 0.01
            and abs(parabola_pts[20][1] - 0.0) < 0.01
            and abs(parabola_pts[-1][0] - 3.0) < 0.01
            and abs(parabola_pts[-1][1] - 9.0) < 0.01
            and len(geom_triangle_spec.get("vertices") or []) == 3
            and len(geom_triangle_spec.get("lines") or []) == 3
        )

        every_cmd_parsed = jarvis_response_parser.parse_command_line('ring(320, 240, 28, "Submit", "blue", "every")')
        every_mark_ok = (
            every_cmd_parsed is not None
            and bool(every_cmd_parsed.extra_data.get("every"))
            and len(jarvis_services.ScreenCaptureService.expand_every_match_commands(every_cmd_parsed)) >= 1
        )

        with tempfile.TemporaryDirectory() as agentic_tmp_dir:
            bound_comm = jarvis_services.CommitmentsReminderService(
                storage_path=os.path.join(agentic_tmp_dir, "comm.json")
            )
            bound_skills = jarvis_services.SkillsManagerService(
                skills_dir=os.path.join(agentic_tmp_dir, "skills")
            )
            bound_jobs = jarvis_services.ScheduledJobsService(
                storage_path=os.path.join(agentic_tmp_dir, "jobs.json")
            )
            bound_facts = jarvis_services.LearnedFactsService(
                storage_path=os.path.join(agentic_tmp_dir, "facts.json")
            )
            bound_focus = jarvis_services.FocusGuardianService(settings_store)
            bound_crew = jarvis_roamies_core.SubagentCrewManager(
                agents_root_dir=os.path.join(agentic_tmp_dir, "crew")
            )
            jarvis_services.DesktopAgentService.bind_companion_services(
                commitments_service=bound_comm,
                skills_service=bound_skills,
                jobs_service=bound_jobs,
                facts_service=bound_facts,
                focus_service=bound_focus,
                subagent_manager=bound_crew,
                subagent_runner=lambda spec_name, brief_txt: f"[{spec_name}] completed {brief_txt}",
            )

            dispatch_t0 = time.perf_counter()
            cmd_rem = jarvis_response_parser.parse_command_line('remember_commitment("Ship v2.5", "today 6pm", "daily")')
            ok_rem, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_rem)
            cmd_snz = jarvis_response_parser.parse_command_line('snooze_reminder("10")')
            ok_snz, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_snz)
            cmd_lst = jarvis_response_parser.parse_command_line('show_list("today")')
            ok_lst, lst_out = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_lst)
            cmd_fstart = jarvis_response_parser.parse_command_line('start_focus("20", "Agentic Test")')
            ok_fstart, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_fstart)
            cmd_fstop = jarvis_response_parser.parse_command_line('stop_focus("done")')
            ok_fstop, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_fstop)
            cmd_sk_save = jarvis_response_parser.parse_command_line('save_skill("Fast Build", "Quick build", "1. Compile\\n2. Test")')
            ok_sk_save, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_sk_save)
            cmd_sk_use = jarvis_response_parser.parse_command_line('use_skill("Fast Build")')
            ok_sk_use, sk_out = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_sk_use)
            cmd_sk_del = jarvis_response_parser.parse_command_line('forget_skill("Fast Build")')
            ok_sk_del, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_sk_del)
            cmd_job = jarvis_response_parser.parse_command_line('schedule_job("Nightly Sync", "in 45 minutes", "Run sync")')
            ok_job, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_job)
            cmd_jlist = jarvis_response_parser.parse_command_line('list_jobs("all")')
            ok_jlist, jlist_out = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_jlist)
            cmd_fact = jarvis_response_parser.parse_command_line('learn_fact("Prefers 60 FPS dark UI")')
            ok_fact, fact_out = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_fact)
            fact_id_match = re.search(r"\[([^\]]+)\]", fact_out)
            fact_id_val = fact_id_match.group(1) if fact_id_match else "Prefers 60 FPS"
            cmd_fforget = jarvis_response_parser.parse_command_line(f'forget_fact("{fact_id_val}")')
            ok_fforget, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_fforget)
            cmd_lfiles = jarvis_response_parser.parse_command_line('list_files("clicky/windows")')
            ok_lfiles, lfiles_out = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_lfiles)
            cmd_sfiles = jarvis_response_parser.parse_command_line('search_files("JarvisApplicationController", "clicky/windows")')
            ok_sfiles, sfiles_out = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_sfiles)
            cmd_lwins = jarvis_response_parser.parse_command_line('list_windows("all")')
            ok_lwins, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_lwins)
            cmd_sub = jarvis_response_parser.parse_command_line('send_agent("builder", "Check build health")')
            ok_sub, _ = jarvis_services.DesktopAgentService.execute_agent_command_blocking(cmd_sub)
            dispatch_elapsed_ms = (time.perf_counter() - dispatch_t0) * 1000.0

            # Reset bound services after temporary directory block
            jarvis_services.DesktopAgentService.bind_companion_services()

        agentic_suite_ok = (
            fast_capture_ok
            and math_graph_ok
            and every_mark_ok
            and ok_rem
            and ok_snz
            and ok_lst
            and ("Ship v2.5" in lst_out)
            and ok_fstart
            and ok_fstop
            and ok_sk_save
            and ok_sk_use
            and ("Compile" in sk_out)
            and ok_sk_del
            and ok_job
            and ok_jlist
            and ("Nightly Sync" in jlist_out)
            and ok_fact
            and ok_fforget
            and ok_lfiles
            and ("jarvis.py" in lfiles_out)
            and ok_sfiles
            and ("jarvis_app.py" in sfiles_out)
            and ok_lwins
            and ok_sub
            and (dispatch_elapsed_ms < 2500.0)
        )

        roamies_github_full_ok = (
            looks_ok
            and visual_cards_ok
            and spoken_ok
            and safety_ok
            and mcp_add_ok
            and mcp_rem_ok
            and (picked_spec is not None)
            and sent_ok
            and rep_ok
            and ("reviewed" in rep_text.lower())
            and prep_ok
            and (unsealed_box == {"hello": "jarvis"})
            and pass_ok
            and local_ok
            and hybrid_ok
            and diag_ok
            and schedule_wake_ok
            and widgets_ok
            and full_parser_ok
            and agentic_suite_ok
        )

        regression_safety_ok = (
            len(twitter_cmds) == 1
            and twitter_cmds[0].kind == "open_url"
            and twitter_cmds[0].body == "https://x.com"
            and len(him_search_cmds) == 0
            and len(person_search_cmds) == 1
            and person_search_cmds[0].kind == "search_web"
            and person_search_cmds[0].body == "Konidela Pavan Tej"
            and multi_tab_side_intent.intent_kind == "multitask_agent"
            and len(multi_tab_side_intent.agent_commands) == 1
            and multi_tab_side_intent.agent_commands[0].kind == "open_side_by_side"
            and ("fast.com" in multi_tab_side_intent.agent_commands[0].body.lower())
            and ("nvidia" in multi_tab_side_intent.agent_commands[0].header.lower())
            and ("system_warning" not in parsed_dsml_turn.spoken_text.lower())
            and ("prompt injection" not in parsed_dsml_turn.spoken_text.lower())
            and ("dsml" not in parsed_dsml_turn.spoken_text.lower())
            and ("Opening Twitter for you now." in parsed_dsml_turn.spoken_text)
            and any(cmd_item.kind == "open_url" and cmd_item.body == "https://x.com" for cmd_item in parsed_dsml_turn.commands)
            and ("system_warning" not in stream_dsml_turn.spoken_text.lower())
            and ("prompt injection" not in stream_dsml_turn.spoken_text.lower())
            and ("dsml" not in stream_dsml_turn.spoken_text.lower())
            and ("Opening Twitter for you now." in stream_dsml_turn.spoken_text)
            and any(cmd_item.kind == "open_url" and cmd_item.body == "https://x.com" for cmd_item in stream_dsml_turn.commands)
            and google_gemini_ok
            and mentor_visuals_ok
            and universal_api_ok
            and jarvis_agent_upgrade_ok
            and roamies_clone_ok
            and roamies_github_full_ok
        )

        parser_ok = (
            "internal reasoning hidden" not in parsed_turn.spoken_text
            and len(collected_commands) >= 4
            and bool(collected_sentences)
            and learned_style is not None
            and scroll_intent_ok
            and scroll_parse_ok
            and agent_ok
            and multitask_agent_ok
            and voice_sync_ok
            and semantic_sync_ok
            and stacked_list_snap_ok
            and adaptive_scroll_ok
            and memory_ok
            and diagram_notes_ok
            and intent_router_ok
            and precise_norm_ok
            and pdf_margin_snap_ok
            and labeled_point_norm_ok
            and universal_agent_ok
            and regression_safety_ok
        )
        suite4_ok = (
            (main_mean_ms < 10.0)
            and hidden_on_close
            and main_personal_ok
            and parser_ok
            and baseline_snap_ok
            and (len(single_char_violations) == 0)
        )
        record_result(
            "Suite 4 (MainWindow & Parser)",
            suite4_ok,
            (
                f"mean_ms={main_mean_ms:.3f}ms, hidden={hidden_on_close}, "
                f"personal={main_personal_ok}, scroll_ok={scroll_intent_ok and scroll_parse_ok}, "
                f"snap_ok={baseline_snap_ok}, agent_ok={agent_ok}, multitask_ok={multitask_agent_ok}, "
                f"sync_ok={voice_sync_ok}, semantic_sync={semantic_sync_ok}, "
                f"stacked_snap={stacked_list_snap_ok}, adaptive_scroll={adaptive_scroll_ok}, "
                f"memory_ok={memory_ok}, diagram_ok={diagram_notes_ok}, "
                f"intent_ok={intent_router_ok}, precise_ok={precise_norm_ok}, "
                f"pdf_margin_ok={pdf_margin_snap_ok}, pt_norm_ok={labeled_point_norm_ok}, "
                f"univ_agent_ok={universal_agent_ok}, regr_safety_ok={regression_safety_ok}, "
                f"roamies_full={roamies_github_full_ok}, agentic_ok={agentic_suite_ok}, "
                f"fast_cap_ms={fast_cap_ms:.1f}ms, dispatch_ms={dispatch_elapsed_ms:.1f}ms, "
                f"single_char={single_char_violations[:8]}"
            ),
        )
    except Exception as exc:
        record_result("Suite 4 (MainWindow & Parser)", False, f"Exception: {exc}")

    # -----------------------------------------------------------------------
    # Suite 5: Live AssemblyAI v3 Token Acquisition
    # -----------------------------------------------------------------------
    try:
        mic_service = jarvis_services.MicrophoneStreamService(settings_store)
        token_value = mic_service._fetch_assemblyai_token()
        token_ok = len(token_value) > 12
        record_result(
            "Suite 5 (AssemblyAI v3 Token)",
            token_ok,
            f"token_length={len(token_value)}",
        )
    except Exception as exc:
        record_result("Suite 5 (AssemblyAI v3 Token)", False, f"Exception: {exc}")

    # -----------------------------------------------------------------------
    # Suite 6: Live Vision SSE (Any Configured Endpoint) + AssemblyAI PCM TTS
    # -----------------------------------------------------------------------
    try:
        candidate_attempts = jarvis_services.VisionBrainService.resolve_all_vision_endpoints(settings_store)
        if not candidate_attempts:
            raise RuntimeError("No Vision/Chat API endpoint or key found in .env or Studio & Keys.")

        test_jpeg_b64 = _build_synthetic_vision_test_jpeg_b64()
        vision_prompt = "Reply in one short sentence: what colored shape and number do you see?"
        vision_start = time.perf_counter()
        vision_answer = ""
        active_provider_label = ""
        last_vision_error = ""

        for attempt_cfg in candidate_attempts:
            attempt_provider = attempt_cfg["provider_label"]
            attempt_wire = attempt_cfg["wire_protocol"]
            attempt_endpoint = attempt_cfg["endpoint_url"]
            attempt_model = attempt_cfg["model_id"]
            attempt_key = attempt_cfg["api_key"]
            model_sequence = (
                jarvis_services.VisionBrainService.get_google_model_fallback_chain(attempt_model)
                if (attempt_wire == "google" and "generativelanguage.googleapis.com" in attempt_endpoint.lower())
                else [attempt_model]
            )
            attempt_succeeded = False
            for candidate_idx, candidate_model in enumerate(model_sequence):
                vision_extractor = jarvis_response_parser.StreamingSentenceExtractor()
                if attempt_wire == "google":
                    request_url = (
                        attempt_endpoint.format(model_id=candidate_model)
                        if "{model_id}" in attempt_endpoint
                        else jarvis_config.normalize_endpoint_url(attempt_endpoint, "google", candidate_model)
                    )
                    request_headers: Dict[str, str] = {"Content-Type": "application/json"}
                    if attempt_key:
                        request_headers["x-goog-api-key"] = attempt_key
                    request_payload = jarvis_services.VisionBrainService.build_google_gemini_payload(
                        system_prompt="",
                        history_messages=[],
                        request_user_text=vision_prompt,
                        jpeg_b64=test_jpeg_b64,
                        max_output_tokens=160,
                    )
                elif attempt_wire == "anthropic":
                    request_url = attempt_endpoint
                    request_headers = {
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    }
                    if attempt_key:
                        request_headers["x-api-key"] = attempt_key
                    request_payload = jarvis_services.VisionBrainService.build_anthropic_messages_payload(
                        system_prompt="",
                        history_messages=[],
                        request_user_text=vision_prompt,
                        jpeg_b64=test_jpeg_b64,
                        max_output_tokens=160,
                        model_id=candidate_model,
                    )
                else:
                    request_url = attempt_endpoint
                    endpoint_lower = request_url.lower()
                    request_headers = {"Content-Type": "application/json"}
                    if attempt_key:
                        if ("openai.azure.com" in endpoint_lower) or ("api-version=" in endpoint_lower):
                            request_headers["api-key"] = attempt_key
                        else:
                            request_headers["Authorization"] = f"Bearer {attempt_key}"
                    if "openrouter.ai" in endpoint_lower:
                        request_headers["HTTP-Referer"] = "https://heyclicky.com"
                        request_headers["X-Title"] = "Jarvis"
                    request_payload = {
                        "model": candidate_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vision_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{test_jpeg_b64}"}},
                                ],
                            }
                        ],
                        "max_tokens": 120,
                        "stream": True,
                    }
                    if "api.river.ai" in endpoint_lower:
                        request_payload["reasoning_effort"] = "none"

                try:
                    with requests.post(
                        request_url,
                        headers=request_headers,
                        json=request_payload,
                        stream=True,
                        timeout=30,
                    ) as vision_response:
                        if not vision_response.ok:
                            last_vision_error = (
                                f"{attempt_provider}:{candidate_model} HTTP {vision_response.status_code}: "
                                f"{vision_response.text[:160]}"
                            )
                            if (
                                attempt_wire == "google"
                                and vision_response.status_code in (400, 404)
                                and (candidate_idx + 1) < len(model_sequence)
                            ):
                                continue
                            break
                        vision_response.encoding = "utf-8"
                        plain_json_lines: List[str] = []
                        saw_sse_prefix = False
                        for sse_line in vision_response.iter_lines(decode_unicode=True):
                            if not sse_line:
                                continue
                            stripped_sse = sse_line.strip()
                            if stripped_sse.startswith("data:"):
                                saw_sse_prefix = True
                                payload_text = stripped_sse[5:].strip()
                                if payload_text == "[DONE]":
                                    break
                                try:
                                    chunk_payload = json.loads(payload_text)
                                except Exception:
                                    continue
                            elif stripped_sse.startswith("event:"):
                                saw_sse_prefix = True
                                continue
                            elif (not saw_sse_prefix) and stripped_sse.startswith("{"):
                                try:
                                    chunk_payload = json.loads(stripped_sse)
                                except Exception:
                                    plain_json_lines.append(sse_line)
                                    continue
                            else:
                                if not saw_sse_prefix:
                                    plain_json_lines.append(sse_line)
                                continue
                            delta_piece = jarvis_services.VisionBrainService.extract_universal_sse_delta_text(
                                chunk_payload,
                                attempt_wire,
                            )
                            if delta_piece:
                                vision_extractor.feed_delta(delta_piece)
                        if (not saw_sse_prefix) and plain_json_lines:
                            try:
                                full_json_payload = json.loads("\n".join(plain_json_lines))
                                full_text_piece = jarvis_services.VisionBrainService.extract_universal_sse_delta_text(
                                    full_json_payload,
                                    attempt_wire,
                                )
                                if full_text_piece:
                                    vision_extractor.feed_delta(full_text_piece)
                            except Exception:
                                pass
                    vision_extractor.flush_remaining()
                    vision_answer = vision_extractor.build_final_response().spoken_text
                    active_provider_label = f"{attempt_provider}:{candidate_model}"
                    attempt_succeeded = True
                    break
                except Exception as request_exc:
                    last_vision_error = f"{attempt_provider}:{candidate_model} error: {request_exc}"
                    break
            if attempt_succeeded:
                break

        if not vision_answer and last_vision_error:
            raise RuntimeError(last_vision_error)

        vision_latency = time.perf_counter() - vision_start
        vision_ok = ("73" in vision_answer) and ("red" in vision_answer.lower())

        tts_service = jarvis_services.StreamingTTSService(settings_store)
        tts_engine, tts_seconds = tts_service.speak_blocking_for_test("Jarvis online.")
        tts_ok = tts_engine == "assemblyai"

        controller = jarvis_app.JarvisApplicationController(settings_store)
        controller.start(auto_demo=False)
        qt_application.processEvents()
        controller.trigger_demo_showcase()
        qt_application.processEvents()
        controller.quit_application()
        qt_application.processEvents()
        controller.deleteLater()
        qt_application.processEvents()

        suite6_ok = vision_ok and tts_ok
        record_result(
            "Suite 6 (Vision SSE + AssemblyAI TTS + Controller)",
            suite6_ok,
            (
                f"provider={active_provider_label}, vision={vision_answer!r} ({vision_latency:.2f}s), "
                f"tts_engine={tts_engine} ({tts_seconds:.2f}s)"
            ),
        )
    except Exception as exc:
        record_result("Suite 6 (Vision SSE + AssemblyAI TTS + Controller)", False, f"Exception: {exc}")

    if failures:
        print(f"SELFTEST FAILED ({len(failures)} failure(s))")
        qt_application.processEvents()
        return 1

    print("ALL 6 SELFTEST SUITES PASSED")
    qt_application.processEvents()
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jarvis for Windows (PySide6 60 FPS Edition)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch with the animated Notch Peek + Overlay Ink demo showcase visible.",
    )
    parser.add_argument(
        "--demo-seconds",
        type=float,
        default=0.0,
        help="Optional auto-exit delay in seconds (useful for automated visual checks).",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="",
        help="Render all Notch, Overlay, and MainWindow states as PNG files into the given directory.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run strict 6-suite verification (assets, 60 FPS benchmarks, parser, live STT/Vision/TTS).",
    )
    parser.add_argument(
        "--talk",
        action="store_true",
        help="Summon push-to-talk listening immediately on launch.",
    )
    parser.add_argument(
        "--ask",
        type=str,
        default="",
        help="Submit a question prompt immediately on launch.",
    )
    parser.add_argument(
        "--summon",
        action="store_true",
        help="Show and center the Jarvis cursor overlay immediately on launch.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Display the interactive Today & Later reminders list overlay card on launch.",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default="",
        help="Open a specific MainWindow panel/tab on launch (e.g. chat, settings:agents, settings:routines).",
    )
    return parser


def main() -> int:
    _configure_clean_startup_logging()
    argument_parser = _build_argument_parser()
    parsed_args = argument_parser.parse_args()

    if parsed_args.snapshot:
        return run_snapshot_export(parsed_args.snapshot)

    if parsed_args.selftest:
        qt_keepalive = _create_qt_application()
        selftest_code = run_strict_selftest()
        qt_keepalive.processEvents()
        return selftest_code

    qt_application = _create_qt_application()
    settings_store = jarvis_config.SettingsStore()
    controller = jarvis_app.JarvisApplicationController(settings_store)

    def handle_sigint(_signal_number, _frame) -> None:
        controller.quit_application()

    signal.signal(signal.SIGINT, handle_sigint)
    controller.start(auto_demo=parsed_args.demo)

    def _run_launch_cli_actions() -> None:
        if parsed_args.panel:
            controller.open_main_window(parsed_args.panel)
        if parsed_args.list:
            controller._on_main_window_agent_command("show_list", "")
        if parsed_args.summon:
            controller.overlay_window.show()
            center_pos = controller.overlay_window.rect().center()
            controller.overlay_window.fly_cursor_to(float(center_pos.x()), float(center_pos.y()), label="Ready", hold_seconds=4.0)
        if (parsed_args.ask or "").strip():
            controller._on_main_window_prompt_submitted(parsed_args.ask.strip())
        if parsed_args.talk:
            controller._on_ptt_pressed()

    if any((parsed_args.panel, parsed_args.list, parsed_args.summon, (parsed_args.ask or "").strip(), parsed_args.talk)):
        QTimer.singleShot(200, _run_launch_cli_actions)

    if parsed_args.demo_seconds > 0:
        exit_delay_ms = max(250, int(parsed_args.demo_seconds * 1000))
        QTimer.singleShot(exit_delay_ms, controller.quit_application)

    return int(qt_application.exec())


if __name__ == "__main__":
    sys.exit(main())
