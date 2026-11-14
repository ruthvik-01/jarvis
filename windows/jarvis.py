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
import signal
import sys
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
    missing_secrets = [
        secret_name
        for secret_name in jarvis_config.ALL_SECRET_NAMES
        if not settings_store.get_secret(secret_name)
    ]
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
                f"univ_agent_ok={universal_agent_ok}, "
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
    # Suite 6: Live River AI Vision SSE + Live AssemblyAI PCM TTS + Controller Smoke
    # -----------------------------------------------------------------------
    try:
        river_key = settings_store.get_secret(jarvis_config.SECRET_RIVER_API_KEY)
        model_id = str(settings_store.get("chat_model", "deepseek-ai/DeepSeek-V4.1-Flash"))
        test_jpeg_b64 = _build_synthetic_vision_test_jpeg_b64()
        vision_extractor = jarvis_response_parser.StreamingSentenceExtractor()

        vision_start = time.perf_counter()
        with requests.post(
            jarvis_services.VisionBrainService.RIVER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {river_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Reply in one short sentence: what colored shape and number do you see?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{test_jpeg_b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 120,
                "stream": True,
                "reasoning_effort": "none",
            },
            stream=True,
            timeout=30,
        ) as vision_response:
            if not vision_response.ok:
                raise RuntimeError(
                    f"River AI HTTP {vision_response.status_code}: {vision_response.text[:160]}"
                )
            vision_response.encoding = "utf-8"
            for sse_line in vision_response.iter_lines(decode_unicode=True):
                if not sse_line or not sse_line.startswith("data: "):
                    continue
                payload_text = sse_line[6:].strip()
                if payload_text == "[DONE]":
                    break
                try:
                    chunk_payload = json.loads(payload_text)
                    delta_piece = (
                        chunk_payload.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                except Exception:
                    continue
                if delta_piece:
                    vision_extractor.feed_delta(delta_piece)
        vision_extractor.flush_remaining()
        vision_latency = time.perf_counter() - vision_start
        vision_answer = vision_extractor.build_final_response().spoken_text
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

        suite6_ok = vision_ok and tts_ok
        record_result(
            "Suite 6 (Vision SSE + AssemblyAI TTS + Controller)",
            suite6_ok,
            (
                f"vision={vision_answer!r} ({vision_latency:.2f}s), "
                f"tts_engine={tts_engine} ({tts_seconds:.2f}s)"
            ),
        )
    except Exception as exc:
        record_result("Suite 6 (Vision SSE + AssemblyAI TTS + Controller)", False, f"Exception: {exc}")

    if failures:
        print(f"SELFTEST FAILED ({len(failures)} failure(s))")
        return 1

    print("ALL 6 SELFTEST SUITES PASSED")
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
    return parser


def main() -> int:
    _configure_clean_startup_logging()
    argument_parser = _build_argument_parser()
    parsed_args = argument_parser.parse_args()

    if parsed_args.snapshot:
        return run_snapshot_export(parsed_args.snapshot)

    if parsed_args.selftest:
        return run_strict_selftest()

    qt_application = _create_qt_application()
    settings_store = jarvis_config.SettingsStore()
    controller = jarvis_app.JarvisApplicationController(settings_store)

    def handle_sigint(_signal_number, _frame) -> None:
        controller.quit_application()

    signal.signal(signal.SIGINT, handle_sigint)
    controller.start(auto_demo=parsed_args.demo)

    if parsed_args.demo_seconds > 0:
        exit_delay_ms = max(250, int(parsed_args.demo_seconds * 1000))
        QTimer.singleShot(exit_delay_ms, controller.quit_application)

    return int(qt_application.exec())


if __name__ == "__main__":
    sys.exit(main())
