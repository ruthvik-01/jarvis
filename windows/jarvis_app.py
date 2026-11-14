"""
Main application controller wiring Jarvis's PySide6 windows and live services.

Connects:
- Top hardware Notch (`jarvis_notch_window.NotchWindow`)
- Companion cursor + 2D screen ink overlay (`jarvis_overlay_windows.OverlayCanvasWindow`)
- Resizable Chat + Settings window (`jarvis_main_window.JarvisMainWindow`)
- Mic STT, River AI Vision Brain, ElevenLabs PCM TTS, Global Hotkeys, and System Tray.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

import jarvis_config
import jarvis_main_window
import jarvis_notch_window
import jarvis_overlay_windows
import jarvis_response_parser
import jarvis_services
import jarvis_theme


class JarvisApplicationController(QObject):
    """Coordinates windows, hotkeys, voice capture, vision streaming, TTS, and screen ink."""

    aboutToQuit = Signal()

    def __init__(self, settings_store: Optional[jarvis_config.SettingsStore] = None) -> None:
        super().__init__()
        jarvis_theme.register_bundled_fonts()

        self.settings_store = settings_store or jarvis_config.SettingsStore()
        self.sound_player = jarvis_services.SoundEffectsPlayer(self.settings_store)
        self.mic_service = jarvis_services.MicrophoneStreamService(self.settings_store)
        self.tts_service = jarvis_services.StreamingTTSService(self.settings_store)
        self.brain_service = jarvis_services.VisionBrainService(self.settings_store)
        self.hotkey_service = jarvis_services.GlobalHotkeyService()

        self.notch_window = jarvis_notch_window.NotchWindow(self.settings_store)
        self.overlay_window = jarvis_overlay_windows.OverlayCanvasWindow(self.settings_store)
        self.main_window = jarvis_main_window.JarvisMainWindow(self.settings_store)

        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._active_voice_mode = "idle"  # "idle" | "ptt" | "dictate" | "mic_test"
        self._last_highlight_points: List[Tuple[float, float]] = []
        self._speak_current_turn = False
        self._force_captions_for_turn = False
        self._last_turn_from_typed_notch = False
        self._active_tts_generation = 0
        self._sentences_enqueued_in_turn = 0
        self._scroll_explain_active = False
        self._scroll_explain_user_prompt = ""
        self._scroll_step_index = 0
        self._max_scroll_steps = 4
        self._pending_scroll_after_speech: Optional[Tuple[str, int]] = None
        self._scroll_generation_id = 0
        self._pre_scroll_image: Optional[object] = None
        self._pending_direct_agent_commands: List[jarvis_response_parser.DrawingCommand] = []
        self._executed_agent_command_in_turn = False
        self._quitting = False

        self._populate_microphone_devices()
        self._connect_signals()
        self._setup_system_tray()

    def _populate_microphone_devices(self) -> None:
        devices = jarvis_services.MicrophoneStreamService.list_input_devices()
        self.main_window.set_microphone_devices(devices)

    def _connect_signals(self) -> None:
        self.settings_store.settingChanged.connect(self._on_setting_changed)

        # Hotkeys
        self.hotkey_service.pushToTalkPressed.connect(self._on_ptt_pressed)
        self.hotkey_service.pushToTalkReleased.connect(self._on_ptt_released)
        self.hotkey_service.doubleCtrlTapped.connect(self.notch_window.toggle_text_input)
        self.hotkey_service.dictationPressed.connect(self._on_dictate_pressed)
        self.hotkey_service.dictationReleased.connect(self._on_dictate_released)
        self.hotkey_service.escapePressed.connect(self._on_escape_pressed)
        self.hotkey_service.leftArrowPressed.connect(self._on_left_arrow_pressed)
        self.hotkey_service.rightArrowPressed.connect(self._on_right_arrow_pressed)

        # Microphone STT
        self.mic_service.audioLevelChanged.connect(self._on_mic_audio_level)
        self.mic_service.partialTranscriptReceived.connect(self._on_partial_transcript)
        self.mic_service.finalTranscriptReady.connect(self._on_final_transcript_ready)
        self.mic_service.microphoneErrorOccurred.connect(self._on_microphone_error)

        # Vision Brain
        self.brain_service.partialTextUpdated.connect(self._on_brain_partial_text)
        self.brain_service.sentenceReady.connect(self._on_brain_sentence_ready)
        self.brain_service.drawingCommandReady.connect(self._on_brain_drawing_command)
        self.brain_service.turnCompleted.connect(self._on_brain_turn_completed)
        self.brain_service.turnFailed.connect(self._on_brain_turn_failed)

        # Streaming TTS
        self.tts_service.audioLevelChanged.connect(self._on_tts_audio_level)
        self.tts_service.speechWordProgress.connect(self._on_tts_word_progress)
        self.tts_service.speechFinished.connect(self._on_tts_finished)

        # Notch Window
        self.notch_window.promptSubmitted.connect(self._on_notch_prompt_submitted)
        self.notch_window.stopRequested.connect(self._on_stop_requested)
        self.notch_window.openMainWindowRequested.connect(self.open_main_window)
        self.notch_window.continueInCaptionsRequested.connect(self._on_continue_in_captions)
        self.notch_window.demoDrawingRequested.connect(lambda _preset: self.trigger_demo_showcase())
        self.notch_window.openArtifactRequested.connect(self._on_open_artifact)
        self.notch_window.scrollExplainRequested.connect(self.start_scroll_and_explain_session)
        self.notch_window.lassoModeRequested.connect(self.start_interactive_lasso_session)

        # Overlay Window
        self.overlay_window.clearRequested.connect(self._on_overlay_cleared)
        self.overlay_window.interactiveLassoCompleted.connect(self._on_interactive_lasso_completed)
        self.overlay_window.strokeAnimationStarted.connect(self._on_stroke_animation_started)
        self.overlay_window.strokeAnimationFinished.connect(self.sound_player.stop_drawing_sound)

        # Main Window
        self.main_window.promptSubmitted.connect(self._on_main_window_prompt_submitted)
        self.main_window.stopRequested.connect(self._on_stop_requested)
        self.main_window.demoDrawingRequested.connect(self.trigger_demo_showcase)
        self.main_window.scrollExplainRequested.connect(self.start_scroll_and_explain_session)
        self.main_window.lassoModeRequested.connect(self.start_interactive_lasso_session)
        self.main_window.agentCommandRequested.connect(self._on_main_window_agent_command)
        self.main_window.clearOverlayRequested.connect(self._on_escape_pressed)
        self.main_window.quitRequested.connect(self.quit_application)
        self.main_window.voicePreviewRequested.connect(self._on_voice_preview_requested)
        self.main_window.microphoneTestRequested.connect(self._on_microphone_test_requested)
        self.main_window.shortcutTestRequested.connect(self._on_shortcut_test_requested)

    def _setup_system_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        icon_path = jarvis_theme.asset_path("jarvis.ico")
        tray_icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self._tray_icon = QSystemTrayIcon(tray_icon, self)
        self._tray_icon.setToolTip(f"Jarvis {jarvis_theme.APP_DISPLAY_VERSION} (Ctrl+Alt Talk · Ctrl×2 Type)")

        tray_menu = QMenu()

        open_home_action = QAction("Open Notch Home", tray_menu)
        open_home_action.triggered.connect(self.notch_window.toggle_home_peek)
        tray_menu.addAction(open_home_action)

        ask_notch_action = QAction("Ask Jarvis (Ctrl × 2)", tray_menu)
        ask_notch_action.triggered.connect(self.notch_window.toggle_text_input)
        tray_menu.addAction(ask_notch_action)

        scroll_explain_action = QAction("Scroll && Explain Page", tray_menu)
        scroll_explain_action.triggered.connect(self.start_scroll_and_explain_session)
        tray_menu.addAction(scroll_explain_action)

        lasso_ask_action = QAction("Lasso && Ask Screen", tray_menu)
        lasso_ask_action.triggered.connect(self.start_interactive_lasso_session)
        tray_menu.addAction(lasso_ask_action)

        open_chat_action = QAction("Open Jarvis Window (Chat)", tray_menu)
        open_chat_action.triggered.connect(lambda: self.open_main_window("chat"))
        tray_menu.addAction(open_chat_action)

        open_settings_action = QAction("Open Settings", tray_menu)
        open_settings_action.triggered.connect(lambda: self.open_main_window("settings:account"))
        tray_menu.addAction(open_settings_action)

        tray_menu.addSeparator()

        demo_action = QAction("Run Screen Sketch Demo", tray_menu)
        demo_action.triggered.connect(self.trigger_demo_showcase)
        tray_menu.addAction(demo_action)

        clear_action = QAction("Clear Screen Drawings (Esc)", tray_menu)
        clear_action.triggered.connect(self._on_escape_pressed)
        tray_menu.addAction(clear_action)

        tray_menu.addSeparator()

        quit_action = QAction("Quit Jarvis", tray_menu)
        quit_action.triggered.connect(self.quit_application)
        tray_menu.addAction(quit_action)

        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.notch_window.toggle_home_peek()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self, auto_demo: bool = False) -> None:
        self.overlay_window.show()
        self.notch_window.show()
        self.hotkey_service.start()
        self.sound_player.play_earcon("reveal-boot", volume=0.45)

        if auto_demo:
            QTimer.singleShot(120, self._run_auto_demo_sequence)

    def _run_auto_demo_sequence(self) -> None:
        self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.HOME_PEEK)
        self.trigger_demo_showcase()

    def quit_application(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self.aboutToQuit.emit()

        self.hotkey_service.stop()
        self.brain_service.stop_current_turn()
        self.tts_service.stop_speaking()

        if self._tray_icon is not None:
            self._tray_icon.hide()

        self.main_window.allow_real_close()
        self.main_window.close()
        self.notch_window.close()
        self.overlay_window.clear_button_window.close()
        self.overlay_window.close()

        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.quit()

    # -----------------------------------------------------------------------
    # Window & Demo Actions
    # -----------------------------------------------------------------------

    def open_main_window(self, target_tab: str = "chat") -> None:
        normalized = (target_tab or "chat").strip().lower()
        if normalized in ("upgrade", "settings", "studio"):
            mapped_tab = "settings:account"
        else:
            mapped_tab = normalized

        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.HOME_PEEK:
            self.notch_window.collapse_to_idle()

        self.main_window.open_tab(mapped_tab)
        if hasattr(self.main_window, "position_under_notch"):
            self.main_window.position_under_notch()
        if self.main_window.isMinimized():
            self.main_window.showNormal()
        else:
            self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def _cancel_scroll_explain_sequence(self) -> None:
        """Stop any multi-step scroll-and-explain walkthrough and invalidate pending timers."""
        self._scroll_explain_active = False
        self._scroll_step_index = 0
        self._pending_scroll_after_speech = None
        self._pre_scroll_image = None
        self._scroll_generation_id += 1

    def start_interactive_lasso_session(self) -> None:
        """Activate click-and-drag neon lasso targeting over the current desktop screen."""
        self._cancel_scroll_explain_sequence()
        self.brain_service.stop_current_turn()
        self.tts_service.stop_speaking()
        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.HOME_PEEK:
            self.notch_window.collapse_to_idle()
        if self.main_window.isVisible():
            self.main_window.hide()
        self.sound_player.play_earcon("jarvis-text-open", volume=0.45)
        self.overlay_window.start_interactive_lasso_mode()
        self.overlay_window.set_bubble_message(
            "Drag a lasso around anything on screen to explain it (Esc to cancel).",
            word_by_word=False,
            hold_seconds=3.8,
        )

    def _on_interactive_lasso_completed(self, lasso_points_object: object) -> None:
        raw_points = lasso_points_object if isinstance(lasso_points_object, list) else []
        validated_points: List[Tuple[float, float]] = [
            (float(point_item[0]), float(point_item[1]))
            for point_item in raw_points
            if isinstance(point_item, (tuple, list)) and len(point_item) >= 2
        ]
        if len(validated_points) < 4:
            return
        self._last_highlight_points = validated_points
        self.sound_player.play_earcon("jarvis-question", volume=0.45)
        self._start_turn(
            prompt_text="Explain clearly what is inside the lassoed region on my screen and highlight its key parts visually.",
            speak_reply=True,
            from_typed_notch=False,
            highlight_points=validated_points,
        )

    def start_scroll_and_explain_session(self, custom_prompt: str = "") -> None:
        """Start an autonomous multi-step Scroll & Explain walkthrough for long pages like ChatGPT."""
        self._cancel_scroll_explain_sequence()
        self._scroll_explain_active = True
        self._scroll_step_index = 1
        self._max_scroll_steps = 4
        self._scroll_generation_id += 1
        cleaned_prompt = (custom_prompt or "").strip()
        self._scroll_explain_user_prompt = cleaned_prompt or (
            "Walk me step by step through this long explanation (for example ChatGPT), "
            "drawing accurate visual underlines, boxes, points, and a concise summary card "
            "before scrolling down to continue."
        )
        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.HOME_PEEK:
            self.notch_window.collapse_to_idle()
        if self.main_window.isVisible():
            self.main_window.hide()
        self.sound_player.play_earcon("jarvis-question", volume=0.45)
        self._start_turn(
            prompt_text=self._scroll_explain_user_prompt,
            speak_reply=True,
            from_typed_notch=False,
            preserve_scroll_session=True,
        )

    def trigger_demo_showcase(self) -> None:
        self.overlay_window.load_demo_showcase()
        self.sound_player.play_drawing_sound("paper-place", volume=0.45)
        self.overlay_window.set_bubble_message(
            "Sketching directly over your screen at 60 FPS! Press Esc anytime to clear.",
            word_by_word=True,
            hold_seconds=4.5,
        )

    def _on_open_artifact(self, artifact_path: str) -> None:
        if artifact_path and os.path.exists(artifact_path):
            try:
                os.startfile(artifact_path)
                return
            except Exception:
                pass
        self.trigger_demo_showcase()

    def _on_overlay_cleared(self) -> None:
        self.overlay_window.clear_caption()
        self.sound_player.play_drawing_sound("page-clear", volume=0.35)

    def _on_setting_changed(self, key_name: str, new_value: object) -> None:
        self.overlay_window.on_setting_changed(key_name, new_value)

    # -----------------------------------------------------------------------
    # Hotkey Handlers
    # -----------------------------------------------------------------------

    def _on_ptt_pressed(self) -> None:
        self._cancel_scroll_explain_sequence()
        self.overlay_window.cancel_interactive_lasso_mode()
        self.tts_service.stop_speaking()
        self.brain_service.stop_current_turn()
        self.sound_player.play_earcon("jarvis-question", volume=0.5)

        self._active_voice_mode = "ptt"
        self._last_highlight_points = []
        self.notch_window.set_live_transcript("")
        self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.LISTENING, status_text="Listening")
        self.overlay_window.set_cursor_visual_state(listening=True, thinking=False, speaking=False)
        if self.settings_store.get("screen_highlight_enabled", True):
            self.overlay_window.begin_highlight_stroke()
        self.mic_service.start_streaming()

    def _on_ptt_released(self) -> None:
        if self._active_voice_mode != "ptt":
            return
        self.overlay_window.set_cursor_visual_state(listening=False, thinking=False, speaking=False)
        self._last_highlight_points = self.overlay_window.end_highlight_stroke()
        self.mic_service.stop_and_finalize()

    def _on_dictate_pressed(self) -> None:
        self._cancel_scroll_explain_sequence()
        self.overlay_window.cancel_interactive_lasso_mode()
        self.tts_service.stop_speaking()
        self.sound_player.play_earcon("jarvis-text-open", volume=0.45)
        self._active_voice_mode = "dictate"
        self.notch_window.set_live_transcript("")
        self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.DICTATING, status_text="Dictating")
        self.mic_service.start_streaming()

    def _on_dictate_released(self) -> None:
        if self._active_voice_mode != "dictate":
            return
        self.mic_service.stop_and_finalize()

    def _on_left_arrow_pressed(self) -> None:
        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.HOME_PEEK:
            self.notch_window.cycle_arc_card(-1)

    def _on_right_arrow_pressed(self) -> None:
        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.HOME_PEEK:
            self.notch_window.cycle_arc_card(1)

    def _on_escape_pressed(self) -> None:
        self._active_voice_mode = "idle"
        self._pending_direct_agent_commands = []
        self._cancel_scroll_explain_sequence()
        self.overlay_window.cancel_interactive_lasso_mode()
        self.tts_service.stop_speaking()
        self.sound_player.stop_drawing_sound()
        self.brain_service.stop_current_turn()
        self.overlay_window.set_cursor_visual_state(listening=False, thinking=False, speaking=False)
        self.overlay_window.set_bubble_message("")
        self.overlay_window.clear_all_drawings()
        self.overlay_window.clear_caption()
        if self.notch_window.current_mode() != jarvis_notch_window.NotchVisualMode.IDLE:
            self.notch_window.collapse_to_idle()

    # -----------------------------------------------------------------------
    # Microphone Callbacks
    # -----------------------------------------------------------------------

    def _on_mic_audio_level(self, level: float) -> None:
        self.notch_window.set_audio_level(level)
        self.overlay_window.set_audio_level(level)
        self.main_window.set_microphone_level(level)

    def _on_partial_transcript(self, text: str) -> None:
        self.notch_window.set_live_transcript(text)
        if self._active_voice_mode == "ptt":
            self.overlay_window.set_bubble_message(text, word_by_word=False, hold_seconds=2.5)

    def _on_final_transcript_ready(self, transcript: str) -> None:
        mode = self._active_voice_mode
        self._active_voice_mode = "idle"
        cleaned = (transcript or "").strip()

        if mode == "dictate":
            if cleaned:
                jarvis_services.SmartDictationInjector.type_text_into_active_window(cleaned)
                self.sound_player.play_earcon("jarvis-text-send", volume=0.45)
            self.notch_window.collapse_to_idle()
            return

        if mode == "mic_test":
            self.overlay_window.set_bubble_message(
                f'Heard: "{cleaned}"' if cleaned else "Microphone active!",
                word_by_word=False,
                hold_seconds=2.8,
            )
            self.notch_window.collapse_to_idle()
            return

        if mode == "ptt":
            if not cleaned and len(self._last_highlight_points) >= 4:
                cleaned = "What is this highlighted area on my screen?"
            if not cleaned:
                self.overlay_window.set_bubble_message("")
                self.overlay_window.set_cursor_visual_state(False, False, False)
                self.notch_window.collapse_to_idle()
                return
            self._start_turn(
                prompt_text=cleaned,
                speak_reply=True,
                from_typed_notch=False,
                highlight_points=self._last_highlight_points,
            )

    def _on_microphone_error(self, error_message: str) -> None:
        self._active_voice_mode = "idle"
        self.overlay_window.set_cursor_visual_state(False, False, False)
        self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.WARNING, status_text=error_message)

    # -----------------------------------------------------------------------
    # Vision Turn & Streaming TTS Pipeline
    # -----------------------------------------------------------------------

    def _on_notch_prompt_submitted(self, prompt_text: str) -> None:
        speak = bool(self.settings_store.get("typed_turns_respond_with_voice", False))
        self._start_turn(prompt_text=prompt_text, speak_reply=speak, from_typed_notch=True)

    def _on_main_window_prompt_submitted(self, prompt_text: str) -> None:
        speak = bool(self.settings_store.get("typed_turns_respond_with_voice", False))
        self._start_turn(prompt_text=prompt_text, speak_reply=speak, from_typed_notch=False)

    def _on_stop_requested(self) -> None:
        self._pending_direct_agent_commands = []
        self._cancel_scroll_explain_sequence()
        self.overlay_window.cancel_interactive_lasso_mode()
        self.brain_service.stop_current_turn()
        self.tts_service.stop_speaking()
        self.sound_player.stop_drawing_sound()
        self.overlay_window.set_cursor_visual_state(False, False, False)
        self.main_window.finish_streaming_assistant_message("")
        if self.notch_window.current_mode() in (
            jarvis_notch_window.NotchVisualMode.THINKING,
            jarvis_notch_window.NotchVisualMode.SPEAKING,
        ):
            self.notch_window.collapse_to_idle()

    def _on_continue_in_captions(self) -> None:
        self._force_captions_for_turn = True

    def _start_turn(
        self,
        prompt_text: str,
        speak_reply: bool,
        from_typed_notch: bool,
        highlight_points: Optional[List[Tuple[float, float]]] = None,
        preserve_scroll_session: bool = False,
    ) -> None:
        self.brain_service.stop_current_turn()
        self.tts_service.stop_speaking()
        self.sound_player.stop_drawing_sound()

        self._executed_agent_command_in_turn = False
        agent_actions_enabled = bool(self.settings_store.get("agent_actions_enabled", True))
        self._pending_direct_agent_commands = (
            jarvis_services.DesktopAgentService.detect_direct_user_agent_commands(prompt_text)
            if agent_actions_enabled and not preserve_scroll_session
            else []
        )

        if not preserve_scroll_session:
            if jarvis_response_parser.is_scroll_and_explain_request(prompt_text):
                self._cancel_scroll_explain_sequence()
                self._scroll_explain_active = True
                self._scroll_step_index = 1
                self._max_scroll_steps = 4
                self._scroll_explain_user_prompt = prompt_text
                self._scroll_generation_id += 1
            else:
                self._cancel_scroll_explain_sequence()

        self._last_turn_from_typed_notch = bool(from_typed_notch)
        self._pending_scroll_after_speech = None

        if preserve_scroll_session and self._scroll_step_index > 1:
            self.overlay_window.clear_viewport_markings_keep_paper()
            displayed_user_message = (
                f"[Scroll step {self._scroll_step_index}/{self._max_scroll_steps}] Explaining next section..."
            )
        else:
            self.overlay_window.clear_all_drawings()
            displayed_user_message = prompt_text

        self._speak_current_turn = bool(speak_reply)
        self._force_captions_for_turn = False
        self._sentences_enqueued_in_turn = 0
        self._active_tts_generation = self.tts_service.begin_turn()

        self.main_window.append_chat_message("user", displayed_user_message)
        self.main_window.update_streaming_assistant_message("Thinking...", is_generating=True)

        if from_typed_notch:
            self.notch_window.set_streaming_response("Thinking...", is_generating=True)
        else:
            self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.THINKING, status_text="Thinking")
            self.overlay_window.set_cursor_visual_state(listening=False, thinking=True, speaking=False)

        if len(self._pending_direct_agent_commands) >= 2:
            action_statuses: List[str] = []
            any_action_ok = False
            for direct_command in self._pending_direct_agent_commands:
                action_ok, status_msg = jarvis_services.DesktopAgentService.execute_agent_command(direct_command)
                any_action_ok = any_action_ok or action_ok
                if status_msg:
                    action_statuses.append(status_msg)
            self._executed_agent_command_in_turn = True
            self._pending_direct_agent_commands = []
            combined_status = ", ".join(action_statuses) if action_statuses else "Done."
            self.sound_player.play_earcon("jarvis-text-send" if any_action_ok else "jarvis-error", volume=0.45)
            self.overlay_window.set_bubble_message(combined_status, word_by_word=False, hold_seconds=2.8)
            direct_summary = f"{combined_status}."
            self._on_brain_turn_completed(
                jarvis_response_parser.ParsedTurnResponse(
                    spoken_text=direct_summary,
                    sentences=[direct_summary],
                    commands=[],
                )
            )
            return

        self.brain_service.start_vision_turn(
            user_prompt=prompt_text,
            include_screenshot=True,
            highlight_points=highlight_points,
            highlight_color_hex=str(
                self.settings_store.get("cursor_color_hex", jarvis_theme.JARVIS_BLUE_HEX)
            ),
            logical_screen_size=(max(640, self.overlay_window.width()), max(480, self.overlay_window.height())),
            scroll_step_index=self._scroll_step_index if self._scroll_explain_active else 0,
            max_scroll_steps=self._max_scroll_steps if self._scroll_explain_active else 0,
        )

    def _on_brain_partial_text(self, partial_text: str) -> None:
        self.main_window.update_streaming_assistant_message(partial_text, is_generating=True)
        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.TEXT_RESPONSE:
            self.notch_window.set_streaming_response(partial_text, is_generating=True)
        if self._force_captions_for_turn or bool(self.settings_store.get("captions_enabled", False)):
            self.overlay_window.set_caption_text(partial_text, force_visible=self._force_captions_for_turn)

    def _on_brain_sentence_ready(self, sentence_text: str) -> None:
        self.overlay_window.set_cursor_visual_state(
            listening=False,
            thinking=False,
            speaking=self._speak_current_turn,
        )
        if self._speak_current_turn:
            self._sentences_enqueued_in_turn += 1
            if self.notch_window.current_mode() not in (
                jarvis_notch_window.NotchVisualMode.TEXT_RESPONSE,
                jarvis_notch_window.NotchVisualMode.HOME_PEEK,
                jarvis_notch_window.NotchVisualMode.TEXT_INPUT,
            ):
                self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.SPEAKING, status_text="Speaking")
            self.tts_service.enqueue_sentence(sentence_text, turn_generation=self._active_tts_generation)

    def _on_brain_drawing_command(self, command: jarvis_response_parser.DrawingCommand) -> None:
        if command.kind == "scroll":
            return
        if command.kind in ("open_browser", "open_url", "search_web", "open_app", "press_keys", "type_text"):
            if bool(self.settings_store.get("agent_actions_enabled", True)):
                action_ok, status_msg = jarvis_services.DesktopAgentService.execute_agent_command(command)
                self._executed_agent_command_in_turn = True
                self._pending_direct_agent_commands = []
                self.sound_player.play_earcon("jarvis-text-send" if action_ok else "jarvis-error", volume=0.45)
                self.overlay_window.set_bubble_message(status_msg, word_by_word=False, hold_seconds=2.8)
            return
        snapped_command = jarvis_services.ScreenCaptureService.snap_drawing_command_to_content(command)
        self.overlay_window.add_drawing_command(snapped_command)

    def _on_stroke_animation_started(self, stroke_kind: str, duration_ms: int) -> None:
        if stroke_kind == "paper":
            self.sound_player.play_drawing_sound("paper-place", volume=0.38)
        else:
            self.sound_player.play_synced_drawing_stroke_sound(duration_ms=duration_ms, volume=0.24)

    def _on_main_window_agent_command(self, action_kind: str, argument_text: str) -> None:
        agent_command = jarvis_response_parser.DrawingCommand(
            kind=action_kind,
            body=argument_text,
            label=argument_text,
        )
        action_ok, status_msg = jarvis_services.DesktopAgentService.execute_agent_command(agent_command)
        self.sound_player.play_earcon("jarvis-text-send" if action_ok else "jarvis-error", volume=0.45)
        self.overlay_window.set_bubble_message(status_msg, word_by_word=False, hold_seconds=2.8)

    def _on_brain_turn_completed(self, parsed_response: jarvis_response_parser.ParsedTurnResponse) -> None:
        if (
            self._pending_direct_agent_commands
            and not self._executed_agent_command_in_turn
            and bool(self.settings_store.get("agent_actions_enabled", True))
        ):
            last_status_msg = ""
            any_action_ok = False
            for direct_command in self._pending_direct_agent_commands:
                action_ok, status_msg = jarvis_services.DesktopAgentService.execute_agent_command(direct_command)
                any_action_ok = any_action_ok or action_ok
                if status_msg:
                    last_status_msg = status_msg
            self._executed_agent_command_in_turn = True
            self.sound_player.play_earcon("jarvis-text-send" if any_action_ok else "jarvis-error", volume=0.45)
            if last_status_msg:
                self.overlay_window.set_bubble_message(last_status_msg, word_by_word=False, hold_seconds=2.8)
        self._pending_direct_agent_commands = []

        final_text = parsed_response.spoken_text or "Done."
        self.main_window.finish_streaming_assistant_message(final_text)

        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.TEXT_RESPONSE:
            self.notch_window.set_streaming_response(final_text, is_generating=False)

        if parsed_response.primary_point_target is not None:
            target_x, target_y, target_label = parsed_response.primary_point_target
            self.overlay_window.fly_cursor_to(target_x, target_y, label=target_label, hold_seconds=4.5)

        scroll_task: Optional[Tuple[str, int]] = None
        if self._scroll_explain_active and self._scroll_step_index >= self._max_scroll_steps:
            self._scroll_explain_active = False
            self._scroll_step_index = 0
        elif parsed_response.scroll_direction in ("down", "up"):
            scroll_task = (parsed_response.scroll_direction, int(parsed_response.scroll_amount or 6))
        elif self._scroll_explain_active and self._scroll_step_index < self._max_scroll_steps:
            scroll_task = ("down", 6)

        has_pending_tts_speech = False
        if self._speak_current_turn:
            if self._sentences_enqueued_in_turn == 0 and parsed_response.spoken_text:
                self._sentences_enqueued_in_turn += 1
                if self.notch_window.current_mode() not in (
                    jarvis_notch_window.NotchVisualMode.TEXT_RESPONSE,
                    jarvis_notch_window.NotchVisualMode.HOME_PEEK,
                    jarvis_notch_window.NotchVisualMode.TEXT_INPUT,
                ):
                    self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.SPEAKING, status_text="Speaking")
                self.tts_service.enqueue_sentence(
                    parsed_response.spoken_text, turn_generation=self._active_tts_generation
                )

            if self._sentences_enqueued_in_turn > 0:
                has_pending_tts_speech = True
                self.overlay_window.set_cursor_visual_state(listening=False, thinking=False, speaking=True)
                self.tts_service.mark_turn_complete(turn_generation=self._active_tts_generation)
            else:
                self.overlay_window.set_cursor_visual_state(False, False, False)
                if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.THINKING:
                    self.notch_window.collapse_to_idle()
        else:
            self.overlay_window.set_cursor_visual_state(False, False, False)
            if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.THINKING:
                self.notch_window.collapse_to_idle()
            if self._force_captions_for_turn and parsed_response.spoken_text:
                self.overlay_window.set_caption_text(parsed_response.spoken_text, force_visible=True)
                QTimer.singleShot(5500, self.overlay_window.clear_caption)

        if scroll_task is not None:
            if has_pending_tts_speech:
                self._pending_scroll_after_speech = scroll_task
            else:
                scheduled_generation_id = self._scroll_generation_id
                QTimer.singleShot(
                    450,
                    lambda task_tuple=scroll_task, gen_id=scheduled_generation_id: self._execute_scroll_and_schedule_next(
                        task_tuple[0], task_tuple[1], gen_id
                    ),
                )

    def _on_brain_turn_failed(self, error_text: str) -> None:
        self._cancel_scroll_explain_sequence()
        self.overlay_window.set_cursor_visual_state(False, False, False)
        self.main_window.finish_streaming_assistant_message(error_text)
        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.TEXT_RESPONSE:
            self.notch_window.set_streaming_response(error_text, is_generating=False)
        else:
            self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.WARNING, status_text=error_text)

    # -----------------------------------------------------------------------
    # TTS & Autonomous Scroll-and-Explain Callbacks
    # -----------------------------------------------------------------------

    def _on_tts_audio_level(self, level: float) -> None:
        self.notch_window.set_audio_level(level)
        self.overlay_window.set_audio_level(level)

    def _on_tts_word_progress(self, sentence_text: str, word_index: int, total_words: int) -> None:
        self.overlay_window.set_karaoke_progress(sentence_text, word_index, total_words)
        if self._force_captions_for_turn or bool(self.settings_store.get("captions_enabled", False)):
            self.overlay_window.set_caption_text(
                sentence_text,
                highlight_word_index=max(0, word_index - 1),
                force_visible=self._force_captions_for_turn,
            )

    def _on_tts_finished(self) -> None:
        self.overlay_window.set_cursor_visual_state(False, False, False)
        if not self._force_captions_for_turn:
            self.overlay_window.clear_caption()
        else:
            QTimer.singleShot(3500, self.overlay_window.clear_caption)
            self._force_captions_for_turn = False

        if self._pending_scroll_after_speech is not None:
            pending_task = self._pending_scroll_after_speech
            self._pending_scroll_after_speech = None
            scheduled_generation_id = self._scroll_generation_id
            self._execute_scroll_and_schedule_next(
                pending_task[0],
                pending_task[1],
                scheduled_generation_id,
            )
            return

        if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.SPEAKING:
            self.notch_window.collapse_to_idle()

    def _execute_scroll_and_schedule_next(
        self,
        scroll_direction: str,
        wheel_ticks: int,
        expected_generation_id: int,
    ) -> None:
        """Scroll the underlying page smoothly, detect end-of-page, and continue explaining."""
        if self._quitting or expected_generation_id != self._scroll_generation_id:
            return

        logical_screen_size = (
            max(640, self.overlay_window.width()),
            max(480, self.overlay_window.height()),
        )
        self._pre_scroll_image = jarvis_services.ScreenCaptureService.capture_logical_snapshot_image(
            logical_screen_size=logical_screen_size
        )

        self.notch_window.set_mode(
            jarvis_notch_window.NotchVisualMode.THINKING,
            status_text=f"Scrolling {scroll_direction}",
        )
        self.overlay_window.set_bubble_message(
            f"Scrolling {scroll_direction} to inspect the next section...",
            word_by_word=False,
            hold_seconds=1.8,
        )
        if self._scroll_explain_active:
            self.overlay_window.clear_viewport_markings_keep_paper()

        jarvis_services.ScreenScrollService.scroll_page_smoothly(
            direction=scroll_direction,
            wheel_ticks=wheel_ticks,
            target_x=logical_screen_size[0] * 0.5,
            target_y=logical_screen_size[1] * 0.5,
        )
        QTimer.singleShot(
            520,
            lambda gen_id=expected_generation_id: self._finish_scroll_after_wheel(gen_id),
        )

    def _finish_scroll_after_wheel(self, expected_generation_id: int) -> None:
        if self._quitting or expected_generation_id != self._scroll_generation_id:
            return
        if not self._scroll_explain_active:
            if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.THINKING:
                self.notch_window.collapse_to_idle()
            return

        logical_screen_size = (
            max(640, self.overlay_window.width()),
            max(480, self.overlay_window.height()),
        )
        post_scroll_image = jarvis_services.ScreenCaptureService.capture_logical_snapshot_image(
            logical_screen_size=logical_screen_size
        )
        if self._pre_scroll_image is not None and post_scroll_image is not None:
            screen_diff_ratio = jarvis_services.ScreenCaptureService.compute_screen_difference_ratio(
                self._pre_scroll_image,
                post_scroll_image,
            )
            if screen_diff_ratio < 0.010:
                self._cancel_scroll_explain_sequence()
                self.overlay_window.set_bubble_message(
                    "Reached the end of the scrollable page.",
                    word_by_word=False,
                    hold_seconds=2.5,
                )
                if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.THINKING:
                    self.notch_window.collapse_to_idle()
                return

        if self._scroll_step_index >= self._max_scroll_steps:
            self._cancel_scroll_explain_sequence()
            if self.notch_window.current_mode() == jarvis_notch_window.NotchVisualMode.THINKING:
                self.notch_window.collapse_to_idle()
            return

        self._scroll_step_index += 1
        continuation_prompt = (
            f"Continue explaining the newly scrolled section visually for: {self._scroll_explain_user_prompt}"
        )
        self._start_turn(
            prompt_text=continuation_prompt,
            speak_reply=self._speak_current_turn,
            from_typed_notch=self._last_turn_from_typed_notch,
            preserve_scroll_session=True,
        )

    # -----------------------------------------------------------------------
    # Settings Test Callbacks
    # -----------------------------------------------------------------------

    def _on_voice_preview_requested(self, voice_id: str) -> None:
        self.sound_player.play_voice_preview(voice_id)

    def _on_microphone_test_requested(self) -> None:
        if self._active_voice_mode != "idle":
            return
        self._active_voice_mode = "mic_test"
        self.notch_window.set_mode(jarvis_notch_window.NotchVisualMode.LISTENING, status_text="Mic Test")
        self.mic_service.start_streaming()
        QTimer.singleShot(2500, self._stop_mic_test_if_active)

    def _stop_mic_test_if_active(self) -> None:
        if self._active_voice_mode == "mic_test":
            self.mic_service.stop_and_finalize()

    def _on_shortcut_test_requested(self, action_id: str) -> None:
        normalized = (action_id or "").strip().lower()
        if normalized == "double_ctrl":
            self.notch_window.toggle_text_input()
        elif normalized == "escape":
            self._on_escape_pressed()
        elif normalized == "demo":
            self.trigger_demo_showcase()
        elif normalized in ("ptt", "dictate", "hands_free_dictate"):
            self._on_microphone_test_requested()
