import cv2
import numpy as np
import sounddevice as sd
import os
import time
import math
import logging
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, QByteArray
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut, QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                               QMessageBox, QProgressBar, QProgressDialog, 
                               QSplitter, QSizePolicy, QTabWidget, QGroupBox)
from app_config import load_settings, save_settings
from ui_styles import dark_stylesheet
from ui_tabs import setup_capture_tab, setup_fx_tab, setup_audio_tab, setup_render_tab
from teleprompter_widget import TeleprompterWidget
from video_engine import VideoEngine
from audio_engine import AudioEngine
from render_worker import RenderWorker
from constants import PALETTES

class WebcamClearVSTPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Recorder - OBS Noise Suppression & Cinematic Palettes")
        self.setStyleSheet(dark_stylesheet())
        
        self.is_recording = False
        self.is_monitoring = False
        self.is_muted = False
        self.video_writer = None
        self.audio_data = []
        self.markers = []
        self.audio_samplerate = 44100
        self.rec_start_time = 0
        self.config_file = "app_settings_obs_ns.json"
        self.auto_stop_timer = None
        self.countdown_timer = None
        self.peak_hold_value = 0.0
        self.peak_hold_time = 0
        self.audio_stream = None
        self.monitor_fade = 0.0   
        self.monitor_logged = False  # Prevents log flooding
        
        # UI Thread-Safety: Variables to hold audio levels for the UI timer to pick up
        self.current_audio_level = 0.0
        self.current_audio_peak = 0.0
        
        self.raw_video_path = "temp_raw_video.avi"
        self.raw_audio_path = "temp_raw_audio.wav"
        self.processed_video_path = "temp_proc_video.avi"
        self.processed_audio_path = "temp_proc_audio.wav"
        
        self.settings = load_settings(self.config_file)
        self.audio_engine = AudioEngine()
        self.video_engine = VideoEngine()
        
        self.restore_geometry()
        self.setup_ui()
        self.populate_audio_devices()
        self.restore_param_values()
        
        self.cap = cv2.VideoCapture(0)
        self.video_timer = QTimer(self)
        self.video_timer.setSingleShot(True)
        self.video_timer.timeout.connect(self.update_video_feed)
        self.video_timer.start(15)
        
        # UI Timer to safely update audio meters without crashing the audio thread
        self.ui_meter_timer = QTimer(self)
        self.ui_meter_timer.timeout.connect(self.update_ui_meters)
        self.ui_meter_timer.start(50) # Update meters 20 times a second

    def restore_geometry(self):
        geo_str = self.settings.get("window_geometry")
        if geo_str:
            try:
                self.restoreGeometry(QByteArray(geo_str.encode('utf-8')))
                return
            except: pass
        self.setGeometry(100, 50, 1400, 860)

    def closeEvent(self, event):
        self.save_settings()
        self.is_recording = False
        self.is_monitoring = False
        self.update_audio_stream()
        if self.cap.isOpened(): self.cap.release()
        event.accept()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)
        
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.main_splitter)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(8)
        self.main_splitter.addWidget(left_widget)
        
        video_group = QLabel(" 📹 Video Preview ")
        video_group.setStyleSheet("font-weight: bold; color: #ff4444; padding: 4px;")
        left_layout.addWidget(video_group)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 4px; border: 1px solid #333;")
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_label, stretch=1)
        
        info_bar = QHBoxLayout()
        info_bar.setSpacing(12)
        self.video_res_label = QLabel("📊 Resolution: --"); self.video_res_label.setStyleSheet("color: #888; font-size: 11px;")
        info_bar.addWidget(self.video_res_label)
        self.video_fps_label = QLabel("🎬 FPS: --"); self.video_fps_label.setStyleSheet("color: #888; font-size: 11px;")
        info_bar.addWidget(self.video_fps_label)
        self.video_filter_label = QLabel("🎨 Filter: --"); self.video_filter_label.setStyleSheet("color: #888; font-size: 11px;")
        info_bar.addWidget(self.video_filter_label)
        info_bar.addStretch()
        self.rec_time_label = QLabel("⏱ 00:00:00"); self.rec_time_label.setStyleSheet("color: #ff4444; font-size: 12px; font-weight: bold; font-family: Consolas;")
        info_bar.addWidget(self.rec_time_label)
        left_layout.addLayout(info_bar)
        
        rec_layout = QHBoxLayout()
        rec_layout.setSpacing(8)
        self.record_btn = QPushButton("▶  Start Recording  (Ctrl+R)")
        self.record_btn.setMinimumHeight(44)
        self.record_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #3a1010; border: 1px solid #ff4444; border-radius: 4px;")
        self.record_btn.clicked.connect(self.toggle_recording)
        rec_layout.addWidget(self.record_btn, stretch=2)
        
        self.process_btn = QPushButton("🔄  Process & Export")
        self.process_btn.setMinimumHeight(44)
        self.process_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #103a10; border: 1px solid #44aa44; border-radius: 4px;")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        rec_layout.addWidget(self.process_btn, stretch=1)
        left_layout.addLayout(rec_layout)
        
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(0)
        self.main_splitter.addWidget(right_widget)
        
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        splitter_sizes = self.settings.get("splitter_sizes", [850, 520])
        self.main_splitter.setSizes(splitter_sizes)
        
        setup_capture_tab(self)
        setup_fx_tab(self)
        setup_audio_tab(self)
        
        self.teleprompter = TeleprompterWidget(self.settings)
        self.tabs.addTab(self.teleprompter, "📝 Teleprompter")
        
        setup_render_tab(self)
        
        self.shortcut_record = QShortcut(QKeySequence("Ctrl+R"), self); self.shortcut_record.activated.connect(self.toggle_recording)
        self.shortcut_mute = QShortcut(QKeySequence("Ctrl+M"), self); self.shortcut_mute.activated.connect(self.toggle_mute)
        self.shortcut_marker = QShortcut(QKeySequence("Ctrl+H"), self); self.shortcut_marker.activated.connect(self.add_marker)
        self.update_video_info()

    def update_video_info(self):
        res_text = self.res_combo.currentText()
        self.video_res_label.setText(f"📊 Resolution: {res_text}")
        fps = "60" if "60fps" in res_text else "30"
        self.video_fps_label.setText(f"🎬 FPS: {fps}")
        if hasattr(self, 'filter_combo'): self.video_filter_label.setText(f"🎨 Filter: {self.filter_combo.currentText()}")

    def refresh_audio_devices(self):
        self.device_combo.clear(); self.populate_audio_devices(); self.status_bar.showMessage("Audio devices refreshed.")

    def restore_param_values(self):
        for key, (sld, scale, default) in self.param_sliders.items():
            if key in self.settings: sld.setValue(int(self.settings[key] * scale))

    def keyPressEvent(self, event):
        if not self.is_recording: super().keyPressEvent(event); return
        self.teleprompter.handle_keypress(event)
        if event.isAccepted(): return
        super().keyPressEvent(event)

    def update_ui_meters(self):
        """Called by UI timer to safely update meters without crashing the audio thread."""
        self.update_audio_level(self.current_audio_level)
        if self.current_audio_peak > 1:
            self.update_audio_peak(self.current_audio_peak)
            self.current_audio_peak = 0.0 # Reset for next interval

    def update_audio_level(self, val):
        self.audio_level_bar.setValue(int(val))
        if val > 85: self.audio_level_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff0000; border-radius: 3px; }")
        elif val > 70: self.audio_level_bar.setStyleSheet("QProgressBar::chunk { background-color: #ffaa00; border-radius: 3px; }")
        else: self.audio_level_bar.setStyleSheet("QProgressBar::chunk { background-color: #44aa44; border-radius: 3px; }")

    def update_audio_peak(self, val):
        now = time.time()
        if val > self.peak_hold_value or now - self.peak_hold_time > 2.0:
            self.peak_hold_value = val; self.peak_hold_time = now
        db_val = 20 * math.log10(max(0.001, self.peak_hold_value / 100.0))
        self.peak_label.setText(f"Peak: {db_val:+.1f} dB")

    def play_beep(self, freq=800, duration=0.15):
        fs = 44100
        t = np.linspace(0, duration, int(fs * duration), False)
        tone = np.sin(freq * t * 2 * np.pi)
        envelope = np.linspace(0, 1, int(fs * 0.01))
        tone[:len(envelope)] *= envelope; tone[-len(envelope):] *= envelope
        sd.play(tone, fs)

    def browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder: self.output_dir_edit.setText(folder)

    def populate_audio_devices(self):
        devices = sd.query_devices()
        input_devices = [d['name'] for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        self.device_combo.addItems(input_devices)
        saved_device = self.settings.get("audio_device", "")
        if saved_device and saved_device in input_devices:
            self.device_combo.setCurrentText(saved_device)

    def apply_preset(self):
        name = self.preset_combo.currentText()
        presets = {
            "Cyberpunk City": {"filter": "RGB Shift", "palette": "Neon Tokyo", "params": {"rgb_shift": 15, "noise": 20, "glitch_bands": 8}},
            "Noir Detective": {"filter": "Sketch Pencil", "palette": "White", "params": {"sketch_scale": 0.8, "blur_k": 7, "invert": 1}},
            "Matrix Code": {"filter": "ASCII Art", "palette": "Matrix Green", "params": {"ascii_block": 8}},
            "Retro VHS": {"filter": "VHS Retro", "palette": "Vaporwave", "params": {"rgb_shift": 10, "noise": 40, "glitch_bands": 12}},
            "Thermal Vision": {"filter": "Thermal Heatmap", "palette": "Inferno", "params": {"blur_k": 3}},
            "Neon Outline": {"filter": "Edge Glow Neon", "palette": "Acid Pink", "params": {"canny_low": 30, "canny_high": 100, "glow": 80}},
            "Pencil Sketch": {"filter": "XDog Sketch", "palette": "White", "params": {"blur_k": 5, "density": 20, "sketch_scale": 0.7}},
            "Anime Ink": {"filter": "Canny Edge", "palette": "White", "params": {"canny_low": 80, "canny_high": 180}},
            "Dreamcore": {"filter": "Edge Glow Neon", "palette": "Cotton Candy", "params": {"canny_low": 40, "canny_high": 120, "glow": 90}}
        }
        if name == "Custom" or name not in presets: return
        p = presets[name]
        self.filter_combo.setCurrentText(p["filter"]); self.palette_combo.setCurrentText(p["palette"])
        QTimer.singleShot(50, lambda: self._apply_preset_params(p["params"]))

    def _apply_preset_params(self, params):
        for key, val in params.items():
            if key in self.param_sliders:
                sld, scale, _ = self.param_sliders[key]
                sld.setValue(int(val * scale))

    def randomize_params(self):
        import random
        for key, (sld, scale, default) in self.param_sliders.items():
            if sld.isVisible():
                min_v, max_v = sld.minimum(), sld.maximum()
                sld.setValue(random.randint(min_v, max_v))
        self.status_bar.showMessage("🎲 Parameters randomized!")

    def reset_current_effect(self):
        for key, (sld, scale, default) in self.param_sliders.items():
            if sld.isVisible(): sld.setValue(int(default / scale))
        self.status_bar.showMessage("⟲ Effect parameters reset to default.")

    def update_palette_preview(self):
        palette_name = self.palette_combo.currentText()
        pixmap = QPixmap(120, 22)
        pixmap.fill(QColor("#2d2d2d"))
        if palette_name != "White":
            stops = PALETTES.get(palette_name)
            if stops:
                painter = QPainter(pixmap)
                gradient = QLinearGradient(0, 0, 120, 0)
                for pos, color in stops:
                    r, g, b = color
                    gradient.setColorAt(pos, QColor(r, g, b))
                painter.fillRect(0, 0, 120, 22, gradient)
                painter.end()
        self.palette_preview_label.setPixmap(pixmap)

    def on_filter_changed(self, *args):
        from constants import FILTER_PARAMS
        filt = self.filter_combo.currentText()
        active_params = set(FILTER_PARAMS.get(filt, []))
        active_groups = set()
        
        for key, (sld, scale, default) in self.param_sliders.items():
            is_active = key in active_params
            sld.setVisible(is_active)
            
            row_widget = sld.parentWidget()
            while row_widget and not isinstance(row_widget, QGroupBox):
                if row_widget.parentWidget() and isinstance(row_widget.parentWidget(), QGroupBox): break
                row_widget = row_widget.parentWidget()
            
            if row_widget:
                row_widget.setVisible(is_active)
            
            parent_widget = sld.parentWidget()
            while parent_widget and not isinstance(parent_widget, QGroupBox):
                parent_widget = parent_widget.parentWidget()
                
            if parent_widget and is_active:
                parent_widget.show()
                active_groups.add(parent_widget.title().strip())
                
        for grp_name, grp_box in self.param_groups.items():
            grp_box.setVisible(grp_name in active_groups)
            
        if hasattr(self, 'filter_combo'): self.video_filter_label.setText(f"🎨 Filter: {filt}")

    def on_param_changed(self, key, val): pass

    def get_video_settings(self):
        return {
            'mirror': self.mirror_check.isChecked(),
            'contrast': self.contrast_slider.value() / 100.0,
            'brightness': self.brightness_slider.value() / 100.0,
            'shorts': self.shorts_check.isChecked(),
            'filter': self.filter_combo.currentText(),
            'palette': self.palette_combo.currentText(),
            'params': {k: (sld.value() / scale) for k, (sld, scale, default) in self.param_sliders.items()}
        }

    def get_audio_settings(self):
        return {
            'gain': self.gain_slider.value() / 100.0,
            'eq_low': self.eq_low_slider.value(), 'eq_mid': self.eq_mid_slider.value(), 'eq_high': self.eq_high_slider.value(),
            'eq_low_q': self.eq_low_q_slider.value() / 100.0, 'eq_mid_q': self.eq_mid_q_slider.value() / 100.0, 'eq_high_q': self.eq_high_q_slider.value() / 100.0,
            'gate_enabled': self.gate_check.isChecked(),
            'gate_thresh': self.gate_slider.value(), 'gate_attack': self.gate_attack_slider.value(), 'gate_release': self.gate_release_slider.value(), 'gate_ratio': self.gate_ratio_slider.value() / 10.0,
            'hp_freq': self.hp_freq_slider.value(), 'yt_loud': self.yt_loud_check.isChecked(),
            'ambient_type': self.ambient_combo.currentText(), 'ambient_vol_db': self.ambient_vol_slider.value(), 'sfx_master_db': self.sfx_master_slider.value(),
            'out_vol': self.out_vol_slider.value() / 100.0,
            'noise_suppress': self.noise_suppress_check.isChecked(), 'noise_suppress_amount': self.ns_amount_slider.value(),
            'noise_suppress_mode': self.ns_mode_combo.currentText(),
            'duck_sfx': self.duck_check.isChecked(), 'deess_thresh': self.deess_slider.value(), 'deess_ratio': self.deess_ratio_slider.value() / 10.0,
            'air_boost': self.air_slider.value() / 10.0,
            'tape_drive': self.tape_slider.value(), 'transient_punch': self.transient_slider.value()
        }

    def toggle_monitor(self):
        self.is_monitoring = self.monitor_btn.isChecked()
        self.monitor_logged = False  # Reset log flag
        success = self.update_audio_stream()
        
        if not success:
            self.is_monitoring = False
            self.monitor_btn.setChecked(False)
            
        if self.is_monitoring:
            self.monitor_btn.setText("⏹  Stop Live Monitor")
            self.status_bar.showMessage("🎧 Live monitor active. Use headphones to prevent feedback!")
            logging.info("Live monitoring enabled.")
        else:
            self.monitor_btn.setText("🎧  Start Live Monitor")
            self.status_bar.showMessage("Monitoring stopped.")
            logging.info("Live monitoring disabled.")

    def update_audio_stream(self):
        should_run = self.is_recording or self.is_monitoring
        
        if not should_run and self.audio_stream is not None:
            try:
                if self.audio_stream.active: self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                logging.error(f"Error stopping audio stream: {e}")
            finally:
                self.audio_stream = None
            logging.info("Audio stream stopped (Idle).")
            return True
            
        if should_run and (self.audio_stream is None or not self.audio_stream.active):
            if self.audio_stream is not None:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                except: pass
                self.audio_stream = None
                
            device_name = self.device_combo.currentText()
            try:
                device_idx = sd.query_devices(device_name, 'input')['index'] if device_name else None
            except Exception:
                device_idx = None
                
            try:
                self.audio_stream = sd.Stream(
                    callback=self.audio_callback, 
                    device=(device_idx, None), 
                    channels=1, 
                    samplerate=self.audio_samplerate, 
                    blocksize=1024, 
                    dtype='float32', 
                    latency='low'
                )
                self.audio_stream.start()
                logging.info(f"Audio stream started (Recording: {self.is_recording}, Monitoring: {self.is_monitoring})")
                return True
            except Exception as e:
                logging.error(f"Failed to start audio stream: {e}", exc_info=True)
                QMessageBox.critical(self, "Audio Error", f"Could not start audio stream:\n{e}\n\nCheck that the device is not in use.")
                self.audio_stream = None
                return False
                
        return True

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            video_settings = self.get_video_settings()
            display_frame = self.video_engine.apply_video_adjustments(frame.copy(), video_settings)
            if self.is_recording:
                if self.is_muted:
                    cv2.circle(display_frame, (20, 20), 10, (0, 0, 255), -1) 
                    cv2.putText(display_frame, "MIC MUTED", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else: cv2.circle(display_frame, (20,20), 10, (0, 255, 0), -1) 
                elapsed = int(time.time() - self.rec_start_time)
                m, s = divmod(elapsed, 60); h, m = divmod(m, 60)
                time_str = f"{h:02d}:{m:02d}:{s:02d}"
                cv2.putText(display_frame, time_str, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.rec_time_label.setText(f"⏱ {time_str}")
            if self.grid_check.isChecked(): display_frame = self.video_engine.draw_grid(display_frame)
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape; bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
            if self.is_recording and self.video_writer is not None: self.video_writer.write(frame) 
        self.video_timer.start(15)

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Unified full-duplex callback. Records input AND routes processed audio to output."""
        try:
            # 1. Capture if recording
            if self.is_recording:
                if self.is_muted:
                    self.audio_data.append(np.zeros_like(indata))
                else:
                    self.audio_data.append(indata.copy())
                
            # 2. Update Level Meter Variables (UI Thread will pick these up safely)
            volume_norm = np.linalg.norm(indata) * 10
            self.current_audio_level = min(volume_norm, 100)
            peak = np.max(np.abs(indata)) * 100
            if peak > self.current_audio_peak:
                self.current_audio_peak = peak
            
            # 3. Monitor output (Hear Myself) with smooth fade to prevent clicks
            target_fade = 1.0 if self.is_monitoring else 0.0
            if self.monitor_fade != target_fade:
                step = (target_fade - self.monitor_fade) / 64.0
                ramp = np.linspace(self.monitor_fade, target_fade, min(64, frames))
                self.monitor_fade = ramp[-1]
                if len(ramp) < frames:
                    ramp = np.pad(ramp, (0, frames - len(ramp)), 'constant', constant_values=target_fade)
            else:
                ramp = np.full(frames, target_fade)

            if self.monitor_fade > 0.0 or target_fade > 0.0:
                audio_settings = self.get_audio_settings()
                
                # SILENCE GUARD: If output volume is 0, skip processing to save CPU and avoid silence
                if audio_settings['out_vol'] <= 0.0:
                    outdata.fill(0)
                    return

                audio_input = np.ascontiguousarray(indata[:, 0])
                processed = self.audio_engine.process_audio_chain(audio_input, self.audio_samplerate, audio_settings, is_live=True)
                
                if processed.ndim > 1:
                    outdata[:, 0] = processed[:, 0] * ramp
                else:
                    outdata[:, 0] = processed * ramp
                
                # Log once to confirm audio is actually being routed
                if not self.monitor_logged and np.max(np.abs(outdata)) > 0.001:
                    logging.info("Monitor audio is being routed to output successfully.")
                    self.monitor_logged = True
                    
            else:
                outdata.fill(0)
                
        except Exception as e:
            # If any error occurs in the audio chain, log it and output silence to prevent stream crash
            if not hasattr(self, '_last_audio_error') or time.time() - self._last_audio_error > 2.0:
                logging.error(f"Audio Callback Error: {e}", exc_info=True)
                self._last_audio_error = time.time()
            outdata.fill(0)

    def toggle_recording(self):
        if not self.is_recording: self.start_recording()
        else: self.stop_recording()

    def toggle_mute(self):
        if not self.is_recording: return
        self.is_muted = not self.is_muted
        if self.is_muted: self.status_bar.showMessage("🔴 Recording | 🔇 MIC MUTED")
        else: self.status_bar.showMessage("🔴 Recording | 🎙️ Mic Live")

    def add_marker(self):
        if not self.is_recording: return
        elapsed = time.time() - self.rec_start_time
        m, s = divmod(int(elapsed), 60); h, m = divmod(m, 60)
        self.markers.append(elapsed)
        self.status_bar.showMessage(f"📌 Marker added at {h:02d}:{m:02d}:{s:02d}")

    def start_recording(self):
        self.record_btn.setEnabled(False)
        self.markers.clear(); self.video_engine.edge_accumulator = None
        countdown_val = int(self.countdown_combo.currentText().replace('s', ''))
        self.countdown(countdown_val)

    def countdown(self, count):
        if count > 0:
            self.status_bar.showMessage(f"Starting in {count}...")
            self.record_btn.setText(f"Starting in {count}...")
            self.play_beep(800, 0.1)
            self.countdown_timer = QTimer(self); self.countdown_timer.setSingleShot(True)
            self.countdown_timer.timeout.connect(lambda: self.countdown(count-1)); self.countdown_timer.start(1000)
        else:
            self.record_btn.setText("Starting..."); self.play_beep(1200, 0.2); self._execute_recording_start()

    def _execute_recording_start(self):
        width, height = map(int, self.res_combo.currentText().split(' ')[0].split('x'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.raw_video_path, fourcc, 20.0, (width, height))
        self.audio_data = []
        
        self.is_recording = True
        success = self.update_audio_stream()
        
        if not success:
            logging.error("Failed to start audio stream, aborting recording.")
            self.is_recording = False
            self.video_writer.release()
            self.record_btn.setText("▶  Start Recording  (Ctrl+R)")
            self.record_btn.setEnabled(True)
            return

        self.is_muted = False
        self.record_btn.setText("⏹ Stop Recording (Ctrl+R)"); self.record_btn.setEnabled(True)
        self.process_btn.setEnabled(False); self.status_bar.showMessage("🔴 Recording Raw Video & Audio...")
        self.rec_start_time = time.time(); self.peak_hold_value = 0
        
        self.teleprompter.start_recording()
        mode = self.teleprompter.tele_mode_combo.currentText()
        if mode == "Bullet Points" and self.teleprompter.bullets:
            self.status_bar.showMessage(f"🔴 Recording | ▶ Bullet 1/{len(self.teleprompter.bullets)}")
            
        auto_stop = self.auto_stop_combo.currentText()
        if auto_stop != "Off":
            ms = 0
            if auto_stop == "15s": ms = 15000
            elif auto_stop == "30s": ms = 30000
            elif auto_stop == "60s": ms = 60000
            elif auto_stop == "3m": ms = 180000
            elif auto_stop == "5m": ms = 300000
            self.auto_stop_timer = QTimer(self); self.auto_stop_timer.setSingleShot(True)
            self.auto_stop_timer.timeout.connect(self.stop_recording); self.auto_stop_timer.start(ms)

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False; self.is_muted = False
        self.record_btn.setText("▶  Start Recording  (Ctrl+R)"); self.process_btn.setEnabled(True)
        self.status_bar.showMessage("⏹ Stopped. Ready to process.")
        if self.auto_stop_timer: self.auto_stop_timer.stop(); self.auto_stop_timer = None
        if self.teleprompter.auto_scroll_timer: self.teleprompter.auto_scroll_timer.stop(); self.teleprompter.auto_scroll_timer = None
        
        self.teleprompter.stop_recording()
        self.rec_time_label.setText("⏱ 00:00:00")
        
        self.update_audio_stream()
        self.video_writer.release()
        logging.info(f"Recording stopped. Captured {len(self.audio_data)} audio chunks.")

    def start_processing(self):
        self.record_btn.setEnabled(False); self.process_btn.setEnabled(False)
        self.progress = QProgressDialog("Rendering final video...", "Cancel", 0, 100, self)
        self.progress.setWindowTitle("Processing"); self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0); self.progress.setValue(0)
        self.video_timer.stop()
        
        title = self.title_edit.text().strip()
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{safe_title}_{timestamp}.mp4" if safe_title else f"Recording_{timestamp}.mp4"
        save_path = os.path.join(self.output_dir_edit.text(), filename)
        
        encoder_str = self.encoder_combo.currentText()
        ffmpeg_codec = "libx264"
        if encoder_str == "NVIDIA NVENC": ffmpeg_codec = "h264_nvenc"
        elif encoder_str == "Intel QSV": ffmpeg_codec = "h264_qsv"
        elif encoder_str == "AMD AMF": ffmpeg_codec = "h264_amf"
        
        render_params = {
            'audio_data': self.audio_data,
            'samplerate': self.audio_samplerate,
            'audio_settings': self.get_audio_settings(),
            'video_settings': self.get_video_settings(),
            'raw_video_path': self.raw_video_path,
            'raw_audio_path': self.raw_audio_path,
            'processed_video_path': self.processed_video_path,
            'processed_audio_path': self.processed_audio_path,
            'save_path': save_path,
            'markers': self.markers,
            'ffmpeg_codec': ffmpeg_codec,
            'render_preset': self.render_combo.currentText()
        }
        
        logging.info(f"Starting render process. Output: {save_path}")
        logging.debug(f"Audio settings for render: {render_params['audio_settings']}")
        
        self.worker = RenderWorker(render_params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_render_finished)
        self.worker.error.connect(self.on_render_error)
        self.worker.start()

    def update_progress(self, val):
        if hasattr(self, 'progress'): self.progress.setValue(val)

    def on_render_finished(self, save_path):
        self.progress.close()
        logging.info(f"Render finished successfully: {save_path}")
        QMessageBox.information(self, "Success", f"✅ Final video saved to:\n{save_path}")
        self.record_btn.setEnabled(True); self.process_btn.setEnabled(True)
        self.status_bar.showMessage("Ready"); self.video_timer.start(15)

    def on_render_error(self, err):
        self.progress.close()
        logging.error(f"Render failed: {err}")
        QMessageBox.critical(self, "Error", f"Processing failed:\n{err}")
        self.record_btn.setEnabled(True); self.process_btn.setEnabled(True)
        self.status_bar.showMessage("Ready"); self.video_timer.start(15)

    def save_settings(self):
        # --- Capture Tab ---
        self.settings["output_dir"] = self.output_dir_edit.text()
        self.settings["resolution"] = self.res_combo.currentText()
        self.settings["countdown_duration"] = self.countdown_combo.currentText()
        self.settings["auto_stop"] = self.auto_stop_combo.currentText()
        
        # --- Live Video Adjustments ---
        self.settings["shorts_mode"] = self.shorts_check.isChecked()
        self.settings["mirror"] = self.mirror_check.isChecked()
        self.settings["show_grid"] = self.grid_check.isChecked()
        self.settings["brightness"] = self.brightness_slider.value() / 100.0
        self.settings["contrast"] = self.contrast_slider.value() / 100.0
        
        # --- Visual FX Tab ---
        self.settings["filter"] = self.filter_combo.currentText()
        self.settings["palette"] = self.palette_combo.currentText()
        for key, (sld, scale, default) in self.param_sliders.items():
            self.settings[key] = sld.value() / scale
            
        # --- Audio Tab ---
        # 1. Input & Monitoring
        self.settings["audio_device"] = self.device_combo.currentText()
        self.settings["input_gain"] = self.gain_slider.value() / 100.0
        self.settings["output_vol"] = self.out_vol_slider.value() / 100.0
        
        # 2. Processing Options
        self.settings["yt_loudness"] = self.yt_loud_check.isChecked()
        self.settings["noise_gate"] = self.gate_check.isChecked()
        self.settings["noise_suppress"] = self.noise_suppress_check.isChecked()
        self.settings["noise_suppress_mode"] = self.ns_mode_combo.currentText()
        self.settings["noise_suppress_amount"] = self.ns_amount_slider.value()
        
        # 3. Highpass & EQ
        self.settings["hp_freq"] = self.hp_freq_slider.value()
        self.settings["eq_low"] = self.eq_low_slider.value()
        self.settings["eq_mid"] = self.eq_mid_slider.value()
        self.settings["eq_high"] = self.eq_high_slider.value()
        self.settings["eq_low_q"] = self.eq_low_q_slider.value() / 100.0
        self.settings["eq_mid_q"] = self.eq_mid_q_slider.value() / 100.0
        self.settings["eq_high_q"] = self.eq_high_q_slider.value() / 100.0
        
        # 4. Dynamics & Saturation
        self.settings["tape_drive"] = self.tape_slider.value()
        self.settings["transient_punch"] = self.transient_slider.value()
        self.settings["deess_thresh"] = self.deess_slider.value()
        self.settings["deess_ratio"] = self.deess_ratio_slider.value() / 10.0
        self.settings["air_boost"] = self.air_slider.value() / 10.0
        
        # 5. Noise Gate Parameters
        self.settings["gate_thresh"] = self.gate_slider.value()
        self.settings["gate_ratio"] = self.gate_ratio_slider.value() / 10.0
        self.settings["gate_attack"] = self.gate_attack_slider.value()
        self.settings["gate_release"] = self.gate_release_slider.value()
        
        # 6. Ambient & SFX
        self.settings["ambient_sound"] = self.ambient_combo.currentText()
        self.settings["ambient_vol_db"] = self.ambient_vol_slider.value()
        self.settings["sfx_master_db"] = self.sfx_master_slider.value()
        self.settings["duck_sfx"] = self.duck_check.isChecked()
        
        # --- Render Tab ---
        self.settings["render_preset"] = self.render_combo.currentText()
        self.settings["encoder"] = self.encoder_combo.currentText()
        
        # --- Teleprompter Tab ---
        self.settings.update(self.teleprompter.save_settings())
        
        # --- Window Layout State ---
        self.settings["splitter_sizes"] = self.main_splitter.sizes()
        self.settings["window_geometry"] = self.saveGeometry().toBase64().data().decode('utf-8')
        
        save_settings(self.config_file, self.settings)