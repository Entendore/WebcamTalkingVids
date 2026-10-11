import cv2
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from pedalboard import (Pedalboard, Gain, Limiter, NoiseGate, 
                        HighpassFilter, LowShelfFilter, PeakFilter, HighShelfFilter)
from moviepy import VideoFileClip, AudioFileClip
import os
import time
import json
import math
import threading
from scipy import signal as scipy_signal
from scipy.ndimage import minimum_filter1d
from datetime import datetime
from numba import jit
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QByteArray
from PySide6.QtGui import QImage, QPixmap, QAction, QKeySequence, QShortcut, QTextCursor, QColor, QPainter, QLinearGradient
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                               QComboBox, QSlider, QCheckBox, QLineEdit, 
                               QFileDialog, QMessageBox, QProgressBar, QSpinBox,
                               QGroupBox, QTextEdit, QScrollBar, QProgressDialog,
                               QScrollArea, QTabWidget, QSplitter, QFrame, QSizePolicy)

@jit(nopython=True, nogil=True)
def apply_gain_numba(audio_data, gain):
    return audio_data * gain

@jit(nopython=True, nogil=True)
def envelope_follower_numba(signal, attack_coeff, release_coeff):
    env = np.zeros_like(signal)
    env[0] = np.abs(signal[0])
    for i in range(1, len(signal)):
        sig_abs = np.abs(signal[i])
        if sig_abs > env[i-1]:
            env[i] = env[i-1] + attack_coeff * (sig_abs - env[i-1])
        else:
            env[i] = env[i-1] + release_coeff * (sig_abs - env[i-1])
    return env

# =========================================================================
# OBS-STYLE NOISE SUPPRESSION CORE (Numba Accelerated)
# =========================================================================
# Based on OBS Studio's Speex and RNNoise implementations:
#   - Speex: minimum statistics noise estimation + Wiener filter + VAD
#   - RNNoise: bark-scale analysis + enhanced VAD + adaptive oversubtraction
# =========================================================================

@jit(nopython=True, nogil=True)
def _obs_speex_core(power, noise_est_init, max_supp, noise_rate):
    """
    OBS Speex-style noise suppression core.
    
    Implements:
      - Minimum statistics noise floor tracking
      - SNR-based Voice Activity Detection (VAD) with threshold
      - Wiener filter gain: G = signal_power / (signal_power + noise_power)
      - Maximum suppression limit (default -30 dB in OBS)
      - Gain smoothing (coefficient 0.9) to prevent musical noise
    """
    n_freqs, n_frames = power.shape
    gain_mask = np.ones_like(power)
    noise_est = noise_est_init.copy()
    prev_gain = np.ones(n_freqs, dtype=np.float32)
    
    # OBS Speex VAD threshold (SNR ratio for speech detection)
    vad_threshold = 2.5
    # Gain smoothing coefficient (OBS uses heavy smoothing)
    gain_smooth = 0.9
    
    for i in range(n_frames):
        # Per-bin SNR for noise update decision
        for f in range(n_freqs):
            ratio = power[f, i] / (noise_est[f] + 1e-10)
            if ratio < vad_threshold:
                # No speech: update noise estimate quickly
                noise_est[f] = (1.0 - noise_rate) * noise_est[f] + noise_rate * power[f, i]
            else:
                # Speech present: update very slowly to preserve signal
                noise_est[f] = (1.0 - 1e-4) * noise_est[f] + 1e-4 * power[f, i]
        
        # Wiener filter gain computation
        signal_pwr = np.maximum(power[:, i] - noise_est, 0.0)
        gain = signal_pwr / (signal_pwr + noise_est + 1e-10)
        
        # Apply maximum suppression limit (OBS default: -30 dB)
        gain = np.maximum(gain, max_supp)
        
        # Smooth gain across frames to prevent musical noise artifacts
        gain = gain_smooth * prev_gain + (1.0 - gain_smooth) * gain
        prev_gain = gain
        
        gain_mask[:, i] = gain
    
    return gain_mask


@jit(nopython=True, nogil=True)
def _obs_rnnoise_core(power, noise_est_init, max_supp):
    """
    OBS RNNoise-inspired noise suppression core.
    
    Implements:
      - Energy-based VAD with hysteresis and state tracking
      - Adaptive noise update rate (fast when silent, very slow during speech)
      - Adaptive oversubtraction factor based on VAD state
      - Enhanced Wiener filter with oversubtraction
      - Stronger gain smoothing (0.92 coefficient)
    
    Note: True RNNoise uses a trained GRU neural network. This implementation
    captures the key signal-processing aspects: bark-scale-inspired band
    analysis, aggressive VAD, and adaptive oversubtraction.
    """
    n_freqs, n_frames = power.shape
    gain_mask = np.ones_like(power)
    noise_est = noise_est_init.copy()
    prev_gain = np.ones(n_freqs, dtype=np.float32)
    
    # VAD state with hysteresis
    vad_state = 0.0
    vad_alpha = 0.95
    vad_onset = 3.0    # SNR to detect speech onset
    vad_offset = 1.5   # SNR below which speech is considered absent
    
    # RNNoise-style stronger smoothing
    gain_smooth = 0.92
    
    for i in range(n_frames):
        # Total frame energy
        total_energy = 0.0
        noise_energy = 0.0
        for f in range(n_freqs):
            total_energy += power[f, i]
            noise_energy += noise_est[f]
        
        # Frame-level SNR for VAD
        snr_estimate = total_energy / (noise_energy + 1e-10)
        
        # VAD with hysteresis
        if snr_estimate > vad_onset:
            vad_state = vad_alpha * vad_state + (1.0 - vad_alpha) * 1.0
        elif snr_estimate > vad_offset:
            vad_state = vad_alpha * vad_state
        else:
            vad_state = vad_alpha * vad_state + (1.0 - vad_alpha) * 0.0
        
        # Adaptive noise update rate
        # Fast update when no speech, very slow during speech
        noise_rate = 0.02 * (1.0 - vad_state) + 0.0001 * vad_state
        
        # Update noise estimate
        for f in range(n_freqs):
            noise_est[f] = (1.0 - noise_rate) * noise_est[f] + noise_rate * power[f, i]
        
        # Enhanced Wiener filter with adaptive oversubtraction
        # More aggressive suppression when no speech detected
        oversub = 1.0 + (1.0 - vad_state) * 2.0
        signal_pwr = np.maximum(power[:, i] - noise_est, 0.0)
        gain = signal_pwr / (signal_pwr + oversub * noise_est + 1e-10)
        gain = np.maximum(gain, max_supp)
        
        # Stronger smoothing for RNNoise mode
        gain = gain_smooth * prev_gain + (1.0 - gain_smooth) * gain
        prev_gain = gain
        
        gain_mask[:, i] = gain
    
    return gain_mask


class WorkerSignals(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    audio_level = Signal(float)
    audio_peak = Signal(float)

class WebcamClearVSTPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Recorder - OBS Noise Suppression & Cinematic Palettes")
        self.setStyleSheet(self.dark_stylesheet())
        
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
        self.current_bullet_idx = 0
        self.bullets = []
        self.edge_accumulator = None
        self.param_sliders = {}
        self.param_groups = {}
        self.monitor_stream = None
        self.peak_hold_value = 0.0
        self.peak_hold_time = 0
        self.auto_scroll_speed = 0
        self.auto_scroll_timer = None
        
        self.raw_video_path = "temp_raw_video.avi"
        self.raw_audio_path = "temp_raw_audio.wav"
        self.processed_video_path = "temp_proc_video.avi"
        self.processed_audio_path = "temp_proc_audio.wav"
        
        self._render_settings = None
        self._render_audio_settings = None
        
        self.signals = WorkerSignals()
        self.signals.audio_level.connect(self.update_audio_level)
        self.signals.audio_peak.connect(self.update_audio_peak)
        self.signals.progress.connect(self.update_progress)
        self.signals.status.connect(self.update_status)
        self.signals.finished.connect(self.on_render_finished)
        self.signals.error.connect(self.on_render_error)
        
        self.load_settings()
        self.restore_geometry()
        self.setup_ui()
        self.populate_audio_devices()
        self.restore_param_values()
        
        self.cap = cv2.VideoCapture(0)
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_feed)
        self.video_timer.start(15)

    def dark_stylesheet(self):
        return """
            QMainWindow, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Segoe UI, Arial; font-size: 13px; }
            QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 12px; padding: 14px 10px 10px 10px; font-weight: bold; color: #ff4444; background-color: #252525; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; background-color: #1e1e1e; }
            QPushButton { background-color: #3a3a3a; color: #ffffff; border: 1px solid #4a4a4a; padding: 7px 14px; border-radius: 4px; }
            QPushButton:hover { background-color: #4a4a4a; border-color: #5a5a5a; }
            QPushButton:pressed { background-color: #2d2d2d; }
            QPushButton:disabled { background-color: #2d2d2d; color: #666666; border-color: #333333; }
            QComboBox, QSpinBox, QLineEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #3a3a3a; border-radius: 4px; padding: 5px 8px; selection-background-color: #ff4444; }
            QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #5a5a5a; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView { background-color: #2d2d2d; color: #ffffff; selection-background-color: #ff4444; border: 1px solid #3a3a3a; outline: none; }
            QSlider::groove:horizontal { border: none; height: 6px; background: #2d2d2d; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #ff4444; border-radius: 3px; }
            QSlider::add-page:horizontal { background: #2d2d2d; border-radius: 3px; }
            QSlider::handle:horizontal { background: #e0e0e0; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; border: 2px solid #ff4444; }
            QSlider::handle:horizontal:hover { background: #ffffff; border-color: #ff6666; width: 18px; height: 18px; }
            QProgressBar { background-color: #2d2d2d; border: 1px solid #3a3a3a; border-radius: 4px; text-align: center; color: #ffffff; }
            QProgressBar::chunk { background-color: #ff4444; border-radius: 3px; }
            QTextEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px; }
            QCheckBox { spacing: 8px; padding: 2px 0; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; }
            QCheckBox::indicator:unchecked { background-color: #2d2d2d; border: 1px solid #555555; border-radius: 3px; }
            QCheckBox::indicator:checked { background-color: #ff4444; border: 1px solid #ff4444; border-radius: 3px; }
            QCheckBox::indicator:hover { border-color: #ff6666; }
            QLabel { color: #d4d4d4; }
            QTabWidget::pane { border: 1px solid #3a3a3a; border-radius: 4px; background: #1e1e1e; }
            QTabBar::tab { background: #2d2d2d; color: #aaaaaa; padding: 8px 16px; border: 1px solid #3a3a3a; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
            QTabBar::tab:selected { background: #252525; color: #ff4444; border-bottom: 2px solid #ff4444; }
            QTabBar::tab:hover { background: #353535; color: #ffffff; }
            QScrollBar:vertical { background: #2d2d2d; width: 10px; border: none; }
            QScrollBar::handle:vertical { background: #4a4a4a; min-height: 30px; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #5a5a5a; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0; }
            QScrollBar:horizontal { background: #2d2d2d; height: 10px; border: none; }
            QScrollBar::handle:horizontal { background: #4a4a4a; min-width: 30px; border-radius: 5px; }
            QScrollBar::handle:horizontal:hover { background: #5a5a5a; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 0; }
            QSplitter::handle { background-color: #3a3a3a; }
            QSplitter::handle:horizontal { width: 4px; }
            QSplitter::handle:vertical { height: 4px; }
            QSplitter::handle:hover { background-color: #ff4444; }
            QFrame[frameShape="4"] { background-color: #333333; max-height: 1px; margin: 4px 0; }
        """

    def _create_section_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("QLabel { color: #888888; font-size: 11px; font-weight: bold; padding: 10px 0 2px 0; border-bottom: 1px solid #333333; }")
        return label

    def _create_slider_row(self, label_text, min_val, max_val, default_val, scale=1.0, suffix="", format_str=None, label_width=130, value_width=70, custom_formatter=None, show_reset=True, tooltip=""):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(8)
        label = QLabel(label_text); label.setMinimumWidth(label_width); label.setToolTip(tooltip)
        layout.addWidget(label)
        slider = QSlider(Qt.Horizontal); slider.setRange(min_val, max_val); slider.setValue(int(default_val / scale)); slider.setToolTip(tooltip)
        layout.addWidget(slider, stretch=1)
        value_label = QLabel(); value_label.setMinimumWidth(value_width); value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setStyleSheet("color: #ff6666; font-weight: bold; font-family: Consolas, 'Courier New', monospace;")
        def update_label(v):
            real_val = v / scale
            if custom_formatter: value_label.setText(custom_formatter(v, real_val))
            elif format_str: value_label.setText(format_str.format(real_val))
            else: value_label.setText(f"{real_val:.2f}{suffix}")
        slider.valueChanged.connect(update_label); update_label(slider.value())
        layout.addWidget(value_label)
        if show_reset:
            reset_btn = QPushButton("⟲"); reset_btn.setFixedSize(22, 22); reset_btn.setStyleSheet("padding: 0; margin: 0; font-size: 11px;")
            reset_btn.setToolTip("Reset to default"); reset_btn.clicked.connect(lambda: slider.setValue(int(default_val / scale)))
            layout.addWidget(reset_btn)
        return widget, slider, value_label

    def load_settings(self):
        defaults = {
            "output_dir": os.getcwd(), "resolution": "1280x720", "filter": "Canny Edge",
            "input_gain": 1.0, "output_vol": 1.0, "mirror": True, "brightness": 1.0, "contrast": 1.0,
            "shorts_mode": False, "yt_loudness": True, "noise_gate": True, "render_preset": "medium",
            "encoder": "Software (x264)", "tele_font": 18, "show_grid": False, "tele_mode": "Manual Scroll",
            "countdown_duration": "3s", "palette": "Neon Tokyo",
            "canny_low": 50, "canny_high": 150, "ghost_decay": 85, "blur_k": 5, "edge_hue": 180, "glow": 50,
            "pix_block": 24, "ascii_block": 10, "mesh_pts": 150, "bin_thresh": 128, "sketch_scale": 0.6, 
            "stipple": 50, "invert": 0, "noise": 30, "density": 15, "rgb_shift": 5, "glitch_bands": 5,
            "eq_low": 0, "eq_mid": 0, "eq_high": 0, "eq_low_q": 0.7, "eq_mid_q": 1.0, "eq_high_q": 0.7,
            "gate_thresh": -45, "gate_attack": 5, "gate_release": 250, "gate_ratio": 4.0,
            "hp_freq": 85,
            "noise_suppress": True, "noise_suppress_amount": 30, "noise_suppress_mode": "Speex",
            "deess_thresh": -35, "deess_ratio": 3.0, "air_boost": 3.0,
            "tape_drive": 15, "transient_punch": 30,
            "ambient_sound": "None", "ambient_vol_db": -18, "sfx_master_db": 0, "duck_sfx": True,
            "splitter_sizes": [850, 520], "tele_auto_scroll_speed": 0, "tele_max_visible": 5,
            "tele_text_color": "#ffffff", "tele_highlight_color": "#ff4444"
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    defaults.update(loaded)
            except: pass
        self.settings = defaults

    def save_settings(self):
        self.settings["output_dir"] = self.output_dir_edit.text()
        self.settings["resolution"] = self.res_combo.currentText()
        self.settings["filter"] = self.filter_combo.currentText()
        self.settings["palette"] = self.palette_combo.currentText()
        self.settings["input_gain"] = self.gain_slider.value() / 100.0
        self.settings["output_vol"] = self.out_vol_slider.value() / 100.0
        self.settings["mirror"] = self.mirror_check.isChecked()
        self.settings["brightness"] = self.brightness_slider.value() / 100.0
        self.settings["contrast"] = self.contrast_slider.value() / 100.0
        self.settings["shorts_mode"] = self.shorts_check.isChecked()
        self.settings["yt_loudness"] = self.yt_loud_check.isChecked()
        self.settings["noise_gate"] = self.gate_check.isChecked()
        self.settings["render_preset"] = self.render_combo.currentText()
        self.settings["encoder"] = self.encoder_combo.currentText()
        self.settings["tele_font"] = self.tele_font_spin.value()
        self.settings["show_grid"] = self.grid_check.isChecked()
        self.settings["tele_mode"] = self.tele_mode_combo.currentText()
        self.settings["countdown_duration"] = self.countdown_combo.currentText()
        self.settings["tele_auto_scroll_speed"] = self.tele_scroll_slider.value()
        self.settings["tele_max_visible"] = self.tele_max_visible_spin.value()
        
        for key, (sld, scale, default) in self.param_sliders.items():
            self.settings[key] = sld.value() / scale
            
        self.settings["eq_low"] = self.eq_low_slider.value()
        self.settings["eq_mid"] = self.eq_mid_slider.value()
        self.settings["eq_high"] = self.eq_high_slider.value()
        self.settings["eq_low_q"] = self.eq_low_q_slider.value() / 100.0
        self.settings["eq_mid_q"] = self.eq_mid_q_slider.value() / 100.0
        self.settings["eq_high_q"] = self.eq_high_q_slider.value() / 100.0
        self.settings["gate_thresh"] = self.gate_slider.value()
        self.settings["gate_attack"] = self.gate_attack_slider.value()
        self.settings["gate_release"] = self.gate_release_slider.value()
        self.settings["gate_ratio"] = self.gate_ratio_slider.value() / 10.0
        self.settings["hp_freq"] = self.hp_freq_slider.value()
        self.settings["ambient_sound"] = self.ambient_combo.currentText()
        self.settings["ambient_vol_db"] = self.ambient_vol_slider.value()
        self.settings["sfx_master_db"] = self.sfx_master_slider.value()
        self.settings["noise_suppress"] = self.noise_suppress_check.isChecked()
        self.settings["noise_suppress_amount"] = self.ns_amount_slider.value()
        self.settings["noise_suppress_mode"] = self.ns_mode_combo.currentText()
        self.settings["duck_sfx"] = self.duck_check.isChecked()
        self.settings["deess_thresh"] = self.deess_slider.value()
        self.settings["deess_ratio"] = self.deess_ratio_slider.value() / 10.0
        self.settings["air_boost"] = self.air_slider.value() / 10.0
        
        self.settings["tape_drive"] = self.tape_slider.value()
        self.settings["transient_punch"] = self.transient_slider.value()
        
        self.settings["splitter_sizes"] = self.main_splitter.sizes()
        self.settings["window_geometry"] = self.saveGeometry().toBase64().data().decode('utf-8')
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f)

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
        if self.cap.isOpened(): self.cap.release()
        if self.is_monitoring: self.toggle_monitor()
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
        
        video_group = QGroupBox(" 📹 Video Preview ")
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(6, 18, 6, 6)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 4px; border: 1px solid #333;")
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_label, stretch=1)
        
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
        video_layout.addLayout(info_bar)
        left_layout.addWidget(video_group, stretch=1)
        
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
        
        # TAB 1: CAPTURE
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(10, 10, 10, 10)
        tab1_layout.setSpacing(8)
        cap_group = QGroupBox(" Capture Settings ")
        cap_layout = QGridLayout(cap_group)
        cap_layout.setSpacing(8); cap_layout.setContentsMargins(12, 18, 12, 12); lbl_w = 110
        lbl = QLabel("Video Title:"); lbl.setMinimumWidth(lbl_w); cap_layout.addWidget(lbl, 0, 0)
        self.title_edit = QLineEdit(); self.title_edit.setPlaceholderText("Leave blank for auto-generated name")
        cap_layout.addWidget(self.title_edit, 0, 1, 1, 3)
        lbl = QLabel("Resolution/FPS:"); lbl.setMinimumWidth(lbl_w); cap_layout.addWidget(lbl, 1, 0)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["640x480", "1280x720", "1280x720 60fps", "1920x1080", "1920x1080 60fps"])
        self.res_combo.setCurrentText(self.settings["resolution"]); self.res_combo.currentTextChanged.connect(self.update_video_info)
        cap_layout.addWidget(self.res_combo, 1, 1, 1, 3)
        cap_layout.addWidget(QLabel("Countdown:"), 2, 0)
        self.countdown_combo = QComboBox(); self.countdown_combo.addItems(["3s", "5s", "10s"])
        self.countdown_combo.setCurrentText(self.settings["countdown_duration"]); cap_layout.addWidget(self.countdown_combo, 2, 1)
        cap_layout.addWidget(QLabel("Auto-Stop:"), 2, 2)
        self.auto_stop_combo = QComboBox(); self.auto_stop_combo.addItems(["Off", "15s", "30s", "60s", "3m", "5m"])
        cap_layout.addWidget(self.auto_stop_combo, 2, 3)
        cap_layout.addWidget(QLabel("Output Folder:"), 3, 0)
        self.output_dir_edit = QLineEdit(self.settings["output_dir"]); cap_layout.addWidget(self.output_dir_edit, 3, 1, 1, 2)
        browse_btn = QPushButton("Browse..."); browse_btn.clicked.connect(self.browse_output_dir)
        cap_layout.addWidget(browse_btn, 3, 3)
        self.shorts_check = QCheckBox("📱  Shorts Mode  (9:16 Vertical with Blurred Background)")
        self.shorts_check.setChecked(self.settings["shorts_mode"]); cap_layout.addWidget(self.shorts_check, 4, 0, 1, 4)
        tab1_layout.addWidget(cap_group)
        
        adj_group = QGroupBox(" Live Video Adjustments ")
        adj_layout = QVBoxLayout(adj_group); adj_layout.setSpacing(4); adj_layout.setContentsMargins(12, 18, 12, 12)
        chk_row = QHBoxLayout(); chk_row.setSpacing(20)
        self.mirror_check = QCheckBox("Mirror Video"); self.mirror_check.setChecked(self.settings["mirror"]); chk_row.addWidget(self.mirror_check)
        self.grid_check = QCheckBox("Rule of Thirds Grid"); self.grid_check.setChecked(self.settings["show_grid"]); chk_row.addWidget(self.grid_check)
        chk_row.addStretch(); adj_layout.addLayout(chk_row)
        bw, self.brightness_slider, _ = self._create_slider_row("Brightness:", 30, 200, int(self.settings["brightness"] * 100), scale=100.0, format_str="{:.2f}x"); adj_layout.addWidget(bw)
        cw, self.contrast_slider, _ = self._create_slider_row("Contrast:", 30, 200, int(self.settings["contrast"] * 100), scale=100.0, format_str="{:.2f}x"); adj_layout.addWidget(cw)
        tab1_layout.addWidget(adj_group); tab1_layout.addStretch()
        self.tabs.addTab(tab1, "🎥 Capture")
        
        # TAB 2: VISUAL FX
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(10, 10, 10, 10); tab2_layout.setSpacing(8)
        preset_group = QGroupBox(" Quick Presets ")
        preset_layout = QHBoxLayout(preset_group); preset_layout.setContentsMargins(12, 18, 12, 12)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Cyberpunk City", "Noir Detective", "Matrix Code", "Retro VHS", "Thermal Vision", "Neon Outline", "Pencil Sketch", "Anime Ink", "Dreamcore"])
        preset_layout.addWidget(self.preset_combo, stretch=1)
        apply_preset_btn = QPushButton("Apply"); apply_preset_btn.clicked.connect(self.apply_preset); preset_layout.addWidget(apply_preset_btn)
        randomize_btn = QPushButton("🎲 Randomize"); randomize_btn.clicked.connect(self.randomize_params); preset_layout.addWidget(randomize_btn)
        reset_btn = QPushButton("⟲ Reset Effect"); reset_btn.clicked.connect(self.reset_current_effect); preset_layout.addWidget(reset_btn)
        tab2_layout.addWidget(preset_group)
        
        fx_group = QGroupBox(" Visual Effects Engine ")
        fx_layout = QGridLayout(fx_group); fx_layout.setSpacing(8); fx_layout.setContentsMargins(12, 18, 12, 12)
        fx_layout.addWidget(QLabel("Visual Effect:"), 0, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Canny Edge", "Ghost Edges", "Laplacian Edge", "Sobel Edge", "XDog Sketch", "Adaptive Lines", "Morph Gradient", "Static Glitch", "Film Grain Edges", "RGB Shift", "Sketch Pencil", "Edge Glow Neon", "Wireframe Mesh", "ASCII Art", "Thermal Heatmap", "Heavy Pixelate", "Dotted Stipple", "Threshold Binary", "VHS Retro", "Posterize", "Contour Fill", "Digital Halftone"])
        self.filter_combo.setCurrentText(self.settings["filter"]); self.filter_combo.currentIndexChanged.connect(self.on_filter_changed)
        fx_layout.addWidget(self.filter_combo, 0, 1)
        
        fx_layout.addWidget(QLabel("Color Palette:"), 1, 0)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems([
            "White", "Neon Cyan", "Matrix Green", "Inferno", "Ice Blue", "Acid Pink",
            "Cyberpunk", "Sunset", "Ocean Depth", "Fire", "Viridis", "Aurora", 
            "Toxic", "Vaporwave", "Gold", "Rainbow Flow", "Plasma", "Magma", 
            "Tropical", "Synthwave", "Neon Tokyo", "Cotton Candy", "Lava Reef", "Deep Space", "Turbo", "Cividis",
            "Emerald Forest", "Sahara Gold", "Arctic Frost", "Mojave Dusk"
        ])
        self.palette_combo.setCurrentText(self.settings.get("palette", "Neon Tokyo"))
        self.palette_combo.currentIndexChanged.connect(self.update_palette_preview)
        fx_layout.addWidget(self.palette_combo, 1, 1)
        
        self.palette_preview_label = QLabel()
        self.palette_preview_label.setMinimumHeight(22)
        self.palette_preview_label.setStyleSheet("border: 1px solid #333; border-radius: 4px;")
        fx_layout.addWidget(self.palette_preview_label, 1, 2)
        tab2_layout.addWidget(fx_group)
        
        param_container = QWidget()
        param_container_layout = QVBoxLayout(param_container)
        param_container_layout.setContentsMargins(0, 0, 0, 0); param_container_layout.setSpacing(8)
        self.PARAM_DEFS = {
            "canny_low":    {"label": "Edge Low Threshold", "group": "Edge Detection",  "min": 10,  "max": 200, "default": 50,  "scale": 1.0,   "tooltip": "Lower bound for edge detection. Lower values detect more edges."},
            "canny_high":   {"label": "Edge High Threshold","group": "Edge Detection",  "min": 50,  "max": 400, "default": 150, "scale": 1.0,   "tooltip": "Upper bound for edge detection. Higher values detect fewer edges."},
            "density":      {"label": "Line Density",      "group": "Edge Detection",  "min": 1,   "max": 50,  "default": 15,  "scale": 1.0,   "tooltip": "Controls the density of lines or morphological operations."},
            "bin_thresh":   {"label": "Binary Threshold",  "group": "Edge Detection",  "min": 0,   "max": 255, "default": 128, "scale": 1.0,   "tooltip": "Threshold value for binary conversion."},
            "blur_k":       {"label": "Blur Strength",     "group": "Pre-processing",  "min": 1,   "max": 31,  "default": 5,   "scale": 1.0,   "tooltip": "Gaussian blur kernel size applied before edge detection."},
            "glow":         {"label": "Glow Intensity",    "group": "Stylization",     "min": 0,   "max": 100, "default": 50,  "scale": 1.0,   "suffix": "%", "tooltip": "Intensity of the neon glow effect."},
            "ghost_decay":  {"label": "Trail Decay",       "group": "Stylization",     "min": 50,  "max": 99,  "default": 85,  "scale": 100.0, "tooltip": "How fast previous frames fade away. 99% = long trails."},
            "pix_block":    {"label": "Pixel Block Size",  "group": "Stylization",     "min": 8,   "max": 80,  "default": 24,  "scale": 1.0,   "suffix": "px","tooltip": "Size of pixelation blocks."},
            "ascii_block":  {"label": "ASCII Density",     "group": "Stylization",     "min": 4,   "max": 24,  "default": 10,  "scale": 1.0,   "suffix": "px","tooltip": "Size of grid blocks for ASCII characters."},
            "mesh_pts":     {"label": "Mesh Density",      "group": "Stylization",     "min": 30,  "max": 400, "default": 150, "scale": 1.0,   "tooltip": "Number of points used to generate the wireframe mesh."},
            "sketch_scale": {"label": "Sketch Scale",      "group": "Stylization",     "min": 10,  "max": 100, "default": 60,  "scale": 100.0, "tooltip": "Brightness/scale multiplier for pencil sketches."},
            "stipple":      {"label": "Stipple Density",   "group": "Stylization",     "min": 10,  "max": 100, "default": 50,  "scale": 1.0,   "suffix": "%", "tooltip": "Density of dots in the stipple effect."},
            "noise":        {"label": "Noise Intensity",   "group": "Glitch & Noise",  "min": 0,   "max": 100, "default": 30,  "scale": 1.0,   "suffix": "%", "tooltip": "Amount of random noise or grain applied."},
            "rgb_shift":    {"label": "Channel Shift",     "group": "Glitch & Noise",  "min": 0,   "max": 50,  "default": 5,   "scale": 1.0,   "suffix": "px","tooltip": "Pixels to shift RGB channels for chromatic aberration."},
            "glitch_bands": {"label": "Glitch Bands",      "group": "Glitch & Noise",  "min": 0,   "max": 30,  "default": 5,   "scale": 1.0,   "tooltip": "Number of horizontal glitch slices."},
            "invert":       {"label": "Invert Colors",     "group": "Colorization",    "min": 0,   "max": 1,   "default": 0,   "scale": 1.0,   "tooltip": "Invert the final output colors (0=Off, 1=On)."}
        }
        categories = {}
        for key, defn in self.PARAM_DEFS.items():
            grp = defn["group"]
            if grp not in categories:
                grp_box = QGroupBox(f" {grp} ")
                grp_layout = QVBoxLayout(grp_box); grp_layout.setSpacing(4); grp_layout.setContentsMargins(12, 18, 12, 12)
                categories[grp] = (grp_box, grp_layout)
                param_container_layout.addWidget(grp_box)
                self.param_groups[grp] = grp_box
        for key, defn in self.PARAM_DEFS.items():
            grp_box, grp_layout = categories[defn["group"]]
            suffix = defn.get("suffix", "")
            w, sld, _ = self._create_slider_row(defn["label"], defn["min"], defn["max"], defn["default"], scale=defn["scale"], suffix=suffix, tooltip=defn.get("tooltip", ""))
            grp_layout.addWidget(w)
            self.param_sliders[key] = (sld, defn["scale"], defn["default"])
            sld.valueChanged.connect(lambda v, k=key: self.on_param_changed(k, v))
        param_scroll = QScrollArea(); param_scroll.setWidgetResizable(True); param_scroll.setFrameShape(QScrollArea.NoFrame); param_scroll.setWidget(param_container)
        tab2_layout.addWidget(param_scroll, stretch=1)
        self.on_filter_changed(); self.update_palette_preview()
        self.tabs.addTab(tab2, "🎨 Visual FX")
        
        # TAB 3: EXPERT AUDIO
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.setContentsMargins(10, 10, 10, 10); tab3_layout.setSpacing(8)
        tab3_content = QWidget()
        tab3_content_layout = QVBoxLayout(tab3_content); tab3_content_layout.setSpacing(8)
        
        audio_group = QGroupBox(" 🎙️ Audio Input & Monitoring ")
        audio_layout = QVBoxLayout(audio_group); audio_layout.setSpacing(6); audio_layout.setContentsMargins(12, 18, 12, 12)
        dev_row = QHBoxLayout(); dev_row.setSpacing(10)
        lbl = QLabel("Device:"); lbl.setMinimumWidth(130); dev_row.addWidget(lbl)
        self.device_combo = QComboBox(); dev_row.addWidget(self.device_combo, stretch=1)
        refresh_btn = QPushButton("🔄"); refresh_btn.setFixedSize(28, 28); refresh_btn.setToolTip("Refresh devices"); refresh_btn.clicked.connect(self.refresh_audio_devices)
        dev_row.addWidget(refresh_btn); audio_layout.addLayout(dev_row)
        
        lvl_row = QHBoxLayout(); lvl_row.setSpacing(10)
        lbl = QLabel("Input Level:"); lbl.setMinimumWidth(130); lvl_row.addWidget(lbl)
        meter_container = QWidget(); meter_layout = QVBoxLayout(meter_container); meter_layout.setContentsMargins(0, 0, 0, 0); meter_layout.setSpacing(1)
        self.audio_level_bar = QProgressBar(); self.audio_level_bar.setMaximum(100); self.audio_level_bar.setTextVisible(True); self.audio_level_bar.setFormat("%v%"); self.audio_level_bar.setMinimumHeight(18)
        meter_layout.addWidget(self.audio_level_bar)
        self.peak_label = QLabel("Peak: -- dB"); self.peak_label.setStyleSheet("color: #888; font-size: 10px; font-family: Consolas;"); self.peak_label.setAlignment(Qt.AlignRight)
        meter_layout.addWidget(self.peak_label)
        lvl_row.addWidget(meter_container, stretch=1); audio_layout.addLayout(lvl_row)
        
        gain_w, self.gain_slider, self.gain_val_lbl = self._create_slider_row("Input Gain:", 0, 500, int(self.settings["input_gain"] * 100), scale=100.0, format_str="{:.2f}x")
        audio_layout.addWidget(gain_w)
        tab3_content_layout.addWidget(audio_group)
        
        proc_group = QGroupBox(" Processing Options ")
        proc_layout = QVBoxLayout(proc_group); proc_layout.setSpacing(5); proc_layout.setContentsMargins(12, 18, 12, 12)
        self.yt_loud_check = QCheckBox("🎙️  Broadcast Loudness  (Limiter)"); self.yt_loud_check.setChecked(self.settings["yt_loudness"]); proc_layout.addWidget(self.yt_loud_check)
        self.gate_check = QCheckBox("🔇  Noise Gate  (Cuts background hum when silent)"); self.gate_check.setChecked(self.settings["noise_gate"]); proc_layout.addWidget(self.gate_check)
        self.noise_suppress_check = QCheckBox("🧹  OBS Noise Suppression  (Speex / RNNoise)")
        self.noise_suppress_check.setChecked(self.settings.get("noise_suppress", True)); proc_layout.addWidget(self.noise_suppress_check)
        tab3_content_layout.addWidget(proc_group)
        
        fx_group = QGroupBox(" 🎚️ Expert Voice Design ")
        fx_layout = QVBoxLayout(fx_group); fx_layout.setSpacing(2); fx_layout.setContentsMargins(12, 18, 12, 12)
        
        fx_layout.addWidget(self._create_section_label("  HIGHPASS FILTER  (Removes low rumble)"))
        hp_w, self.hp_freq_slider, self.hp_freq_lbl = self._create_slider_row("Cutoff Frequency:", 20, 300, self.settings.get("hp_freq", 85), format_str="{:.0f} Hz")
        fx_layout.addWidget(hp_w)
        
        fx_layout.addWidget(self._create_section_label("  EQUALIZER  (3-BAND with Q)"))
        eq_low_w, self.eq_low_slider, self.eq_low_lbl = self._create_slider_row("Low (Bass):", -24, 24, self.settings.get("eq_low", 0), format_str="{:+.1f} dB"); fx_layout.addWidget(eq_low_w)
        eq_low_q_w, self.eq_low_q_slider, _ = self._create_slider_row("  └ Low Q:", 10, 500, int(self.settings.get("eq_low_q", 0.7) * 100), scale=100.0, format_str="{:.2f}"); fx_layout.addWidget(eq_low_q_w)
        eq_mid_w, self.eq_mid_slider, self.eq_mid_lbl = self._create_slider_row("Mid:", -24, 24, self.settings.get("eq_mid", 0), format_str="{:+.1f} dB"); fx_layout.addWidget(eq_mid_w)
        eq_mid_q_w, self.eq_mid_q_slider, _ = self._create_slider_row("  └ Mid Q:", 10, 500, int(self.settings.get("eq_mid_q", 1.0) * 100), scale=100.0, format_str="{:.2f}"); fx_layout.addWidget(eq_mid_q_w)
        eq_high_w, self.eq_high_slider, self.eq_high_lbl = self._create_slider_row("High (Treble):", -24, 24, self.settings.get("eq_high", 0), format_str="{:+.1f} dB"); fx_layout.addWidget(eq_high_w)
        eq_high_q_w, self.eq_high_q_slider, _ = self._create_slider_row("  └ High Q:", 10, 500, int(self.settings.get("eq_high_q", 0.7) * 100), scale=100.0, format_str="{:.2f}"); fx_layout.addWidget(eq_high_q_w)
        
        fx_layout.addWidget(self._create_section_label("  EXPERT DYNAMICS  (Saturation & Punch)"))
        tape_w, self.tape_slider, self.tape_lbl = self._create_slider_row("Tape Saturation:", 0, 100, self.settings.get("tape_drive", 15), format_str="{:.0f}%", tooltip="Adds analog warmth and harmonic distortion.")
        fx_layout.addWidget(tape_w)
        trans_w, self.transient_slider, self.trans_lbl = self._create_slider_row("Transient Punch:", 0, 100, self.settings.get("transient_punch", 30), format_str="{:.0f}%", tooltip="Enhances the attack of consonants for crispness.")
        fx_layout.addWidget(trans_w)
        
        fx_layout.addWidget(self._create_section_label("  DYNAMIC DE-ESSER  (Tames harsh 'S' sounds)"))
        deess_w, self.deess_slider, self.deess_lbl = self._create_slider_row("De-Ess Threshold:", -60, 0, self.settings.get("deess_thresh", -35), format_str="{:+.0f} dB"); fx_layout.addWidget(deess_w)
        deess_ratio_w, self.deess_ratio_slider, _ = self._create_slider_row("De-Ess Ratio:", 10, 100, int(self.settings.get("deess_ratio", 3.0) * 10), scale=10.0, format_str="{:.1f}:1"); fx_layout.addWidget(deess_ratio_w)
        
        fx_layout.addWidget(self._create_section_label("  OBS NOISE SUPPRESSION  (Speex / RNNoise)"))
        ns_mode_row = QHBoxLayout(); ns_mode_row.setSpacing(10)
        lbl = QLabel("Algorithm:"); lbl.setMinimumWidth(130); ns_mode_row.addWidget(lbl)
        self.ns_mode_combo = QComboBox()
        self.ns_mode_combo.addItems(["Speex", "RNNoise"])
        self.ns_mode_combo.setCurrentText(self.settings.get("noise_suppress_mode", "Speex"))
        self.ns_mode_combo.setToolTip("Speex: Lightweight spectral subtraction (OBS default).\nRNNoise: Neural-inspired with adaptive oversubtraction.")
        ns_mode_row.addWidget(self.ns_mode_combo, stretch=1)
        fx_layout.addLayout(ns_mode_row)
        ns_w, self.ns_amount_slider, self.ns_amount_lbl = self._create_slider_row("Suppression:", 3, 60, self.settings.get("noise_suppress_amount", 30), format_str="{:.0f} dB", tooltip="Maximum noise attenuation. OBS default: 30 dB."); fx_layout.addWidget(ns_w)
        
        fx_layout.addWidget(self._create_section_label("  NOISE GATE "))
        gate_w, self.gate_slider, self.gate_lbl = self._create_slider_row("Gate Threshold:", -90, -10, self.settings.get("gate_thresh", -45), format_str="{:.0f} dB"); fx_layout.addWidget(gate_w)
        gate_ratio_w, self.gate_ratio_slider, _ = self._create_slider_row("Gate Ratio:", 10, 100, int(self.settings.get("gate_ratio", 4.0) * 10), scale=10.0, format_str="{:.1f}:1"); fx_layout.addWidget(gate_ratio_w)
        gate_attack_w, self.gate_attack_slider, _ = self._create_slider_row("Gate Attack:", 1, 100, self.settings.get("gate_attack", 5), format_str="{:.0f} ms"); fx_layout.addWidget(gate_attack_w)
        gate_release_w, self.gate_release_slider, _ = self._create_slider_row("Gate Release:", 10, 1000, self.settings.get("gate_release", 250), format_str="{:.0f} ms"); fx_layout.addWidget(gate_release_w)
        
        fx_layout.addWidget(self._create_section_label("  VOICE PRESENCE  (Adds 'Air' when speaking)"))
        air_w, self.air_slider, self.air_lbl = self._create_slider_row("Air Enhancer:", 0, 120, int(self.settings.get("air_boost", 3.0) * 10), scale=10.0, format_str="{:+.1f} dB"); fx_layout.addWidget(air_w)
        
        fx_layout.addWidget(self._create_section_label("  AMBIENT SOUND GENERATOR "))
        amb_row = QHBoxLayout(); amb_row.setSpacing(10)
        lbl = QLabel("Background:"); lbl.setMinimumWidth(130); amb_row.addWidget(lbl)
        self.ambient_combo = QComboBox()
        self.ambient_combo.addItems(["None", "White Noise", "Pink Noise", "Brown Noise", "Light Rain", "Heavy Rain", "Thunderstorm", "Rain on Tin Roof", "Ocean Waves", "Gentle Stream", "Waterfall", "Wind", "Fireplace", "Forest Night", "Birdsong Morning", "Café Ambience", "Keyboard Typing", "Clock Ticking", "Distant Traffic", "Heartbeat", "Space Drone"])
        self.ambient_combo.setCurrentText(self.settings.get("ambient_sound", "None"))
        amb_row.addWidget(self.ambient_combo, stretch=1); fx_layout.addLayout(amb_row)
        
        def amb_vol_formatter(v, real_val):
            if v <= -60: return "Muted"
            return f"{v:+d} dB"
        amb_vol_w, self.ambient_vol_slider, self.amb_vol_lbl = self._create_slider_row("SFX Volume:", -60, 6, self.settings.get("ambient_vol_db", -18), custom_formatter=amb_vol_formatter); fx_layout.addWidget(amb_vol_w)
        sfx_master_w, self.sfx_master_slider, self.sfx_master_lbl = self._create_slider_row("SFX Master:", -12, 12, self.settings.get("sfx_master_db", 0), format_str="{:+.0f} dB"); fx_layout.addWidget(sfx_master_w)
        self.duck_check = QCheckBox("🤖  Auto-Duck SFX  (Lowers ambient when you speak)"); self.duck_check.setChecked(self.settings.get("duck_sfx", True)); fx_layout.addWidget(self.duck_check)
        
        fx_layout.addWidget(self._create_section_label("  OUTPUT"))
        out_vol_w, self.out_vol_slider, self.out_vol_val_lbl = self._create_slider_row("Output Volume:", 0, 500, int(self.settings["output_vol"] * 100), scale=100.0, format_str="{:.2f}x"); fx_layout.addWidget(out_vol_w)
        self.monitor_btn = QPushButton("🎧  Start Live Monitor  (Full Chain)"); self.monitor_btn.setMinimumHeight(36); self.monitor_btn.clicked.connect(self.toggle_monitor); fx_layout.addWidget(self.monitor_btn)
        tab3_content_layout.addWidget(fx_group); tab3_content_layout.addStretch()
        
        audio_scroll = QScrollArea(); audio_scroll.setWidgetResizable(True); audio_scroll.setFrameShape(QScrollArea.NoFrame); audio_scroll.setWidget(tab3_content)
        tab3_layout.addWidget(audio_scroll)
        self.tabs.addTab(tab3, "🎙️ Audio")
        
        # TAB 4: TELEPROMPTER
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        tab4_layout.setContentsMargins(10, 10, 10, 10); tab4_layout.setSpacing(8)
        tele_group = QGroupBox(" 📝 Teleprompter ")
        tele_layout = QVBoxLayout(tele_group); tele_layout.setSpacing(8); tele_layout.setContentsMargins(12, 18, 12, 12)
        self.tele_text = QTextEdit()
        self.tele_text.setStyleSheet(f"font-size: {self.settings['tele_font']}pt; line-height: 1.6; padding: 16px;")
        self.tele_text.setMinimumHeight(200); tele_layout.addWidget(self.tele_text, stretch=1)
        self.tele_progress = QProgressBar(); self.tele_progress.setMaximumHeight(8); self.tele_progress.setTextVisible(False)
        self.tele_progress.setStyleSheet("QProgressBar { background-color: #2d2d2d; border: none; border-radius: 4px; } QProgressBar::chunk { background-color: #ff4444; border-radius: 4px; }")
        tele_layout.addWidget(self.tele_progress)
        self.tele_counter_label = QLabel("📊 Bullet: 0 / 0"); self.tele_counter_label.setStyleSheet("color: #888; font-size: 12px; font-weight: bold; padding: 2px 0;")
        tele_layout.addWidget(self.tele_counter_label)
        tele_ctrl = QHBoxLayout(); tele_ctrl.setSpacing(8)
        lbl = QLabel("Mode:"); lbl.setMinimumWidth(50); tele_ctrl.addWidget(lbl)
        self.tele_mode_combo = QComboBox(); self.tele_mode_combo.addItems(["Manual Scroll", "Bullet Points", "Auto Scroll"])
        self.tele_mode_combo.setCurrentText(self.settings["tele_mode"]); self.tele_mode_combo.currentTextChanged.connect(self.on_tele_mode_changed); tele_ctrl.addWidget(self.tele_mode_combo)
        tele_ctrl.addSpacing(12); lbl = QLabel("Font:"); lbl.setMinimumWidth(40); tele_ctrl.addWidget(lbl)
        self.tele_font_spin = QSpinBox(); self.tele_font_spin.setRange(8, 72); self.tele_font_spin.setValue(self.settings["tele_font"]); self.tele_font_spin.valueChanged.connect(self.update_tele_font); tele_ctrl.addWidget(self.tele_font_spin)
        tele_ctrl.addSpacing(12); lbl = QLabel("Visible:"); lbl.setMinimumWidth(50); tele_ctrl.addWidget(lbl)
        self.tele_max_visible_spin = QSpinBox(); self.tele_max_visible_spin.setRange(3, 15); self.tele_max_visible_spin.setValue(self.settings.get("tele_max_visible", 5))
        self.tele_max_visible_spin.setToolTip("Number of bullets visible in preview"); self.tele_max_visible_spin.valueChanged.connect(self.update_bullet_display); tele_ctrl.addWidget(self.tele_max_visible_spin)
        tele_ctrl.addStretch()
        self.tele_prev_btn = QPushButton("◀ Prev"); self.tele_prev_btn.setFixedSize(70, 28); self.tele_prev_btn.clicked.connect(self.prev_bullet); tele_ctrl.addWidget(self.tele_prev_btn)
        self.tele_next_btn = QPushButton("Next ▶"); self.tele_next_btn.setFixedSize(70, 28); self.tele_next_btn.clicked.connect(self.next_bullet); tele_ctrl.addWidget(self.tele_next_btn)
        tele_layout.addLayout(tele_ctrl)
        
        auto_scroll_row = QHBoxLayout(); auto_scroll_row.setSpacing(8)
        lbl = QLabel("Auto-Scroll Speed:"); lbl.setMinimumWidth(130); auto_scroll_row.addWidget(lbl)
        self.tele_scroll_slider = QSlider(Qt.Horizontal); self.tele_scroll_slider.setRange(0, 100); self.tele_scroll_slider.setValue(self.settings.get("tele_auto_scroll_speed", 0))
        self.tele_scroll_slider.valueChanged.connect(self.update_auto_scroll_speed); auto_scroll_row.addWidget(self.tele_scroll_slider, stretch=1)
        self.tele_scroll_label = QLabel("Off"); self.tele_scroll_label.setMinimumWidth(50); self.tele_scroll_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.tele_scroll_label.setStyleSheet("color: #ff6666; font-weight: bold;"); auto_scroll_row.addWidget(self.tele_scroll_label)
        tele_layout.addLayout(auto_scroll_row)
        self.tele_hint_label = QLabel("▶  [Space / →] Next    ◀  [←] Previous    [Ctrl + / −] Font Size    [Ctrl+Enter] Jump to Start    [Ctrl+End] Jump to End")
        self.tele_hint_label.setStyleSheet("color: #888888; font-size: 11px; padding: 4px 0;"); self.tele_hint_label.setWordWrap(True); tele_layout.addWidget(self.tele_hint_label)
        help_label = QLabel("💡 <b>Manual Scroll:</b> Paste your script and use arrow keys to scroll.<br>💡 <b>Bullet Points:</b> Each line becomes a bullet. Current is highlighted, next/prev shown for context.<br>💡 <b>Auto Scroll:</b> Script scrolls automatically at the set speed.")
        help_label.setStyleSheet("color: #666; font-size: 11px; padding: 8px; background: #2d2d2d; border-radius: 4px;"); help_label.setWordWrap(True); tele_layout.addWidget(help_label)
        tab4_layout.addWidget(tele_group)
        self.tabs.addTab(tab4, "📝 Teleprompter")
        
        # TAB 5: RENDER
        tab5 = QWidget()
        tab5_layout = QVBoxLayout(tab5)
        tab5_layout.setContentsMargins(10, 10, 10, 10); tab5_layout.setSpacing(8)
        render_group = QGroupBox(" Render & Export Settings ")
        render_layout = QGridLayout(render_group); render_layout.setSpacing(8); render_layout.setContentsMargins(12, 18, 12, 12); lbl_w = 130
        lbl = QLabel("Render Quality:"); lbl.setMinimumWidth(lbl_w); render_layout.addWidget(lbl, 0, 0)
        self.render_combo = QComboBox(); self.render_combo.addItems(["ultrafast", "fast", "medium", "slow"])
        self.render_combo.setCurrentText(self.settings["render_preset"]); render_layout.addWidget(self.render_combo, 0, 1)
        lbl = QLabel("Encoder:"); lbl.setMinimumWidth(lbl_w); render_layout.addWidget(lbl, 1, 0)
        self.encoder_combo = QComboBox(); self.encoder_combo.addItems(["Software (x264)", "NVIDIA NVENC", "Intel QSV", "AMD AMF"])
        self.encoder_combo.setCurrentText(self.settings["encoder"]); render_layout.addWidget(self.encoder_combo, 1, 1)
        tab5_layout.addWidget(render_group); tab5_layout.addStretch()
        self.tabs.addTab(tab5, "⚙️ Render")
        
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

    def on_tele_mode_changed(self):
        mode = self.tele_mode_combo.currentText()
        if mode == "Auto Scroll": self.tele_text.setReadOnly(True)
        else: self.tele_text.setReadOnly(False)
        if mode == "Bullet Points": self.tele_scroll_slider.setEnabled(False)
        else: self.tele_scroll_slider.setEnabled(True)

    def update_auto_scroll_speed(self, val):
        if val == 0:
            self.tele_scroll_label.setText("Off")
            if self.auto_scroll_timer: self.auto_scroll_timer.stop(); self.auto_scroll_timer = None
        else:
            self.tele_scroll_label.setText(f"{val}%")
            if self.is_recording and self.tele_mode_combo.currentText() in ["Auto Scroll", "Manual Scroll"]: self.start_auto_scroll(val)

    def start_auto_scroll(self, speed):
        if self.auto_scroll_timer: self.auto_scroll_timer.stop()
        interval = max(10, int(200 - speed * 1.9))
        self.auto_scroll_timer = QTimer(self); self.auto_scroll_timer.timeout.connect(lambda: self.scroll_manual(1)); self.auto_scroll_timer.start(interval)

    def restore_param_values(self):
        for key, (sld, scale, default) in self.param_sliders.items():
            if key in self.settings: sld.setValue(int(self.settings[key] * scale))

    def keyPressEvent(self, event):
        if not self.is_recording: super().keyPressEvent(event); return
        mode = self.tele_mode_combo.currentText()
        if mode == "Bullet Points":
            if event.key() in [Qt.Key_Space, Qt.Key_Right, Qt.Key_Down]: self.next_bullet(); event.accept(); return
            elif event.key() in [Qt.Key_Left, Qt.Key_Up]: self.prev_bullet(); event.accept(); return
        elif mode in ["Manual Scroll", "Auto Scroll"]:
            if event.key() in [Qt.Key_Space, Qt.Key_Right, Qt.Key_Down]: self.scroll_manual(1); event.accept(); return
            elif event.key() in [Qt.Key_Left, Qt.Key_Up]: self.scroll_manual(-1); event.accept(); return
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal: self.increase_tele_font(); event.accept(); return
            elif event.key() == Qt.Key_Minus: self.decrease_tele_font(); event.accept(); return
            elif event.key() == Qt.Key_Return: self.current_bullet_idx = 0; self.update_bullet_display(); event.accept(); return
        super().keyPressEvent(event)

    def increase_tele_font(self):
        self.tele_font_spin.setValue(self.tele_font_spin.value() + 1); self.status_bar.showMessage(f"Teleprompter Font Size: {self.tele_font_spin.value()}")

    def decrease_tele_font(self):
        self.tele_font_spin.setValue(self.tele_font_spin.value() - 1); self.status_bar.showMessage(f"Teleprompter Font Size: {self.tele_font_spin.value()}")

    def update_tele_font(self):
        if self.tele_mode_combo.currentText() == "Bullet Points" and self.is_recording: self.update_bullet_display()
        else: self.tele_text.setStyleSheet(f"font-size: {self.tele_font_spin.value()}pt; line-height: 1.6; padding: 16px;")

    def scroll_manual(self, direction):
        scrollbar = self.tele_text.verticalScrollBar()
        step = self.tele_text.fontMetrics().height() * 2
        scrollbar.setValue(scrollbar.value() + (step * direction)); self.status_bar.showMessage("🔴 Recording...")

    def next_bullet(self):
        if self.current_bullet_idx < len(self.bullets) - 1:
            self.current_bullet_idx += 1; self.update_bullet_display()
            pct = int((self.current_bullet_idx + 1) / max(1, len(self.bullets)) * 100)
            self.status_bar.showMessage(f"▶ Bullet {self.current_bullet_idx + 1}/{len(self.bullets)} ({pct}%)")

    def prev_bullet(self):
        if self.current_bullet_idx > 0:
            self.current_bullet_idx -= 1; self.update_bullet_display()
            pct = int((self.current_bullet_idx + 1) / max(1, len(self.bullets)) * 100)
            self.status_bar.showMessage(f"▶ Bullet {self.current_bullet_idx + 1}/{len(self.bullets)} ({pct}%)")

    def update_bullet_display(self):
        if not self.bullets: return
        base_font = self.tele_font_spin.value(); max_visible = self.tele_max_visible_spin.value()
        half = max_visible // 2
        start_idx = max(0, self.current_bullet_idx - half)
        end_idx = min(len(self.bullets), start_idx + max_visible)
        if end_idx - start_idx < max_visible: start_idx = max(0, end_idx - max_visible)
        html = ""
        if start_idx > 0: html += f"<p style='color:#444445; font-size:{base_font - 4}pt; text-align:center; margin:4px 0;'>··· ({start_idx} more above) ···</p>"
        for i in range(start_idx, end_idx):
            bullet = self.bullets[i]; distance = abs(i - self.current_bullet_idx)
            if i == self.current_bullet_idx:
                html += f"<div style='background-color: #2a1a1a; border-left: 4px solid #ff4444; padding: 16px 12px; margin: 12px 0; border-radius: 4px;'><p style='color:#ff4444; font-size:{base_font + 8}pt; font-weight:bold; margin:0; line-height:1.4;'>▶ {bullet}</p></div>"
            elif distance == 1: html += f"<p style='color:#aaaabb; font-size:{base_font}pt; margin:8px 0; padding-left:20px;'>{bullet}</p>"
            elif distance == 2: html += f"<p style='color:#777788; font-size:{base_font - 2}pt; margin:6px 0; padding-left:20px;'>{bullet}</p>"
            else: html += f"<p style='color:#555566; font-size:{base_font - 4}pt; margin:4px 0; padding-left:20px;'>{bullet}</p>"
        if end_idx < len(self.bullets):
            remaining = len(self.bullets) - end_idx
            html += f"<p style='color:#444455; font-size:{base_font - 4}pt; text-align:center; margin:4px 0;'>··· ({remaining} more below) ···</p>"
        self.tele_text.setHtml(html)
        if len(self.bullets) > 0:
            progress_val = int((self.current_bullet_idx + 1) / len(self.bullets) * 100)
            self.tele_progress.setValue(progress_val)
            self.tele_counter_label.setText(f"📊 Bullet: {self.current_bullet_idx + 1} / {len(self.bullets)}  ({progress_val}%)")
        cursor = self.tele_text.textCursor()
        block = self.tele_text.document().findBlockByNumber(self.current_bullet_idx - start_idx)
        if block.isValid(): cursor.setPosition(block.position()); self.tele_text.setTextCursor(cursor); self.tele_text.ensureCursorVisible()

    def update_audio_level(self, val):
        self.audio_level_bar.setValue(int(val))
        if val > 85: self.audio_level_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff0000; border-radius: 3px; }")
        elif val > 70: self.audio_level_bar.setStyleSheet("QProgressBar::chunk { background-color: #ffaa00; border-radius: 3px; }")
        else: self.audio_level_bar.setStyleSheet("QProgressBar::chunk { background-color: #44aa44; border-radius: 3px; }")

    def update_audio_peak(self, val):
        import time
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
            all_stops = self.get_palette_stops_dict()
            stops = all_stops.get(palette_name)
            if stops:
                painter = QPainter(pixmap)
                gradient = QLinearGradient(0, 0, 120, 0)
                for pos, color in stops:
                    r, g, b = color
                    gradient.setColorAt(pos, QColor(r, g, b))
                painter.fillRect(0, 0, 120, 22, gradient)
                painter.end()
        self.palette_preview_label.setPixmap(pixmap)

    def get_palette_stops(self, palette_name=None):
        """Returns a dict mapping palette name -> list of (pos, [r,g,b]) stops."""
        return {
            "White":          [(0.0, [0, 0, 0]), (1.0, [255, 255, 255])],
            "Neon Cyan":      [(0.0, [0, 0, 0]),    (0.25, [0, 25, 45]),  (0.55, [0, 160, 200]), (0.80, [60, 240, 255]), (1.0, [200, 255, 255])],
            "Matrix Green":   [(0.0, [0, 0, 0]),    (0.25, [0, 30, 0]),   (0.50, [0, 110, 25]),  (0.75, [80, 230, 80]),  (1.0, [220, 255, 220])],
            "Ice Blue":       [(0.0, [0, 5, 20]),   (0.30, [20, 70, 110]),(0.60, [80, 160, 230]),(0.85, [180, 230, 255]),(1.0, [255, 255, 255])],
            "Acid Pink":      [(0.0, [0, 0, 0]),    (0.30, [50, 0, 80]),  (0.60, [190, 0, 180]), (0.85, [255, 60, 230]), (1.0, [255, 210, 255])],
            "Inferno":        [(0.00, [0, 0, 0]),   (0.15, [40, 0, 60]),  (0.35, [110, 0, 130]), (0.55, [200, 40, 80]),  (0.75, [255, 120, 10]), (0.90, [255, 220, 80]), (1.00, [255, 255, 220])],
            "Cyberpunk":      [(0.00, [0, 0, 0]),   (0.15, [20, 0, 50]),  (0.30, [120, 0, 180]), (0.50, [255, 0, 150]),  (0.70, [0, 200, 255]),  (0.85, [100, 255, 255]),(1.00, [255, 255, 255])],
            "Sunset":         [(0.00, [10, 0, 30]), (0.20, [60, 10, 80]), (0.40, [150, 20, 100]),(0.60, [220, 30, 60]),  (0.80, [255, 140, 30]), (1.00, [255, 240, 150])],
            "Ocean Depth":    [(0.00, [0, 5, 20]),  (0.20, [0, 20, 60]),  (0.40, [0, 60, 120]),  (0.60, [0, 120, 160]),  (0.80, [80, 220, 220]), (1.00, [240, 255, 255])],
            "Fire":           [(0.00, [0, 0, 0]),   (0.20, [80, 0, 0]),   (0.40, [180, 20, 0]),  (0.60, [240, 80, 0]),   (0.80, [255, 160, 30]), (0.95, [255, 230, 80]), (1.00, [255, 255, 230])],
            "Viridis":        [(0.00, [68, 1, 84]), (0.20, [72, 35, 116]),(0.40, [64, 67, 135]), (0.60, [52, 94, 141]),  (0.80, [38, 130, 142]), (1.00, [253, 231, 37])],
            "Aurora":         [(0.00, [0, 5, 20]),  (0.20, [0, 40, 30]),  (0.40, [0, 100, 80]),  (0.60, [0, 230, 130]),  (0.80, [80, 220, 220]), (1.00, [180, 100, 240])],
            "Toxic":          [(0.00, [0, 0, 0]),   (0.25, [0, 40, 0]),   (0.50, [40, 120, 10]), (0.75, [120, 220, 30]), (0.90, [220, 255, 60]), (1.00, [255, 255, 200])],
            "Vaporwave":      [(0.00, [10, 0, 30]), (0.20, [50, 0, 80]),  (0.40, [120, 20, 140]),(0.60, [255, 60, 180]), (0.80, [100, 220, 255]),(1.00, [255, 240, 255])],
            "Gold":           [(0.00, [0, 0, 0]),   (0.25, [40, 10, 0]),  (0.50, [120, 60, 10]), (0.75, [220, 150, 40]), (0.90, [255, 215, 80]), (1.00, [255, 250, 220])],
            "Rainbow Flow":   [(0.00, [255, 0, 0]), (0.17, [255, 165, 0]),(0.33, [255, 255, 0]), (0.50, [0, 200, 0]),    (0.67, [0, 200, 255]),  (0.83, [0, 0, 255]),    (1.00, [180, 0, 255])],
            "Plasma":         [(0.00, [13, 8, 135]),(0.25, [85, 15, 140]),(0.50, [166, 40, 133]),(0.75, [226, 100, 98]), (1.00, [240, 249, 33])],
            "Magma":          [(0.00, [0, 0, 4]),   (0.20, [40, 15, 60]), (0.40, [120, 30, 80]), (0.60, [200, 60, 60]),  (0.80, [240, 120, 40]), (1.00, [255, 240, 180])],
            "Tropical":       [(0.00, [0, 60, 80]), (0.25, [0, 120, 110]),(0.50, [80, 180, 70]), (0.75, [220, 200, 50]), (1.00, [240, 100, 30])],
            "Synthwave":      [(0.00, [20, 0, 40]), (0.25, [70, 15, 100]),(0.50, [160, 40, 130]),(0.75, [230, 80, 110]), (1.00, [255, 150, 80])],
            "Neon Tokyo":     [(0.00, [5, 5, 20]),  (0.15, [20, 5, 80]),  (0.30, [220, 0, 120]), (0.50, [255, 50, 150]), (0.70, [0, 240, 255]),  (0.85, [200, 255, 255]),(1.00, [255, 255, 255])],
            "Cotton Candy":   [(0.00, [50, 0, 80]), (0.20, [100, 50, 200]),(0.40, [255, 100, 200]),(0.60, [255, 200, 220]),(0.80, [100, 200, 255]),(1.00, [255, 255, 255])],
            "Lava Reef":      [(0.00, [10, 5, 5]),  (0.20, [80, 10, 0]),  (0.40, [200, 40, 10]), (0.60, [255, 120, 20]), (0.80, [255, 220, 60]), (1.00, [255, 255, 200])],
            "Deep Space":     [(0.00, [0, 0, 10]),  (0.25, [10, 10, 60]), (0.50, [80, 0, 120]),  (0.75, [200, 50, 200]), (1.00, [255, 200, 255])],
            "Turbo":          [(0.00, [48, 18, 59]),(0.10, [62, 73, 137]),(0.20, [64, 120, 179]),(0.30, [63, 163, 175]),(0.40, [70, 200, 122]),(0.50, [94, 221, 74]),(0.60, [140, 220, 42]),(0.70, [191, 211, 36]),(0.80, [236, 189, 43]),(0.90, [250, 147, 58]),(1.00, [248, 99, 69])],
            "Cividis":        [(0.00, [0, 32, 76]), (0.20, [60, 70, 110]),(0.40, [120, 100, 100]),(0.60, [180, 140, 80]),(0.80, [220, 180, 60]),(1.00, [255, 233, 69])],
            "Emerald Forest": [(0.0, [0, 10, 5]),   (0.25, [10, 40, 15]), (0.50, [20, 110, 45]), (0.75, [90, 200, 110]), (1.0, [220, 255, 220])],
            "Sahara Gold":    [(0.0, [20, 10, 0]),  (0.25, [80, 40, 10]), (0.50, [180, 110, 30]),(0.75, [240, 180, 60]), (1.0, [255, 240, 180])],
            "Arctic Frost":   [(0.0, [10, 20, 30]), (0.30, [50, 100, 140]),(0.60, [150, 200, 230]),(0.85, [220, 240, 250]),(1.0, [255, 255, 255])],
            "Mojave Dusk":    [(0.0, [20, 10, 30]), (0.25, [80, 30, 60]), (0.50, [180, 60, 80]), (0.75, [240, 130, 60]), (1.0, [255, 220, 150])]
        }

    # =========================================================================
    # OBS-STYLE NOISE SUPPRESSION (Speex & RNNoise modes)
    # =========================================================================
    def obs_noise_suppression(self, audio, sr, suppression_db=30.0, mode="Speex"):
        """
        OBS Studio-style noise suppression.
        
        Mode "Speex": Minimum statistics + Wiener filter + VAD
          - Lightweight spectral subtraction approach
          - OBS's default noise suppression algorithm
          
        Mode "RNNoise": Neural-inspired + adaptive oversubtraction + enhanced VAD
          - More aggressive suppression with bark-scale-inspired analysis
          - Adaptive oversubtraction based on VAD state
          - Closer to OBS's RNNoise neural network approach
        """
        if len(audio) < sr * 0.1: return audio
        
        # STFT parameters (OBS uses similar frame sizes)
        n_fft = 2048; hop = 512
        f, t, Zxx = scipy_signal.stft(audio, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, boundary='zeros', padded=True)
        mag = np.abs(Zxx).astype(np.float32)
        phase = np.angle(Zxx).astype(np.float32)
        power = mag ** 2
        
        # Estimate initial noise floor from first ~0.8 seconds (OBS approach)
        init_frames = max(1, int(0.8 * sr / hop))
        noise_power_init = np.mean(power[:, :init_frames], axis=1).astype(np.float32)
        
        # Maximum suppression factor (OBS default: -30 dB)
        max_supp = 10.0 ** (-suppression_db / 20.0)
        
        if mode == "RNNoise":
            gain_mask = _obs_rnnoise_core(power, noise_power_init, max_supp)
        else:
            # Speex mode with configurable noise update rate
            noise_rate = 0.01  # OBS Speex slow update rate
            gain_mask = _obs_speex_core(power, noise_power_init, max_supp, noise_rate)
        
        # Apply gain mask
        cleaned_mag = mag * gain_mask
        cleaned_Zxx = cleaned_mag * np.exp(1j * phase)
        
        # Inverse STFT
        _, cleaned = scipy_signal.istft(cleaned_Zxx, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, boundary=True)
        
        # Match output length
        if len(cleaned) > len(audio): cleaned = cleaned[:len(audio)]
        elif len(cleaned) < len(audio): cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))
        
        # Fade in/out to prevent clicks at boundaries
        n_fade = min(512, len(cleaned) // 4)
        if n_fade > 1:
            fade = np.linspace(0, 1, n_fade, dtype=np.float32)
            cleaned[:n_fade]  = audio[:n_fade]  * fade + cleaned[:n_fade]  * (1 - fade)
            cleaned[-n_fade:] = audio[-n_fade:] * fade[::-1] + cleaned[-n_fade:] * (1 - fade[::-1])
        
        return cleaned.astype(np.float32)

    def apply_deesser(self, audio, sr, threshold_db=-35.0, ratio=3.0):
        b, a = scipy_signal.butter(4, 6000.0 / (sr / 2.0), btype='high')
        high_band = scipy_signal.lfilter(b, a, audio).astype(np.float32)
        low_band = audio - high_band
        env = np.abs(high_band)
        b_env, a_env = scipy_signal.butter(2, 100.0 / (sr / 2.0), btype='low')
        env = scipy_signal.lfilter(b_env, a_env, env)
        env_db = 20 * np.log10(env + 1e-8)
        over = np.maximum(env_db - threshold_db, 0.0)
        gr_db = -over * (ratio - 1.0)
        gain = (10 ** (gr_db / 20.0)).astype(np.float32)
        return (low_band + high_band * gain).astype(np.float32)

    def apply_tape_saturation(self, audio, drive_pct):
        if drive_pct <= 0: return audio
        drive = 1.0 + (drive_pct / 100.0) * 3.0
        return np.tanh(audio * drive) / np.tanh(drive)

    def apply_transient_shaper(self, audio, sr, punch_pct):
        if punch_pct <= 0: return audio
        env = np.abs(audio).astype(np.float32)
        attack_env = scipy_signal.lfilter(*scipy_signal.butter(2, 100.0/(sr/2.0)), env)
        sustain_env = scipy_signal.lfilter(*scipy_signal.butter(2, 10.0/(sr/2.0)), env)
        transient = np.maximum(0, attack_env - sustain_env)
        if transient.max() > 0: transient /= transient.max()
        return (audio + (audio * transient * (punch_pct / 100.0) * 1.5)).astype(np.float32)

    def _transient_env(self, n, sr, attack_ms=2.0, decay_rate=100.0):
        if n <= 0: return np.zeros(0, dtype=np.float32)
        t = np.arange(n, dtype=np.float32) / sr
        env = np.exp(-t * decay_rate).astype(np.float32)
        a_len = max(1, int(sr * attack_ms / 1000.0))
        if a_len < n:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(a_len) / a_len)).astype(np.float32)
            env[:a_len] *= ramp
        elif n > 0:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).astype(np.float32)
            env *= ramp
        return env

    def _normalize_layer(self, x, target_peak=0.7):
        p = float(np.max(np.abs(x))) if x.size else 0.0
        if p > 1e-9: return (x * (target_peak / p)).astype(np.float32)
        return x.astype(np.float32)

    def _remove_dc(self, x):
        return (x - np.mean(x)).astype(np.float32)

    def _soft_clip(self, x, drive=1.2):
        return (np.tanh(x * drive) / np.tanh(drive)).astype(np.float32)

    def _smooth_sfx_bus(self, x, sr):
        if sr > 24000:
            b, a = scipy_signal.butter(2, 12000.0 / (sr / 2.0), btype='low')
            x = scipy_signal.lfilter(b, a, x).astype(np.float32)
        return self._remove_dc(x)

    def generate_ambient_sound(self, duration_sec, sr, sound_type):
        n_samples = int(duration_sec * sr)
        if n_samples <= 0: return np.zeros(n_samples, dtype=np.float32)
        audio = np.zeros(n_samples, dtype=np.float32)
        t_full = np.arange(n_samples) / sr

        if sound_type == "White Noise":
            audio = np.random.randn(n_samples).astype(np.float32) * 0.5
            b, a = scipy_signal.butter(2, 8000 / (sr / 2), btype='low')
            audio = scipy_signal.lfilter(b, a, audio).astype(np.float32)
        elif sound_type == "Pink Noise":
            white = np.random.randn(n_samples).astype(np.float32)
            num_rows = 16
            rows = np.random.randn(num_rows, n_samples)
            rows[:, 1:] = np.cumsum(rows[:, 1:], axis=1)
            rows *= np.exp(-np.linspace(0, 6, num_rows))[:, None]
            audio = np.mean(rows, axis=0).astype(np.float32)
        elif sound_type == "Brown Noise":
            white = np.random.randn(n_samples).astype(np.float32)
            leak = 0.997
            brown = np.zeros_like(white)
            for i in range(1, n_samples): brown[i] = leak * brown[i-1] + white[i]
            audio = brown.astype(np.float32)
        elif sound_type in ("Light Rain", "Heavy Rain"):
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [400 / (sr / 2), 8000 / (sr / 2)], btype='band')
            wash = scipy_signal.lfilter(b, a, white).astype(np.float32)
            density = n_samples // 80 if sound_type == "Light Rain" else n_samples // 25
            drops = np.zeros(n_samples, dtype=np.float32)
            for idx in np.random.randint(0, n_samples, density):
                drop_len = int(sr * np.random.uniform(0.002, 0.008))
                if idx + drop_len < n_samples:
                    t_drop = np.arange(drop_len) / sr
                    env = self._transient_env(drop_len, sr, attack_ms=1.0, decay_rate=400.0)
                    freq = np.random.uniform(2000, 6000)
                    drops[idx:idx + drop_len] += np.sin(2 * np.pi * freq * t_drop) * env * 0.3
            rumble = np.random.randn(n_samples).astype(np.float32)
            b_lp, a_lp = scipy_signal.butter(2, 200 / (sr / 2), btype='low')
            rumble = scipy_signal.lfilter(b_lp, a_lp, rumble).astype(np.float32) * 0.15
            intensity = 0.7 if sound_type == "Light Rain" else 1.0
            audio = (wash * 0.5 + drops * 0.5 + rumble) * intensity
        elif sound_type == "Rain on Tin Roof":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [600 / (sr / 2), 10000 / (sr / 2)], btype='band')
            wash = scipy_signal.lfilter(b, a, white).astype(np.float32) * 0.3
            drops = np.zeros(n_samples, dtype=np.float32)
            for idx in np.random.randint(0, n_samples, n_samples // 30):
                drop_len = int(sr * 0.02)
                if idx + drop_len < n_samples:
                    t_drop = np.arange(drop_len) / sr
                    env = self._transient_env(drop_len, sr, attack_ms=1.5, decay_rate=150.0)
                    fund = np.random.uniform(1500, 3500)
                    sig = (np.sin(2 * np.pi * fund * t_drop) + 0.5 * np.sin(2 * np.pi * fund * 2.7 * t_drop) + 0.3 * np.sin(2 * np.pi * fund * 5.2 * t_drop))
                    drops[idx:idx + drop_len] += sig * env * 0.25
            audio = wash + drops
        elif sound_type == "Thunderstorm":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [400 / (sr / 2), 8000 / (sr / 2)], btype='band')
            wash = scipy_signal.lfilter(b, a, white).astype(np.float32)
            drops = np.zeros(n_samples, dtype=np.float32)
            for idx in np.random.randint(0, n_samples, n_samples // 25):
                drop_len = int(sr * np.random.uniform(0.002, 0.008))
                if idx + drop_len < n_samples:
                    t_drop = np.arange(drop_len) / sr
                    env = self._transient_env(drop_len, sr, attack_ms=1.0, decay_rate=400.0)
                    freq = np.random.uniform(2000, 6000)
                    drops[idx:idx + drop_len] += np.sin(2 * np.pi * freq * t_drop) * env * 0.3
            audio = wash * 0.5 + drops * 0.5
            for _ in range(max(1, int(duration_sec / 12.0))):
                if n_samples > sr * 8:
                    start = np.random.randint(0, n_samples - sr * 8)
                    boom_len = np.random.randint(sr * 3, sr * 6)
                    t_boom = np.arange(boom_len) / sr
                    crack_env = np.exp(-t_boom * 8)
                    crack = (np.random.randn(boom_len) * crack_env * 0.4).astype(np.float32)
                    rumble_env = np.exp(-t_boom * 1.5) * (1 - np.exp(-t_boom * 15))
                    rumble_freq = np.random.uniform(30, 60)
                    rumble = ((np.sin(2 * np.pi * rumble_freq * t_boom) + 0.5 * np.sin(2 * np.pi * rumble_freq * 1.5 * t_boom)) * rumble_env * 0.7).astype(np.float32)
                    noise_rumble = np.random.randn(boom_len) * rumble_env * 0.3
                    b_lp, a_lp = scipy_signal.butter(3, 150 / (sr / 2), btype='low')
                    noise_rumble = scipy_signal.lfilter(b_lp, a_lp, noise_rumble).astype(np.float32)
                    boom_fade_in = np.minimum(1.0, np.arange(boom_len) / (sr * 0.2)).astype(np.float32)
                    audio[start:start + boom_len] += (crack + rumble + noise_rumble) * 0.9 * boom_fade_in
        elif sound_type == "Ocean Waves":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [100 / (sr / 2), 3000 / (sr / 2)], btype='band')
            wash = scipy_signal.lfilter(b, a, white).astype(np.float32)
            lfo1 = (0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t_full)).astype(np.float32)
            lfo2 = (0.3 + 0.3 * np.sin(2 * np.pi * 0.07 * t_full + 1.5)).astype(np.float32)
            lfo3 = (0.2 + 0.2 * np.sin(2 * np.pi * 0.2 * t_full + 0.5)).astype(np.float32)
            mod = lfo1 * lfo2 + lfo3
            hiss = np.random.randn(n_samples).astype(np.float32)
            b_h, a_h = scipy_signal.butter(2, 4000 / (sr / 2), btype='high')
            hiss = scipy_signal.lfilter(b_h, a_h, hiss).astype(np.float32) * np.maximum(mod - 0.5, 0) * 0.3
            sub = np.sin(2 * np.pi * 50 * t_full) * mod * 0.1
            audio = wash * mod * 0.7 + hiss + sub
        elif sound_type == "Gentle Stream":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [1500 / (sr / 2), 8000 / (sr / 2)], btype='band')
            bubbles = scipy_signal.lfilter(b, a, white).astype(np.float32)
            lfo = (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t_full) * np.sin(2 * np.pi * 0.13 * t_full)).astype(np.float32)
            bubbles *= lfo
            b_lp, a_lp = scipy_signal.butter(2, 800 / (sr / 2), btype='low')
            murmur = scipy_signal.lfilter(b_lp, a_lp, white).astype(np.float32) * 0.4
            audio = bubbles * 0.6 + murmur
        elif sound_type == "Waterfall":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [800 / (sr / 2), 12000 / (sr / 2)], btype='band')
            hiss = scipy_signal.lfilter(b, a, white).astype(np.float32) * 0.7
            b_lp, a_lp = scipy_signal.butter(2, 500 / (sr / 2), btype='low')
            roar = scipy_signal.lfilter(b_lp, a_lp, white).astype(np.float32) * 0.5
            lfo = (0.85 + 0.15 * np.sin(2 * np.pi * 0.3 * t_full)).astype(np.float32)
            audio = (hiss + roar) * lfo
        elif sound_type == "Wind":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [80 / (sr / 2), 1500 / (sr / 2)], btype='band')
            wind = scipy_signal.lfilter(b, a, white).astype(np.float32)
            lfo1 = (0.4 + 0.6 * np.sin(2 * np.pi * 0.1 * t_full)).astype(np.float32)
            lfo2 = (0.3 + 0.7 * np.sin(2 * np.pi * 0.23 * t_full + 1.0)).astype(np.float32)
            wind *= lfo1 * lfo2
            whistle = np.sin(2 * np.pi * 1200 * t_full + 0.5 * np.sin(2 * np.pi * 3 * t_full))
            whistle *= np.maximum(lfo1 * lfo2 - 0.5, 0) * 0.08
            audio = wind * 0.8 + whistle
        elif sound_type == "Fireplace":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [100 / (sr / 2), 1500 / (sr / 2)], btype='band')
            roar = scipy_signal.lfilter(b, a, white).astype(np.float32) * 0.5
            crackles = np.zeros(n_samples, dtype=np.float32)
            for idx in np.random.randint(0, n_samples, n_samples // 25):
                cl = int(sr * np.random.uniform(0.005, 0.02))
                if idx + cl < n_samples:
                    t_c = np.arange(cl) / sr
                    env = self._transient_env(cl, sr, attack_ms=0.8, decay_rate=float(np.random.uniform(80, 200)))
                    freq = np.random.uniform(3000, 8000)
                    crackles[idx:idx + cl] += np.sin(2 * np.pi * freq * t_c) * env * 0.4
            pops = np.zeros(n_samples, dtype=np.float32)
            for _ in range(max(1, int(duration_sec * 2))):
                idx = np.random.randint(0, n_samples)
                pl = int(sr * 0.05)
                if idx + pl < n_samples:
                    t_p = np.arange(pl) / sr
                    env = self._transient_env(pl, sr, attack_ms=2.0, decay_rate=30.0)
                    freq = np.random.uniform(200, 500)
                    pops[idx:idx + pl] += np.sin(2 * np.pi * freq * t_p) * env * 0.5
            audio = roar + crackles + pops
        elif sound_type == "Forest Night":
            white = np.random.randn(n_samples).astype(np.float32)
            b_lp, a_lp = scipy_signal.butter(2, 400 / (sr / 2), btype='low')
            bed = scipy_signal.lfilter(b_lp, a_lp, white).astype(np.float32) * 0.3
            chirps = np.zeros(n_samples, dtype=np.float32)
            for _ in range(int(duration_sec * 3)):
                start = np.random.randint(0, max(1, n_samples - sr // 2))
                for b_i in range(np.random.randint(3, 8)):
                    b_start = start + b_i * int(sr * 0.05)
                    if b_start + int(sr * 0.02) < n_samples:
                        chirp_len = int(sr * 0.015)
                        t_c = np.arange(chirp_len) / sr
                        freq = np.random.uniform(4500, 5500)
                        fm = 1 + 0.05 * np.sin(2 * np.pi * 80 * t_c)
                        env = self._transient_env(chirp_len, sr, attack_ms=0.5, decay_rate=60.0)
                        chirps[b_start:b_start + chirp_len] += np.sin(2 * np.pi * freq * fm * t_c) * env * 0.25
            owl = np.zeros(n_samples, dtype=np.float32)
            for _ in range(max(1, int(duration_sec / 20))):
                start = np.random.randint(0, max(1, n_samples - sr * 2))
                hoot_len = int(sr * 0.6)
                if start + hoot_len < n_samples:
                    t_h = np.arange(hoot_len) / sr
                    env = np.exp(-((t_h - 0.3) / 0.2) ** 2)
                    env[:max(1, int(sr*0.01))] *= np.linspace(0, 1, max(1, int(sr*0.01)))
                    freq = np.random.uniform(350, 500)
                    freq_mod = freq * (1 - 0.1 * t_h / 0.6)
                    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
                    owl[start:start + hoot_len] += np.sin(phase) * env * 0.3
            audio = bed + chirps + owl
        elif sound_type == "Birdsong Morning":
            white = np.random.randn(n_samples).astype(np.float32)
            b_lp, a_lp = scipy_signal.butter(2, 600 / (sr / 2), btype='low')
            bed = scipy_signal.lfilter(b_lp, a_lp, white).astype(np.float32) * 0.2
            songs = np.zeros(n_samples, dtype=np.float32)
            for _ in range(int(duration_sec * 1.5)):
                start = np.random.randint(0, max(1, n_samples - sr))
                song_len = int(sr * np.random.uniform(0.1, 0.4))
                if start + song_len < n_samples:
                    t_s = np.arange(song_len) / sr
                    env = np.exp(-((t_s - song_len / sr / 2) / (song_len / sr / 2.5)) ** 2)
                    a = max(1, int(sr * 0.005))
                    env[:a] *= np.linspace(0, 1, a)
                    base_freq = np.random.uniform(2000, 5000)
                    fm_rate = np.random.uniform(10, 30)
                    fm_depth = np.random.uniform(200, 800)
                    freq = base_freq + fm_depth * np.sin(2 * np.pi * fm_rate * t_s)
                    phase = 2 * np.pi * np.cumsum(freq) / sr
                    songs[start:start + song_len] += np.sin(phase) * env * 0.3
            audio = bed + songs
        elif sound_type == "Café Ambience":
            white = np.random.randn(n_samples).astype(np.float32)
            b_lp, a_lp = scipy_signal.butter(2, 1000 / (sr / 2), btype='low')
            bed = scipy_signal.lfilter(b_lp, a_lp, white).astype(np.float32) * 0.35
            chatter = np.zeros(n_samples, dtype=np.float32)
            for _ in range(int(duration_sec * 1.5)):
                start = np.random.randint(0, max(1, n_samples - sr * 2))
                ch_len = int(sr * np.random.uniform(0.5, 2.0))
                if start + ch_len < n_samples:
                    t_ch = np.arange(ch_len) / sr
                    env = np.exp(-((t_ch - ch_len / sr / 2) / (ch_len / sr / 2)) ** 2)
                    am = 0.5 + 0.5 * np.sin(2 * np.pi * np.random.uniform(3, 6) * t_ch)
                    noise_burst = np.random.randn(ch_len)
                    b_bp, a_bp = scipy_signal.butter(2, [300 / (sr / 2), 3000 / (sr / 2)], btype='band')
                    filtered = scipy_signal.lfilter(b_bp, a_bp, noise_burst).astype(np.float32)
                    burst_fade = np.ones(ch_len, dtype=np.float32)
                    fe = max(1, int(sr * 0.01))
                    burst_fade[:fe]  = np.linspace(0, 1, fe)
                    burst_fade[-fe:] = np.linspace(1, 0, fe)
                    chatter[start:start + ch_len] += filtered * env * am * burst_fade * 0.3
            clinks = np.zeros(n_samples, dtype=np.float32)
            for _ in range(int(duration_sec * 0.3)):
                start = np.random.randint(0, max(1, n_samples - sr // 4))
                cl = int(sr * 0.08)
                if start + cl < n_samples:
                    t_c = np.arange(cl) / sr
                    env = self._transient_env(cl, sr, attack_ms=1.0, decay_rate=50.0)
                    f1 = np.random.uniform(2500, 4000)
                    clinks[start:start + cl] += (np.sin(2 * np.pi * f1 * t_c) + 0.5 * np.sin(2 * np.pi * f1 * 1.8 * t_c)) * env * 0.25
            audio = bed + chatter + clinks
        elif sound_type == "Keyboard Typing":
            clicks = np.zeros(n_samples, dtype=np.float32)
            pos = 0
            while pos < n_samples:
                burst_len = int(sr * np.random.uniform(0.3, 2.5))
                n_keys = int(burst_len / sr * np.random.uniform(4, 8))
                for _ in range(n_keys):
                    kpos = pos + np.random.randint(0, burst_len)
                    if kpos < n_samples:
                        klen = int(sr * 0.015)
                        if kpos + klen < n_samples:
                            t_k = np.arange(klen) / sr
                            env = self._transient_env(klen, sr, attack_ms=0.5, decay_rate=150.0)
                            body_freq = np.random.uniform(120, 250)
                            click_freq = np.random.uniform(2000, 4000)
                            body = np.sin(2 * np.pi * body_freq * t_k) * 0.4
                            click = np.sin(2 * np.pi * click_freq * t_k) * 0.3
                            clicks[kpos:kpos + klen] += (body + click) * env * 0.5
                pos += burst_len + int(sr * np.random.uniform(0.3, 1.5))
            audio = clicks
        elif sound_type == "Clock Ticking":
            ticks = np.zeros(n_samples, dtype=np.float32)
            tick_count = 0
            for pos in range(0, n_samples, sr):
                tlen = int(sr * 0.03)
                if pos + tlen < n_samples:
                    t_t = np.arange(tlen) / sr
                    env = self._transient_env(tlen, sr, attack_ms=0.5, decay_rate=200.0)
                    freq = 2500 if tick_count % 2 == 0 else 2200
                    body = np.sin(2 * np.pi * 400 * t_t) * 0.3
                    tick = np.sin(2 * np.pi * freq * t_t) * 0.5
                    ticks[pos:pos + tlen] += (tick + body) * env * 0.4
                tick_count += 1
            audio = ticks
        elif sound_type == "Distant Traffic":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [50 / (sr / 2), 800 / (sr / 2)], btype='band')
            bed = scipy_signal.lfilter(b, a, white).astype(np.float32) * 0.5
            b_h, a_h = scipy_signal.butter(4, [1000 / (sr / 2), 5000 / (sr / 2)], btype='band')
            hiss = scipy_signal.lfilter(b_h, a_h, white).astype(np.float32) * 0.2
            pass_by = 0.5 + 0.5 * np.sin(2 * np.pi * 0.08 * t_full)
            audio = (bed + hiss * pass_by) * 0.7
        elif sound_type == "Heartbeat":
            beats = np.zeros(n_samples, dtype=np.float32)
            bpm = 70
            beat_interval = 60.0 / bpm * sr
            for beat_start in np.arange(0, n_samples, beat_interval):
                tlen = int(sr * 0.12)
                if int(beat_start) + tlen < n_samples:
                    t_b = np.arange(tlen) / sr
                    env = self._transient_env(tlen, sr, attack_ms=3.0, decay_rate=25.0)
                    beats[int(beat_start):int(beat_start) + tlen] += np.sin(2 * np.pi * 60 * t_b) * env * 0.9
                dub_start = int(beat_start) + int(sr * 0.18)
                if dub_start + tlen < n_samples:
                    t_b = np.arange(tlen) / sr
                    env = self._transient_env(tlen, sr, attack_ms=3.0, decay_rate=30.0)
                    beats[dub_start:dub_start + tlen] += np.sin(2 * np.pi * 55 * t_b) * env * 0.6
            audio = beats
        elif sound_type == "Space Drone":
            drone = np.zeros(n_samples, dtype=np.float32)
            freqs = [55, 82.5, 110, 165, 220]
            for i, f in enumerate(freqs):
                amp = 1.0 / (i + 1)
                lfo = np.sin(2 * np.pi * (0.05 + i * 0.02) * t_full)
                freq_mod = f * (1 + 0.005 * lfo)
                phase = 2 * np.pi * np.cumsum(freq_mod) / sr
                drone += np.sin(phase) * amp
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [200 / (sr / 2), 2000 / (sr / 2)], btype='band')
            wash = scipy_signal.lfilter(b, a, white).astype(np.float32) * 0.1
            am = 0.7 + 0.3 * np.sin(2 * np.pi * 0.03 * t_full)
            audio = (drone + wash) * am * 0.5
        else:
            return np.zeros(n_samples, dtype=np.float32)

        audio = self._smooth_sfx_bus(audio.astype(np.float32), sr)
        audio = self._soft_clip(audio, drive=1.2)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-9: audio = audio * (0.85 / peak)
        fade_samples = int(sr * 1.0)
        if len(audio) > fade_samples * 2:
            t_in  = np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)
            t_out = np.linspace(np.pi / 2, 0, fade_samples, dtype=np.float32)
            audio[:fade_samples] *= np.sin(t_in)
            audio[-fade_samples:] *= np.sin(t_out)
        return self._remove_dc(audio)

    def process_audio_chain(self, audio_np, fs, is_live=False):
        if hasattr(self, '_render_audio_settings') and self._render_audio_settings is not None:
            s = self._render_audio_settings
        else:
            s = {
                'gain': self.gain_slider.value() / 100.0,
                'eq_low': self.eq_low_slider.value(), 'eq_mid': self.eq_mid_slider.value(), 'eq_high': self.eq_high_slider.value(),
                'eq_low_q': self.eq_low_q_slider.value() / 100.0, 'eq_mid_q': self.eq_mid_q_slider.value() / 100.0, 'eq_high_q': self.eq_high_q_slider.value() / 100.0,
                'gate_enabled': self.gate_check.isChecked(),
                'gate_thresh': self.gate_slider.value(), 'gate_attack': self.gate_attack_slider.value(), 'gate_release': self.gate_release_slider.value(), 'gate_ratio': self.gate_ratio_slider.value() / 10.0,
                'hp_freq': self.hp_freq_slider.value(),
                'yt_loud': self.yt_loud_check.isChecked(),
                'ambient_type': self.ambient_combo.currentText(), 'ambient_vol_db': self.ambient_vol_slider.value(), 'sfx_master_db': self.sfx_master_slider.value(),
                'out_vol': self.out_vol_slider.value() / 100.0,
                'noise_suppress': self.noise_suppress_check.isChecked() if not is_live else False,
                'noise_suppress_amount': self.ns_amount_slider.value(),
                'noise_suppress_mode': self.ns_mode_combo.currentText(),
                'duck_sfx': self.duck_check.isChecked(),
                'deess_thresh': self.deess_slider.value(), 'deess_ratio': self.deess_ratio_slider.value() / 10.0,
                'air_boost': self.air_slider.value() / 10.0,
                'tape_drive': self.tape_slider.value(),
                'transient_punch': self.transient_slider.value()
            }

        gained_audio = apply_gain_numba(audio_np, s['gain'])
        if s.get('noise_suppress', False) and len(gained_audio) > fs * 0.1:
            gained_audio = self.obs_noise_suppression(
                gained_audio, fs, 
                suppression_db=s['noise_suppress_amount'],
                mode=s.get('noise_suppress_mode', 'Speex')
            )

        # EQ and filtering chain (no compressor)
        plugins = [
            HighpassFilter(cutoff_frequency_hz=s['hp_freq']),
            LowShelfFilter(cutoff_frequency_hz=150, gain_db=s['eq_low']),
            PeakFilter(cutoff_frequency_hz=1200, gain_db=s['eq_mid'], q=s['eq_mid_q']),
            HighShelfFilter(cutoff_frequency_hz=4000, gain_db=s['eq_high'])
        ]
        if s['gate_enabled']:
            plugins.append(NoiseGate(threshold_db=s['gate_thresh'], attack_ms=s['gate_attack'], release_ms=s['gate_release'], ratio=s['gate_ratio']))
        
        board = Pedalboard(plugins)
        processed_voice = board(gained_audio, fs)
        
        # Apply transient shaper and tape saturation (no compressor)
        processed_voice = self.apply_transient_shaper(processed_voice, fs, s.get('transient_punch', 0))
        processed_voice = self.apply_tape_saturation(processed_voice, s.get('tape_drive', 0))
        processed_voice = self.apply_deesser(processed_voice, fs, threshold_db=s['deess_thresh'], ratio=s['deess_ratio'])
        
        if not is_live and s['ambient_type'] != "None":
            duration = len(processed_voice) / fs
            ambient_mono = self.generate_ambient_sound(duration, fs, s['ambient_type'])
            total_db = s['ambient_vol_db'] + s['sfx_master_db']
            ambient_gain = 0.0 if total_db <= -60 else 10.0 ** (total_db / 20.0)
            
            n_amb = len(ambient_mono)
            t_amb = np.arange(n_amb) / fs
            pan_lfo = (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t_amb)).astype(np.float32)
            ambient_stereo = np.zeros((n_amb, 2), dtype=np.float32)
            ambient_stereo[:, 0] = ambient_mono * np.sqrt(pan_lfo)
            ambient_stereo[:, 1] = ambient_mono * np.sqrt(1.0 - pan_lfo)

            if s.get('duck_sfx', True):
                voice_env = np.abs(processed_voice).astype(np.float32)
                attack_coeff = 1.0 / (fs * 0.02)
                release_coeff = 1.0 / (fs * 0.2)
                duck_env = envelope_follower_numba(voice_env, attack_coeff, release_coeff)
                if duck_env.max() > 0: duck_env /= duck_env.max()
                duck_gain = 1.0 - (duck_env * 0.5)
                ambient_stereo *= duck_gain[:, np.newaxis]

            amb_plugins = [HighpassFilter(cutoff_frequency_hz=120), PeakFilter(cutoff_frequency_hz=1000, gain_db=-3.0, q=1.5), HighShelfFilter(cutoff_frequency_hz=3000, gain_db=2.0)]
            ambient_stereo = Pedalboard(amb_plugins)(ambient_stereo, fs)
            ambient_stereo *= ambient_gain
            voice_stereo = np.column_stack((processed_voice, processed_voice))
            min_len = min(len(voice_stereo), len(ambient_stereo))
            processed_voice = (voice_stereo[:min_len] + ambient_stereo[:min_len]).astype(np.float32)
            
        vol_db = 20 * math.log10(max(0.001, s['out_vol']))
        
        if s['air_boost'] > 0.1:
            voice_env = np.abs(processed_voice if processed_voice.ndim == 1 else processed_voice[:, 0]).astype(np.float32)
            env_smooth = scipy_signal.lfilter(*scipy_signal.butter(2, 5.0 / (fs / 2.0)), voice_env)
            if env_smooth.max() > 0: env_smooth /= env_smooth.max()
            env_smooth = np.power(env_smooth, 2)
            b_air, a_air = scipy_signal.butter(4, 8000.0 / (fs / 2.0), btype='high')
            if processed_voice.ndim == 1:
                air_band = scipy_signal.lfilter(b_air, a_air, processed_voice)
                processed_voice = processed_voice + (air_band * (10**(s['air_boost']/20.0) - 1.0) * env_smooth)
            else:
                air_l = scipy_signal.lfilter(b_air, a_air, processed_voice[:, 0])
                air_r = scipy_signal.lfilter(b_air, a_air, processed_voice[:, 1])
                processed_voice[:, 0] += (air_l * (10**(s['air_boost']/20.0) - 1.0) * env_smooth)
                processed_voice[:, 1] += (air_r * (10**(s['air_boost']/20.0) - 1.0) * env_smooth)

        # Final output stage: gain + limiter (no compressor)
        final_plugins = [
            Gain(gain_db=vol_db),
            HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-1.0),
            Limiter(threshold_db=-1.0, release_ms=100)
        ]
        return Pedalboard(final_plugins)(processed_voice, fs)

    def toggle_monitor(self):
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_stream:
                self.monitor_stream.stop(); self.monitor_stream.close()
            self.monitor_btn.setText("🎧  Start Live Monitor  (Full Chain)")
            self.status_bar.showMessage("Monitoring stopped.")
        else:
            if QMessageBox.warning(self, "Monitor Warning", "Ensure you are wearing headphones to prevent feedback. Continue?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes: return
            self.is_monitoring = True
            self.monitor_btn.setText("⏹  Stop Live Monitor")
            device_name = self.device_combo.currentText()
            device_idx = sd.query_devices(device_name, 'input')['index'] if device_name else None
            fs = self.audio_samplerate
            def monitor_callback(indata, outdata, frames, time, status):
                processed = self.process_audio_chain(indata[:, 0], fs, is_live=True)
                outdata[:, 0] = processed
            self.monitor_stream = sd.Stream(callback=monitor_callback, device=(device_idx, None), channels=1, samplerate=fs, blocksize=1024, dtype='float32', latency='low')
            self.monitor_stream.start()
            self.status_bar.showMessage("Live monitor active.")

    def apply_shorts_mode(self, frame):
        h, w = frame.shape[:2]; target_w, target_h = 720, 1280
        scale = target_w / w; new_w = int(w * scale); new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        bg = cv2.resize(frame, (target_w, target_h)); bg = cv2.GaussianBlur(bg, (51, 51), 0)
        y_offset = (target_h - new_h) // 2; bg[y_offset:y_offset+new_h, :] = resized
        return bg

    def draw_grid(self, frame):
        h, w = frame.shape[:2]; color = (255, 255, 255); thickness = 1
        cv2.line(frame, (w//3, 0), (w//3, h), color, thickness)
        cv2.line(frame, (2*w//3, 0), (2*w//3, h), color, thickness)
        cv2.line(frame, (0, h//3), (w, h//3), color, thickness)
        cv2.line(frame, (0, 2*h//3), (w, 2*h//3), color, thickness)
        return frame

    FILTER_PARAMS = {
        "Canny Edge":       ["canny_low", "canny_high", "invert"],
        "Ghost Edges":      ["canny_low", "canny_high", "ghost_decay"],
        "Laplacian Edge":   ["blur_k", "invert"],
        "Sobel Edge":       ["blur_k", "invert"],
        "XDog Sketch":      ["blur_k", "density", "sketch_scale", "invert"],
        "Adaptive Lines":   ["blur_k", "density", "invert"],
        "Morph Gradient":   ["blur_k", "density", "invert"],
        "Sketch Pencil":    ["blur_k", "sketch_scale", "invert"],
        "Edge Glow Neon":   ["canny_low", "canny_high", "glow"],
        "Wireframe Mesh":   ["mesh_pts"],
        "ASCII Art":        ["ascii_block", "invert"],
        "Thermal Heatmap":  ["blur_k", "invert"],
        "Heavy Pixelate":   ["pix_block"],
        "Dotted Stipple":   ["stipple", "invert"],
        "Threshold Binary": ["bin_thresh", "invert"],
        "Static Glitch":    ["canny_low", "canny_high", "noise", "glitch_bands", "invert"],
        "Film Grain Edges": ["canny_low", "canny_high", "noise", "glitch_bands", "invert"],
        "RGB Shift":        ["canny_low", "canny_high", "rgb_shift", "noise", "glitch_bands"],
        "VHS Retro":        ["rgb_shift", "noise", "glitch_bands"],
        "Posterize":        ["density", "invert"],
        "Contour Fill":     ["canny_low", "density", "invert"],
        "Digital Halftone": ["stipple", "invert"]
    }

    def on_filter_changed(self, *args):
        filt = self.filter_combo.currentText()
        active_params = set(self.FILTER_PARAMS.get(filt, []))
        active_groups = set()
        for key, (sld, scale, default) in self.param_sliders.items():
            is_active = key in active_params
            sld.setVisible(is_active)
            parent_widget = sld.parentWidget()
            while parent_widget and not isinstance(parent_widget, QGroupBox):
                parent_widget = parent_widget.parentWidget()
            if parent_widget:
                if is_active:
                    parent_widget.show(); active_groups.add(parent_widget.title().strip())
                else:
                    row_widget = sld.parentWidget()
                    while row_widget and not isinstance(row_widget, QGroupBox):
                        if row_widget.parentWidget() and isinstance(row_widget.parentWidget(), QGroupBox): break
                        row_widget = row_widget.parentWidget()
                    if row_widget: row_widget.hide()
        for grp_name, grp_box in self.param_groups.items():
            if grp_name in active_groups: grp_box.show()
            else: grp_box.hide()
        if hasattr(self, 'filter_combo'): self.video_filter_label.setText(f"🎨 Filter: {filt}")

    def _p(self, key):
        if hasattr(self, '_render_settings') and self._render_settings is not None:
            return self._render_settings['params'].get(key, 0)
        if key in self.param_sliders:
            sld, scale, default = self.param_sliders[key]
            return sld.value() / scale
        return 0

    def on_param_changed(self, key, val): pass

    def _build_lut(self, stops):
        stops = sorted(stops, key=lambda s: s[0])
        lut = np.zeros((256, 3), dtype=np.uint8)
        if not stops:
            for i in range(256):
                lut[i] = [i, i, i]
            return lut
        for i in range(256):
            t = i / 255.0
            if t <= stops[0][0]:
                lut[i] = stops[0][1]
                continue
            if t >= stops[-1][0]:
                lut[i] = stops[-1][1]
                continue
            for j in range(len(stops) - 1):
                p0, c0 = stops[j]
                p1, c1 = stops[j + 1]
                if p0 <= t <= p1:
                    ratio = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
                    lut[i] = [int(c0[k] + (c1[k] - c0[k]) * ratio) for k in range(3)]
                    break
        return lut

    def get_palette_stops_dict(self):
        return self.get_palette_stops(None)

    def apply_palette(self, gray_img, palette_name):
        gray_img = gray_img.astype(np.uint8)
        if palette_name == "White":
            return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        all_palettes = self.get_palette_stops_dict()
        if palette_name not in all_palettes:
            return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        lut_rgb = self._build_lut(all_palettes[palette_name])
        lut_bgr = np.ascontiguousarray(lut_rgb[:, ::-1])  # RGB -> BGR for OpenCV
        return lut_bgr[gray_img]

    def apply_visual_effect(self, frame):
        if hasattr(self, '_render_settings') and self._render_settings is not None:
            filt = self._render_settings['filter']; palette = self._render_settings['palette']
        else:
            filt = self.filter_combo.currentText(); palette = self.palette_combo.currentText()
        if filt != "Ghost Edges": self.edge_accumulator = None
        invert = self._p("invert") > 0.5
        
        def odd_k(v):
            v = max(1, int(v)); return v if v % 2 == 1 else v + 1

        if filt == "Canny Edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p("canny_low")), int(self._p("canny_high")))
            if invert: edges = 255 - edges
            frame = self.apply_palette(edges, palette)
        elif filt == "Ghost Edges":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p("canny_low")), int(self._p("canny_high")))
            if self.edge_accumulator is None or self.edge_accumulator.shape != edges.shape:
                self.edge_accumulator = np.zeros_like(edges, dtype=np.float32)
            self.edge_accumulator *= self._p("ghost_decay")
            self.edge_accumulator += edges.astype(np.float32)
            np.clip(self.edge_accumulator, 0, 255, out=self.edge_accumulator)
            frame = self.apply_palette(self.edge_accumulator.astype(np.uint8), palette)
        elif filt == "Laplacian Edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p("blur_k")); blurred = cv2.GaussianBlur(gray, (k, k), 0)
            lap = cv2.Laplacian(blurred, cv2.CV_64F); lap = np.absolute(lap)
            lap = (lap / lap.max() * 255).astype(np.uint8) if lap.max() > 0 else lap.astype(np.uint8)
            if invert: lap = 255 - lap
            frame = self.apply_palette(lap, palette)
        elif filt == "Sobel Edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p("blur_k")); blurred = cv2.GaussianBlur(gray, (k, k), 0)
            sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3); sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sx**2 + sy**2)
            mag = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
            if invert: mag = 255 - mag
            frame = self.apply_palette(mag, palette)
        elif filt == "XDog Sketch":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p("blur_k")); sigma1 = k / 2.0 if k > 1 else 0.5
            g1 = cv2.GaussianBlur(gray, (k, k), sigma1); g2 = cv2.GaussianBlur(gray, (k, k), sigma1 * 1.6)
            dog = g1 - g2; dog = dog / 255.0
            dog = np.where(dog < 0.01, 1.0, 1.0 + np.tanh(10.0 * (dog - 0.01)))
            dog = (dog * 255).clip(0, 255).astype(np.uint8); dog = 255 - dog 
            if invert: dog = 255 - dog
            frame = self.apply_palette(dog, palette)
        elif filt == "Adaptive Lines":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p("blur_k")); gray = cv2.GaussianBlur(gray, (k, k), 0)
            block_size = max(3, int(self._p("density")) * 2 + 1)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 5)
            if invert: edges = 255 - edges
            frame = self.apply_palette(edges, palette)
        elif filt == "Morph Gradient":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = max(1, int(self._p("density"))); kernel = np.ones((k, k), np.uint8)
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            if invert: grad = 255 - grad
            frame = self.apply_palette(grad, palette)
        elif filt == "Static Glitch":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p("canny_low")), int(self._p("canny_high")))
            noise_amt = self._p("noise") / 100.0
            noise = np.random.randint(0, 256, gray.shape, dtype=np.uint8)
            _, noise = cv2.threshold(noise, 200, 255, cv2.THRESH_BINARY)
            combined = cv2.addWeighted(edges, 1.0, noise, noise_amt * 2.0, 0)
            bands = int(self._p("glitch_bands"))
            if bands > 0:
                h, w = combined.shape
                for _ in range(bands):
                    y = np.random.randint(0, h-10); h_slice = np.random.randint(2, 10); shift_x = np.random.randint(-20, 20)
                    combined[y:y+h_slice] = np.roll(combined[y:y+h_slice], shift_x, axis=1)
            if invert: combined = 255 - combined
            frame = self.apply_palette(combined, palette)
        elif filt == "Film Grain Edges":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p("canny_low")), int(self._p("canny_high")))
            noise_amt = self._p("noise") / 100.0
            grain = np.random.normal(0, 25, gray.shape); grain = np.clip(grain, 0, 255).astype(np.uint8)
            combined = cv2.addWeighted(edges, 1.0, grain, noise_amt, 0)
            bands = int(self._p("glitch_bands"))
            if bands > 0:
                h, w = combined.shape
                for _ in range(bands):
                    y = np.random.randint(0, h-10); h_slice = np.random.randint(2, 10); shift_x = np.random.randint(-20,20)
                    combined[y:y+h_slice] = np.roll(combined[y:y+h_slice], shift_x, axis=1)
            if invert: combined = 255 - combined
            frame = self.apply_palette(combined, palette)
        elif filt == "RGB Shift":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p("canny_low")), int(self._p("canny_high")))
            shift = int(self._p("rgb_shift"))
            r = np.roll(edges, shift, axis=1); g = edges; b = np.roll(edges, -shift, axis=1)
            shifted = cv2.merge([b, g, r])
            noise_amt = self._p("noise") / 100.0
            if noise_amt > 0:
                glitch_noise = np.random.randint(0, 256, shifted.shape, dtype=np.uint8)
                shifted = cv2.addWeighted(shifted, 1.0, glitch_noise, noise_amt * 0.5, 0)
            frame = shifted
        elif filt == "VHS Retro":
            shift = int(self._p("rgb_shift"))
            b, g, r = cv2.split(frame)
            r = np.roll(r, shift, axis=1); b = np.roll(b, -shift, axis=1)
            frame = cv2.merge([b, g, r])
            bands = int(self._p("glitch_bands"))
            if bands > 0:
                h, w = frame.shape[:2]
                for _ in range(bands):
                    y = np.random.randint(0, h-10); h_slice = np.random.randint(2, 10); shift_x = np.random.randint(-30, 30)
                    frame[y:y+h_slice] = np.roll(frame[y:y+h_slice], shift_x, axis=1)
            for y in range(0, frame.shape[0], 2): frame[y] = (frame[y] * 0.7).astype(np.uint8)
            noise_amt = self._p("noise") / 100.0
            if noise_amt > 0:
                noise = np.random.normal(0, 15, frame.shape)
                frame = cv2.addWeighted(frame, 1.0, noise.astype(np.uint8), noise_amt, 0)
        elif filt == "Posterize":
            levels = max(2, int(self._p("density"))); factor = 256 // levels
            posterized = (frame // factor) * factor
            if palette != "White":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); colored = self.apply_palette(gray, palette)
                factor_c = 256 // levels; posterized = (colored // factor_c) * factor_c
            if invert: posterized = 255 - posterized
            frame = posterized
        elif filt == "Contour Fill":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p("canny_low")), int(self._p("canny_low")*2))
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(frame); colored_palette = self.apply_palette(gray, palette)
            for i, cnt in enumerate(contours):
                color = colored_palette[gray.shape[0]//2, i % gray.shape[1]].tolist()
                cv2.drawContours(mask, [cnt], -1, color, int(self._p("density")))
            if invert: mask = 255 - mask
            frame = mask
        elif filt == "Digital Halftone":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]; block = max(4, int(self._p("stipple")))
            canvas = np.zeros_like(frame); colored = self.apply_palette(gray, palette)
            for y in range(0, h, block):
                for x in range(0, w, block):
                    val = gray[y, x]; radius = int((255 - val) / 255 * (block // 2))
                    if radius > 0: cv2.circle(canvas, (x + block//2, y + block//2), radius, colored[y, x].tolist(), -1)
            if invert: canvas = 255 - canvas
            frame = canvas
        elif filt == "Sketch Pencil":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p("blur_k")); inv = 255 - gray; blur_inv = cv2.GaussianBlur(inv, (k, k), 0)
            sketch = cv2.divide(gray, blur_inv, scale=256.0)
            sketch = np.clip(sketch * self._p("sketch_scale"), 0, 255).astype(np.uint8)
            if invert: sketch = 255 - sketch
            frame = self.apply_palette(sketch, palette)
        elif filt == "Edge Glow Neon":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p("canny_low")), int(self._p("canny_high")))
            colored = self.apply_palette(edges, palette)
            glow_pct = self._p("glow"); blur_size = odd_k(int(3 + glow_pct * 0.5))
            glow1 = cv2.GaussianBlur(colored, (blur_size, blur_size), 0)
            glow2 = cv2.GaussianBlur(colored, (blur_size*2+1, blur_size*2+1), 0)
            glow = cv2.addWeighted(glow1, 0.6, glow2, 0.4, 0)
            frame = cv2.addWeighted(glow, 0.8, colored, 1.2, 0)
        elif filt == "Wireframe Mesh":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]; n_pts = int(self._p("mesh_pts"))
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=n_pts, qualityLevel=0.02, minDistance=10, blockSize=7)
            mask = np.zeros_like(frame)
            if corners is not None and len(corners) > 3:
                pts = corners.reshape(-1, 2).astype(np.int32); rect = (0, 0, w, h); subdiv = cv2.Subdiv2D(rect)
                for p in pts: subdiv.insert((int(p[0]), int(p[1])))
                triangles = subdiv.getTriangleList()
                palette_colors = {
                    "Matrix Green": (0, 255, 0), "Neon Cyan": (255, 255, 0), "Acid Pink": (255, 0, 255), "Ice Blue": (255, 100, 0),
                    "Cyberpunk": (255, 0, 200), "Sunset": (0, 140, 255), "Fire": (0, 140, 255), "Vaporwave": (255, 60, 180),
                    "Toxic": (30, 255, 80), "Viridis": (37, 201, 94), "Plasma": (33, 249, 240), "Rainbow Flow": (255, 200, 0),
                    "Synthwave": (80, 50, 220), "Aurora": (130, 230, 0), "Neon Tokyo": (255, 0, 150), "Cotton Candy": (255, 100, 200),
                    "Lava Reef": (255, 120, 20), "Deep Space": (200, 50, 200), "Turbo": (250, 147, 58), "Cividis": (255, 233, 69)
                }
                color = palette_colors.get(palette, (255, 255, 255))
                for t in triangles:
                    p1 = (int(t[0]), int(t[1])); p2 = (int(t[2]), int(t[3])); p3 = (int(t[4]), int(t[5]))
                    cv2.line(mask, p1, p2, color, 1); cv2.line(mask, p2, p3, color, 1); cv2.line(mask, p3, p1, color, 1)
            frame = mask
        elif filt == "ASCII Art":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            block = max(4, int(self._p("ascii_block"))); h, w = gray.shape[:2]
            chars = " .:-=+*#%@"; small = cv2.resize(gray, (w // block, h // block))
            canvas = np.zeros_like(frame); colored = self.apply_palette(gray, palette)
            fh, fw = canvas.shape[:2]; cell_h = fh / small.shape[0]; cell_w = fw / small.shape[1]
            for r in range(small.shape[0]):
                for c in range(small.shape[1]):
                    val = small[r, c]; ch = chars[min(len(chars) - 1, val * len(chars) // 256)]
                    if ch != ' ':
                        x, y = int(c * cell_w), int(r * cell_h)
                        cv2.putText(canvas, ch, (x, y + int(cell_h * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colored[min(r*block, h-1), min(c*block, w-1)].tolist(), 1)
            if invert: canvas = 255 - canvas
            frame = canvas
        elif filt == "Thermal Heatmap":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p("blur_k")); blurred = cv2.GaussianBlur(gray, (k, k), 0)
            if invert: blurred = 255 - blurred
            frame = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        elif filt == "Heavy Pixelate":
            block = max(2, int(self._p("pix_block"))); h, w = frame.shape[:2]
            small = cv2.resize(frame, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
            frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif filt == "Dotted Stipple":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]; density = self._p("stipple") / 100.0
            canvas = np.zeros_like(frame); n_dots = int(h * w * density * 0.05)
            ys = np.random.randint(0, h, n_dots); xs = np.random.randint(0, w, n_dots); vals = gray[ys, xs]
            colored = self.apply_palette(gray, palette)
            for i in range(n_dots):
                if vals[i] > 60: cv2.circle(canvas, (xs[i], ys[i]), 1, colored[ys[i], xs[i]].tolist(), -1)
            if invert: canvas = 255 - canvas
            frame = canvas
        elif filt == "Threshold Binary":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = int(self._p("bin_thresh")); _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            if invert: bw = 255 - bw
            frame = self.apply_palette(bw, palette)
        return frame

    def apply_video_adjustments(self, frame):
        if hasattr(self, '_render_settings') and self._render_settings is not None:
            mirror = self._render_settings['mirror']
            contrast = self._render_settings['contrast']
            brightness = (self._render_settings['brightness'] - 1.0) * 100
            shorts = self._render_settings['shorts']
        else:
            mirror = self.mirror_check.isChecked()
            contrast = self.contrast_slider.value() / 100.0
            brightness = (self.brightness_slider.value() / 100.0 - 1.0) * 100
            shorts = self.shorts_check.isChecked()
        if mirror:
            frame = cv2.flip(frame, 1)
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        fx_frame = self.apply_visual_effect(adjusted.copy())
        if shorts:
            fx_frame = self.apply_shorts_mode(fx_frame)
        return fx_frame

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            display_frame = self.apply_video_adjustments(frame.copy())
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
            if self.grid_check.isChecked(): display_frame = self.draw_grid(display_frame)
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape; bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
            if self.is_recording and self.video_writer is not None: self.video_writer.write(frame) 
        self.video_timer.start(15)

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            if self.is_muted: self.audio_data.append(np.zeros_like(indata))
            else: self.audio_data.append(indata.copy())
        volume_norm = np.linalg.norm(indata) * 10
        self.signals.audio_level.emit(min(volume_norm, 100))
        peak = np.max(np.abs(indata)) * 100
        if peak > 1: self.signals.audio_peak.emit(peak)

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
        if self.is_monitoring: self.toggle_monitor()
        self.record_btn.setEnabled(False)
        self.markers.clear(); self.edge_accumulator = None
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
        device_name = self.device_combo.currentText()
        device_idx = sd.query_devices(device_name, 'input')['index'] if device_name else None
        self.audio_stream = sd.InputStream(callback=self.audio_callback, device=device_idx, channels=1, samplerate=self.audio_samplerate)
        self.audio_stream.start()
        self.is_recording = True; self.is_muted = False
        self.record_btn.setText("⏹ Stop Recording (Ctrl+R)"); self.record_btn.setEnabled(True)
        self.process_btn.setEnabled(False); self.status_bar.showMessage("🔴 Recording Raw Video & Audio...")
        self.rec_start_time = time.time(); self.peak_hold_value = 0
        self.tele_text.setReadOnly(True); self.tele_text.setFocusPolicy(Qt.NoFocus)
        mode = self.tele_mode_combo.currentText()
        if mode == "Bullet Points":
            self.bullets = [line.strip() for line in self.tele_text.toPlainText().split('\n') if line.strip()]
            self.current_bullet_idx = 0
            if self.bullets:
                self.update_bullet_display()
                self.status_bar.showMessage(f"🔴 Recording | ▶ Bullet 1/{len(self.bullets)}")
        elif mode == "Manual Scroll":
            cursor = self.tele_text.textCursor(); cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.tele_text.setTextCursor(cursor); self.tele_text.verticalScrollBar().setValue(0)
        elif mode == "Auto Scroll":
            cursor = self.tele_text.textCursor(); cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.tele_text.setTextCursor(cursor); self.tele_text.verticalScrollBar().setValue(0)
            if self.tele_scroll_slider.value() > 0: self.start_auto_scroll(self.tele_scroll_slider.value())
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
        if self.auto_scroll_timer: self.auto_scroll_timer.stop(); self.auto_scroll_timer = None
        self.tele_text.setReadOnly(False); self.tele_text.setFocusPolicy(Qt.StrongFocus)
        self.tele_text.setHtml(""); self.tele_progress.setValue(0)
        self.tele_counter_label.setText("📊 Bullet: 0 / 0"); self.rec_time_label.setText("⏱ 00:00:00")
        self.audio_stream.stop(); self.audio_stream.close(); self.video_writer.release()

    def start_processing(self):
        self.record_btn.setEnabled(False); self.process_btn.setEnabled(False)
        self.progress = QProgressDialog("Rendering final video...", "Cancel", 0, 100, self)
        self.progress.setWindowTitle("Processing"); self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0); self.progress.setValue(0)
        threading.Thread(target=self.post_process, daemon=True).start()

    def post_process(self):
        try:
            self.signals.progress.emit(10); self.video_timer.stop(); self._thread_error = None

            def _audio_worker():
                try:
                    self._render_audio_settings = {
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
                    if len(self.audio_data) > 0:
                        audio_np = np.concatenate(self.audio_data, axis=0).flatten().astype(np.float32)
                        processed_audio = self.process_audio_chain(audio_np, self.audio_samplerate)
                        wav.write(self.processed_audio_path, self.audio_samplerate, (processed_audio * 32767).astype(np.int16))
                except Exception as e: self._thread_error = e
                finally: self._render_audio_settings = None

            def _video_worker():
                try:
                    self._render_settings = {
                        'mirror': self.mirror_check.isChecked(), 'contrast': self.contrast_slider.value() / 100.0,
                        'brightness': self.brightness_slider.value() / 100.0, 'shorts': self.shorts_check.isChecked(),
                        'filter': self.filter_combo.currentText(), 'palette': self.palette_combo.currentText(),
                        'params': { k: (sld.value() / scale) for k, (sld, scale, default) in self.param_sliders.items() }
                    }
                    cap = cv2.VideoCapture(self.raw_video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out = cv2.VideoWriter(self.processed_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); current_frame = 0; self.edge_accumulator = None
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        proc_frame = self.apply_video_adjustments(frame); out.write(proc_frame)
                        current_frame += 1
                        if total_frames > 0:
                            progress = 10 + int((current_frame / total_frames) * 70)
                            self.signals.progress.emit(progress)
                    cap.release(); out.release()
                except Exception as e: self._thread_error = e
                finally: self._render_settings = None

            audio_thread = threading.Thread(target=_audio_worker, daemon=True)
            video_thread = threading.Thread(target=_video_worker, daemon=True)
            audio_thread.start(); video_thread.start()
            audio_thread.join(); video_thread.join()
            if self._thread_error: raise self._thread_error

            self.signals.progress.emit(85)
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
            
            video_clip = VideoFileClip(self.processed_video_path)
            audio_clip = AudioFileClip(self.processed_audio_path)
            final_clip = video_clip.with_audio(audio_clip)
            final_clip.write_videofile(
                save_path, codec=ffmpeg_codec, audio_codec="aac", 
                temp_audiofile="temp_audio.m4a", remove_temp=True, logger=None,
                ffmpeg_params=['-preset', self.render_combo.currentText()]
            )
            
            if self.markers:
                marker_path = os.path.join(self.output_dir_edit.text(), filename.replace('.mp4', '_markers.txt'))
                with open(marker_path, 'w') as f:
                    f.write("--- Highlight Markers ---\n")
                    for m_time in self.markers:
                        m, s = divmod(int(m_time), 60); h, m = divmod(m, 60)
                        f.write(f"[{h:02d}:{m:02d}:{s:02d}] Highlight\n")
            
            self.signals.progress.emit(100)
            self.signals.finished.emit(save_path)

        except Exception as e: self.signals.error.emit(str(e))
        finally:
            for f in [self.raw_video_path, self.raw_audio_path, self.processed_video_path, self.processed_audio_path]:
                if os.path.exists(f): os.remove(f)

    def update_progress(self, val):
        if hasattr(self, 'progress'): self.progress.setValue(val)
    def update_status(self, msg): self.status_bar.showMessage(msg)

    def on_render_finished(self, save_path):
        self.progress.close()
        QMessageBox.information(self, "Success", f"✅ Final video saved to:\n{save_path}")
        self.record_btn.setEnabled(True); self.process_btn.setEnabled(True)
        self.status_bar.showMessage("Ready"); self.video_timer.start(15)

    def on_render_error(self, err):
        self.progress.close()
        QMessageBox.critical(self, "Error", f"Processing failed:\n{err}")
        self.record_btn.setEnabled(True); self.process_btn.setEnabled(True)
        self.status_bar.showMessage("Ready"); self.video_timer.start(15)

if __name__ == "__main__":
    import sys
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = WebcamClearVSTPro()
    window.show()
    sys.exit(app.exec())