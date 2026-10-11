from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                               QPushButton, QComboBox, QSlider, QCheckBox, QLineEdit, 
                               QSpinBox, QGroupBox, QScrollArea, QSizePolicy, QProgressBar)
from PySide6.QtCore import Qt
from constants import PARAM_DEFS, PALETTES
from ui_styles import create_slider_row, create_section_label

def setup_capture_tab(window):
    tab1 = QWidget()
    tab1_layout = QVBoxLayout(tab1)
    tab1_layout.setContentsMargins(10, 10, 10, 10)
    tab1_layout.setSpacing(8)
    
    cap_group = QGroupBox(" Capture Settings ")
    cap_layout = QGridLayout(cap_group)
    cap_layout.setSpacing(8); cap_layout.setContentsMargins(12, 18, 12, 12); lbl_w = 110
    
    lbl = QLabel("Video Title:"); lbl.setMinimumWidth(lbl_w); cap_layout.addWidget(lbl, 0, 0)
    window.title_edit = QLineEdit(); window.title_edit.setPlaceholderText("Leave blank for auto-generated name")
    cap_layout.addWidget(window.title_edit, 0, 1, 1, 3)
    
    lbl = QLabel("Resolution/FPS:"); lbl.setMinimumWidth(lbl_w); cap_layout.addWidget(lbl, 1, 0)
    window.res_combo = QComboBox()
    window.res_combo.addItems(["640x480", "1280x720", "1280x720 60fps", "1920x1080", "1920x1080 60fps"])
    window.res_combo.setCurrentText(window.settings["resolution"]); window.res_combo.currentTextChanged.connect(window.update_video_info)
    cap_layout.addWidget(window.res_combo, 1, 1, 1, 3)
    
    cap_layout.addWidget(QLabel("Countdown:"), 2, 0)
    window.countdown_combo = QComboBox(); window.countdown_combo.addItems(["3s", "5s", "10s"])
    window.countdown_combo.setCurrentText(window.settings["countdown_duration"]); cap_layout.addWidget(window.countdown_combo, 2, 1)
    cap_layout.addWidget(QLabel("Auto-Stop:"), 2, 2)
    window.auto_stop_combo = QComboBox(); window.auto_stop_combo.addItems(["Off", "15s", "30s", "60s", "3m", "5m"])
    window.auto_stop_combo.setCurrentText(window.settings.get("auto_stop", "Off"))
    cap_layout.addWidget(window.auto_stop_combo, 2, 3)
    
    cap_layout.addWidget(QLabel("Output Folder:"), 3, 0)
    window.output_dir_edit = QLineEdit(window.settings["output_dir"]); cap_layout.addWidget(window.output_dir_edit, 3, 1, 1, 2)
    browse_btn = QPushButton("Browse..."); browse_btn.clicked.connect(window.browse_output_dir)
    cap_layout.addWidget(browse_btn, 3, 3)
    
    window.shorts_check = QCheckBox("📱  Shorts Mode  (9:16 Vertical with Blurred Background)")
    window.shorts_check.setChecked(window.settings["shorts_mode"]); cap_layout.addWidget(window.shorts_check, 4, 0, 1, 4)
    tab1_layout.addWidget(cap_group)
    
    adj_group = QGroupBox(" Live Video Adjustments ")
    adj_layout = QVBoxLayout(adj_group); adj_layout.setSpacing(4); adj_layout.setContentsMargins(12, 18, 12, 12)
    chk_row = QHBoxLayout(); chk_row.setSpacing(20)
    window.mirror_check = QCheckBox("Mirror Video"); window.mirror_check.setChecked(window.settings["mirror"]); chk_row.addWidget(window.mirror_check)
    window.grid_check = QCheckBox("Rule of Thirds Grid"); window.grid_check.setChecked(window.settings["show_grid"]); chk_row.addWidget(window.grid_check)
    chk_row.addStretch(); adj_layout.addLayout(chk_row)
    bw, window.brightness_slider, _ = create_slider_row("Brightness:", 30, 200, int(window.settings["brightness"] * 100), scale=100.0, format_str="{:.2f}x"); adj_layout.addWidget(bw)
    cw, window.contrast_slider, _ = create_slider_row("Contrast:", 30, 200, int(window.settings["contrast"] * 100), scale=100.0, format_str="{:.2f}x"); adj_layout.addWidget(cw)
    tab1_layout.addWidget(adj_group); tab1_layout.addStretch()
    
    # Save immediately when live adjustments change
    window.shorts_check.stateChanged.connect(lambda s: window.save_settings())
    window.mirror_check.stateChanged.connect(lambda s: window.save_settings())
    window.grid_check.stateChanged.connect(lambda s: window.save_settings())
    window.brightness_slider.sliderReleased.connect(lambda: window.save_settings())
    window.contrast_slider.sliderReleased.connect(lambda: window.save_settings())
    
    window.tabs.addTab(tab1, "🎥 Capture")

def setup_fx_tab(window):
    tab2 = QWidget()
    tab2_layout = QVBoxLayout(tab2)
    tab2_layout.setContentsMargins(10, 10, 10, 10); tab2_layout.setSpacing(8)
    
    preset_group = QGroupBox(" Quick Presets ")
    preset_layout = QHBoxLayout(preset_group); preset_layout.setContentsMargins(12, 18, 12, 12)
    window.preset_combo = QComboBox()
    window.preset_combo.addItems(["Custom", "Cyberpunk City", "Noir Detective", "Matrix Code", "Retro VHS", "Thermal Vision", "Neon Outline", "Pencil Sketch", "Anime Ink", "Dreamcore"])
    preset_layout.addWidget(window.preset_combo, stretch=1)
    apply_preset_btn = QPushButton("Apply"); apply_preset_btn.clicked.connect(window.apply_preset); preset_layout.addWidget(apply_preset_btn)
    randomize_btn = QPushButton("🎲 Randomize"); randomize_btn.clicked.connect(window.randomize_params); preset_layout.addWidget(randomize_btn)
    reset_btn = QPushButton("⟲ Reset Effect"); reset_btn.clicked.connect(window.reset_current_effect); preset_layout.addWidget(reset_btn)
    tab2_layout.addWidget(preset_group)
    
    fx_group = QGroupBox(" Visual Effects Engine ")
    fx_layout = QGridLayout(fx_group); fx_layout.setSpacing(8); fx_layout.setContentsMargins(12, 18, 12, 12)
    fx_layout.addWidget(QLabel("Visual Effect:"), 0, 0)
    window.filter_combo = QComboBox()
    window.filter_combo.addItems(["Canny Edge", "Ghost Edges", "Laplacian Edge", "Sobel Edge", "XDog Sketch", "Adaptive Lines", "Morph Gradient", "Static Glitch", "Film Grain Edges", "RGB Shift", "Sketch Pencil", "Edge Glow Neon", "Wireframe Mesh", "ASCII Art", "Thermal Heatmap", "Heavy Pixelate", "Dotted Stipple", "Threshold Binary", "VHS Retro", "Posterize", "Contour Fill", "Digital Halftone"])
    window.filter_combo.setCurrentText(window.settings["filter"]); window.filter_combo.currentIndexChanged.connect(window.on_filter_changed)
    fx_layout.addWidget(window.filter_combo, 0, 1)
    
    fx_layout.addWidget(QLabel("Color Palette:"), 1, 0)
    window.palette_combo = QComboBox()
    window.palette_combo.addItems(list(PALETTES.keys()))  # Dynamically loaded from constants
    window.palette_combo.setCurrentText(window.settings.get("palette", "Neon Tokyo"))
    window.palette_combo.currentIndexChanged.connect(window.update_palette_preview)
    fx_layout.addWidget(window.palette_combo, 1, 1)
    
    window.palette_preview_label = QLabel()
    window.palette_preview_label.setMinimumHeight(22)
    window.palette_preview_label.setStyleSheet("border: 1px solid #333; border-radius: 4px;")
    fx_layout.addWidget(window.palette_preview_label, 1, 2)
    tab2_layout.addWidget(fx_group)
    
    param_container = QWidget()
    param_container_layout = QVBoxLayout(param_container)
    param_container_layout.setContentsMargins(0, 0, 0, 0); param_container_layout.setSpacing(8)
    
    window.param_sliders = {}
    window.param_groups = {}
    categories = {}
    for key, defn in PARAM_DEFS.items():
        grp = defn["group"]
        if grp not in categories:
            grp_box = QGroupBox(f" {grp} ")
            grp_layout = QVBoxLayout(grp_box); grp_layout.setSpacing(4); grp_layout.setContentsMargins(12, 18, 12, 12)
            categories[grp] = (grp_box, grp_layout)
            param_container_layout.addWidget(grp_box)
            window.param_groups[grp] = grp_box
    for key, defn in PARAM_DEFS.items():
        grp_box, grp_layout = categories[defn["group"]]
        suffix = defn.get("suffix", "")
        w, sld, _ = create_slider_row(defn["label"], defn["min"], defn["max"], defn["default"], scale=defn["scale"], suffix=suffix, tooltip=defn.get("tooltip", ""))
        grp_layout.addWidget(w)
        window.param_sliders[key] = (sld, defn["scale"], defn["default"])
        sld.valueChanged.connect(lambda v, k=key: window.on_param_changed(k, v))
        
    param_scroll = QScrollArea(); param_scroll.setWidgetResizable(True); param_scroll.setFrameShape(QScrollArea.NoFrame); param_scroll.setWidget(param_container)
    tab2_layout.addWidget(param_scroll, stretch=1)
    window.on_filter_changed(); window.update_palette_preview()
    window.tabs.addTab(tab2, "🎨 Visual FX")

def setup_audio_tab(window):
    tab3 = QWidget()
    tab3_layout = QVBoxLayout(tab3)
    tab3_layout.setContentsMargins(10, 10, 10, 10); tab3_layout.setSpacing(8)
    tab3_content = QWidget()
    tab3_content_layout = QVBoxLayout(tab3_content); tab3_content_layout.setSpacing(8)
    
    audio_group = QGroupBox(" 🎙️ Audio Input & Monitoring ")
    audio_layout = QVBoxLayout(audio_group); audio_layout.setSpacing(6); audio_layout.setContentsMargins(12, 18, 12, 12)
    dev_row = QHBoxLayout(); dev_row.setSpacing(10)
    lbl = QLabel("Device:"); lbl.setMinimumWidth(130); dev_row.addWidget(lbl)
    window.device_combo = QComboBox(); dev_row.addWidget(window.device_combo, stretch=1)
    refresh_btn = QPushButton("🔄"); refresh_btn.setFixedSize(28, 28); refresh_btn.setToolTip("Refresh devices"); refresh_btn.clicked.connect(window.refresh_audio_devices)
    dev_row.addWidget(refresh_btn); audio_layout.addLayout(dev_row)
    
    lvl_row = QHBoxLayout(); lvl_row.setSpacing(10)
    lbl = QLabel("Input Level:"); lbl.setMinimumWidth(130); lvl_row.addWidget(lbl)
    meter_container = QWidget(); meter_layout = QVBoxLayout(meter_container); meter_layout.setContentsMargins(0, 0, 0, 0); meter_layout.setSpacing(1)
    window.audio_level_bar = QProgressBar(); window.audio_level_bar.setMaximum(100); window.audio_level_bar.setTextVisible(True); window.audio_level_bar.setFormat("%v%"); window.audio_level_bar.setMinimumHeight(18)
    meter_layout.addWidget(window.audio_level_bar)
    window.peak_label = QLabel("Peak: -- dB"); window.peak_label.setStyleSheet("color: #888; font-size: 10px; font-family: Consolas;"); window.peak_label.setAlignment(Qt.AlignRight)
    meter_layout.addWidget(window.peak_label)
    lvl_row.addWidget(meter_container, stretch=1); audio_layout.addLayout(lvl_row)
    
    gain_w, window.gain_slider, window.gain_val_lbl = create_slider_row("Input Gain:", 0, 500, int(window.settings["input_gain"] * 100), scale=100.0, format_str="{:.2f}x")
    audio_layout.addWidget(gain_w)
    tab3_content_layout.addWidget(audio_group)
    
    proc_group = QGroupBox(" Processing Options ")
    proc_layout = QVBoxLayout(proc_group); proc_layout.setSpacing(5); proc_layout.setContentsMargins(12, 18, 12, 12)
    window.yt_loud_check = QCheckBox("🎙️  Broadcast Loudness  (Limiter)"); window.yt_loud_check.setChecked(window.settings["yt_loudness"]); proc_layout.addWidget(window.yt_loud_check)
    window.gate_check = QCheckBox("🔇  Noise Gate  (Cuts background hum when silent)"); window.gate_check.setChecked(window.settings["noise_gate"]); proc_layout.addWidget(window.gate_check)
    window.noise_suppress_check = QCheckBox("🧹  OBS Noise Suppression  (Speex / RNNoise)")
    window.noise_suppress_check.setChecked(window.settings.get("noise_suppress", True)); proc_layout.addWidget(window.noise_suppress_check)
    tab3_content_layout.addWidget(proc_group)
    
    fx_group = QGroupBox(" 🎚️ Expert Voice Design ")
    fx_layout = QVBoxLayout(fx_group); fx_layout.setSpacing(2); fx_layout.setContentsMargins(12, 18, 12, 12)
    
    fx_layout.addWidget(create_section_label("  HIGHPASS FILTER  (Removes low rumble)"))
    hp_w, window.hp_freq_slider, window.hp_freq_lbl = create_slider_row("Cutoff Frequency:", 20, 300, window.settings.get("hp_freq", 85), format_str="{:.0f} Hz")
    fx_layout.addWidget(hp_w)
    
    fx_layout.addWidget(create_section_label("  EQUALIZER  (3-BAND with Q)"))
    eq_low_w, window.eq_low_slider, window.eq_low_lbl = create_slider_row("Low (Bass):", -24, 24, window.settings.get("eq_low", 0), format_str="{:+.1f} dB"); fx_layout.addWidget(eq_low_w)
    eq_low_q_w, window.eq_low_q_slider, _ = create_slider_row("  └ Low Q:", 10, 500, int(window.settings.get("eq_low_q", 0.7) * 100), scale=100.0, format_str="{:.2f}"); fx_layout.addWidget(eq_low_q_w)
    eq_mid_w, window.eq_mid_slider, window.eq_mid_lbl = create_slider_row("Mid:", -24, 24, window.settings.get("eq_mid", 0), format_str="{:+.1f} dB"); fx_layout.addWidget(eq_mid_w)
    eq_mid_q_w, window.eq_mid_q_slider, _ = create_slider_row("  └ Mid Q:", 10, 500, int(window.settings.get("eq_mid_q", 1.0) * 100), scale=100.0, format_str="{:.2f}"); fx_layout.addWidget(eq_mid_q_w)
    eq_high_w, window.eq_high_slider, window.eq_high_lbl = create_slider_row("High (Treble):", -24, 24, window.settings.get("eq_high", 0), format_str="{:+.1f} dB"); fx_layout.addWidget(eq_high_w)
    eq_high_q_w, window.eq_high_q_slider, _ = create_slider_row("  └ High Q:", 10, 500, int(window.settings.get("eq_high_q", 0.7) * 100), scale=100.0, format_str="{:.2f}"); fx_layout.addWidget(eq_high_q_w)
    
    fx_layout.addWidget(create_section_label("  EXPERT DYNAMICS  (Saturation & Punch)"))
    tape_w, window.tape_slider, window.tape_lbl = create_slider_row("Tape Saturation:", 0, 100, window.settings.get("tape_drive", 15), format_str="{:.0f}%", tooltip="Adds analog warmth and harmonic distortion.")
    fx_layout.addWidget(tape_w)
    trans_w, window.transient_slider, window.trans_lbl = create_slider_row("Transient Punch:", 0, 100, window.settings.get("transient_punch", 30), format_str="{:.0f}%", tooltip="Enhances the attack of consonants for crispness.")
    fx_layout.addWidget(trans_w)
    
    fx_layout.addWidget(create_section_label("  DYNAMIC DE-ESSER  (Tames harsh 'S' sounds)"))
    deess_w, window.deess_slider, window.deess_lbl = create_slider_row("De-Ess Threshold:", -60, 0, window.settings.get("deess_thresh", -35), format_str="{:+.0f} dB"); fx_layout.addWidget(deess_w)
    deess_ratio_w, window.deess_ratio_slider, _ = create_slider_row("De-Ess Ratio:", 10, 100, int(window.settings.get("deess_ratio", 3.0) * 10), scale=10.0, format_str="{:.1f}:1"); fx_layout.addWidget(deess_ratio_w)
    
    fx_layout.addWidget(create_section_label("  OBS NOISE SUPPRESSION  (Speex / RNNoise)"))
    ns_mode_row = QHBoxLayout(); ns_mode_row.setSpacing(10)
    lbl = QLabel("Algorithm:"); lbl.setMinimumWidth(130); ns_mode_row.addWidget(lbl)
    window.ns_mode_combo = QComboBox()
    window.ns_mode_combo.addItems(["Speex", "RNNoise"])
    window.ns_mode_combo.setCurrentText(window.settings.get("noise_suppress_mode", "Speex"))
    window.ns_mode_combo.setToolTip("Speex: Lightweight spectral subtraction (OBS default).\nRNNoise: Neural-inspired with adaptive oversubtraction.")
    ns_mode_row.addWidget(window.ns_mode_combo, stretch=1)
    fx_layout.addLayout(ns_mode_row)
    ns_w, window.ns_amount_slider, window.ns_amount_lbl = create_slider_row("Suppression:", 3, 60, window.settings.get("noise_suppress_amount", 30), format_str="{:.0f} dB", tooltip="Maximum noise attenuation. OBS default: 30 dB."); fx_layout.addWidget(ns_w)
    
    fx_layout.addWidget(create_section_label("  NOISE GATE "))
    gate_w, window.gate_slider, window.gate_lbl = create_slider_row("Gate Threshold:", -90, -10, window.settings.get("gate_thresh", -45), format_str="{:.0f} dB"); fx_layout.addWidget(gate_w)
    gate_ratio_w, window.gate_ratio_slider, _ = create_slider_row("Gate Ratio:", 10, 100, int(window.settings.get("gate_ratio", 4.0) * 10), scale=10.0, format_str="{:.1f}:1"); fx_layout.addWidget(gate_ratio_w)
    gate_attack_w, window.gate_attack_slider, _ = create_slider_row("Gate Attack:", 1, 100, window.settings.get("gate_attack", 5), format_str="{:.0f} ms"); fx_layout.addWidget(gate_attack_w)
    gate_release_w, window.gate_release_slider, _ = create_slider_row("Gate Release:", 10, 1000, window.settings.get("gate_release", 250), format_str="{:.0f} ms"); fx_layout.addWidget(gate_release_w)
    
    fx_layout.addWidget(create_section_label("  VOICE PRESENCE  (Adds 'Air' when speaking)"))
    air_w, window.air_slider, window.air_lbl = create_slider_row("Air Enhancer:", 0, 120, int(window.settings.get("air_boost", 3.0) * 10), scale=10.0, format_str="{:+.1f} dB"); fx_layout.addWidget(air_w)
    
    fx_layout.addWidget(create_section_label("  AMBIENT SOUND GENERATOR "))
    amb_row = QHBoxLayout(); amb_row.setSpacing(10)
    lbl = QLabel("Background:"); lbl.setMinimumWidth(130); amb_row.addWidget(lbl)
    window.ambient_combo = QComboBox()
    window.ambient_combo.addItems(["None", "White Noise", "Pink Noise", "Brown Noise", "Light Rain", "Heavy Rain", "Thunderstorm", "Rain on Tin Roof", "Ocean Waves", "Gentle Stream", "Waterfall", "Wind", "Fireplace", "Forest Night", "Birdsong Morning", "Café Ambience", "Keyboard Typing", "Clock Ticking", "Distant Traffic", "Heartbeat", "Space Drone"])
    window.ambient_combo.setCurrentText(window.settings.get("ambient_sound", "None"))
    amb_row.addWidget(window.ambient_combo, stretch=1); fx_layout.addLayout(amb_row)
    
    def amb_vol_formatter(v, real_val):
        if v <= -60: return "Muted"
        return f"{v:+d} dB"
    amb_vol_w, window.ambient_vol_slider, window.amb_vol_lbl = create_slider_row("SFX Volume:", -60, 6, window.settings.get("ambient_vol_db", -18), custom_formatter=amb_vol_formatter); fx_layout.addWidget(amb_vol_w)
    sfx_master_w, window.sfx_master_slider, window.sfx_master_lbl = create_slider_row("SFX Master:", -12, 12, window.settings.get("sfx_master_db", 0), format_str="{:+.0f} dB"); fx_layout.addWidget(sfx_master_w)
    window.duck_check = QCheckBox("🤖  Auto-Duck SFX  (Lowers ambient when you speak)"); window.duck_check.setChecked(window.settings.get("duck_sfx", True)); fx_layout.addWidget(window.duck_check)
    
    fx_layout.addWidget(create_section_label("  OUTPUT"))
    out_vol_w, window.out_vol_slider, window.out_vol_val_lbl = create_slider_row("Output Volume:", 0, 500, int(window.settings["output_vol"] * 100), scale=100.0, format_str="{:.2f}x"); fx_layout.addWidget(out_vol_w)
    
    # UPDATED MONITOR BUTTON (Now a checkable toggle)
    window.monitor_btn = QPushButton("🎧  Start Live Monitor  (Full Chain)")
    window.monitor_btn.setMinimumHeight(36)
    window.monitor_btn.setCheckable(True)
    window.monitor_btn.clicked.connect(window.toggle_monitor)
    fx_layout.addWidget(window.monitor_btn)
    
    tab3_content_layout.addWidget(fx_group); tab3_content_layout.addStretch()
    
    audio_scroll = QScrollArea(); audio_scroll.setWidgetResizable(True); audio_scroll.setFrameShape(QScrollArea.NoFrame); audio_scroll.setWidget(tab3_content)
    tab3_layout.addWidget(audio_scroll)
    window.tabs.addTab(tab3, "🎙️ Audio")

def setup_render_tab(window):
    tab5 = QWidget()
    tab5_layout = QVBoxLayout(tab5)
    tab5_layout.setContentsMargins(10, 10, 10, 10); tab5_layout.setSpacing(8)
    render_group = QGroupBox(" Render & Export Settings ")
    render_layout = QGridLayout(render_group); render_layout.setSpacing(8); render_layout.setContentsMargins(12, 18, 12, 12); lbl_w = 130
    lbl = QLabel("Render Quality:"); lbl.setMinimumWidth(lbl_w); render_layout.addWidget(lbl, 0, 0)
    window.render_combo = QComboBox(); window.render_combo.addItems(["ultrafast", "fast", "medium", "slow"])
    window.render_combo.setCurrentText(window.settings["render_preset"]); render_layout.addWidget(window.render_combo, 0, 1)
    lbl = QLabel("Encoder:"); lbl.setMinimumWidth(lbl_w); render_layout.addWidget(lbl, 1, 0)
    window.encoder_combo = QComboBox(); window.encoder_combo.addItems(["Software (x264)", "NVIDIA NVENC", "Intel QSV", "AMD AMF"])
    window.encoder_combo.setCurrentText(window.settings["encoder"]); render_layout.addWidget(window.encoder_combo, 1, 1)
    tab5_layout.addWidget(render_group); tab5_layout.addStretch()
    window.tabs.addTab(tab5, "⚙️ Render")