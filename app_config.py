import os
import json

DEFAULT_SETTINGS = {
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
    "tele_text_color": "#ffffff", "tele_highlight_color": "#ff4444",
    "auto_stop": "Off", "audio_device": ""  # <-- ADDED MISSING DEFAULTS
}

def load_settings(config_file):
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded = json.load(f)
                settings.update(loaded)
        except:
            pass
    return settings

def save_settings(config_file, settings):
    try:
        with open(config_file, 'w') as f:
            json.dump(settings, f, indent=4) 
    except:
        pass