import colorsys
import numpy as np

def _gen_sine_hsv(h_start, h_range, s_base, s_range, v_base, v_range, steps=100):
    """Generates palettes with oscillating hues/saturations/values for complex gradients."""
    stops = []
    for i in range(steps + 1):
        t = i / steps
        wave = np.sin(t * np.pi)
        inv_wave = 1.0 - wave
        
        h = (h_start + h_range * t) % 360.0
        s = s_base + s_range * wave
        v = v_base + v_range * inv_wave
        
        r, g, b = colorsys.hsv_to_rgb(h/360.0, np.clip(s, 0, 1), np.clip(v, 0, 1))
        stops.append((t, [int(r*255), int(g*255), int(b*255)]))
    return stops

def _gen_multi_stop(stops_list, steps=100):
    """Smoothly interpolates through a list of specific distinct color points [(t, [r,g,b]), ...]"""
    stops = []
    stops_list = sorted(stops_list, key=lambda x: x[0])
    for i in range(steps + 1):
        t = i / steps
        p1, c1 = stops_list[0]
        p2, c2 = stops_list[-1]
        for j in range(len(stops_list) - 1):
            if stops_list[j][0] <= t <= stops_list[j+1][0]:
                p1, c1 = stops_list[j]
                p2, c2 = stops_list[j+1]
                break
        
        if p2 == p1:
            ratio = 0.0
        else:
            ratio = (t - p1) / (p2 - p1)
        
        r = int(c1[0] + (c2[0] - c1[0]) * ratio)
        g = int(c1[1] + (c2[1] - c1[1]) * ratio)
        b = int(c1[2] + (c2[2] - c1[2]) * ratio)
        stops.append((t, [r, g, b]))
    return stops

PARAM_DEFS = {
    # Edge Detection
    "canny_low":    {"label": "Edge Low Threshold", "group": "Edge Detection",  "min": 10,  "max": 200, "default": 50,  "scale": 1.0,   "tooltip": "Lower bound for edge detection. Lower values detect more edges."},
    "canny_high":   {"label": "Edge High Threshold","group": "Edge Detection",  "min": 50,  "max": 400, "default": 150, "scale": 1.0,   "tooltip": "Upper bound for edge detection. Higher values detect fewer edges."},
    "density":      {"label": "Line Density",      "group": "Edge Detection",  "min": 1,   "max": 50,  "default": 15,  "scale": 1.0,   "tooltip": "Controls the density of lines or morphological operations."},
    "bin_thresh":   {"label": "Binary Threshold",  "group": "Edge Detection",  "min": 0,   "max": 255, "default": 128, "scale": 1.0,   "tooltip": "Threshold value for binary conversion."},
    
    # Pre-processing
    "blur_k":       {"label": "Blur Strength",     "group": "Pre-processing",  "min": 1,   "max": 31,  "default": 5,   "scale": 1.0,   "tooltip": "Gaussian blur kernel size applied before edge detection."},
    
    # Stylization
    "glow":         {"label": "Glow Intensity",    "group": "Stylization",     "min": 0,   "max": 100, "default": 50,  "scale": 1.0,   "suffix": "%", "tooltip": "Intensity of the neon glow effect."},
    "ghost_decay":  {"label": "Trail Decay",       "group": "Stylization",     "min": 50,  "max": 99,  "default": 85,  "scale": 100.0, "tooltip": "How fast previous frames fade away. 99% = long trails."},
    "pix_block":    {"label": "Pixel Block Size",  "group": "Stylization",     "min": 8,   "max": 80,  "default": 24,  "scale": 1.0,   "suffix": "px","tooltip": "Size of pixelation blocks."},
    "ascii_block":  {"label": "ASCII Density",     "group": "Stylization",     "min": 4,   "max": 24,  "default": 10,  "scale": 1.0,   "suffix": "px","tooltip": "Size of grid blocks for ASCII characters."},
    "ascii_bg":     {"label": "ASCII Background",  "group": "Stylization",     "min": 0,   "max": 255, "default": 0,   "scale": 1.0,   "tooltip": "Background brightness for ASCII art."},
    "mesh_pts":     {"label": "Mesh Density",      "group": "Stylization",     "min": 30,  "max": 400, "default": 150, "scale": 1.0,   "tooltip": "Number of points used to generate the wireframe mesh."},
    "mesh_thick":   {"label": "Mesh Line Width",   "group": "Stylization",     "min": 1,   "max": 5,   "default": 1,   "scale": 1.0,   "tooltip": "Thickness of the wireframe lines."},
    "sketch_scale": {"label": "Sketch Scale",      "group": "Stylization",     "min": 10,  "max": 100, "default": 60,  "scale": 100.0, "tooltip": "Brightness/scale multiplier for pencil sketches."},
    "stipple":      {"label": "Stipple Density",   "group": "Stylization",     "min": 10,  "max": 100, "default": 50,  "scale": 1.0,   "suffix": "%", "tooltip": "Density of dots in the stipple effect."},
    
    # Glitch & Noise
    "noise":        {"label": "Noise Intensity",   "group": "Glitch & Noise",  "min": 0,   "max": 100, "default": 30,  "scale": 1.0,   "suffix": "%", "tooltip": "Amount of random noise or grain applied."},
    "rgb_shift":    {"label": "Channel Shift",     "group": "Glitch & Noise",  "min": 0,   "max": 50,  "default": 5,   "scale": 1.0,   "suffix": "px","tooltip": "Pixels to shift RGB channels for chromatic aberration."},
    "glitch_bands": {"label": "Glitch Bands",      "group": "Glitch & Noise",  "min": 0,   "max": 30,  "default": 5,   "scale": 1.0,   "tooltip": "Number of horizontal glitch slices."},
    "scanlines":    {"label": "Scanline Intensity","group": "Glitch & Noise",  "min": 0,   "max": 100, "default": 0,   "scale": 1.0,   "suffix": "%", "tooltip": "CRT/VHS horizontal black line intensity."},
    "distortion":   {"label": "Wave Distortion",   "group": "Glitch & Noise",  "min": 0,   "max": 50,  "default": 0,   "scale": 1.0,   "suffix": "px","tooltip": "Sine-wave horizontal distortion."},
    
    # Colorization
    "invert":       {"label": "Invert Colors",     "group": "Colorization",    "min": 0,   "max": 1,   "default": 0,   "scale": 1.0,   "tooltip": "Invert the final output colors (0=Off, 1=On)."},
    "vignette":     {"label": "Vignette Amount",   "group": "Colorization",    "min": 0,   "max": 100, "default": 0,   "scale": 1.0,   "suffix": "%", "tooltip": "Darkens the edges of the screen."},
    "bloom":        {"label": "Bloom Intensity",   "group": "Colorization",    "min": 0,   "max": 100, "default": 0,   "scale": 1.0,   "suffix": "%", "tooltip": "Glowing effect for bright areas."}
}

FILTER_PARAMS = {
    "Canny Edge":       ["canny_low", "canny_high", "invert", "vignette", "bloom"],
    "Ghost Edges":      ["canny_low", "canny_high", "ghost_decay", "vignette", "bloom"],
    "Laplacian Edge":   ["blur_k", "invert", "vignette", "bloom"],
    "Sobel Edge":       ["blur_k", "invert", "vignette", "bloom"],
    "XDog Sketch":      ["blur_k", "density", "sketch_scale", "invert", "vignette"],
    "Adaptive Lines":   ["blur_k", "density", "invert", "vignette"],
    "Morph Gradient":   ["blur_k", "density", "invert", "vignette"],
    "Sketch Pencil":    ["blur_k", "sketch_scale", "invert", "vignette"],
    "Edge Glow Neon":   ["canny_low", "canny_high", "glow", "bloom", "vignette"],
    "Wireframe Mesh":   ["mesh_pts", "mesh_thick", "vignette"],
    "ASCII Art":        ["ascii_block", "ascii_bg", "invert"],
    "Thermal Heatmap":  ["blur_k", "invert", "bloom", "vignette"],
    "Heavy Pixelate":   ["pix_block", "vignette"],
    "Dotted Stipple":   ["stipple", "invert", "vignette"],
    "Threshold Binary": ["bin_thresh", "invert", "vignette"],
    "Static Glitch":    ["canny_low", "canny_high", "noise", "glitch_bands", "invert", "scanlines", "distortion"],
    "Film Grain Edges": ["canny_low", "canny_high", "noise", "glitch_bands", "invert", "scanlines"],
    "RGB Shift":        ["canny_low", "canny_high", "rgb_shift", "noise", "glitch_bands", "scanlines", "distortion"],
    "VHS Retro":        ["rgb_shift", "noise", "glitch_bands", "scanlines", "distortion"],
    "Posterize":        ["density", "invert", "bloom", "vignette"],
    "Contour Fill":     ["canny_low", "density", "invert", "vignette"],
    "Digital Halftone": ["stipple", "invert", "vignette"]
}

PALETTES = {
    "White":          [(0.0, [0, 0, 0]), (1.0, [255, 255, 255])],
    
    # --- Highly Detailed Artistic Multi-Stops (101 colors) ---
    "Bioluminescent": _gen_multi_stop([
        (0.0, [0, 0, 0]), (0.2, [0, 15, 40]), (0.4, [0, 80, 120]), 
        (0.6, [0, 200, 220]), (0.8, [100, 255, 180]), (1.0, [230, 255, 255])
    ], 100),
    
    "Holographic": _gen_multi_stop([
        (0.0, [10, 0, 20]), (0.15, [80, 0, 120]), (0.30, [200, 0, 255]), 
        (0.45, [0, 150, 255]), (0.60, [0, 255, 200]), (0.75, [180, 255, 220]), 
        (0.90, [255, 100, 200]), (1.0, [255, 240, 150])
    ], 100),
    
    "Radioactive": _gen_multi_stop([
        (0.0, [0, 0, 0]), (0.2, [0, 30, 0]), (0.4, [20, 80, 0]), 
        (0.6, [60, 150, 0]), (0.8, [180, 255, 20]), (1.0, [255, 255, 200])
    ], 100),
    
    "Coral Reef": _gen_multi_stop([
        (0.0, [0, 0, 10]), (0.2, [10, 0, 40]), (0.4, [100, 0, 100]), 
        (0.6, [200, 50, 100]), (0.8, [255, 150, 120]), (1.0, [255, 240, 220])
    ], 100),
    
    "Ember & Ash": _gen_multi_stop([
        (0.0, [5, 5, 5]), (0.2, [30, 10, 5]), (0.4, [100, 20, 0]), 
        (0.6, [200, 60, 0]), (0.8, [255, 150, 20]), (1.0, [255, 230, 150])
    ], 100),
    
    "Steampunk": _gen_multi_stop([
        (0.0, [10, 5, 0]), (0.2, [40, 20, 0]), (0.4, [100, 50, 10]), 
        (0.6, [180, 100, 20]), (0.8, [220, 160, 50]), (1.0, [255, 230, 150])
    ], 100),
    
    "Tie-Dye": _gen_multi_stop([
        (0.0, [10, 0, 30]), (0.15, [200, 0, 100]), (0.30, [255, 100, 0]), 
        (0.45, [255, 230, 0]), (0.60, [50, 200, 50]), (0.75, [0, 200, 255]), 
        (0.90, [100, 50, 255]), (1.0, [255, 50, 200])
    ], 100),
    
    "Neon Genesis": _gen_multi_stop([
        (0.0, [0, 0, 0]), (0.15, [20, 0, 40]), (0.30, [80, 0, 120]), 
        (0.45, [150, 0, 180]), (0.60, [255, 0, 150]), (0.75, [0, 200, 255]), 
        (0.90, [150, 240, 255]), (1.0, [255, 255, 255])
    ], 100),

    # --- 100-Color Dynamic Sine Wave Palettes ---
    "100 Color Rainbow":   _gen_sine_hsv(0, 360, 1.0, 0.0, 0.0, 1.0, 100),
    "100 Color Inferno":   _gen_sine_hsv(0, 60, 1.0, 0.0, 0.0, 1.0, 100),
    "100 Color Ocean":     _gen_sine_hsv(180, 60, 1.0, 0.0, 0.0, 1.0, 100),
    "100 Color Cyberpunk": _gen_sine_hsv(300, 240, 1.0, 0.0, 0.0, 1.0, 100),
    "100 Color Forest":    _gen_sine_hsv(80, 80, 1.0, 0.0, 0.0, 1.0, 100),
    "100 Color Galaxy":    _gen_sine_hsv(240, 60, 1.0, 0.0, 0.0, 1.0, 100),

    # --- Highly Expanded Standard Palettes ---
    "Neon Cyan":      [(0.0, [0, 0, 0]), (0.1, [0, 10, 25]), (0.2, [0, 25, 45]), (0.3, [0, 50, 80]), (0.4, [0, 90, 120]), (0.5, [0, 130, 160]), (0.55, [0, 160, 200]), (0.7, [40, 200, 230]), (0.8, [60, 240, 255]), (0.9, [150, 250, 255]), (1.0, [200, 255, 255])],
    "Matrix Green":   [(0.0, [0, 0, 0]), (0.1, [0, 10, 0]), (0.2, [0, 25, 5]), (0.3, [0, 40, 10]), (0.4, [0, 60, 15]), (0.5, [0, 90, 20]), (0.6, [0, 110, 25]), (0.7, [10, 150, 40]), (0.8, [40, 190, 60]), (0.9, [100, 230, 100]), (1.0, [220, 255, 220])],
    "Ice Blue":       [(0.0, [0, 5, 20]), (0.1, [5, 15, 40]), (0.2, [10, 30, 60]), (0.3, [20, 50, 90]), (0.4, [30, 70, 110]), (0.5, [50, 100, 150]), (0.6, [80, 160, 230]), (0.7, [120, 200, 240]), (0.8, [160, 220, 250]), (0.9, [200, 240, 255]), (1.0, [255, 255, 255])],
    "Acid Pink":      [(0.0, [0, 0, 0]), (0.1, [10, 0, 20]), (0.2, [25, 0, 40]), (0.3, [40, 0, 60]), (0.4, [60, 0, 80]), (0.5, [100, 0, 120]), (0.6, [150, 10, 150]), (0.7, [190, 20, 170]), (0.8, [230, 40, 200]), (0.9, [255, 100, 230]), (1.0, [255, 210, 255])],
    "Inferno":        [(0.00, [0, 0, 0]), (0.05, [20, 0, 30]), (0.10, [40, 0, 60]), (0.15, [60, 0, 90]), (0.20, [80, 0, 110]), (0.30, [120, 10, 100]), (0.40, [160, 20, 80]), (0.50, [200, 40, 60]), (0.60, [230, 80, 30]), (0.70, [245, 120, 20]), (0.80, [255, 160, 30]), (0.90, [255, 210, 60]), (1.00, [255, 255, 220])],
    "Cyberpunk":      [(0.00, [0, 0, 0]), (0.05, [10, 0, 20]), (0.10, [20, 0, 50]), (0.15, [40, 0, 80]), (0.20, [60, 0, 110]), (0.25, [90, 0, 140]), (0.30, [120, 10, 160]), (0.35, [150, 20, 170]), (0.40, [180, 30, 160]), (0.45, [210, 40, 150]), (0.50, [240, 50, 140]), (0.60, [255, 80, 120]), (0.70, [255, 120, 100]), (0.80, [150, 200, 230]), (0.90, [180, 230, 255]), (1.00, [255, 255, 255])],
    "Sunset":         [(0.00, [10, 0, 30]), (0.10, [30, 5, 50]), (0.20, [50, 10, 70]), (0.30, [80, 15, 90]), (0.40, [120, 20, 100]), (0.50, [160, 30, 90]), (0.60, [200, 50, 70]), (0.70, [230, 90, 50]), (0.80, [245, 140, 30]), (0.90, [255, 200, 40]), (1.00, [255, 240, 150])],
    "Ocean Depth":    [(0.00, [0, 5, 20]), (0.10, [0, 15, 40]), (0.20, [0, 25, 60]), (0.30, [0, 40, 90]), (0.40, [0, 60, 120]), (0.50, [0, 90, 150]), (0.60, [0, 120, 170]), (0.70, [20, 160, 190]), (0.80, [60, 200, 220]), (0.90, [150, 230, 240]), (1.00, [240, 255, 255])],
    "Fire":           [(0.00, [0, 0, 0]), (0.05, [20, 0, 0]), (0.10, [40, 0, 0]), (0.15, [60, 5, 0]), (0.20, [80, 10, 0]), (0.30, [120, 20, 0]), (0.40, [160, 40, 0]), (0.50, [200, 70, 0]), (0.60, [230, 110, 10]), (0.70, [245, 150, 20]), (0.80, [255, 190, 40]), (0.90, [255, 230, 80]), (1.00, [255, 255, 230])],
    "Viridis":        [(0.00, [68, 1, 84]), (0.10, [72, 20, 100]), (0.20, [72, 35, 116]), (0.30, [68, 50, 128]), (0.40, [62, 67, 135]), (0.50, [52, 84, 140]), (0.60, [42, 102, 142]), (0.70, [32, 120, 140]), (0.80, [28, 140, 136]), (0.90, [30, 160, 120]), (1.00, [253, 231, 37])],
    "Aurora":         [(0.00, [0, 5, 20]), (0.10, [0, 20, 25]), (0.20, [0, 40, 30]), (0.30, [0, 70, 50]), (0.40, [0, 100, 70]), (0.50, [0, 140, 90]), (0.60, [0, 180, 110]), (0.70, [40, 210, 140]), (0.80, [80, 230, 180]), (0.90, [150, 240, 220]), (1.00, [180, 100, 240])],
    "Toxic":          [(0.00, [0, 0, 0]), (0.10, [0, 15, 0]), (0.20, [0, 30, 0]), (0.30, [10, 50, 5]), (0.40, [20, 70, 10]), (0.50, [40, 100, 15]), (0.60, [60, 130, 20]), (0.70, [90, 170, 30]), (0.80, [130, 210, 40]), (0.90, [180, 240, 60]), (1.00, [255, 255, 200])],
    "Vaporwave":      [(0.00, [10, 0, 30]), (0.10, [30, 0, 60]), (0.20, [50, 10, 90]), (0.30, [80, 20, 120]), (0.40, [110, 40, 150]), (0.50, [140, 60, 170]), (0.60, [180, 80, 180]), (0.70, [220, 100, 190]), (0.80, [255, 120, 200]), (0.90, [200, 180, 230]), (1.00, [255, 240, 255])],
    "Gold":           [(0.00, [0, 0, 0]), (0.10, [15, 5, 0]), (0.20, [30, 10, 0]), (0.30, [50, 20, 5]), (0.40, [80, 40, 10]), (0.50, [120, 70, 15]), (0.60, [160, 100, 25]), (0.70, [200, 140, 40]), (0.80, [230, 180, 60]), (0.90, [250, 220, 90]), (1.00, [255, 250, 220])],
    "Rainbow Flow":   [(0.00, [255, 0, 0]), (0.10, [255, 100, 0]), (0.17, [255, 165, 0]), (0.25, [255, 220, 0]), (0.33, [255, 255, 0]), (0.42, [150, 230, 0]), (0.50, [0, 200, 0]), (0.58, [0, 200, 120]), (0.67, [0, 200, 255]), (0.75, [0, 100, 255]), (0.83, [0, 0, 255]), (0.92, [120, 0, 200]), (1.00, [180, 0, 255])],
    "Plasma":         [(0.00, [13, 8, 135]), (0.15, [50, 12, 140]), (0.30, [85, 15, 140]), (0.45, [120, 25, 135]), (0.60, [166, 40, 133]), (0.75, [200, 70, 115]), (0.90, [226, 100, 98]), (1.00, [240, 249, 33])],
    "Magma":          [(0.00, [0, 0, 4]), (0.10, [15, 5, 25]), (0.20, [30, 10, 50]), (0.30, [60, 15, 70]), (0.40, [100, 25, 80]), (0.50, [140, 40, 75]), (0.60, [180, 60, 65]), (0.70, [215, 90, 55]), (0.80, [240, 120, 45]), (0.90, [252, 160, 45]), (1.00, [255, 240, 180])],
    "Tropical":       [(0.00, [0, 60, 80]), (0.15, [0, 90, 100]), (0.30, [0, 120, 110]), (0.45, [40, 150, 90]), (0.60, [80, 180, 70]), (0.75, [150, 210, 50]), (0.90, [220, 200, 50]), (1.00, [240, 100, 30])],
    "Synthwave":      [(0.00, [20, 0, 40]), (0.15, [40, 10, 70]), (0.30, [70, 15, 100]), (0.45, [110, 30, 120]), (0.60, [160, 40, 130]), (0.75, [210, 60, 115]), (0.90, [245, 90, 90]), (1.00, [255, 150, 80])],
    "Neon Tokyo":     [(0.00, [5, 5, 20]), (0.10, [20, 5, 80]), (0.20, [60, 0, 120]), (0.30, [120, 0, 160]), (0.40, [180, 10, 140]), (0.50, [220, 20, 120]), (0.60, [255, 50, 150]), (0.70, [200, 80, 180]), (0.80, [100, 150, 230]), (0.90, [50, 220, 255]), (1.00, [200, 255, 255])],
    "Cotton Candy":   [(0.00, [50, 0, 80]), (0.10, [70, 10, 100]), (0.20, [90, 20, 130]), (0.30, [120, 40, 160]), (0.40, [150, 60, 190]), (0.50, [180, 80, 210]), (0.60, [210, 110, 220]), (0.70, [240, 150, 230]), (0.80, [255, 190, 235]), (0.90, [200, 220, 245]), (1.00, [255, 255, 255])],
    "Lava Reef":      [(0.00, [10, 5, 5]), (0.10, [30, 10, 5]), (0.20, [60, 15, 5]), (0.30, [90, 20, 0]), (0.40, [130, 30, 0]), (0.50, [170, 50, 5]), (0.60, [210, 80, 10]), (0.70, [240, 120, 20]), (0.80, [255, 160, 40]), (0.90, [255, 210, 80]), (1.00, [255, 255, 200])],
    "Deep Space":     [(0.00, [0, 0, 10]), (0.10, [5, 5, 30]), (0.20, [10, 10, 60]), (0.30, [20, 0, 90]), (0.40, [40, 10, 110]), (0.50, [60, 20, 120]), (0.60, [90, 30, 140]), (0.70, [130, 50, 160]), (0.80, [170, 80, 180]), (0.90, [210, 120, 200]), (1.00, [255, 200, 255])],
    "Turbo":          [(0.00, [48, 18, 59]), (0.10, [62, 73, 137]), (0.20, [64, 120, 179]), (0.30, [63, 163, 175]), (0.40, [70, 200, 122]), (0.50, [94, 221, 74]), (0.60, [140, 220, 42]), (0.70, [191, 211, 36]), (0.80, [236, 189, 43]), (0.90, [250, 147, 58]), (1.00, [248, 99, 69])],
    "Cividis":        [(0.00, [0, 32, 76]), (0.10, [30, 50, 90]), (0.20, [60, 70, 105]), (0.30, [90, 90, 115]), (0.40, [120, 110, 115]), (0.50, [150, 130, 105]), (0.60, [180, 150, 90]), (0.70, [200, 170, 75]), (0.80, [220, 190, 65]), (0.90, [240, 215, 60]), (1.00, [255, 233, 69])],
    "Emerald Forest": [(0.0, [0, 10, 5]), (0.1, [5, 20, 10]), (0.2, [10, 40, 15]), (0.3, [15, 60, 25]), (0.4, [20, 80, 35]), (0.5, [25, 110, 45]), (0.6, [40, 140, 60]), (0.7, [60, 170, 80]), (0.8, [90, 200, 110]), (0.9, [150, 230, 160]), (1.0, [220, 255, 220])],
    "Sahara Gold":    [(0.0, [20, 10, 0]), (0.1, [40, 25, 5]), (0.2, [60, 40, 10]), (0.3, [90, 60, 15]), (0.4, [120, 80, 25]), (0.5, [150, 100, 40]), (0.6, [180, 130, 60]), (0.7, [210, 160, 90]), (0.8, [235, 195, 130]), (0.9, [250, 220, 170]), (1.0, [255, 240, 180])],
    "Arctic Frost":   [(0.0, [10, 20, 30]), (0.1, [20, 40, 60]), (0.2, [30, 60, 90]), (0.3, [40, 80, 110]), (0.4, [60, 100, 130]), (0.5, [90, 130, 160]), (0.6, [120, 160, 190]), (0.7, [160, 190, 215]), (0.8, [190, 215, 235]), (0.9, [220, 235, 245]), (1.0, [255, 255, 255])],
    "Mojave Dusk":    [(0.0, [20, 10, 30]), (0.1, [40, 20, 50]), (0.2, [60, 25, 65]), (0.3, [80, 30, 75]), (0.4, [110, 40, 80]), (0.5, [140, 50, 80]), (0.6, [170, 70, 75]), (0.7, [200, 100, 65]), (0.8, [225, 130, 55]), (0.9, [245, 175, 65]), (1.0, [255, 220, 150])]
}