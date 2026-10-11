import cv2
import numpy as np
from constants import PALETTES, FILTER_PARAMS

class VideoEngine:
    def __init__(self):
        self.edge_accumulator = None

    def _build_lut(self, stops):
        """Vectorized LUT generator that flawlessly interpolates all provided colors."""
        if not stops:
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256): lut[i] = [i, i, i]
            return lut
            
        stops = sorted(stops, key=lambda s: s[0])
        positions = np.array([s[0] for s in stops], dtype=np.float32)
        colors = np.array([s[1] for s in stops], dtype=np.float32)
        
        t = np.linspace(0, 1, 256, dtype=np.float32)
        
        lut = np.zeros((256, 3), dtype=np.uint8)
        for c in range(3):
            lut[:, c] = np.clip(np.interp(t, positions, colors[:, c]), 0, 255).astype(np.uint8)
            
        return lut

    def apply_palette(self, gray_img, palette_name):
        gray_img = gray_img.astype(np.uint8)
        if palette_name == "White":
            return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        if palette_name not in PALETTES:
            return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        lut_rgb = self._build_lut(PALETTES[palette_name])
        lut_bgr = np.ascontiguousarray(lut_rgb[:, ::-1])  # RGB -> BGR for OpenCV
        return lut_bgr[gray_img]

    def _p(self, params, key):
        return params.get(key, 0)

    def _apply_post_fx(self, frame, params):
        """Applies universal post-processing effects (Bloom, Vignette, Scanlines, Distortion)."""
        
        # 1. Bloom
        bloom_amt = self._p(params, "bloom") / 100.0
        if bloom_amt > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            if len(frame.shape) == 3:
                bright_mask = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
            blurred_bright = cv2.GaussianBlur(bright_mask, (21, 21), 0)
            frame = cv2.addWeighted(frame, 1.0, blurred_bright, bloom_amt * 1.5, 0)
            
        # 2. Vignette
        vig_amt = self._p(params, "vignette") / 100.0
        if vig_amt > 0:
            h, w = frame.shape[:2]
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            mask = 1.0 - (dist_from_center / max_dist)
            mask = np.clip(mask, 0, 1)
            mask = mask * vig_amt + (1.0 - vig_amt) # Blend from full bright to vignette dark
            frame = (frame * mask[..., None]).astype(np.uint8)
            
        # 3. Scanlines
        scan_amt = self._p(params, "scanlines") / 100.0
        if scan_amt > 0:
            frame[::2] = (frame[::2] * (1.0 - scan_amt)).astype(np.uint8)
            
        # 4. Wave Distortion
        dist_amt = self._p(params, "distortion")
        if dist_amt > 0:
            h, w = frame.shape[:2]
            map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
            map_y = np.tile(np.arange(h).reshape(-1, 1), (1, w)).astype(np.float32)
            map_x = map_x + (np.sin(map_y / 10.0) * dist_amt).astype(np.float32)
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
        return frame

    def apply_visual_effect(self, frame, filt, palette, params):
        if filt != "Ghost Edges": self.edge_accumulator = None
        invert = self._p(params, "invert") > 0.5
        
        def odd_k(v):
            v = max(1, int(v)); return v if v % 2 == 1 else v + 1

        if filt == "Canny Edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p(params, "canny_low")), int(self._p(params, "canny_high")))
            if invert: edges = 255 - edges
            frame = self.apply_palette(edges, palette)
        elif filt == "Ghost Edges":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p(params, "canny_low")), int(self._p(params, "canny_high")))
            if self.edge_accumulator is None or self.edge_accumulator.shape != edges.shape:
                self.edge_accumulator = np.zeros_like(edges, dtype=np.float32)
            self.edge_accumulator *= self._p(params, "ghost_decay")
            self.edge_accumulator += edges.astype(np.float32)
            np.clip(self.edge_accumulator, 0, 255, out=self.edge_accumulator)
            frame = self.apply_palette(self.edge_accumulator.astype(np.uint8), palette)
        elif filt == "Laplacian Edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p(params, "blur_k")); blurred = cv2.GaussianBlur(gray, (k, k), 0)
            lap = cv2.Laplacian(blurred, cv2.CV_64F); lap = np.absolute(lap)
            lap = (lap / lap.max() * 255).astype(np.uint8) if lap.max() > 0 else lap.astype(np.uint8)
            if invert: lap = 255 - lap
            frame = self.apply_palette(lap, palette)
        elif filt == "Sobel Edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p(params, "blur_k")); blurred = cv2.GaussianBlur(gray, (k, k), 0)
            sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3); sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sx**2 + sy**2)
            mag = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
            if invert: mag = 255 - mag
            frame = self.apply_palette(mag, palette)
        elif filt == "XDog Sketch":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p(params, "blur_k")); sigma1 = k / 2.0 if k > 1 else 0.5
            g1 = cv2.GaussianBlur(gray, (k, k), sigma1); g2 = cv2.GaussianBlur(gray, (k, k), sigma1 * 1.6)
            dog = g1 - g2; dog = dog / 255.0
            dog = np.where(dog < 0.01, 1.0, 1.0 + np.tanh(10.0 * (dog - 0.01)))
            dog = (dog * 255).clip(0, 255).astype(np.uint8); dog = 255 - dog 
            if invert: dog = 255 - dog
            frame = self.apply_palette(dog, palette)
        elif filt == "Adaptive Lines":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p(params, "blur_k")); gray = cv2.GaussianBlur(gray, (k, k), 0)
            block_size = max(3, int(self._p(params, "density")) * 2 + 1)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 5)
            if invert: edges = 255 - edges
            frame = self.apply_palette(edges, palette)
        elif filt == "Morph Gradient":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = max(1, int(self._p(params, "density"))); kernel = np.ones((k, k), np.uint8)
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            if invert: grad = 255 - grad
            frame = self.apply_palette(grad, palette)
        elif filt == "Static Glitch":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p(params, "canny_low")), int(self._p(params, "canny_high")))
            noise_amt = self._p(params, "noise") / 100.0
            noise = np.random.randint(0, 256, gray.shape, dtype=np.uint8)
            _, noise = cv2.threshold(noise, 200, 255, cv2.THRESH_BINARY)
            combined = cv2.addWeighted(edges, 1.0, noise, noise_amt * 2.0, 0)
            bands = int(self._p(params, "glitch_bands"))
            if bands > 0:
                h, w = combined.shape
                for _ in range(bands):
                    y = np.random.randint(0, h-10); h_slice = np.random.randint(2, 10); shift_x = np.random.randint(-20, 20)
                    combined[y:y+h_slice] = np.roll(combined[y:y+h_slice], shift_x, axis=1)
            if invert: combined = 255 - combined
            frame = self.apply_palette(combined, palette)
        elif filt == "Film Grain Edges":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p(params, "canny_low")), int(self._p(params, "canny_high")))
            noise_amt = self._p(params, "noise") / 100.0
            grain = np.random.normal(0, 25, gray.shape); grain = np.clip(grain, 0, 255).astype(np.uint8)
            combined = cv2.addWeighted(edges, 1.0, grain, noise_amt, 0)
            bands = int(self._p(params, "glitch_bands"))
            if bands > 0:
                h, w = combined.shape
                for _ in range(bands):
                    y = np.random.randint(0, h-10); h_slice = np.random.randint(2, 10); shift_x = np.random.randint(-20,20)
                    combined[y:y+h_slice] = np.roll(combined[y:y+h_slice], shift_x, axis=1)
            if invert: combined = 255 - combined
            frame = self.apply_palette(combined, palette)
        elif filt == "RGB Shift":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p(params, "canny_low")), int(self._p(params, "canny_high")))
            shift = int(self._p(params, "rgb_shift"))
            r = np.roll(edges, shift, axis=1); g = edges; b = np.roll(edges, -shift, axis=1)
            shifted = cv2.merge([b, g, r])
            noise_amt = self._p(params, "noise") / 100.0
            if noise_amt > 0:
                glitch_noise = np.random.randint(0, 256, shifted.shape, dtype=np.uint8)
                shifted = cv2.addWeighted(shifted, 1.0, glitch_noise, noise_amt * 0.5, 0)
            frame = shifted
        elif filt == "VHS Retro":
            shift = int(self._p(params, "rgb_shift"))
            b, g, r = cv2.split(frame)
            r = np.roll(r, shift, axis=1); b = np.roll(b, -shift, axis=1)
            frame = cv2.merge([b, g, r])
            bands = int(self._p(params, "glitch_bands"))
            if bands > 0:
                h, w = frame.shape[:2]
                for _ in range(bands):
                    y = np.random.randint(0, h-10); h_slice = np.random.randint(2, 10); shift_x = np.random.randint(-30, 30)
                    frame[y:y+h_slice] = np.roll(frame[y:y+h_slice], shift_x, axis=1)
            for y in range(0, frame.shape[0], 2): frame[y] = (frame[y] * 0.7).astype(np.uint8)
            noise_amt = self._p(params, "noise") / 100.0
            if noise_amt > 0:
                noise = np.random.normal(0, 15, frame.shape)
                frame = cv2.addWeighted(frame, 1.0, noise.astype(np.uint8), noise_amt, 0)
        elif filt == "Posterize":
            levels = max(2, int(self._p(params, "density"))); factor = 256 // levels
            posterized = (frame // factor) * factor
            if palette != "White":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); colored = self.apply_palette(gray, palette)
                factor_c = 256 // levels; posterized = (colored // factor_c) * factor_c
            if invert: posterized = 255 - posterized
            frame = posterized
        elif filt == "Contour Fill":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p(params, "canny_low")), int(self._p(params, "canny_low")*2))
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(frame); colored_palette = self.apply_palette(gray, palette)
            for i, cnt in enumerate(contours):
                color = colored_palette[gray.shape[0]//2, i % gray.shape[1]].tolist()
                cv2.drawContours(mask, [cnt], -1, color, int(self._p(params, "density")))
            if invert: mask = 255 - mask
            frame = mask
        elif filt == "Digital Halftone":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]; block = max(4, int(self._p(params, "stipple")))
            canvas = np.zeros_like(frame); colored = self.apply_palette(gray, palette)
            for y in range(0, h, block):
                for x in range(0, w, block):
                    val = gray[y, x]; radius = int((255 - val) / 255 * (block // 2))
                    if radius > 0: cv2.circle(canvas, (x + block//2, y + block//2), radius, colored[y, x].tolist(), -1)
            if invert: canvas = 255 - canvas
            frame = canvas
        elif filt == "Sketch Pencil":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = odd_k(self._p(params, "blur_k")); inv = 255 - gray; blur_inv = cv2.GaussianBlur(inv, (k, k), 0)
            sketch = cv2.divide(gray, blur_inv, scale=256.0)
            sketch = np.clip(sketch * self._p(params, "sketch_scale"), 0, 255).astype(np.uint8)
            if invert: sketch = 255 - sketch
            frame = self.apply_palette(sketch, palette)
        elif filt == "Edge Glow Neon":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p(params, "canny_low")), int(self._p(params, "canny_high")))
            colored = self.apply_palette(edges, palette)
            glow_pct = self._p(params, "glow"); blur_size = odd_k(int(3 + glow_pct * 0.5))
            glow1 = cv2.GaussianBlur(colored, (blur_size, blur_size), 0)
            glow2 = cv2.GaussianBlur(colored, (blur_size*2+1, blur_size*2+1), 0)
            glow = cv2.addWeighted(glow1, 0.6, glow2, 0.4, 0)
            frame = cv2.addWeighted(glow, 0.8, colored, 1.2, 0)
        elif filt == "Wireframe Mesh":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]; n_pts = int(self._p(params, "mesh_pts"))
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=n_pts, qualityLevel=0.02, minDistance=10, blockSize=7)
            mask = np.zeros_like(frame)
            if corners is not None and len(corners) > 3:
                pts = corners.reshape(-1, 2).astype(np.int32); rect = (0, 0, w, h); subdiv = cv2.Subdiv2D(rect)
                for p in pts: subdiv.insert((int(p[0]), int(p[1])))
                triangles = subdiv.getTriangleList()
                
                white_pixel = np.array([[255]], dtype=np.uint8)
                color = self.apply_palette(white_pixel, palette)[0, 0].tolist()
                thickness = int(self._p(params, "mesh_thick"))
                
                for t in triangles:
                    p1 = (int(t[0]), int(t[1])); p2 = (int(t[2]), int(t[3])); p3 = (int(t[4]), int(t[5]))
                    cv2.line(mask, p1, p2, color, thickness); cv2.line(mask, p2, p3, color, thickness); cv2.line(mask, p3, p1, color, thickness)
            frame = mask
        elif filt == "ASCII Art":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            block = max(4, int(self._p(params, "ascii_block"))); h, w = gray.shape[:2]
            chars = " .:-=+*#%@"; small = cv2.resize(gray, (w // block, h // block))
            
            bg_val = int(self._p(params, "ascii_bg"))
            canvas = np.full_like(frame, bg_val)
            colored = self.apply_palette(gray, palette)
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
            k = odd_k(self._p(params, "blur_k")); blurred = cv2.GaussianBlur(gray, (k, k), 0)
            if invert: blurred = 255 - blurred
            frame = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        elif filt == "Heavy Pixelate":
            block = max(2, int(self._p(params, "pix_block"))); h, w = frame.shape[:2]
            small = cv2.resize(frame, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
            frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif filt == "Dotted Stipple":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]; density = self._p(params, "stipple") / 100.0
            canvas = np.zeros_like(frame); n_dots = int(h * w * density * 0.05)
            ys = np.random.randint(0, h, n_dots); xs = np.random.randint(0, w, n_dots); vals = gray[ys, xs]
            colored = self.apply_palette(gray, palette)
            for i in range(n_dots):
                if vals[i] > 60: cv2.circle(canvas, (xs[i], ys[i]), 1, colored[ys[i], xs[i]].tolist(), -1)
            if invert: canvas = 255 - canvas
            frame = canvas
        elif filt == "Threshold Binary":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = int(self._p(params, "bin_thresh")); _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            if invert: bw = 255 - bw
            frame = self.apply_palette(bw, palette)
            
        # Apply universal Post-FX before returning
        frame = self._apply_post_fx(frame, params)
        return frame

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

    def apply_video_adjustments(self, frame, settings):
        mirror = settings.get('mirror', True)
        contrast = settings.get('contrast', 1.0)
        brightness = (settings.get('brightness', 1.0) - 1.0) * 100
        shorts = settings.get('shorts', False)
        filt = settings.get('filter', 'Canny Edge')
        palette = settings.get('palette', 'Neon Tokyo')
        params = settings.get('params', {})

        if mirror:
            frame = cv2.flip(frame, 1)
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        fx_frame = self.apply_visual_effect(adjusted.copy(), filt, palette, params)
        if shorts:
            fx_frame = self.apply_shorts_mode(fx_frame)
        return fx_frame