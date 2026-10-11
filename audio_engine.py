import numpy as np
import scipy.signal as scipy_signal
from numba import jit
from pedalboard import (Pedalboard, Gain, Limiter, NoiseGate, 
                        HighpassFilter, LowShelfFilter, PeakFilter, HighShelfFilter)
import math

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

@jit(nopython=True, nogil=True)
def _obs_speex_core(power, noise_est_init, max_supp, noise_rate):
    n_freqs, n_frames = power.shape
    gain_mask = np.ones_like(power)
    noise_est = noise_est_init.copy()
    prev_gain = np.ones(n_freqs, dtype=np.float32)
    vad_threshold = 2.5
    gain_smooth = 0.9
    for i in range(n_frames):
        for f in range(n_freqs):
            ratio = power[f, i] / (noise_est[f] + 1e-10)
            if ratio < vad_threshold:
                noise_est[f] = (1.0 - noise_rate) * noise_est[f] + noise_rate * power[f, i]
            else:
                noise_est[f] = (1.0 - 1e-4) * noise_est[f] + 1e-4 * power[f, i]
        signal_pwr = np.maximum(power[:, i] - noise_est, 0.0)
        gain = signal_pwr / (signal_pwr + noise_est + 1e-10)
        gain = np.maximum(gain, max_supp)
        gain = gain_smooth * prev_gain + (1.0 - gain_smooth) * gain
        prev_gain = gain
        gain_mask[:, i] = gain
    return gain_mask

@jit(nopython=True, nogil=True)
def _obs_rnnoise_core(power, noise_est_init, max_supp):
    n_freqs, n_frames = power.shape
    gain_mask = np.ones_like(power)
    noise_est = noise_est_init.copy()
    prev_gain = np.ones(n_freqs, dtype=np.float32)
    vad_state = 0.0
    vad_alpha = 0.95
    vad_onset = 3.0
    vad_offset = 1.5
    gain_smooth = 0.92
    for i in range(n_frames):
        total_energy = 0.0
        noise_energy = 0.0
        for f in range(n_freqs):
            total_energy += power[f, i]
            noise_energy += noise_est[f]
        snr_estimate = total_energy / (noise_energy + 1e-10)
        if snr_estimate > vad_onset:
            vad_state = vad_alpha * vad_state + (1.0 - vad_alpha) * 1.0
        elif snr_estimate > vad_offset:
            vad_state = vad_alpha * vad_state
        else:
            vad_state = vad_alpha * vad_state + (1.0 - vad_alpha) * 0.0
        noise_rate = 0.02 * (1.0 - vad_state) + 0.0001 * vad_state
        for f in range(n_freqs):
            noise_est[f] = (1.0 - noise_rate) * noise_est[f] + noise_rate * power[f, i]
        oversub = 1.0 + (1.0 - vad_state) * 2.0
        signal_pwr = np.maximum(power[:, i] - noise_est, 0.0)
        gain = signal_pwr / (signal_pwr + oversub * noise_est + 1e-10)
        gain = np.maximum(gain, max_supp)
        gain = gain_smooth * prev_gain + (1.0 - gain_smooth) * gain
        prev_gain = gain
        gain_mask[:, i] = gain
    return gain_mask

@jit(nopython=True, nogil=True)
def _generate_rain_drops_numba(n_samples, n_drops, sr, freq_min, freq_max, len_min, len_max, amp, attack_ms, decay_rate):
    drops = np.zeros(n_samples, dtype=np.float32)
    indices = np.random.randint(0, n_samples, n_drops)
    for i in range(n_drops):
        idx = indices[i]
        l = int(sr * np.random.uniform(len_min, len_max))
        if idx + l < n_samples:
            t = np.arange(l) / sr
            env = np.exp(-t * decay_rate)
            a_len = max(1, int(sr * attack_ms / 1000.0))
            if a_len < l:
                env[:a_len] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(a_len) / a_len))
            freq = np.random.uniform(freq_min, freq_max)
            drops[idx:idx+l] += np.sin(2 * np.pi * freq * t) * env * amp
    return drops

@jit(nopython=True, nogil=True)
def _generate_tin_roof_drops_numba(n_samples, n_drops, sr, freq_min, freq_max, len_min, len_max, amp, attack_ms, decay_rate):
    drops = np.zeros(n_samples, dtype=np.float32)
    indices = np.random.randint(0, n_samples, n_drops)
    for i in range(n_drops):
        idx = indices[i]
        l = int(sr * np.random.uniform(len_min, len_max))
        if idx + l < n_samples:
            t = np.arange(l) / sr
            env = np.exp(-t * decay_rate)
            a_len = max(1, int(sr * attack_ms / 1000.0))
            if a_len < l:
                env[:a_len] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(a_len) / a_len))
            fund = np.random.uniform(freq_min, freq_max)
            sig = (np.sin(2 * np.pi * fund * t) + 0.5 * np.sin(2 * np.pi * fund * 2.7 * t) + 0.3 * np.sin(2 * np.pi * fund * 5.2 * t))
            drops[idx:idx+l] += sig * env * amp
    return drops

@jit(nopython=True, nogil=True)
def _generate_crackles_numba(n_samples, n_drops, sr, freq_min, freq_max, len_min, len_max, amp, attack_ms, decay_rate_min, decay_rate_max):
    crackles = np.zeros(n_samples, dtype=np.float32)
    indices = np.random.randint(0, n_samples, n_drops)
    for i in range(n_drops):
        idx = indices[i]
        l = int(sr * np.random.uniform(len_min, len_max))
        if idx + l < n_samples:
            t = np.arange(l) / sr
            decay_rate = np.random.uniform(decay_rate_min, decay_rate_max)
            env = np.exp(-t * decay_rate)
            a_len = max(1, int(sr * attack_ms / 1000.0))
            if a_len < l:
                env[:a_len] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(a_len) / a_len))
            freq = np.random.uniform(freq_min, freq_max)
            crackles[idx:idx+l] += np.sin(2 * np.pi * freq * t) * env * amp
    return crackles

class AudioEngine:
    def __init__(self):
        pass

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

    def _remove_dc(self, x):
        return (x - np.mean(x)).astype(np.float32)

    def _soft_clip(self, x, drive=1.2):
        return (np.tanh(x * drive) / np.tanh(drive)).astype(np.float32)

    def _smooth_sfx_bus(self, x, sr):
        if sr > 24000:
            b, a = scipy_signal.butter(2, 12000.0 / (sr / 2.0), btype='low')
            x = scipy_signal.lfilter(b, a, x).astype(np.float32)
        return self._remove_dc(x)

    def obs_noise_suppression(self, audio, sr, suppression_db=30.0, mode="Speex"):
        if len(audio) < sr * 0.1: return audio
        n_fft = 2048; hop = 512
        f, t, Zxx = scipy_signal.stft(audio, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, boundary='zeros', padded=True)
        mag = np.abs(Zxx).astype(np.float32)
        phase = np.angle(Zxx).astype(np.float32)
        power = mag ** 2
        init_frames = max(1, int(0.8 * sr / hop))
        noise_power_init = np.mean(power[:, :init_frames], axis=1).astype(np.float32)
        max_supp = 10.0 ** (-suppression_db / 20.0)
        if mode == "RNNoise":
            gain_mask = _obs_rnnoise_core(power, noise_power_init, max_supp)
        else:
            noise_rate = 0.01
            gain_mask = _obs_speex_core(power, noise_power_init, max_supp, noise_rate)
        cleaned_mag = mag * gain_mask
        cleaned_Zxx = cleaned_mag * np.exp(1j * phase)
        _, cleaned = scipy_signal.istft(cleaned_Zxx, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, boundary=True)
        if len(cleaned) > len(audio): cleaned = cleaned[:len(audio)]
        elif len(cleaned) < len(audio): cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))
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
        return (np.tanh(audio * drive) / np.tanh(drive)).astype(np.float32)

    def apply_transient_shaper(self, audio, sr, punch_pct):
        if punch_pct <= 0: return audio
        env = np.abs(audio).astype(np.float32)
        attack_env = scipy_signal.lfilter(*scipy_signal.butter(2, 100.0/(sr/2.0)), env)
        sustain_env = scipy_signal.lfilter(*scipy_signal.butter(2, 10.0/(sr/2.0)), env)
        transient = np.maximum(0, attack_env - sustain_env)
        if transient.max() > 0: transient /= transient.max()
        return (audio + (audio * transient * (punch_pct / 100.0) * 1.5)).astype(np.float32)

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
            
            drops = _generate_rain_drops_numba(n_samples, density, sr, 2000.0, 6000.0, 0.002, 0.008, 0.3, 1.0, 400.0)
            
            rumble = np.random.randn(n_samples).astype(np.float32)
            b_lp, a_lp = scipy_signal.butter(2, 200 / (sr / 2), btype='low')
            rumble = scipy_signal.lfilter(b_lp, a_lp, rumble).astype(np.float32) * 0.15
            intensity = 0.7 if sound_type == "Light Rain" else 1.0
            audio = (wash * 0.5 + drops * 0.5 + rumble) * intensity
        elif sound_type == "Rain on Tin Roof":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [600 / (sr / 2), 10000 / (sr / 2)], btype='band')
            wash = scipy_signal.lfilter(b, a, white).astype(np.float32) * 0.3
            
            drops = _generate_tin_roof_drops_numba(n_samples, n_samples // 30, sr, 1500.0, 3500.0, 0.015, 0.025, 0.25, 1.5, 150.0)
            
            audio = wash + drops
        elif sound_type == "Thunderstorm":
            white = np.random.randn(n_samples).astype(np.float32)
            b, a = scipy_signal.butter(4, [400 / (sr / 2), 8000 / (sr / 2)], btype='band')
            wash = scipy_signal.lfilter(b, a, white).astype(np.float32)
            
            drops = _generate_rain_drops_numba(n_samples, n_samples // 25, sr, 2000.0, 6000.0, 0.002, 0.008, 0.3, 1.0, 400.0)
            
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
            
            crackles = _generate_crackles_numba(n_samples, n_samples // 25, sr, 3000.0, 8000.0, 0.005, 0.02, 0.4, 0.8, 80.0, 200.0)
            
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

    def process_audio_chain(self, audio_np, fs, s, is_live=False):
        if len(audio_np) == 0: 
            return np.zeros(0, dtype=np.float32)
            
        gained_audio = apply_gain_numba(audio_np, s['gain'])
        if s.get('noise_suppress', False) and len(gained_audio) > fs * 0.1:
            gained_audio = self.obs_noise_suppression(
                gained_audio, fs, 
                suppression_db=s['noise_suppress_amount'],
                mode=s.get('noise_suppress_mode', 'Speex')
            )

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
        
        processed_voice = self.apply_transient_shaper(processed_voice, fs, s.get('transient_punch', 0))
        processed_voice = self.apply_tape_saturation(processed_voice, s.get('tape_drive', 0))
        processed_voice = self.apply_deesser(processed_voice, fs, threshold_db=s['deess_thresh'], ratio=s['deess_ratio'])
        
        # SFX & Ambient sounds are skipped in live mode to save extreme amounts of CPU 
        # and prevent context switching / audio dropouts while monitoring.
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

        final_plugins = [
            Gain(gain_db=vol_db),
            HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-1.0),
            Limiter(threshold_db=-1.0, release_ms=100)
        ]
        return Pedalboard(final_plugins)(processed_voice, fs)