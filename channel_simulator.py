"""
channel_simulator.py

Модуль для симуляции различных помех и искажений в акустическом канале связи.
Используется для тестирования модема (например, test_modem_simple.py) без изменения его кода.
"""

import numpy as np
from typing import Optional, List


class ChannelSimulator:
    """
    Класс для симуляции канала связи.
    Позволяет добавлять шум, многолучевость, джиттер и сдвиг фаз.
    """
    
    def __init__(self, fs: int = 48000, seed: Optional[int] = None):
        self.fs = fs
        self.rng = np.random.RandomState(seed)
        
    def add_awgn(self, sig: np.ndarray, snr_db: float) -> np.ndarray:
        """Добавляет белый гауссовский шум (AWGN) к сигналу."""
        if snr_db is None or snr_db <= -np.inf:
            return sig.copy()
        
        sig_power = np.mean(np.abs(sig) ** 2)
        if sig_power < 1e-15:
            return sig.copy() + self.rng.normal(0, 1e-6, len(sig))
            
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = np.sqrt(noise_power)
        
        noise = self.rng.normal(0, noise_std, len(sig))
        return sig + noise.astype(sig.dtype)

    def apply_multipath(self, sig: np.ndarray, 
                        delays_ms: List[float], 
                        gains: List[float]) -> np.ndarray:
        """Применяет модель многолучевого распространения (Multipath)."""
        if len(delays_ms) != len(gains):
            raise ValueError("delays_ms and gains must have the same length")
            
        out = np.zeros_like(sig, dtype=np.float64)
        for delay_ms, gain in zip(delays_ms, gains):
            delay_samples = int(delay_ms / 1000.0 * self.fs)
            if delay_samples == 0:
                out += gain * sig
            elif delay_samples < len(sig):
                shifted = np.zeros_like(sig)
                shifted[delay_samples:] = sig[:-delay_samples]
                out += gain * shifted
        return out

    def apply_frequency_offset(self, sig: np.ndarray, offset_hz: float) -> np.ndarray:
        """Вносит сдвиг частоты (Доплеровский сдвиг)."""
        if offset_hz == 0:
            return sig.copy()
            
        t = np.arange(len(sig)) / float(self.fs)
        return np.real(sig * np.exp(1j * 2.0 * np.pi * offset_hz * t))

    def apply_phase_jitter(self, sig: np.ndarray, jitter_std_rad: float) -> np.ndarray:
        """Вносит случайный шум фазы (Phase Jitter)."""
        if jitter_std_rad == 0:
            return sig.copy()
            
        phase_noise = self.rng.normal(0, jitter_std_rad, len(sig))
        return sig * np.exp(1j * phase_noise)

    def apply_sample_jitter(self, sig: np.ndarray, jitter_ms: float) -> np.ndarray:
        """Симулирует джиттер семплов (нестабильность тактового генератора)."""
        if jitter_ms <= 0:
            return sig.copy()
            
        n_samples = len(sig)
        t_ideal = np.arange(n_samples) / float(self.fs)
        
        jitter_samples = (jitter_ms / 1000.0) * self.fs
        jitter_noise = self.rng.uniform(-jitter_samples, jitter_samples, n_samples)
        t_actual = t_ideal + jitter_noise / self.fs
        
        return np.interp(t_ideal, t_actual, sig).astype(sig.dtype)

    def apply_fading(self, sig: np.ndarray, 
                     fader_type: str = 'rayleigh', 
                     block_size: int = 512) -> np.ndarray:
        """Симулирует замирания (Fading)."""
        out = np.zeros_like(sig)
        num_blocks = int(np.ceil(len(sig) / block_size))
        
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, len(sig))
            
            if fader_type == 'rayleigh':
                gain = self.rng.rayleigh(1.0)
            else: # rician
                gain = np.abs(self.rng.normal(1.0, 0.5) + 1j * self.rng.normal(0, 0.5))
                
            out[start:end] = sig[start:end] * gain
            
        return out

    def apply_hard_clipping(self, sig: np.ndarray, threshold: float) -> np.ndarray:
        """Применяет жесткое ограничение (hard clipping) сигнала."""
        return np.clip(sig, -threshold, threshold)

    def apply_soft_clipping(self, sig: np.ndarray, threshold: float, 
                           clip_type: str = 'tanh') -> np.ndarray:
        """Применяет мягкое ограничение (soft clipping) сигнала."""
        if clip_type == 'tanh':
            return threshold * np.tanh(sig / threshold)
        elif clip_type == 'cubic':
            normalized = sig / threshold
            clipped = np.where(np.abs(normalized) < 1.0,
                             normalized - normalized**3 / 3.0,
                             np.sign(normalized))
            return threshold * clipped
        elif clip_type == 'arctan':
            return (2 * threshold / np.pi) * np.arctan(sig / threshold)
        else:
            raise ValueError(f"Unknown clip_type: {clip_type}")

    def apply_distortion(self, sig: np.ndarray, 
                        drive: float = 1.0, 
                        type: str = 'even') -> np.ndarray:
        """Применяет нелинейные искажения (distortion)."""
        max_val = np.max(np.abs(sig))
        if max_val < 1e-15:
            return sig.copy()
        normalized = sig / max_val
        
        if type == 'even':
            distorted = normalized + drive * normalized**2 / 2.0
        elif type == 'odd':
            distorted = np.tanh(normalized * (1.0 + drive))
        elif type == 'full':
            distorted = normalized + drive * (normalized**2 + normalized**3) / 2.0
        else:
            raise ValueError(f"Unknown distortion type: {type}")
            
        return distorted * max_val

    def apply_amplitude_compression(self, sig: np.ndarray, 
                                   threshold: float = 0.5,
                                   ratio: float = 4.0,
                                   attack_ms: float = 1.0,
                                   release_ms: float = 10.0) -> np.ndarray:
        """Симулирует компрессию амплитуды (audio compressor)."""
        attack_coeff = np.exp(-1.0 / (attack_ms * self.fs / 1000.0))
        release_coeff = np.exp(-1.0 / (release_ms * self.fs / 1000.0))
        
        envelope = np.abs(sig)
        gain = np.zeros_like(envelope)
        gain[0] = envelope[0]
        for i in range(1, len(envelope)):
            if envelope[i] > gain[i-1]:
                gain[i] = attack_coeff * gain[i-1] + (1 - attack_coeff) * envelope[i]
            else:
                gain[i] = release_coeff * gain[i-1] + (1 - release_coeff) * envelope[i]
        
        compressed = np.zeros_like(sig)
        for i in range(len(sig)):
            if gain[i] > threshold:
                reduction = (gain[i] - threshold) / (gain[i] * ratio) + (1 - 1/ratio)
                compressed[i] = sig[i] * reduction
            else:
                compressed[i] = sig[i]
                
        return compressed

    def apply_random_frequency_response(self, sig: np.ndarray, 
                                       f_low: float = 300.0, 
                                       f_high: float = 5000.0,
                                       smoothness: int = 15) -> np.ndarray:
        """Применяет случайную частотную характеристику канала в заданной полосе частот."""
        n = len(sig)
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        spec = np.fft.rfft(sig)
        
        mask = (freqs >= f_low) & (freqs <= f_high)
        random_response = np.ones_like(freqs, dtype=np.float64)
        
        if np.any(mask):
            raw_random = self.rng.uniform(0.3, 1.7, size=np.sum(mask))
            if smoothness > 1 and len(raw_random) > smoothness:
                kernel = np.ones(smoothness) / smoothness
                raw_random = np.convolve(raw_random, kernel, mode='same')
            random_response[mask] = raw_random
        
        spec_modified = spec * random_response
        return np.fft.irfft(spec_modified, n=n)

    def apply_distance_attenuation(self, sig: np.ndarray,
                                  distance_m: float,
                                  ref_distance: float = 1.0,
                                  air_absorption_db_per_m: float = 0.01) -> np.ndarray:
        """Симулирует затухание сигнала при удалении динамика от микрофона."""
        if distance_m <= 0:
            return sig.copy()
            
        geom_attenuation = (ref_distance / distance_m) ** 2
        
        n = len(sig)
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        spec = np.fft.rfft(sig)
        
        f_norm = freqs / (self.fs / 2)
        f0 = 0.5
        base_curve = (f_norm - f0)**2
        base_strength = 4.0
        freq_attenuation = 1.0 + base_strength * base_curve
        freq_attenuation = np.clip(freq_attenuation, 1.0, 10.0)
        
        distance_factor = min(distance_m / 5.0, 10.0)
        freq_attenuation = 1.0 + (freq_attenuation - 1.0) * distance_factor
        
        spec_attenuated = spec / freq_attenuation
        result = np.fft.irfft(spec_attenuated, n=n) * geom_attenuation
        
        air_attenuation_linear = 10 ** (-air_absorption_db_per_m * distance_m / 20.0)
        result *= air_attenuation_linear
        
        return result.astype(sig.dtype)

    def apply_impulse_noise(self, sig: np.ndarray,
                            impulse_level: float = 1.0,
                            min_spacing_ms: float = 50.0,
                            max_spacing_ms: float = 200.0,
                            impulse_duration_ms: float = 1.0) -> np.ndarray:
        """
        Добавляет импульсные помехи (короткие импульсы белого шума).
        
        Параметры:
        - impulse_level: амплитуда импульсов шума (среднеквадратичное отклонение)
        - min_spacing_ms: минимальное расстояние между импульсами в миллисекундах
        - max_spacing_ms: максимальное расстояние между импульсами в миллисекундах
        - impulse_duration_ms: длительность каждого импульса в миллисекундах
        """
        result = sig.copy().astype(np.float64)
        n_samples = len(sig)
        
        min_spacing_samples = int(min_spacing_ms / 1000.0 * self.fs)
        max_spacing_samples = int(max_spacing_ms / 1000.0 * self.fs)
        impulse_duration_samples = max(1, int(impulse_duration_ms / 1000.0 * self.fs))
        
        pos = 0
        while pos < n_samples:
            # Генерируем импульс
            end_pos = min(pos + impulse_duration_samples, n_samples)
            noise = self.rng.normal(0, impulse_level, end_pos - pos)
            result[pos:end_pos] += noise
            
            # Следующий интервал
            if min_spacing_samples >= max_spacing_samples:
                next_interval = min_spacing_samples
            else:
                next_interval = self.rng.randint(min_spacing_samples, max_spacing_samples)
                
            pos += next_interval
        
        return result.astype(sig.dtype)


def simulate_channel(sig: np.ndarray, 
                     fs: int = 48000, 
                     snr_db: Optional[float] = None,
                     freq_offset_hz: float = 0.0,
                     multipath_delays: Optional[List[float]] = None,
                     multipath_gains: Optional[List[float]] = None,
                     phase_jitter_rad: float = 0.0,
                     seed: Optional[int] = None) -> np.ndarray:
    """Универсальная функция для применения набора искажений канала."""
    sim = ChannelSimulator(fs=fs, seed=seed)
    result = sig.copy()
    
    if freq_offset_hz != 0:
        result = sim.apply_frequency_offset(result, freq_offset_hz)
        
    if multipath_delays and multipath_gains:
        result = sim.apply_multipath(result, multipath_delays, multipath_gains)
        
    if phase_jitter_rad > 0:
        if not np.iscomplexobj(result):
            result = result.astype(np.complex128)
        result = sim.apply_phase_jitter(result, phase_jitter_rad)
        result = np.real(result)
        
    if snr_db is not None:
        result = sim.add_awgn(result, snr_db)
        
    return result.astype(sig.dtype)


if __name__ == "__main__":
    print("Тестирование симулятора канала...")
    fs = 48000
    t = np.arange(0, 1.0, 1.0/fs)
    test_sig = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    
    distorted = simulate_channel(
        test_sig,
        fs=fs,
        snr_db=20,
        freq_offset_hz=5.0,
        multipath_delays=[0.0, 2.0],
        multipath_gains=[0.7, 0.3],
        seed=42
    )
    
    print(f"Original RMS: {np.sqrt(np.mean(test_sig**2)):.4f}")
    print(f"Distorted RMS: {np.sqrt(np.mean(distorted**2)):.4f}")
    print("Симулятор канала готов к использованию.")