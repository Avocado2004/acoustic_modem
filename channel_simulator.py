"""
channel_simulator.py

Модуль для симуляции различних помех і искажень в акустичному каналі зв'язку.
Використовується для тестування модема (наприклад, test_modem_simple.py) без зміни його коду.
"""

import numpy as np
from typing import Optional, List


class ChannelSimulator:
    """
    Клас для симуляції каналу зв'язку.
    Дозволяє додавати шум, багатопроменевість, джитер і зсув фаз.
    """
    
    def __init__(self, fs: int = 48000, seed: Optional[int] = None):
        """
        Ініціалізація симулятора каналу.
        
        Параметри:
        - fs: частота дискретизації
        - seed: seed для генератора випадкових чисел
        """
        self.fs = fs
        self.rng = np.random.RandomState(seed)
        # Зберігаємо згенеровану АЧХ для забезпечення постійності протягом сеансу
        self._current_freq_response = None
        # Зберігаємо параметри останнього виклику для відлагодження
        self._last_distance_m = None
        self._last_percent = None
        
    def reset_channel_state(self):
        """
        Скидає збережену АЧХ при ініціалізації нового сеансу.
        Викликається перед початком нової симуляції.
        """
        self._current_freq_response = None
        self._last_distance_m = None
        self._last_percent = None
        print("[CHANNEL] Стан каналу скинуто")
        
    def add_awgn(self, sig: np.ndarray, snr_db: float) -> np.ndarray:
        """Додає білий гауссівський шум (AWGN) до сигналу."""
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
        """Застосовує модель багатопроменевого поширення (Multipath)."""
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
        """Вносить зсув частоти (Доплерівський зсув)."""
        if offset_hz == 0:
            return sig.copy()
            
        t = np.arange(len(sig)) / float(self.fs)
        return np.real(sig * np.exp(1j * 2.0 * np.pi * offset_hz * t))

    def apply_phase_jitter(self, sig: np.ndarray, jitter_std_rad: float) -> np.ndarray:
        """Вносить випадковий шум фази (Phase Jitter)."""
        if jitter_std_rad == 0:
            return sig.copy()
            
        phase_noise = self.rng.normal(0, jitter_std_rad, len(sig))
        return sig * np.exp(1j * phase_noise)

    def apply_sample_jitter(self, sig: np.ndarray, jitter_ms: float) -> np.ndarray:
        """Симулює джиттер семплів (нестабільність тактового генератора)."""
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
        """Симулює замирання (Fading)."""
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
        """Застосовує жорстке обмеження (hard clipping) сигналу."""
        return np.clip(sig, -threshold, threshold)

    def apply_soft_clipping(self, sig: np.ndarray, threshold: float, 
                           clip_type: str = 'tanh') -> np.ndarray:
        """Застосовує м'яке обмеження (soft clipping) сигналу."""
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
        """Застосовує нелінійні спотворення (distortion)."""
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
        """Симулює компресію амплітуди (audio compressor)."""
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
                                       percent: float = 0.0,
                                       f_low: float = 300.0, 
                                       f_high: float = 5000.0,
                                       smoothness: int = 15) -> np.ndarray:
        """
        Застосовує випадкову частотну характеристику (АЧХ) каналу в заданій смузі частот.
        
        Параметри:
        - sig: вхідний сигнал
        - percent: відсоток спотворень (0-100), керує інтенсивність АЧХ
        - f_low: нижня межа смуги частот (Гц)
        - f_high: верхня межа смуги частот (Гц)
        - smoothness: ступінь згладжування АЧХ
        
        Повертає:
        - сигнал з застосованою АЧХ
        """
        n = len(sig)
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        spec = np.fft.rfft(sig)
        
        # Якщо percent = 0, повертаємо сигнал без змін (рівна АЧХ)
        if percent == 0.0:
            print("[CHANNEL] АЧХ рівна (percent=0%)")
            return sig.copy()
        
        # Генеруємо АЧХ один раз за сеанс, якщо ще не згенерована
        if self._current_freq_response is None or self._last_percent != percent:
            print(f"[CHANNEL] Генерація нової АЧХ для percent={percent}%...")
            
            # Масштабуємо діапазон АЧХ від 0% (рівна) до 100% (±12 дБ)
            max_db_range = 12.0  # максимальне відхилення в дБ
            db_range = (percent / 100.0) * max_db_range
            
            mask = (freqs >= f_low) & (freqs <= f_high)
            random_response_db = np.zeros_like(freqs, dtype=np.float64)
            
            if np.any(mask):
                # Генеруємо випадкову АЧХ в дБ в смузі частот
                raw_random_db = self.rng.uniform(-db_range, db_range, size=np.sum(mask))
                
                # Згладжування для уникнення різких стрибків
                if smoothness > 1 and len(raw_random_db) > smoothness:
                    kernel = np.ones(smoothness) / smoothness
                    raw_random_db = np.convolve(raw_random_db, kernel, mode='same')
                
                random_response_db[mask] = raw_random_db
            
            # Переведення з дБ у лінійний масштаб (для амплітуди)
            random_response_linear = 10 ** (random_response_db / 20.0)
            
            # Зберігаємо згенеровану АЧХ для постійності в сеансі
            self._current_freq_response = random_response_linear
            self._last_percent = percent
            
            # Відлагоджувальний вивід
            if np.any(mask):
                db_values = random_response_db[mask]
                print(f"[CHANNEL] АЧХ згенерована: діапазон {np.min(db_values):.2f}..{np.max(db_values):.2f} дБ, "
                      f"середнє {np.mean(db_values):.2f} дБ")
        else:
            print(f"[CHANNEL] Використовуємо збережену АЧХ (percent={percent}%)")
        
        # Застосовуємо АЧХ до спектру
        spec_modified = spec * self._current_freq_response
        return np.fft.irfft(spec_modified, n=n)

    def apply_distance_attenuation(self, sig: np.ndarray,
                                  percent: float = 0.0,
                                  ref_distance: float = 1.0,
                                  air_absorption_db_per_m: float = 0.01) -> np.ndarray:
        """
        Симулює затухання сигналу при віддаленні динаміка від мікрофона.
        Зменшення рівня сигналу відбувається пропорційно 1/distance.
        Високі частоти затухають сильніше низьких.
        
        Параметри:
        - sig: вхідний сигнал
        - percent: відсоток спотворень (0-100)
                  0% -> відстань 1 м (практично немає затухання)
                  100% -> відстань 5 м (максимальне затухання)
        - ref_distance: опорна відстань (м)
        - air_absorption_db_per_m: поглинання звуку повітрям (дБ/м)
        
        Повертає:
        - сигнал з застосованим затуханням
        """
        # Розрахунок відстані від відсотка (1 м при 0%, 5 м при 100%)
        distance_m = 1.0 + (5.0 - 1.0) * (percent / 100.0)
        
        # При 0% просто повертаємо копію сигналу (немає затухання)
        if percent == 0.0:
            print(f"[CHANNEL] Затухання: відстань={distance_m:.2f} м, без затухання (0%)")
            return sig.copy()
        
        # Геометричне затухання: амплітуда зменшується пропорційно 1/distance
        geom_attenuation = (ref_distance / distance_m) ** 2
        
        n = len(sig)
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        spec = np.fft.rfft(sig)
        
        # Частотно-залежне затухання: високі частоти затухають сильніше
        # Масштабуємо сили ефекту від 0 (при 0%) до максимуму (при 100%)
        alpha = 4.0 * (percent / 100.0)  # сила частотної залежності
        f_norm = freqs / (self.fs / 2)  # нормалізована частота [0, 1]
        # Використовуємо монотонно зростаючу функцію від частоти
        # freq_attenuation = 1.0 + alpha * f_norm (високі частоти мають більший коефіцієнт)
        freq_attenuation = 1.0 + alpha * f_norm
        freq_attenuation = np.clip(freq_attenuation, 1.0, 10.0)
        
        # Масштабуємо частотне затухання відстанню
        distance_factor = min(distance_m / 5.0, 10.0)
        freq_attenuation = 1.0 + (freq_attenuation - 1.0) * distance_factor
        
        # Підсумкове затухання у частотній області
        # Високі частоти мають більший freq_attenuation, тому ділення зменшує їх сильніше
        spec_attenuated = spec / freq_attenuation
        
        # Зворотне перетворення у часову область з урахуванням геометричного затухання
        result = np.fft.irfft(spec_attenuated, n=n) * geom_attenuation
        
        # Поглинання звуку в повітрі
        air_attenuation_linear = 10 ** (-air_absorption_db_per_m * distance_m / 20.0)
        result *= air_attenuation_linear
        
        # Відлагоджувальний вивід
        print(f"[CHANNEL] Затухання: відстань={distance_m:.2f} м, geom_factor={geom_attenuation:.4f}")
        print(f"[CHANNEL]   Частотне затухання: alpha={alpha:.2f}, макс. freq_factor={np.max(freq_attenuation):.2f}")
        print(f"[CHANNEL]   Рівень сигналу зменшено в {1.0/geom_attenuation:.1f} раз")
        
        return result.astype(sig.dtype)

    def apply_impulse_noise(self, sig: np.ndarray,
                            impulse_level: float = 1.0,
                            min_spacing_ms: float = 50.0,
                            max_spacing_ms: float = 200.0,
                            impulse_duration_ms: float = 1.0) -> np.ndarray:
        """
        Додає імпульсні перешкоди (короткі імпульси білого шуму).
        
        Параметри:
        - impulse_level: амплітуда імпульсів шуму (середньоквадратичне відхилення)
        - min_spacing_ms: мінімальна відстань між імпульсами в мілісекундах
        - max_spacing_ms: максимальна відстань між імпульсами в мілісекундах
        - impulse_duration_ms: тривалість кожного імпульсу в мілісекундах
        """
        result = sig.copy().astype(np.float64)
        n_samples = len(sig)
        
        min_spacing_samples = int(min_spacing_ms / 1000.0 * self.fs)
        max_spacing_samples = int(max_spacing_ms / 1000.0 * self.fs)
        impulse_duration_samples = max(1, int(impulse_duration_ms / 1000.0 * self.fs))
        
        pos = 0
        while pos < n_samples:
            # Генеруємо імпульс
            end_pos = min(pos + impulse_duration_samples, n_samples)
            noise = self.rng.normal(0, impulse_level, end_pos - pos)
            result[pos:end_pos] += noise
            
            # Наступний інтервал
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
    """Універсальна функція для застосування набору спотворень каналу."""
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
    print("Тестування симулятора каналу...")
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
    print("Симулятор каналу готовий до використання.")
