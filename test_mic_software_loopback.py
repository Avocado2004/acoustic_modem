#!/usr/bin/env python3
"""
Тест режима M с программным loopback.

Вместо акустической передачи (динамик → микрофон),
используем программный loopback: TX сигнал подаётся
напрямую в RX процессор через pipe/queue.

Это позволяет проверить что код приёма работает корректно
без влияния акустического канала.
"""

import sys
import os
import time
import numpy as np
from scipy.io import wavfile

project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.insert(0, project_dir)

from data_source import MicrophoneChunkDataSource
from modem_config import fs, init_phases, set_modulation
from signal_processor import receive_from_microphone_chunked


class SoftwareLoopbackSource(MicrophoneChunkDataSource):
    """
    Программный loopback — подаёт TX сигнал как будто он записан с микрофона.
    Добавляет реалистичный шум и затухание.
    """

    def __init__(self, wav_path: str, attenuation: float = 0.3, noise_level: float = 0.001):
        """
        Параметры
        ----------
        wav_path : str
            Путь к WAV файлу с TX сигналом
        attenuation : float
            Коэффициент затухания (0.0 - 1.0)
        noise_level : float
            Уровень шума
        """
        self.fs = fs
        self.channels = 1
        self.chunk = 2048
        self.sleep_time = 0.001

        self._buffer_lock = None
        self._ring = None
        self._total_samples_in_buffer = 0
        self._samples_read = 0
        self._audio_backend = None
        self.active = False

        # Загружаем файл
        fs_wav, raw_data = wavfile.read(wav_path)
        if raw_data.ndim > 1:
            raw_data = raw_data[:, 0]

        if np.issubdtype(raw_data.dtype, np.integer):
            self._data = raw_data.astype(np.float64) / np.iinfo(raw_data.dtype).max
        else:
            self._data = raw_data.astype(np.float64)

        # Применяем затухание (как в реальном акустическом канале)
        self._data = self._data * attenuation

        # Добавляем шум
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(self._data))
            self._data = self._data + noise

        self._current_pos = 0
        self._total_samples = len(self._data)

        print(f"[SoftLoopback] Loaded: {self._total_samples} samples, "
              f"attenuation={attenuation}, noise_level={noise_level}, "
              f"RMS={np.sqrt(np.mean(self._data**2)):.6f}")

    def start(self):
        self.active = True
        print("[SoftLoopback] Started")

    def read_chunk(self):
        if self._current_pos >= self._total_samples:
            return np.array([], dtype=np.float64)

        end = min(self._current_pos + self.chunk, self._total_samples)
        chunk = self._data[self._current_pos:end].copy()
        self._current_pos = end
        self._samples_read = self._current_pos
        return chunk

    def has_more(self):
        return self._current_pos < self._total_samples

    def get_all_data(self):
        return self._data.copy()

    def get_samples_read(self):
        return self._samples_read

    def stop(self):
        self.active = False
        print("[SoftLoopback] Stopped")


def test_software_loopback():
    """Тест с программным loopback."""
    print("=" * 60)
    print("ТЕСТ РЕЖИМА M С ПРОГРАММНЫМ LOOPBACK")
    print("=" * 60)

    set_modulation("QPSK")
    init_phases()

    wav_path = "ofdm_acoustic_tx_with_noise.wav"
    if not os.path.exists(wav_path):
        print(f"Файл {wav_path} не найден!")
        return False

    # Тест 1: Высокое затухание (как реальный микрофон)
    print("\n--- Тест 1: attenuation=0.3, noise=0.001 ---")
    sim = SoftwareLoopbackSource(wav_path, attenuation=0.3, noise_level=0.001)

    import data_source
    original = data_source.MicrophoneChunkDataSource
    data_source.MicrophoneChunkDataSource = lambda *a, **kw: sim

    result = receive_from_microphone_chunked()

    data_source.MicrophoneChunkDataSource = original

    if result and os.path.exists("rx_text.txt"):
        with open("rx_text.txt", "r", encoding="utf-8") as f:
            received = f.read()
        print(f"Принято: '{received[:100]}'")
        if len(received) > 0:
            print("✅ Тест 1 пройден!")
        else:
            print("❌ Тест 1: пустой результат")
    else:
        print(f"❌ Тест 1: result={result}")

    # Тест 2: Среднее затухание
    print("\n--- Тест 2: attenuation=0.5, noise=0.0005 ---")
    sim2 = SoftwareLoopbackSource(wav_path, attenuation=0.5, noise_level=0.0005)
    data_source.MicrophoneChunkDataSource = lambda *a, **kw: sim2

    result2 = receive_from_microphone_chunked()

    data_source.MicrophoneChunkDataSource = original

    if result2 and os.path.exists("rx_text.txt"):
        with open("rx_text.txt", "r", encoding="utf-8") as f:
            received2 = f.read()
        print(f"Принято: '{received2[:100]}'")
        if len(received2) > 0:
            print("✅ Тест 2 пройден!")
        else:
            print("❌ Тест 2: пустой результат")
    else:
        print(f"❌ Тест 2: result={result2}")

    return result or result2


if __name__ == "__main__":
    try:
        success = test_software_loopback()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
