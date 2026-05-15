#!/usr/bin/env python3
"""
Тест: воспроизведение сигнала через динамики + запись с микрофона.
"""

import numpy as np
import time
import threading
from scipy.io import wavfile
from data_source import MicrophoneDataSource
from audio_backend import get_audio_backend

fs = 48000

# Загружаем тестовый сигнал
print("Загрузка тестового сигнала...")
from modem_tx import transmit_text
import random
import string

random.seed(42)
test_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(100))

# Генерируем сигнал (без воспроизведения)
print("Генерация сигнала...")
import io
import sys
from contextlib import redirect_stdout

# Перенаправляем stdout, чтобы не засорять вывод
f = io.StringIO()
with redirect_stdout(f):
    transmit_text(test_text)

# Загружаем сгенерированный WAV
fs_wav, sig = wavfile.read("ofdm_acoustic_tx_with_noise.wav")
print(f"Загружен WAV: {len(sig)} сэмплов, SR={fs_wav}")
print(f"RMS сигнала: {np.sqrt(np.mean(sig.astype(float)**2)):.6f}")

# Нормализуем сигнал
sig_float = sig.astype(np.float64) / np.iinfo(sig.dtype).max
print(f"RMS после нормализации: {np.sqrt(np.mean(sig_float**2)):.6f}")

# Запускаем микрофон
print("\nЗапуск микрофона...")
mic = MicrophoneDataSource(fs=fs, channels=1, chunk=2048)
mic.start()

# Ждём 1 сек для накопления буфера
time.sleep(1)

# Воспроизводим сигнал
print("Воспроизведение сигнала через динамики...")
audio = get_audio_backend()

# Запускаем запись в отдельном потоке
recorded_chunks = []
recording = True

def record_thread():
    while recording:
        snapshot = mic.read_snapshot()
        if len(snapshot) > 0:
            recorded_chunks.append(snapshot.copy())
        time.sleep(0.01)

t = threading.Thread(target=record_thread, daemon=True)
t.start()

# Воспроизводим
audio.play(sig_float, samplerate=fs)

# Ждём завершения воспроизведения
time.sleep(len(sig_float) / fs + 1)

# Останавливаем запись
recording = False
time.sleep(0.5)
mic.stop()

# Объединяем записанные данные
if recorded_chunks:
    recorded = np.concatenate(recorded_chunks)
    print(f"\nЗаписано {len(recorded)} сэмплов ({len(recorded)/fs:.2f} сек)")
    print(f"RMS: {np.sqrt(np.mean(recorded**2)):.6f}")
    print(f"Min: {np.min(recorded):.6f}, Max: {np.max(recorded):.6f}")
    
    # Сохраняем
    wavfile.write("mic_recorded.wav", fs, (recorded * 32767).astype(np.int16))
    print("Сохранено в mic_recorded.wav")
else:
    print("Нет данных!")
