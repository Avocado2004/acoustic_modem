#!/usr/bin/env python3
"""
Тест: воспроизведение сигнала через динамики + запись с микрофона.
Сигнал усилен в 10 раз.
"""

import numpy as np
import time
import threading
from scipy.io import wavfile
from data_source import MicrophoneDataSource
from audio_backend import get_audio_backend
from modem_config import fs
from modem_tx import transmit_text
import random
import string
import io
from contextlib import redirect_stdout

# Генерируем сигнал
print("Генерация сигнала...")
random.seed(42)
test_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(100))

f = io.StringIO()
with redirect_stdout(f):
    transmit_text(test_text)

# Загружаем WAV
fs_wav, sig = wavfile.read("ofdm_acoustic_tx_with_noise.wav")
sig_float = sig.astype(np.float64) / np.iinfo(sig.dtype).max
print(f"Загружен WAV: {len(sig_float)} сэмплов, RMS={np.sqrt(np.mean(sig_float**2)):.6f}")

# Усиливаем сигнал в 10 раз
sig_boosted = sig_float * 10.0
# Ограничиваем до [-1, 1]
sig_boosted = np.clip(sig_boosted, -1.0, 1.0)
print(f"После усиления: RMS={np.sqrt(np.mean(sig_boosted**2)):.6f}")

# Запускаем микрофон
print("\nЗапуск микрофона...")
mic = MicrophoneDataSource(fs=fs, channels=1, chunk=2048)
mic.start()

# Ждём 1 сек
time.sleep(1)

# Воспроизводим сигнал
print("Воспроизведение сигнала через динамики...")
audio = get_audio_backend()

# Запись в отдельном потоке
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
audio.play(sig_boosted.astype(np.float32), samplerate=fs)

# Ждём завершения
time.sleep(len(sig_boosted) / fs + 2)

# Останавливаем запись
recording = False
time.sleep(0.5)
mic.stop()

# Объединяем
if recorded_chunks:
    recorded = np.concatenate(recorded_chunks)
    print(f"\nЗаписано {len(recorded)} сэмплов ({len(recorded)/fs:.2f} сек)")
    print(f"RMS: {np.sqrt(np.mean(recorded**2)):.6f}")
    print(f"Min: {np.min(recorded):.6f}, Max: {np.max(recorded):.6f}")
    
    # Сохраняем
    wavfile.write("mic_recorded2.wav", fs, (recorded * 32767).astype(np.int16))
    print("Сохранено в mic_recorded2.wav")
else:
    print("Нет данных!")
