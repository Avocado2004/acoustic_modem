#!/usr/bin/env python3
"""
Тест записи с микрофона.
Записывает 5 секунд с микрофона и сохраняет в WAV.
"""

import numpy as np
import time
from scipy.io import wavfile
from data_source import MicrophoneDataSource

fs = 48000
duration = 5  # секунд

print(f"Запись {duration} секунд с микрофона...")

# Создаём источник данных
mic = MicrophoneDataSource(fs=fs, channels=1, chunk=2048)
mic.start()

# Ждём и собираем данные
chunks = []
start = time.time()
while time.time() - start < duration:
    snapshot = mic.read_snapshot()
    if len(snapshot) > 0:
        chunks.append(snapshot.copy())
    time.sleep(0.01)

mic.stop()

# Объединяем данные
if chunks:
    data = np.concatenate(chunks)
    print(f"Записано {len(data)} сэмплов ({len(data)/fs:.2f} сек)")
    print(f"RMS: {np.sqrt(np.mean(data**2)):.6f}")
    print(f"Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")
    
    # Сохраняем в WAV
    wavfile.write("mic_test.wav", fs, (data * 32767).astype(np.int16))
    print("Сохранено в mic_test.wav")
else:
    print("Нет данных с микрофона!")
