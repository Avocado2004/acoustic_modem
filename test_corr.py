#!/usr/bin/env python3
"""
Тест корреляции: ищем преамбулу в записанном с микрофона файле.
"""

import numpy as np
from scipy.io import wavfile
from modem_config import fs, Nfft, Ncp, Nsub
from modem_modulation import build_preamble
from signal_utils import fftconvolve

# Загружаем записанный файл
print("Загрузка записанного файла...")
fs_wav, sig = wavfile.read("mic_recorded.wav")
print(f"Загружено: {len(sig)} сэмплов, SR={fs_wav}")

# Нормализуем
sig_float = sig.astype(np.float64) / np.iinfo(sig.dtype).max
print(f"RMS: {np.sqrt(np.mean(sig_float**2)):.6f}")

# Строим преамбулу
preamble_td = build_preamble()
print(f"Преамбула: {len(preamble_td)} сэмплов")

# Вычисляем корреляцию
print("Вычисление корреляции...")
corr = fftconvolve(sig_float, preamble_td[::-1], mode='valid')
norm_corr = np.abs(corr) / np.sqrt(np.sum(preamble_td**2))

# Находим пик
peak_idx = np.argmax(norm_corr)
peak_val = norm_corr[peak_idx]
print(f"Пик корреляции: idx={peak_idx}, val={peak_val:.6f}")
print(f"Время пика: {peak_idx/fs:.3f} сек")

# Статистика
print(f"Средняя корреляция: {np.mean(norm_corr):.6f}")
print(f"Std корреляция: {np.std(norm_corr):.6f}")
print(f"Максимальная корреляция: {np.max(norm_corr):.6f}")

# Порог
threshold = 0.35
print(f"Порог: {threshold}")
print(f"Пик > порога: {peak_val > threshold}")
