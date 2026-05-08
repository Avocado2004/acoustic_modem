#!/usr/bin/env python3
"""
Сравнение обычной передачи и чтения из файла с Loop-режимом.
Результаты должны быть идентичны.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Отключаем отладочный вывод AGC
import modem_config
modem_config.AGC_DEBUG = False

# Генерируем тестовый текст
test_text = "Hello World! This is a test message for comparing transmission modes."

# === Режим 1: Обычная передача ===
print("=" * 60)
print("РЕЖИМ 1: Обычная передача")
print("=" * 60)

from modem_tx import transmit_text
from modem_rx import receive_from_file

# Передаём текст
transmit_text(test_text)

# Принимаем из файла
print("\nПриём из файла...")
success1 = receive_from_file("ofdm_acoustic_tx_with_noise.wav")

# Читаем результат
received_text1 = ""
if os.path.exists("rx_text.txt"):
    with open("rx_text.txt", "r", encoding='utf-8') as f:
        received_text1 = f.read()

print(f"Обычная передача: RS_OK=?, текст='{received_text1[:50]}...'")

# === Режим 2: Loop (передача -> искажения -> приём) ===
print("\n" + "=" * 60)
print("РЕЖИМ 2: Loop (передача -> искажения 0% -> приём)")
print("=" * 60)

# Передаём текст заново
transmit_text(test_text)

# Применяем искажения (0% - сигнал не изменяется)
from scipy.io import wavfile
fs, sig = wavfile.read("ofdm_acoustic_tx_with_noise.wav")
sig_float = sig.astype(np.float64) / 32767.0

# Сохраняем без изменений
wavfile.write("ofdm_acoustic_tx_with_noise.wav", fs, sig)

# Принимаем из файла
print("\nПриём из файла (после искажений 0%)...")
# Удаляем старый файл результата
if os.path.exists("rx_text.txt"):
    os.remove("rx_text.txt")

success2 = receive_from_file("ofdm_acoustic_tx_with_noise.wav")

# Читаем результат
received_text2 = ""
if os.path.exists("rx_text.txt"):
    with open("rx_text.txt", "r", encoding='utf-8') as f:
        received_text2 = f.read()

print(f"Loop режим: RS_OK=?, текст='{received_text2[:50]}...'")

# === Сравнение ===
print("\n" + "=" * 60)
print("СРАВНЕНИЕ")
print("=" * 60)

if received_text1 == test_text:
    print("✅ Обычная передача: текст совпадает")
else:
    print(f"❌ Обычная передача: текст отличается")
    print(f"   Ожидается: '{test_text[:50]}...'")
    print(f"   Получено:  '{received_text1[:50]}...'")

if received_text2 == test_text:
    print("✅ Loop режим: текст совпадает")
else:
    print(f"❌ Loop режим: текст отличается")
    print(f"   Ожидается: '{test_text[:50]}...'")
    print(f"   Получено:  '{received_text2[:50]}...'")

if received_text1 == received_text2:
    print("✅ Результаты идентичны")
else:
    print("❌ Результаты отличаются!")
    print(f"   Обычная: '{received_text1[:50]}...'")
    print(f"   Loop:    '{received_text2[:50]}...'")
