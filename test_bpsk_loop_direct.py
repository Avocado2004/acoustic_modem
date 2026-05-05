#!/usr/bin/env python3
"""
Прямой тест BPSK loop без интерактивного ввода.
Тестирует полный цикл: генерация -> передача -> прием -> проверка
"""

import random
import string
import sys
import os
import numpy as np

# Устанавливаем BPSK модуляцию ДО всех импортов
from modem_config import set_modulation, init_phases, fs, wavfile, MODULATION, Nfft, Ncp, Nsub, BITS_PER_SYMBOL, OFDM_SYMBOLS_PER_BLOCK

print(f"[TEST] Устанавливаем BPSK модуляцию...")
set_modulation("BPSK")
init_phases()

print(f"[TEST] Модуляция: {MODULATION}, BITS_PER_SYMBOL={BITS_PER_SYMBOL}, OFDM_SYMBOLS_PER_BLOCK={OFDM_SYMBOLS_PER_BLOCK}")

from modem_modulation import build_preamble
from modem_tx import transmit_text
from signal_processor import receive_from_file

# Генерируем случайный текст
random_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(300))
original_data = random_text.encode('utf-8')
print(f"[TEST] Сгенерирован текст: {random_text[:20]}...")
print(f"[TEST] Размер данных: {len(original_data)} байт")

# Сохраняем оригинал
temp_original = "loop_original_text.bin"
with open(temp_original, "wb") as f:
    f.write(original_data)

# Передаём текст
print(f"\n[TEST] Передача текста...")
transmit_text(random_text)
wav_filename = "ofdm_acoustic_tx_with_noise.wav"
print(f"[TEST] Передача завершена: {wav_filename}")

# Проверяем что файл создан
if not os.path.exists(wav_filename):
    print(f"[TEST-ERR] Файл {wav_filename} не создан!")
    sys.exit(1)

# Загружаем и проверяем WAV
fs_wav, sig = wavfile.read(wav_filename)
print(f"[TEST] WAV загружен: {len(sig)} samples, dtype={sig.dtype}, fs={fs_wav}")

# Принимаем
print(f"\n[TEST] Приём данных...")
received_file = "rx_text.txt"
if os.path.exists(received_file):
    os.remove(received_file)

success = receive_from_file(wav_filename)

# Сравниваем
if os.path.exists(received_file):
    with open(received_file, "r", encoding='utf-8') as f:
        received_text = f.read()
    print(f"\n[TEST] Оригинальный текст: {random_text[:30]}...")
    print(f"[TEST] Принятый текст:    {received_text[:30]}...")
    if random_text == received_text:
        print(f"[TEST] ✅ УСПЕХ: Текст совпадает полностью!")
        sys.exit(0)
    else:
        print(f"[TEST] ❌ ОШИБКА: Текст отличается")
        min_len = min(len(random_text), len(received_text))
        for i in range(min_len):
            if random_text[i] != received_text[i]:
                print(f"[TEST] Первое отличие в позиции {i}: ориг='{random_text[i]}', принято='{received_text[i]}'")
                break
        if len(random_text) != len(received_text):
            print(f"[TEST] Длина отличается: ориг={len(random_text)}, принято={len(received_text)}")
        sys.exit(1)
else:
    print(f"[TEST-ERR] Файл {received_file} не найден. Приём не удался.")
    sys.exit(1)
