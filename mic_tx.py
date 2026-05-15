#!/usr/bin/env python3
"""
Режим M — передача через динамик.
Запускать ВТОРЫМ, после запуска mic_rx.py в другом терминале.
"""

import sys
import os
import time
import random
import string

project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.insert(0, project_dir)

from modem_config import init_phases, fs, set_modulation, MODULATION
from modem_tx import transmit_text

print("=" * 60)
print("MIC TX — Передача через динамик")
print("=" * 60)

# Модуляция по умолчанию QPSK
modulation_type = "QPSK"
print(f"[TX] Модуляция: {modulation_type}")

# Устанавливаем модуляцию
set_modulation(modulation_type)
init_phases()

# Генерируем 1 КБ случайных данных
random_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(1024))
print(f"[TX] Сгенерирован текст: {random_text[:20]}...")
print(f"[TX] Размер данных: {len(random_text.encode('utf-8'))} байт")

# Сохраняем оригинал для сравнения
with open("mic_original_text.txt", "w", encoding='utf-8') as f:
    f.write(random_text)

# Сигнализируем приёмнику о начале передачи
signal_file = "mic_tx_signal.bin"
with open(signal_file, 'w') as f:
    f.write(modulation_type)

print(f"\n[TX] Сигнал отправлен приёмнику")
print(f"[TX] Ожидание 2 секунды перед передачей...")
time.sleep(2)

# Передаём данные через динамик
print(f"\n[TX] Передача данных через динамик...")
transmit_text(random_text)
print(f"[TX] Передача завершена")

# Ждём завершения приёма
print(f"[TX] Ожидание завершения приёма...")
time.sleep(5)

# Сравниваем результат
received_file = "rx_text.txt"
if os.path.exists(received_file):
    with open(received_file, "r", encoding='utf-8') as f:
        received_text = f.read()
    
    print(f"\n[TX] Сравнение данных...")
    print(f"[TX] Оригинал:  {random_text[:30]}...")
    print(f"[TX] Принято:   {received_text[:30]}...")
    
    if random_text == received_text:
        print(f"\n[TX] ✅ УСПЕХ: Текст совпадает полностью!")
    else:
        print(f"\n[TX] ❌ ОШИБКА: Текст отличается")
        min_len = min(len(random_text), len(received_text))
        for i in range(min_len):
            if random_text[i] != received_text[i]:
                print(f"[TX] Первое отличие в позиции {i}: ориг='{random_text[i]}', принято='{received_text[i]}'")
                break
        if len(random_text) != len(received_text):
            print(f"[TX] Длина: ориг={len(random_text)}, принято={len(received_text)}")
else:
    print(f"\n[TX] ❌ Файл {received_file} не найден")
