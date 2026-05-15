#!/usr/bin/env python3
"""
Режим M — приём с микрофона.
Запускать ПЕРВЫМ, затем запустить mic_tx.py в другом терминале.
"""

import sys
import os
import time
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.insert(0, project_dir)

from modem_config import init_phases, fs, set_modulation, MODULATION
from signal_processor import receive_from_microphone_chunked

print("=" * 60)
print("MIC RX — Приём с микрофона")
print("=" * 60)

# Ждём файл-сигнал от передатчика
signal_file = "mic_tx_signal.bin"
print(f"\n[RX] Ожидание сигнала от передатчика ({signal_file})...")
print("[RX] Запустите mic_tx.py в другом терминале!")

timeout = 30  # секунд ожидания
start = time.time()
while not os.path.exists(signal_file):
    time.sleep(0.5)
    if time.time() - start > timeout:
        print("[RX-ERR] Таймаут ожидания передатчика")
        sys.exit(1)

# Читаем параметры передачи
with open(signal_file, 'r') as f:
    modulation_type = f.read().strip()

print(f"[RX] Получен сигнал: модуляция {modulation_type}")

# Устанавливаем модуляцию
set_modulation(modulation_type)
init_phases()
print(f"[RX] Модуляция: {MODULATION}")

# Удаляем файл-сигнал
os.remove(signal_file)

# Запускаем приём
print(f"\n[RX] Запуск записи с микрофона...")
print(f"[RX] Ожидание преамбулы...")

result = receive_from_microphone_chunked()

if result:
    print("\n[RX] ✅ Приём завершён успешно!")
    if os.path.exists("rx_text.txt"):
        with open("rx_text.txt", "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[RX] Принято {len(text)} символов: {text[:50]}...")
else:
    print("\n[RX] ❌ Ошибка приёма")
