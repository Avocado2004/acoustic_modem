#!/usr/bin/env python3
"""
Скрипт для исправления live_receive_and_process в modem_rx.py.
Проблема: функция использует modem_config.OFDM_SYMBOLS_PER_BLOCK до того, 
как модуляция станет известна.
"""
import re

# Читаем файл
with open('modem_rx.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Замена 1: Исправляем physical_symbols_guess
# Используем максимальное значение (для BPSK: OFDM_SYMBOLS_PER_BLOCK=2)
old_code_1 = """            packet_blocks_guess = DEFAULT_PACKET_BLOCKS
            # Переводим логические блоки в физические символы для расчета общего количества сэмплов
            physical_symbols_guess = packet_blocks_guess * modem_config.OFDM_SYMBOLS_PER_BLOCK"""

new_code_1 = """            packet_blocks_guess = DEFAULT_PACKET_BLOCKS
            # Переводим логические блоки в физические символы для расчета общего количества сэмплов
            # Используем максимальное значение OFDM_SYMBOLS_PER_BLOCK (для BPSK это 2)
            # чтобы гарантировать, что мы прочитаем достаточно данных
            physical_symbols_guess = packet_blocks_guess * 2  # максимум для BPSK"""

if old_code_1 in content:
    content = content.replace(old_code_1, new_code_1)
    print("[OK] Замена 1 выполнена: physical_symbols_guess")
else:
    print("[ERR] Замена 1 не выполнена: не найден шаблон")

# Замена 2: Исправляем physical_symbols_refined
old_code_2 = """            # Переводим логические блоки в физические символы
            physical_symbols_refined = packet_blocks_guess * modem_config.OFDM_SYMBOLS_PER_BLOCK"""

new_code_2 = """            # Переводим логические блоки в физические символы
            # Используем максимальное значение OFDM_SYMBOLS_PER_BLOCK (для BPSK это 2)
            physical_symbols_refined = packet_blocks_guess * 2  # максимум для BPSK"""

if old_code_2 in content:
    content = content.replace(old_code_2, new_code_2)
    print("[OK] Замена 2 выполнена: physical_symbols_refined")
else:
    print("[ERR] Замена 2 не выполнена: не найден шаблон")

# Замена 3: Исправляем physical_symbols_val в цикле обработки пакетов
# Нужно использовать текущее значение modem_config.OFDM_SYMBOLS_PER_BLOCK после того, как модуляция определена
old_code_3 = """                    # Переводим логические блоки в физические символы
                    physical_symbols_val = packet_blocks_val * modem_config.OFDM_SYMBOLS_PER_BLOCK"""

new_code_3 = """                    # Переводим логические блоки в физические символы
                    # Используем текущее значение OFDM_SYMBOLS_PER_BLOCK (обновляется после определения модуляции)
                    physical_symbols_val = packet_blocks_val * modem_config.OFDM_SYMBOLS_PER_BLOCK"""

if old_code_3 in content:
    content = content.replace(old_code_3, new_code_3)
    print("[OK] Замена 3 выполнена: physical_symbols_val")
else:
    print("[ERR] Замена 3 не выполнена: не найден шаблон")

# Записываем файл
with open('modem_rx.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nФайл modem_rx.py обновлен!")
