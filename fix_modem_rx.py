#!/usr/bin/env python3
"""
Скрипт для исправления _try_decode_with_modulation в modem_rx.py.
Проблема: функция использует глобальные настройки modem_config, 
которые не соответствуют тестируемой модуляции.
"""
import re

# Читаем файл
with open('modem_rx.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Замена 1: Добавляем локальные переменные для параметров модуляции
# Ищем начало функции _try_decode_with_modulation и добавляем переменные после проверки глобальных переменных
old_code_1 = """    if 'abs_corr' not in globals() or 'rx' not in globals():
        return None
    
    try:"""

new_code_1 = """    if 'abs_corr' not in globals() or 'rx' not in globals():
        return None
    
    # Определяем параметры для конкретной модуляции
    if modulation == "BPSK":
        bits_per_symbol = 1
        ofdm_symbols_per_block = 2  # 2 физических символа = 1 логический блок для BPSK
        bits_per_ofdm_symbol = Nsub * bits_per_symbol  # 48 для BPSK
    else:  # QPSK
        bits_per_symbol = 2
        ofdm_symbols_per_block = 1  # 1 физический символ = 1 логический блок для QPSK
        bits_per_ofdm_symbol = Nsub * bits_per_symbol  # 96 для QPSK
    
    try:"""

if old_code_1 in content:
    content = content.replace(old_code_1, new_code_1)
    print("[OK] Замена 1 выполнена: добавлены локальные переменные")
else:
    print("[ERR] Замена 1 не выполнена: не найден шаблон")
    # Попробуем найти с другим форматированием
    if 'if \'abs_corr\' not in globals() or \'rx\' not in globals():' in content:
        print("  Но похожий код найден, проверьте отступы")

# Замена 2: Исправляем расчет physical_symbols_expected
old_code_2 = """    # Переводим логические блоки в физические символы
    physical_symbols_expected = packet_blocks_expected * modem_config.OFDM_SYMBOLS_PER_BLOCK"""

new_code_2 = """    # Переводим логические блоки в физические символы
    physical_symbols_expected = packet_blocks_expected * ofdm_symbols_per_block"""

if old_code_2 in content:
    content = content.replace(old_code_2, new_code_2)
    print("[OK] Замена 2 выполнена: physical_symbols_expected")
else:
    print("[ERR] Замена 2 не выполнена: не найден шаблон")

# Замена 3: Исправляем вызов deinterleave_bits
old_code_3 = """    # Применяем деинтерливинг
    bits_pkt = deinterleave_bits(bits_pkt, block_size=modem_config.BITS_PER_OFDM_SYMBOL)"""

new_code_3 = """    # Применяем деинтерливинг с правильным размером блока для данной модуляции
    bits_pkt = deinterleave_bits(bits_pkt, block_size=bits_per_ofdm_symbol)"""

if old_code_3 in content:
    content = content.replace(old_code_3, new_code_3)
    print("[OK] Замена 3 выполнена: deinterleave_bits")
else:
    print("[ERR] Замена 3 не выполнена: не найден шаблон")

# Записываем файл
with open('modem_rx.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nФайл modem_rx.py обновлен!")
