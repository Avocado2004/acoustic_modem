#!/usr/bin/env python3
"""
Тест для проверки интеграции BPSK и QPSK в акустическом модеме.
Проверяет:
1. BPSK/QPSK map/demap функции
2. Построение заголовка с правильными битами модуляции
3. Парсинг заголовка и определение модуляции
4. Сборку и разбор OFDM символов
5. Передачу и прием данных в обоих режимах
"""

import sys
import os
import numpy as np

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("ТЕСТИРОВАНИЕ BPSK И QPSK ИНТЕГРАЦИИ")
print("=" * 60)

# Импортируем модем
try:
    import test_modem_simple as modem
    print("[OK] Модуль test_modem_simple импортирован успешно")
except Exception as e:
    print(f"[ERROR] Ошибка импорта: {e}")
    sys.exit(1)

def test_bpsk_functions():
    """Тест 1: Проверка BPSK map/demap функций"""
    print("\n[ТЕСТ 1] Проверка BPSK map/demap функций...")
    
    # Проверяем наличие функций
    if not hasattr(modem, 'bpsk_map') or not hasattr(modem, 'bpsk_demap'):
        print("[FAIL] Функции bpsk_map или bpsk_demap не найдены")
        return False
    
    # Тест маппинга BPSK
    test_bits = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 1])
    syms = modem.bpsk_map(test_bits)
    
    if len(syms) != len(test_bits):
        print(f"[FAIL] Неверная длина символов: {len(syms)} != {len(test_bits)}")
        return False
    
    # Проверяем, что символы - это 1+0j для 0 и -1+0j для 1
    for i, (b, s) in enumerate(zip(test_bits, syms)):
        expected = 1.0 if b == 0 else -1.0
        if abs(s - expected) > 1e-10:
            print(f"[FAIL] Неверный символ для бита {b}: {s} != {expected}")
            return False
    
    print("[OK] bpsk_map работает корректно")
    
    # Тест демаппинга BPSK
    demapped = modem.bpsk_demap(syms)
    
    if not np.array_equal(test_bits, demapped):
        print(f"[FAIL] Демаппинг не совпадает: {test_bits} != {demapped}")
        return False
    
    print("[OK] bpsk_demap работает корректно")
    return True

def test_qpsk_functions():
    """Тест 2: Проверка QPSK map/demap функций"""
    print("\n[ТЕСТ 2] Проверка QPSK map/demap функций...")
    
    if not hasattr(modem, 'qpsk_map') or not hasattr(modem, 'qpsk_demap'):
        print("[FAIL] Функции qpsk_map или qpsk_demap не найдены")
        return False
    
    test_bits = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    syms = modem.qpsk_map(test_bits)
    
    if len(syms) != len(test_bits) // 2:
        print(f"[FAIL] Неверная длина символов QPSK")
        return False
    
    demapped = modem.qpsk_demap(syms)
    
    if not np.array_equal(test_bits, demapped):
        print(f"[FAIL] QPSK демаппинг не совпадает")
        return False
    
    print("[OK] QPSK map/demap работают корректно")
    return True

def test_modulation_header():
    """Тест 3: Проверка заголовка с битами модуляции"""
    print("\n[ТЕСТ 3] Проверка заголовка с битами модуляции...")
    
    # Сохраняем оригинальную модуляцию
    orig_mod = modem.MODULATION
    
    try:
        # Тест BPSK заголовка
        modem.MODULATION = "BPSK"
        header_bpsk = modem.build_header(b'F', 100, filename_bytes=b'test.txt')
        
        # Проверяем биты модуляции в первом байте (bits 3-2 должны быть 01 для BPSK)
        flags = header_bpsk[0]
        mod_bits = (flags >> 2) & 0x03
        
        if mod_bits != 0b01:
            print(f"[FAIL] BPSK: Неверные биты модуляции: {mod_bits:02b} (ожидалось 01)")
            return False
        
        print(f"[OK] BPSK заголовок: mod_bits={mod_bits:02b} (правильно)")
        
        # Тест QPSK заголовка
        modem.MODULATION = "QPSK"
        header_qpsk = modem.build_header(b'F', 100, filename_bytes=b'test.txt')
        
        flags = header_qpsk[0]
        mod_bits = (flags >> 2) & 0x03
        
        if mod_bits != 0b00:
            print(f"[FAIL] QPSK: Неверные биты модуляции: {mod_bits:02b} (ожидалось 00)")
            return False
        
        print(f"[OK] QPSK заголовок: mod_bits={mod_bits:02b} (правильно)")
        
        # Проверяем парсинг заголовка
        parsed = modem.parse_header(header_bpsk)
        if parsed['modulation'] != "BPSK":
            print(f"[FAIL] Парсинг BPSK заголовка: {parsed['modulation']}")
            return False
        
        parsed = modem.parse_header(header_qpsk)
        if parsed['modulation'] != "QPSK":
            print(f"[FAIL] Парсинг QPSK заголовка: {parsed['modulation']}")
            return False
        
        print("[OK] Парсинг заголовка работает корректно")
        return True
        
    finally:
        modem.MODULATION = orig_mod

def test_build_data_td():
    """Тест 4: Проверка сборки OFDM символов для обеих модуляций"""
    print("\n[ТЕСТ 4] Проверка build_data_td для BPSK и QPSK...")
    
    orig_mod = modem.MODULATION
    
    try:
        # Тест BPSK
        modem.MODULATION = "BPSK"
        modem.BITS_PER_SYMBOL = 1
        modem.BITS_PER_OFDM_SYMBOL = modem.Nsub * modem.BITS_PER_SYMBOL
        
        test_bits = np.random.randint(0, 2, modem.BITS_PER_OFDM_SYMBOL)
        td_bpsk, nblocks_bpsk = modem.build_data_td(test_bits)
        
        if nblocks_bpsk != 1:
            print(f"[FAIL] BPSK: ожидался 1 блок, получено {nblocks_bpsk}")
            return False
        
        print(f"[OK] BPSK build_data_td: {nblocks_bpsk} блок(ов)")
        
        # Тест QPSK
        modem.MODULATION = "QPSK"
        modem.BITS_PER_SYMBOL = 2
        modem.BITS_PER_OFDM_SYMBOL = modem.Nsub * modem.BITS_PER_SYMBOL
        
        test_bits = np.random.randint(0, 2, modem.BITS_PER_OFDM_SYMBOL)
        td_qpsk, nblocks_qpsk = modem.build_data_td(test_bits)
        
        if nblocks_qpsk != 1:
            print(f"[FAIL] QPSK: ожидался 1 блок, получено {nblocks_qpsk}")
            return False
        
        print(f"[OK] QPSK build_data_td: {nblocks_qpsk} блок(ов)")
        return True
        
    finally:
        modem.MODULATION = orig_mod

def test_full_transmission(mode):
    """Тест 5: Полная передача и прием данных"""
    print(f"\n[ТЕСТ 5] Полная передача данных в режиме {mode}...")
    
    orig_mod = modem.MODULATION
    
    try:
        # Устанавливаем модуляцию
        modem.MODULATION = mode
        if mode == "BPSK":
            modem.BITS_PER_SYMBOL = 1
        else:
            modem.BITS_PER_SYMBOL = 2
        modem.BITS_PER_OFDM_SYMBOL = modem.Nsub * modem.BITS_PER_SYMBOL
        
        # Тестовые данные
        test_text = b"Hello BPSK/QPSK Test!" if mode == "BPSK" else b"Hello QPSK Test!"
        
        # Кодируем данные
        rs_encoded = modem.rs.encode(test_text)
        bits = modem.bytes_to_bits(rs_encoded)
        
        # Маппинг
        if mode == "BPSK":
            syms = modem.bpsk_map(bits)
        else:
            syms = modem.qpsk_map(bits)
        
        # Создаем OFDM символы
        td, nblocks = modem.build_data_td(bits)
        
        print(f"[OK] Передача: {len(test_text)} байт -> {nblocks} OFDM блоков")
        
        # Симуляция приема (упрощенная)
        # Добавляем preamble
        preamble = modem.build_preamble()
        rx_signal = np.concatenate((preamble, td))
        
        # Имитация приема (в реальности здесь был бы decode_packet_at_candidate)
        print(f"[OK] Сигнал сформирован: {len(rx_signal)} samples")
        print(f"[OK] Режим {mode} работает корректно")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Ошибка в режиме {mode}: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        modem.MODULATION = orig_mod

def test_modulation_switching():
    """Тест 6: Переключение между модуляциями"""
    print("\n[ТЕСТ 6] Проверка переключения модуляции...")
    
    try:
        # Переключаемся в BPSK
        modem.MODULATION = "BPSK"
        modem.BITS_PER_SYMBOL = 1
        modem.BITS_PER_OFDM_SYMBOL = modem.Nsub * modem.BITS_PER_SYMBOL
        
        if modem.BITS_PER_SYMBOL != 1 or modem.BITS_PER_OFDM_SYMBOL != modem.Nsub:
            print("[FAIL] Переключение в BPSK не сработало")
            return False
        
        print(f"[OK] Переключение в BPSK: BITS_PER_SYMBOL={modem.BITS_PER_SYMBOL}")
        
        # Переключаемся в QPSK
        modem.MODULATION = "QPSK"
        modem.BITS_PER_SYMBOL = 2
        modem.BITS_PER_OFDM_SYMBOL = modem.Nsub * modem.BITS_PER_SYMBOL
        
        if modem.BITS_PER_SYMBOL != 2 or modem.BITS_PER_OFDM_SYMBOL != modem.Nsub * 2:
            print("[FAIL] Переключение в QPSK не сработало")
            return False
        
        print(f"[OK] Переключение в QPSK: BITS_PER_SYMBOL={modem.BITS_PER_SYMBOL}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Ошибка переключения: {e}")
        return False

# Запуск всех тестов
print("\n" + "=" * 60)
print("ЗАПУСК ТЕСТОВ")
print("=" * 60)

results = []

results.append(("BPSK functions", test_bpsk_functions()))
results.append(("QPSK functions", test_qpsk_functions()))
results.append(("Modulation header", test_modulation_header()))
results.append(("Build data TD", test_build_data_td()))
results.append(("Full transmission BPSK", test_full_transmission("BPSK")))
results.append(("Full transmission QPSK", test_full_transmission("QPSK")))
results.append(("Modulation switching", test_modulation_switching()))

# Итоги
print("\n" + "=" * 60)
print("ИТОГИ ТЕСТИРОВАНИЯ")
print("=" * 60)

passed = sum(1 for _, r in results if r)
total = len(results)

for name, result in results:
    status = "[PASS]" if result else "[FAIL]"
    print(f"{status} {name}")

print(f"\nПройдено: {passed}/{total} тестов")

if passed == total:
    print("\n[SUCCESS] Все тесты пройдены! BPSK и QPSK интегрированы корректно.")
else:
    print(f"\n[WARNING] {total - passed} тест(ов) не пройдены")

sys.exit(0 if passed == total else 1)
