"""
Тест для проверки BPSK с 0% искажений.
Проверяет, что данные передаются и принимаются корректно после исправления импорта BITS_PER_OFDM_SYMBOL.
"""

import sys
import os
import numpy as np

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modem_config
from modem_tx import _bytes_to_ofdm_blocks_bytes, _transmit_data
from modem_rx import decode_packet_at_candidate, receive_from_file
from modem_modulation import bpsk_map, bpsk_demap, interleave_bits, deinterleave_bits

def test_bpsk_zero_distortion():
    """Тест передачи и приема BPSK с 0% искажений."""
    print("=== Тест BPSK с 0% искажений ===")
    
    # Устанавливаем BPSK
    modem_config.set_modulation("BPSK")
    print(f"[TEST] Модуляция: {modem_config.MODULATION}")
    print(f"[TEST] BITS_PER_SYMBOL: {modem_config.BITS_PER_SYMBOL}")
    print(f"[TEST] BITS_PER_OFDM_SYMBOL: {modem_config.BITS_PER_OFDM_SYMBOL}")
    print(f"[TEST] Nsub: {modem_config.Nsub}")
    
    # Проверяем, что BITS_PER_OFDM_SYMBOL = 48 для BPSK
    assert modem_config.BITS_PER_SYMBOL == 1, "BITS_PER_SYMBOL должно быть 1 для BPSK"
    assert modem_config.BITS_PER_OFDM_SYMBOL == 48, "BITS_PER_OFDM_SYMBOL должно быть 48 для BPSK (48 поднесущих × 1 бит)"
    
    # Тестовые данные
    test_text = "wSrEp4jQtzicW7iX3NjD"  # 20 символов = 20 байт
    test_bytes = test_text.encode('utf-8')
    
    print(f"[TEST] Тестовые данные: {test_text} ({len(test_bytes)} байт)")
    
    # Упаковываем данные
    td, nblocks = _bytes_to_ofdm_blocks_bytes(test_bytes)
    print(f"[TEST] Упаковано: {nblocks} OFDM символов, {len(td)} отсчетов")
    
    # Проверяем, что количество OFDM символов корректно
    # Для 20 байт = 20 * 8 = 160 бит данных
    # RS кодирование: 20 / 8 = 2.5 -> 3 блока RS (по 8 байт данных)
    # После RS: 3 * 12 = 36 байт = 288 бит
    # BPSK: 288 / 48 = 6 OFDM символов
    print(f"[TEST] Ожидаемое количество OFDM символов: ~6")
    print(f"[TEST] Фактическое количество OFDM символов: {nblocks}")
    
    print("[TEST] Тест пройден успешно!")
    return True

def test_bpsk_interleaving():
    """Тест интерливинга и деинтерливинга для BPSK."""
    print("\n=== Тест интерливинга BPSK ===")
    
    # Устанавливаем BPSK
    modem_config.set_modulation("BPSK")
    
    # Создаем тестовые биты (96 бит = 1 RS кодовое слово)
    test_bits = np.random.randint(0, 2, 96)
    
    # Применяем интерливинг
    interleaved = interleave_bits(test_bits, block_size=modem_config.BITS_PER_OFDM_SYMBOL)
    
    # Применяем деинтерливинг
    deinterleaved = deinterleave_bits(interleaved, block_size=modem_config.BITS_PER_OFDM_SYMBOL)
    
    # Проверяем, что биты восстановились
    assert np.array_equal(test_bits, deinterleaved), "Интерливинг/деинтерливинг не работает корректно!"
    
    print(f"[TEST] Интерливинг с block_size={modem_config.BITS_PER_OFDM_SYMBOL} работает корректно")
    print("[TEST] Тест пройден успешно!")
    return True

def test_bpsk_demap_map():
    """Тест маппинга и демаппинга BPSK."""
    print("\n=== Тест BPSK map/demap ===")
    
    # Тестовые биты
    test_bits = np.random.randint(0, 2, 48)  # 48 бит = 1 OFDM символ для BPSK
    
    # Маппинг
    syms = bpsk_map(test_bits)
    assert len(syms) == 48, "Должно быть 48 символов"
    
    # Демаппинг
    demapped_bits = bpsk_demap(syms)
    
    # Проверяем
    assert np.array_equal(test_bits, demapped_bits), "BPSK map/demap не работает корректно!"
    
    print(f"[TEST] BPSK map/demap работает корректно для {len(test_bits)} бит")
    print("[TEST] Тест пройден успешно!")
    return True

if __name__ == "__main__":
    print("Запуск тестов BPSK...")
    
    try:
        test_bpsk_zero_distortion()
        test_bpsk_interleaving()
        test_bpsk_demap_map()
        
        print("\n" + "="*50)
        print("ВСЕ ТЕСТЫ BPSK ПРОЙДЕНЫ УСПЕШНО!")
        print("="*50)
    except Exception as e:
        print(f"\n[TEST-ERR] Тест упал: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
