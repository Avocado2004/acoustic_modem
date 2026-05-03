"""
Тест полного цикла для BPSK: упаковка, передача, прием, распаковка.
Проверяет, что данные не искажаются на позиции 104.
"""

import sys
import os
import numpy as np

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modem_config
from modem_tx import _bytes_to_ofdm_blocks_bytes
from modem_modulation import (bpsk_map, bpsk_demap, interleave_bits, deinterleave_bits,
                                build_data_td, bytes_to_bits, bits_to_bytes)
from reedsolo import RSCodec

def test_bpsk_full_cycle():
    """Тест полного цикла BPSK."""
    print("=== Тест полного цикла BPSK ===")
    
    # Устанавливаем BPSK
    modem_config.set_modulation("BPSK")
    print(f"[TEST] Модуляция: {modem_config.MODULATION}")
    print(f"[TEST] BITS_PER_SYMBOL: {modem_config.BITS_PER_SYMBOL}")
    print(f"[TEST] BITS_PER_OFDM_SYMBOL: {modem_config.BITS_PER_OFDM_SYMBOL}")
    
    # Тестовые данные: 300 байт (как в логе)
    test_data = os.urandom(300)
    print(f"[TEST] Тестовые данные: {len(test_data)} байт")
    
    # 1. RS кодирование
    rs = RSCodec(4)  # 4 байта паритета
    rs_encoded = []
    for i in range(0, len(test_data), 8):
        block = test_data[i:i+8]
        if len(block) < 8:
            block = block + b'\x00' * (8 - len(block))
        encoded = rs.encode(block)
        rs_encoded.append(encoded)
    
    print(f"[TEST] RS кодирование: {len(rs_encoded)} блоков")
    print(f"[TEST] Размер каждого блока: {len(rs_encoded[0])} байт (8 данных + 4 паритета)")
    
    # 2. Преобразование в биты
    all_bits = []
    for block in rs_encoded:
        bits = bytes_to_bits(block)
        all_bits.extend(bits)
    
    print(f"[TEST] Всего бит после RS: {len(all_bits)}")
    print(f"[TEST] Ожидается: {len(rs_encoded)} * 96 = {len(rs_encoded) * 96} бит")
    
    # 3. Интерливинг
    bits_array = np.array(all_bits, dtype=int)
    interleaved = interleave_bits(bits_array, block_size=modem_config.BITS_PER_OFDM_SYMBOL)
    
    print(f"[TEST] Интерливинг с block_size={modem_config.BITS_PER_OFDM_SYMBOL}")
    print(f"[TEST] После интерливинга: {len(interleaved)} бит")
    
    # 4. Маппинг BPSK
    syms = bpsk_map(interleaved)
    print(f"[TEST] BPSK маппинг: {len(syms)} символов")
    print(f"[TEST] Ожидается: {len(interleaved)} символов (1 бит = 1 символ)")
    
    # 5. Проверяем, сколько OFDM символов получится
    # Для BPSK: 48 поднесущих = 48 бит/OFDM символ
    ofdm_symbols_count = len(syms) // modem_config.Nsub
    if len(syms) % modem_config.Nsub != 0:
        ofdm_symbols_count += 1
    
    print(f"[TEST] Количество OFDM символов: ~{ofdm_symbols_count}")
    print(f"[TEST] Бит на OFDM символ: {modem_config.Nsub}")
    print(f"[TEST] Всего бит данных: {ofdm_symbols_count * modem_config.Nsub}")
    
    # 6. Демаппинг (имитируем прием)
    demapped_bits = bpsk_demap(syms)
    print(f"[TEST] Демаппинг: {len(demapped_bits)} бит")
    
    # 7. Деинтерливинг
    deinterleaved = deinterleave_bits(demapped_bits, block_size=modem_config.BITS_PER_OFDM_SYMBOL)
    print(f"[TEST] Деинтерливинг: {len(deinterleaved)} бит")
    
    # 8. Проверяем совпадение
    if np.array_equal(interleaved, demapped_bits):
        print("[TEST] ✓ Интерливинг/Демаппинг работает корректно!")
    else:
        print("[TEST] ✗ Ошибка в цикле интерливинг/демаппинг")
        # Находим первое отличие
        for i in range(min(len(interleaved), len(demapped_bits))):
            if interleaved[i] != demapped_bits[i]:
                print(f"[TEST] Первое отличие на позиции {i}: было {interleaved[i]}, стало {demapped_bits[i]}")
                break
        return False
    
    # 9. RS декодирование
    rs_decoded = []
    n_cw = len(deinterleaved) // 96
    print(f"[TEST] RS декодирование: {n_cw} кодовых слов")
    
    for i in range(n_cw):
        cw_bits = deinterleaved[i*96:(i+1)*96]
        cw_bytes = bits_to_bytes(cw_bits)
        try:
            decoded = rs.decode(cw_bytes)[0]
            rs_decoded.append(decoded[:8])  # Берем только 8 байт данных
        except Exception as e:
            print(f"[TEST] Ошибка RS декодирования для блока {i}: {e}")
            rs_decoded.append(b'\x00' * 8)
    
    # Собираем данные
    decoded_data = b''.join(rs_decoded)
    print(f"[TEST] Декодировано: {len(decoded_data)} байт")
    
    # Сравниваем с оригиналом (первые 300 байт)
    original_300 = test_data[:300]
    decoded_300 = decoded_data[:300]
    
    if decoded_300 == original_300:
        print("[TEST] ✓ Данные совпадают!")
        return True
    else:
        print("[TEST] ✗ Данные не совпадают!")
        for i in range(min(len(original_300), len(decoded_300))):
            if original_300[i] != decoded_300[i]:
                print(f"[TEST] Первое отличие на позиции {i}: ориг={original_300[i]:02x}, принято={decoded_300[i]:02x}")
                break
        return False

if __name__ == "__main__":
    print("Запуск теста полного цикла BPSK...")
    
    try:
        result = test_bpsk_full_cycle()
        if result:
            print("\n" + "="*50)
            print("ВСЕ ТЕСТЫ BPSK ПРОЙДЕНЫ УСПЕШНО!")
            print("="*50)
        else:
            print("\n[TEST-ERR] Тест не пройден!")
            sys.exit(1)
    except Exception as e:
        print(f"\n[TEST-ERR] Тест упал: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
