#!/usr/bin/env python3
"""
Скрипт для обновления test_modem_simple.py: 
добавляет динамическое определение модуляции BPSK/QPSK при приеме
"""

import re

with open('test_modem_simple.py', 'r') as f:
    content = f.read()

# Старый код демаппинга (точное совпадение)
old_code = """        # Use appropriate modulation demapping
        if MODULATION == "BPSK":
            bits_pkt = bpsk_demap(rx_syms_pkt)
        else:
            bits_pkt = qpsk_demap(rx_syms_pkt)"""

# Новый код с динамическим определением модуляции
new_code = """        # Динамическое определение модуляции из заголовка
        # Пробуем QPSK по умолчанию
        bits_pkt = qpsk_demap(rx_syms_pkt)
        
        # Для первого пакета пытаемся определить модуляцию из заголовка
        if packet_idx == 0:
            cw_bits = RS_CW_BITS
            n_cw = len(bits_pkt) // cw_bits
            detected_mod = None
            
            # Пробуем QPSK демаппинг
            if n_cw > 0:
                bstart = 0
                bbits = bits_pkt[bstart:bstart+cw_bits]
                if len(bbits) < cw_bits:
                    bbits = np.concatenate((bbits, np.zeros(cw_bits - len(bbits), dtype=int)))
                bts = bits_to_bytes(bbits)
                try:
                    msg = rs.decode(bts)[0]
                    if len(msg) > 0 and (msg[0] & 0xF0) != 0:
                        # Валидный заголовок, проверяем биты модуляции
                        mod_bits = (msg[0] >> 2) & 0x03
                        detected_mod = "BPSK" if mod_bits == 0b01 else "QPSK"
                except Exception:
                    pass
            
            # Если QPSK не сработал, пробуем BPSK
            if detected_mod is None:
                bits_pkt_bpsk = bpsk_demap(rx_syms_pkt)
                n_cw_bpsk = len(bits_pkt_bpsk) // cw_bits
                if n_cw_bpsk > 0:
                    bbits = bits_pkt_bpsk[:cw_bits]
                    if len(bbits) < cw_bits:
                        bbits = np.concatenate((bbits, np.zeros(cw_bits - len(bbits), dtype=int)))
                    bts = bits_to_bytes(bbits)
                    try:
                        msg = rs.decode(bts)[0]
                        if len(msg) > 0 and (msg[0] & 0xF0) != 0:
                            mod_bits = (msg[0] >> 2) & 0x03
                            detected_mod = "BPSK" if mod_bits == 0b01 else "QPSK"
                            if detected_mod == "BPSK":
                                bits_pkt = bits_pkt_bpsk
                    except Exception:
                        pass
            
            # Обновляем глобальные переменные если модуляция определена
            if detected_mod is not None:
                global MODULATION, BITS_PER_SYMBOL, BITS_PER_OFDM_SYMBOL
                if MODULATION != detected_mod:
                    MODULATION = detected_mod
                    if MODULATION == "BPSK":
                        BITS_PER_SYMBOL = 1
                    else:
                        BITS_PER_SYMBOL = 2
                    BITS_PER_OFDM_SYMBOL = Nsub * BITS_PER_SYMBOL
                    print(f"[RX] Обнаружена модуляция: {MODULATION}, BITS_PER_SYMBOL={BITS_PER_SYMBOL}")
        else:
            # Для последующих пакетов используем текущую настройку MODULATION
            if MODULATION == "BPSK":
                bits_pkt = bpsk_demap(rx_syms_pkt)
            else:
                bits_pkt = qpsk_demap(rx_syms_pkt)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('test_modem_simple.py', 'w') as f:
        f.write(content)
    print("Успешно обновлена функция decode_packet_at_candidate")
    print("Добавлено динамическое определение модуляции BPSK/QPSK")
else:
    print("ОШИБКА: Не удалось найти точное совпадение для замены")
    print("Проверьте форматирование в файле test_modem_simple.py")
