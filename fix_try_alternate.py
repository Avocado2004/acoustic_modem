#!/usr/bin/env python3
"""
Скрипт для исправления _try_alternate_demaps_and_rs в test_modem_simple.py
Добавляет поддержку BPSK демаппинга
"""

with open('test_modem_simple.py', 'r') as f:
    lines = f.readlines()

# Находим начало функции _try_alternate_demaps_and_rs
start_line = None
for i, line in enumerate(lines):
    if 'def _try_alternate_demaps_and_rs' in line:
        start_line = i
        break

if start_line is None:
    print("ОШИБКА: Не найдена функция _try_alternate_demaps_and_rs")
    exit(1)

# Находим конец функции (следующая строка, которая не начинается с пробела после start_line)
end_line = None
for i in range(start_line + 1, len(lines)):
    if lines[i].strip() == '' or lines[i].startswith('    ') or lines[i].startswith('\t'):
        continue
    else:
        end_line = i
        break

if end_line is None:
    end_line = len(lines)

print(f"Найдена функция с {end_line - start_line} строк")

# Новая версия функции
new_function = '''def _try_alternate_demaps_and_rs(rx_syms_pkt, n_cw, packet_idx, bytes_before_packet):
    transforms = [
        lambda x: x,
        lambda x: np.conj(x),
        lambda x: -x,
        lambda x: x * 1j,
        lambda x: x * -1j,
    ]
    best = (None, 0)
    # Пробуем оба типа демаппинга: QPSK и BPSK
    for mod in ["QPSK", "BPSK"]:
        for tr in transforms:
            try:
                rx_t = tr(rx_syms_pkt)
            except Exception:
                continue
            # Используем соответствующий демаппинг
            if mod == "BPSK":
                bits_t = bpsk_demap(rx_t)
            else:
                bits_t = qpsk_demap(rx_t)
            cw_bits = RS_CW_BITS
            n_cw_t = len(bits_t) // cw_bits
            rs_ok_t = 0
            decoded_blocks_t = []
            for ci in range(min(n_cw_t, n_cw)):
                bstart = ci * cw_bits
                bbits = bits_t[bstart:bstart+cw_bits]
                if len(bbits) < cw_bits:
                    bbits = np.concatenate((bbits, np.zeros(cw_bits - len(bbits), dtype=int)))
                bts = bits_to_bytes(bbits)
                try:
                    msg = rs.decode(bts)[0]
                    rs_ok_t += 1
                except Exception:
                    msg = b'\\x00' * RS_DATA_BYTES
                decoded_blocks_t.append(msg)
            if rs_ok_t > best[1]:
                best = (b"".join(decoded_blocks_t), rs_ok_t)
    return best

'''

# Заменяем старую функцию новой
new_lines = lines[:start_line] + [new_function] + lines[end_line:]

with open('test_modem_simple.py', 'w') as f:
    f.writelines(new_lines)

print("Успешно обновлена функция _try_alternate_demaps_and_rs")
print("Добавлена поддержка BPSK демаппинга")
