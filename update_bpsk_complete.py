#!/usr/bin/env python3
"""
Скрипт для полного обновления test_modem_simple.py: 
- Обновляет _try_alternate_demaps_and_rs для поддержки BPSK
- Обновляет live_receive_and_process для динамического определения модуляции
- Обновляет make_training_blocks для учета текущей модуляции
"""

import re

with open('test_modem_simple.py', 'r') as f:
    content = f.read()

# 1. Обновляем _try_alternate_demaps_and_rs
old_try = """def _try_alternate_demaps_and_rs(rx_syms_pkt, n_cw, packet_idx, bytes_before_packet):
    transforms = [
        lambda x: x,
        lambda x: np.conj(x),
        lambda x: -x,
        lambda x: x * 1j,
        lambda x: x * -1j,
    ]
    best = (None, 0)
    for tr in transforms:
        try:
            rx_t = tr(rx_syms_pkt)
        except Exception:
            continue
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
                msg = b'\x00' * RS_DATA_BYTES
            decoded_blocks_t.append(msg)
        if rs_ok_t > best[1]:
            best = (b"".join(decoded_blocks_t), rs_ok_t)
    return best"""

new_try = """def _try_alternate_demaps_and_rs(rx_syms_pkt, n_cw, packet_idx, bytes_before_packet):
    transforms = [
        lambda x: x,
        lambda x: np.conj(x),
        lambda x: -x,
        lambda x: x * 1j,
        lambda x: x * -1j,
    ]
    best = (None, 0)
    # Пробуем оба типа демаппинга
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
                    msg = b'\x00' * RS_DATA_BYTES
                decoded_blocks_t.append(msg)
            if rs_ok_t > best[1]:
                best = (b"".join(decoded_blocks_t), rs_ok_t)
    return best"""

if old_try in content:
    content = content.replace(old_try, new_try)
    print("1. Обновлена функция _try_alternate_demaps_and_rs для поддержки BPSK")
else:
    print("1. ВНИМАНИЕ: Не удалось найти _try_alternate_demaps_and_rs")

# 2. Обновляем make_training_blocks для учета модуляции
old_training = """def make_training_blocks(n_blocks=HABR_SAMPLE_BLOCKS, seed=HABR_SEED):
  rng = np.random.RandomState(seed)
  blocks = []
  for _ in range(n_blocks):
    bits = rng.randint(0, 2, Nsub * 2)
    syms = qpsk_map(bits)
    blocks.append(syms)
  return np.array(blocks)"""

new_training = """def make_training_blocks(n_blocks=HABR_SAMPLE_BLOCKS, seed=HABR_SEED):
  rng = np.random.RandomState(seed)
  blocks = []
  for _ in range(n_blocks):
    bits = rng.randint(0, 2, Nsub * 2)
    # Используем текущую модуляцию для тренировочных блоков
    if MODULATION == "BPSK":
        syms = bpsk_map(bits[:Nsub])  # BPSK: 1 бит на поднесущую
    else:
        syms = qpsk_map(bits)
    blocks.append(syms)
  return np.array(blocks)"""

if old_training in content:
    content = content.replace(old_training, new_training)
    print("2. Обновлена функция make_training_blocks для учета модуляции")
else:
    print("2. ВНИМАНИЕ: Не удалось найти make_training_blocks")

# 3. Добавляем глобальную переменную для отслеживания модуляции в live_receive_and_process
# Проверяем, есть ли уже объявление detected_modulation в начале функции
if "def live_receive_and_process():" in content:
    # Добавляем инициализацию переменной в начало функции
    old_live_start = """def live_receive_and_process():
    CHUNK = 2048
    RECORD_CHANNELS = 1
    BUFFER_LOCK = threading.Lock()
    ring = deque()
    total_samples_in_buffer = 0"""
    
    new_live_start = """def live_receive_and_process():
    CHUNK = 2048
    RECORD_CHANNELS = 1
    BUFFER_LOCK = threading.Lock()
    ring = deque()
    total_samples_in_buffer = 0
    # Для отслеживания модуляции при live приеме
    global MODULATION, BITS_PER_SYMBOL, BITS_PER_OFDM_SYMBOL"""
    
    if old_live_start in content:
        content = content.replace(old_live_start, new_live_start)
        print("3. Обновлена функция live_receive_and_process (добавлена глобальная область)")
    else:
        print("3. ВНИМАНИЕ: Не удалось найти начало live_receive_and_process")

# Записываем обновленный контент
with open('test_modem_simple.py', 'w') as f:
    f.write(content)

print("\nОбновление завершено!")
print("Проверьте файл test_modem_simple.py на наличие ошибок синтаксиса.")
