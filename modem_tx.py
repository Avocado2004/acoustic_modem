"""
Модуль передачи OFDM Acoustic Modem.
Содержит функции сборки пакетов, мягкого клиппинга и сохранения WAV.
"""

import os
import numpy as np
import struct
import zlib
import modem_config
from modem_config import (fs, Nfft, Ncp, Nsub, subc_inds,
                           DEFAULT_PACKET_BLOCKS, GAP_OFDM_SYMBOLS, SYMBOL_LEN,
                           GAP_SAMPLES_DEFAULT, MAX_PAYLOAD_SIZE, RS_DATA_BYTES,
                           RS_CW_BYTES, RS_CW_BITS, rs, SYMBOL_TX_TARGET,
                           TARGET_RMS, MIN_RMS, SYMBOL_TARGET_RMS, AGC_ALPHA,
                           AGC_DEBUG, PLOTTING_AVAILABLE, plt,
                           PLOT_CONSTELLATION, CONSTELLATION_TX_FILENAME,
                           CONSTELLATION_USE_GRADIENT,
                           PLOT_CONSTELLATION_TX_OFDM, CONSTELLATION_TX_OFDM_FILENAME,
                           CONSTELLATION_TX_OFDM_COMPENSATE_PHASE,
                           CONSTELLATION_TX_OFDM_COMPENSATED_FILENAME,
                           PREAMBLE_PILOT_SYMBOLS, WARMUP_SYMBOLS, WARMUP_LEVEL,
                           PILOT_INTERVAL)
from modem_modulation import (qpsk_map, bpsk_map, bytes_to_bits, bits_to_bytes,
                               ofdm_symbol, build_preamble, build_data_td, zc_root_sequence,
                               interleave_bits)
from modem_packet import build_header, make_packet_header_bytes, simulate_packet_positions 


# -----------------------
# soft clipping utilities
# -----------------------
TARGET_CREST_DB = 6
CREST_TOLERANCE_DB = 0.1
MAX_SOFTCLIP_ITERS = 40

def softclip_tanh(x, g):
    """Мягкий клиппинг с использованием tanh."""
    if g <= 0:
        return x
    denom = np.tanh(g)
    if denom == 0:
        return x
    return np.tanh(g * x) / denom

def crest_db(sig):
    """Расчет пик-фактора сигнала в дБ."""
    peak = np.max(np.abs(sig))
    rms = np.sqrt(np.mean(sig**2))
    if rms == 0 or peak == 0:
        return np.inf
    return 20.0 * np.log10(peak / rms)

def apply_softclip_to_target_crest(x, target_db=TARGET_CREST_DB, tol_db=CREST_TOLERANCE_DB, max_iters=MAX_SOFTCLIP_ITERS):
    """Применение мягкого клиппинга для достижения целевого пик-фактора."""
    if x.size == 0:
        return x, 0.0
    peak_in = np.max(np.abs(x))
    if peak_in == 0:
        return x, crest_db(x)
    x_norm = x / peak_in
    if crest_db(x_norm) <= target_db + tol_db:
        return x, crest_db(x)
    g_lo = 1e-8
    g_hi = 1.0
    for _ in range(30):
        y = softclip_tanh(x_norm, g_hi)
        if crest_db(y) <= target_db:
            break
        g_hi *= 2.0
    best_g = g_hi
    for _ in range(max_iters):
        g_mid = 0.5 * (g_lo + g_hi)
        y_mid = softclip_tanh(x_norm, g_mid)
        c_mid = crest_db(y_mid)
        if abs(c_mid - target_db) <= tol_db:
            best_g = g_mid
            break
        if c_mid > target_db:
            g_lo = g_mid
        else:
            g_hi = g_mid
        best_g = g_mid
    y_final = softclip_tanh(x / peak_in, best_g) * peak_in
    final_crest = crest_db(y_final)
    return y_final, final_crest


def transmit_text(text, packet_blocks=DEFAULT_PACKET_BLOCKS):
    """Передача текста."""
    text_bytes = text.encode("utf-8")
    total_data_len = len(text_bytes)
    filename_bytes = b''
    return _transmit_data(text_bytes, total_data_len, filename_bytes, mode='T', packet_blocks=packet_blocks)


def transmit_file(file_path, packet_blocks=DEFAULT_PACKET_BLOCKS):
    """Передача файла."""
    if not os.path.isfile(file_path):
        print(f"Файл не найден: {file_path}")
        return False
    file_size = os.path.getsize(file_path)
    if file_size > MAX_PAYLOAD_SIZE:
        print("Файл слишком большой (>1GB).")
        return False
    filename = os.path.basename(file_path)
    filename_bytes = filename.encode("utf-8")
    if len(filename_bytes) > 255:
        print("Имя файла слишком длинное (>255 байт).")
        return False
    with open(file_path, "rb") as f:
        file_data = f.read()
    total_data_len = file_size
    return _transmit_data(file_data, total_data_len, filename_bytes, mode='F', packet_blocks=packet_blocks)


def _collect_tx_constellation_symbols(data_bytes, header_bytes, filename_bytes, mode, packet_blocks):
    """
    Сбор символов созвездия на стороне передачи (TX).
    
    Выполняет модуляцию данных и собирает комплексные символы (I+Qj)
    для последующей визуализации созвездия.
    
    Параметры
    ----------
    data_bytes : bytes
        Данные для передачи
    header_bytes : bytes
        Заголовок передачи (тройной для первого пакета)
    filename_bytes : bytes
        Имя файла (пустое для текста)
    mode : str
        Режим передачи ('T' для текста, 'F' для файла)
    packet_blocks : int
        Количество логических блоков в пакете
    
    Возвращает
    -------
    list
        Список комплексных символов созвездия
    """
    constellation_symbols = []
    
    try:
        # Определяем параметры модуляции
        modulation = modem_config.MODULATION
        bits_per_symbol = modem_config.BITS_PER_SYMBOL
        ofdm_symbols_per_block = modem_config.OFDM_SYMBOLS_PER_BLOCK
        
        # Собираем все байты для передачи (заголовок + данные)
        all_bytes = header_bytes + data_bytes
        
        # Разбиваем на блоки по RS_DATA_BYTES
        blocks = [all_bytes[i:i+RS_DATA_BYTES] for i in range(0, len(all_bytes), RS_DATA_BYTES)]
        if len(blocks[-1]) < RS_DATA_BYTES:
            blocks[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks[-1]))
        
        # RS кодирование
        encoded_blocks = [rs.encode(b) for b in blocks]
        
        # Преобразуем в биты
        all_bits = []
        for enc_block in encoded_blocks:
            all_bits.extend(bytes_to_bits(enc_block))
        
        # Применяем интерливинг
        all_bits = interleave_bits(all_bits, block_size=modem_config.BITS_PER_OFDM_SYMBOL)
        
        # Модулируем биты в символы
        n_sub = Nsub
        
        if modulation == "BPSK":
            # BPSK: 1 бит на поднесущую
            for i in range(0, len(all_bits), n_sub):
                bits_chunk = all_bits[i:i+n_sub]
                if len(bits_chunk) < n_sub:
                    bits_chunk = np.concatenate([bits_chunk, np.zeros(n_sub - len(bits_chunk), dtype=int)])
                
                # BPSK модуляция: 0 -> +1, 1 -> -1
                symbols = 1 - 2 * bits_chunk.astype(complex)
                constellation_symbols.extend(symbols)
        else:
            # QPSK: 2 бита на поднесущую
            for i in range(0, len(all_bits), n_sub * 2):
                bits_chunk = all_bits[i:i+n_sub*2]
                if len(bits_chunk) < n_sub * 2:
                    bits_chunk = np.concatenate([bits_chunk, np.zeros(n_sub * 2 - len(bits_chunk), dtype=int)])
                
                # QPSK модуляция
                symbols = []
                for j in range(0, len(bits_chunk), 2):
                    b0, b1 = bits_chunk[j], bits_chunk[j+1]
                    # QPSK созвездие: (1+1j), (1-1j), (-1+1j), (-1-1j)
                    real = 1 - 2*b0
                    imag = 1 - 2*b1
                    symbols.append(complex(real, imag))
                
                # Нормализация
                symbols = np.array(symbols) / np.sqrt(2)
                constellation_symbols.extend(symbols)
        
        print(f"[TX-CONSTELLATION] Collected {len(constellation_symbols)} symbols for TX constellation")
        
    except Exception as e:
        print(f"[TX-CONSTELLATION] Error collecting symbols: {e}")
        import traceback
        traceback.print_exc()
    
    return constellation_symbols


def _collect_tx_ofdm_constellation_symbols(data_bytes, header_bytes, filename_bytes, mode, packet_blocks):
    """
    Сбор символов созвездия после OFDM-модуляции (FD символы до IFFT).
    
    Выполняет полную цепочку модуляции (RS-кодирование, интерливинг, BPSK/QPSK,
    OFDM-модуляция) и собирает комплексные поднесущие в частотной области
    после всех обработок (нормализация, ACE, фазы), но до IFFT.
    
    Параметры
    ----------
    data_bytes : bytes
        Данные для передачи
    header_bytes : bytes
        Заголовок передачи (тройной для первого пакета)
    filename_bytes : bytes
        Имя файла (пустое для текста)
    mode : str
        Режим передачи ('T' для текста, 'F' для файла)
    packet_blocks : int
        Количество логических блоков в пакете
    
    Возвращает
    -------
    list
        Список комплексных символов созвездия после OFDM-обработки
    """
    ofdm_constellation_symbols = []
    
    try:
        # Собираем все байты для передачи (заголовок + данные)
        all_bytes = header_bytes + data_bytes
        
        # Разбиваем на блоки по RS_DATA_BYTES
        blocks = [all_bytes[i:i+RS_DATA_BYTES] for i in range(0, len(all_bytes), RS_DATA_BYTES)]
        if len(blocks[-1]) < RS_DATA_BYTES:
            blocks[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks[-1]))
        
        # RS кодирование
        encoded_blocks = [rs.encode(b) for b in blocks]
        
        # Преобразуем в биты
        all_bits = []
        for enc_block in encoded_blocks:
            all_bits.extend(bytes_to_bits(enc_block))
        
        # Применяем интерливинг
        all_bits = interleave_bits(all_bits, block_size=modem_config.BITS_PER_OFDM_SYMBOL)
        
        # Вызываем build_data_td с collect_fd=True для сбора FD символов
        if len(all_bits) > 0:
            _, n_symbols, fd_symbols = build_data_td(all_bits, collect_fd=True)
            ofdm_constellation_symbols = list(fd_symbols)
            print(f"[TX-OFDM-CONSTELLATION] Collected {len(ofdm_constellation_symbols)} FD symbols "
                  f"from {n_symbols} OFDM symbols (modulation={modem_config.MODULATION})")
        else:
            print("[TX-OFDM-CONSTELLATION] No bits to process")
        
    except Exception as e:
        print(f"[TX-OFDM-CONSTELLATION] Error collecting OFDM symbols: {e}")
        import traceback
        traceback.print_exc()
    
    return ofdm_constellation_symbols


def _generate_pilot_symbols(n_symbols):
    """
    Генерация пилотных OFDM символов для настройки эквалайзера и AGC приёмника.
    
    Пилотный символ: все поднесущие = (1+1j)/√2 (как в преамбуле).
    Это позволяет приёмнику:
    1. Получить точную оценку канала Hk по 16 пилотам (вместо 2 в преамбуле)
    2. Плавно настроить AGC на известном сигнале
    3. Усреднить шум по 16 символам
    
    Параметры
    ----------
    n_symbols : int
        Количество пилотных OFDM символов
    
    Возвращает
    -------
    np.ndarray
        Сигнал пилотных символов во временной области
    """
    if n_symbols <= 0:
        return np.array([], dtype=np.float64)
    
    pilot_samples = []
    
    # Пилотный символ: все поднесущие = (1+1j)/√2
    pilot_syms = (1 + 1j) / np.sqrt(2) * np.ones(Nsub, dtype=complex)
    
    # Нормализуем к целевому RMS
    cur_rms = np.sqrt(np.mean(np.abs(pilot_syms)**2))
    if cur_rms > 0:
        pilot_syms = pilot_syms / cur_rms * SYMBOL_TX_TARGET
    
    print(f"[TX-PILOT] Pilot symbol: all subcarriers = (1+1j)/√2, RMS={cur_rms:.4f}, target={SYMBOL_TX_TARGET}")
    
    for i in range(n_symbols):
        # Используем ofdm_symbol() для генерации временного сигнала
        # ofdm_symbol() уже включает CP, поэтому добавляем x напрямую
        x = ofdm_symbol(pilot_syms)
        pilot_samples.append(x)
    
    return np.concatenate(pilot_samples)


def _insert_data_pilots(td_data, n_data_symbols):
    """
    Вставка пилотных символов внутрь потока данных через каждые PILOT_INTERVAL символов.
    
    Пилотный символ: все поднесущие = (1+1j)/√2 (как в преамбуле).
    Пилоты вставляются после каждых PILOT_INTERVAL символов данных для непрерывной
    подстройки эквалайзера и фазы приёмника.
    
    Параметры
    ----------
    td_data : np.ndarray
        Временной сигнал данных (все OFDM символы подряд)
    n_data_symbols : int
        Количество символов данных в td_data
    
    Возвращает
    -------
    np.ndarray
        Временной сигнал с вставленными пилотами (данные + пилоты)
    int
        Количество вставленных пилотных символов
    """
    if PILOT_INTERVAL <= 0 or n_data_symbols <= 0:
        return td_data, 0
    
    # Вычисляем количество пилотов: один пилот после каждых PILOT_INTERVAL символов
    n_pilots = n_data_symbols // PILOT_INTERVAL
    
    if n_pilots == 0:
        return td_data, 0
    
    # Генерируем один пилотный временной символ (с CP)
    pilot_syms = (1 + 1j) / np.sqrt(2) * np.ones(Nsub, dtype=complex)
    cur_rms = np.sqrt(np.mean(np.abs(pilot_syms)**2))
    if cur_rms > 0:
        pilot_syms = pilot_syms / cur_rms * SYMBOL_TX_TARGET
    pilot_td = ofdm_symbol(pilot_syms)  # уже включает CP
    
    # Разбиваем данные на блоки по SYMBOL_LEN сэмплов и вставляем пилоты
    result_parts = []
    symbols_processed = 0
    
    for i in range(0, n_data_symbols, PILOT_INTERVAL):
        # Берём блок из до PILOT_INTERVAL символов данных
        block_end = min(i + PILOT_INTERVAL, n_data_symbols)
        block_samples = td_data[i * SYMBOL_LEN : block_end * SYMBOL_LEN]
        result_parts.append(block_samples)
        symbols_processed = block_end
        
        # Вставляем пилот после блока (если это не последний неполный блок)
        if block_end < n_data_symbols:
            result_parts.append(pilot_td)
    
    # Добавляем оставшиеся сэмплы (если есть)
    remaining_start = symbols_processed * SYMBOL_LEN
    if remaining_start < len(td_data):
        result_parts.append(td_data[remaining_start:])
    
    result = np.concatenate(result_parts)
    
    print(f"[TX-DATA-PILOTS] Inserted {n_pilots} pilots into {n_data_symbols} data symbols "
          f"(every {PILOT_INTERVAL} symbols). Total: {n_data_symbols + n_pilots} symbols, "
          f"{len(result)} samples")
    
    return result, n_pilots


def _transmit_data(data_bytes, total_data_len, filename_bytes, mode='T', packet_blocks=DEFAULT_PACKET_BLOCKS):
    """Внутренняя функция передачи данных.
    
    packet_blocks - количество ЛОГИЧЕСКИХ блоков в пакете.
    Для QPSK: 1 логический блок = 1 физический OFDM символ.
    Для BPSK: 1 логический блок = OFDM_SYMBOLS_PER_BLOCK (2) физических OFDM символов.
    """
    # TX params
    LOGICAL_BLOCKS_PER_PACKET = packet_blocks
    # Переводим логические блоки в физические OFDM символы
    physical_symbols_per_packet = LOGICAL_BLOCKS_PER_PACKET * modem_config.OFDM_SYMBOLS_PER_BLOCK
    
    gap_ofdm_symbols = GAP_OFDM_SYMBOLS
    symbol_len_samples = SYMBOL_LEN
    gap_samples = gap_ofdm_symbols * symbol_len_samples

    PREROLL_SEC = 0.25
    POSTROLL_SEC = 0.50
    NOISE_LEVEL = 0.0  # Отключаем шум для режима Loop (для отладки)
    NOISE_SEED = 20231107

    rng_global = np.random.RandomState(NOISE_SEED)

    tx_type_bits = 0b11 if mode == "F" else (0b10 if mode == "T" else 0b00)
    version = 0

    # compute CRC32 над данными
    crc_val = zlib.crc32(data_bytes) & 0xFFFFFFFF
    transmission_header = build_header(b'F' if mode == "F" else b'T', total_data_len,
                                   filename_bytes=filename_bytes, packet_no=0,
                                   version=version, packet_blocks=packet_blocks, crc32=crc_val)
    remaining_data = data_bytes

    samples_packets = []
    packet_no = 0
    running_sample_offset = 0
    packet_preamble_offsets_no_preroll = []

    while True:
        is_first = (packet_no == 0)
        if is_first:
            header_bytes = transmission_header + transmission_header + transmission_header
        else:
            header_bytes = make_packet_header_bytes(packet_no, tx_type_bits)
        
        # allowed_physical_symbols - максимальное количество физических символов в пакете
        allowed_physical_symbols = physical_symbols_per_packet
        
        lo = 0
        hi = len(remaining_data)
        best_sz = 0
        best_td = None
        while lo <= hi:
            mid = (lo + hi) // 2
            test_bytes = header_bytes + remaining_data[:mid]
            td_local, n_physical_symbols = _bytes_to_ofdm_blocks_bytes(test_bytes)
            # Сравниваем физические символы с допустимым количеством
            if n_physical_symbols <= allowed_physical_symbols:
                best_sz = mid
                best_td = td_local
                lo = mid + 1
            else:
                hi = mid - 1
                
        if best_td is None:
            best_sz = 0
            td_local, n_physical_symbols = _bytes_to_ofdm_blocks_bytes(header_bytes)
            if n_physical_symbols > allowed_physical_symbols:
                samples_keep = allowed_physical_symbols * symbol_len_samples
                td_local = td_local[:samples_keep]
            best_td = td_local

        td_now = best_td
        if SYMBOL_LEN <= 0:
            n_physical_now = 0
        else:
            n_physical_now = len(td_now) // SYMBOL_LEN
            
        # Дополняем пакет до полного размера (в физических символах)
        if n_physical_now < allowed_physical_symbols:
            needed_physical_symbols = allowed_physical_symbols - n_physical_now
            filler_bytes_len = needed_physical_symbols * RS_DATA_BYTES
            td_filler, nblk_f = _bytes_to_ofdm_blocks_bytes(b'\x00' * filler_bytes_len)
            if nblk_f >= needed_physical_symbols and len(td_filler) >= needed_physical_symbols * SYMBOL_LEN:
                td_now = np.concatenate((td_now, td_filler[:needed_physical_symbols * SYMBOL_LEN]))
                n_physical_now = allowed_physical_symbols
            else:
                pad_samples = needed_physical_symbols * SYMBOL_LEN
                td_now = np.concatenate((td_now, np.zeros(pad_samples, dtype=td_now.dtype)))
                n_physical_now = allowed_physical_symbols
        elif n_physical_now > allowed_physical_symbols:
            td_now = td_now[:allowed_physical_symbols * SYMBOL_LEN]
            n_physical_now = allowed_physical_symbols

        # === Вставка пилотных символов внутрь потока данных ===
        # Пилоты вставляются через каждые PILOT_INTERVAL символов данных
        # для непрерывной подстройки эквалайзера приёмника
        n_data_pilots = 0
        if PILOT_INTERVAL > 0 and n_physical_now > 0:
            td_now, n_data_pilots = _insert_data_pilots(td_now, n_physical_now)
            # Обновляем количество физических символов (данные + пилоты)
            n_physical_now = n_physical_now + n_data_pilots

        # Для отладки: вычисляем, сколько логических блоков мы отправили
        n_logical_sent = n_physical_now // modem_config.OFDM_SYMBOLS_PER_BLOCK
        print(f"[TX-DBG] packet_no={packet_no} is_first={is_first} payload_bytes={best_sz} "
              f"physical_symbols={n_physical_now} (data={n_physical_now - n_data_pilots} pilots={n_data_pilots}) "
              f"logical_blocks={n_logical_sent} header_first16={header_bytes[:16].hex()}")

        # Добавляем пилотные символы перед преамбулой первого пакета
        # для настройки AGC и эквалайзера приёмника
        # Пилотные символы дают точную оценку канала Hk и плавную настройку AGC
        if is_first and PREAMBLE_PILOT_SYMBOLS > 0:
            pilot = _generate_pilot_symbols(PREAMBLE_PILOT_SYMBOLS)
            print(f"[TX] Pilot: {PREAMBLE_PILOT_SYMBOLS} OFDM symbols ({len(pilot)} samples) before preamble")
        else:
            pilot = np.array([], dtype=np.float64)

        preamble = build_preamble()
        packet_samples = np.concatenate((pilot, preamble, td_now)) if len(td_now) > 0 else np.concatenate((pilot, preamble))
        packet_preamble_offsets_no_preroll.append(running_sample_offset + len(pilot))

        def make_gap_noise(n_samps):
            if NOISE_LEVEL is None or NOISE_LEVEL <= 0:
                return np.zeros(n_samps, dtype=np.float64)
            w = rng_global.normal(loc=0.0, scale=1.0, size=n_samps).astype(np.float64)
            cur_rms = np.sqrt(np.mean(w**2)) if w.size > 0 else 1.0
            if cur_rms == 0:
                return np.zeros(n_samps, dtype=np.float64)
            pkt_peak = np.max(np.abs(packet_samples)) if np.max(np.abs(packet_samples)) > 0 else 1.0
            noise_rms = NOISE_LEVEL * pkt_peak
            w = w * (noise_rms / cur_rms)
            return w.astype(packet_samples.dtype)

        gap_noise = make_gap_noise(gap_samples)
        packet_samples = np.concatenate((packet_samples, gap_noise))

        samples_packets.append(packet_samples)

        packet_sample_count = len(packet_samples)
        print(f"[TX-DBG] packet {packet_no} sample_offset_start(no_preroll)={running_sample_offset} sample_count={packet_sample_count}")
        running_sample_offset += packet_sample_count

        remaining_data = remaining_data[best_sz:]
        packet_no += 1

        if len(remaining_data) == 0:
            break
        if packet_no > 1000000:
            print("[TX] too many packets, aborting")
            break

    tx_packets_concat = np.concatenate(samples_packets) if len(samples_packets) > 0 else np.array([], dtype=float)
    preroll_len = int(PREROLL_SEC * fs)
    postroll_len = int(POSTROLL_SEC * fs)
    preroll = np.zeros(preroll_len, dtype=tx_packets_concat.dtype)
    postroll = np.zeros(postroll_len, dtype=tx_packets_concat.dtype)
    if NOISE_LEVEL is not None and NOISE_LEVEL > 0.0:
        rnd = np.random.RandomState(NOISE_SEED)
        tx_peak = np.max(np.abs(tx_packets_concat)) if np.max(np.abs(tx_packets_concat)) > 0 else 1.0
        noise_rms = float(NOISE_LEVEL) * float(tx_peak)
        def make_noise(n_samples):
            w = rnd.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float64)
            cur_rms = np.sqrt(np.mean(w**2)) if w.size > 0 else 1.0
            if cur_rms == 0:
                return np.zeros(n_samples, dtype=np.float64)
            w = w * (noise_rms / cur_rms)
            return w
        preroll_noise = make_noise(preroll_len)
        postroll_noise = make_noise(postroll_len)
        preroll = preroll_noise.astype(tx_packets_concat.dtype)
        postroll = postroll_noise.astype(tx_packets_concat.dtype)

    tx_out = np.concatenate((preroll, tx_packets_concat, postroll))
    tx_clipped, achieved_crest = apply_softclip_to_target_crest(tx_out, target_db=TARGET_CREST_DB)
    print(f"[SOFTCLIP] achieved crest = {achieved_crest:.3f} dB (target {TARGET_CREST_DB} dB)")
    
    # === Отключено: Диаграммы переданного сигнала ===
    # if PLOTTING_AVAILABLE:
    #     try:
    #         from plot_utils import plot_tx_diagrams
    #         plot_tx_diagrams(tx_clipped, fs)
    #     except Exception as e:
    #         print(f"[TX] Ошибка при сохранении диаграмм: {e}")
    
    # === Отключено: Созвездие передачи (TX constellation) ===
    # if PLOTTING_AVAILABLE and PLOT_CONSTELLATION:
    #     try:
    #         from plot_utils import plot_constellation
    #         tx_constellation_symbols = _collect_tx_constellation_symbols(
    #             data_bytes, transmission_header, filename_bytes, mode, packet_blocks
    #         )
    #         if tx_constellation_symbols and len(tx_constellation_symbols) > 0:
    #             plot_constellation(
    #                 tx_constellation_symbols,
    #                 title="TX Constellation",
    #                 filename=CONSTELLATION_TX_FILENAME,
    #                 use_gradient=CONSTELLATION_USE_GRADIENT,
    #                 modulation_type=modem_config.MODULATION
    #             )
    #             print(f"[TX] Constellation diagram saved to {CONSTELLATION_TX_FILENAME}")
    #     except Exception as e:
    #         print(f"[TX] Ошибка при сохранении созвездия: {e}")
    
    # === Отключено: Созвездие после OFDM-модуляции ===
    # if PLOTTING_AVAILABLE and PLOT_CONSTELLATION and PLOT_CONSTELLATION_TX_OFDM:
    #     try:
    #         from plot_utils import plot_constellation
    #         tx_ofdm_constellation_symbols = _collect_tx_ofdm_constellation_symbols(
    #             data_bytes, transmission_header, filename_bytes, mode, packet_blocks
    #         )
    #         if tx_ofdm_constellation_symbols and len(tx_ofdm_constellation_symbols) > 0:
    #             plot_constellation(
    #                 tx_ofdm_constellation_symbols,
    #                 title="TX Constellation (after OFDM, before clipping)",
    #                 filename=CONSTELLATION_TX_OFDM_FILENAME,
    #                 use_gradient=CONSTELLATION_USE_GRADIENT,
    #                 modulation_type=modem_config.MODULATION
    #             )
    #             print(f"[TX] OFDM constellation diagram saved to {CONSTELLATION_TX_OFDM_FILENAME}")
    #
    #             if CONSTELLATION_TX_OFDM_COMPENSATE_PHASE:
    #                 print(f"[TX] Generating phase-compensated OFDM constellation diagram...")
    #                 subc_phases = modem_config.subc_phases
    #                 if subc_phases is not None and len(subc_phases) > 0:
    #                     n_ofdm_symbols = len(tx_ofdm_constellation_symbols) // Nsub
    #                     phases_repeated = np.tile(subc_phases, n_ofdm_symbols)
    #                     plot_constellation(
    #                         tx_ofdm_constellation_symbols,
    #                         title="TX Constellation (after OFDM, phase compensated)",
    #                         filename=CONSTELLATION_TX_OFDM_COMPENSATED_FILENAME,
    #                         use_gradient=CONSTELLATION_USE_GRADIENT,
    #                         modulation_type=modem_config.MODULATION,
    #                         phase_compensation=phases_repeated
    #                     )
    #                     print(f"[TX] Phase-compensated OFDM constellation saved to {CONSTELLATION_TX_OFDM_COMPENSATED_FILENAME}")
    #     except Exception as e:
    #         print(f"[TX] Ошибка при сохранении OFDM созвездия: {e}")
    
    max_abs = np.max(np.abs(tx_clipped)) if tx_clipped.size else 0.0
    if max_abs == 0:
        scale_to_int16 = 1.0
    else:
        scale_to_int16 = 1 / max_abs
    tx_int16 = (tx_clipped * scale_to_int16 * np.iinfo(np.int16).max).astype(np.int16)
    from wav_utils import wavfile
    wavfile.write("ofdm_acoustic_tx_with_noise.wav", fs, tx_int16)
    packet_preamble_offsets_with_preroll = [preroll_len + int(x) for x in packet_preamble_offsets_no_preroll]
    print("TX saved to ofdm_acoustic_tx_with_noise.wav (packetized, gaps inserted)")
    print(f"[TX-DBG] total_packets={packet_no} total_samples={len(tx_packets_concat)} preroll={preroll_len} postroll={postroll_len}")
    print(f"[TX-DBG] transmission_header_hex={transmission_header.hex()[:256]}")
    print("[TX-DBG] preamble positions (sample indices in WAV, including preroll):")
    for i, pos in enumerate(packet_preamble_offsets_with_preroll):
        print(f"  packet {i}: preamble_sample_index={pos}")

    # Play the generated signal
    print(f"[TX] Preparing to play sound...")
    print(f"[TX-DBG] Signal amplitude: max_abs={max_abs:.6f}, RMS={np.sqrt(np.mean(tx_clipped**2)):.6f}")
    print(f"[TX-DBG] Signal duration: {len(tx_clipped)/fs:.3f} seconds")

    from modem_config import get_audio
    audio = get_audio()
    if audio is not None:
        print(f"[TX] Playing through audio backend: {audio.__class__.__name__}")
        tx_float32 = tx_clipped.astype(np.float32)
        max_val = np.max(np.abs(tx_float32))
        if max_val > 0:
            tx_float32 = tx_float32 / max_val
        audio.play(tx_float32, samplerate=fs)
        print(f"[TX] Playback completed")
    else:
        print("[TX-ERR] Audio backend not available!")

    return True


def _bytes_to_ofdm_blocks_bytes(bstream: bytes):
    """
    Вспомогательная функция для передачи: разбивка потока байт на блоки OFDM.
    
    Алгоритм:
    1. Разбивает поток байт на блоки по RS_DATA_BYTES (8 байт)
    2. Дополняет последний блок нулями, если он неполный
    3. Применяет RS-кодирование к каждому блоку (получаем 12 байт = 96 бит на блок)
    4. Преобразует байты в биты
    5. Применяет интерливинг для повышения устойчивости к пачкам ошибок
    
    Возвращает (time_domain_samples, n_physical_symbols):
    - time_domain_samples: сигнал во временной области
    - n_physical_symbols: количество ФИЗИЧЕСКИХ OFDM символов
    
    Особенность для BPSK:
    - BPSK: 1 бит на поднесущую, 48 поднесущих = 48 бит/OFDM символ
    - Для формирования одного RS слова (96 бит) требуется 2 OFDM символа BPSK
    - Интерливинг работает с block_size=BITS_PER_OFDM_SYMBOL (48 для BPSK, 96 для QPSK)
    """
    if len(bstream) == 0:
        return np.array([], dtype=float), 0
    blocks_local = [bstream[i:i+RS_DATA_BYTES] for i in range(0, len(bstream), RS_DATA_BYTES)]
    if len(blocks_local[-1]) < RS_DATA_BYTES:
        blocks_local[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks_local[-1]))
    encoded = [rs.encode(b) for b in blocks_local]
    bits_blocks_local = [bytes_to_bits(b) for b in encoded]
    
    data_bits_local = np.concatenate(bits_blocks_local) if len(bits_blocks_local) > 0 else np.array([], dtype=int)
    # Применяем интерливинг для повышения устойчивости к пачкам ошибок
    # block_size равен BITS_PER_OFDM_SYMBOL (48 для BPSK, 96 для QPSK)
    data_bits_local = interleave_bits(data_bits_local, block_size=modem_config.BITS_PER_OFDM_SYMBOL)
    
    
    td_local, n_physical_symbols = build_data_td(data_bits_local) if data_bits_local.size > 0 else (np.array([], dtype=float), 0)
    return td_local, n_physical_symbols

# Alias for backward compatibility with tests
bytes_to_ofdm_blocks_bytes = _bytes_to_ofdm_blocks_bytes
