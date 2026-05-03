"""
Модуль приема OFDM Acoustic Modem.
Содержит функции синхронизации, декодирования пакетов и живого приема.
"""

import numpy as np
import zlib
import modem_config
from modem_config import (Nfft, Ncp, Nsub, subc_inds, fs, SYMBOL_LEN, DEFAULT_PACKET_BLOCKS,
                           RS_CW_BITS, RS_DATA_BYTES,
                           RS_CW_BYTES, rs, SYMBOL_TARGET_RMS, AGC_ALPHA, AGC_DEBUG, MIN_RMS,
                           PLOTTING_AVAILABLE, _MAX_RS_FAIL_PRINTS_GLOBAL, SYNC_WINDOW_HALF, TARGET_RMS)

# Параметр эквалайзера для совместимости с run_loop.py
eq_alpha = 0.02
from modem_modulation import (qpsk_demap, bpsk_demap, ofdm_symbol, build_preamble, bytes_to_bits, bits_to_bytes,
                           sync_by_corr, deinterleave_bits, AdaptiveEqualizer)

from modem_packet import parse_header, build_header, make_packet_header_bytes, simulate_packet_positions

# Глобальные переменные состояния (инициализируются при работе)
rx = None
abs_corr = None
# subc_phases импортируется из modem_config, не переопределяем здесь!
preamble_td = None
last_agc_rms = 1.0  # Инициализируем значением по умолчанию, а не None
post_sync = False
sync_sample_abs = None
last_packet_end_sample = None
last_packet_rs_ok = None
pkt0_header_offset = None
pkt0_header_bytes = None
rx_syms_list = []
Hk_smooth_list = []
equalizer_history_list = []  # Список для хранения истории эквалайзера по пакетам
last_packet_used_pre = None
# Глобальный эквалайзер для сохранения состояния между пакетами
global_equalizer = None

# Список для хранения истории AGC по всем символам всех пакетов
agc_history_list = []
# Глобальный счётчик символов (нумеруется по всем пакетам)
_global_symbol_counter = 0

# Список для хранения символов созвездия (для градиентного вывода)
rx_constellation_symbols = []


# -----------------------
# Вспомогательная функция для декодирования с конкретной модуляцией
# -----------------------
def _try_decode_with_modulation(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, modulation):
    """
    Попытка декодирования пакета с конкретной модуляцией.
    Использует отдельный экземпляр эквалайзера для каждой попытки.
    
    packet_blocks_expected - количество ЛОГИЧЕСКИХ блоков.
    
    Возвращает (decoded_bytes, rs_ok_count, used_preamble, equalizer_instance) или None при ошибке.
    """
    global abs_corr, rx, preamble_td, last_agc_rms, agc_history_list, _global_symbol_counter
    
    if 'abs_corr' not in globals() or 'rx' not in globals():
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
    
    try:
        # Оценка частотной ошибки и канала (общая для всех модуляций)
        zc_seq_ideal = (np.exp(-1j * np.pi * 1 * np.arange(Nsub) * (np.arange(Nsub) + 1) / float(Nsub)))
        zc_seq_ideal = zc_seq_ideal / np.sqrt(np.mean(np.abs(zc_seq_ideal)**2))
        S_zc_fd = np.zeros(Nfft, dtype=complex)
        S_zc_fd[subc_inds] = zc_seq_ideal
        S_zc_fd[-subc_inds] = np.conj(zc_seq_ideal)
    except Exception:
        S_zc_fd = None
    
    try:
        rx_zc1 = rx[pref_abs + Ncp : pref_abs + Ncp + Nfft]
        rx_zc2 = rx[pref_abs + SYMBOL_LEN + Ncp : pref_abs + SYMBOL_LEN + Ncp + Nfft]
        cross = np.vdot(rx_zc1, rx_zc2)
        delta_phi = np.angle(cross)
        T_between = SYMBOL_LEN / float(fs)
        f_err_loc = delta_phi / (2.0 * np.pi * T_between)
        
        pilot1_start = pref_abs + 2*SYMBOL_LEN
        rx_pre1 = rx[pilot1_start + Ncp : pilot1_start + Ncp + Nfft]
        pilot2_start = pilot1_start + SYMBOL_LEN
        rx_pre2 = rx[pilot2_start + Ncp : pilot2_start + Ncp + Nfft]
        t1_offset = (pilot1_start) / float(fs)
        t2_offset = (pilot2_start) / float(fs)
        time_vec = np.arange(Nfft) / float(fs)
        rx_pre1_corr = rx_pre1 * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t1_offset + time_vec))
        rx_pre2_corr = rx_pre2 * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t2_offset + time_vec))
        R1t = np.fft.fft(rx_pre1_corr) / Nfft
        R2t = np.fft.fft(rx_pre2_corr) / Nfft
        
        phi_est = 0.0
        if S_zc_fd is not None:
            try:
                Rzc = np.fft.fft(rx_zc1)
                phi_est = np.angle(np.vdot(S_zc_fd, Rzc))
            except Exception:
                phi_est = 0.0
        if phi_est != 0.0:
            R1t = R1t * np.exp(-1j * phi_est)
            R2t = R2t * np.exp(-1j * phi_est)
        
        S_ref = np.fft.fft(preamble_td[2*SYMBOL_LEN + Ncp : 2*SYMBOL_LEN + Ncp + Nfft])
        Hk_est = (R1t[subc_inds] / S_ref[subc_inds] + R2t[subc_inds] / S_ref[subc_inds]) / 2
        Hk_mag = np.clip(np.median(np.abs(Hk_est)) if hasattr(np, 'median') else np.mean(np.abs(Hk_est)), 1/2.0, None)
        Hk_s = Hk_mag * np.exp(1j*np.angle(Hk_est))
        
        # Создаем отдельный экземпляр эквалайзера для этой попытки
        equalizer = AdaptiveEqualizer(initial_Hk=Hk_s, alpha=0.02, modulation=modulation)
        print(f"[EQ] Modulation {modulation}: AdaptiveEqualizer initialized with alpha=0.02")
        
    except Exception as e:
        print(f"[RX-DBG-DECODE] Exception in channel estimation for {modulation}: {e}")
        return None
    
    try:
        pre_segment = rx[pref_abs : pref_abs + len(preamble_td)]
        pre_rms = np.sqrt(np.mean(pre_segment**2)) if pre_segment.size > 0 else MIN_RMS
        if pre_rms < MIN_RMS:
            pre_rms = MIN_RMS
        packet_gain = 0.5 / pre_rms
    except Exception:
        packet_gain = 1.0
    
    # Переводим логические блоки в физические символы
    physical_symbols_expected = packet_blocks_expected * ofdm_symbols_per_block
    
    pkt_data_start = pref_abs + len(preamble_td)
    pkt_payload_samples = physical_symbols_expected * SYMBOL_LEN
    seg = packet_gain * rx[pkt_data_start : pkt_data_start + pkt_payload_samples]
    if len(seg) < pkt_payload_samples:
        return None
    frames = seg.reshape(physical_symbols_expected, SYMBOL_LEN)
    
    rx_syms_pkt_list = []
    
    for idxf, fr in enumerate(frames):
        frame_start_abs = pkt_data_start + idxf * SYMBOL_LEN
        t_frame_start = frame_start_abs / float(fs)
        useful = fr[Ncp:]
        tv = t_frame_start + np.arange(Nfft) / float(fs)
        useful_corr = useful * np.exp(-1j * 2.0 * np.pi * f_err_loc * tv)
        
        cur_rms = np.sqrt(np.mean(np.abs(useful)**2)) if useful.size > 0 else 1e-12
        est_rms = (1.0 - AGC_ALPHA) * last_agc_rms + AGC_ALPHA * cur_rms
        last_agc_rms = est_rms
        if est_rms < 1e-12:
            est_rms = 1e-12
        gain_sym = SYMBOL_TARGET_RMS / est_rms
        
        # Сохраняем данные AGC для последующего построения графика
        agc_history_list.append({
            'symbol_idx': _global_symbol_counter,
            'pkt_idx': packet_idx,
            'frame_idx': idxf,
            'cur_rms': cur_rms,
            'est_rms': est_rms,
            'gain_sym': gain_sym
        })
        _global_symbol_counter += 1
        
        useful = useful * gain_sym
        
        try:
            F = np.fft.fft(useful_corr) / Nfft
            subc = equalizer.process(F[subc_inds])
        except Exception:
            subc = np.zeros(Nsub, dtype=complex)
        
        if modem_config.subc_phases is not None and np.any(modem_config.subc_phases != 0):
            subc = subc * np.exp(-1j * modem_config.subc_phases)
        
        rx_syms_pkt_list.append(subc)
        
        # Сохраняем символы для градиентного созвездия
        global rx_constellation_symbols
        rx_constellation_symbols.extend(subc)
    
    if len(rx_syms_pkt_list) == 0:
        return None
    
    try:
        rx_syms_pkt = np.concatenate(rx_syms_pkt_list)
    except Exception:
        return None
    
    if phi_est != 0.0:
        rx_syms_pkt = rx_syms_pkt * np.exp(-1j * phi_est)
    
    # Демаппинг в зависимости от модуляции
    if modulation == "BPSK":
        bits_pkt = bpsk_demap(rx_syms_pkt)
        print(f"[RX] Using {modulation} demapping")
    else:
        bits_pkt = qpsk_demap(rx_syms_pkt)
        print(f"[RX] Using {modulation} demapping")
    
    # Применяем деинтерливинг с правильным размером блока для данной модуляции
    bits_pkt = deinterleave_bits(bits_pkt, block_size=bits_per_ofdm_symbol)
    
    # RS декодирование
    cw_bits = RS_CW_BITS
    n_cw = len(bits_pkt) // cw_bits
    rs_ok = 0
    decoded_blocks = []
    
    for ci in range(n_cw):
        bstart = ci * cw_bits
        bbits = bits_pkt[bstart:bstart+cw_bits]
        if len(bbits) < cw_bits:
            bbits = np.concatenate((bbits, np.zeros(cw_bits - len(bbits), dtype=int)))
        bts = bits_to_bytes(bbits)
        try:
            msg = rs.decode(bts)[0]
            rs_ok += 1
            decoded_blocks.append((msg[:RS_DATA_BYTES], True))  # Успешное декодирование
        except Exception as e:
            msg = b'\x00' * RS_DATA_BYTES
            decoded_blocks.append((msg, False))  # Ошибка декодирования
    
    # Возвращаем результат с экземпляром эквалайзера
    return (decoded_blocks, rs_ok, pref_abs, equalizer)


# -----------------------
# decode_packet_at_candidate
# -----------------------
def decode_packet_at_candidate(pref_abs, packet_blocks_expected, packet_idx=0, bytes_before_packet=0, expected_total=0):
    """Декодирование пакета по кандидату синхронизации.
    
    packet_blocks_expected - количество ЛОГИЧЕСКИХ блоков.
    """
    global abs_corr, rx, preamble_td, last_agc_rms, global_equalizer
    best_result = (b'',0, None)
    
    if 'abs_corr' not in globals() or 'rx' not in globals():
        return best_result
    
    if '_rs_fail_prints_count' not in globals():
        globals()['_rs_fail_prints_count'] = 0
    
    # Для первого пакета пробуем обе модуляции параллельно
    if packet_idx == 0:
        print(f"[RX] First packet: trying both BPSK and QPSK modulation...")
        results = []
        
        for try_mod in ["QPSK", "BPSK"]:
            print(f"[RX] Trying {try_mod} modulation...")
            result = _try_decode_with_modulation(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, try_mod)
            if result is not None:
                decoded_blocks, rs_ok, used_pre, eq_instance = result
                print(f"[RX] {try_mod}: RS_OK={rs_ok}, blocks={len(decoded_blocks)}")
                results.append((decoded_blocks, rs_ok, used_pre, eq_instance, try_mod))
        
        if results:
            # Выбираем лучший результат по количеству RS_OK
            best = max(results, key=lambda x: x[1])
            decoded_blocks, rs_ok, used_pre, eq_instance, best_mod = best
            
            print(f"[RX] Best modulation: {best_mod} with RS_OK={rs_ok}")
            
            # Устанавливаем глобальный эквалайзер от лучшей попытки
            global_equalizer = eq_instance
            
            # Устанавливаем глобальные настройки модуляции
            modem_config.MODULATION = best_mod
            if best_mod == "BPSK":
                modem_config.BITS_PER_SYMBOL = 1
                modem_config.OFDM_SYMBOLS_PER_BLOCK = 2
            else:
                modem_config.BITS_PER_SYMBOL = 2
                modem_config.OFDM_SYMBOLS_PER_BLOCK = 1
            modem_config.BITS_PER_OFDM_SYMBOL = modem_config.Nsub * modem_config.BITS_PER_SYMBOL
            
            # Формируем возвращаемый результат
            if packet_idx == 0:
                expected_hdr_len = 3 * 64
                cw_per_hdr = (64 + RS_DATA_BYTES - 1) // RS_DATA_BYTES
                if len(decoded_blocks) < cw_per_hdr * 3:
                    for _p in range(len(decoded_blocks), cw_per_hdr * 3):
                        decoded_blocks.append((b'\x00'*RS_DATA_BYTES, False))
                
                header_by_cw = bytearray(expected_hdr_len)
                for i in range(cw_per_hdr):
                    chosen_msg = None
                    candidates_idx = [i, i + cw_per_hdr, i + 2 * cw_per_hdr]
                    for ci in candidates_idx:
                        if ci < len(decoded_blocks):
                            msg_bytes, ok_flag = decoded_blocks[ci]
                            if ok_flag:
                                chosen_msg = msg_bytes[:RS_DATA_BYTES]
                                break
                    if chosen_msg is None:
                        for ci in candidates_idx:
                            if ci < len(decoded_blocks):
                                chosen_msg = decoded_blocks[ci][0][:RS_DATA_BYTES]
                                break
                    if chosen_msg is None:
                        chosen_msg = b'\x00' * RS_DATA_BYTES
                    start_b = i * RS_DATA_BYTES
                    header_by_cw[start_b:start_b+RS_DATA_BYTES] = chosen_msg
                
                single_selected_hdr = bytes(header_by_cw[:64])
                header_by_cw = bytearray(single_selected_hdr + single_selected_hdr + single_selected_hdr)
                ret_bytes = bytes(header_by_cw) + (b"".join([b for (b,ok) in decoded_blocks])[expected_hdr_len:])
                globals()['pkt0_header_bytes'] = bytes(header_by_cw[:64])
            else:
                ret_bytes = b"".join([b for (b,ok) in decoded_blocks])
            
            return ret_bytes, rs_ok, used_pre
        else:
            print("[RX-ERR] Both BPSK and QPSK failed for first packet")
            return best_result
    else:
        # Для последующих пакетов используем известную модуляцию
        modulation = modem_config.MODULATION
        print(f"[RX] Packet {packet_idx}: using known modulation {modulation}")
        
        result = _try_decode_with_modulation(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, modulation)
        if result is not None:
            decoded_blocks, rs_ok, used_pre, eq_instance = result
            
            # Обновляем глобальный эквалайзер
            global_equalizer = eq_instance
            
            ret_bytes = b"".join([b for (b,ok) in decoded_blocks])
            return ret_bytes, rs_ok, used_pre
        else:
            return best_result


# -----------------------
# Функция приема из WAV-файла
# -----------------------
def receive_from_file(wav_path):
    """Прием из WAV-файла."""
    from wav_utils import wavfile
    from signal_utils import fftconvolve, find_peaks
    
    # Инициализируем фазы перед приемом
    from modem_config import init_phases
    init_phases()
    
    _, wavd = wavfile.read(wav_path)
    sig = wavd[:,0] if wavd.ndim > 1 else wavd
    global rx, abs_corr, preamble_td
    rx = sig.astype(float) / np.iinfo(wavd.dtype).max
    
    # Предварительный AGC для слабых сигналов перед поиском преамбулы
    signal_rms = np.sqrt(np.mean(rx**2))
    print(f"[AGC-PRE] До AGC: signal_rms={signal_rms:.6f}")
    if signal_rms > 0:
        rx = rx * (TARGET_RMS / signal_rms)
        print(f"[AGC-PRE] После AGC: target={TARGET_RMS}, gain={TARGET_RMS/signal_rms:.3f}, новый RMS={np.sqrt(np.mean(rx**2)):.6f}")
    
    # Получаем актуальную преамбулу
    from modem_modulation import build_preamble
    preamble_td_local = build_preamble()
    preamble_td = preamble_td_local  # Обновляем глобальную переменную
    corr = fftconvolve(rx, preamble_td_local[::-1], mode='valid')
    abs_corr = np.abs(corr)
    
    if abs_corr.size == 0:
        print("[RX-ERR] correlation empty, cannot find preambles")
        return False
        
    peak_val = np.max(abs_corr)
    threshold = 0.5 * peak_val if peak_val != 0 else 0.0
    peaks, props = find_peaks(abs_corr, height=threshold, distance=len(preamble_td_local)//2)
    peak_heights = props['peak_heights'] if 'peak_heights' in props else abs_corr[peaks]
    candidates = sorted(zip(peaks, peak_heights), key=lambda x: -x[1])
    print(f"[RX-DBG] found {len(candidates)} preamble peak candidates (threshold={threshold:.6g})")
    
    # Используем фиксированную позицию преамбулы
    expected_sync = 12000
    # Отладочный вывод: значение окна поиска (хотя в текущей реализации используется фиксированная позиция)
    print(f"[SYNC] SYNC_WINDOW_HALF={SYNC_WINDOW_HALF}, фиксированная позиция синхронизации: {expected_sync}")
    sync_idx = expected_sync

    pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = decode_packet_at_candidate(sync_idx, DEFAULT_PACKET_BLOCKS, packet_idx=0, bytes_before_packet=0, expected_total=0)
    if pkt0_decoded is None or len(pkt0_decoded) < 64:
        print("[RX-ERR] unable to decode first packet header")
        return False

    header64 = pkt0_decoded[:64]
    try:
        hdr = parse_header(header64)
        
        # Проверяем валидность заголовка
        if not hdr.get('is_valid', False):
            print(f"[RX-ERR] Invalid header (is_valid=False), discarding")
            return False
            
    except Exception as e:
        print("[RX-ERR] parse_header failed:", e)
        return False
        
    global post_sync, sync_sample_abs
    post_sync = True
    sync_sample_abs = int(pkt0_used_pre) if pkt0_used_pre is not None else int(sync_idx)
    
    mode_rx = hdr['mode']
    total_sz = hdr['data_len']
    fname = hdr['filename']
    packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)
    
    # Используем модуляцию из заголовка
    modulation_from_header = hdr.get('modulation', 'QPSK')
    print(f"[RX] Modulation from header: {modulation_from_header}")
    modem_config.MODULATION = modulation_from_header
    if modulation_from_header == "BPSK":
        modem_config.BITS_PER_SYMBOL = 1
        modem_config.OFDM_SYMBOLS_PER_BLOCK = 2
    else:
        modem_config.BITS_PER_SYMBOL = 2
        modem_config.OFDM_SYMBOLS_PER_BLOCK = 1
    modem_config.BITS_PER_OFDM_SYMBOL = modem_config.Nsub * modem_config.BITS_PER_SYMBOL
    
    positions_no_preroll = simulate_packet_positions(total_sz, fname.encode('utf-8') if fname else b'', (mode_rx == b'T'),
                                             packet_blocks_from_hdr, preamble_len=len(preamble_td_local), symbol_len=SYMBOL_LEN,
                                             gap_samples=0)
    if len(positions_no_preroll) == 0:
        print("[RX-ERR] simulate_packet_positions returned no positions")
        return False
        
    base_abs = sync_idx - positions_no_preroll[0]
    expected_abs = [base_abs + int(x) for x in positions_no_preroll]
    
    assembled_chunks = []
    bytes_collected = 0
    expected_total = total_sz
    packet_blocks_val = packet_blocks_from_hdr
    
    for pkt_idx, pref in enumerate(expected_abs):
        if pkt_idx == 0:
            # Используем уже декодированные данные первого пакета, чтобы избежать двойного декодирования
            decoded_bytes = pkt0_decoded
            rs_ok_count = pkt0_rs_ok
            used_preamble = pkt0_used_pre
            print(f"[RX-DBG] pkt{pkt_idx}: pref={pref} used_pre={used_preamble} RS_OK={rs_ok_count} bytes={len(decoded_bytes)} (reused)")
        else:
            decoded_bytes, rs_ok_count, used_preamble = decode_packet_at_candidate(pref, packet_blocks_val, packet_idx=pkt_idx, bytes_before_packet=bytes_collected, expected_total=expected_total)
            print(f"[RX-DBG] pkt{pkt_idx}: pref={pref} used_pre={used_preamble} RS_OK={rs_ok_count} bytes={len(decoded_bytes)}")
        
        if pkt_idx == 0:
            if len(decoded_bytes) < 64:
                print("[RX-ERR] first packet decoded <64 bytes")
                return False
            payload_start = 3 * 64
            if payload_start >= len(decoded_bytes):
                payload_chunk = b''
            else:
                payload_chunk = decoded_bytes[payload_start:]
            need = max(0, expected_total - bytes_collected)
            if need <= 0:
                payload_chunk = b''
            else:
                payload_chunk = payload_chunk[:need]
            assembled_chunks.append(payload_chunk)
            bytes_collected += len(payload_chunk)
        else:
            if len(decoded_bytes) >= 4:
                payload_chunk = decoded_bytes[4:]
                assembled_chunks.append(payload_chunk)
                bytes_collected += len(payload_chunk)
                
        print(f"[RX-INFO] collected {bytes_collected}/{expected_total} bytes")
        if bytes_collected >= expected_total:
            break
    
    assembled = b"".join(assembled_chunks)[:expected_total]
    
    # CRC check
    try:
        hdr_crc = hdr.get('crc32', None)
        if hdr_crc is not None and hdr_crc != 0:  # Запрещаем нулевой CRC
            calc_crc = zlib.crc32(assembled) & 0xFFFFFFFF
            if calc_crc == int(hdr_crc):
                print(f"[RX-CRC] OK: CRC32 matched (0x{calc_crc:08X})")
            else:
                print(f"[RX-CRC] MISMATCH: received 0x{int(hdr_crc):08X}, calculated 0x{calc_crc:08X}")
        else:
            print(f"[RX-CRC] SKIP: CRC is 0 or None, skipping CRC check")
    except Exception as e:
        print("[RX-CRC] crc check failed:", e)
        
    if mode_rx == b"F":
        out_fname = fname if fname else "rx_file"
        out_path = "rx_" + out_fname
        with open(out_path, "wb") as f:
            f.write(assembled)
        print(f"[RX FILE] Saved: {out_path} ({len(assembled)} bytes)")
    else:
        rec_text = assembled.decode("utf-8", errors="ignore")
        print("[RX TEXT]", rec_text)
        # Сохраняем текст в файл
        out_path = "rx_text.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(rec_text)
        print(f"[RX TEXT] Saved to: {out_path}")
    
    # Построение итогового графика эквалайзера
    if PLOTTING_AVAILABLE:
        try:
            from plot_utils import plot_rx_equalizer_final
            global Hk_smooth_list
            if 'Hk_smooth_list' in globals() and Hk_smooth_list:
                print(f"[RX] Построение итогового графика эквалайзера...")
                plot_rx_equalizer_final(Hk_smooth_list, subc_inds, fs, Nfft)
        except Exception as e:
            print(f"[RX] Ошибка при построении итогового графика эквалайзера: {e}")
    
    # Построение графика AGC по окончании приёма из файла
    if PLOTTING_AVAILABLE:
        try:
            from plot_utils import plot_agc_per_symbol
            global agc_history_list
            if 'agc_history_list' in globals() and agc_history_list:
                print(f"[RX] Построение графика AGC по окончании приёма из файла...")
                plot_agc_per_symbol(agc_history_list, title="AGC per Symbol (File Mode)")
                # Очищаем список после построения графика
                agc_history_list.clear()
                global _global_symbol_counter
                _global_symbol_counter = 0
        except Exception as e:
            print(f"[RX] Ошибка при построении графика AGC: {e}")
    
    # Отрисовка созвездия с градиентом (синий->красный) по порядку символов
    if PLOTTING_AVAILABLE:
        try:
            from plot_utils import plot_constellation
            global rx_constellation_symbols
            if rx_constellation_symbols:
                print(f"[DEBUG] Вызов plot_constellation с {len(rx_constellation_symbols)} символами")
                plot_constellation(rx_constellation_symbols, "RX Constellation (gradient)", use_gradient=True)
                rx_constellation_symbols.clear()  # Очищаем после отрисовки
        except Exception as e:
            print(f"[RX] Ошибка при построении созвездия: {e}")
    
    return True


# -----------------------
# Live receive and main flow
# -----------------------
def live_receive_and_process():
    """Живой прием с микрофона."""
    import threading
    import time
    from collections import deque
    from modem_config import get_audio, PLOTTING_AVAILABLE, plt
    
    # Инициализируем фазы и строим преамбулу
    from modem_config import init_phases
    init_phases()
    from modem_modulation import build_preamble
    preamble_td_local = build_preamble()
    if preamble_td_local is None:
        print("[LIVE-ERR] Failed to build preamble")
        return
    pre_len = len(preamble_td_local)
    # Обновляем глобальную переменную
    global preamble_td
    preamble_td = preamble_td_local
    
    CHUNK = 2048
    RECORD_CHANNELS = 1
    BUFFER_LOCK = threading.Lock()
    ring = deque()
    total_samples_in_buffer = 0
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal total_samples_in_buffer
        if status:
            pass
        s = indata[:, 0].astype(np.float64)
        with BUFFER_LOCK:
            ring.append(s)
            total_samples_in_buffer += s.size
    
    audio = get_audio()
    audio.start_stream(audio_callback, samplerate=fs, channels=RECORD_CHANNELS, blocksize=CHUNK)
    print("[LIVE] microphone stream started, buffering...")
    
    def read_buffer_snapshot():
        with BUFFER_LOCK:
            if not ring:
                return np.array([], dtype=float)
            arr = np.concatenate(list(ring))
            return arr.copy()
    
    def wait_for_samples(n_needed, timeout=None):
        start = time.time()
        while True:
            with BUFFER_LOCK:
                cur = total_samples_in_buffer
            if cur >= n_needed:
                return True
            if timeout is not None and (time.time() - start) > timeout:
                return False
            time.sleep(0.01)
    
    try:
        print("[LIVE] searching first preamble (continuous scan)")
        found_first = False
        first_sync_abs = None
        
        while not found_first:
            buf = read_buffer_snapshot()
            if buf.size < pre_len:
                time.sleep(0.02)
                continue
            
            # Предварительный AGC для слабых сигналов перед поиском преамбулы
            signal_rms = np.sqrt(np.mean(buf**2))
            if signal_rms > 0:
                buf = buf * (TARGET_RMS / signal_rms)
            
            corr_full = np.correlate(buf, preamble_td_local, mode='valid')
            energy = np.convolve(buf * buf, np.ones(pre_len)[::-1], mode='valid')
            denom = np.sqrt(energy * np.sum(preamble_td_local * preamble_td_local))
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_corr = np.abs(corr_full) / denom
                norm_corr[~np.isfinite(norm_corr)] = 0.0
            peak_idx = int(np.argmax(norm_corr))
            peak_val = float(norm_corr[peak_idx])
            
            # Вычисляем уровень шума как медиану корреляции
            noise_floor = np.median(norm_corr)  # Медиана корреляции (шум)
            # Повышаем порог: минимум в 10 раз выше шума или абсолютный минимум 0.1
            threshold = max(noise_floor * 10.0, 0.1)
            # Проверяем энергию сигнала в окне преамбулы
            signal_energy = np.mean(buf[peak_idx:peak_idx+pre_len]**2) if peak_idx + pre_len <= len(buf) else 0
            min_energy = 1e-6  # Минимальная энергия для валидного сигнала
            print(f"[SYNC] noise_floor={noise_floor:.6f}, threshold={threshold:.6f}, peak_val={peak_val:.6f}, energy={signal_energy:.6f}")
            
            if peak_val < threshold or signal_energy < min_energy:
                reason = "peak too low" if peak_val < threshold else "energy too low"
                print(f"[SYNC] skip: {reason} (peak={peak_val:.3f}, threshold={threshold:.3f}, energy={signal_energy:.6f})")
                time.sleep(0.02)
                continue
            
            cand_abs = peak_idx
            packet_blocks_guess = DEFAULT_PACKET_BLOCKS
            # Переводим логические блоки в физические символы для расчета общего количества сэмплов
            # Используем максимальное значение OFDM_SYMBOLS_PER_BLOCK (для BPSK это 2)
            # чтобы гарантировать, что мы прочитаем достаточно данных
            physical_symbols_guess = packet_blocks_guess * 2  # максимум для BPSK
            needed_total = cand_abs + pre_len + physical_symbols_guess * SYMBOL_LEN + SYMBOL_LEN
            
            if not wait_for_samples(needed_total, timeout=10.0):
                time.sleep(0.05)
                continue
            
            refine_center = max(0, cand_abs - SYMBOL_LEN)
            refine_window_len = pre_len + 2 * SYMBOL_LEN
            buf2 = read_buffer_snapshot()
            if refine_center + refine_window_len > buf2.size:
                time.sleep(0.02)
                continue
            local_segment = buf2[refine_center: refine_center + refine_window_len]
            from signal_utils import correlate
            corr_local = correlate(local_segment, preamble_td_local, mode='valid')
            energy_local = np.convolve(local_segment * local_segment, np.ones(pre_len)[::-1], mode='valid')
            denom_local = np.sqrt(energy_local * np.sum(preamble_td_local * preamble_td_local))
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_local = np.abs(corr_local) / denom_local
                norm_local[~np.isfinite(norm_local)] = 0.0
            loc_peak = int(np.argmax(norm_local))
            refined_abs = refine_center + loc_peak
            print(f"[LIVE] first preamble candidate found: peak={peak_val:.3f} cand={cand_abs} refined={refined_abs}")
            
            # Переводим логические блоки в физические символы
            # Используем максимальное значение OFDM_SYMBOLS_PER_BLOCK (для BPSK это 2)
            physical_symbols_refined = packet_blocks_guess * 2  # максимум для BPSK
            needed_total_refined = refined_abs + pre_len + physical_symbols_refined * SYMBOL_LEN + SYMBOL_LEN
            if not wait_for_samples(needed_total_refined, timeout=10.0):
                print("[LIVE] waiting for more samples for refined packet timed out, continue scanning")
                time.sleep(0.05)
                continue
            
            first_sync_abs = refined_abs
            found_first = True
        
        if first_sync_abs is None:
            raise RuntimeError("no preamble found in live stream")
        
        print(f"[LIVE] attempting decode of first packet at abs={first_sync_abs}")
        
        buf_all = read_buffer_snapshot()
        from signal_utils import correlate
        abs_corr = np.abs(correlate(buf_all, preamble_td_local, mode='valid'))
        
        globals_backup = {}
        for name in ('rx', 'abs_corr'):
            globals_backup[name] = globals().get(name, None)
            globals()[name] = buf_all
        
        try:
            pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = decode_packet_at_candidate(first_sync_abs, DEFAULT_PACKET_BLOCKS, packet_idx=0, bytes_before_packet=0, expected_total=0)
            if pkt0_decoded is None or len(pkt0_decoded) < 64:
                print("[LIVE] failed to decode first packet from live buffer")
                globals()['rx'] = globals_backup['rx']
                globals()['abs_corr'] = globals_backup['abs_corr']
                audio.stop()
                return
            
            # Детальная отладка первого пакета
            print(f"[LIVE-DEBUG] pkt0_decoded length: {len(pkt0_decoded)}")
            print(f"[LIVE-DEBUG] First 64 bytes hex: {pkt0_decoded[:64].hex()}")
            
            header64 = pkt0_decoded[:64]
            
            try:
                hdr = parse_header(header64)
                print(f"[LIVE] first packet decoded: RS_OK={pkt0_rs_ok} header={header64[:16].hex()}")
                
                # Проверяем валидность заголовка
                if not hdr.get('is_valid', False):
                    print(f"[LIVE] Invalid header (is_valid=False), discarding packet")
                    globals()['rx'] = globals_backup['rx']
                    globals()['abs_corr'] = globals_backup['abs_corr']
                    audio.stop()
                    return
                
                # Используем модуляцию из заголовка
                modulation_from_header = hdr.get('modulation', 'QPSK')
                print(f"[LIVE] Modulation from header: {modulation_from_header}")
                
                mode_rx = hdr['mode']
                total_sz = hdr['data_len']
                fname = hdr['filename']
                packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)
                
                # Устанавливаем глобальные настройки модуляции
                modem_config.MODULATION = modulation_from_header
                if modulation_from_header == "BPSK":
                    modem_config.BITS_PER_SYMBOL = 1
                    modem_config.OFDM_SYMBOLS_PER_BLOCK = 2
                else:
                    modem_config.BITS_PER_SYMBOL = 2
                    modem_config.OFDM_SYMBOLS_PER_BLOCK = 1
                modem_config.BITS_PER_OFDM_SYMBOL = modem_config.Nsub * modem_config.BITS_PER_SYMBOL
                
                positions_no_preroll = simulate_packet_positions(total_sz, fname.encode('utf-8') if fname else b'', (mode_rx == b'T'),
                                                                 packet_blocks_from_hdr, preamble_len=len(preamble_td_local), symbol_len=SYMBOL_LEN,
                                                                 gap_samples=0)
                if len(positions_no_preroll) == 0:
                    print("[LIVE] simulate_packet_positions returned no positions")
                    audio.stop()
                    return
                
                base_abs = first_sync_abs - positions_no_preroll[0]
                expected_abs = [base_abs + int(x) for x in positions_no_preroll]
                
                assembled_chunks = []
                bytes_collected = 0
                expected_total = total_sz
                packet_blocks_val = packet_blocks_from_hdr
                
                for pkt_idx, pref in enumerate(expected_abs):
                    pref_backoff = max(0, pref - SYMBOL_LEN)
                    # Переводим логические блоки в физические символы
                    # Используем текущее значение OFDM_SYMBOLS_PER_BLOCK (обновляется после определения модуляции)
                    physical_symbols_val = packet_blocks_val * modem_config.OFDM_SYMBOLS_PER_BLOCK
                    needed_for_pkt = pref_backoff + SYMBOL_LEN + pre_len + physical_symbols_val * SYMBOL_LEN + SYMBOL_LEN
                    if not wait_for_samples(needed_for_pkt, timeout=20.0):
                        print(f"[LIVE] timeout waiting for packet {pkt_idx} samples (needed {needed_for_pkt}), aborting")
                        break
                    
                    buf_now = read_buffer_snapshot()
                    abs_corr = np.abs(correlate(buf_now, preamble_td_local, mode='valid'))
                    
                    refine_center = pref_backoff
                    refine_window_len = pre_len + 2 * SYMBOL_LEN
                    if refine_center + refine_window_len > buf_now.size:
                        if not wait_for_samples(refine_center + refine_window_len, timeout=5.0):
                            print("[LIVE] not enough samples for refine, abort packet")
                            break
                        buf_now = read_buffer_snapshot()
                    abs_corr = np.abs(correlate(buf_now, preamble_td_local, mode='valid'))
                    
                    local_segment = buf_now[refine_center: refine_center + refine_window_len]
                    corr_local = correlate(local_segment, preamble_td_local, mode='valid')
                    energy_local = np.convolve(local_segment * local_segment, np.ones(pre_len)[::-1], mode='valid')
                    denom_local = np.sqrt(energy_local * np.sum(preamble_td_local * preamble_td_local))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        norm_local = np.abs(corr_local) / denom_local
                        norm_local[~np.isfinite(norm_local)] = 0.0
                    loc_peak = int(np.argmax(norm_local))
                    refined_abs = refine_center + loc_peak
                    
                    globals()['rx'] = buf_now
                    globals()['abs_corr'] = abs_corr
                    decoded_bytes, rs_ok_count, used_preamble = decode_packet_at_candidate(refined_abs, packet_blocks_val, packet_idx=pkt_idx, bytes_before_packet=bytes_collected, expected_total=expected_total)
                    print(f"[LIVE] pkt{pkt_idx}: refined={refined_abs} used_pre={used_preamble} RS_OK={rs_ok_count} bytes={len(decoded_bytes)}")
                    
                    if pkt_idx == 0:
                        if len(decoded_bytes) < 64:
                            print("[LIVE] first packet decoded <64 bytes, abort")
                            break
                        payload_start = 3 * 64
                        if payload_start >= len(decoded_bytes):
                            payload_chunk = b''
                        else:
                            payload_chunk = decoded_bytes[payload_start:]
                        need = max(0, expected_total - bytes_collected)
                        if need <= 0:
                            payload_chunk = b''
                        else:
                            payload_chunk = payload_chunk[:need]
                        assembled_chunks.append(payload_chunk)
                        bytes_collected += len(payload_chunk)
                    else:
                        if len(decoded_bytes) >= 4:
                            payload_chunk = decoded_bytes[4:]
                            assembled_chunks.append(payload_chunk)
                            bytes_collected += len(payload_chunk)
                    
                    print(f"[LIVE] collected {bytes_collected}/{expected_total} bytes")
                    if bytes_collected >= expected_total:
                        print("[LIVE] collected all expected bytes")
                        break
                
                assembled = b"".join(assembled_chunks)[:expected_total]
                
                try:
                    hdr_crc = hdr.get('crc32', None)
                    if hdr_crc is not None and hdr_crc != 0:  # Запрещаем нулевой CRC
                        calc_crc = zlib.crc32(assembled) & 0xFFFFFFFF
                        if calc_crc == int(hdr_crc):
                            print(f"[RX-CRC] OK: CRC32 matched (0x{calc_crc:08X})")
                        else:
                            print(f"[RX-CRC] MISMATCH: received 0x{int(hdr_crc):08X}, calculated 0x{calc_crc:08X}")
                    else:
                        print(f"[RX-CRC] SKIP: CRC is 0 or None, skipping CRC check")
                except Exception as e:
                    print("[RX-CRC] crc check failed:", e)
                
                if mode_rx == b"F":
                    out_fname = fname if fname else "rx_file"
                    out_path = "rx_" + out_fname
                    with open(out_path, "wb") as f:
                        f.write(assembled)
                    print(f"[RX FILE] Saved: {out_path} ({len(assembled)} bytes)")
                else:
                    rec_text = assembled.decode("utf-8", errors="ignore")
                    print("[RX TEXT]", rec_text)
                    
            except Exception as e:
                print(f"[LIVE] Error processing header: {e}")
                import traceback
                traceback.print_exc()
        
        finally:
            globals()['rx'] = globals_backup['rx']
            globals()['abs_corr'] = globals_backup['abs_corr']
    
    except KeyboardInterrupt:
        print("[LIVE] Interrupted by user")
    except Exception as e:
        print("[LIVE] exception during live receive:", e)
        import traceback
        traceback.print_exc()
    finally:
        try:
            if audio.is_active():
                audio.stop()
        except Exception:
            pass
        print("[LIVE] microphone stream stopped")
        
        # Построение итогового графика эквалайзера
        if PLOTTING_AVAILABLE:
            try:
                from plot_utils import plot_rx_equalizer_final
                global Hk_smooth_list
                if 'Hk_smooth_list' in globals() and Hk_smooth_list:
                    print(f"[RX] Построение итогового графика эквалайзера...")
                    plot_rx_equalizer_final(Hk_smooth_list, subc_inds, fs, Nfft)
            except Exception as e:
                print(f"[RX] Ошибка при построении итогового графика эквалайзера: {e}")
        
        # Построение графика AGC по окончании живого приёма
        if PLOTTING_AVAILABLE:
            try:
                from plot_utils import plot_agc_per_symbol
                global agc_history_list
                if 'agc_history_list' in globals() and agc_history_list:
                    print(f"[RX] Построение графика AGC по окончании живого приёма...")
                    plot_agc_per_symbol(agc_history_list, title="AGC per Symbol (Live Mode)")
                    # Очищаем список после построения графика
                    agc_history_list.clear()
                    global _global_symbol_counter
                    _global_symbol_counter = 0
            except Exception as e:
                print(f"[RX] Ошибка при построении графика AGC: {e}")
        # Отрисовка созвездия с градиентом (синий->красный) по порядку символов
        if PLOTTING_AVAILABLE:
            try:
                from plot_utils import plot_constellation
                global rx_constellation_symbols
                if rx_constellation_symbols:
                    print(f"[DEBUG] Вызов plot_constellation с {len(rx_constellation_symbols)} символами")
                    plot_constellation(rx_constellation_symbols, "RX Constellation (gradient)", use_gradient=True)
                    rx_constellation_symbols.clear()  # Очищаем после отрисовки
            except Exception as e:
                print(f"[RX] Ошибка при построении созвездия: {e}")


# -----------------------
# Audio backend wrapper (for backward compatibility with tests)
# -----------------------
def get_audio():
    """
    Возвращает аудио бэкенд для кроссплатформенной поддержки.
    Для обратной совместимости с тестами.
    """
    from audio_backend import get_audio_backend
    return get_audio_backend()
