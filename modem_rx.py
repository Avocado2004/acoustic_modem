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
                          PLOTTING_AVAILABLE, _MAX_RS_FAIL_PRINTS_GLOBAL, SYNC_WINDOW_HALF)
from modem_modulation import (qpsk_demap, bpsk_demap, ofdm_symbol, build_preamble, bytes_to_bits, bits_to_bytes,
                              sync_by_corr, deinterleave_bits, BITS_PER_OFDM_SYMBOL)
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
last_packet_used_pre = None


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
    
    # Получаем актуальную преамбулу (теперь с правильными фазами)
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
    
    # Используем фиксированную позицию преамбулы (preroll = 12000)
    # Так как мы точно знаем, где находится преамбула
    expected_sync = 12000
    print(f"[RX-DBG] Using FIXED preroll position: {expected_sync}")
    sync_idx = expected_sync

    pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = decode_packet_at_candidate(sync_idx, DEFAULT_PACKET_BLOCKS, packet_idx=0, bytes_before_packet=0, expected_total=0)
    if pkt0_decoded is None or len(pkt0_decoded) < 64:
        print("[RX-ERR] unable to decode first packet header")
        return False

    header64 = pkt0_decoded[:64]
    try:
        hdr = parse_header(header64)
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
        if hdr_crc is not None:
            calc_crc = zlib.crc32(assembled) & 0xFFFFFFFF
            if calc_crc == int(hdr_crc):
                print(f"[RX-CRC] OK: CRC32 matched (0x{calc_crc:08X})")
            else:
                print(f"[RX-CRC] MISMATCH: received 0x{int(hdr_crc):08X}, calculated 0x{calc_crc:08X}")
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
        # Сохраняем текст в файл для возможности проверки
        out_path = "rx_text.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(rec_text)
        print(f"[RX TEXT] Saved to: {out_path}")
        
    return True


# -----------------------
# decode_packet_at_candidate
# -----------------------
def decode_packet_at_candidate(pref_abs, packet_blocks_expected, packet_idx=0, bytes_before_packet=0, expected_total=0):
    """Декодирование пакета по кандидату синхронизации."""
    best_result = (b'', 0, None)
    global abs_corr, rx, preamble_td, last_agc_rms
    if 'abs_corr' not in globals() or 'rx' not in globals():
        return best_result

    if '_rs_fail_prints_count' not in globals():
        globals()['_rs_fail_prints_count'] = 0

    # ИСПРАВЛЕНИЕ: используем переданный индекс напрямую, без поиска в окне
    # Так как мы уже знаем точную позицию преамбулы (12000)
    combined = [pref_abs]

    try:
        zc_seq_ideal = (np.exp(-1j * np.pi * 1 * np.arange(Nsub) * (np.arange(Nsub) + 1) / float(Nsub)))
        zc_seq_ideal = zc_seq_ideal / np.sqrt(np.mean(np.abs(zc_seq_ideal)**2))
        S_zc_fd = np.zeros(Nfft, dtype=complex)
        S_zc_fd[subc_inds] = zc_seq_ideal
        S_zc_fd[-subc_inds] = np.conj(zc_seq_ideal)
    except Exception:
        S_zc_fd = None

    for cand in combined:
        try:
            print(f"[RX-DBG-DECODE] cand={cand}, Nfft={Nfft}, Ncp={Ncp}, SYMBOL_LEN={SYMBOL_LEN}")
            rx_zc1 = rx[cand + Ncp : cand + Ncp + Nfft]
            rx_zc2 = rx[cand + SYMBOL_LEN + Ncp : cand + SYMBOL_LEN + Ncp + Nfft]
            cross = np.vdot(rx_zc1, rx_zc2)
            delta_phi = np.angle(cross)
            T_between = SYMBOL_LEN / float(fs)
            f_err_loc = delta_phi / (2.0 * np.pi * T_between)
            print(f"[RX-DBG-DECODE] f_err_loc={f_err_loc:.6f} Hz")
            
            pilot1_start = cand + 2*SYMBOL_LEN
            rx_pre1 = rx[pilot1_start + Ncp : pilot1_start + Ncp + Nfft]
            pilot2_start = pilot1_start + SYMBOL_LEN
            rx_pre2 = rx[pilot2_start + Ncp : pilot2_start + Ncp + Nfft]
            t1_offset = (pilot1_start) / float(fs)
            t2_offset = (pilot2_start) / float(fs)
            time_vec = np.arange(Nfft) / float(fs)
            rx_pre1_corr = rx_pre1 * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t1_offset + time_vec))
            rx_pre2_corr = rx_pre2 * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t2_offset + time_vec))
            R1t = np.fft.fft(rx_pre1_corr) / Nfft  # ДЕЛИМ НА Nfft!
            R2t = np.fft.fft(rx_pre2_corr) / Nfft  # ДЕЛИМ НА Nfft!
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
            
            # ИСПРАВЛЕНИЕ: правильная оценка канала
            Hk_est = (R1t[subc_inds] / S_ref[subc_inds] + R2t[subc_inds] / S_ref[subc_inds]) / 2
            
            # Используем Hk_est напрямую, только ограничиваем экстремальные значения
            Hk_mag = np.clip(np.median(np.abs(Hk_est)) if hasattr(np, 'median') else np.mean(np.abs(Hk_est)), 1/2.0, None)
            Hk_s = Hk_mag * np.exp(1j*np.angle(Hk_est))
            print(f"[RX-DBG-DECODE] Hk_est magnitude range: {np.min(np.abs(Hk_est)):.6f} - {np.max(np.abs(Hk_est)):.6f}")
            print(f"[RX-DBG-DECODE] Hk_s magnitude: {Hk_mag:.6f}")
            
            # Сохраняем график эквалайзера для первого пакета
            if packet_idx == 0 and PLOTTING_AVAILABLE:
                try:
                    import matplotlib.pyplot as plt
                    # График амплитуды и фазы эквалайзера
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Амплитудная характеристика
                    freqs = subc_inds * fs / float(Nfft)
                    ax1.plot(freqs, 20*np.log10(np.abs(Hk_s) + 1e-12), 'b-o', markersize=3)
                    ax1.set_xlabel('Frequency (Hz)')
                    ax1.set_ylabel('Magnitude (dB)')
                    ax1.set_title('Equalizer Magnitude Response (Hk_s)')
                    ax1.grid(True)
                    
                    # Фазовая характеристика
                    ax2.plot(freqs, np.angle(Hk_s) * 180/np.pi, 'r-o', markersize=3)
                    ax2.set_xlabel('Frequency (Hz)')
                    ax2.set_ylabel('Phase (degrees)')
                    ax2.set_title('Equalizer Phase Response (Hk_s)')
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig('rx_equalizer.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"[RX] График эквалайзера сохранен: rx_equalizer.png")
                except Exception as e:
                    print(f"[RX] Ошибка при сохранении графика эквалайзера: {e}")
        except Exception as e:
            print(f"[RX-DBG-DECODE] Exception in channel estimation: {e}")
            continue

        try:
            pre_segment = rx[cand : cand + len(preamble_td)]
            pre_rms = np.sqrt(np.mean(pre_segment**2)) if pre_segment.size > 0 else MIN_RMS
            if pre_rms < MIN_RMS:
                pre_rms = MIN_RMS
            packet_gain = 0.5 / pre_rms
        except Exception:
            packet_gain = 1.0

        pkt_data_start = cand + len(preamble_td)
        pkt_payload_samples = packet_blocks_expected * SYMBOL_LEN
        seg = packet_gain * rx[pkt_data_start : pkt_data_start + pkt_payload_samples]
        if len(seg) < pkt_payload_samples:
            continue
        frames = seg.reshape(packet_blocks_expected, SYMBOL_LEN)
        
        # ИСПРАВЛЕНИЕ: инициализируем список для символов ПЕРЕД циклом
        rx_syms_pkt_list = []  # Список для накопления символов
        
        for idxf, fr in enumerate(frames):
            frame_start_abs = pkt_data_start + idxf * SYMBOL_LEN
            t_frame_start = frame_start_abs / float(fs)
            useful = fr[Ncp:]
            tv = t_frame_start + np.arange(Nfft) / float(fs)
            useful_corr = useful * np.exp(-1j * 2.0 * np.pi * f_err_loc * tv)

            cur_rms = np.sqrt(np.mean(np.abs(useful)**2)) if useful.size > 0 else 1e-12
            # Исправление: используем глобальную переменную last_agc_rms напрямую
            global last_agc_rms
            # Если last_agc_rms все еще None (по какой-то причине), инициализируем
            if last_agc_rms is None or not isinstance(last_agc_rms, (int, float)):
                last_agc_rms = cur_rms
            est_rms = (1.0 - AGC_ALPHA) * last_agc_rms + AGC_ALPHA * cur_rms
            globals()['last_agc_rms'] = est_rms
            if est_rms < 1e-12:
                est_rms = 1e-12
            gain_sym = SYMBOL_TARGET_RMS / est_rms
            # ВСЕГДА выводим AGC для первого пакета (для отладки)
            if packet_idx == 0:
                print(f"[AGC-DEBUG] pkt={packet_idx} frame={idxf} cur_rms={cur_rms:.6e} est_rms={est_rms:.6e} gain_sym={gain_sym:.3f}")
                print(f"[AGC-DEBUG] useful stats: min={np.min(useful):.6f}, max={np.max(useful):.6f}, rms={np.sqrt(np.mean(useful**2)):.6f}")

            useful = useful * gain_sym

            try:
                F = np.fft.fft(useful_corr) / Nfft  # ДЕЛИМ НА Nfft ДЛЯ СОГЛАСОВАНИЯ С Hk_s
                subc = F[subc_inds] / Hk_s
            except Exception:
                subc = np.zeros(Nsub, dtype=complex)
            
            # Применяем фазы поднесущих к каждому символу (если они есть)
            if modem_config.subc_phases is not None and np.any(modem_config.subc_phases != 0):
                subc = subc * np.exp(-1j * modem_config.subc_phases)
            
            rx_syms_pkt_list.append(subc)

        # Проверяем, что список не пустой
        if len(rx_syms_pkt_list) == 0:
            continue
        
        try:
            rx_syms_pkt = np.concatenate(rx_syms_pkt_list)
        except Exception:
            continue
        
        if phi_est != 0.0:
            rx_syms_pkt = rx_syms_pkt * np.exp(-1j * phi_est)
        
        # Сохраняем диаграмму созвездия для первого пакета
        if packet_idx == 0 and PLOTTING_AVAILABLE:
            try:
                from plot_utils import plot_constellation
                plot_constellation(rx_syms_pkt[:min(4096, len(rx_syms_pkt))], 
                                 title=f"RX Constellation pkt{packet_idx}")
                print(f"[RX] Диаграмма созвездия сохранена для пакета {packet_idx}")
            except Exception as e:
                print(f"[RX] Ошибка при сохранении диаграммы созвездия: {e}")

        # ВСЕГДА используем QPSK (BPSK отключен до отладки)
        bits_pkt = qpsk_demap(rx_syms_pkt)
        
        # Применяем деинтерливинг для восстановления исходного порядка битов
        bits_pkt = deinterleave_bits(bits_pkt, block_size=BITS_PER_OFDM_SYMBOL)
        
        # ОТЛАДКА: выводим первые символы
        if packet_idx == 0:
            print(f"[RX-DEBUG] First 10 rx_syms_pkt: {rx_syms_pkt[:10]}")
            print(f"[RX-DEBUG] First 10 bits_pkt: {bits_pkt[:20]}")
            # Проверяем, не нулевые ли символы
            if np.all(np.abs(rx_syms_pkt[:10]) < 0.1):
                print("[RX-DEBUG] WARNING: First symbols have very low magnitude!")
            # Проверяем фазы
            if modem_config.subc_phases is not None:
                print(f"[RX-DEBUG] subc_phases[:5]: {modem_config.subc_phases[:5]}")
        
        if packet_idx == 0:
            # Просто выводим информацию о модуляции
            print(f"[RX] Using QPSK modulation (BPSK auto-detection disabled)")
            # Используем modem_config для изменения глобальных настроек
            modem_config.MODULATION = "QPSK"
            modem_config.BITS_PER_SYMBOL = 2
            modem_config.BITS_PER_OFDM_SYMBOL = modem_config.Nsub * modem_config.BITS_PER_SYMBOL

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
                decoded_ok = True
            except Exception as e:
                msg = b'\x00' * RS_DATA_BYTES
                decoded_ok = False
                header_len = 64 if packet_idx == 0 else 4
                local_payload_offset = ci * RS_DATA_BYTES - header_len
                if local_payload_offset < 0:
                    global_offset = bytes_before_packet + 0
                else:
                    global_offset = bytes_before_packet + local_payload_offset
                do_print = False
                if packet_idx == 0:
                    if globals()['_rs_fail_prints_count'] < _MAX_RS_FAIL_PRINTS_GLOBAL:
                        do_print = True
                else:
                    if globals()['_rs_fail_prints_count'] < int(_MAX_RS_FAIL_PRINTS_GLOBAL * 0.25):
                        do_print = True
                if do_print:
                    globals()['_rs_fail_prints_count'] += 1
                    try:
                        symbols_per_cw = (cw_bits) // 2
                        s_start = ci * symbols_per_cw
                        subc_idx = int(s_start % Nsub)
                        freq_hz = subc_inds[subc_idx] * fs / float(Nfft)
                        freq_msg = f"{freq_hz:.1f} Hz (subcarrier idx {subc_inds[subc_idx]})"
                    except Exception:
                        freq_msg = "unknown"
                    if packet_idx == 0:
                        print(f"[RX-DBG-PKT] pkt{packet_idx} cw_idx={ci} RS_FAIL global_offset={global_offset} cw_in={bts.hex()} exc={e} freq={freq_msg}")
                    else:
                        if globals().get('post_sync', False):
                            print(f"[RX-DBG-PKT] pkt{packet_idx} cw_idx={ci} RS_FAIL global_offset={global_offset} freq={freq_msg}")
                        else:
                            print(f"[RX-DBG-PKT] pkt{packet_idx} cw_idx={ci} RS_FAIL global_offset={global_offset}")
            decoded_blocks.append((msg, decoded_ok) if packet_idx == 0 else msg)

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

            candidate_return_bytes = bytes(header_by_cw) + (b"".join([b for (b,ok) in decoded_blocks])[expected_hdr_len:])
            assembled_try = candidate_return_bytes[:expected_hdr_len]

            print(f"[RX-DBG-PKT] pkt0 header merged, first 16 bytes: {candidate_return_bytes[:16].hex()}")
            globals()['pkt0_header_bytes'] = bytes(header_by_cw[:64])
            ret_bytes = candidate_return_bytes
        else:
            ret_bytes = b"".join(decoded_blocks)

        if rs_ok > best_result[1]:
            best_result = (ret_bytes, rs_ok, cand)
            try:
                if 'rx_syms_list' not in globals():
                    globals()['rx_syms_list'] = []
                if 'Hk_smooth_list' not in globals():
                    globals()['Hk_smooth_list'] = []
                rx_slice = rx_syms_pkt[:4096].copy() if isinstance(rx_syms_pkt, np.ndarray) else np.array([], dtype=complex)
                globals()['rx_syms_list'].append(rx_slice)
                globals()['Hk_smooth_list'].append(Hk_s.copy() if isinstance(Hk_s, np.ndarray) else np.array([], dtype=complex))
                globals()['last_packet_used_pre'] = cand
                globals()['last_packet_end_sample'] = int(cand + len(preamble_td) + packet_blocks_expected * SYMBOL_LEN)
                globals()['last_packet_rs_ok'] = rs_ok
            except Exception:
                pass
            if rs_ok >= n_cw * 0.9:
                break
    return best_result


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

            corr_full = np.correlate(buf, preamble_td_local, mode='valid')
            energy = np.convolve(buf * buf, np.ones(pre_len)[::-1], mode='valid')
            denom = np.sqrt(energy * np.sum(preamble_td_local * preamble_td_local))
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_corr = np.abs(corr_full) / denom
                norm_corr[~np.isfinite(norm_corr)] = 0.0
            peak_idx = int(np.argmax(norm_corr))
            peak_val = float(norm_corr[peak_idx])

            if peak_val < 0.35:
                time.sleep(0.02)
                continue

            cand_abs = peak_idx
            packet_blocks_guess = DEFAULT_PACKET_BLOCKS
            needed_total = cand_abs + pre_len + packet_blocks_guess * SYMBOL_LEN + SYMBOL_LEN

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

            needed_total_refined = refined_abs + pre_len + packet_blocks_guess * SYMBOL_LEN + SYMBOL_LEN
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
            print(f"[LIVE-DEBUG] First 64 bytes as array: {list(pkt0_decoded[:64])}")
            
            header64 = pkt0_decoded[:64]
            print(f"[LIVE-DEBUG] header64 length: {len(header64)}")
            print(f"[LIVE-DEBUG] header64 hex: {header64[:16].hex()}")
            
            # Проверяем, не нулевой ли заголовок
            if all(b == 0 for b in header64[:16]):
                print("[LIVE-ERR] Header is all zeros! Possible modulation mismatch or sync error.")
                print(f"[LIVE-ERR] Try checking if transmitter uses same modulation (QPSK) as receiver.")
                print(f"[LIVE-ERR] first_sync_abs = {first_sync_abs}")
                print(f"[LIVE-ERR] pkt0_rs_ok = {pkt0_rs_ok} (should be >0 for valid packet)")
                # Попробуем найти следующий пик и попробовать снова
                try:
                    audio.stop()
                except Exception:
                    pass
                return
            
            try:
                hdr = parse_header(header64)
                print(f"[LIVE] first packet decoded: RS_OK={pkt0_rs_ok} header={header64[:16].hex()}")
                print(f"[LIVE] parsed header type: {type(hdr)}, content: {hdr}")
                
                # Проверяем, что заголовок не нулевой
                if all(b == 0 for b in header64[:16]):
                    print("[LIVE] ERROR: header is all zeros, invalid transmission")
                    try:
                        audio.stop()
                    except Exception:
                        pass
                    return
                
                # Проверяем, что заголовок корректный
                if not isinstance(hdr, dict):
                    print(f"[LIVE] ERROR: hdr is not a dict. hdr={hdr}")
                    try:
                        audio.stop()
                    except Exception:
                        pass
                    return
                    
                required_keys = ['mode', 'data_len', 'filename', 'packet_blocks']
                missing_keys = [key for key in required_keys if key not in hdr]
                if missing_keys:
                    print(f"[LIVE] invalid header format, missing keys: {missing_keys}. hdr={hdr}")
                    try:
                        audio.stop()
                    except Exception:
                        pass
                    print("[LIVE] aborting receive due to invalid header")
                    return
            except KeyError as ke:
                print(f"[LIVE] KeyError during header processing: {ke}")
                print(f"[LIVE] header64[:32] = {header64[:32].hex()}")
                try:
                    audio.stop()
                except Exception:
                    pass
                return
            except Exception as e:
                print("[LIVE] parse_header failed:", e)
                try:
                    audio.stop()
                except Exception:
                    pass
                print("[LIVE] aborting receive due to invalid transmission header")
                return
        
            globals()['post_sync'] = True
            globals()['sync_sample_abs'] = int(pkt0_used_pre) if pkt0_used_pre is not None else int(first_sync_abs)

            mode_rx = hdr['mode']
            total_sz = hdr['data_len']
            fname = hdr['filename']
            packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)
            if not isinstance(packet_blocks_from_hdr, int) or packet_blocks_from_hdr <= 0:
                try:
                    audio.stop()
                except Exception:
                    pass
                print(f"[LIVE] invalid packet_blocks in header ({packet_blocks_from_hdr}), aborting receive")
                return
            print(f"[LIVE] transmission header: data_len={total_sz} filename={fname} packet_blocks={packet_blocks_from_hdr}")

            positions_no_preroll = simulate_packet_positions(total_sz, fname.encode('utf-8') if fname else b'', (mode_rx == b'T'),
                                                             packet_blocks_from_hdr, preamble_len=len(preamble_td_local), symbol_len=SYMBOL_LEN,
                                                             gap_samples=0)
            if len(positions_no_preroll) == 0:
                print("[LIVE] simulate_packet_positions returned no positions")
                audio.stop()
                return

            base_abs = first_sync_abs - positions_no_preroll[0]
            expected_abs = [base_abs + int(x) for x in positions_no_preroll]
            print("[LIVE] expected preamble absolute indices (in buffer snapshot coords):", expected_abs)

            assembled_chunks = []
            bytes_collected = 0
            expected_total = total_sz
            packet_blocks_val = packet_blocks_from_hdr

            for pkt_idx, pref in enumerate(expected_abs):
                pref_backoff = max(0, pref - SYMBOL_LEN)
                needed_for_pkt = pref_backoff + SYMBOL_LEN + pre_len + packet_blocks_val * SYMBOL_LEN + SYMBOL_LEN
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
                print(f"[LIVE] pkt{pkt_idx} predicted={pref} backoff={pref_backoff} refined={refined_abs} peak={float(norm_local[loc_peak]):.3f}")

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
                hdr_crc = hdr.get('crc32', None) if isinstance(hdr, dict) else None
                if hdr_crc is not None:
                    calc_crc = zlib.crc32(assembled) & 0xFFFFFFFF
                    if calc_crc == int(hdr_crc):
                        print(f"[RX-CRC] OK: CRC32 matched (0x{calc_crc:08X})")
                    else:
                        print(f"[RX-CRC] MISMATCH: received 0x{int(hdr_crc):08X}, calculated 0x{calc_crc:08X}")
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