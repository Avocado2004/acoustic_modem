"""
Модуль приема из файла OFDM Acoustic Modem.
Содержит функцию приема из WAV-файла.
"""

import numpy as np
import zlib
import modem_config
from modem_config import (Nfft, Ncp, Nsub, subc_inds, fs, SYMBOL_LEN, DEFAULT_PACKET_BLOCKS,
                           RS_CW_BITS, RS_DATA_BYTES,
                           RS_CW_BYTES, rs, SYMBOL_TARGET_RMS, AGC_ALPHA, AGC_DEBUG, MIN_RMS,
                           PLOTTING_AVAILABLE, _MAX_RS_FAIL_PRINTS_GLOBAL, SYNC_WINDOW_HALF)

from modem_modulation import build_preamble
from modem_packet import parse_header, simulate_packet_positions
from rx_decoder import decode_packet_at_candidate

# Импортируем модуль состояния
import rx_state as _rx_st

def receive_from_file(wav_path):
    """Прием из WAV-файла."""
    from wav_utils import wavfile
    from signal_utils import fftconvolve, find_peaks
    
    # Инициализируем фазы перед приемом
    from modem_config import init_phases
    init_phases()
    
    _, wavd = wavfile.read(wav_path)
    sig = wavd[:,0] if wavd.ndim > 1 else wavd
    # Используем переменные из rx_state
    _rx_st.rx = sig.astype(float) / np.iinfo(wavd.dtype).max
    
    # Получаем актуальную преамбулу
    from modem_modulation import build_preamble
    preamble_td_local = build_preamble()
    _rx_st.preamble_td = preamble_td_local  # Обновляем глобальную переменную
    
    # Исправлено: используем _rx_st.rx вместо rx
    corr = fftconvolve(_rx_st.rx, preamble_td_local[::-1], mode='valid')
    _rx_st.abs_corr = np.abs(corr)
    
    # Исправлено: используем _rx_st.abs_corr вместо abs_corr
    if _rx_st.abs_corr.size == 0:
        print("[RX-ERR] correlation empty, cannot find preambles")
        return False
        
    peak_val = np.max(_rx_st.abs_corr)
    threshold = 0.5 * peak_val if peak_val != 0 else 0.0
    peaks, props = find_peaks(_rx_st.abs_corr, height=threshold, distance=len(preamble_td_local)//2)
    peak_heights = props['peak_heights'] if 'peak_heights' in props else _rx_st.abs_corr[peaks]
    candidates = sorted(zip(peaks, peak_heights), key=lambda x: -x[1])
    print(f"[RX-DBG] found {len(candidates)} preamble peak candidates (threshold={threshold:.6g})")
    
    # Используем фиксированную позицию преамбулы
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
        
    # Используем переменные из rx_state
    _rx_st.post_sync = True
    _rx_st.sync_sample_abs = int(pkt0_used_pre) if pkt0_used_pre is not None else int(sync_idx)
    
    # Исправлено: mode_rx должно быть hdr['mode']
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
        # Исправлено: добавлена запятая
        with open(out_path, "wb") as f:
            f.write(assembled)
        print(f"[RX FILE] Saved: {out_path} ({len(assembled)} bytes)")
    else:
        rec_text = assembled.decode("utf-8", errors="ignore")
        print("[RX TEXT]", rec_text)
        # Сохраняем текст в файл
        out_path = "rx_text.txt"
        # Исправлено: добавлена запятая
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(rec_text)
        print(f"[RX TEXT] Saved to: {out_path}")
    
    # === Отключено: Построение итогового графика эквалайзера ===
    # if PLOTTING_AVAILABLE:
    #     try:
    #         from plot_utils import plot_rx_equalizer_final
    #         if _rx_st.Hk_smooth_list and len(_rx_st.Hk_smooth_list) > 0:
    #             print(f"[RX] Построение итогового графика эквалайзера...")
    #             plot_rx_equalizer_final(_rx_st.Hk_smooth_list, subc_inds, fs, Nfft)
    #     except Exception as e:
    #         print(f"[RX] Ошибка при построении итогового графика эквалайзера: {e}")
    
    # === Отключено: Построение графика AGC ===
    # if PLOTTING_AVAILABLE:
    #     try:
    #         from plot_utils import plot_agc_per_symbol
    #         if _rx_st.agc_history_list and len(_rx_st.agc_history_list) > 0:
    #             print(f"[RX] Построение графика AGC по окончании приёма из файла...")
    #             plot_agc_per_symbol(_rx_st.agc_history_list, title="AGC per Symbol (File Mode)")
    #             _rx_st.agc_history_list.clear()
    #             _rx_st._global_symbol_counter = 0
    #     except Exception as e:
    #         print(f"[RX] Ошибка при построении графика AGC: {e}")
    
    # === Оставлено: Отрисовка созвездия с градиентом (синий->красный) по порядку символов ===
    if PLOTTING_AVAILABLE:
        try:
            from plot_utils import plot_constellation
            if _rx_st.rx_constellation_symbols and len(_rx_st.rx_constellation_symbols) > 0:
                print(f"[DEBUG] Вызов plot_constellation с {len(_rx_st.rx_constellation_symbols)} символами")
                plot_constellation(_rx_st.rx_constellation_symbols, "RX Constellation (gradient)", use_gradient=True)
                _rx_st.rx_constellation_symbols.clear()  # Очищаем после отрисовки
        except Exception as e:
            print(f"[RX] Ошибка при построении созвездия: {e}")
    
    return True
