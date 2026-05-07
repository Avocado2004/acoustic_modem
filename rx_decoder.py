"""
Модуль декодирования OFDM Acoustic Modem.
Содержит функции декодирования пакетов с различными модуляциями.
"""

import numpy as np
import zlib
import modem_config
from modem_config import (Nfft, Ncp, Nsub, subc_inds, fs, SYMBOL_LEN, DEFAULT_PACKET_BLOCKS,
                           RS_CW_BITS, RS_DATA_BYTES,
                           RS_CW_BYTES, rs, SYMBOL_TARGET_RMS, AGC_ALPHA, AGC_DEBUG, MIN_RMS,
                           PLOTTING_AVAILABLE, _MAX_RS_FAIL_PRINTS_GLOBAL, SYNC_WINDOW_HALF,
                           PREAMBLE_PILOT_SYMBOLS)

from modem_modulation import (qpsk_demap, bpsk_demap, ofdm_symbol, build_preamble, bytes_to_bits, bits_to_bytes,
                           sync_by_corr, deinterleave_bits, AdaptiveEqualizer)

from modem_packet import parse_header, build_header, make_packet_header_bytes, simulate_packet_positions

# Импортируем модуль состояния
import rx_state as _rx_st

# Импортируем модуль водопадной диаграммы (headless, без отображения окон)
try:
    from equalizer_waterfall import EqualizerWaterfall, PLOTTING_AVAILABLE as WF_AVAILABLE
except ImportError:
    EqualizerWaterfall = None
    WF_AVAILABLE = False
    print("[RX] equalizer_waterfall не импортирован, водопад отключён")

# Проверка что rx_state импортирован корректно
assert hasattr(_rx_st, 'reset_state'), "rx_state должен содержать reset_state()"
assert hasattr(_rx_st, '_rs_fail_prints_count'), "rx_state должен содержать _rs_fail_prints_count"
assert hasattr(_rx_st, 'rx'), "rx_state должен содержать rx"
assert hasattr(_rx_st, 'abs_corr'), "rx_state должен содержать abs_corr"
assert hasattr(_rx_st, 'preamble_td'), "rx_state должен содержать preamble_td"
assert hasattr(_rx_st, 'last_agc_rms'), "rx_state должен содержать last_agc_rms"
assert hasattr(_rx_st, 'global_equalizer'), "rx_state должен содержать global_equalizer"
assert hasattr(_rx_st, 'agc_history_list'), "rx_state должен содержать agc_history_list"
assert hasattr(_rx_st, '_global_symbol_counter'), "rx_state должен содержать _global_symbol_counter"
assert hasattr(_rx_st, 'rx_constellation_symbols'), "rx_state должен содержать rx_constellation_symbols"


# -----------------------
# Функция извлечения пилотных символов для настройки эквалайзера и AGC
# -----------------------
def _extract_pilot_symbols(pref_abs, f_err_loc):
    """
    Извлечение и усреднение пилотных символов перед преамбулой.
    
    Пилотные символы идут ДО преамбулы (ZC+ZC+P+P) и содержат известные данные:
    все поднесущие = (1+1j)/√2.
    
    Это позволяет:
    1. Получить точную оценку канала Hk по 16 пилотам (вместо 2 в преамбуле)
    2. Инициализировать AGC из реального RMS пилотов
    
    Параметры
    ----------
    pref_abs : int
        Абсолютная позиция начала преамбулы (ZC1)
    f_err_loc : float
        Локальная частотная ошибка для компенсации
    
    Возвращает
    -------
    tuple (Hk_pilot, pilot_rms)
        Hk_pilot : np.ndarray - усреднённая оценка канала по пилотам
        pilot_rms : float - RMS пилотных символов для инициализации AGC
    """
    n_pilots = PREAMBLE_PILOT_SYMBOLS
    
    if n_pilots <= 0:
        return None, None
    
    # Позиция пилотов: перед преамбулой
    pilot_start = pref_abs - n_pilots * SYMBOL_LEN
    
    if pilot_start < 0:
        print(f"[RX-PILOT] pilot_start={pilot_start} < 0, skipping pilot extraction")
        return None, None
    
    # Эталонный пилотный символ: все поднесущие = (1+1j)/√2
    S_pilot_ref = (1 + 1j) / np.sqrt(2) * np.ones(Nsub, dtype=complex)
    
    Hk_sum = np.zeros(Nsub, dtype=complex)
    pilot_rms_sum = 0.0
    valid_pilots = 0
    
    for i in range(n_pilots):
        # Начало i-го пилотного символа
        sym_start = pilot_start + i * SYMBOL_LEN
        
        # Проверяем, что символ полностью в буфере
        if sym_start + SYMBOL_LEN > _rx_st.rx.size:
            print(f"[RX-PILOT] pilot {i} out of buffer, skipping")
            continue
        
        # Вырезаем useful часть (без CP)
        useful = _rx_st.rx[sym_start + Ncp : sym_start + Ncp + Nfft]
        
        if useful.size < Nfft:
            print(f"[RX-PILOT] pilot {i} too short, skipping")
            continue
        
        # Компенсируем частотный сдвиг
        t_sym_start = sym_start / float(fs)
        time_vec = np.arange(Nfft) / float(fs)
        useful_corr = useful * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t_sym_start + time_vec))
        
        # FFT → получаем принятые поднесущие
        R = np.fft.fft(useful_corr) / Nfft
        
        # Оценка канала для этого символа: Hk_i = R / S_ref
        Hk_i = R[subc_inds] / S_pilot_ref
        
        Hk_sum += Hk_i
        pilot_rms_sum += np.sqrt(np.mean(np.abs(useful)**2))
        valid_pilots += 1
    
    if valid_pilots == 0:
        print("[RX-PILOT] no valid pilots extracted")
        return None, None
    
    # Усредняем оценки канала по всем валидным пилотам
    Hk_pilot = Hk_sum / valid_pilots
    pilot_rms = pilot_rms_sum / valid_pilots
    
    print(f"[RX-PILOT] Extracted {valid_pilots}/{n_pilots} pilots, "
          f"Hk_avg_mag={np.mean(np.abs(Hk_pilot)):.4f}, pilot_rms={pilot_rms:.6f}")
    
    return Hk_pilot, pilot_rms


# -----------------------
# Вспомогательная функция для декодирования с конкретной модуляцией
# -----------------------
def _try_decode_with_modulation(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, modulation, show_waterfall=False):
    """
    Попытка декодирования пакета с конкретной модуляцией.
    Использует отдельный экземпляр эквалайзера для каждой попытки.
    
    packet_blocks_expected - количество ЛОГИЧЕСКИХ блоков.
    show_waterfall - если True, показывать водопадную диаграмму эквалайзера в реальном времени.
    
    Возвращает (decoded_bytes, rs_ok_count, used_preamble, equalizer_instance) или None при ошибке.
    """
    # Проверяем, что переменные состояния инициализированы
    if _rx_st.rx is None or _rx_st.abs_corr is None:
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
        rx_zc1 = _rx_st.rx[pref_abs + Ncp : pref_abs + Ncp + Nfft]
        rx_zc2 = _rx_st.rx[pref_abs + SYMBOL_LEN + Ncp : pref_abs + SYMBOL_LEN + Ncp + Nfft]
        cross = np.vdot(rx_zc1, rx_zc2)
        delta_phi = np.angle(cross)
        T_between = SYMBOL_LEN / float(fs)
        f_err_loc = delta_phi / (2.0 * np.pi * T_between)
        
        pilot1_start = pref_abs + 2*SYMBOL_LEN
        rx_pre1 = _rx_st.rx[pilot1_start + Ncp : pilot1_start + Ncp + Nfft]
        pilot2_start = pilot1_start + SYMBOL_LEN
        rx_pre2 = _rx_st.rx[pilot2_start + Ncp : pilot2_start + Ncp + Nfft]
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
        
        S_ref = np.fft.fft(_rx_st.preamble_td[2*SYMBOL_LEN + Ncp : 2*SYMBOL_LEN + Ncp + Nfft]) / Nfft
        Hk_est = (R1t[subc_inds] / S_ref[subc_inds] + R2t[subc_inds] / S_ref[subc_inds]) / 2
        Hk_mag_raw = np.median(np.abs(Hk_est)) if hasattr(np, 'median') else np.mean(np.abs(Hk_est))
        Hk_mag = np.clip(Hk_mag_raw, 1/2.0, None)
        Hk_s = Hk_mag * np.exp(1j*np.angle(Hk_est))
        
        # Отладочный вывод начального состояния эквалайзера
        print(f"[EQ-INIT] Hk_est: median_abs={Hk_mag_raw:.4f}, clipped={Hk_mag:.4f}, mean_phase={np.mean(np.angle(Hk_est)):.4f} rad")
        print(f"[EQ-INIT] R1t[subc] rms={np.sqrt(np.mean(np.abs(R1t[subc_inds])**2)):.6f}, S_ref[subc] rms={np.sqrt(np.mean(np.abs(S_ref[subc_inds])**2)):.6f}")
        
        # Пытаемся использовать пилотные символы для более точной оценки канала
        Hk_pilot = None
        pilot_rms = None
        if PREAMBLE_PILOT_SYMBOLS > 0:
            Hk_pilot, pilot_rms = _extract_pilot_symbols(pref_abs, f_err_loc)
        
        # Если пилоты успешно извлечены - используем их для инициализации эквалайзера
        if Hk_pilot is not None:
            # Усредняем Hk от пилотов и Hk от преамбулы для лучшей оценки
            Hk_combined = (Hk_pilot + Hk_s) / 2
            Hk_init = Hk_combined
            print(f"[EQ-INIT] Using combined Hk (pilots + preamble): avg_mag={np.mean(np.abs(Hk_init)):.4f}")
        else:
            Hk_init = Hk_s
            print(f"[EQ-INIT] Using preamble-only Hk: avg_mag={np.mean(np.abs(Hk_init)):.4f}")
        
        # Создаем отдельный экземпляр эквалайзера для этой попытки
        equalizer = AdaptiveEqualizer(initial_Hk=Hk_init, alpha=0.02, modulation=modulation)
        print(f"[EQ] Modulation {modulation}: AdaptiveEqualizer initialized with alpha=0.02, initial_avg_mag={np.mean(np.abs(Hk_init)):.4f}")
        
        # Создаём визуализацию водопадной диаграммы (если запрошена)
        waterfall = None
        if show_waterfall:
            try:
                from equalizer_waterfall import EqualizerWaterfall
                waterfall = EqualizerWaterfall(
                    subc_inds=subc_inds,
                    fs=fs,
                    Nfft=Nfft,
                    max_symbols=500,
                    title_prefix=f"EQ Waterfall pkt={packet_idx} mod={modulation}"
                )
                # Показываем начальное состояние Hk
                waterfall.update(equalizer.get_current_Hk())
                print(f"[EQ-WF] Водопадная диаграмма создана для pkt={packet_idx} mod={modulation}")
            except Exception as e:
                print(f"[EQ-WF] Ошибка создания водопадной диаграммы: {e}")
                waterfall = None
        
    except Exception as e:
        print(f"[RX-DBG-DECODE] Exception in channel estimation for {modulation}: {e}")
        return None
    
    try:
        # Вычисляем pre_rms по преамбуле (ZC+ZC+P+P), без пилотных символов
        # Пилоты идут ДО преамбулы, поэтому не включаем их
        pre_segment = _rx_st.rx[pref_abs : pref_abs + len(_rx_st.preamble_td)]
        pre_rms = np.sqrt(np.mean(pre_segment**2)) if pre_segment.size > 0 else MIN_RMS
        if pre_rms < MIN_RMS:
            pre_rms = MIN_RMS
        packet_gain = 0.5 / pre_rms
        
        # Инициализируем AGC из RMS пилотов, если они доступны
        # Это даст правильный gain для данных с первого символа
        if pilot_rms is not None and pilot_rms > MIN_RMS:
            # Инициализируем last_agc_rms из реального RMS пилотов
            # Масштабируем на packet_gain для согласованности
            _rx_st.last_agc_rms = pilot_rms * packet_gain
            print(f"[AGC-INIT] Initialized last_agc_rms from pilots: {pilot_rms:.6f} * {packet_gain:.6f} = {_rx_st.last_agc_rms:.6f}")
        else:
            # Fallback: используем pre_rms
            _rx_st.last_agc_rms = pre_rms * packet_gain
            print(f"[AGC-INIT] Initialized last_agc_rms from preamble: {pre_rms:.6f} * {packet_gain:.6f} = {_rx_st.last_agc_rms:.6f}")
    except Exception:
        packet_gain = 1.0
    
    # Переводим логические блоки в физические символы
    physical_symbols_expected = packet_blocks_expected * ofdm_symbols_per_block
    
    pkt_data_start = pref_abs + len(_rx_st.preamble_td)
    pkt_payload_samples = physical_symbols_expected * SYMBOL_LEN
    seg = packet_gain * _rx_st.rx[pkt_data_start : pkt_data_start + pkt_payload_samples]
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
        est_rms = (1.0 - AGC_ALPHA) * _rx_st.last_agc_rms + AGC_ALPHA * cur_rms
        _rx_st.last_agc_rms = est_rms
        if est_rms < 1e-12:
            est_rms = 1e-12
        gain_sym = SYMBOL_TARGET_RMS / est_rms
        # Ограничиваем gain_sym, чтобы избежать перегрузки или слишком слабого сигнала
        gain_sym = np.clip(gain_sym, 0.1, 10.0)
        
        # Отладочный вывод AGC
        if AGC_DEBUG:
            print(f"[AGC] pkt={packet_idx} frame={idxf} cur_rms={cur_rms:.6f} est_rms={est_rms:.6f} gain_sym={gain_sym:.3f}")
        
        # Сохраняем данные AGC для последующего построения графика
        _rx_st.agc_history_list.append({
            'symbol_idx': _rx_st._global_symbol_counter,
            'pkt_idx': packet_idx,
            'frame_idx': idxf,
            'cur_rms': cur_rms,
            'est_rms': est_rms,
            'gain_sym': gain_sym
        })
        _rx_st._global_symbol_counter += 1
        
        useful = useful * gain_sym
        # Применяем AGC-усиление к сигналу с компенсацией частотного сдвига
        useful_corr = useful * np.exp(-1j * 2.0 * np.pi * f_err_loc * tv)
        
        try:
            F = np.fft.fft(useful_corr) / Nfft
            subc = equalizer.process(F[subc_inds])
            # Отладочный вывод амплитуды после эквалайзера
            if AGC_DEBUG and idxf < 3:
                print(f"[EQ-OUT] pkt={packet_idx} frame={idxf} subc_rms={np.sqrt(np.mean(np.abs(subc)**2)):.6f} subc_max={np.max(np.abs(subc)):.6f}")
        except Exception:
            subc = np.zeros(Nsub, dtype=complex)

        # Сохраняем текущее состояние Hk в историю эквалайзера для водопадной диаграммы
        try:
            _rx_st.equalizer_history_list.append(equalizer.get_current_Hk().copy())
            # print(f"[EQ-HIST] Сохранен снимок Hk #{len(_rx_st.equalizer_history_list)} "
            #       f"для символа {idxf} пакета {packet_idx}")
        except Exception as e:
            print(f"[EQ-HIST] Ошибка сохранения Hk: {e}")
        
        # Обновляем водопадную диаграмму эквалайзера (если включена)
        if waterfall is not None:
            try:
                waterfall.update(equalizer.get_current_Hk())
            except Exception as e:
                print(f"[EQ-WF] Ошибка обновления водопада: {e}")
        
        if modem_config.subc_phases is not None and np.any(modem_config.subc_phases != 0):
            subc = subc * np.exp(-1j * modem_config.subc_phases)
        
        # Отладочный вывод амплитуды перед сохранением
        if AGC_DEBUG and idxf < 3:
            print(f"[POST-PHASE] pkt={packet_idx} frame={idxf} subc_rms={np.sqrt(np.mean(np.abs(subc)**2)):.6f}")
        
        rx_syms_pkt_list.append(subc)
        
        # Сохраняем символы для градиентного созвездия
        _rx_st.rx_constellation_symbols.extend(subc)
        # print(f"[DEBUG] Добавлено {len(subc)} символов, всего: {len(_rx_st.rx_constellation_symbols)}")
    
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
    else:
        bits_pkt = qpsk_demap(rx_syms_pkt)
    
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
    
    # Закрываем водопадную диаграмму если она была создана
    if waterfall is not None:
        try:
            waterfall.close()
            print(f"[EQ-WF] Водопадная диаграмма закрыта для pkt={packet_idx}")
        except Exception as e:
            print(f"[EQ-WF] Ошибка закрытия водопада: {e}")

    # Возвращаем результат с экземпляром эквалайзера
    return (decoded_blocks, rs_ok, pref_abs, equalizer)


# -----------------------
# decode_packet_at_candidate
# -----------------------
def decode_packet_at_candidate(pref_abs, packet_blocks_expected, packet_idx=0, bytes_before_packet=0, expected_total=0, show_waterfall=False):
    """Декодирование пакета по кандидату синхронизации.
    
    packet_blocks_expected - количество ЛОГИЧЕСКИХ блоков.
    show_waterfall - если True, показывать водопадную диаграмму эквалайзера в реальном времени.
    """
    best_result = (b'', 0, None)
    
    if _rx_st.rx is None or _rx_st.abs_corr is None:
        return best_result
    
    # Используем счётчик ошибок RS из rx_state (инициализируется автоматически)
    # Сбрасываем счётчик при необходимости
    if not hasattr(_rx_st, '_rs_fail_prints_count'):
        _rx_st._rs_fail_prints_count = 0
    
    # Для первого пакета пробуем обе модуляции параллельно
    if packet_idx == 0:
        print(f"[RX] First packet: trying both BPSK and QPSK modulation...")
        results = []
        
        for try_mod in ["QPSK", "BPSK"]:
            print(f"[RX] Trying {try_mod} modulation...")
            result = _try_decode_with_modulation(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, try_mod, show_waterfall=show_waterfall)
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
            _rx_st.global_equalizer = eq_instance
            
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
                _rx_st.pkt0_header_bytes = bytes(header_by_cw[:64])
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
        
        result = _try_decode_with_modulation(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, modulation, show_waterfall=show_waterfall)
        if result is not None:
            decoded_blocks, rs_ok, used_pre, eq_instance = result
            
            # Обновляем глобальный эквалайзер
            _rx_st.global_equalizer = eq_instance
            
            ret_bytes = b"".join([b for (b,ok) in decoded_blocks])
            return ret_bytes, rs_ok, used_pre
        else:
            return best_result


# -----------------------
# Функция автосохранения водопадной диаграммы эквалайзера
# -----------------------
def save_equalizer_waterfall(filename="rx_equalizer_waterfall.png"):
    """
    Сохраняет водопадную диаграмму эквалайзера в PNG файл.
    
    Использует equalizer_history_list из rx_state для построения водопада.
    Вызывается автоматически по окончании приёма всех пакетов.
    
    :param filename: Имя файла для сохранения (по умолчанию 'rx_equalizer_waterfall.png').
    :return: True если сохранение успешно, False в случае ошибки.
    """
    if not WF_AVAILABLE or EqualizerWaterfall is None:
        print("[RX-WF] Водопад эквалайзера отключён (модуль недоступен)")
        return False
    
    if not _rx_st.equalizer_history_list or len(_rx_st.equalizer_history_list) == 0:
        print("[RX-WF] Нет данных истории эквалайзера для сохранения")
        return False
    
    try:
        print(f"[RX-WF] Сохранение водопадной диаграммы эквалайзера: {filename}")
        
        # Создаём экземпляр водопада
        waterfall = EqualizerWaterfall(
            subc_inds=subc_inds,
            fs=fs,
            Nfft=Nfft,
            max_symbols=500,
            title_prefix="Equalizer Waterfall"
        )
        
        # Загружаем историю эквалайзера
        waterfall.update_from_history(_rx_st.equalizer_history_list)
        
        # Сохраняем в файл
        result = waterfall.save(filename)
        
        if result:
            print(f"[RX-WF] Водопадная диаграмма успешно сохранена: {filename}")
        else:
            print(f"[RX-WF] Ошибка сохранения водопадной диаграммы")
        
        return result
        
    except Exception as e:
        print(f"[RX-WF] Ошибка при сохранении водопада: {e}")
        import traceback
        traceback.print_exc()
        return False
