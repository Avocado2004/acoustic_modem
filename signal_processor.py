"""
Универсальный процессор сигнала OFDM Acoustic Modem.

Содержит единый цикл обработки для всех режимов приёма:
  - microphone (live)  : чтение из микрофона в реальном времени
  - file               : чтение из WAV-файла
  - loop               : loopback-режим (то же, что file)

Основная функция:
    process_signal_stream(data_source, preamble_td, config) -> bool

Архитектура:
    process_signal_stream()
        ├── _search_first_preamble()   — поиск первой преамбулы (нормализованная корреляция)
        ├── _refine_sync()             — двухпроходная синхронизация (грубая + уточнение)
        ├── _decode_first_packet()     — декодирование первого пакета через rx_decoder
        ├── _parse_header_and_plan()   — разбор заголовка, определение позиций пакетов
        ├── _decode_remaining_packets()— цикл по пакетам: уточнение → декодирование
        └── _assemble_and_verify()     — сборка данных, CRC32-проверка, сохранение

Все комментарии и документация — на русском языке.
"""

import time
import zlib
from collections import deque

import numpy as np

import modem_config
from modem_config import (
    Nfft, Ncp, Nsub, subc_inds, fs, SYMBOL_LEN, DEFAULT_PACKET_BLOCKS,
    RS_CW_BITS, RS_DATA_BYTES, RS_CW_BYTES, rs,
    SYMBOL_TARGET_RMS, AGC_ALPHA, AGC_DEBUG, MIN_RMS,
    PLOTTING_AVAILABLE, _MAX_RS_FAIL_PRINTS_GLOBAL, SYNC_WINDOW_HALF,
)

from modem_modulation import build_preamble
from modem_packet import parse_header, simulate_packet_positions
from rx_decoder import decode_packet_at_candidate
from signal_utils import fftconvolve

# Импортируем модуль состояния приёмника (единый источник глобальных переменных)
import rx_state as _rx_st


# ---------------------------------------------------------------------------
# Константы процессора
# ---------------------------------------------------------------------------

# Порог нормализованной корреляции для обнаружения преамбулы
PREAMBLE_CORR_THRESHOLD = 0.35

# Размер чанка чтения из источника данных (в сэмплах)
CHUNK_SIZE = 2048

# Задержка между итерациями чтения (в секундах)
_SLEEP_LIVE = 0.01   # live-режим (микрофон) — больше задержка, меньше CPU
_SLEEP_FILE = 0.001  # file/loop-режим — минимальная задержка


# ---------------------------------------------------------------------------
# Внутренние вспомогательные функции
# ---------------------------------------------------------------------------

def _normalized_correlation(buf, preamble_td_local):
    """
    Вычисление нормализованной корреляции между буфером и преамбулой.

    Использует fftconvolve для быстрой корреляции.
    Нормализация: |corr| / sqrt(energy * pre_energy)

    Параметры
    ----------
    buf : np.ndarray
        Буфер сигнала
    preamble_td_local : np.ndarray
        Временное представление преамбулы

    Возвращает
    -------
    norm_corr : np.ndarray
        Нормализованная корреляция (значения 0..1)
    """
    pre_len = len(preamble_td_local)
    pre_energy = np.sum(preamble_td_local * preamble_td_local)

    # Корреляция через fftconvolve (быстрее np.correlate для длинных буферов)
    corr = fftconvolve(buf, preamble_td_local[::-1], mode='valid')

    # Энергия скользящего окна
    energy = np.convolve(buf * buf, np.ones(pre_len)[::-1], mode='valid')

    # Нормализация
    denom = np.sqrt(energy * pre_energy)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_corr = np.abs(corr) / denom
        norm_corr[~np.isfinite(norm_corr)] = 0.0

    return norm_corr


def _refine_sync(buf, center, preamble_td_local):
    """
    Уточнение позиции преамбулы в окне вокруг center.

    Вырезает локальный сегмент [center : center + window_len],
    вычисляет нормализованную корреляцию и возвращает уточнённую позицию.

    Параметры
    ----------
    buf : np.ndarray
        Буфер сигнала (достаточно длинный, чтобы покрыть окно)
    center : int
        Приблизительная позиция (начало окна поиска)
    preamble_td_local : np.ndarray
        Временное представление преамбулы

    Возвращает
    -------
    refined_abs : int
        Уточнённая абсолютная позиция преамбулы в буфере
    peak_val : float
        Значение корреляции в найденном пике
    """
    pre_len = len(preamble_td_local)
    window_len = pre_len + 2 * SYMBOL_LEN

    # Проверяем, что буфер достаточно длинный
    if center + window_len > buf.size:
        # Если буфер короткий — уточнение невозможно, возвращаем center
        print(f"[SP-REFINE] buffer too short ({buf.size}) for window at center={center}, window_len={window_len}")
        return center, 0.0

    local_segment = buf[center: center + window_len]
    norm_local = _normalized_correlation(local_segment, preamble_td_local)

    loc_peak = int(np.argmax(norm_local))
    peak_val = float(norm_local[loc_peak])
    refined_abs = center + loc_peak

    return refined_abs, peak_val


def _accumulate_buffer(data_source, ring_buffer, total_samples, needed_samples, sleep_time, label=""):
    """
    Накопление данных из data_source в ring_buffer до достижения needed_samples.

    Читает чанки из data_source, добавляет в deque (ring_buffer),
    обновляет счётчик total_samples.

    Параметры
    ----------
    data_source : DataSource
        Источник данных (файл, микрофон, loopback)
    ring_buffer : deque
        Буфер накопления (collections.deque of np.ndarray)
    total_samples : int
        Текущее количество накопленных сэмплов
    needed_samples : int
        Сколько сэмплов нужно накопить
    sleep_time : float
        Задержка между чтениями (секунды)
    label : str
        Метка для отладочного вывода

    Возвращает
    -------
    total_samples : int
        Обновлённое количество накопленных сэмплов
    success : bool
        True если удалось накопить достаточно данных
    """
    # Проверяем, есть ли уже достаточно данных в буфере
    if total_samples >= needed_samples:
        return total_samples, True

    # Проверяем, есть ли данные в источнике через has_more()
    # (для file/loop — всегда True пока не прочитан весь файл,
    #  для live — True пока поток активен)
    iteration = 0
    while total_samples < needed_samples:
        iteration += 1

        # Проверяем, есть ли ещё данные в источнике
        if hasattr(data_source, 'has_more') and not data_source.has_more():
            print(f"[SP-ACCUM-{label}] data_source has no more data, "
                  f"collected {total_samples}/{needed_samples}")
            return total_samples, False

        # Пробуем прочитать чанк
        chunk = None
        if hasattr(data_source, 'read_chunk'):
            try:
                chunk = data_source.read_chunk()
            except Exception as e:
                print(f"[SP-ACCUM-{label}] read_chunk exception: {e}")
        elif hasattr(data_source, 'read_snapshot'):
            # Совместимость со старым интерфейсом DataSource
            snapshot = data_source.read_snapshot()
            # Вычисляем, сколько новых сэмплов появилось
            if hasattr(data_source, 'get_samples_read'):
                new_total = data_source.get_samples_read()
            else:
                new_total = len(snapshot)
            new_samples = new_total - total_samples
            if new_samples > 0:
                if snapshot.size > new_samples:
                    chunk = snapshot[-new_samples:].copy()
                else:
                    chunk = snapshot.copy()

        if chunk is not None and chunk.size > 0:
            ring_buffer.append(chunk)
            total_samples += chunk.size
            if iteration % 50 == 0:
                print(f"[SP-ACCUM-{label}] accumulated {total_samples}/{needed_samples} samples")
        else:
            # Нет данных — ждём
            time.sleep(sleep_time)

        # Защита от бесконечного цикла для file/loop
        # (если файл закончился, has_more() вернёт False)
        if iteration > 10000:
            print(f"[SP-ACCUM-{label}] too many iterations, breaking")
            break

    success = total_samples >= needed_samples
    return total_samples, success


def _buffer_to_array(ring_buffer, total_samples):
    """
    Конвертация ring_buffer (deque of chunks) в единый np.ndarray.

    Параметры
    ----------
    ring_buffer : deque
        Буфер накопления
    total_samples : int
        Общее количество сэмплов

    Возвращает
    -------
    buf : np.ndarray
        Единый массив всех накопленных данных
    """
    if not ring_buffer:
        return np.array([], dtype=np.float64)
    return np.concatenate(list(ring_buffer))


# ---------------------------------------------------------------------------
# Основная функция обработки
# ---------------------------------------------------------------------------

def process_signal_stream(data_source, preamble_td_local, config=None):
    """
    Основной цикл обработки сигнала из потока данных.

    Универсальный процессор для всех режимов (microphone, file, loop).
    Выполняет:
    1. Поиск первой преамбулы (нормализованная корреляция, порог 0.35)
    2. Двухпроходную синхронизацию (грубая + уточнение)
    3. Декодирование первого пакета
    4. Разбор заголовка → определение количества и позиций пакетов
    5. Цикл по пакетам: уточнение синхронизации → декодирование
    6. Сборка данных → CRC32-проверка → сохранение/возврат результата

    Параметры
    ----------
    data_source : DataSource
        Источник данных. Должен реализовывать:
        - read_chunk() -> np.ndarray  (прочитать следующий чанк)
        - has_more() -> bool          (есть ли ещё данные)
        - get_all_data() -> np.ndarray (получить все накопленные данные)
        - get_samples_read() -> int   (сколько сэмплов прочитано)
        - stop()                      (остановить источник)
        Также поддерживается старый интерфейс:
        - read_snapshot() -> np.ndarray
        - wait_for_samples(n, timeout) -> bool
    preamble_td_local : np.ndarray
        Временное представление преамбулы
    config : object, optional
        Объект конфигурации. Если None — используется modem_config.

    Возвращает
    -------
    bool
        True если приём успешен, False в случае ошибки
    """
    # Определяем конфигурацию
    if config is None:
        config = modem_config

    pre_len = len(preamble_td_local)
    print(f"[SP] process_signal_stream started, preamble_len={pre_len}, "
          f"threshold={PREAMBLE_CORR_THRESHOLD}, chunk_size={CHUNK_SIZE}")

    # Определяем режим для выбора задержки
    # Если data_source имеет атрибут 'sleep_time' — используем его
    # Иначе определяем по типу
    if hasattr(data_source, 'sleep_time'):
        sleep_time = data_source.sleep_time
    else:
        # По умолчанию используем file-задержку
        sleep_time = _SLEEP_FILE

    print(f"[SP] sleep_time={sleep_time}")

    # Сбрасываем состояние приёмника
    _rx_st.reset_state()
    _rx_st.preamble_td = preamble_td_local

    # Определяем, какой интерфейс использует data_source
    use_legacy = hasattr(data_source, 'read_snapshot') and hasattr(data_source, 'wait_for_samples')
    print(f"[SP] data_source legacy interface: {use_legacy}")

    # -----------------------------------------------------------------------
    # Режим 1: Legacy interface (read_snapshot / wait_for_samples)
    # -----------------------------------------------------------------------
    if use_legacy:
        return _process_legacy_interface(data_source, preamble_td_local, config, pre_len, sleep_time)

    # -----------------------------------------------------------------------
    # Режим 2: Новый интерфейс (read_chunk / has_more / ring buffer)
    # -----------------------------------------------------------------------
    return _process_chunk_interface(data_source, preamble_td_local, config, pre_len, sleep_time)


def _process_legacy_interface(data_source, preamble_td_local, config, pre_len, sleep_time):
    """
    Обработка сигнала через legacy-интерфейс DataSource (read_snapshot/wait_for_samples).

    Используется для совместимости с существующими FileDataSource и MicrophoneDataSource.

    Параметры
    ----------
    data_source : DataSource
        Источник данных с методами read_snapshot() и wait_for_samples()
    preamble_td_local : np.ndarray
        Временное представление преамбулы
    config : object
        Конфигурация
    pre_len : int
        Длина преамбулы
    sleep_time : float
        Задержка между итерациями

    Возвращает
    -------
    bool
        True если приём успешен
    """
    print("[SP] Using legacy interface (read_snapshot/wait_for_samples)")

    # -----------------------------------------------------------------------
    # Этап 1: Поиск первой преамбулы
    # -----------------------------------------------------------------------
    print("[SP] Stage 1: searching first preamble (continuous scan)")
    found_first = False
    first_sync_abs = None
    search_iteration = 0
    prev_buf_size_legacy = 0
    stuck_count_legacy = 0

    while not found_first:
        search_iteration += 1
        buf = data_source.read_snapshot()

        # Проверка: если буфер перестал расти и файл неактивен — выходим из цикла поиска
        if buf.size == prev_buf_size_legacy and hasattr(data_source, 'is_active') and not data_source.is_active():
            stuck_count_legacy += 1
            if stuck_count_legacy >= 3:
                print(f"[SP-SEARCH] buffer stuck at {buf.size} samples, file inactive. "
                      f"Breaking search loop.")
                break
        else:
            stuck_count_legacy = 0
        prev_buf_size_legacy = buf.size

        if buf.size < pre_len:
            if search_iteration % 50 == 0:
                print(f"[SP-SEARCH] iteration {search_iteration}, "
                      f"buf.size={buf.size} < pre_len={pre_len}")
            time.sleep(sleep_time)
            continue

        # Вычисляем нормализованную корреляцию
        norm_corr = _normalized_correlation(buf, preamble_td_local)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        if search_iteration % 20 == 0 or peak_val > PREAMBLE_CORR_THRESHOLD * 0.8:
            print(f"[SP-SEARCH] iteration {search_iteration}, "
                  f"peak_idx={peak_idx}, peak_val={peak_val:.4f}, "
                  f"threshold={PREAMBLE_CORR_THRESHOLD}, buf.size={buf.size}")

        if peak_val < PREAMBLE_CORR_THRESHOLD:
            time.sleep(sleep_time)
            continue

        # Найден кандидат — уточняем синхронизацию
        cand_abs = peak_idx
        print(f"[SP-SEARCH] candidate found at {cand_abs}, peak_val={peak_val:.4f}")

        # Двухпроходная синхронизация: уточнение в окне ±SYMBOL_LEN
        refine_center = max(0, cand_abs - SYMBOL_LEN)
        refine_window_len = pre_len + 2 * SYMBOL_LEN

        buf2 = data_source.read_snapshot()
        if refine_center + refine_window_len > buf2.size:
            print(f"[SP-SEARCH] buffer too short for refine, waiting...")
            if not data_source.wait_for_samples(refine_center + refine_window_len, timeout=10.0):
                print("[SP-SEARCH] wait_for_samples timeout, continue scanning")
                time.sleep(sleep_time)
                continue
            buf2 = data_source.read_snapshot()

        refined_abs, refined_peak = _refine_sync(buf2, refine_center, preamble_td_local)
        print(f"[SP-SEARCH] refined: cand={cand_abs} -> refined={refined_abs}, "
              f"refined_peak={refined_peak:.4f}")

        # Проверяем, достаточно ли данных для декодирования первого пакета
        packet_blocks_guess = DEFAULT_PACKET_BLOCKS
        # Для QPSK: OFDM_SYMBOLS_PER_BLOCK=1, для BPSK: OFDM_SYMBOLS_PER_BLOCK=2
        # Используем максимальное значение (BPSK) чтобы гарантировать достаточно данных
        # для любой модуляции (модуляция определится после декодирования первого пакета)
        physical_symbols_guess = packet_blocks_guess * 2  # максимум для BPSK
        needed_total = refined_abs + pre_len + physical_symbols_guess * SYMBOL_LEN + SYMBOL_LEN

        print(f"[SP-SEARCH] refined_abs={refined_abs}, needed_total={needed_total}, "
              f"packet_blocks_guess={packet_blocks_guess}, "
              f"physical_symbols_guess={physical_symbols_guess} (max for BPSK)")

        if not data_source.wait_for_samples(needed_total, timeout=10.0):
            # Файл может быть дочитан до конца — проверяем
            if hasattr(data_source, 'is_active') and not data_source.is_active():
                print(f"[SP-SEARCH] file exhausted, but preamble was found at {refined_abs}. "
                      f"Attempting to decode with available data.")
                # Проверяем что есть хоть какие-то данные после преамбулы
                buf_check = data_source.read_snapshot()
                available_after_preamble = buf_check.size - refined_abs - pre_len
                if available_after_preamble > SYMBOL_LEN:
                    print(f"[SP-SEARCH] {available_after_preamble} samples available after preamble, "
                          f"proceeding with decoding")
                    first_sync_abs = refined_abs
                    found_first = True
                else:
                    print(f"[SP-SEARCH] not enough data after preamble "
                          f"({available_after_preamble} samples), aborting")
                    break
            else:
                print(f"[SP-SEARCH] not enough samples ({needed_total}), continue scanning")
                time.sleep(sleep_time)
                continue
        else:
            first_sync_abs = refined_abs
            found_first = True

    if first_sync_abs is None:
        print("[SP-ERR] no preamble found in stream")
        return False

    print(f"[SP] First preamble found at abs={first_sync_abs}")

    # -----------------------------------------------------------------------
    # Этап 2: Декодирование первого пакета
    # -----------------------------------------------------------------------
    print(f"[SP] Stage 2: decoding first packet at abs={first_sync_abs}")

    buf_all = data_source.read_snapshot()

    # Устанавливаем глобальные переменные состояния для декодера
    _rx_st.rx = buf_all.copy()
    _rx_st.abs_corr = np.abs(fftconvolve(buf_all, preamble_td_local[::-1], mode='valid'))

    print(f"[SP] buf_all.size={buf_all.size}, abs_corr.size={_rx_st.abs_corr.size}")

    pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = decode_packet_at_candidate(
        first_sync_abs, DEFAULT_PACKET_BLOCKS, packet_idx=0,
        bytes_before_packet=0, expected_total=0
    )

    if pkt0_decoded is None or len(pkt0_decoded) < 64:
        print(f"[SP-ERR] failed to decode first packet (decoded={len(pkt0_decoded) if pkt0_decoded else 0} bytes)")
        data_source.stop()
        return False

    print(f"[SP] First packet decoded: RS_OK={pkt0_rs_ok}, "
          f"used_pre={pkt0_used_pre}, bytes={len(pkt0_decoded)}")
    print(f"[SP] First 16 bytes hex: {pkt0_decoded[:16].hex()}")

    # -----------------------------------------------------------------------
    # Этап 3: Разбор заголовка и планирование пакетов
    # -----------------------------------------------------------------------
    print("[SP] Stage 3: parsing header and planning packets")

    header64 = pkt0_decoded[:64]

    try:
        hdr = parse_header(header64)
    except Exception as e:
        print(f"[SP-ERR] parse_header failed: {e}")
        import traceback
        traceback.print_exc()
        data_source.stop()
        return False

    print(f"[SP] Header parsed: mode={hdr.get('mode')}, data_len={hdr.get('data_len')}, "
          f"filename={hdr.get('filename')}, packet_blocks={hdr.get('packet_blocks')}, "
          f"modulation={hdr.get('modulation')}")

    modulation_from_header = hdr.get('modulation', 'QPSK')
    mode_rx = hdr['mode']
    total_sz = hdr['data_len']
    fname = hdr['filename']
    packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)

    # Устанавливаем модуляцию
    modem_config.MODULATION = modulation_from_header
    if modulation_from_header == "BPSK":
        modem_config.BITS_PER_SYMBOL = 1
        modem_config.OFDM_SYMBOLS_PER_BLOCK = 2
    else:
        modem_config.BITS_PER_SYMBOL = 2
        modem_config.OFDM_SYMBOLS_PER_BLOCK = 1
    modem_config.BITS_PER_OFDM_SYMBOL = Nsub * modem_config.BITS_PER_SYMBOL

    print(f"[SP] Modulation set to {modulation_from_header}: "
          f"BITS_PER_SYMBOL={modem_config.BITS_PER_SYMBOL}, "
          f"OFDM_SYMBOLS_PER_BLOCK={modem_config.OFDM_SYMBOLS_PER_BLOCK}")

    # Симулируем позиции пакетов
    positions_no_preroll = simulate_packet_positions(
        total_sz,
        fname.encode('utf-8') if fname else b'',
        (mode_rx == b'T'),
        packet_blocks_from_hdr,
        preamble_len=pre_len,
        symbol_len=SYMBOL_LEN,
        gap_samples=0
    )

    if len(positions_no_preroll) == 0:
        print("[SP-ERR] simulate_packet_positions returned no positions")
        data_source.stop()
        return False

    print(f"[SP] Expected {len(positions_no_preroll)} packets at positions: "
          f"{positions_no_preroll[:5]}{'...' if len(positions_no_preroll) > 5 else ''}")

    # Вычисляем абсолютные позиции
    base_abs = first_sync_abs - positions_no_preroll[0]
    expected_abs = [base_abs + int(x) for x in positions_no_preroll]

    # -----------------------------------------------------------------------
    # Этап 4: Цикл по пакетам
    # -----------------------------------------------------------------------
    print("[SP] Stage 4: decoding packets")

    assembled_chunks = []
    bytes_collected = 0
    expected_total = total_sz
    packet_blocks_val = packet_blocks_from_hdr

    for pkt_idx, pref in enumerate(expected_abs):
        # Для первого пакета используем уже декодированные данные
        if pkt_idx == 0:
            decoded_bytes = pkt0_decoded
            rs_ok_count = pkt0_rs_ok
            used_preamble = pkt0_used_pre
            print(f"[SP-PKT{pkt_idx}] reused: pref={pref}, used_pre={used_preamble}, "
                  f"RS_OK={rs_ok_count}, bytes={len(decoded_bytes)}")
        else:
            # Вычисляем сколько данных нужно для этого пакета
            pref_backoff = max(0, pref - SYMBOL_LEN)
            physical_symbols_val = packet_blocks_val * modem_config.OFDM_SYMBOLS_PER_BLOCK
            needed_for_pkt = (pref_backoff + SYMBOL_LEN + pre_len +
                              physical_symbols_val * SYMBOL_LEN + SYMBOL_LEN)

            # Ждём достаточно данных
            if not data_source.wait_for_samples(needed_for_pkt, timeout=20.0):
                print(f"[SP-PKT{pkt_idx}] timeout waiting for {needed_for_pkt} samples, aborting")
                break

            buf_now = data_source.read_snapshot()

            # Обновляем глобальные переменные
            _rx_st.rx = buf_now.copy()
            _rx_st.abs_corr = np.abs(fftconvolve(buf_now, preamble_td_local[::-1], mode='valid'))

            # Уточняем синхронизацию
            refine_center = pref_backoff
            if refine_center + pre_len + 2 * SYMBOL_LEN > buf_now.size:
                if not data_source.wait_for_samples(refine_center + pre_len + 2 * SYMBOL_LEN, timeout=5.0):
                    print(f"[SP-PKT{pkt_idx}] not enough samples for refine, abort")
                    break
                buf_now = data_source.read_snapshot()
                _rx_st.rx = buf_now.copy()
                _rx_st.abs_corr = np.abs(fftconvolve(buf_now, preamble_td_local[::-1], mode='valid'))

            refined_abs, refined_peak = _refine_sync(buf_now, refine_center, preamble_td_local)
            print(f"[SP-PKT{pkt_idx}] refine: pref={pref} -> refined={refined_abs}, "
                  f"peak={refined_peak:.4f}")

            # Декодируем
            decoded_bytes, rs_ok_count, used_preamble = decode_packet_at_candidate(
                refined_abs, packet_blocks_val, packet_idx=pkt_idx,
                bytes_before_packet=bytes_collected, expected_total=expected_total
            )
            print(f"[SP-PKT{pkt_idx}] decoded: used_pre={used_preamble}, "
                  f"RS_OK={rs_ok_count}, bytes={len(decoded_bytes)}")

        # Извлекаем payload из пакета
        if pkt_idx == 0:
            if len(decoded_bytes) < 64:
                print("[SP-ERR] first packet decoded <64 bytes, abort")
                break
            payload_start = 3 * 64  # Пропускаем 3 заголовка по 64 байта
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
                payload_chunk = decoded_bytes[4:]  # Пропускаем 4-байтный заголовок пакета
                assembled_chunks.append(payload_chunk)
                bytes_collected += len(payload_chunk)

        print(f"[SP-PROGRESS] collected {bytes_collected}/{expected_total} bytes")
        if bytes_collected >= expected_total:
            print("[SP] All expected bytes collected")
            break

    # -----------------------------------------------------------------------
    # Этап 5: Сборка данных, CRC32-проверка, сохранение
    # -----------------------------------------------------------------------
    print("[SP] Stage 5: assembling and verifying data")

    assembled = b"".join(assembled_chunks)[:expected_total]
    print(f"[SP] Assembled {len(assembled)} bytes (expected {expected_total})")

    # CRC32 проверка
    try:
        hdr_crc = hdr.get('crc32', None)
        if hdr_crc is not None and int(hdr_crc) != 0:
            calc_crc = zlib.crc32(assembled) & 0xFFFFFFFF
            if calc_crc == int(hdr_crc):
                print(f"[SP-CRC] OK: CRC32 matched (0x{calc_crc:08X})")
            else:
                print(f"[SP-CRC] MISMATCH: received 0x{int(hdr_crc):08X}, "
                      f"calculated 0x{calc_crc:08X}")
        else:
            print("[SP-CRC] No CRC32 in header, skipping check")
    except Exception as e:
        print(f"[SP-CRC] CRC check failed: {e}")

    # Сохранение результата
    out_path = None
    if mode_rx == b"F":
        out_fname = fname if fname else "rx_file"
        out_path = "rx_" + out_fname
        with open(out_path, "wb") as f:
            f.write(assembled)
        print(f"[SP-FILE] Saved: {out_path} ({len(assembled)} bytes)")
    else:
        rec_text = assembled.decode("utf-8", errors="ignore")
        print(f"[SP-TEXT] Received text ({len(rec_text)} chars):")
        print(rec_text[:200] + ("..." if len(rec_text) > 200 else ""))
        out_path = "rx_text.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(rec_text)
        print(f"[SP-TEXT] Saved to: {out_path}")

    # Построение графиков (если доступно)
    if PLOTTING_AVAILABLE:
        _plot_results()

    # Останавливаем источник данных
    try:
        data_source.stop()
    except Exception:
        pass

    return True


def _process_chunk_interface(data_source, preamble_td_local, config, pre_len, sleep_time):
    """
    Обработка сигнала через новый интерфейс DataSource (read_chunk/has_more/ring buffer).

    Использует ring buffer (deque) для накопления чанков.

    Параметры
    ----------
    data_source : DataSource
        Источник данных с методами read_chunk(), has_more()
    preamble_td_local : np.ndarray
        Временное представление преамбулы
    config : object
        Конфигурация
    pre_len : int
        Длина преамбулы
    sleep_time : float
        Задержка между итерациями

    Возвращает
    -------
    bool
        True если приём успешен
    """
    print("[SP] Using chunk interface (read_chunk/has_more/ring buffer)")

    # Ring buffer для накопления данных
    ring_buffer = deque()
    total_samples = 0

    # -----------------------------------------------------------------------
    # Этап 1: Поиск первой преамбулы
    # -----------------------------------------------------------------------
    print("[SP] Stage 1: searching first preamble (continuous scan)")
    found_first = False
    first_sync_abs = None
    search_iteration = 0
    prev_total_chunk = 0
    stuck_count_chunk = 0

    while not found_first:
        search_iteration += 1

        # Читаем чанк из источника
        chunk = None
        try:
            if hasattr(data_source, 'read_chunk'):
                chunk = data_source.read_chunk()
        except Exception as e:
            print(f"[SP-SEARCH] read_chunk exception: {e}")

        if chunk is not None and chunk.size > 0:
            ring_buffer.append(chunk)
            total_samples += chunk.size
        else:
            # Проверка: если total_samples не растёт и данных больше нет — выходим
            if total_samples == prev_total_chunk and hasattr(data_source, 'has_more') and not data_source.has_more():
                stuck_count_chunk += 1
                if stuck_count_chunk >= 3:
                    print(f"[SP-SEARCH] no new data, total_samples={total_samples}, "
                          f"source exhausted. Breaking search loop.")
                    break
            else:
                stuck_count_chunk = 0
        prev_total_chunk = total_samples

        # Проверяем, достаточно ли данных для корреляции
        if total_samples < pre_len:
            if search_iteration % 50 == 0:
                print(f"[SP-SEARCH] iteration {search_iteration}, "
                      f"total_samples={total_samples} < pre_len={pre_len}")
            time.sleep(sleep_time)
            continue

        # Конвертируем буфер в массив для корреляции
        buf = _buffer_to_array(ring_buffer, total_samples)

        # Вычисляем нормализованную корреляцию
        norm_corr = _normalized_correlation(buf, preamble_td_local)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        if search_iteration % 20 == 0 or peak_val > PREAMBLE_CORR_THRESHOLD * 0.8:
            print(f"[SP-SEARCH] iteration {search_iteration}, "
                  f"peak_idx={peak_idx}, peak_val={peak_val:.4f}, "
                  f"threshold={PREAMBLE_CORR_THRESHOLD}, total_samples={total_samples}")

        if peak_val < PREAMBLE_CORR_THRESHOLD:
            time.sleep(sleep_time)
            continue

        # Найден кандидат — уточняем синхронизацию
        cand_abs = peak_idx
        print(f"[SP-SEARCH] candidate found at {cand_abs}, peak_val={peak_val:.4f}")

        # Двухпроходная синхронизация
        refine_center = max(0, cand_abs - SYMBOL_LEN)
        refine_window_len = pre_len + 2 * SYMBOL_LEN

        if refine_center + refine_window_len > total_samples:
            # Недостаточно данных для уточнения — ждём
            print(f"[SP-SEARCH] waiting for refine data: need {refine_center + refine_window_len}, "
                  f"have {total_samples}")
            total_samples, success = _accumulate_buffer(
                data_source, ring_buffer, total_samples,
                refine_center + refine_window_len, sleep_time, "refine"
            )
            if not success:
                print("[SP-SEARCH] failed to accumulate refine data, continue scanning")
                time.sleep(sleep_time)
                continue

        buf2 = _buffer_to_array(ring_buffer, total_samples)
        refined_abs, refined_peak = _refine_sync(buf2, refine_center, preamble_td_local)
        print(f"[SP-SEARCH] refined: cand={cand_abs} -> refined={refined_abs}, "
              f"refined_peak={refined_peak:.4f}")

        # Проверяем, достаточно ли данных для декодирования
        packet_blocks_guess = DEFAULT_PACKET_BLOCKS
        # Для QPSK: OFDM_SYMBOLS_PER_BLOCK=1, для BPSK: OFDM_SYMBOLS_PER_BLOCK=2
        # Используем максимальное значение (BPSK) чтобы гарантировать достаточно данных
        # для любой модуляции (модуляция определится после декодирования первого пакета)
        physical_symbols_guess = packet_blocks_guess * 2  # максимум для BPSK
        needed_total = refined_abs + pre_len + physical_symbols_guess * SYMBOL_LEN + SYMBOL_LEN

        print(f"[SP-SEARCH] refined_abs={refined_abs}, needed_total={needed_total}, "
              f"packet_blocks_guess={packet_blocks_guess}, "
              f"physical_symbols_guess={physical_symbols_guess} (max for BPSK)")

        if total_samples < needed_total:
            print(f"[SP-SEARCH] waiting for decode data: need {needed_total}, "
                  f"have {total_samples}")
            total_samples, success = _accumulate_buffer(
                data_source, ring_buffer, total_samples,
                needed_total, sleep_time, "decode"
            )
            if not success:
                # Проверяем, может файл закончился, но данных достаточно
                if hasattr(data_source, 'has_more') and not data_source.has_more():
                    print(f"[SP-SEARCH] data source exhausted at {total_samples} samples. "
                          f"Attempting to decode with available data.")
                    available_after_preamble = total_samples - refined_abs - pre_len
                    if available_after_preamble > SYMBOL_LEN:
                        print(f"[SP-SEARCH] {available_after_preamble} samples available "
                              f"after preamble, proceeding with decoding")
                        first_sync_abs = refined_abs
                        found_first = True
                    else:
                        print(f"[SP-SEARCH] not enough data after preamble "
                              f"({available_after_preamble} samples), aborting")
                        break
                else:
                    print("[SP-SEARCH] failed to accumulate decode data, continue scanning")
                    time.sleep(sleep_time)
                    continue

        if not found_first:
            first_sync_abs = refined_abs
            found_first = True

    if first_sync_abs is None:
        print("[SP-ERR] no preamble found in stream")
        return False

    print(f"[SP] First preamble found at abs={first_sync_abs}")

    # -----------------------------------------------------------------------
    # Этап 2: Декодирование первого пакета
    # -----------------------------------------------------------------------
    print(f"[SP] Stage 2: decoding first packet at abs={first_sync_abs}")

    buf_all = _buffer_to_array(ring_buffer, total_samples)

    # Устанавливаем глобальные переменные состояния для декодера
    _rx_st.rx = buf_all.copy()
    _rx_st.abs_corr = np.abs(fftconvolve(buf_all, preamble_td_local[::-1], mode='valid'))

    print(f"[SP] buf_all.size={buf_all.size}, abs_corr.size={_rx_st.abs_corr.size}")

    pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = decode_packet_at_candidate(
        first_sync_abs, DEFAULT_PACKET_BLOCKS, packet_idx=0,
        bytes_before_packet=0, expected_total=0
    )

    if pkt0_decoded is None or len(pkt0_decoded) < 64:
        print(f"[SP-ERR] failed to decode first packet "
              f"(decoded={len(pkt0_decoded) if pkt0_decoded else 0} bytes)")
        try:
            data_source.stop()
        except Exception:
            pass
        return False

    print(f"[SP] First packet decoded: RS_OK={pkt0_rs_ok}, "
          f"used_pre={pkt0_used_pre}, bytes={len(pkt0_decoded)}")
    print(f"[SP] First 16 bytes hex: {pkt0_decoded[:16].hex()}")

    # -----------------------------------------------------------------------
    # Этап 3: Разбор заголовка и планирование пакетов
    # -----------------------------------------------------------------------
    print("[SP] Stage 3: parsing header and planning packets")

    header64 = pkt0_decoded[:64]

    try:
        hdr = parse_header(header64)
    except Exception as e:
        print(f"[SP-ERR] parse_header failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            data_source.stop()
        except Exception:
            pass
        return False

    print(f"[SP] Header parsed: mode={hdr.get('mode')}, data_len={hdr.get('data_len')}, "
          f"filename={hdr.get('filename')}, packet_blocks={hdr.get('packet_blocks')}, "
          f"modulation={hdr.get('modulation')}")

    modulation_from_header = hdr.get('modulation', 'QPSK')
    mode_rx = hdr['mode']
    total_sz = hdr['data_len']
    fname = hdr['filename']
    packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)

    # Устанавливаем модуляцию
    modem_config.MODULATION = modulation_from_header
    if modulation_from_header == "BPSK":
        modem_config.BITS_PER_SYMBOL = 1
        modem_config.OFDM_SYMBOLS_PER_BLOCK = 2
    else:
        modem_config.BITS_PER_SYMBOL = 2
        modem_config.OFDM_SYMBOLS_PER_BLOCK = 1
    modem_config.BITS_PER_OFDM_SYMBOL = Nsub * modem_config.BITS_PER_SYMBOL

    print(f"[SP] Modulation set to {modulation_from_header}: "
          f"BITS_PER_SYMBOL={modem_config.BITS_PER_SYMBOL}, "
          f"OFDM_SYMBOLS_PER_BLOCK={modem_config.OFDM_SYMBOLS_PER_BLOCK}")

    # Симулируем позиции пакетов
    positions_no_preroll = simulate_packet_positions(
        total_sz,
        fname.encode('utf-8') if fname else b'',
        (mode_rx == b'T'),
        packet_blocks_from_hdr,
        preamble_len=pre_len,
        symbol_len=SYMBOL_LEN,
        gap_samples=0
    )

    if len(positions_no_preroll) == 0:
        print("[SP-ERR] simulate_packet_positions returned no positions")
        try:
            data_source.stop()
        except Exception:
            pass
        return False

    print(f"[SP] Expected {len(positions_no_preroll)} packets at positions: "
          f"{positions_no_preroll[:5]}{'...' if len(positions_no_preroll) > 5 else ''}")

    # Вычисляем абсолютные позиции
    base_abs = first_sync_abs - positions_no_preroll[0]
    expected_abs = [base_abs + int(x) for x in positions_no_preroll]

    # -----------------------------------------------------------------------
    # Этап 4: Цикл по пакетам
    # -----------------------------------------------------------------------
    print("[SP] Stage 4: decoding packets")

    assembled_chunks = []
    bytes_collected = 0
    expected_total = total_sz
    packet_blocks_val = packet_blocks_from_hdr

    for pkt_idx, pref in enumerate(expected_abs):
        # Для первого пакета используем уже декодированные данные
        if pkt_idx == 0:
            decoded_bytes = pkt0_decoded
            rs_ok_count = pkt0_rs_ok
            used_preamble = pkt0_used_pre
            print(f"[SP-PKT{pkt_idx}] reused: pref={pref}, used_pre={used_preamble}, "
                  f"RS_OK={rs_ok_count}, bytes={len(decoded_bytes)}")
        else:
            # Вычисляем сколько данных нужно для этого пакета
            pref_backoff = max(0, pref - SYMBOL_LEN)
            physical_symbols_val = packet_blocks_val * modem_config.OFDM_SYMBOLS_PER_BLOCK
            needed_for_pkt = (pref_backoff + SYMBOL_LEN + pre_len +
                              physical_symbols_val * SYMBOL_LEN + SYMBOL_LEN)

            # Накопляем данные если нужно
            if total_samples < needed_for_pkt:
                print(f"[SP-PKT{pkt_idx}] accumulating: need {needed_for_pkt}, "
                      f"have {total_samples}")
                total_samples, success = _accumulate_buffer(
                    data_source, ring_buffer, total_samples,
                    needed_for_pkt, sleep_time, f"pkt{pkt_idx}"
                )
                if not success:
                    print(f"[SP-PKT{pkt_idx}] failed to accumulate, aborting")
                    break

            buf_now = _buffer_to_array(ring_buffer, total_samples)

            # Обновляем глобальные переменные
            _rx_st.rx = buf_now.copy()
            _rx_st.abs_corr = np.abs(fftconvolve(buf_now, preamble_td_local[::-1], mode='valid'))

            # Уточняем синхронизацию
            refine_center = pref_backoff
            refine_window_len = pre_len + 2 * SYMBOL_LEN

            if refine_center + refine_window_len > total_samples:
                total_samples, success = _accumulate_buffer(
                    data_source, ring_buffer, total_samples,
                    refine_center + refine_window_len, sleep_time, f"refine{pkt_idx}"
                )
                if not success:
                    print(f"[SP-PKT{pkt_idx}] not enough samples for refine, abort")
                    break
                buf_now = _buffer_to_array(ring_buffer, total_samples)
                _rx_st.rx = buf_now.copy()
                _rx_st.abs_corr = np.abs(fftconvolve(buf_now, preamble_td_local[::-1], mode='valid'))

            refined_abs, refined_peak = _refine_sync(buf_now, refine_center, preamble_td_local)
            print(f"[SP-PKT{pkt_idx}] refine: pref={pref} -> refined={refined_abs}, "
                  f"peak={refined_peak:.4f}")

            # Декодируем
            decoded_bytes, rs_ok_count, used_preamble = decode_packet_at_candidate(
                refined_abs, packet_blocks_val, packet_idx=pkt_idx,
                bytes_before_packet=bytes_collected, expected_total=expected_total
            )
            print(f"[SP-PKT{pkt_idx}] decoded: used_pre={used_preamble}, "
                  f"RS_OK={rs_ok_count}, bytes={len(decoded_bytes)}")

        # Извлекаем payload из пакета
        if pkt_idx == 0:
            if len(decoded_bytes) < 64:
                print("[SP-ERR] first packet decoded <64 bytes, abort")
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

        print(f"[SP-PROGRESS] collected {bytes_collected}/{expected_total} bytes")
        if bytes_collected >= expected_total:
            print("[SP] All expected bytes collected")
            break

    # -----------------------------------------------------------------------
    # Этап 5: Сборка данных, CRC32-проверка, сохранение
    # -----------------------------------------------------------------------
    print("[SP] Stage 5: assembling and verifying data")

    assembled = b"".join(assembled_chunks)[:expected_total]
    print(f"[SP] Assembled {len(assembled)} bytes (expected {expected_total})")

    # CRC32 проверка
    try:
        hdr_crc = hdr.get('crc32', None)
        if hdr_crc is not None and int(hdr_crc) != 0:
            calc_crc = zlib.crc32(assembled) & 0xFFFFFFFF
            if calc_crc == int(hdr_crc):
                print(f"[SP-CRC] OK: CRC32 matched (0x{calc_crc:08X})")
            else:
                print(f"[SP-CRC] MISMATCH: received 0x{int(hdr_crc):08X}, "
                      f"calculated 0x{calc_crc:08X}")
        else:
            print("[SP-CRC] No CRC32 in header, skipping check")
    except Exception as e:
        print(f"[SP-CRC] CRC check failed: {e}")

    # Сохранение результата
    out_path = None
    if mode_rx == b"F":
        out_fname = fname if fname else "rx_file"
        out_path = "rx_" + out_fname
        with open(out_path, "wb") as f:
            f.write(assembled)
        print(f"[SP-FILE] Saved: {out_path} ({len(assembled)} bytes)")
    else:
        rec_text = assembled.decode("utf-8", errors="ignore")
        print(f"[SP-TEXT] Received text ({len(rec_text)} chars):")
        print(rec_text[:200] + ("..." if len(rec_text) > 200 else ""))
        out_path = "rx_text.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(rec_text)
        print(f"[SP-TEXT] Saved to: {out_path}")

    # Построение графиков (если доступно)
    if PLOTTING_AVAILABLE:
        _plot_results()

    # Останавливаем источник данных
    try:
        data_source.stop()
    except Exception:
        pass

    return True


def _plot_results():
    """
    Построение графиков после завершения приёма.

    Включает:
    - Итоговый график эквалайзера (Hk_smooth_list)
    - График AGC по символам
    - График созвездия (градиентный)
    """
    try:
        from plot_utils import plot_rx_equalizer_final, plot_agc_per_symbol, plot_constellation

        if _rx_st.Hk_smooth_list and len(_rx_st.Hk_smooth_list) > 0:
            print("[SP-PLOT] Построение итогового графика эквалайзера...")
            plot_rx_equalizer_final(_rx_st.Hk_smooth_list, subc_inds, fs, Nfft)

        if _rx_st.agc_history_list and len(_rx_st.agc_history_list) > 0:
            print("[SP-PLOT] Построение графика AGC...")
            plot_agc_per_symbol(_rx_st.agc_history_list, title="AGC per Symbol")
            _rx_st.agc_history_list.clear()
            _rx_st._global_symbol_counter = 0

        if _rx_st.rx_constellation_symbols and len(_rx_st.rx_constellation_symbols) > 0:
            print(f"[SP-PLOT] Построение созвездия с "
                  f"{len(_rx_st.rx_constellation_symbols)} символами...")
            plot_constellation(_rx_st.rx_constellation_symbols,
                               "RX Constellation (gradient)", use_gradient=True)
            _rx_st.rx_constellation_symbols.clear()

    except Exception as e:
        print(f"[SP-PLOT] Ошибка при построении графиков: {e}")


# ---------------------------------------------------------------------------
# Удобные обёртки для разных режимов
# ---------------------------------------------------------------------------

def receive_from_file(wav_path, config=None):
    """
    Приём из WAV-файла через универсальный процессор.

    Параметры
    ----------
    wav_path : str
        Путь к WAV-файлу
    config : object, optional
        Конфигурация. Если None — используется modem_config.

    Возвращает
    -------
    bool
        True если приём успешен
    """
    from data_source import FileDataSource
    from modem_config import init_phases

    print(f"[SP] receive_from_file: {wav_path}")

    # Инициализируем фазы
    init_phases()

    # Строим преамбулу
    preamble_td_local = build_preamble()
    if preamble_td_local is None:
        print("[SP-ERR] Failed to build preamble")
        return False

    # Создаём источник данных
    data_source = FileDataSource(wav_path)

    # Запускаем обработку
    return process_signal_stream(data_source, preamble_td_local, config)


def receive_from_microphone(config=None):
    """
    Живой приём с микрофона через универсальный процессор.

    Параметры
    ----------
    config : object, optional
        Конфигурация. Если None — используется modem_config.

    Возвращает
    -------
    bool
        True если приём успешен
    """
    from data_source import MicrophoneDataSource
    from modem_config import init_phases

    print("[SP] receive_from_microphone")

    # Инициализируем фазы
    init_phases()

    # Строим преамбулу
    preamble_td_local = build_preamble()
    if preamble_td_local is None:
        print("[SP-ERR] Failed to build preamble")
        return False

    # Создаём источник данных
    data_source = MicrophoneDataSource(fs, channels=1, chunk=CHUNK_SIZE)
    data_source.start()

    try:
        # Запускаем обработку
        return process_signal_stream(data_source, preamble_td_local, config)
    except KeyboardInterrupt:
        print("[SP] Interrupted by user")
        data_source.stop()
        return False
