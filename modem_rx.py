"""
Модуль приема OFDM Acoustic Modem — ТОНКИЙ ФАСАД для обратной совместимости.

Этот модуль НЕ содержит реализации логики приёма. Вся работа делегирована:
  - rx_decoder.py       — decode_packet_at_candidate(), _try_decode_with_modulation()
  - signal_processor.py  — receive_from_file() (через process_signal_stream)
  - rx_live.py          — live_receive_and_process()
  - rx_state.py          — глобальные переменные состояния

Глобальные переменные состояния сохранены здесь для совместимости с тестами
(test_modem_rx.py напрямую патчит modem_rx.rx, modem_rx.abs_corr и т.д.).

Все комментарии — на русском языке.
"""

import numpy as np
import modem_config
from modem_config import (Nfft, Ncp, Nsub, subc_inds, fs, SYMBOL_LEN, DEFAULT_PACKET_BLOCKS,
                           RS_CW_BITS, RS_DATA_BYTES,
                           RS_CW_BYTES, rs, SYMBOL_TARGET_RMS, AGC_ALPHA, AGC_DEBUG, MIN_RMS,
                           PLOTTING_AVAILABLE, _MAX_RS_FAIL_PRINTS_GLOBAL, SYNC_WINDOW_HALF)

# Параметр эквалайзера для совместимости с run_loop.py
eq_alpha = 0.02

from modem_modulation import (qpsk_demap, bpsk_demap, ofdm_symbol, build_preamble, bytes_to_bits, bits_to_bytes,
                           sync_by_corr, deinterleave_bits, AdaptiveEqualizer)

from modem_packet import parse_header, build_header, make_packet_header_bytes, simulate_packet_positions

# Импортируем модуль состояния для синхронизации
import rx_state as _rx_st


# =============================================================================
# Глобальные переменные состояния
# Сохранены для обратной совместимости — тесты напрямую патчат эти переменные.
# Фасадные функции синхронизируют их с rx_state перед делегированием.
# =============================================================================

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


# =============================================================================
# Вспомогательные функции синхронизации
# =============================================================================

def _sync_to_rx_state():
    """
    Копирует глобальные переменные из этого модуля в rx_state.
    Вызывается перед делегированием в rx_decoder, чтобы тесты,
    которые патчат modem_rx.rx, modem_rx.abs_corr и т.д.,
    корректно работали с rx_decoder (который читает из rx_state).
    """
    _rx_st.rx = rx
    _rx_st.abs_corr = abs_corr
    _rx_st.preamble_td = preamble_td
    _rx_st.last_agc_rms = last_agc_rms
    _rx_st.post_sync = post_sync
    _rx_st.sync_sample_abs = sync_sample_abs
    _rx_st.last_packet_end_sample = last_packet_end_sample
    _rx_st.last_packet_rs_ok = last_packet_rs_ok
    _rx_st.pkt0_header_offset = pkt0_header_offset
    _rx_st.pkt0_header_bytes = pkt0_header_bytes
    _rx_st.rx_syms_list = rx_syms_list
    _rx_st.Hk_smooth_list = Hk_smooth_list
    _rx_st.equalizer_history_list = equalizer_history_list
    _rx_st.last_packet_used_pre = last_packet_used_pre
    _rx_st.global_equalizer = global_equalizer
    _rx_st.agc_history_list = agc_history_list
    _rx_st._global_symbol_counter = _global_symbol_counter


def _sync_from_rx_state():
    """
    Копирует глобальные переменные из rx_state обратно в этот модуль.
    Вызывается после делегирования в rx_decoder, чтобы изменения,
    сделанные в rx_state во время декодирования, были видны в modem_rx.
    """
    global rx, abs_corr, preamble_td, last_agc_rms, post_sync
    global sync_sample_abs, last_packet_end_sample, last_packet_rs_ok
    global pkt0_header_offset, pkt0_header_bytes, rx_syms_list
    global Hk_smooth_list, equalizer_history_list, last_packet_used_pre
    global global_equalizer, agc_history_list, _global_symbol_counter

    rx = _rx_st.rx
    abs_corr = _rx_st.abs_corr
    preamble_td = _rx_st.preamble_td
    last_agc_rms = _rx_st.last_agc_rms
    post_sync = _rx_st.post_sync
    sync_sample_abs = _rx_st.sync_sample_abs
    last_packet_end_sample = _rx_st.last_packet_end_sample
    last_packet_rs_ok = _rx_st.last_packet_rs_ok
    pkt0_header_offset = _rx_st.pkt0_header_offset
    pkt0_header_bytes = _rx_st.pkt0_header_bytes
    rx_syms_list = _rx_st.rx_syms_list
    Hk_smooth_list = _rx_st.Hk_smooth_list
    equalizer_history_list = _rx_st.equalizer_history_list
    last_packet_used_pre = _rx_st.last_packet_used_pre
    global_equalizer = _rx_st.global_equalizer
    agc_history_list = _rx_st.agc_history_list
    _global_symbol_counter = _rx_st._global_symbol_counter


# =============================================================================
# Фасадные функции — делегируют работу новым модулям
# =============================================================================

def _try_decode_with_modulation(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, modulation):
    """
    Фасад: делегирует в rx_decoder._try_decode_with_modulation().
    Синхронизирует глобальные переменные состояния перед и после вызова.
    """
    from rx_decoder import _try_decode_with_modulation as _impl

    _sync_to_rx_state()
    try:
        result = _impl(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, modulation)
        return result
    finally:
        _sync_from_rx_state()


def decode_packet_at_candidate(pref_abs, packet_blocks_expected, packet_idx=0, bytes_before_packet=0, expected_total=0):
    """
    Фасад: делегирует в rx_decoder.decode_packet_at_candidate().
    Синхронизирует глобальные переменные состояния перед и после вызова.

    packet_blocks_expected — количество ЛОГИЧЕСКИХ блоков.
    """
    from rx_decoder import decode_packet_at_candidate as _impl

    _sync_to_rx_state()
    try:
        result = _impl(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total)
        return result
    finally:
        _sync_from_rx_state()


def receive_from_file(wav_path):
    """
    Фасад: делегирует в signal_processor.receive_from_file().
    Синхронизирует глобальные переменные состояния после вызова.
    """
    from signal_processor import receive_from_file as _impl

    result = _impl(wav_path)
    _sync_from_rx_state()
    return result


def live_receive_and_process():
    """
    Фасад: делегирует в rx_live.live_receive_and_process().
    Синхронизирует глобальные переменные состояния после вызова.
    """
    from rx_live import live_receive_and_process as _impl

    result = _impl()
    _sync_from_rx_state()
    return result


def get_audio():
    """
    Возвращает аудио бэкенд для кроссплатформенной поддержки.
    Для обратной совместимости с тестами и ofdm_gui_flet.py.
    """
    from audio_backend import get_audio_backend
    return get_audio_backend()
