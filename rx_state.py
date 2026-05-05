"""
Модуль состояния приемника OFDM Acoustic Modem.
Содержит глобальные переменные состояния для использования в других модулях приема.
"""

import numpy as np
import modem_config

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

# Счётчик печати ошибок RS (для ограничения вывода в консоль)
_rs_fail_prints_count = 0


def reset_state():
    """
    Сбрасывает все глобальные переменные состояния к значениям по умолчанию.
    Полезно для тестирования или перезапуска приема.
    """
    global rx, abs_corr, preamble_td, last_agc_rms, post_sync
    global sync_sample_abs, last_packet_end_sample, last_packet_rs_ok
    global pkt0_header_offset, pkt0_header_bytes, rx_syms_list
    global Hk_smooth_list, equalizer_history_list, last_packet_used_pre
    global global_equalizer, agc_history_list, _global_symbol_counter
    global rx_constellation_symbols, _rs_fail_prints_count
    
    rx = None
    abs_corr = None
    preamble_td = None
    last_agc_rms = 1.0
    post_sync = False
    sync_sample_abs = None
    last_packet_end_sample = None
    last_packet_rs_ok = None
    pkt0_header_offset = None
    pkt0_header_bytes = None
    rx_syms_list = []
    Hk_smooth_list = []
    equalizer_history_list = []
    last_packet_used_pre = None
    global_equalizer = None
    agc_history_list = []
    _global_symbol_counter = 0
    rx_constellation_symbols = []
    _rs_fail_prints_count = 0
    
    print("[RX-STATE] State reset to defaults")
