"""
Модуль конфигурации OFDM Acoustic Modem.
Содержит параметры системы, глобальные переменные, настройки фаз и импорты.
"""

import os
import math
import numpy as np
from signal_utils import fftconvolve, medfilt, correlate, find_peaks
from wav_utils import wavfile
from reedsolo import RSCodec
import threading
import time
from collections import deque

# --- Рабочая директория ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# -----------------------
# Параметры системы
# -----------------------
fs = 48000
Nfft = 512
Ncp = 128
df = fs / Nfft

# Поднесущие: установить так, чтобы было ровно 48 поднесущих в диапазоне, начинающемся от ~300 Hz
k_low = int(math.ceil(300 / df))
REQUIRED_NSUB = 48
k_high = k_low + REQUIRED_NSUB - 1
subc_inds = np.arange(k_low, k_high + 1)
Nsub = len(subc_inds)

f_low_hz = k_low * df
f_high_hz = k_high * df
print(f"[CFG] df={df:.3f} Hz, k_low={k_low} -> {f_low_hz:.1f} Hz, k_high={k_high} -> {f_high_hz:.1f} Hz, Nsub={Nsub}")

# FEC (Reed-Solomon)
RS_DATA_BYTES = 8
RS_PARITY_BYTES = 4
rs = RSCodec(RS_PARITY_BYTES)
# stronger RS for Transmission Header: 8 data bytes + 16 parity bytes
RS_HDR_PARITY = 16
RS_HDR_CW_BYTES = RS_DATA_BYTES + RS_HDR_PARITY

RS_CW_BYTES = RS_DATA_BYTES + RS_PARITY_BYTES
RS_CW_BITS = RS_CW_BYTES * 8

MAX_PAYLOAD_SIZE = 1 << 30

# PHASE METHOD
PHASE_METHOD = "schroeder"

# Modulation type: "QPSK" or "BPSK"
MODULATION = "QPSK"  # Default QPSK for backward compatibility

# Количество физических OFDM символов, составляющих один логический блок.
# Логический блок всегда несет 96 бит (одно RS кодовое слово).
# QPSK: 1 символ * 96 бит = 96 бит -> OFDM_SYMBOLS_PER_BLOCK = 1
# BPSK: 2 символа * 48 бит = 96 бит -> OFDM_SYMBOLS_PER_BLOCK = 2
OFDM_SYMBOLS_PER_BLOCK = 1  # По умолчанию для QPSK

# BITS_PER_OFDM_SYMBOL is now dynamic based on modulation
if MODULATION == "BPSK":
    BITS_PER_SYMBOL = 1  # BPSK: 1 bit per subcarrier
else:
    BITS_PER_SYMBOL = 2  # QPSK: 2 bits per subcarrier
BITS_PER_OFDM_SYMBOL = Nsub * BITS_PER_SYMBOL

# Check compatibility: BITS_PER_OFDM_SYMBOL should equal RS_CW_BITS for 1 RS cw per OFDM symbol
if BITS_PER_OFDM_SYMBOL != RS_CW_BITS:
    print(f"[WARN] bits per OFDM symbol = {BITS_PER_OFDM_SYMBOL}, RS cw bits = {RS_CW_BITS}. Expected equality for 1 cw/symbol.")
else:
    print(f"[CFG] One OFDM symbol carries exactly one RS codeword ({RS_CW_BYTES} bytes, {RS_CW_BITS} bits).")


def set_modulation(modulation_type):
    """
    Динамически устанавливает тип модуляции и обновляет все зависимые параметры.
    
    Параметры:
    - modulation_type: строка "BPSK" или "QPSK"
    
    Возвращает:
    - True если модуляция успешно изменена, False если тип не поддерживается
    """
    global MODULATION, BITS_PER_SYMBOL, BITS_PER_OFDM_SYMBOL, OFDM_SYMBOLS_PER_BLOCK
    
    modulation_type = modulation_type.upper()
    if modulation_type not in ["BPSK", "QPSK"]:
        print(f"[CFG-ERR] Неизвестный тип модуляции: {modulation_type}. Используйте 'BPSK' или 'QPSK'.")
        return False
    
    MODULATION = modulation_type
    
    if MODULATION == "BPSK":
        BITS_PER_SYMBOL = 1  # BPSK: 1 bit per subcarrier
        OFDM_SYMBOLS_PER_BLOCK = 2  # 2 символа BPSK = 1 логический блок (96 бит)
    else:
        BITS_PER_SYMBOL = 2  # QPSK: 2 bits per subcarrier
        OFDM_SYMBOLS_PER_BLOCK = 1  # 1 символ QPSK = 1 логический блок (96 бит)
    
    BITS_PER_OFDM_SYMBOL = Nsub * BITS_PER_SYMBOL
    
    print(f"[CFG] Модуляция изменена на {MODULATION}: BITS_PER_SYMBOL={BITS_PER_SYMBOL}, BITS_PER_OFDM_SYMBOL={BITS_PER_OFDM_SYMBOL}, OFDM_SYMBOLS_PER_BLOCK={OFDM_SYMBOLS_PER_BLOCK}")
    
    # Проверка совместимости с RS кодом
    if BITS_PER_OFDM_SYMBOL != RS_CW_BITS:
        print(f"[WARN] bits per OFDM symbol = {BITS_PER_OFDM_SYMBOL}, RS cw bits = {RS_CW_BITS}. Expected equality for 1 cw/symbol.")
    else:
        print(f"[CFG] One OFDM symbol carries exactly one RS codeword ({RS_CW_BYTES} bytes, {RS_CW_BITS} bits).")
    
    return True

# Habr params (unchanged)
HABR_SAMPLE_BLOCKS = 200
HABR_PHASE_GRID = 36
HABR_MAX_ITERS = 2
HABR_SEED = 12345
HABR_SAVE_FILE = "habr_phases.npy"

subc_phases = None

# Globals shared TX/RX
# DEFAULT_PACKET_BLOCKS теперь означает количество ЛОГИЧЕСКИХ блоков (по 96 бит каждый)
DEFAULT_PACKET_BLOCKS = 75  # default logical OFDM blocks per packet
GAP_OFDM_SYMBOLS = 0
SYMBOL_LEN = Nfft + Ncp
GAP_SAMPLES_DEFAULT = GAP_OFDM_SYMBOLS * SYMBOL_LEN
# ширина окна поиска кандидата (в отсчетах) — полуширина окна вокруг pref_abs
SYNC_WINDOW_HALF = 20  # можно изменить (типично Ncp или SYMBOL_LEN//2)

# limit diagnostic RS_FAIL prints globally per receive to avoid flood
_MAX_RS_FAIL_PRINTS_GLOBAL = 1

# --- PACKET-LEVEL AGC (insert after cand chosen, before pkt_data_start computation) ---
# параметры AGC
TARGET_RMS = 0.5    # целевой RMS для полезной части пакета (подберите экспериментально)
MIN_RMS = 1e-12
# параметры символьного AGC
SYMBOL_TARGET_RMS = 0.5  # Целевой RMS для OFDM символа (передача и прием)
AGC_ALPHA = 0.14           # экспоненциальный коэффициент для скользящего оценщика
# debug: вывод AGC по каждому OFDM-блоку (False в нормальной работе)
AGC_DEBUG = False

# target RMS (амплитуда) для каждого OFDM-символа на стороне TX
SYMBOL_TX_TARGET = SYMBOL_TARGET_RMS  # Теперь одинаковые значения

# -----------------------
# Параметры визуализации созвездия (Constellation Diagram)
# -----------------------
PLOT_CONSTELLATION = True  # Включить/отключить визуализацию созвездия
CONSTELLATION_TX_FILENAME = 'constellation_tx.png'  # Имя файла для передачи
CONSTELLATION_RX_FILENAME = 'constellation_rx.png'  # Имя файла для приема
CONSTELLATION_USE_GRADIENT = True  # Использовать градиент цвета (синий -> красный)
CONSTELLATION_DPI = 150  # DPI для сохранения графиков

# Созвездие после OFDM-модуляции (FD символы до IFFT, до soft clipping)
PLOT_CONSTELLATION_TX_OFDM = True  # Включить созвездие после OFDM
CONSTELLATION_TX_OFDM_FILENAME = 'constellation_tx_ofdm.png'  # Имя файла для OFDM созвездия

# Компенсация фазового сдвига в созвездии TX OFDM
# Позволяет увидеть только влияние ACE и других обработок, без начального фазового сдвига
CONSTELLATION_TX_OFDM_COMPENSATE_PHASE = True  # Компенсировать начальный фазовый сдвиг
CONSTELLATION_TX_OFDM_COMPENSATED_FILENAME = 'constellation_tx_ofdm_compensated.png'  # Имя файла для компенсированного созвездия

# -----------------------
# Active Constellation Extension (ACE) – simplified implementation
# -----------------------
ACE_MAX_ITERS = 5
ACE_PEAK_THRESHOLD = 1.01
ACE_STEP = 0.3
ACE_ALLOW_EXPANSION = True

# Conditional imports for plotting (not available on mobile platforms)
try:
    import matplotlib
    # Use non-GUI backend to avoid creating windows or requiring a display
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    PLOTTING_AVAILABLE = False
    print("[WARN] matplotlib not available, plotting disabled")

try:
    from plot_utils import plot_constellation
except ImportError:
    plot_constellation = None
    print("[WARN] plot_utils not available, constellation plotting disabled")

def init_phases():
    """Инициализация фаз поднесущих (Habr, Schroeder или случайные)."""
    global subc_phases
    if PHASE_METHOD == "habr":
        if os.path.isfile(HABR_SAVE_FILE):
            try:
                subc_phases = np.load(HABR_SAVE_FILE)
                print(f"[HABR] loaded phases from {HABR_SAVE_FILE}")
                return
            except Exception as e:
                print("[HABR] failed load, will optimize:", e)
        # Если нет сохраненных фаз, запускаем оптимизацию
        # (здесь может потребоваться импорт функций из modem_modulation, 
        # поэтому оптимизация вызывается из основного файла после всех импортов)
        print("[HABR] phase file not found, will optimize after imports")
    else:
        subc_phases = make_subcarrier_phases(PHASE_METHOD, Nsub, seed=HABR_SEED)
        print(f"[PHASE] method={PHASE_METHOD}, example degs[:8]={np.degrees(subc_phases[:8])}")

def make_subcarrier_phases(method=None, N=Nsub, seed=0, habr_phases=None):
    """Генерация фаз поднесущих различными методами."""
    if method is None:
        return np.zeros(N)
    if method == "random":
        rnd = np.random.RandomState(seed)
        return rnd.uniform(0, 2*np.pi, size=N)
    if method == "schroeder":
        k = np.arange(N)
        return np.mod(np.pi * k * (k - 1) / float(N), 2*np.pi)
    if method == "habr":
        if habr_phases is not None:
            return np.array(habr_phases) % (2*np.pi)
        k = np.arange(N)
        return np.mod(np.pi * k * (k - 1) / float(N), 2*np.pi)
    return np.zeros(N)

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
