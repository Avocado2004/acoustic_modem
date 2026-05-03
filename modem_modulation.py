"""
Модуль модуляции и демодуляции OFDM Acoustic Modem.
Содержит функции для работы с битами, QPSK/BPSK, OFDM символами и ACE.
"""

import os
import numpy as np
import struct
import zlib
import modem_config
from modem_config import (Nfft, Ncp, Nsub, subc_inds, fs, PHASE_METHOD,
                          BITS_PER_SYMBOL,
                          BITS_PER_OFDM_SYMBOL, RS_DATA_BYTES, RS_CW_BYTES, RS_CW_BITS, rs,
                          SYMBOL_TX_TARGET, ACE_MAX_ITERS, ACE_PEAK_THRESHOLD, ACE_STEP, ACE_ALLOW_EXPANSION,
                          make_subcarrier_phases, HABR_SAMPLE_BLOCKS, HABR_SEED, HABR_MAX_ITERS,
                          HABR_PHASE_GRID, HABR_SAVE_FILE)

# -----------------------
# Биты/текст/битовые утилиты
# -----------------------
def text_to_bits(text, encoding='utf-8'):
    """Преобразование текста в массив битов."""
    bs = ''.join(f"{b:08b}" for b in text.encode(encoding))
    return np.array(list(bs), dtype=int)

def bits_to_text(bits, encoding='utf-8'):
    """Преобразование массива битов в текст."""
    L = (len(bits) // 8) * 8
    b = bits[:L].reshape(-1, 8)
    data = bytes(int(''.join(str(x) for x in row), 2) for row in b)
    return data.decode(encoding, errors="ignore")

def sync_by_corr(rx, pre):
    """Синхронизация по корреляции."""
    from signal_utils import fftconvolve
    corr = fftconvolve(rx, pre[::-1], mode='valid')
    return int(np.argmax(np.abs(corr)))

def bytes_to_bits(data: bytes) -> np.ndarray:
    """Преобразование байтов в массив битов."""
    bits = []
    for byte in data:
        bits.extend([int(b) for b in f"{byte:08b}"])
    return np.array(bits, dtype=int)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Преобразование массива битов в байты."""
    L = (len(bits) // 8) * 8
    b = bits[:L].reshape(-1, 8)
    return bytes(int(''.join(str(bit) for bit in row), 2) for row in b)

# -----------------------
# Interleaving (Bit Permutation)
# -----------------------
def interleave_bits(bits, block_size=96):
    """
    Перемежение (интерливинг) битов для повышения устойчивости к пачкам ошибок.
    Блочный интерливинг: биты переставляются внутри блока размера block_size.
    Записываем в матрицу по строкам, читаем по столбцам.
    """
    if len(bits) == 0:
        return bits
    
    # Работаем с блоками по block_size бит
    result = []
    for start in range(0, len(bits), block_size):
        block = bits[start:start+block_size]
        if len(block) < block_size:
            # Дополняем последний блок нулями
            block = np.concatenate([block, np.zeros(block_size - len(block), dtype=int)])
        
        # Записываем блок в матрицу по строкам (row-major)
        n = len(block)
        cols = 8  # Фиксированное количество столбцов
        rows = (n + cols - 1) // cols
        
        matrix = np.zeros((rows, cols), dtype=int)
        # Заполняем по строкам
        for i in range(n):
            matrix[i // cols, i % cols] = block[i]
        
        # Читаем по столбцам (column-major) - это и есть интерливинг
        interleaved_block = []
        for j in range(cols):
            for i in range(rows):
                idx = j * rows + i
                if idx < n:
                    interleaved_block.append(matrix[i, j])
        
        result.extend(interleaved_block)
    
    return np.array(result, dtype=int)

def deinterleave_bits(bits, block_size=96):
    """
    Обратное перемежение (деинтерливинг) битов.
    Восстанавливает исходный порядок битов после интерливинга.
    Обратная операция к interleave_bits.
    """
    if len(bits) == 0:
        return bits
    
    result = []
    for start in range(0, len(bits), block_size):
        block = bits[start:start+block_size]
        if len(block) < block_size:
            block = np.concatenate([block, np.zeros(block_size - len(block), dtype=int)])
        
        # Восстанавливаем матрицу из блока (как читали при интерливинге)
        n = len(block)
        cols = 8
        rows = (n + cols - 1) // cols
        
        matrix = np.zeros((rows, cols), dtype=int)
        # Заполняем по столбцам (как читали при интерливинге)
        idx = 0
        for j in range(cols):
            for i in range(rows):
                if idx < n:
                    matrix[i, j] = block[idx]
                    idx += 1
        
        # Читаем по строкам (как записывали при интерливинге)
        deinterleaved_block = []
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < n:
                    deinterleaved_block.append(matrix[i, j])
        
        result.extend(deinterleaved_block)
    
    return np.array(result, dtype=int)

# -----------------------
# QPSK map/demap
# -----------------------
def qpsk_map(bits):
    """Маппинг битов в QPSK символы."""
    M = {
    '00': (1 + 1j) / np.sqrt(2),
    '01': (-1 + 1j) / np.sqrt(2),
    '11': (-1 - 1j) / np.sqrt(2),
    '10': (1 - 1j) / np.sqrt(2),
    }
    if len(bits) % 2:
        bits = np.append(bits, 0)
    syms = [M[f"{bits[i]}{bits[i+1]}"] for i in range(0, len(bits), 2)]
    return np.array(syms)

def qpsk_demap(syms):
    """Демаппинг QPSK символов в биты."""
    bits = []
    for s in syms:
        b0 = 0 if s.real > 0 else 1
        b1 = 0 if s.imag > 0 else 1
        bits += [b1, b0]
    return np.array(bits, dtype=int)

# -----------------------
# BPSK map/demap
# -----------------------
def bpsk_map(bits):
    """Map bits to BPSK symbols: 0->1, 1->-1 (BPSK: 1 bit per symbol)"""
    syms = []
    for b in bits:
        syms.append(1.0 if b == 0 else -1.0)
    return np.array(syms, dtype=complex)

def bpsk_demap(syms):
    """Demap BPSK symbols to bits."""
    bits = []
    for s in syms:
        bits.append(0 if s.real > 0 else 1)
    return np.array(bits, dtype=int)

# -----------------------
# crest utilities
# -----------------------
def crest_factor(sig):
    """Расчет пик-фактора сигнала в дБ."""
    peak = np.max(np.abs(sig))
    rms = np.sqrt(np.mean(sig**2))
    if rms == 0:
        return np.inf
    return 20 * np.log10(peak / rms)

# -----------------------
# Active Constellation Extension (ACE) – simplified implementation
# -----------------------
def ace_reduce_peaks(ds, Nfft_local, subc_inds_local):
    """Снижение пиков сигнала методом ACE."""
    try:
        if ds is None or len(ds) == 0:
            return ds
        X = np.zeros(Nfft_local, dtype=complex)
        X[subc_inds_local] = ds
        X[-subc_inds_local] = np.conj(ds)
        for it in range(ACE_MAX_ITERS):
            x_td = np.fft.ifft(X)
            x_td_real = np.real(x_td)
            mag = np.abs(x_td_real)
            mean_mag = np.mean(mag) if mag.size > 0 else 0.0
            if mean_mag <= 0:
                break
            thresh = ACE_PEAK_THRESHOLD * mean_mag
            peak_idx = np.where(mag > thresh)[0]
            if peak_idx.size == 0:
                break
            corr_td = np.zeros_like(x_td, dtype=complex)
            exceed = mag[peak_idx] - thresh
            corr_td[peak_idx] = - (exceed / (mag[peak_idx] + 1e-12)) * x_td[peak_idx] * ACE_STEP
            Corr_fd = np.fft.fft(corr_td)
            Corr_proj = np.zeros_like(Corr_fd)
            Corr_proj[subc_inds_local] = Corr_fd[subc_inds_local]
            Corr_proj[-subc_inds_local] = Corr_fd[-subc_inds_local]
            X_new = X + Corr_proj
            ds_cand = X_new[subc_inds_local].copy()
            if ACE_ALLOW_EXPANSION:
                phases = np.angle(ds)
                mags_old = np.abs(ds)
                mags_cand = np.abs(ds_cand)
                mags_new = np.maximum(mags_old, mags_cand)
                mags_new = np.minimum(mags_new, mags_old * (1.0 + ACE_STEP))
                ds = mags_new * np.exp(1j * phases)
                X = np.zeros(Nfft_local, dtype=complex)
                X[subc_inds_local] = ds
                X[-subc_inds_local] = np.conj(ds)
            else:
                ds = ds_cand
                X = X_new
        return ds
    except Exception:
        return ds

# -----------------------
# OFDM symbol build
# -----------------------
def ofdm_symbol(data_syms):
    """Сборка OFDM символа из поднесущих."""
    if len(data_syms) < Nsub:
        ds = np.concatenate((data_syms, np.zeros(Nsub - len(data_syms), dtype=complex)))
    else:
        ds = np.array(data_syms[:Nsub], dtype=complex)

    try:
        eps = 1e-12
        cur_rms = np.sqrt(np.mean(np.abs(ds)**2)) if ds.size > 0 else 0.0
        if cur_rms < eps:
            cur_rms = eps
        ds = ds / cur_rms * SYMBOL_TX_TARGET
    except Exception:
        pass

    ds = ace_reduce_peaks(ds, Nfft, subc_inds)

    if modem_config.subc_phases is not None and np.any(modem_config.subc_phases != 0):
        ds = ds * np.exp(1j * modem_config.subc_phases)

    X = np.zeros(Nfft, dtype=complex)
    X[subc_inds] = ds
    X[-subc_inds] = np.conj(ds)
    X[0] = X[0].real
    if Nfft % 2 == 0:
        X[Nfft//2] = X[Nfft//2].real
    x = np.fft.ifft(X)
    x = np.real(x)
    return np.concatenate((x[-Ncp:], x))

# -----------------------
# Zadoff-Chu generator (frequency-domain placement)
# -----------------------
def zc_root_sequence(u: int, L: int):
    """Генерация последовательности Zadoff-Chu."""
    n = np.arange(L)
    z = np.exp(-1j * np.pi * u * n * (n + 1) / float(L))
    z = z / np.sqrt(np.mean(np.abs(z)**2))
    return z

# -----------------------
# build_preamble: ZC, ZC, pilot, pilot
# -----------------------
def build_preamble(reps=1, zc_root=1):
    """Сборка преамбулы из ZC и пилот-символов."""
    zc_seq = zc_root_sequence(zc_root, Nsub)
    S_zc = ofdm_symbol(zc_seq)
    pilot = np.full(Nsub, (1 + 1j) / np.sqrt(2))
    S_pilot = ofdm_symbol(pilot)
    preamble = np.concatenate((S_zc, S_zc, S_pilot, S_pilot))
    
    # Нормализуем преамбулу к SYMBOL_TX_TARGET (как и ofdm_symbol())
    eps = 1e-12
    cur_rms = np.sqrt(np.mean(np.abs(preamble)**2))
    if cur_rms < eps:
        cur_rms = eps
    preamble = preamble / cur_rms * SYMBOL_TX_TARGET
    return preamble

def build_data_td(bits):
    """
    Сборка модулированных данных во временную область.
    
    Алгоритм:
    1. Преобразование битов в символы модуляции (BPSK или QPSK)
    2. Дополнение нулями до кратности Nsub (количество поднесущих)
    3. Разбиение на блоки по Nsub символов (каждый блок = один OFDM символ)
    4. Преобразование каждого блока в OFDM символ во временной области
    
    Особенности для BPSK:
    - BPSK48: 48 поднесущих × 1 бит = 48 бит/OFDM символ
    - Для формирования одного RS слова (96 бит) требуется 2 OFDM символа BPSK
    - BPSK96 (будущая поддержка): 96 поднесущих × 1 бит = 96 бит/OFDM символ
    
    Возвращает:
    - td: сигнал во временной области (все OFDM символы подряд)
    - nblocks: количество OFDM символов
    """
    if modem_config.MODULATION == "BPSK":
        syms = bpsk_map(bits)
    else:
        syms = qpsk_map(bits)
    pad = (-len(syms)) % Nsub
    if pad:
        syms = np.concatenate((syms, np.zeros(pad, dtype=complex)))
    blk = syms.reshape(-1, Nsub)
    td = [ofdm_symbol(b) for b in blk]
    return np.concatenate(td), blk.shape[0]

# -----------------------
# Habr optimizer (unchanged)
# -----------------------
def make_training_blocks(n_blocks=HABR_SAMPLE_BLOCKS, seed=HABR_SEED):
    """Создание блоков для тренировки фаз."""
    rng = np.random.RandomState(seed)
    blocks = []
    for _ in range(n_blocks):
        bits = rng.randint(0, 2, Nsub * 2)
        if modem_config.MODULATION == "BPSK":
            syms = bpsk_map(bits[:Nsub])
        else:
            syms = qpsk_map(bits)
        blocks.append(syms)
    return np.array(blocks)

def optimize_phases_habr(sample_blocks=None, n_iter=HABR_MAX_ITERS, grid_size=HABR_PHASE_GRID, seed=HABR_SEED, verbose=True):
    """Оптимизация фаз методом Habr."""
    if sample_blocks is None:
        sample_blocks = make_training_blocks()
    M = sample_blocks.shape[0]
    phases = make_subcarrier_phases("schroeder", Nsub)

    def compute_total_td(phs):
        old = modem_config.subc_phases
        modem_config.subc_phases = phs
        td_list = [ofdm_symbol(sample_blocks[m]) for m in range(M)]
        td = np.concatenate(td_list)
        modem_config.subc_phases = old
        return td

    best_td = compute_total_td(phases)
    best_crest = crest_factor(best_td)
    if verbose:
        print(f"[HABR] init crest = {best_crest:.3f} dB (Schroeder start)")
    grid = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
    for it in range(n_iter):
        if verbose:
            print(f"[HABR] Iteration {it+1}/{n_iter}")
        improved = False
        for k_idx in range(Nsub):
            cur_ph = phases[k_idx]
            best_local_phase = cur_ph
            best_local_crest = best_crest
            for phi in grid:
                cand = phases.copy()
                cand[k_idx] = phi
                td_cand = compute_total_td(cand)
                c = crest_factor(td_cand)
                if c < best_local_crest:
                    best_local_crest = c
                    best_local_phase = phi
            if best_local_phase != cur_ph:
                phases[k_idx] = best_local_phase
                best_crest = best_local_crest
                improved = True
        if not improved:
            if verbose:
                print("[HABR] no improvement in iteration, stopping early")
            break
    if verbose:
        print(f"[HABR] final crest = {best_crest:.3f} dB")
    return phases % (2*np.pi), best_crest

def init_phases():
    """Инициализация фаз поднесущих (Habr, Schroeder или случайные)."""
    if PHASE_METHOD == "habr":
        if os.path.isfile(HABR_SAVE_FILE):
            try:
                modem_config.subc_phases = np.load(HABR_SAVE_FILE)
                print(f"[HABR] loaded phases from {HABR_SAVE_FILE}")
                return
            except Exception as e:
                print("[HABR] failed load, will optimize:", e)
        train_blocks = make_training_blocks(n_blocks=HABR_SAMPLE_BLOCKS, seed=HABR_SEED)
        ph, c = optimize_phases_habr(sample_blocks=train_blocks,
            n_iter=HABR_MAX_ITERS,
            grid_size=HABR_PHASE_GRID,
            seed=HABR_SEED,
            verbose=True)
        modem_config.subc_phases = ph
        try:
            np.save(HABR_SAVE_FILE, modem_config.subc_phases)
            print(f"[HABR] saved optimized phases to {HABR_SAVE_FILE} (crest={c:.3f} dB)")
        except Exception as e:
            print("[HABR] failed save:", e)
    else:
        modem_config.subc_phases = make_subcarrier_phases(PHASE_METHOD, Nsub, seed=HABR_SEED)
        print(f"[PHASE] method={PHASE_METHOD}, example degs[:8]={np.degrees(modem_config.subc_phases[:8])}")

# -----------------------
# Вспомогательные функции для работы с блоками данных
# -----------------------
def bytes_to_ofdm_blocks_bytes(bstream: bytes):
    """Разбивка потока байт на блоки OFDM с RS кодированием."""
    if len(bstream) == 0:
        return np.array([], dtype=float), 0
    blocks_local = [bstream[i:i+RS_DATA_BYTES] for i in range(0, len(bstream), RS_DATA_BYTES)]
    if len(blocks_local[-1]) < RS_DATA_BYTES:
        blocks_local[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks_local[-1]))
    encoded = [rs.encode(b) for b in blocks_local]
    bits_blocks_local = [bytes_to_bits(b) for b in encoded]

    data_bits_local = np.concatenate(bits_blocks_local) if len(bits_blocks_local) > 0 else np.array([], dtype=int)
    td_local, nblocks_local = build_data_td(data_bits_local) if data_bits_local.size > 0 else (np.array([], dtype=float), 0)
    return td_local, nblocks_local

# -----------------------
# Адаптивный эквалайзер (Decision-Directed)
# -----------------------
class AdaptiveEqualizer:
    """
    Класс для адаптивного выравнивания частотной характеристики канала.
    Использует подход Decision-Directed (DD) для уточнения оценки канала Hk
    по мере поступления символов данных.
    """
    
    def __init__(self, initial_Hk, alpha=0.05, modulation='QPSK', store_history=True, history_step=1):
        """
        Инициализация эквалайзера.
        
        :param initial_Hk: Начальная оценка канала (из преамбулы).
        :param alpha: Коэффициент сглаживания (0.0 - мгновенный отклик, 1.0 - игнор новых данных).
        :param modulation: Тип модуляции ('QPSK' или 'BPSK').
        :param store_history: Сохранять ли историю изменения Hk.
        :param history_step: Сохранять состояние каждые N символов (для экономии памяти).
        """
        self.Hk = np.array(initial_Hk, dtype=complex)
        self.alpha = alpha
        self.modulation = modulation
        self.symbol_count = 0
        self.store_history = store_history
        self.history_step = history_step
        self.history = []  # Список для хранения истории Hk
        
        # Сохраняем начальное состояние
        if self.store_history:
            self.history.append(self.Hk.copy())
        
    def process(self, rx_fd):
        """
        Обработка одного принятого OFDM символа (в частотной области).
        
        :param rx_fd: Комплексный массив поднесущих принятого символа (после FFT).
        :return: Выровненный символ (x_hat).
        """
        # 1. Применяем текущий Hk (предварительное выравнивание)
        # Добавляем малую константу для избежания деления на ноль
        x_hat = rx_fd / (self.Hk + 1e-12)
        
        # 2. Принимаем "решение" (Decision) - маппим обратно в идеальный символ
        if self.modulation == 'BPSK':
            # Для BPSK: real > 0 -> 1+0j, иначе -1+0j
            decision = np.where(np.real(x_hat) > 0, 1.0, -1.0) + 0j
        else:
            # Для QPSK: квадранты -> идеальные точки
            re = np.where(np.real(x_hat) > 0, 1.0, -1.0)
            im = np.where(np.imag(x_hat) > 0, 1.0, -1.0)
            decision = (re + 1j * im) / np.sqrt(2)
        
        # 3. Оценка нового канала на основе решения: H_new = Y / X_decision
        # Если decision близко к нулю, не обновляем (защита от шума)
        mask = np.abs(decision) > 0.1
        if np.any(mask):
            Hk_new = rx_fd[mask] / decision[mask]
            
            # 4. Экспоненциальное скользящее среднее (EMA) для сглаживания
            # Hk = (1 - alpha) * Hk_old + alpha * Hk_new
            # Для векторов разной длины используем усреднение по маске
            if len(Hk_new) > 0:
                # Обновляем только те поднесущие, где есть надежное решение
                self.Hk[mask] = (1.0 - self.alpha) * self.Hk[mask] + self.alpha * Hk_new
        
        self.symbol_count += 1
        
        # Сохраняем историю с заданным шагом
        if self.store_history and (self.symbol_count % self.history_step == 0 or self.symbol_count == 1):
            self.history.append(self.Hk.copy())
            if self.symbol_count % 100 == 0:
                print(f"[EQ-DEBUG] Сохранено {len(self.history)} состояний Hk (символ {self.symbol_count})")
        
        return x_hat
    
    def get_current_Hk(self):
        """Возвращает текущую оценку канала Hk."""
        return self.Hk.copy()
    
    def get_history(self):
        """
        Возвращает историю изменения Hk за все время работы.
        Каждый элемент списка — это массив Hk в определенный момент времени.
        """
        return self.history
    
    def get_debug_info(self):
        """Возвращает отладочную информацию: средние амплитуда и фаза."""
        avg_mag = np.mean(np.abs(self.Hk))
        avg_phase = np.mean(np.angle(self.Hk))
        return avg_mag, avg_phase, self.symbol_count