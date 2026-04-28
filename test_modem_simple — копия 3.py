#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-carrier QPSK-модем с LDPC-кодированием,
полосой 300–5000 Гц, 6-порядковым BPF и визуализацией спектра.

Поток данных:
  Unicode-строка → UTF-8 байты → биты → LDPC → QPSK → WAV → QPSK → LDPC → биты → UTF-8 → Unicode-строка
"""

import numpy as np
import wave
import struct
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ========== Параметры полосы и фильтра ==========
f_low   = 300      # Гц
f_high  = 10000     # Гц
band    = f_high - f_low
fs      = 10 * band           # сэмпл/с, 10×ширина полосы
order   = 6                   # порядок фильтра

# ========== Параметры LDPC ==========
# Код (n, k), регулярная матрица проверок с dv единицами на столбец
n_ldpc = 240
k_ldpc = 120
dv     = 3
seed   = 42
max_iter_decode = 50

# -----------------------------------------------------------------------------
# Вспомогательные функции для GF(2)
# -----------------------------------------------------------------------------
def gf2_row_echelon(A):
    """Ступенчатый вид над GF(2). Возвращает (A_reduced, pivots)."""
    A = (A.copy() & 1).astype(np.uint8)
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        pivot = None
        for rr in range(r, m):
            if A[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for rr in range(m):
            if rr != r and A[rr, c]:
                A[rr, :] ^= A[r, :]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A, pivots

def gf2_inv(A):
    """Обратная матрица над GF(2)."""
    A = (A.copy() & 1).astype(np.uint8)
    n = A.shape[0]
    I = np.eye(n, dtype=np.uint8)
    Aug = np.concatenate([A, I], axis=1)
    r = 0
    for c in range(n):
        pivot = None
        for rr in range(r, n):
            if Aug[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            Aug[[r, pivot]] = Aug[[pivot, r]]
        for rr in range(n):
            if rr != r and Aug[rr, c]:
                Aug[rr, :] ^= Aug[r, :]
        r += 1
        if r == n:
            break
    if not np.array_equal(Aug[:, :n], np.eye(n, dtype=np.uint8)):
        raise ValueError("Матрица не обратима над GF(2).")
    return Aug[:, n:]

def gf2_rank(A):
    """Ранг над GF(2)."""
    _, pivots = gf2_row_echelon(A)
    return len(pivots)

# -----------------------------------------------------------------------------
# Построение LDPC (H и систематическое кодирование)
# -----------------------------------------------------------------------------
def build_ldpc(n, k, dv=3, seed=0, max_tries=50):
    """Строит H (m x n) и перестановку столбцов P: H[:,P] = [H_p | H_v], H_p обратима."""
    rng = np.random.default_rng(seed)
    m = n - k
    for _ in range(max_tries):
        H = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            rows = rng.choice(m, size=dv, replace=False)
            H[rows, j] = 1
        if gf2_rank(H) < m:
            continue
        _, pivots = gf2_row_echelon(H)
        if len(pivots) < m:
            continue
        pivots = pivots[:m]
        non_pivots = [j for j in range(n) if j not in pivots]
        P = np.array(pivots + non_pivots, dtype=int)
        P_inv = np.empty_like(P); P_inv[P] = np.arange(n)
        H_perm = H[:, P]
        H_p = H_perm[:, :m]
        try:
            H_p_inv = gf2_inv(H_p)
        except ValueError:
            continue
        return H, P, P_inv, H_p_inv, m
    raise RuntimeError("Не удалось построить пригодную LDPC H. Увеличьте n или измените dv/seed.")

H, P, P_inv, H_p_inv, m_ldpc = build_ldpc(n_ldpc, k_ldpc, dv=dv, seed=seed)

def ldpc_encode_blocks(bit_string: str):
    """Биты → LDPC-код (систематический), возвращает (кодовые_биты, исходная_длина_без_паддинга)."""
    orig_len = len(bit_string)
    pad = (-orig_len) % k_ldpc
    bit_string += '0' * pad
    num_blk = len(bit_string) // k_ldpc

    H_perm = H[:, P]
    H_v = H_perm[:, m_ldpc:]  # (m x k)

    coded_bits = []
    for i in range(num_blk):
        v_bits = np.fromiter(bit_string[i*k_ldpc:(i+1)*k_ldpc], dtype=np.uint8) & 1
        s = (H_v @ v_bits) & 1  # синдромная часть
        s %= 2
        p = (H_p_inv @ s) & 1
        p %= 2
        cw_perm = np.concatenate([p, v_bits])  # [parity | info] в переставленном порядке
        cw = np.zeros(n_ldpc, dtype=np.uint8)
        cw[P] = cw_perm
        coded_bits.extend(cw.tolist())

    return ''.join('1' if b else '0' for b in coded_bits), orig_len

def ldpc_decode_blocks_hard(rx_bits: str, orig_bits_len: int, max_iter=max_iter_decode) -> str:
    """Итеративный bit-flipping. Возвращает исходные биты длиной orig_bits_len."""
    assert len(rx_bits) % n_ldpc == 0, "Длина принятых бит должна быть кратна n_ldpc"
    num_blk = len(rx_bits) // n_ldpc
    out_bits = []

    var_degrees = H.sum(axis=0).astype(int)
    thresholds = np.maximum(1, (var_degrees + 1) // 2)

    for i in range(num_blk):
        cw = np.fromiter(rx_bits[i*n_ldpc:(i+1)*n_ldpc], dtype=np.uint8) & 1
        for _ in range(max_iter):
            syndrome = (H @ cw) & 1
            syndrome %= 2
            if not syndrome.any():
                break
            unsat_counts = (H.T @ syndrome).astype(int)
            flip_mask = (unsat_counts >= thresholds).astype(np.uint8)
            if not flip_mask.any():
                cw[int(np.argmax(unsat_counts))] ^= 1
            else:
                cw ^= flip_mask
        cw_perm = cw[P]
        v_hat = cw_perm[m_ldpc:]
        out_bits.extend(v_hat.tolist())

    return ''.join('1' if b else '0' for b in out_bits)[:orig_bits_len]

# -----------------------------------------------------------------------------
# Сервис: фильтр и спектр
# -----------------------------------------------------------------------------
def design_bpf(fs, f1, f2, order=6):
    nyq = fs / 2
    b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
    return b, a

def plot_spectrum(signal: np.ndarray, fs: int, title: str = "Spectrum"):
    """Строит амплитудный спектр сигнала в dB."""
    Nfft = len(signal)
    windowed = signal * np.hanning(Nfft)
    fft_vals = np.fft.rfft(windowed)
    fft_freq = np.fft.rfftfreq(Nfft, d=1/fs)
    magnitude = np.abs(fft_vals) / Nfft
    plt.figure(figsize=(8, 4))
    plt.plot(fft_freq, 20*np.log10(magnitude + 1e-12), color='C0')
    plt.title(title)
    plt.xlabel("Частота, Гц")
    plt.ylabel("Уровень, dB")
    plt.xlim(0, fs/2)
    plt.ylim(-100, 0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Кодирование/декодирование текста (Unicode ↔ UTF-8)
# -----------------------------------------------------------------------------
def text_to_bits(s: str) -> str:
    data = s.encode('utf-8')
    return ''.join(f"{b:08b}" for b in data)

def bits_to_text(bstr: str) -> str:
    if len(bstr) % 8 != 0:
        bstr = bstr[:len(bstr) - (len(bstr) % 8)]
    data = bytes(int(bstr[i:i+8], 2) for i in range(0, len(bstr), 8))
    return data.decode('utf-8', errors='replace')

# -----------------------------------------------------------------------------
# Модуляция/демодуляция мульти-QPSK
# -----------------------------------------------------------------------------
def modulate_multi(bit_string: str, M: int) -> np.ndarray:
    pad = (-len(bit_string)) % (2*M)
    bit_string += '0'*pad

    Rs_sub   = band / M
    T_sym    = 1 / Rs_sub
    N        = int(fs * T_sym)
    dt       = 1/ fs
    t_sym    = np.arange(N) * dt

    fcs = [f_low + (i + 0.5)*Rs_sub for i in range(M)]
    map_b2iq = {
        "00": ( 1,  1),
        "01": (-1,  1),
        "11": (-1, -1),
        "10": ( 1, -1),
    }

    num_sym = len(bit_string) // (2*M)
    out = np.zeros(num_sym * N, dtype=np.float32)

    for n in range(num_sym):
        base = np.zeros(N, dtype=np.float32)
        for m, fc in enumerate(fcs):
            bits = bit_string[(n*2*M + 2*m):(n*2*M + 2*m + 2)]
            I, Q      = map_b2iq[bits]
            carrier_I = np.cos(2*np.pi*fc*t_sym)
            carrier_Q = np.sin(2*np.pi*fc*t_sym)
            base += I*carrier_I - Q*carrier_Q
        out[n*N:(n+1)*N] = base

    b, a = design_bpf(fs, f_low, f_high, order)
    return filtfilt(b, a, out)

def demodulate_multi(rx: np.ndarray, M: int) -> str:
    Rs_sub   = band / M
    T_sym    = 1 / Rs_sub
    N        = int(fs * T_sym)
    dt       = 1/ fs
    t_sym    = np.arange(N) * dt

    fcs = [f_low + (i + 0.5)*Rs_sub for i in range(M)]
    map_iq2b = {
        ( 1,  1): "00",
        (-1,  1): "01",
        (-1, -1): "11",
        ( 1, -1): "10",
    }

    num_sym = len(rx) // N
    bits_out = ""

    for n in range(num_sym):
        seg = rx[n*N:(n+1)*N]
        for fc in fcs:
            I_rec = np.dot(seg, np.cos(2*np.pi*fc*t_sym))
            Q_rec = -np.dot(seg, np.sin(2*np.pi*fc*t_sym))
            I_hat = 1 if I_rec >= 0 else -1
            Q_hat = 1 if Q_rec >= 0 else -1
            bits_out += map_iq2b[(I_hat, Q_hat)]

    return bits_out

# -----------------------------------------------------------------------------
# WAV I/O
# -----------------------------------------------------------------------------
def save_wave(fn: str, signal: np.ndarray):
    mx = np.max(np.abs(signal)) or 1.0
    sig16 = (signal/mx * 32767).astype(np.int16)
    with wave.open(fn, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(sig16.tobytes())

def load_wave(fn: str) -> np.ndarray:
    with wave.open(fn,'rb') as wf:
        n     = wf.getnframes()
        data  = wf.readframes(n)
        ints  = struct.unpack('<' + 'h'*n, data)
    return np.array(ints, dtype=np.float32)/32767.0

# -----------------------------------------------------------------------------
# Главная программа
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    WAV_FILE = "qpsk_ldpc.wav"

    try:
        M = int(input("Число поднесущих [целое >0, по умолчанию 50]: ") or "50")
        assert M > 0
    except:
        M = 50

    print(f"LDPC код: (n, k, rate) = ({n_ldpc}, {k_ldpc}, {k_ldpc/n_ldpc:.3f}), dv = {dv}, итераций декодера: {max_iter_decode}")

    text = input("Введите текст для передачи (Unicode): ")
    bits = text_to_bits(text)

    # LDPC кодирование
    bits_enc, orig_len = ldpc_encode_blocks(bits)
    print(f"Исходных бит: {len(bits)}, после LDPC: {len(bits_enc)}")

    # Модуляция и передача
    tx = modulate_multi(bits_enc, M)
    print(f"Сгенерировано сэмплов: {len(tx)}")

    save_wave(WAV_FILE, tx)
    print("Сигнал сохранён в", WAV_FILE)

    rx = load_wave(WAV_FILE)
    print(f"Прочитано сэмплов: {len(rx)}")

    # Визуализация спектра сигнала
    plot_spectrum(tx, fs, title="Спектр мульти-QPSK сигнала с LDPC")

    # Демодуляция
    rec_bits = demodulate_multi(rx, M)
    rec_bits = rec_bits[:len(bits_enc)]  # обрезаем к кратности n_ldpc

    # LDPC декодирование
    bits_dec = ldpc_decode_blocks_hard(rec_bits, orig_bits_len=orig_len)

    # Unicode вывод
    rec_text = bits_to_text(bits_dec)

    print("Восстановленный текст (Unicode):")
    print(rec_text)
    print("Статус:", "OK" if rec_text == text else "ERROR")

