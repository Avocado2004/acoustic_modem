#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
import sounddevice as sd
from scipy.signal import firwin, filtfilt, correlate
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math

# -------------------------
# 1) Параметры системы
# -------------------------
fs               = 48000
Nfft             = 512
CP_LEN           = 64
f_low, f_high    = 300, 5000

df          = fs / Nfft
k_low       = int(math.ceil(f_low  / df))
k_high      = int(math.floor(f_high / df))
subc_inds   = np.arange(k_low, k_high + 1)
subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
subc_data   = np.setdiff1d(subc_inds, subc_pilots)

# -------------------------
# 2) Фильтр и AGC
# -------------------------
def design_bpf(f_low, f_high, fs, numtaps=129):
    taps = firwin(numtaps, cutoff=[f_low, f_high], fs=fs, pass_zero=False)
    return taps, [1.0]

b_bpf, a_bpf = design_bpf(f_low, f_high, fs)

def apply_agc(sig, target=0.9, eps=1e-12):
    peak = np.max(np.abs(sig)) + eps
    gain = target / peak
    return sig * gain

# -------------------------
# 3) Text ↔ Bits
# -------------------------
def text_to_bits(text, enc='utf-8'):
    bs = ''.join(f'{b:08b}' for b in text.encode(enc))
    return np.array(list(bs), int)

def bits_to_text(bits, enc='utf-8'):
    L   = (len(bits)//8)*8
    arr = bits[:L].reshape(-1, 8)
    data= bytes(int(''.join(str(x) for x in row), 2) for row in arr)
    return data.decode(enc, errors='ignore')

# -------------------------
# 4) QPSK map/demap
# -------------------------
def qpsk_map(bits):
    if len(bits) % 2:
        bits = np.append(bits, 0)
    re = 1 - 2*bits[0::2]
    im = 1 - 2*bits[1::2]
    return (re + 1j*im) / np.sqrt(2)

def qpsk_demap(symbols):
    bits = []
    for s in symbols:
        bits += [0 if s.real>0 else 1,
                 0 if s.imag>0 else 1]
    return np.array(bits, int)

# -------------------------
# 5) Zadoff–Chu preamble
# -------------------------
def generate_zc(N, u=1):
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n+1) / N)

# -------------------------
# 6) OFDM-модулятор
# -------------------------
def ofdm_mod(text):
    data_bits = text_to_bits(text)
    n_bytes   = len(data_bits)//8
    rem       = len(data_bits) % 8
    last_idx  = 3 if rem == 0 else (rem//2 - 1)

    header = np.array(list(f"{n_bytes:016b}") + list(f"{last_idx:02b}"), int)
    bits   = np.hstack((header, data_bits))
    pad    = (-len(bits)) % 8
    if pad:
        bits = np.hstack((bits, np.ones(pad, int)))

    syms = qpsk_map(bits)
    pad  = (-len(syms)) % len(subc_data)
    if pad:
        syms = np.concatenate((syms, np.zeros(pad, complex)))
    data_blk = syms.reshape(-1, len(subc_data))

    # ZC-преамбла
    Nzc = len(subc_inds)
    zc  = generate_zc(Nzc)
    Xz  = np.zeros(Nfft, complex)
    Xz[subc_inds]   = zc
    Xz[-subc_inds]  = np.conj(zc)
    pre_nocp = np.fft.ifft(Xz)
    pre_cp   = np.hstack((pre_nocp[-CP_LEN:], pre_nocp))

    # блоки данных
    data_blocks = []
    for blk in data_blk:
        X = np.zeros(Nfft, complex)
        X[subc_data]    = blk
        X[subc_pilots]  = 1+0j
        X[-subc_data]   = np.conj(blk)
        X[-subc_pilots] = np.conj(1+0j)
        td = np.fft.ifft(X)
        data_blocks.append(np.hstack((td[-CP_LEN:], td)))

    tx = np.concatenate((pre_cp, *data_blocks))
    return tx, pre_cp

# -------------------------
# 7) OFDM-демодулятор (SC coarse + ZC fine timing)
# -------------------------
def ofdm_dem(rx, pre_cp):
    """
    OFDM-демодулятор с двухэтапной синхронизацией:
      1) coarse timing — корреляция ZC-only с decimation
      2) fine timing   — корреляция ZC-only по ±D вокруг coarse
    В обоих этапах используем одну и ту же preamble—последовательность
    (с CP или без, по флагу use_cp_corr).
    Остальная логика не менялась.
    """
    # 1) BPF + AGC
    rx_f     = filtfilt(b_bpf, a_bpf, rx)
    rx_f     = apply_agc(rx_f)
    frame_len = Nfft + CP_LEN

    # 2) Подготовка preamble для корреляции
    # Если True — берем полную pre_cp (CP+symbol),
    # если False — только чистый OFDM-символ без CP.
    use_cp_corr = False
    if use_cp_corr:
        corr_seq = pre_cp
    else:
        corr_seq = pre_cp[CP_LEN:]
    corr_len = len(corr_seq)

    # 3) Coarse timing: decimated correlation ZC-only
    D = 8  # decimation factor
    max_lag = len(rx_f) - corr_len
    lags    = np.arange(0, max_lag, D)
    corr_vals = np.empty_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        segment = rx_f[lag : lag + corr_len]
        corr_vals[i] = np.abs(np.vdot(segment, corr_seq.conj()))

    idx_coarse = np.argmax(corr_vals)
    lag_coarse = lags[idx_coarse]
    corr_coarse = corr_vals[idx_coarse] / corr_len

    print(f"[SYNC] coarse ZC corr (decim D={D}) = {corr_coarse:.4f}")
    print(f"[SYNC] coarse lag = {lag_coarse}, offset = {lag_coarse % frame_len}")

    # 4) Fine timing: full-rate correlation ±D around coarse
    win_s = max(0, lag_coarse - D)
    win_e = min(lag_coarse + corr_len + D, len(rx_f))
    rx_win = rx_f[win_s:win_e]

    corr_fine = correlate(rx_win, corr_seq.conj(), mode='valid')
    idx_fine  = np.argmax(np.abs(corr_fine))
    corr_fval = np.abs(corr_fine[idx_fine]) / corr_len

    final_nocp = win_s + idx_fine
    start_data = final_nocp + corr_len
    offset_f   = start_data % frame_len

    print(f"[SYNC] fine ZC corr = {corr_fval:.4f}")
    print(f"[SYNC] fine idx = {idx_fine}, final offset = {offset_f}")

    # округляем start_data для срезов
    start_data_int = int(np.round(start_data))

    # 5) Оценка канала по preamble (без CP)
    #    используем ту же corr_seq, если use_cp_corr=False, то это и есть чистый символ
    pre_start = final_nocp
    pre_block = rx_f[pre_start : pre_start + corr_len]
    H_full    = np.fft.fft(pre_block, n=Nfft)
    H_est     = H_full[subc_data]

    # 6) Извлечение и эквализация данных
    rx_blocks = rx_f[start_data_int:]
    n_blk     = len(rx_blocks) // frame_len
    recv_syms = []

    for i in range(n_blk):
        blk     = rx_blocks[i*frame_len:(i+1)*frame_len]
        no_cp   = blk[CP_LEN:]
        X       = np.fft.fft(no_cp, n=Nfft)

        data_raw = X[subc_data]
        data_eq  = data_raw / (H_est + 1e-12)

        pilot_rx = X[subc_pilots]
        phi_off  = np.angle(np.mean(pilot_rx * np.conj(1+0j)))
        data_eq *= np.exp(-1j * phi_off)

        recv_syms.append(data_eq)

    recv_syms = np.hstack(recv_syms) if recv_syms else np.array([], complex)

    # 7) Demap → text
    bits    = qpsk_demap(recv_syms)
    decoded = bits_to_text(bits)
    return decoded, recv_syms

# -------------------------
# 8) Тестовые функции и визуализация
# -------------------------
def test_cfo(rx_signal, preamble_cp, fs):
    corr = correlate(rx_signal, preamble_cp.conj(), mode='valid')
    idx  = np.argmax(np.abs(corr))
    peak = corr[idx]
    T_pre = len(preamble_cp) / fs
    delta_f = np.angle(peak) / (2 * np.pi * T_pre)
    print(f"[CFO] idx={idx}, Δf={delta_f:.2f} Hz")
    t      = np.arange(len(rx_signal)) / fs
    rx_corr = rx_signal * np.exp(-1j * 2 * np.pi * delta_f * t)
    return rx_corr, delta_f, idx

def test_timing(idx, pre_len, Nfft, CP_LEN):
    start_data = idx + pre_len
    offset     = start_data % (Nfft + CP_LEN)
    print(f"[TIMING] start_data={start_data}, offset={offset}")
    if offset != CP_LEN:
        print("  → timing misaligned!")
    else:
        print("  → timing OK")

def test_preprocessing(raw, filt, agc, pre_cp):
    def peak(x):
        return np.max(np.abs(correlate(x, pre_cp, mode='valid')))
    peaks = {'raw': peak(raw), 'filt': peak(filt), 'agc': peak(agc)}
    print("[PRE] peaks:", peaks)
    plt.bar(peaks.keys(), peaks.values())
    plt.title("Preprocessing effect")
    plt.show()

def test_equalization(rx_corr, idx, pre_cp):
    start_data = idx + len(pre_cp)
    blk = rx_corr[int(start_data):int(start_data)+CP_LEN+Nfft]
    no_cp = blk[CP_LEN:]
    Xd    = np.fft.fft(no_cp, n=Nfft)

    Hp = Xd[subc_pilots]
    H_interp = (np.interp(subc_data, subc_pilots, Hp.real)
                + 1j * np.interp(subc_data, subc_pilots, Hp.imag))

    X_before = Xd[subc_data]
    X_after  = X_before / H_interp

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(X_before.real, X_before.imag, s=5)
    plt.title("Before EQ"); plt.axis('equal')
    plt.subplot(1,2,2)
    plt.scatter(X_after.real, X_after.imag, s=5)
    plt.title("After EQ"); plt.axis('equal')
    plt.show()

def plot_constellation(syms, title):
    plt.figure(figsize=(4,4))
    plt.scatter(syms.real, syms.imag, s=20, alpha=0.6)
    plt.grid(True); plt.axis('equal'); plt.title(title)
    plt.show()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    txt  = input("Enter text: ")
    mode = input("Mode (test/file/mic): ").strip().lower()

    if mode == "test":
        bits = text_to_bits(txt)
        syms = qpsk_map(bits)
        print(bits_to_text(qpsk_demap(syms)))
        sys.exit(0)

    tx, pre_cp = ofdm_mod(txt)

    # вставляем случайную тишину 0–5 с в начало
    silence_len = random.randint(0, fs * 5)
    silence     = np.zeros(silence_len, dtype=complex)
    tx          = np.concatenate((silence, tx))

    tx = apply_agc(tx)
    wavfile.write("ofdm_tx.wav", fs,
                  (tx.real * np.iinfo(np.int16).max).astype(np.int16))
    print(f"Saved ofdm_tx.wav with {silence_len} samples of leading silence")

    if mode == "file":
        _, wavd = wavfile.read("ofdm_tx.wav")
        sig     = wavd[:,0] if wavd.ndim > 1 else wavd
        rx_raw  = sig.astype(float) / np.iinfo(wavd.dtype).max
    else:
        rec    = sd.rec(len(tx), fs, 1, 'float32'); sd.wait()
        rx_raw = rec[:,0] / np.max(np.abs(rec))

    rx_filt = filtfilt(b_bpf, a_bpf, rx_raw)
    rx_agc  = apply_agc(rx_filt)

    rx_corr, delta_f, idx = test_cfo(rx_agc if mode!="file" else rx_raw, pre_cp, fs)
    test_timing(idx, len(pre_cp), Nfft, CP_LEN)
    test_preprocessing(rx_raw, rx_filt, rx_agc, pre_cp)
    test_equalization(rx_corr, idx, pre_cp)

    decoded, rx_syms = ofdm_dem(rx_raw, pre_cp)
    print("Decoded text:", decoded)
    plot_constellation(rx_syms, "RX Constellation")

