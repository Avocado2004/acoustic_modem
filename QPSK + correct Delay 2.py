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
# 0) Sampling frequency
# -------------------------
fs = 48000  # частота дискретизации, Гц

# -------------------------
# 1) Параметры системы
# -------------------------
Nfft = 512
CP_LEN = 64
f_low, f_high = 300, 5000
df = fs / Nfft
k_low = int(math.ceil(f_low / df))
k_high = int(math.floor(f_high / df))
subc_inds = np.arange(k_low, k_high + 1)
subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
subc_data = np.setdiff1d(subc_inds, subc_pilots)
# эталон пилотов (все равны 1+0j)
pilot_ref = np.ones(len(subc_pilots), dtype=complex)

# -------------------------
# 2) Фильтрация и AGC
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
    L = (len(bits)//8)*8
    arr = bits[:L].reshape(-1,8)
    data = bytes(int(''.join(str(x) for x in row),2) for row in arr)
    return data.decode(enc, errors='ignore')

# -------------------------
# 4) QPSK map/demap
# -------------------------
def qpsk_map(bits):
    if len(bits)%2:
        bits = np.append(bits,0)
    re = 1 - 2*bits[0::2]
    im = 1 - 2*bits[1::2]
    return (re+1j*im)/np.sqrt(2)

def qpsk_demap(symbols):
    bits = []
    for s in symbols:
        bits += [
            0 if s.real>0 else 1,
            0 if s.imag>0 else 1
        ]
    return np.array(bits, int)

# -------------------------
# 5) Zadoff–Chu preamble
# -------------------------
def generate_zc(N, u=1):
    n = np.arange(N)
    return np.exp(-1j*np.pi*u*n*(n+1)/N)

# -------------------------
# 6) OFDM-модулятор
# -------------------------
def ofdm_mod(text):
    # Преобразование текста в биты и QPSK-символы
    data_bits = text_to_bits(text)
    # Заголовок: длина в байтах и остаток бит
    n_bytes = len(data_bits)//8
    rem = len(data_bits)%8
    last_idx = 3 if rem==0 else (rem//2 - 1)
    header = np.array(
        list(f"{n_bytes:016b}") + list(f"{last_idx:02b}"), int)
    bits = np.hstack((header, data_bits))
    # padding до кратности 8
    pad = (-len(bits))%8
    if pad:
        bits = np.hstack((bits, np.zeros(pad, int)))
    syms = qpsk_map(bits)
    pad = (-len(syms))%len(subc_data)
    if pad:
        syms = np.concatenate((syms, np.zeros(pad, complex)))
    data_blk = syms.reshape(-1, len(subc_data))

    # ZC-преамбула: генерируем один OFDM-символ (без CP), добавляем CP
    Nzc = len(subc_inds)
    zc_freq = generate_zc(Nzc)
    zc_time = np.fft.ifft(zc_freq, n=Nfft)
    pre_cp = np.hstack((zc_time[-CP_LEN:], zc_time))

    # блоки данных
    data_blocks = []
    for blk in data_blk:
        F = np.zeros(Nfft, complex)
        F[subc_data] = blk
        F[subc_pilots] = pilot_ref
        t_sym = np.fft.ifft(F, n=Nfft)
        t_cp = np.hstack((t_sym[-CP_LEN:], t_sym))
        data_blocks.append(t_cp)

    tx = np.concatenate((pre_cp, *data_blocks))
    return tx, pre_cp

# -------------------------
# 7) OFDM-демодулятор
# -------------------------
def ofdm_dem(rx, pre_cp):
    # 1) BPF + AGC
    rx_f = filtfilt(b_bpf, a_bpf, rx)
    rx_f = apply_agc(rx_f)

    frame_len = Nfft + CP_LEN

    # 2) Coarse timing — корреляция с полем preamble (с CP)
    corr_cp = correlate(rx_f, pre_cp.conj(), mode='valid')
    idx_cp = np.argmax(np.abs(corr_cp))
    corr_val_cp = np.abs(corr_cp[idx_cp]) / len(pre_cp)
    offset_coarse = idx_cp
    print(f"[SYNC] Coarse corr (with CP) = {corr_val_cp:.4f}")
    print(f"[SYNC] Coarse offset = {offset_coarse}")

    # 3) Оценка CFO по CP преамбулы
    if offset_coarse + CP_LEN + Nfft <= len(rx_f):
        cp1 = rx_f[offset_coarse : offset_coarse + CP_LEN]
        cp2 = rx_f[offset_coarse + Nfft : offset_coarse + Nfft + CP_LEN]
        phi = np.angle(np.vdot(cp1, cp2.conj()))
        freq_off = phi / CP_LEN  # rad/сэмпл
        print(f"[SYNC] Estimated CFO = {freq_off:.4e} rad/sample")
        n = np.arange(len(rx_f))
        rx_f = rx_f * np.exp(-1j * freq_off * n)
    else:
        print(" [SYNC] CFO estimation skipped (window overflow)")

    # 4) Fine timing — корреляция чистого символа (без CP)
    zc_nocp = pre_cp[CP_LEN:]
    corr_nocp = correlate(rx_f, zc_nocp.conj(), mode='valid')
    idx_nc = np.argmax(np.abs(corr_nocp))
    corr_val_nc = np.abs(corr_nocp[idx_nc]) / len(zc_nocp)
    offset_fine = idx_nc
    print(f"[SYNC] Fine corr (no CP) = {corr_val_nc:.4f}")
    print(f"[SYNC] Fine offset = {offset_fine}")

    # 5) Оценка канала по преамбуле (без CP)
    start_nocp = offset_fine
    pre_block = rx_f[start_nocp : start_nocp + len(zc_nocp)]
    H_full = np.fft.fft(pre_block, n=Nfft)
    H_est = H_full[subc_data]

    # 6) Извлечение данных и эквализация
    start_data = start_nocp + Nfft
    print(f"[SYNC] Data start offset = {start_data}")
    rx_blocks = rx_f[int(start_data):]
    n_blk = len(rx_blocks) // frame_len
    recv_syms = []
    for i in range(n_blk):
        blk = rx_blocks[i*frame_len:(i+1)*frame_len]
        no_cp = blk[CP_LEN:]
        X = np.fft.fft(no_cp, n=Nfft)
        # эквализация по оценке канала
        data_eq = X[subc_data] / (H_est + 1e-12)
        # фазовая поправка по пилотам
        pilot_rx = X[subc_pilots]
        phi_off = np.angle(np.vdot(pilot_rx, pilot_ref.conj()))
        data_eq *= np.exp(-1j * phi_off)
        print(f"[SYNC] Pilot phase offset sym {i}: {phi_off:+.4f}")
        recv_syms.append(data_eq)
    recv_syms = np.hstack(recv_syms) if recv_syms else np.array([], complex)

    # 7) Demap → text
    bits = qpsk_demap(recv_syms)
    decoded = bits_to_text(bits)
    return decoded, recv_syms, start_data

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    txt = input("Enter text: ")
    mode = input("Mode (test/file/mic): ").strip().lower()

    if mode == "test":
        bits = text_to_bits(txt)
        syms = qpsk_map(bits)
        print(bits_to_text(qpsk_demap(syms)))
        sys.exit(0)

    tx, pre_cp = ofdm_mod(txt)

    # добавляем случайную тишину перед фреймом
    silence_len = random.randint(0, fs * 5)
    silence = np.zeros(silence_len, dtype=complex)
    tx = np.concatenate((silence, tx))
    print(f"[DEBUG] Actual preamble offset = {silence_len}")

    # AGC
    tx = apply_agc(tx)

    # carrier modulation for file mode
    fc = (f_low + f_high) / 2
    t_audio = np.arange(len(tx)) / fs
    tx_audio = np.real(tx * np.exp(1j * 2 * np.pi * fc * t_audio))
    tx_audio /= np.max(np.abs(tx_audio))

    wavfile.write("ofdm_tx.wav", fs,
                  (tx_audio * np.iinfo(np.int16).max).astype(np.int16))

    if mode == "file":
        _, wavd = wavfile.read("ofdm_tx.wav")
        sig = wavd[:,0] if wavd.ndim>1 else wavd
        rx_audio = sig.astype(float) / np.iinfo(wavd.dtype).max

        # carrier demodulation
        t_rx = np.arange(len(rx_audio)) / fs
        rx_raw = rx_audio * np.exp(-1j * 2 * np.pi * fc * t_rx)

    else:
        rec = sd.rec(len(tx), fs, 1, 'float32'); sd.wait()
        rx_audio = rec[:,0] / np.max(np.abs(rec))
        rx_raw = rx_audio  # уже baseband

    decoded, rx_syms, start_data = ofdm_dem(rx_raw, pre_cp)

    # diagnostic difference
    diff = start_data - silence_len
    print(f"[DEBUG] Start offset diff = {diff} samples")

    print("Decoded text:", decoded)

    plt.figure(figsize=(4,4))
    plt.scatter(rx_syms.real, rx_syms.imag, s=20, alpha=0.6)
    plt.title("RX Constellation")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

