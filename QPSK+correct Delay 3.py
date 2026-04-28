#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
import math
import numpy as np
from scipy.signal import firwin, filtfilt
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt

# -------------------------
# 0) Sampling frequency
# -------------------------
fs = 48000  # Гц

# -------------------------
# 1) OFDM parameters
# -------------------------
Nfft   = 512
CP_LEN = 64
f_low, f_high = 300, 5000
df     = fs / Nfft
k_low  = int(math.ceil(f_low / df))
k_high = int(math.floor(f_high / df))
subc_inds   = np.arange(k_low, k_high+1)
subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
subc_data   = np.setdiff1d(subc_inds, subc_pilots)
pilot_ref   = np.ones(len(subc_pilots), dtype=complex)
fc          = (f_low + f_high) / 2

# -------------------------
# 2) BPF + AGC
# -------------------------
def design_bpf(f1, f2, fs, numtaps=129):
    taps = firwin(numtaps, cutoff=[f1, f2], fs=fs, pass_zero=False)
    return taps, [1.0]

b_bpf, a_bpf = design_bpf(f_low, f_high, fs)

def apply_agc(sig, target=0.9, eps=1e-12):
    peak = np.max(np.abs(sig)) + eps
    return sig * (target / peak)

# -------------------------
# 3) Text ↔ Bits
# -------------------------
def text_to_bits(text, enc='utf-8'):
    bs = ''.join(f'{b:08b}' for b in text.encode(enc))
    return np.array(list(bs), int)

def bits_to_text(bits, enc='utf-8'):
    L   = (len(bits)//8)*8
    arr = bits[:L].reshape(-1,8)
    data = bytes(int(''.join(str(x) for x in row),2) for row in arr)
    return data.decode(enc, errors='ignore')

# -------------------------
# 4) QPSK map/demap
# -------------------------
def qpsk_map(bits):
    if len(bits)%2:
        bits = np.append(bits, 0)
    re = 1 - 2*bits[0::2]
    im = 1 - 2*bits[1::2]
    return (re + 1j*im)/math.sqrt(2)

def qpsk_demap(symbols):
    bits = []
    for s in symbols:
        bits += [0 if s.real>0 else 1,
                 0 if s.imag>0 else 1]
    return np.array(bits, int)

# -------------------------
# 5) Golay complementary preamble
# -------------------------
def generate_golay_seq(N):
    """
    Generate one Golay complementary sequence of length N = 2^m.
    Recursive construction:
      A0 = [1], B0 = [1]
      A_{k+1} = [A_k, B_k]
      B_{k+1} = [A_k, -B_k]
    Returns sequence A of length N.
    """
    m = int(math.log2(N))
    A = np.array([1], int)
    B = np.array([1], int)
    for _ in range(m):
        A, B = np.hstack((A, B)), np.hstack((A, -B))
    return A.astype(complex)

# -------------------------
# New) Schmidl–Cox coarse sync metric
# -------------------------
def schmidl_cox_metric(rx, CP_LEN, Nfft):
    """
    Compute Schmidl–Cox timing metric M[n].
    """
    L = len(rx) - Nfft - CP_LEN + 1
    P = np.empty(L, complex)
    R = np.empty(L)
    for n in range(L):
        a = rx[n : n + CP_LEN]
        b = rx[n + Nfft : n + Nfft + CP_LEN]
        P[n] = np.vdot(a, b.conj())
        R[n] = np.sum(np.abs(b)**2)
    M = np.abs(P)**2 / (R**2 + 1e-12)
    return M

# -------------------------
# 6) OFDM modulator
# -------------------------
def ofdm_mod(text):
    data_bits = text_to_bits(text)
    n_bytes   = len(data_bits)//8
    rem       = len(data_bits)%8
    last_idx  = 3 if rem==0 else (rem//2 - 1)
    header    = np.array(list(f"{n_bytes:016b}") +
                         list(f"{last_idx:02b}"), int)
    bits      = np.hstack((header, data_bits))
    pad       = (-len(bits)) % 8
    if pad:
        bits = np.hstack((bits, np.zeros(pad, int)))

    syms = qpsk_map(bits)
    pad  = (-len(syms)) % len(subc_data)
    if pad:
        syms = np.hstack((syms, np.zeros(pad, complex)))
    data_blk = syms.reshape(-1, len(subc_data))

    # Golay‐based preamble (frequency domain)
    G = generate_golay_seq(Nfft)      # length Nfft, ±1
    g_time = np.fft.ifft(G, n=Nfft)   # time‐domain Golay
    preamble = np.hstack((g_time[-CP_LEN:], g_time))

    # data symbols
    blocks = []
    for blk in data_blk:
        F = np.zeros(Nfft, complex)
        F[subc_data]   = blk
        F[subc_pilots] = pilot_ref
        t_sym = np.fft.ifft(F, n=Nfft)
        blocks.append(np.hstack((t_sym[-CP_LEN:], t_sym)))

    tx = np.concatenate((preamble, *blocks))
    return tx, preamble

# -------------------------
# 7) OFDM demodulator with Schmidl–Cox + Golay sync
# -------------------------
def ofdm_dem(rx, preamble):
    # 1) BPF + AGC
    rx_f = filtfilt(b_bpf, a_bpf, rx)
    rx_f = apply_agc(rx_f)

    # 2) Coarse sync – Schmidl–Cox on CP
    M = schmidl_cox_metric(rx_f, CP_LEN, Nfft)
    offset_coarse = int(np.argmax(M))
    print(f"[SYNC] Coarse offset (Schmidl–Cox) = {offset_coarse}, metric = {M[offset_coarse]:.4f}")

    # (diagnostic) full‐preamble correlation
    # pr = preamble[::-1].conj()
    # corr_pr = np.correlate(rx_f, pr, mode='valid')

    # 3) CFO estimation & correction
    a = rx_f[offset_coarse : offset_coarse + CP_LEN]
    b = rx_f[offset_coarse + Nfft : offset_coarse + Nfft + CP_LEN]
    phi      = np.angle(np.vdot(a, b))
    freq_off = phi / Nfft
    print(f"[SYNC] Estimated CFO = {freq_off:.4e} rad/sample")
    rx_f *= np.exp(-1j * freq_off * np.arange(len(rx_f)))

    # 4) Fine sync – Golay time‐domain cross‐corr
    g_nocp = preamble[CP_LEN:]
    g_rev  = g_nocp[::-1].conj()
    corr_g = np.correlate(rx_f, g_rev, mode='valid')
    offset_fine = int(np.argmax(np.abs(corr_g)))
    print(f"[SYNC] Fine offset (Golay) = {offset_fine}, corr = {np.abs(corr_g[offset_fine]):.4f}")

    # 5) Channel estimation from Golay preamble
    pre_blk = rx_f[offset_fine : offset_fine + Nfft]
    H_full  = np.fft.fft(pre_blk, n=Nfft)
    H_est   = H_full[subc_data]

    # 6) Data extraction
    start_data = offset_fine + Nfft - 1
    print(f"[DEBUG] Estimated start_data index = {start_data}")

    rx_blocks = rx_f[start_data:]
    frame_len = Nfft + CP_LEN
    n_blk = len(rx_blocks) // frame_len
    recv_syms = []
    for i in range(n_blk):
        blk = rx_blocks[i*frame_len:(i+1)*frame_len]
        X   = np.fft.fft(blk[CP_LEN:], n=Nfft)
        eq  = X[subc_data] / (H_est + 1e-12)
        prx = X[subc_pilots]
        phi_off = np.angle(np.vdot(prx, pilot_ref.conj()))
        print(f"[SYNC] Pilot phase offset sym {i}: {phi_off:+.4f}")
        recv_syms.append(eq * np.exp(-1j*phi_off))
    recv_syms = np.hstack(recv_syms) if recv_syms else np.array([], dtype=complex)
    bits    = qpsk_demap(recv_syms)
    decoded = bits_to_text(bits)
    return decoded, recv_syms, offset_coarse, start_data, M, corr_g

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

    tx, preamble = ofdm_mod(txt)
    silence_len = random.randint(0, fs*5)
    tx = np.concatenate((np.zeros(silence_len, complex), tx))
    print(f"[DEBUG] Actual preamble offset (silence) = {silence_len}")

    tx = apply_agc(tx)
    t = np.arange(len(tx)) / fs
    tx_audio = np.real(tx * np.exp(1j*2*np.pi*fc*t))
    tx_audio /= np.max(np.abs(tx_audio))
    wavfile.write("ofdm_tx.wav", fs,
                  (tx_audio * np.iinfo(np.int16).max).astype(np.int16))

    if mode == "file":
        _, wavd = wavfile.read("ofdm_tx.wav")
        sig = wavd[:,0] if wavd.ndim>1 else wavd
        rx_raw = sig.astype(float) / np.iinfo(wavd.dtype).max
        rx_raw = rx_raw.astype(np.complex128)
        t_rx = np.arange(len(rx_raw)) / fs
        rx_raw *= np.exp(-1j*2*np.pi*fc*t_rx)
    else:
        rec = sd.rec(len(tx), fs, 1, 'float32'); sd.wait()
        rx_audio = rec[:,0] / np.max(np.abs(rec))
        rx_raw = rx_audio

    decoded, rx_syms, off_coarse, start_data, M, corr_g = ofdm_dem(rx_raw, preamble)

    expected_data_start = silence_len + len(preamble)
    print(f"[DEBUG] Actual data start index = {expected_data_start}")
    print(f"[DEBUG] Length preamble (samples) = {len(preamble)}")
    diff = start_data - silence_len
    print(f"[DEBUG] Start offset diff = {diff} samples")
    print("Decoded text:", decoded)

    # plots
    plt.figure(figsize=(4,4))
    plt.scatter(rx_syms.real, rx_syms.imag, s=20, alpha=0.6)
    plt.title("RX Constellation")
    plt.grid(True)
    plt.axis('equal')

    plt.figure(figsize=(6,3))
    plt.plot(M)
    plt.title("Schmidl–Cox metric M(n)")
    plt.xlabel("Sample index")
    plt.ylabel("M(n)")
    plt.grid(True)

    plt.figure(figsize=(6,3))
    plt.plot(np.abs(corr_g))
    plt.title("Golay fine‐sync correlation")
    plt.xlabel("Sample index")
    plt.ylabel("|corr_g|")
    plt.grid(True)

    plt.show()

