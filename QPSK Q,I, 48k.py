#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, correlate
from scipy.io import wavfile
import matplotlib.pyplot as plt

# -----------------------------------
# 1. SYSTEM PARAMETERS
# -----------------------------------
fs       = 48000            # sampling rate, Hz
Nfft     = 512              # FFT size
CP_LEN   = 64               # cyclic prefix length

f_low    = 300              # lower edge, Hz
f_high   = 5000             # upper edge, Hz

# compute OFDM subcarrier indices and frequencies
df        = fs / Nfft
k_low     = int(math.ceil(f_low  / df))
k_high    = int(math.floor(f_high / df))
subc_inds = np.arange(k_low, k_high + 1)
subc_freqs= subc_inds * df

print("OFDM subcarriers (k → f):")
for k, f in zip(subc_inds, subc_freqs):
    print(f"  k={k:3d} → {f:7.1f} Hz")
print()

subc_pilots     = np.hstack((subc_inds[:2], subc_inds[-2:]))
subc_data       = np.setdiff1d(subc_inds, subc_pilots)
pilot_sym       = 1+0j
pilot_sym_end   = 2+2j
N_END_PILOT_BLK = 2

AGC_TARGET = 0.9
AGC_EPS    = 1e-12

# -----------------------------------
# 2. 8th-order Butterworth BPF as SOS
# -----------------------------------
wn_low     = f_low  / (fs/2)
wn_high    = f_high / (fs/2)
sos_bp     = butter(8, [wn_low, wn_high], btype='bandpass', output='sos')

def apply_agc(x, target=AGC_TARGET, eps=AGC_EPS):
    peak = np.max(np.abs(x))
    if peak < eps:
        return x
    return x * (target/peak)

# -----------------------------------
# 3. HELPERS: spectrum & constellation
# -----------------------------------
def plot_spectrum(x, fs, title):
    N     = len(x)
    X     = np.fft.fft(x, n=N)
    freqs = np.fft.fftfreq(N, d=1/fs)
    mask  = freqs >= 0
    freqs = freqs[mask]
    mag   = 20 * np.log10(np.abs(X[mask]) + 1e-12)

    plt.figure(figsize=(6,3))
    plt.plot(freqs, mag, color='navy')
    plt.xlim(0, fs/2)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.tight_layout()
    plt.show()

def plot_constellation(syms, title):
    plt.figure(figsize=(4,4))
    plt.scatter(syms.real, syms.imag, s=20, alpha=0.6, color='navy')
    plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
    plt.grid(True); plt.axis('equal')
    plt.title(title)
    plt.show()

# -----------------------------------
# 4. TEXT ↔ BITS
# -----------------------------------
def text_to_bits(text, enc='utf-8'):
    bs = ''.join(f'{b:08b}' for b in text.encode(enc))
    return np.array(list(bs), int)

def bits_to_text(bits, enc='utf-8'):
    L   = (len(bits)//8)*8
    arr = bits[:L].reshape(-1, 8)
    data=bytes(int(''.join(str(x) for x in row),2) for row in arr)
    return data.decode(enc, errors='ignore')

# -----------------------------------
# 5. QPSK MAP/DEMAP
# -----------------------------------
def qpsk_map(bits):
    if len(bits) % 2:
        bits = np.append(bits, 0)
    re = 1 - 2*bits[0::2]
    im = 1 - 2*bits[1::2]
    return (re + 1j*im) / math.sqrt(2)

def qpsk_demap(symbols):
    bits = []
    for s in symbols:
        bits += [0 if s.real>0 else 1,
                 0 if s.imag>0 else 1]
    return np.array(bits, int)

# -----------------------------------
# 6. Zadoff–Chu PREAMBLE
# -----------------------------------
def generate_zc(N, u=1):
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n+1) / N)

# -----------------------------------
# 7. OFDM Modulator
# -----------------------------------
def ofdm_mod(text):
    data_bits = text_to_bits(text)
    n_bytes   = len(data_bits)//8
    rem       = len(data_bits) % 8
    last_idx  = 3 if rem==0 else (rem//2 -1)

    header = np.array(
        list(f"{n_bytes:016b}") + list(f"{last_idx:02b}"),
        int
    )
    bits = np.hstack((header, data_bits))
    pad  = (-len(bits)) % 8
    if pad:
        bits = np.hstack((bits, np.ones(pad,int)))

    syms = qpsk_map(bits)
    pad2 = (-len(syms)) % len(subc_data)
    if pad2:
        syms = np.concatenate((syms, np.zeros(pad2, complex)))
    blocks = syms.reshape(-1, len(subc_data))

    # preamble
    Nzc = len(subc_inds)
    zc  = generate_zc(Nzc)
    Xz  = np.zeros(Nfft, complex)
    Xz[subc_inds]   = zc
    Xz[-subc_inds]  = np.conj(zc)
    pre_nocp = np.fft.ifft(Xz)
    pre_cp   = np.hstack((pre_nocp[-CP_LEN:], pre_nocp))

    # data blocks
    data_blocks = []
    for blk in blocks:
        X = np.zeros(Nfft, complex)
        X[subc_data]    = blk
        X[subc_pilots]  = pilot_sym
        X[-subc_data]   = np.conj(blk)
        X[-subc_pilots] = np.conj(pilot_sym)
        td = np.fft.ifft(X)
        data_blocks.append(np.hstack((td[-CP_LEN:], td)))

    # end-pilots
    end_blocks = []
    for _ in range(N_END_PILOT_BLK):
        X = np.zeros(Nfft, complex)
        X[subc_pilots]   = pilot_sym_end
        X[-subc_pilots]  = np.conj(pilot_sym_end)
        td = np.fft.ifft(X)
        end_blocks.append(np.hstack((td[-CP_LEN:], td)))

    tx = np.concatenate((pre_cp, *data_blocks, *end_blocks))
    return tx, pre_cp

# -----------------------------------
# 8. OFDM Demodulator
# -----------------------------------
def ofdm_dem(rx, pre_cp):
    corr = correlate(rx, pre_cp, mode='valid')
    idx  = np.argmax(np.abs(corr))

    phi     = np.angle(corr[idx])
    Delta_f = phi / (2 * np.pi * (len(pre_cp)/fs))
    t       = np.arange(len(rx)) / fs
    rx_cfo  = rx * np.exp(-1j * 2 * np.pi * Delta_f * t)

    start     = idx + len(pre_cp)
    blk_len   = Nfft + CP_LEN
    first_blk = int(math.ceil(start/blk_len))

    rec_syms = []
    for i in range(first_blk, len(rx_cfo)//blk_len):
        block = rx_cfo[i*blk_len:(i+1)*blk_len]
        no_cp  = block[CP_LEN:]
        X      = np.fft.fft(no_cp)
        if np.allclose(X[subc_pilots], pilot_sym_end, atol=0.1):
            break
        H_p     = X[subc_pilots] / pilot_sym
        Hr      = np.interp(subc_data, subc_pilots, H_p.real)
        Hi      = np.interp(subc_data, subc_pilots, H_p.imag)
        H_i     = Hr + 1j*Hi
        # protect against zero
        H_i[np.abs(H_i) < AGC_EPS] = AGC_EPS
        phi_blk = np.angle(np.mean(H_p))
        Xd      = X[subc_data] / H_i * np.exp(-1j * phi_blk)
        rec_syms.append(Xd)

    syms = np.hstack(rec_syms) if rec_syms else np.array([], complex)
    bits = qpsk_demap(syms)
    if len(bits) < 18:
        return "", syms

    hdr_bits   = bits[:18]
    n_bytes    = int("".join(map(str, hdr_bits[:16])), 2)
    last_idx   = int("".join(map(str, hdr_bits[16:])), 2)
    total_bits = n_bytes*8 if last_idx==3 else (n_bytes-1)*8 + (last_idx+1)*2
    data_bits  = bits[18:18+total_bits]
    return bits_to_text(data_bits), syms

# -----------------------------------
# 9. MAIN (file mode)
# -----------------------------------
if __name__ == "__main__":
    txt  = input("Enter text: ")
    mode = input("Mode (test/file): ").strip().lower()

    if mode == "test":
        bits = text_to_bits(txt)
        syms = qpsk_map(bits)
        print("Decoded (test):", bits_to_text(qpsk_demap(syms)))
        sys.exit(0)

    # TX: OFDM mod + spectra + constellation
    tx, pre_cp = ofdm_mod(txt)
    plot_spectrum(tx, fs, "Spectrum before filtering")
    plot_constellation(qpsk_map(text_to_bits(txt)), "TX Constellation")

    # 1) bandpass filter complex baseband via SOS-filtfilt
    tx_filt = sosfiltfilt(sos_bp, tx)

    plot_spectrum(tx_filt, fs, "Spectrum after filtering")

    # 2) AGC on complex baseband
    tx_filt = apply_agc(tx_filt)

    # 3) interleave I/Q into real stream
    I = tx_filt.real; Q = tx_filt.imag
    s = np.empty(2*len(I), float)
    s[0::2], s[1::2] = I, Q

    # 4) AGC on real stream
    s = apply_agc(s)

    # 5) save mono-WAV
    wavfile.write(
        "ofdm_iq.wav",
        fs,
        (s * np.iinfo(np.int16).max).astype(np.int16)
    )
    print("Saved ofdm_iq.wav (8th-order BPF, interleaved I/Q)")

    # RX: read, de-interleave, demodulate
    _, wavd = wavfile.read("ofdm_iq.wav")
    sig     = wavd.astype(float) / np.iinfo(wavd.dtype).max

    I_rx = sig[0::2]; Q_rx = sig[1::2]
    rx   = I_rx + 1j*Q_rx

    decoded, rx_syms = ofdm_dem(rx, pre_cp)
    plot_constellation(rx_syms, "RX Constellation")
    print("Decoded text:", decoded)

