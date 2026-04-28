#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import sounddevice as sd
from scipy.signal import firwin, filtfilt, correlate
from scipy.io import wavfile
import matplotlib.pyplot as plt

# -------------------------
# 0) Параметры системы
# -------------------------
fs         = 48000
Nfft       = 512
CP_LEN     = 64
AGC_TARGET = 0.9

f_low  = 300
f_high = 5000
df     = fs / Nfft

print(f"[INIT] fs={fs}, Nfft={Nfft}, CP_LEN={CP_LEN}, AGC_TARGET={AGC_TARGET}")
print(f"[INIT] f_low={f_low}, f_high={f_high}, df={df:.2f}")

k_low       = int(math.ceil(f_low/df))
k_high      = int(math.floor(f_high/df))
subc_inds   = np.arange(k_low, k_high+1)
subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
subc_data   = np.setdiff1d(subc_inds, subc_pilots)

print(f"[INIT] subc_inds   = {subc_inds}")
print(f"[INIT] subc_pilots = {subc_pilots}")
print(f"[INIT] subc_data   = {subc_data}")

pilot_sym       = 1+0j
pilot_sym_end   = 2+2j
N_END_PILOT_BLK = 2

# -------------------------
# 1) Фильтр BPF и AGC
# -------------------------
def design_bpf(f_low, f_high, fs, numtaps=129):
    print(f"[design_bpf] f_low={f_low}, f_high={f_high}, fs={fs}, taps={numtaps}")
    taps = firwin(numtaps, cutoff=[f_low, f_high], pass_zero=False, fs=fs)
    return taps, 1.0

b_bpf, a_bpf = design_bpf(f_low, f_high, fs)

def apply_agc(sig, target=AGC_TARGET, eps=1e-12):
    peak = np.max(np.abs(sig)) + eps
    gain = target / peak
    print(f"[apply_agc] peak={peak:.5f}, gain={gain:.5f}")
    return sig * gain

# -------------------------
# 2) text ↔ bits
# -------------------------
def text_to_bits(text, enc='utf-8'):
    bs   = ''.join(f"{b:08b}" for b in text.encode(enc))
    bits = np.array(list(bs), int)
    print(f"[text_to_bits] text_len={len(text)}, bits_len={len(bits)}")
    return bits

def bits_to_text(bits, enc='utf-8'):
    L   = (len(bits)//8)*8
    arr = bits[:L].reshape(-1,8)
    data = bytes(int(''.join(str(x) for x in row), 2) for row in arr)
    text = data.decode(enc, errors='ignore')
    print(f"[bits_to_text] bits_len={len(bits)}, decoded_text='{text}'")
    return text

# -------------------------
# 3) QPSK map / demap
# -------------------------
def qpsk_map(bits):
    if len(bits) % 2:
        bits = np.append(bits, 0)
        print("[qpsk_map] odd bit count, appended 0")
    b0    = bits[0::2]
    b1    = bits[1::2]
    re    = 1 - 2*b0
    im    = 1 - 2*b1
    syms  = (re + 1j*im) / math.sqrt(2)
    print(f"[qpsk_map] mapping {len(bits)} bits -> {len(syms)} symbols")
    return syms

def qpsk_demap(symbols):
    bits = []
    for i, s in enumerate(symbols):
        b_re = 0 if s.real>0 else 1
        b_im = 0 if s.imag>0 else 1
        bits += [b_re, b_im]
        print(f"[qpsk_demap] sym#{i}: {s:.3f} -> bits {b_re}{b_im}")
    print(f"[qpsk_demap] demapped {len(symbols)} symbols -> {len(bits)} bits")
    return np.array(bits, int)

# -------------------------
# 4) Zadoff–Chu preamble
# -------------------------
def generate_zc(N, u=1):
    print(f"[generate_zc] N={N}, u={u}")
    n = np.arange(N)
    return np.exp(-1j * math.pi * u * n * (n+1) / N)

# -------------------------
# 5) OFDM-модулятор
# -------------------------
def ofdm_mod(text):
    print(f"[ofdm_mod] start modulation for text='{text}'")
    data_bits = text_to_bits(text)
    n_bytes   = len(data_bits)//8
    rem       = len(data_bits) % 8
    last_idx  = 3 if rem==0 else (rem//2 - 1)
    print(f"[ofdm_mod] n_bytes={n_bytes}, rem={rem}, last_idx={last_idx}")

    hdr = np.hstack((
        np.array(list(f"{n_bytes:016b}"), int),
        np.array(list(f"{last_idx:02b}"),  int)
    ))
    print(f"[ofdm_mod] header bits: {''.join(map(str,hdr))}")

    bits     = np.hstack((hdr, data_bits))
    pad_bits = (-len(bits)) % 8
    bits     = np.hstack((bits, np.zeros(pad_bits, int)))
    print(f"[ofdm_mod] pad_bits={pad_bits}, total bits={len(bits)}")

    syms   = qpsk_map(bits)
    pad_s  = (-len(syms)) % len(subc_data)
    syms   = np.hstack((syms, np.zeros(pad_s, complex)))
    print(f"[ofdm_mod] pad_s={pad_s}, total syms={len(syms)}")

    data_blk = syms.reshape(-1, len(subc_data))
    print(f"[ofdm_mod] data blocks={data_blk.shape[0]}")

    # преамбула
    Nzc = len(subc_inds)
    zc  = generate_zc(Nzc)
    Xz  = np.zeros(Nfft, complex)
    Xz[subc_inds]  = zc
    Xz[-subc_inds] = np.conj(zc)
    pre_nocp = np.fft.ifft(Xz)
    pre_cp   = np.hstack((pre_nocp[-CP_LEN:], pre_nocp))
    print(f"[ofdm_mod] preamble length={len(pre_cp)}")

    # data blocks
    data_blocks = []
    for i, blk in enumerate(data_blk):
        X = np.zeros(Nfft, complex)
        X[subc_data]    = blk
        X[subc_pilots]  = pilot_sym
        X[-subc_data]   = np.conj(blk)
        X[-subc_pilots] = np.conj(pilot_sym)
        td = np.fft.ifft(X)
        data_blocks.append(np.hstack((td[-CP_LEN:], td)))
        print(f"[ofdm_mod] data block #{i} length={len(td)+CP_LEN}")

    # end-пилоты
    end_blocks = []
    for i in range(N_END_PILOT_BLK):
        X = np.zeros(Nfft, complex)
        X[subc_pilots]    = pilot_sym_end
        X[-subc_pilots]   = np.conj(pilot_sym_end)
        td = np.fft.ifft(X)
        end_blocks.append(np.hstack((td[-CP_LEN:], td)))
        print(f"[ofdm_mod] end pilot block #{i} length={len(td)+CP_LEN}")

    tx = np.hstack((pre_cp, *data_blocks, *end_blocks))
    print(f"[ofdm_mod] total TX samples={len(tx)}")
    return tx, pre_cp
# -------------------------
# 6) OFDM-демодулятор с header-trim
# -------------------------
def ofdm_dem(rx, pre_cp):
    print("[ofdm_dem] start demodulation")

    # пропустить BPF
    rx_f = rx
    print("[ofdm_dem] skipped bandpass filter")

    # AGC до синхронизации
    rx_f = apply_agc(rx_f)

    # синхронизация по преамбле
    corr = correlate(rx_f, pre_cp, mode='valid')
    idx  = np.argmax(np.abs(corr))
    print(f"[ofdm_dem] preamble idx={idx}")

    # оценка и компенсация CFO
    phi     = np.angle(corr[idx])
    Delta_f = phi/(2*math.pi*(len(pre_cp)/fs))
    print(f"[ofdm_dem] CFO phi={phi:.5f}, Δf={Delta_f:.2f}Hz")
    t      = np.arange(len(rx_f))/fs
    rx_cfo = rx_f * np.exp(-1j*2*math.pi*Delta_f*t)
    print("[ofdm_dem] applied CFO compensation")

    # AGC после CFO
    rx_cfo = apply_agc(rx_cfo)
    print("[ofdm_dem] applied AGC post-CFO")

    # нарезка на блоки
    start   = idx + len(pre_cp)
    blk_len = Nfft + CP_LEN
    first   = int(math.ceil(start/blk_len))
    print(f"[ofdm_dem] data starts at {start}, first block={first}")

    rec_syms = []
    for blk_idx in range(first, len(rx_cfo)//blk_len):
        print(f"[ofdm_dem] processing block #{blk_idx-first}")
        blk = rx_cfo[blk_idx*blk_len:(blk_idx+1)*blk_len]
        blk *= AGC_TARGET/(np.max(np.abs(blk))+1e-12)
        no_cp = blk[CP_LEN:]
        X     = np.fft.fft(no_cp)

        # равнение по пилотам
        H_p  = X[subc_pilots]/pilot_sym
        Hr   = np.interp(subc_data, subc_pilots, H_p.real)
        Hi   = np.interp(subc_data, subc_pilots, H_p.imag)
        H_i  = Hr + 1j*Hi
        phi_blk = np.angle(np.mean(H_p))

        Xd = X[subc_data]/H_i * np.exp(-1j*phi_blk)
        rec_syms.append(Xd)
        print(f"[ofdm_dem] collected {len(Xd)} symbols from block")

    rx_syms = np.hstack(rec_syms) if rec_syms else np.array([], complex)
    print(f"[ofdm_dem] total received symbols={len(rx_syms)}")

    # QPSK-демап и header-trim
    all_bits = qpsk_demap(rx_syms)
    print(f"[ofdm_dem] raw bits len={len(all_bits)}")

    # разбор header
    hdr_bits = all_bits[:18]
    n_bytes  = int(''.join(str(b) for b in hdr_bits[:16]), 2)
    last_idx = int(''.join(str(b) for b in hdr_bits[16:18]), 2)
    print(f"[ofdm_dem] parsed header: n_bytes={n_bytes}, last_idx={last_idx}")

    # выборка payload_bits
    payload_len = n_bytes*8
    payload_bits = all_bits[18:18+payload_len]
    print(f"[ofdm_dem] extracted payload bits len={len(payload_bits)}")

    decoded = bits_to_text(payload_bits)
    return decoded, rx_syms

# -------------------------
# 7) Отрисовка созвездия
# -------------------------
def plot_constellation(syms, title):
    print(f"[plot] drawing constellation '{title}', points={len(syms)}")
    plt.figure(figsize=(4,4))
    plt.scatter(syms.real, syms.imag, s=30, c='navy', alpha=0.7)
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# -------------------------
# 8) main
# -------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_wav    = os.path.join(script_dir, "ofdm_tx.wav")
    print(f"[MAIN] script_dir={script_dir}")

    txt  = input("Введите текст: ")
    mode = input("Режим работы (test/file/mic): ").strip().lower()

    tx, pre_cp = ofdm_mod(txt)

    tx_agc = apply_agc(tx)
    wavfile.write(out_wav,
                  fs,
                  (tx_agc * np.iinfo(np.int16).max).astype(np.int16))
    print(f"[MAIN] Saved WAV to {out_wav}")

    if mode == "test":
        syms   = qpsk_map(text_to_bits(txt))
        plot_constellation(syms, "TX Constellation")
        decoded = bits_to_text(qpsk_demap(syms))
        print("[MAIN] test decoded:", decoded)
        sys.exit(0)

    if mode == "file":
        _, wavd = wavfile.read(out_wav)
        sig     = wavd[:,0] if wavd.ndim>1 else wavd
        rx      = sig.astype(float)/np.iinfo(wavd.dtype).max

    elif mode == "mic":
        rec = sd.rec(int(fs*len(pre_cp)), fs, 1, 'float32')
        sd.wait()
        rx = rec[:,0]/np.max(np.abs(rec))

    else:
        print("[MAIN] unsupported mode")
        sys.exit(1)

    decoded_rx, rx_syms = ofdm_dem(rx, pre_cp)
    plot_constellation(rx_syms, "RX Constellation")
    print("[MAIN] Decoded text:", decoded_rx)

