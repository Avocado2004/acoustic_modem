#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import sounddevice as sd
from scipy.signal import firwin, correlate
from scipy.io import wavfile
import matplotlib.pyplot as plt
from collections import deque
import threading
import random

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

# OFDM-пилоты
pilot_sym       = AGC_TARGET + 0j
pilot_sym_end   = 2+2j
N_END_PILOT_BLK = 2

# старт-пилоты перед преамбулой
N_START_PILOT_BLK = 10

# детект если E_pilots/E_tot > PILOT_DET_THRESH
PILOT_DET_THRESH = 0.1

# индексы всех задействованных несущих
k_low       = int(math.ceil(f_low/df))
k_high      = int(math.floor(f_high/df))
subc_inds   = np.arange(k_low, k_high+1)
subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
subc_data   = np.setdiff1d(subc_inds, subc_pilots)

sym_len = Nfft + CP_LEN

# -------------------------
# 1) BPF и AGC
# -------------------------
def design_bpf(f_low, f_high, fs, numtaps=129):
    taps = firwin(numtaps, cutoff=[f_low, f_high],
                  pass_zero=False, fs=fs)
    return taps, 1.0

b_bpf, a_bpf = design_bpf(f_low, f_high, fs)

def apply_agc(sig, target=1.0):
    """
    Робастная AGC по 95%-квантилю, чтобы игнорировать единичные выбросы.
    """
    mag = np.abs(sig)
    p95 = np.percentile(mag, 95)
    gain = target/(p95 + 1e-12)
    return sig * gain


# -------------------------
# 2) text↔bits, QPSK map/demap
# -------------------------
def text_to_bits(text, enc='utf-8'):
    bs   = ''.join(f"{b:08b}" for b in text.encode(enc))
    return np.array(list(bs), int)

def bits_to_text(bits, enc='utf-8'):
    L   = (len(bits)//8)*8
    arr = bits[:L].reshape(-1,8)
    data= bytes(int(''.join(str(x) for x in row),2)
                for row in arr)
    return data.decode(enc, errors='ignore')

def qpsk_map(bits):
    if len(bits)%2:
        bits = np.append(bits,0)
    b0, b1 = bits[0::2], bits[1::2]
    re = 1-2*b0; im = 1-2*b1
    return (re+1j*im)/math.sqrt(2)

def qpsk_demap(symbols):
    bits=[]
    for s in symbols:
        bits += [0 if s.real>0 else 1,
                 0 if s.imag>0 else 1]
    return np.array(bits,int)

# -------------------------
# 3) Golay-преамбла
# -------------------------
def golay_pair(n):
    """
    Рекурсивная генерация Golay complementary pair длины n=2^k.
    Возвращает два массива из ±1, у которых сумма автокорреляций = δ[n].
    """
    if n == 1:
        return np.array([1], int), np.array([1], int)
    a, b = golay_pair(n // 2)
    A = np.hstack((a, b))
    B = np.hstack((a, -b))
    return A, B


# -------------------------
# 4) OFDM-модулятор
# -------------------------
def ofdm_mod(text):
    """
    OFDM-модулятор с Golay-преамблой.
    Возвращает:
      tx    — комплексный сигнал (start-pilots + pre_A + pre_B + data + end-pilots)
      pre_A — первый Golay-блок с CP
      pre_B — второй Golay-блок с CP
    """
    # 1) Header + data → биты → QPSK-символы
    data_bits = text_to_bits(text)
    n_bytes   = len(data_bits) // 8
    rem       = len(data_bits) % 8
    last_idx  = 3 if rem == 0 else (rem // 2 - 1)
    hdr_bits  = np.hstack((
        np.array(list(f"{n_bytes:016b}"), int),
        np.array(list(f"{last_idx:02b}"),  int)
    ))
    bits      = np.hstack((hdr_bits, data_bits))
    pad_bits  = (-len(bits)) % 8
    bits      = np.hstack((bits, np.zeros(pad_bits, int)))
    syms      = qpsk_map(bits)
    pad_s     = (-len(syms)) % len(subc_data)
    syms      = np.hstack((syms, np.zeros(pad_s, complex)))
    data_blk  = syms.reshape(-1, len(subc_data))

    # 2) Старт-пилоты
    start_blocks = []
    for _ in range(N_START_PILOT_BLK):
        X       = np.zeros(Nfft, complex)
        X[subc_pilots]    = pilot_sym
        X[-subc_pilots]   = np.conj(pilot_sym)
        td      = np.fft.ifft(X)
        start_blocks.append(np.hstack((td[-CP_LEN:], td)))

    # 3) Golay-преамбла (два real-блока A и B)
    Nzc = Nfft  # power of two
    A, B = golay_pair(Nzc)
    pre_A = np.hstack((A[-CP_LEN:], A)).astype(float)
    pre_B = np.hstack((B[-CP_LEN:], B)).astype(float)

    # 4) Data-блоки
    data_blocks = []
    for blk in data_blk:
        X = np.zeros(Nfft, complex)
        X[subc_data]    = blk
        X[subc_pilots]  = 1+0j
        X[-subc_data]   = np.conj(blk)
        X[-subc_pilots] = np.conj(1+0j)
        td = np.fft.ifft(X)
        data_blocks.append(np.hstack((td[-CP_LEN:], td)))

    # 5) End-пилоты
    end_blocks = []
    for _ in range(N_END_PILOT_BLK):
        X = np.zeros(Nfft, complex)
        X[subc_pilots]    = pilot_sym_end
        X[-subc_pilots]   = np.conj(pilot_sym_end)
        td = np.fft.ifft(X)
        end_blocks.append(np.hstack((td[-CP_LEN:], td)))

    tx = np.hstack((
        *start_blocks,
        pre_A,
        pre_B,
        *data_blocks,
        *end_blocks
    ))
    return tx, pre_A, pre_B


# -------------------------
# 5) OFDM-демодулятор
# -------------------------
def ofdm_dem(rx, pre_A, pre_B):
    """
    OFDM-демодулятор с Golay-преамблой:
      - AGC
      - Golay A+B correlation
      - CFO compensation
      - skip exactly two preamble blocks
      - FFT, equalization, demapping, header parsing
    """
    # 1) AGC
    rx_f = apply_agc(rx)

    # 2) correlation
    corrA = correlate(rx_f, pre_A, mode="valid")
    corrB = correlate(rx_f, pre_B, mode="valid")
    corr  = corrA + corrB
    idx   = int(np.argmax(np.abs(corr)))
    peak  = np.abs(corr[idx])
    print(f"[ofdm_dem] preamble peak @ sample {idx}, amp={peak:.3f}")

    # 3) CFO
    phi    = np.angle(corr[idx])
    delta  = phi / (2 * np.pi * (len(pre_A)/fs))
    t      = np.arange(len(rx_f)) / fs
    rx_cfo = apply_agc(rx_f * np.exp(-1j * 2 * np.pi * delta * t))

    # 4) SKIP TWO PREAMBLE BLOCKS <<<<<
    sym_len    = Nfft + CP_LEN
    pre_blocks = 2
    start      = idx + pre_blocks * sym_len
    blk_len    = sym_len
    first_blk  = int(np.ceil(start / blk_len))
    print(f"[ofdm_dem] data starts at sample {start}, block #{first_blk}")

    # 5) Process data blocks
    rec_syms = []
    for bi in range(first_blk, len(rx_cfo)//blk_len):
        sidx, eidx = bi*blk_len, (bi+1)*blk_len
        blk         = rx_cfo[sidx:eidx]

        # optional per-block AGC
        gain_blk = AGC_TARGET/(np.percentile(np.abs(blk),95)+1e-12)
        blk     *= gain_blk

        no_cp = blk[CP_LEN:]
        X     = np.fft.fft(no_cp)

        # end-pilot detection
        if np.allclose(X[subc_pilots], pilot_sym_end, atol=0.1):
            print(f"[ofdm_dem] end-pilot at block {bi}, stopping")
            break

        # equalize
        H_p      = X[subc_pilots]/(1+0j)
        Hr       = np.interp(subc_data, subc_pilots, H_p.real)
        Hi       = np.interp(subc_data, subc_pilots, H_p.imag)
        H_i      = Hr + 1j*Hi
        phi_blk  = np.angle(np.mean(H_p))
        Xd       = X[subc_data]/H_i * np.exp(-1j * phi_blk)
        rec_syms.append(Xd)

    # 6) Demap & header
    rx_syms    = np.hstack(rec_syms) if rec_syms else np.array([], complex)
    bits       = qpsk_demap(rx_syms)
    hdr_bits   = bits[:18]
    hb16       = ''.join(str(b) for b in hdr_bits[:16])
    hb2        = ''.join(str(b) for b in hdr_bits[16:18])
    n_bytes    = int(hb16, 2) if hdr_bits.size>=16 else 0
    last_idx   = int(hb2, 2) if hdr_bits.size>=18 else 0
    payload    = bits[18:18 + n_bytes*8]
    decoded    = bits_to_text(payload)

    print(f"[ofdm_dem] hdr={hb16}|{hb2}, n_bytes={n_bytes}, last_idx={last_idx}")
    print(f"[ofdm_dem] decoded = '{decoded}'")
    return decoded, rx_syms

# -------------------------
# 6) Goertzel-энергия
# -------------------------
def goertzel_energy(x, k, N=Nfft):
    coeff = 2*math.cos(2*math.pi*k/N)
    s0=s1=s2=0.0
    for xn in x:
        s0 = xn + coeff*s1 - s2
        s2, s1 = s1, s0
    real = s1 - s2*math.cos(2*math.pi*k/N)
    imag = s2*math.sin(2*math.pi*k/N)
    return real*real + imag*imag

# -------------------------
# 7) Слушаем микрофон
# -------------------------
#import threading
import numpy as np
import sounddevice as sd
from collections import deque

# доля от идеального пика корреляции, при которой считаем преамбулу «найденной»
CORR_FRAC = 0.02

def listen_mic(tx, pre_A, pre_B):
    """
    Слушаем микрофон, детектим полный Golay-преамбл (pre_A+pre_B)
    по скользящей корреляции. Ложных срабатываний в тишине не будет.
    
    tx     — массив переданного сигнала (для длины)
    pre_A  — первый Golay-блок с CP
    pre_B  — второй Golay-блок с CP
    """
    sym_len  = len(pre_A)                      # Nfft+CP_LEN
    # составляем всю преамблу, по которой будем коррелировать
    preamble = np.hstack((pre_A, pre_B))
    L        = len(preamble)                   # = 2*sym_len

    # идеальный корреляционный пик = <preamble, preamble> = sum(preamble²)
    energy_ref = np.sum(preamble * preamble)
    thresh     = CORR_FRAC * energy_ref

    buf         = []
    ring        = deque(maxlen=L)
    detected    = False
    detect_idx  = None
    stop_event  = threading.Event()

    def callback(indata, frames, time_info, status):
        nonlocal detected, detect_idx
        samples = indata[:,0]
        buf.extend(samples)
        ring.extend(samples)

        # как только накопили ровно L сэмплов, считаем корреляцию
        if not detected and len(ring) == L:
                x        = np.array(ring)
                corr_val = np.dot(x, preamble)
                abs_val  = abs(corr_val)
                print(f"[MIC] corr_val = {corr_val:.1f}, |corr_val| = {abs_val:.1f}, threshold = {thresh:.1f}")
                if abs_val > thresh:
                    detect_idx = len(buf) - L
                    print(f"[MIC] preamble detected at sample {detect_idx} "
                          f"(sign = {'+' if corr_val>0 else '–'})")
                    detected = True


        # как только захватили полный пакет — выходим
        if detected and len(buf) >= detect_idx + len(tx):
            stop_event.set()

    with sd.InputStream(channels=1,
                        samplerate=fs,
                        blocksize=512,     # или sym_len, не критично
                        dtype='float32',
                        callback=callback):
        stop_event.wait()

    buff = np.array(buf)
    # откуда брать полный tx (учитываем старт-пилоты)
    tx_start = detect_idx - N_START_PILOT_BLK * sym_len
    if tx_start < 0:
        print(f"[MIC] warning: tx_start={tx_start} clamped to 0")
        tx_start = 0

    full = buff[tx_start : tx_start + len(tx)]
    if full.size < len(tx):
        raise RuntimeError(f"[MIC] full too short: {full.size} < {len(tx)}")

    print(f"[MIC] full_tx starts at {tx_start}, length={full.size}")

    # выкидываем старт-пилоты
    offset = N_START_PILOT_BLK * sym_len
    rx0    = full[offset:]
    rx0   /= np.max(np.abs(rx0))
    print(f"[MIC] returning rx0 len = {len(rx0)}")
    return rx0


# 8) main
# -------------------------
# -----------------------------
# main.py (полностью)
# -----------------------------
if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    import sounddevice as sd
    from scipy.io import wavfile
    import matplotlib.pyplot as plt

    # 1) Ввод текста и режима
    txt  = input("Введите текст: ")
    mode = input("Режим (test/file/mic): ").strip().lower()

    # 2) OFDM-модуляция → tx, pre_A, pre_B
    tx, pre_A, pre_B = ofdm_mod(txt)
    tx_agc          = apply_agc(tx)

    # 3) Сохранение WAV (только real-часть)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_wav    = os.path.join(script_dir, "ofdm_tx.wav")
    wavfile.write(
        out_wav,
        fs,
        (np.real(tx_agc) * np.iinfo(np.int16).max).astype(np.int16)
    )
    print(f"[MAIN] saved '{out_wav}', total {len(tx)} samples")

    # 4) Сбор rx0 в зависимости от режима
    if mode == "file":
        _, wavd = wavfile.read(out_wav)
        sig     = wavd[:,0] if wavd.ndim>1 else wavd
        rx0     = sig.astype(float) / np.iinfo(wavd.dtype).max

    elif mode == "mic":
        # В режиме mic вы сами воспроизводите ofdm_tx.wav во внешней программе
        rx0 = listen_mic(tx, pre_A, pre_B)

    elif mode == "test":
        syms = qpsk_map(text_to_bits(txt))
        plt.figure()
        plt.scatter(syms.real, syms.imag, c='teal', s=30, alpha=0.7)
        plt.title("TX Constellation")
        plt.axis('equal'); plt.grid(True); plt.show()
        print("test decoded:", bits_to_text(qpsk_demap(syms)))
        sys.exit(0)

    else:
        print(f"[MAIN] unsupported mode '{mode}'")
        sys.exit(1)

    # 5) Для отладки: фиксированная задержка перед демодуляцией
    fixed_delay_samps = 1152
    print(f"[MAIN] injecting FIXED silence of {fixed_delay_samps} samples")
    rx0 = np.hstack((np.zeros(fixed_delay_samps, dtype=rx0.dtype), rx0))

    # 6) Демодуляция и декодирование
    decoded, rx_syms = ofdm_dem(rx0, pre_A, pre_B)

    # 7) Отрисовка RX-созвездия и вывод текста
    plt.figure()
    plt.scatter(rx_syms.real, rx_syms.imag, c='navy', s=30, alpha=0.7)
    plt.title("RX Constellation")
    plt.axis('equal'); plt.grid(True); plt.show()

    print("Decoded:", decoded)
