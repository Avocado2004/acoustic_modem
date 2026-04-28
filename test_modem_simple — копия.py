#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-carrier QPSK-модем с полосой 300–5000 Гц, 6-порядковым фильтром краёв
и визуализацией спектра сигнала

1. Ввод числа поднесущих (M) и текста
2. UTF-8 → битовая строка, паддинг до 2·M
3. Генерация M QPSK-поднесущих
4. Суммирование и 6-порядковый Butterworth-BPF (300–5000 Гц)
5. Визуализация спектра мульти-QPSK сигнала
6. Сохранение и чтение WAV
7. Демодуляция по каждой поднесущей
8. Биты → UTF-8 → вывод текста
"""

import numpy as np
import wave
import struct
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ========== Параметры полосы и фильтра ==========
f_low   = 300      # Гц
f_high  = 5000     # Гц
band    = f_high - f_low
fs      = 10 * band           # сэмпл/с, 10×ширина полосы
order   = 6                   # порядок фильтра

def design_bpf(fs, f1, f2, order=6):
    nyq = fs / 2
    b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
    return b, a

def plot_spectrum(signal: np.ndarray, fs: int, title: str = "Spectrum"):
    """
    Строит амплитудный спектр сигнала в dB.
    """
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

# ========== Кодирование/декодирование текста ==========
def text_to_bits(s: str) -> str:
    data = s.encode('utf-8')
    return ''.join(f"{b:08b}" for b in data)

def bits_to_text(bstr: str) -> str:
    ba = bytearray()
    for i in range(0, len(bstr), 8):
        ba.append(int(bstr[i:i+8], 2))
    return ba.decode('utf-8', errors='replace')

# ========== Модуляция нескольких поднесущих ==========
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

# ========== Демодуляция ==========
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

# ========== WAV I/O ==========
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

# ========== Главная программа ==========
if __name__ == "__main__":
    WAV_FILE = "qpsk_multi.wav"

    try:
        M = int(input("Число поднесущих [целое >0, по умолчанию 60]: ") or "60")
        assert M > 0
    except:
        M = 60

    text = input("Введите текст для передачи: ")
    bits = text_to_bits(text)
    print(f"Длина бит: {len(bits)}, поднесущих: {M}")

    tx = modulate_multi(bits, M)
    print(f"Сгенерировано сэмплов: {len(tx)}")

    # Визуализация спектра сигнала
    plot_spectrum(tx, fs, title="Спектр мульти-QPSK сигнала")

    save_wave(WAV_FILE, tx)
    print("Сигнал сохранён в", WAV_FILE)

    rx = load_wave(WAV_FILE)
    print(f"Прочитано сэмплов: {len(rx)}")

    rec_bits = demodulate_multi(rx, M)
    rec_bits = rec_bits[:len(bits)]
    rec_text = bits_to_text(rec_bits)

    print("Восстановленный текст:")
    print(rec_text)
    print("Статус:", "OK" if rec_text == text else "ERROR")

