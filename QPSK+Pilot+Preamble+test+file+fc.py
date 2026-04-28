#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import sounddevice as sd
from scipy.signal import firwin, filtfilt, correlate
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math

# -------------------------
# 1) Параметры системы
# -------------------------
fs      = 48000
Nfft    = 512
CP_LEN  = 64

f_low, f_high = 300, 5000
df = fs / Nfft

k_low   = int(math.ceil(f_low/df))
k_high  = int(math.floor(f_high/df))
subc_inds  = np.arange(k_low, k_high+1)

subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
subc_data   = np.setdiff1d(subc_inds, subc_pilots)

pilot_sym      = 1+0j               # обычные пилоты
pilot_sym_end  = 2+2j               # маркер конца
N_END_PILOT_BLK = 2                 # число концевых блоков

AGC_TARGET = 0.9

# -------------------------
# 2) Линейно-фазный BPF + AGC
# -------------------------
def design_bpf(f_low, f_high, fs, numtaps=129):
    print(f"[design_bpf] f_low={f_low}, f_high={f_high}, fs={fs}, numtaps={numtaps}")
    taps = firwin(numtaps,
                  cutoff=[f_low, f_high],
                  fs=fs,
                  pass_zero=False)
    return taps, [1.0]

b_bpf, a_bpf = design_bpf(f_low, f_high, fs)

def apply_agc(sig, target=AGC_TARGET, eps=1e-12):
    peak = np.max(np.abs(sig)) + eps
    gain = target / peak
    print(f"[apply_agc] peak={peak:.5f}, gain={gain:.5f}")
    return sig * gain

# -------------------------
# 3) Text ↔ Bits
# -------------------------
def text_to_bits(text, enc='utf-8'):
    bs = ''.join(f'{b:08b}' for b in text.encode(enc))
    bits = np.array(list(bs), int)
    print(f"[text_to_bits] text_len={len(text)}, bits_len={len(bits)}")
    return bits

def bits_to_text(bits, enc='utf-8'):
    L = (len(bits)//8)*8
    arr = bits[:L].reshape(-1,8)
    data = bytes(int(''.join(str(x) for x in row), 2)
                 for row in arr)
    text = data.decode(enc, errors='ignore')
    print(f"[bits_to_text] bits_len={len(bits)}, decoded_text='{text}'")
    return text

# -------------------------
# 4) QPSK map/demap
# -------------------------
def qpsk_map(bits):
    """
    Преобразует поток бит в QPSK-символы:
      b0 → знак реальной части,
      b1 → знак мнимой части.
    """
    if len(bits) % 2:
        bits = np.append(bits, 0)
        print("[qpsk_map] odd bit count, appended 0")
    # группируем по 2: bits[0::2] → b0, bits[1::2] → b1
    re = 1 - 2*bits[0::2]       # 0→+1, 1→-1
    im = 1 - 2*bits[1::2]       # 0→+1, 1→-1
    syms = (re + 1j*im) / np.sqrt(2)
    print(f"[qpsk_map] mapping {len(bits)} bits → {len(syms)} symbols")
    return syms


def qpsk_demap(symbols):
    bits = []
    for s in symbols:
        bits += [0 if s.real>0 else 1, 0 if s.imag>0 else 1]
    print(f"[qpsk_demap] demapped {len(symbols)} symbols → {len(bits)} bits")
    return np.array(bits, int)

# -------------------------
# 5) Zadoff–Chu preamble
# -------------------------
def generate_zc(N, u=1):
    print(f"[generate_zc] N={N}, u={u}")
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n+1) / N)

# -------------------------
# 6) OFDM-модулятор
# -------------------------
def ofdm_mod(text):
    print(f"[ofdm_mod] start modulation for text='{text}'")
    data_bits = text_to_bits(text)
    n_bytes   = len(data_bits)//8
    rem       = len(data_bits) % 8
    last_idx  = 3 if rem == 0 else (rem//2 - 1)

    header = np.array(
      list(f"{n_bytes:016b}") +
      list(f"{last_idx:02b}"),
      int
    )
    print(f"[ofdm_mod] header bytes={n_bytes}, last_idx={last_idx}, header_bits={''.join(map(str,header))}")

    bits = np.hstack((header, data_bits))
    pad_bits = (-len(bits)) % 8
    if pad_bits:
        print(f"[ofdm_mod] pad_bits={pad_bits}")
        bits = np.hstack((bits, np.ones(pad_bits, int)))

    syms = qpsk_map(bits)

    pad = (-len(syms)) % len(subc_data)
    if pad:
        print(f"[ofdm_mod] pad syms for block alignment, pad={pad}")
        syms = np.concatenate((syms, np.zeros(pad, complex)))
    data_blk = syms.reshape(-1, len(subc_data))
    print(f"[ofdm_mod] data blocks={data_blk.shape[0]}")

    # ZC-преамбула
    Nzc  = len(subc_inds)
    zc   = generate_zc(Nzc, u=1)
    Xz   = np.zeros(Nfft, complex)
    Xz[subc_inds]   = zc
    Xz[-subc_inds]  = np.conj(zc)
    pre_nocp = np.fft.ifft(Xz)
    pre_cp   = np.hstack((pre_nocp[-CP_LEN:], pre_nocp))
    print(f"[ofdm_mod] preamble length={len(pre_cp)} samples")

    # OFDM-блоки данных
    data_blocks = []
    for blk in data_blk:
        X = np.zeros(Nfft, complex)
        X[subc_data]    = blk
        X[subc_pilots]  = pilot_sym
        X[-subc_data]   = np.conj(blk)
        X[-subc_pilots] = np.conj(pilot_sym)
        td = np.fft.ifft(X)
        data_blocks.append(np.hstack((td[-CP_LEN:], td)))

    # концевые блоки
    end_blocks = []
    for _ in range(N_END_PILOT_BLK):
        X = np.zeros(Nfft, complex)
        X[subc_pilots]  = pilot_sym_end
        X[-subc_pilots] = np.conj(pilot_sym_end)
        td = np.fft.ifft(X)
        end_blocks.append(np.hstack((td[-CP_LEN:], td)))

    tx = np.concatenate((pre_cp, *data_blocks, *end_blocks))
    return tx, pre_cp

# -------------------------
# 7) OFDM-демодулятор
# -------------------------
def ofdm_dem(rx, pre_cp):
    print("[ofdm_dem] start demodulation")
    # BPF + AGC
    rx_f = filtfilt(b_bpf, a_bpf, rx)
    print("[ofdm_dem] applied bandpass filter")
    rx_f = apply_agc(rx_f)

    # синхро по преамбле
    corr = correlate(rx_f, pre_cp, mode='valid')
    idx  = np.argmax(np.abs(corr))
    print(f"[ofdm_dem] preamble detected at sample index={idx}")

    # глобальная CFO-компенсация
    phi     = np.angle(corr[idx])
    Delta_f = phi / (2*np.pi * (len(pre_cp)/fs))
    print(f"[ofdm_dem] CFO phi={phi:.5f} rad, Δf={Delta_f:.2f} Hz")
    t       = np.arange(len(rx_f)) / fs
    rx_cfo  = rx_f * np.exp(-1j * 2*np.pi * Delta_f * t)

    # начало данных
    start    = idx + len(pre_cp)
    blk_len  = Nfft + CP_LEN
    # Рассчитываем первый блок (целое и с округлением вверх)
    first_block = int(np.ceil(start/blk_len))
    print(f"[ofdm_dem] data symbols start at sample index={start}")
    print(f"[ofdm_dem] blk_len={blk_len}, first_block={first_block}")

    rec_syms           = []
    eqs                = []
    phase_corrections  = []

    for blk_idx in range(first_block, len(rx_cfo)//blk_len):
        block = rx_cfo[blk_idx*blk_len:(blk_idx+1)*blk_len]
        print(f"[ofdm_dem] processing block #{blk_idx-first_block}")

        # AGC для каждого блока
        peak_blk = np.max(np.abs(block))
        gain_blk = AGC_TARGET / (peak_blk + 1e-12)
        print(f"  AGC block peak={peak_blk:.5f}, gain={gain_blk:.5f}")
        block *= gain_blk

        no_cp = block[CP_LEN:]
        X     = np.fft.fft(no_cp)

        # проверка конца
        if np.allclose(X[subc_pilots], pilot_sym_end, atol=0.1):
            print("[ofdm_dem] end-of-data pilot detected, stopping")
            break

        # эквализация
        H_p = X[subc_pilots] / pilot_sym
        Hr  = np.interp(subc_data, subc_pilots, H_p.real)
        Hi  = np.interp(subc_data, subc_pilots, H_p.imag)
        H_i = Hr + 1j*Hi
        eqs.append(H_i)

        # фазовая коррекция блока
        phi_blk = np.angle(np.mean(H_p))
        phase_corrections.append(phi_blk)
        print(f"  Phase correction phi_blk={phi_blk:.5f} rad")

        # применяем коррекцию
        Xd = X[subc_data] / H_i
        Xd *= np.exp(-1j * phi_blk)
        rec_syms.append(Xd)

    # (построение графиков опущено для краткости)

    syms = np.hstack(rec_syms) if rec_syms else np.array([], complex)
    bits = qpsk_demap(syms)

    # вставляем отладку по битам
    print(f"[ofdm_dem] full demapped bits: {''.join(map(str, bits))}")
    hdr_bits = bits[:18]
    n_bytes  = int(''.join(map(str, hdr_bits[:16])), 2)
    last_idx = int(''.join(map(str, hdr_bits[16:])), 2)
    # корректный подсчёт количества data bits
    if last_idx == 3:
        total_data_bits = n_bytes * 8
    else:
        total_data_bits = (n_bytes - 1)*8 + (last_idx + 1)*2
    data_bits = bits[18:18 + total_data_bits]

    print(f"[ofdm_dem] hdr_bits   : {''.join(map(str, hdr_bits))}")
    print(f"[ofdm_dem] n_bytes    : {n_bytes}, last_idx={last_idx}")
    print(f"[ofdm_dem] data_bits  : {''.join(map(str, data_bits))}")

    decoded = bits_to_text(data_bits)
    return decoded, syms

# -------------------------
# 8) Констелляция
# -------------------------
def plot_constellation(syms, title):
    plt.figure(figsize=(4,4))
    plt.scatter(syms.real, syms.imag, s=20, alpha=0.6, color='navy')
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.grid(True)
    plt.axis('equal')
    plt.title(title)
    plt.show()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # 1) Ввод текста и выбор режима работы
    txt = input("Введите текст: ")
    mode = input("Режим работы (test/file/mic): ").strip().lower()

    # 2) Режим test: простой круговой тест QPSK
    if mode == "test":
        bits_orig = text_to_bits(txt)
        syms = qpsk_map(bits_orig)
        bits_rec = qpsk_demap(syms)
        decoded = bits_to_text(bits_rec)
        print("Test mode - decoded text:", decoded)
        sys.exit(0)

    # 3) OFDM-модуляция
    tx, pre_cp = ofdm_mod(txt)

    # 4) Показываем TX-констелляцию
    plot_constellation(qpsk_map(text_to_bits(txt)), "TX Constellation")

    # 5) AGC на базовой полосе
    tx = apply_agc(tx)

    # 6) Модуляция всего сигнала на поднесущую fc = f_high - f_low
    fc = f_high - f_low
    t = np.arange(len(tx)) / fs
    # переводим комплексный baseband в пассбэнд
    tx_passband = tx * np.exp(1j * 2 * np.pi * fc * t)

    # 7) Сохраняем только реальную часть в WAV
    wavfile.write(
        "ofdm_tx.wav",
        fs,
        (np.real(tx_passband) * np.iinfo(np.int16).max).astype(np.int16)
    )
    print("Saved ofdm_tx.wav (passband)")

    # 8) Приём: из файла или с микрофона
    if mode == "file":
        # читаем из сохранённого WAV
        _, wavd = wavfile.read("ofdm_tx.wav")
        sig = wavd[:,0] if wavd.ndim > 1 else wavd
        rx_real = sig.astype(float) / np.iinfo(wavd.dtype).max

        # демодуляция с той же частоты fc → возвращаем комплексный baseband
        t_rx = np.arange(len(rx_real)) / fs
        rx = rx_real * np.exp(-1j * 2 * np.pi * fc * t_rx)

    else:
        rec = sd.rec(len(tx), fs, 1, 'float32')
        sd.wait()
        # при прямом микрофонном вводе сохраняем комплексность, нормируем
        rx = rec[:,0] / np.max(np.abs(rec))

    # 9) Демодулятор и вывод
    decoded, rx_syms = ofdm_dem(rx, pre_cp)

    # 10) Показываем RX-констелляцию с градиентом времени
    plt.figure(figsize=(4,4))
    idxs = np.arange(len(rx_syms))
    plt.scatter(
        rx_syms.real,
        rx_syms.imag,
        c=idxs,
        cmap='RdBu',
        s=20,
        alpha=0.6
    )
    plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
    plt.grid(True); plt.axis('equal')
    plt.title("RX Constellation (gradient)")
    plt.show()

    print("Decoded text:", decoded)
