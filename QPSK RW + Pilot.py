import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math

def text_to_bits(text, encoding='utf-8'):
    bits_str = ''.join(f'{b:08b}' for b in text.encode(encoding))
    return np.array(list(bits_str), dtype=int)

def bits_to_text(bits, encoding='utf-8'):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    data = bytes(int(''.join(str(b) for b in byte), 2) for byte in chars)
    return data.decode(encoding, errors='ignore')

def qpsk_map(bits):
    mapping = {
        '00':  (1+1j)/np.sqrt(2),
        '01':  (-1+1j)/np.sqrt(2),
        '11':  (-1-1j)/np.sqrt(2),
        '10':  (1-1j)/np.sqrt(2),
    }
    if len(bits) % 2:
        bits = np.append(bits, 0)
    syms = []
    for i in range(0, len(bits), 2):
        pair = f"{bits[i]}{bits[i+1]}"
        syms.append(mapping[pair])
    return np.array(syms)

def qpsk_demap(symbols):
    bits = []
    for s in symbols:
        b0 = 0 if s.real > 0 else 1
        b1 = 0 if s.imag > 0 else 1
        bits += [b1, b0]
    return np.array(bits, dtype=int)

def ofdm_mod(bits, Nfft, subcarriers):
    # резервируем 2 нижних + 2 верхних поднесущих под пилоты
    subc_pilots = np.hstack((subcarriers[:2], subcarriers[-2:]))
    subc_data   = np.setdiff1d(subcarriers, subc_pilots)
    pilot_sym   = (1+1j)/np.sqrt(2)

    data_syms = qpsk_map(bits)
    n_data    = len(subc_data)
    pad       = (-len(data_syms)) % n_data
    if pad:
        data_syms = np.concatenate([data_syms, np.zeros(pad, dtype=complex)])
    blocks = data_syms.reshape(-1, n_data)

    tx_time = []
    for blk in blocks:
        X = np.zeros(Nfft, dtype=complex)
        X[subc_data]   = blk
        X[subc_pilots] = pilot_sym
        # зеркальное заполнение для получения вещественного сигнала
        X[-subc_data]   = np.conj(blk)
        X[-subc_pilots] = np.conj(pilot_sym)
        tx_time.append(np.real(np.fft.ifft(X)))
    return np.concatenate(tx_time), blocks.shape[0]

def ofdm_dem(rx, Nfft, subcarriers, num_blocks, orig_bit_len):
    subc_pilots = np.hstack((subcarriers[:2], subcarriers[-2:]))
    subc_data   = np.setdiff1d(subcarriers, subc_pilots)
    pilot_sym   = (1+1j)/np.sqrt(2)

    # обрезаем и разбиваем по символам
    rx = rx[: num_blocks * Nfft]
    frames = rx.reshape(num_blocks, Nfft)

    rec_syms = []
    for idx, frame in enumerate(frames):
        X = np.fft.fft(frame)

        # 1) Оценка g по текущим пилотам
        pilots_rx = X[subc_pilots]
        g = np.mean(pilots_rx / pilot_sym)

        # 2) Коррекция всего спектра
        X_corr = X / g

        # 3) Сбор данных
        rec_syms.append(X_corr[subc_data])

        # дебаг: печать амплитуды и фазы g
        print(f"Frame {idx:2d}: |g|={abs(g):.3f}, ∠g={np.angle(g, deg=True):.1f}°")

    rec_syms = np.concatenate(rec_syms)
    bits = qpsk_demap(rec_syms)
    return bits[:orig_bit_len]

def plot_constellation(symbols, title):
    plt.figure(figsize=(5,5))
    plt.scatter(symbols.real, symbols.imag, s=30, alpha=0.6, color='navy')
    plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
    plt.title(title); plt.xlabel('I'); plt.ylabel('Q')
    plt.grid(True); plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # параметры
    fs     = 48000
    f_low  = 300
    f_high = 4300
    Nfft   = 512
    df     = fs / Nfft

    k_low  = int(math.ceil(f_low  / df))
    k_high = int(math.floor(f_high / df))
    subc_inds = np.arange(k_low, k_high+1)

    print("Активные поднесущие:", subc_inds)
    print(f"Символная скорость: {fs/Nfft:.2f} Гц\n")

    # ввод и модуляция
    text     = input("Введите текст: ")
    bits     = text_to_bits(text)
    orig_len = len(bits)
    tx_ofdm, nblocks = ofdm_mod(bits, Nfft, subc_inds)

    # BPF + нормировка
    taps    = signal.firwin(9, [f_low, f_high], fs=fs, pass_zero=False)
    tx_filt = signal.filtfilt(taps, [1], tx_ofdm)
    tx_norm = tx_filt * (0.9/np.max(np.abs(tx_filt)))

    # запись и чтение
    wavfile.write("tx.wav", fs, (tx_norm*np.iinfo(np.int16).max).astype(np.int16))
    _, rx_wav = wavfile.read("tx.wav")
    if rx_wav.ndim>1: rx_wav = rx_wav[:,0]
    rx_norm = rx_wav.astype(float)/np.iinfo(rx_wav.dtype).max

    # демодуляция
    rx_bits = ofdm_dem(rx_norm, Nfft, subc_inds, nblocks, orig_len)
    print("\nDecoded:", bits_to_text(rx_bits))

    # созвездия
    plot_constellation(qpsk_map(bits),    "Созвездие TX")
    rx_syms = []
    for i in range(nblocks):
        X = np.fft.fft(rx_norm[i*Nfft:(i+1)*Nfft])
        rx_syms.append(X[subc_inds])
    plot_constellation(np.concatenate(rx_syms), "Созвездие RX (до корр.)")
