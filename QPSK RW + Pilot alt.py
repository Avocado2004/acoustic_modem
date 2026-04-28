import numpy as np
import scipy.signal as signal
from scipy.signal import freqz
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math

def text_to_bits(text, encoding='utf-8'):
    bs = ''.join(f'{b:08b}' for b in text.encode(encoding))
    return np.array(list(bs), dtype=int)

def bits_to_text(bits, encoding='utf-8'):
    chunks = [bits[i:i+8] for i in range(0, len(bits), 8)]
    data = bytes(int(''.join(str(b) for b in ch), 2) for ch in chunks)
    return data.decode(encoding, errors='ignore')

def qpsk_map(bits):
    mapping = {
        '00': (1+1j)/np.sqrt(2),
        '01': (-1+1j)/np.sqrt(2),
        '11': (-1-1j)/np.sqrt(2),
        '10': (1-1j)/np.sqrt(2),
    }
    if len(bits) % 2:
        bits = np.append(bits, 0)
    syms = []
    for i in range(0, len(bits), 2):
        key = f"{bits[i]}{bits[i+1]}"
        syms.append(mapping[key])
    return np.array(syms)

def qpsk_demap(symbols):
    bits = []
    for s in symbols:
        b_real = 0 if s.real > 0 else 1
        b_imag = 0 if s.imag > 0 else 1
        bits += [b_imag, b_real]
    return np.array(bits, dtype=int)

def ofdm_mod(bits, Nfft, subc_inds):
    # зарезервировать 2 нижних + 2 верхних поднесущих под пилоты
    subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
    subc_data   = np.setdiff1d(subc_inds, subc_pilots)
    pilot_sym   = (1+1j)/np.sqrt(2)

    data_syms = qpsk_map(bits)
    n_data    = len(subc_data)
    pad       = (-len(data_syms)) % n_data
    if pad:
        data_syms = np.concatenate([data_syms,
                                    np.zeros(pad, dtype=complex)])
    blocks = data_syms.reshape(-1, n_data)

    tx_time = []
    for blk in blocks:
        X = np.zeros(Nfft, dtype=complex)
        X[subc_data]   = blk
        X[subc_pilots] = pilot_sym
        X[-subc_data]   = np.conj(blk)
        X[-subc_pilots] = np.conj(pilot_sym)
        tx_time.append(np.real(np.fft.ifft(X)))
    return np.concatenate(tx_time), blocks.shape[0]

def ofdm_dem(rx, Nfft, subc_inds, nblocks, orig_len):
    subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
    subc_data   = np.setdiff1d(subc_inds, subc_pilots)
    pilot_sym   = (1+1j)/np.sqrt(2)

    rx = rx[: nblocks * Nfft]
    frames = rx.reshape(nblocks, Nfft)
    rec_syms = []

    for idx, frame in enumerate(frames):
        X = np.fft.fft(frame)

        # оценить канал+фильтр по пилотам
        pilots_rx = X[subc_pilots]
        g = np.mean(pilots_rx / pilot_sym)

        # скорректировать всю ОФДМ-спектрограмму
        X_corr = X / g

        print(f"Frame {idx:2d}: |g|={abs(g):.3f}, ∠g={np.angle(g, deg=True):.1f}°")
        rec_syms.append(X_corr[subc_data])

    rec_syms = np.concatenate(rec_syms)
    bits = qpsk_demap(rec_syms)
    return bits[:orig_len]

def plot_constellation(symbols, title):
    plt.figure(figsize=(5,5))
    plt.scatter(symbols.real, symbols.imag, s=30, alpha=0.6)
    plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
    plt.title(title); plt.xlabel('I'); plt.ylabel('Q')
    plt.grid(True); plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # Параметры
    fs     = 48000
    f_low  = 300
    f_high = 4300
    Nfft   = 512
    df     = fs / Nfft

    # активные поднесущие
    k_low  = int(math.ceil(f_low  / df))
    k_high = int(math.floor(f_high / df))
    subc_inds = np.arange(k_low, k_high+1)

    print("Поднесущие:", subc_inds)
    print(f"Скорость символов: {fs/Nfft:.2f} Гц\n")

    # 1) ввод текста
    text     = input("Введите текст: ")
    bits     = text_to_bits(text)
    orig_len = len(bits)

    # 2) OFDM-модуляция с пилотами
    tx_ofdm, nblocks = ofdm_mod(bits, Nfft, subc_inds)

    # 3) FIR-BPF + нормализация по пилотным частотам
    taps = signal.firwin(numtaps=9,
                         cutoff=[f_low, f_high],
                         fs=fs,
                         pass_zero=False)

    # расчет амплитудного отклика на пилотных частотах
    subc_pilots = np.hstack((subc_inds[:2], subc_inds[-2:]))
    pilot_freqs = subc_pilots * df
    w, h = freqz(taps, [1.0], worN=pilot_freqs, fs=fs)
    gain = np.mean(np.abs(h))
    taps /= gain  # нормируем так, чтобы средний |H|=1

    # 4) фильтрация без фазовых искажений
    tx_filt = signal.filtfilt(taps, [1.0], tx_ofdm)

    # 5) нормировка амплитуды ±0.9
    tx_norm = tx_filt * (0.9 / np.max(np.abs(tx_filt)))

    # 6) сохранить WAV
    wavfile.write("ofdm_acoustic_tx.wav",
                  fs,
                  (tx_norm*np.iinfo(np.int16).max).astype(np.int16))
    print("Сохранено как ofdm_acoustic_tx.wav\n")

    # 7) спектр передатчика
    freqs    = np.fft.rfftfreq(len(tx_norm), 1/fs)
    spectrum = np.abs(np.fft.rfft(tx_norm))
    plt.figure(figsize=(8,4))
    plt.plot(freqs, 20*np.log10(spectrum + 1e-12))
    plt.title("Спектр TX"); plt.xlabel("Гц"); plt.ylabel("дБ")
    plt.grid(True); plt.show()

    # 8) созвездие TX
    plot_constellation(qpsk_map(bits), "Созвездие TX")

    # 9) чтение и подготовка RX
    _, rx_wav = wavfile.read("ofdm_acoustic_tx.wav")
    rx = rx_wav.astype(float)
    if rx.ndim > 1:
        rx = rx[:,0]
    rx_norm = rx / np.iinfo(rx_wav.dtype).max

    # 10) демодуляция с пилотной коррекцией
    rx_bits = ofdm_dem(rx_norm, Nfft, subc_inds, nblocks, orig_len)
    print("\nDecoded:", bits_to_text(rx_bits))

    # 11) созвездие RX до коррекции
    rx_syms = []
    for i in range(nblocks):
        X = np.fft.fft(rx_norm[i*Nfft:(i+1)*Nfft])
        rx_syms.append(X[subc_inds])
    plot_constellation(np.concatenate(rx_syms),
                       "Созвездие RX (до корр.)")

