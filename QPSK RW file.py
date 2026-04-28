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
    symbols = []
    for i in range(0, len(bits), 2):
        pair = f"{bits[i]}{bits[i+1]}"
        symbols.append(mapping[pair])
    return np.array(symbols)

def qpsk_demap(symbols):
    bits = []
    for s in symbols:
        r, i = s.real, s.imag
        b0 = 0 if r > 0 else 1
        b1 = 0 if i > 0 else 1
        bits += [b1, b0]
    return np.array(bits, dtype=int)

def ofdm_mod(bits, Nfft, subcarriers):
    qsyms = qpsk_map(bits)
    Nsub = len(subcarriers)
    pad = (-len(qsyms)) % Nsub
    if pad:
        qsyms = np.concatenate([qsyms, np.zeros(pad, dtype=complex)])
    qsyms = qsyms.reshape(-1, Nsub)
    ofdm_time = []
    for block in qsyms:
        X = np.zeros(Nfft, dtype=complex)
        X[subcarriers] = block
        X[-subcarriers] = np.conj(block)
        x = np.fft.ifft(X)
        ofdm_time.append(np.real(x))
    return np.concatenate(ofdm_time), len(qsyms)

def ofdm_dem(rx, Nfft, subcarriers, num_blocks, orig_bit_len):
    rx = rx[: num_blocks * Nfft ]
    frames = rx.reshape(num_blocks, Nfft)
    rec_syms = []
    for x in frames:
        X = np.fft.fft(x)
        rec_syms.append(X[subcarriers])
    rec_syms = np.concatenate(rec_syms)
    bits = qpsk_demap(rec_syms)
    return bits[:orig_bit_len]

def plot_constellation(symbols, title):
    plt.figure(figsize=(5,5))
    plt.scatter(symbols.real, symbols.imag,
                s=30, alpha=0.6, color='navy')
    plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
    plt.title(title); plt.xlabel('I'); plt.ylabel('Q')
    plt.grid(True); plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # Параметры канала и OFDM
    fs     = 48000      # Гц
    f_low  = 300        # Гц
    f_high = 4300       # Гц
    Nfft   = 512
    df     = fs / Nfft

    # Индексы активных поднесущих
    k_low     = int(math.ceil(f_low  / df))
    k_high    = int(math.floor(f_high / df))
    subc_inds = np.arange(k_low, k_high+1)

    # Частота смены OFDM-символов
    symbol_rate = fs / Nfft
    print("Используемые поднесущие (FFT индексы):", subc_inds)
    print(f"Частота смены символов: {symbol_rate:.2f} Гц")

    # 1) Ввод текста
    text     = input("Введите текст для передачи: ")
    bits     = text_to_bits(text)
    orig_len = len(bits)

    # 2) OFDM-модуляция
    tx_ofdm, nblocks = ofdm_mod(bits, Nfft, subc_inds)

    # 3) Полосно-пропускающий фильтр
    taps    = signal.firwin(
        numtaps=9,
        cutoff=[f_low, f_high],
        fs=fs,
        pass_zero=False
    )
    tx_filt = signal.filtfilt(taps, [1.0], tx_ofdm)

    # 3.1) Нормализация амплитуды до ±0.9
    norm_coef = 0.9 / np.max(np.abs(tx_filt))
    tx_norm   = tx_filt * norm_coef

    # 4) Запись в WAV
    wavfile.write(
        "ofdm_acoustic_tx.wav",
        fs,
        (tx_norm * np.iinfo(np.int16).max).astype(np.int16)
    )
    print("Сигнал сохранен в ofdm_acoustic_tx.wav (нормирован до ±0.9)")

    # 5) Вывод спектра сохраненного сигнала
    N        = len(tx_norm)
    freqs    = np.fft.rfftfreq(N, 1/fs)
    spectrum = np.abs(np.fft.rfft(tx_norm))
    plt.figure(figsize=(8,4))
    plt.plot(freqs, 20 * np.log10(spectrum + 1e-12))
    plt.title("Спектр сохраненного сигнала")
    plt.xlabel("Частота, Гц")
    plt.ylabel("Уровень, дБ")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6) Созвездие TX
    tx_syms = qpsk_map(bits)
    plot_constellation(tx_syms, "Созвездие TX (все поднесущие)")

    # 7) Чтение и нормирование принятого сигнала
    fs2, rx_wav = wavfile.read("ofdm_acoustic_tx.wav")
    if rx_wav.ndim > 1:
        rx_wav = rx_wav[:,0]
    rx_norm = rx_wav.astype(float) / np.iinfo(rx_wav.dtype).max

    # 8) Демодуляция OFDM → биты
    rx_bits  = ofdm_dem(rx_norm, Nfft, subc_inds, nblocks, orig_len)
    rec_text = bits_to_text(rx_bits)
    print("Восстановленный текст:", rec_text)

    # 9) Созвездие RX
    rx_syms = np.concatenate([
        np.fft.fft(rx_norm[i*Nfft:(i+1)*Nfft])[subc_inds]
        for i in range(nblocks)
    ])
    plot_constellation(rx_syms, "Созвездие RX (все поднесущие)")

