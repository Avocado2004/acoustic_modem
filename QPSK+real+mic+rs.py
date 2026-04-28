#!/usr/bin/env python3
"""
OFDM Acoustic Modem с медианным фильтром для сглаживания амплитуды канала.
Фильтруем только |H_est| медианой, сохраняем файлы в директории скрипта.
"""

import os
import sys

# ─── I. Устанавливаем CWD = папка скрипта ──────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"[INFO] Рабочая директория → {script_dir}")

# ─── II. Импорты и параметры ────────────────────────────────────────────────
import numpy as np
from scipy.signal import fftconvolve, medfilt
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt
import math
from reedsolo import RSCodec, ReedSolomonError

fs        = 48000      # Гц
Nfft      = 512        # размер FFT
Ncp       = 128        # длина CP

# Параметры FEC
RS_DATA_BYTES   = 10   # k = 10 байт пользовательских данных
RS_PARITY_BYTES = 2    # n−k = 2 байта контроля → исправляет до floor(2/2)=1 байта ошибок
rs = RSCodec(RS_PARITY_BYTES)

# Индексы активных поднесущих (броадкаст-диапазон)
df        = fs / Nfft
k_low     = int(math.ceil(300   / df))
k_high    = int(math.floor(5000 / df))
subc_inds = np.arange(k_low, k_high + 1)
Nsub      = len(subc_inds)


# ─── III. Text ↔ Bits ─────────────────────────────────────────────────────
def text_to_bits(text, encoding='utf-8'):
    bs = ''.join(f"{b:08b}" for b in text.encode(encoding))
    return np.array(list(bs), dtype=int)

def bits_to_text(bits, encoding='utf-8'):
    L    = (len(bits)//8)*8
    b    = bits[:L].reshape(-1,8)
    data = bytes(int("".join(str(x) for x in row), 2) for row in b)
    return data.decode(encoding, errors="ignore")



def bytes_to_bits(data: bytes) -> np.ndarray:
    """Преобразовать каждый байт в 8 бит big-endian."""
    bits = []
    for byte in data:
        bits.extend([int(b) for b in f"{byte:08b}"])
    return np.array(bits, dtype=int)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Собрать битовый массив в байты (отбрасывая хвостовую неполную группу)."""
    L = (len(bits) // 8) * 8
    b = bits[:L].reshape(-1, 8)
    return bytes(int("".join(str(bit) for bit in row), 2) for row in b)



# ─── IV. QPSK Mapper / Demapper ────────────────────────────────────────────
def qpsk_map(bits):
    M = {
        '00': (1+1j)/np.sqrt(2),
        '01': (-1+1j)/np.sqrt(2),
        '11': (-1-1j)/np.sqrt(2),
        '10': (1-1j)/np.sqrt(2),
    }
    if len(bits) % 2:
        bits = np.append(bits, 0)
    syms = [M[f"{bits[i]}{bits[i+1]}"] for i in range(0, len(bits), 2)]
    return np.array(syms)

def qpsk_demap(syms):
    bits = []
    for s in syms:
        b0 = 0 if s.real > 0 else 1
        b1 = 0 if s.imag > 0 else 1
        bits += [b1, b0]
    return np.array(bits, dtype=int)


# ─── V. OFDM symbol + Hermitian symmetry ──────────────────────────────────
def ofdm_symbol(data_syms):
    X = np.zeros(Nfft, dtype=complex)
    X[subc_inds]  = data_syms
    X[-subc_inds] = np.conj(data_syms)
    X[0] = X[0].real
    if Nfft % 2 == 0:
        X[Nfft//2] = X[Nfft//2].real
    x = np.fft.ifft(X)
    x = np.real(x)
    return np.concatenate((x[-Ncp:], x))

def build_preamble(reps=2):
    pilot = np.full(Nsub, (1+1j)/np.sqrt(2))
    S     = ofdm_symbol(pilot)
    return np.tile(S, reps)

def build_data_td(bits):
    syms = qpsk_map(bits)
    pad  = (-len(syms)) % Nsub
    if pad:
        syms = np.concatenate((syms, np.zeros(pad, dtype=complex)))
    blk = syms.reshape(-1, Nsub)
    td  = [ofdm_symbol(b) for b in blk]
    return np.concatenate(td), blk.shape[0]


# ─── VI. Синхронизация по корреляции ───────────────────────────────────────
def sync_by_corr(rx, pre):
    corr = fftconvolve(rx, pre[::-1], mode='valid')
    return int(np.argmax(np.abs(corr)))


# ─── VII. Визуализация созвездия ───────────────────────────────────────────
def plot_constellation(syms, title):
    plt.figure(figsize=(5,5))
    plt.scatter(syms.real, syms.imag, s=20, alpha=0.6, color='navy')
    plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
    plt.title(title); plt.xlabel('I'); plt.ylabel('Q')
    plt.grid(True); plt.axis('equal')
    plt.show()


# ─── Добавляем функции для оценки ошибок ─────────────────────────────────

def compute_bit_errors(orig_bits, rx_bits):
    """
    Считает количество и долю ошибочных бит.
    orig_bits: исходный массив бит (0/1)
    rx_bits:   демодулированные биты
    возвращает: (num_errors, ber)
    """
    L = min(len(orig_bits), len(rx_bits))
    errs = np.count_nonzero(orig_bits[:L] != rx_bits[:L])
    ber  = errs / L if L > 0 else 0.0
    return errs, ber

def levenshtein_distance(s, t):
    """
    Вычисляет редакционное расстояние (Левенштейна) между строками s и t.
    """
    m, n = len(s), len(t)
    # инициализируем матрицу (m+1)x(n+1)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s[i-1] == t[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # удаление
                dp[i][j-1] + 1,      # вставка
                dp[i-1][j-1] + cost  # замена
            )
    return dp[m][n]

def compute_text_errors(orig_text, rx_text):
    """
    Считает количество символных ошибок и Character Error Rate (CER).
    """
    dist = levenshtein_distance(orig_text, rx_text)
    L    = max(len(orig_text), len(rx_text), 1)
    cer  = dist / L
    return dist, cer

# ─── VIII. Main workflow ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Ввод текста и бит
    text = input("Введите текст для передачи: ")
    data_bytes = text.encode('utf-8')

    # 1) Шардирование на куски по RS_DATA_BYTES
    blocks = [data_bytes[i:i+RS_DATA_BYTES]
              for i in range(0, len(data_bytes), RS_DATA_BYTES)]
    # 2) Дозаполнение последнего блока нулями до полной длины
    if len(blocks[-1]) < RS_DATA_BYTES:
        blocks[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks[-1]))
    nblk = len(blocks)

    # 3) RS-кодирование каждого блока
    encoded_blocks = [rs.encode(block) for block in blocks]

    # 4) Перевод кодированных байт в битовый массив и объединение
    bits_blocks = [bytes_to_bits(b) for b in encoded_blocks]
    data_bits  = np.concatenate(bits_blocks)
    # Общая длина битов понадобится для BER (до модуляции нужно знать orig_len_bits)
    orig_len_bits = len(data_bits)


    # 2) Построение TX
    preamble_td  = build_preamble(reps=2)
    data_td, nblk = build_data_td(data_bits)
    tx = np.concatenate((preamble_td, data_td))
    tx *= 0.9 / np.max(np.abs(tx))

    # 3) Сохранение WAV 16-bit
    # добавляем 0.25 с тишины перед сигналом (preroll)
    preroll = np.zeros(int(0.25 * fs), dtype=tx.dtype)
    tx_with_preroll = np.concatenate((preroll, tx))
    
    wavfile.write(
        "ofdm_acoustic_tx.wav",
        fs,
        (tx_with_preroll * np.iinfo(np.int16).max).astype(np.int16)
    )
    print("TX saved to ofdm_acoustic_tx.wav (с 0.25 с преролом)")

    # 4) Приём (file / mic)
    src = input("Откуда демодулировать? (file/mic): ").strip().lower()
    if src == 'file':
        _, wavd = wavfile.read("ofdm_acoustic_tx.wav")
        sig     = wavd[:,0] if wavd.ndim>1 else wavd
        rx      = sig.astype(float) / np.iinfo(wavd.dtype).max
    else:
        print("Запись с микрофона 10 секунд...")
        rec = sd.rec(int(10*fs), samplerate=fs, channels=1)
        sd.wait()
        rx = rec[:,0].astype(float)
        rx /= np.max(np.abs(rx))

    # 5) Синхронизация
    sync_idx = sync_by_corr(rx, preamble_td)
    print(f"Sync index: {sync_idx}")
    


    # 6) Извлечение данных
    start   = sync_idx + len(preamble_td)
    needed  = nblk * (Nfft + Ncp)
    segment = rx[start:start+needed]
    if len(segment) < needed:
        segment = np.concatenate((segment, np.zeros(needed - len(segment))))

    # 7) Оценка канального отклика
     # Вычисляем начало первого и второго символа преамбулы
    sym_len    = Nfft + Ncp
    base       = sync_idx
    
    # идеальный OFDM-символ (у вас два одинаковых, берём один)
    ideal_sym  = tx[          : sym_len][Ncp:]
    
    # приёмные символы
    start1     = base + 0*sym_len + Ncp
    start2     = base + 1*sym_len + Ncp
    rx_pre1    = rx[start1     : start1+Nfft]
    rx_pre2    = rx[start2     : start2+Nfft]
    
    # FFT и усреднение
    # … после синхронизации и извлечения rx_pre1, rx_pre2 …



    # 1) FFT идеального и приёмного преамбульных символов
    S_ref = np.fft.fft(ideal_sym)
    R1    = np.fft.fft(rx_pre1)
    R2    = np.fft.fft(rx_pre2)
 
    # 2) Инициализируем канал нулями
    H_est = np.zeros(Nfft, dtype=complex)
 
    # 3) Оцениваем H только на активных поднесущих
    S_ref_k = S_ref[subc_inds]
    R1_k    = R1[subc_inds]
    R2_k    = R2[subc_inds]
 
    H1_k    = R1_k / S_ref_k
    H2_k    = R2_k / S_ref_k
    Hk      = (H1_k + H2_k) / 2
 
    # 4) Записываем обратно в общий вектор
    H_est[subc_inds] = Hk

    # 8) Сглаживание |H_est| медианным фильтром
    Hk          = H_est[subc_inds]
    Hk_mag      = np.abs(Hk)
    Hk_mag_smooth = medfilt(Hk_mag, kernel_size=5)
    # … после медианного фильтра над Hk_mag_smooth …

   # 1) Ограничиваем минимальную Hk_mag_smooth так, чтобы 1/Hk_mag_smooth ≤ 2
    max_gain      = 2.0
    min_H_mag     = 1.0 / max_gain
    Hk_mag_smooth = np.clip(Hk_mag_smooth, a_min=min_H_mag, a_max=None)

   # 2) Восстанавливаем комплексный отклик с ограничением
    Hk_phase    = np.angle(Hk)
    Hk_smooth   = Hk_mag_smooth * np.exp(1j * Hk_phase)


    # 9) Эквализация и сбор символов
    frames = segment.reshape(nblk, Nfft+Ncp)
    rx_syms = []
    for frm in frames:
        Xf   = np.fft.fft(frm[Ncp:])
        eq   = Xf[subc_inds] / Hk_smooth
        rx_syms.append(eq)
    rx_syms = np.concatenate(rx_syms)

    # 10) Демаппинг и восстановление текста
    # Демаппинг всей последовательности
    rx_bits       = qpsk_demap(rx_syms)[:orig_len_bits]

    # Разбиваем RX-биты на кодированные RS-блоки
    codeword_len_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
    decoded_bytes     = []

    print("=== RS-декодирование по блокам ===")
    for i in range(nblk):
        start      = i * codeword_len_bits
        end        = start + codeword_len_bits
        block_bits = rx_bits[start:end]
        block_bytes= bits_to_bytes(block_bits)
        
        try:
            decoded_tuple = rs.decode(block_bytes)
            msg           = decoded_tuple[0]
            status        = "OK"
        except ReedSolomonError:
            msg           = b'\x00' * RS_DATA_BYTES
            status        = "ERR"
        
        print(f"Block {i+1}/{nblk}: RS decode = {status}")
        decoded_bytes.append(msg)

    # Сборка и вывод текста
    all_bytes = b"".join(decoded_bytes).rstrip(b'\x00')
    rec_text  = all_bytes.decode('utf-8', errors='ignore')
    

    # Подсчёт ошибок бит
    bit_errs, ber = compute_bit_errors(data_bits, rx_bits)
    print(f"[METRICS] Bit errors: {bit_errs}/{orig_len_bits}  (BER = {ber:.2%})")

    # Подсчёт ошибок символов
    char_errs, cer = compute_text_errors(text, rec_text)
    print(f"[METRICS] Char errors: {char_errs}/"
          f"{len(text)}  (CER = {cer:.2%})")

    print("Восстановленный текст:", rec_text)

    # 11) Отрисовка созвездия
    plot_constellation(rx_syms, "RX Constellation (equalized)")

    # 1) Частоты активных поднесущих в Гц
    freqs    = subc_inds * fs / Nfft

    # 2) Усиление эквалайзера = |1 / Hk_smooth|
    eq_gain  = 1.0 / np.abs(Hk_smooth)

    # 3) Построение графика
    plt.figure(figsize=(8,4))
    plt.plot(freqs, eq_gain, marker='o', linestyle='-',
             color='darkorange', label='Gain = 1/|H_smooth|')
    plt.title("Equalizer Gain vs Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
