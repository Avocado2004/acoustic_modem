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
import struct



fs        = 48000      # Гц
Nfft      = 512        # размер FFT
Ncp       = 128        # длина CP

# Параметры FEC
RS_DATA_BYTES   = 8   # k = 10 байт пользовательских данных
RS_PARITY_BYTES = 4    # n−k = 2 байта контроля → исправляет до floor(2/2)=1 байта ошибок
rs = RSCodec(RS_PARITY_BYTES)

# Индексы активных поднесущих (броадкаст-диапазон)
df        = fs / Nfft
k_low     = int(math.ceil(300   / df))
k_high    = int(math.floor(4300 / df))
subc_inds = np.arange(k_low, k_high + 1)
Nsub      = len(subc_inds)

# Максимальный размер передаваемых данных (1 ГБ)
MAX_PAYLOAD_SIZE = 1 << 30  # 1 073 741 824 байта




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
# вверху файла, после всех импортов
# чтобы везде была доступна преамбула
    preamble_td = build_preamble(reps=2)
    # I. Выбор режима работы: [T]ransmit (передача) или [R]eceive (приём)
    op = input("Режим работы — [T]ransmit или [R]eceive (по умолчанию T): ").strip().upper()
    op = "R" if op == "R" else "T"

    # II. ————————————— TX-блок —————————————
    if op == "T":
        # 1) Выбор режима полезной нагрузки: File/Text
        mode = input("Режим передачи — [F]ile (по умолчанию) или [T]ext: ").strip().upper()
        mode = "T" if mode == "T" else "F"

        # 2) Формируем payload = header + данные
        if mode == "F":
            file_path = input("Путь к файлу для передачи: ").strip()
            if not os.path.isfile(file_path):
                print(f"Файл не найден: {file_path}")
                sys.exit(1)
            file_size = os.path.getsize(file_path)
            if file_size > MAX_PAYLOAD_SIZE:
                print("Файл слишком крупный (больше 1 ГБ).")
                sys.exit(1)

            filename   = os.path.basename(file_path)
            name_bytes = filename.encode("utf-8")
            if len(name_bytes) > 255:
                print("Имя файла слишком длинное (>255 байт).")
                sys.exit(1)
            with open(file_path, "rb") as f:
                file_data = f.read()

            header  = (
                b"F" +
                struct.pack(">Q", file_size) +
                struct.pack("B", len(name_bytes)) +
                name_bytes
            )
            payload = header + file_data

        else:
            text        = input("Введите текст для передачи: ")
            text_bytes  = text.encode("utf-8")
            header      = b"T" + struct.pack(">Q", len(text_bytes))
            payload     = header + text_bytes

        # 3) RS-кодирование → bits → OFDM TX → WAV
        data_bytes     = payload
        blocks         = [data_bytes[i:i+RS_DATA_BYTES]
                          for i in range(0, len(data_bytes), RS_DATA_BYTES)]
        if len(blocks[-1]) < RS_DATA_BYTES:
            blocks[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks[-1]))
        nblk           = len(blocks)
        encoded_blocks = [rs.encode(b) for b in blocks]
        bits_blocks    = [bytes_to_bits(b) for b in encoded_blocks]
        data_bits      = np.concatenate(bits_blocks)
        orig_len_bits  = len(data_bits)

        preamble_td = build_preamble(reps=2)
        data_td, _  = build_data_td(data_bits)
        tx          = np.concatenate((preamble_td, data_td))
        tx         *= 0.9 / np.max(np.abs(tx))

        preroll = np.zeros(int(0.25*fs), dtype=tx.dtype)
        tx_out  = np.concatenate((preroll, tx))
        wavfile.write("ofdm_acoustic_tx.wav", fs,
                      (tx_out * np.iinfo(np.int16).max).astype(np.int16))
        print("TX saved to ofdm_acoustic_tx.wav (с 0.25 с преролом)")

    # III. ————————————— RX-блок —————————————
    # Если мы были в режиме TX, то сначала спросим откуда демодулировать,
    # иначе — сразу слушаем.
    src = input("Откуда демодулировать? (file/mic): ").strip().lower()
    if src == "file":
        # В режиме RX-only спрашиваем путь к WAV
        if op == "R":
            wav_path = input("Путь к WAV-файлу для приёма: ").strip()
        else:
            wav_path = "ofdm_acoustic_tx.wav"
        _, wavd = wavfile.read(wav_path)
        sig     = wavd[:,0] if wavd.ndim>1 else wavd
        rx      = sig.astype(float) / np.iinfo(wavd.dtype).max
    else:
        print("Запись с микрофона 10 секунд…")
        rec = sd.rec(int(10*fs), samplerate=fs, channels=1)
        sd.wait()
        rx = rec[:,0].astype(float)
        rx /= np.max(np.abs(rx))

    # 1) Синхронизация
    sync_idx = sync_by_corr(rx, preamble_td)
    print(f"Sync index: {sync_idx}")
    

    # 2) Извлекаем сегмент OFDM-данных
    # 6) Извлечение сегмента OFDM-данных (динамическое nblk в режиме приёма)
    symbol_len = Nfft + Ncp
    start      = sync_idx + len(preamble_td)
    remaining  = len(rx) - start

    # вычисляем, сколько целых OFDM-символов (блоков) успело прийти
    nblk = remaining // symbol_len
    if nblk == 0:
        raise RuntimeError("Не удалось найти ни одного OFDM-блока после синхронизации")

    print(f"[RX] Обнаружено OFDM-блоков: {nblk}")

    needed  = nblk * symbol_len
    segment = rx[start:start+needed]


    # 3) Оценка канального отклика и эквализация
    sym_len    = Nfft + Ncp
    # берём первый символ преамбулы из preamble_td
    # (preamble_td = build_preamble(reps=2) лежит вне main)
    ideal_sym  = preamble_td[Ncp : Ncp + Nfft]
    rx_pre1    = rx[sync_idx + Ncp                : sync_idx + Ncp + Nfft]
    rx_pre2    = rx[sync_idx + sym_len + Ncp       : sync_idx + sym_len + Ncp + Nfft]
    S_ref      = np.fft.fft(ideal_sym)
    R1, R2     = np.fft.fft(rx_pre1), np.fft.fft(rx_pre2)
    Hk         = ((R1[subc_inds]/S_ref[subc_inds]) + (R2[subc_inds]/S_ref[subc_inds]))/2
    Hk_mag_s   = np.clip(medfilt(np.abs(Hk), 5), 1/2.0, None)
    Hk_smooth  = Hk_mag_s * np.exp(1j*np.angle(Hk))

    # 4) Сбор и демаппинг символов
    frames    = segment.reshape(nblk, Nfft+Ncp)
    rx_syms   = np.concatenate([np.fft.fft(frm[Ncp:])[subc_inds]/Hk_smooth
                                for frm in frames])
        
    # Для RS-декодирования нужно ровно nblk * codeword_len_bits бит
    codeword_len_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
    total_cw_bits     = codeword_len_bits * nblk

    all_rx_bits = qpsk_demap(rx_syms)
    # демапим ровно столько, чтобы разбить на RS-блоки
    rx_bits     = all_rx_bits[:total_cw_bits]


    # 5) RS-декодирование
    cw_len_bits = (RS_DATA_BYTES+RS_PARITY_BYTES)*8
    decoded     = []
    rs_statuses = []
    print("=== RS-декодирование по блокам ===")
    for i in range(nblk):
        bts = bits_to_bytes(rx_bits[i*cw_len_bits:(i+1)*cw_len_bits])
        try:
             msg    = rs.decode(bts)[0]
             status = "OK"
        except ReedSolomonError:
             msg    = b'\x00' * RS_DATA_BYTES
             status = "ERR"
 
        print(f"Block {i+1}/{nblk}: RS decode = {status}")
        rs_statuses.append(status)
        decoded.append(msg)
        # после завершения RS-декодирования собираем весь stream байт
        all_bytes = b"".join(decoded).rstrip(b"\x00")


    # 6) Парсинг header и вывод
    pos               = 0
    mode_rx           = all_bytes[pos:pos+1]; pos += 1
    total_sz          = struct.unpack(">Q", all_bytes[pos:pos+8])[0]; pos += 8
    # длина полезной нагрузки в битах (для демаппинга)
    payload_bits_len  = total_sz * 8

    if mode_rx == b"F":
        name_len = all_bytes[pos]; pos+=1
        fname    = all_bytes[pos:pos+name_len].decode("utf-8",errors="ignore")
        pos     += name_len
        body     = all_bytes[pos:pos+total_sz]
        out_path = "rx_"+fname
        with open(out_path,"wb") as f: f.write(body)
        print(f"[RX FILE] Сохранён файл: {out_path} ({len(body)} байт)")

        # FER по total_sz
        
        # 7) Block-level FER: считаем только реальные блоки с данными
        # число блоков с реальными байтами
        n_data_blocks = (total_sz + RS_DATA_BYTES - 1) // RS_DATA_BYTES
 
        # сколько из этих первых n_data_blocks упало
        err_blocks = sum(1 for s in rs_statuses[:n_data_blocks] if s == "ERR")
        fer_blocks = err_blocks / n_data_blocks
        print(f"[FER-blocks] Ошибок в RS-блоках: {err_blocks}/{n_data_blocks}  "
           f"(FER = {fer_blocks:.2%})")
 
      # дальше можно считать CER/BER по payload_bits_len, как было
 

    else:
        text_bytes = all_bytes[pos:pos+total_sz]
        rec_text   = text_bytes.decode("utf-8",errors="ignore")
        print("[RX TEXT]", rec_text)

        # CER по total_sz (сравниваем реально переданные байты и принятый текст)
        # 1) Исходная строка, извлечённая из header
        orig_str   = text_bytes.decode("utf-8", errors="ignore")
        # 2) Сравниваем символы
        char_errs  = sum(o != r for o, r in zip(orig_str, rec_text))
        cer        = char_errs / total_sz
        print(f"[CER] Char errors: {char_errs}/{total_sz}  (CER = {cer:.2%})")

    # 7) BER (только если в этом же запуске был TX)
    if 'data_bits' in locals():
        tx_bits        = data_bits[:payload_bits_len]
        bit_errs, ber  = compute_bit_errors(tx_bits, rx_bits)
        print(f"[BER] Bit errors: {bit_errs}/{payload_bits_len}  (BER = {ber:.2%})")
    else:
        print("[BER] пропущено (неизвестны исходные биты)")

    # 8) Констелляция и Gain
    plot_constellation(rx_syms, "RX Constellation (eq.)")
    freqs   = subc_inds * fs / Nfft
    eq_gain = 1.0/np.abs(Hk_smooth)
    plt.figure(figsize=(8,4))
    plt.plot(freqs, eq_gain, "o-", color="darkorange", label="Gain")
    plt.title("Equalizer Gain vs Frequency")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Gain")
    plt.grid(True); plt.legend(); plt.show()
