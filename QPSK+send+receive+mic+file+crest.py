#!/usr/bin/env python3
"""
OFDM Acoustic Modem с фазированием поднесущих и реализацией метода Habr для уменьшения CREST.
Содержит: PHASE_METHOD = None | "schroeder" | "random" | "habr".
"""
import os
import sys
import math
import struct
import numpy as np
from scipy.signal import fftconvolve, medfilt
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
from reedsolo import RSCodec, ReedSolomonError

# --- Рабочая директория ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# -----------------------
# Параметры системы
# -----------------------
fs = 48000          # sample rate
Nfft = 512          # FFT size
Ncp = 128           # cyclic prefix
df = fs / Nfft

# поднесущие в диапазоне 300..4300 Гц
k_low = int(math.ceil(300 / df))
k_high = int(math.floor(4300 / df))
subc_inds = np.arange(k_low, k_high + 1)
Nsub = len(subc_inds)

# FEC (Reed-Solomon)
RS_DATA_BYTES = 8
RS_PARITY_BYTES = 4
rs = RSCodec(RS_PARITY_BYTES)

MAX_PAYLOAD_SIZE = 1 << 30

# PHASE METHOD
# None | "schroeder" | "random" | "habr"
PHASE_METHOD = "schroeder"

# Параметры метода Habr (grid-based per-subcarrier search)
HABR_SAMPLE_BLOCKS = 200     # количество тренировочных OFDM-блоков
HABR_PHASE_GRID = 36         # число точек в сетке фаз (напр. 36 -> шаг 10°)
HABR_MAX_ITERS = 2           # число проходов по всем поднесущим
HABR_SEED = 12345            # seed для воспроизводимости
HABR_SAVE_FILE = "habr_phases.npy"

# Инициализация глобальной переменной фаз (будет заполнена ниже)
subc_phases = None

# -----------------------
# Базовые вспомогательные функции
# -----------------------
def text_to_bits(text, encoding='utf-8'):
    bs = ''.join(f"{b:08b}" for b in text.encode(encoding))
    return np.array(list(bs), dtype=int)

def bits_to_text(bits, encoding='utf-8'):
    L = (len(bits)//8)*8
    b = bits[:L].reshape(-1,8)
    data = bytes(int("".join(str(x) for x in row), 2) for row in b)
    return data.decode(encoding, errors="ignore")

def bytes_to_bits(data: bytes) -> np.ndarray:
    bits = []
    for byte in data:
        bits.extend([int(b) for b in f"{byte:08b}"])
    return np.array(bits, dtype=int)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    L = (len(bits) // 8) * 8
    b = bits[:L].reshape(-1, 8)
    return bytes(int("".join(str(bit) for bit in row), 2) for row in b)

# QPSK mapping / demapping (consistent with original)
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

# crest factor
def crest_factor(sig):
    peak = np.max(np.abs(sig))
    rms = np.sqrt(np.mean(sig**2))
    if rms == 0:
        return np.inf
    return 20 * np.log10(peak / rms)

# -----------------------
# Функции фазирования поднесущих
# -----------------------
def make_subcarrier_phases(method=None, N=Nsub, seed=0, habr_phases=None):
    """
    Возвращает массив фаз длины N.
    Поддерживает: None, "random", "schroeder", "habr".
    Если method == "habr" и habr_phases задан, возвращаем их.
    """
    if method is None:
        return np.zeros(N)
    if method == "random":
        rnd = np.random.RandomState(seed)
        return rnd.uniform(0, 2*np.pi, size=N)
    if method == "schroeder":
        k = np.arange(N)
        return np.mod(np.pi * k * (k - 1) / float(N), 2*np.pi)
    if method == "habr":
        if habr_phases is not None:
            return np.array(habr_phases) % (2*np.pi)
        # start from Schroeder as initialization
        k = np.arange(N)
        return np.mod(np.pi * k * (k - 1) / float(N), 2*np.pi)
    return np.zeros(N)

# -----------------------
# OFDM symbol building with phase application
# -----------------------
def ofdm_symbol(data_syms):
    """
    Формирует OFDM временной символ с Hermitian-симметрией и CP.
    data_syms: комплексные символы длины <= Nsub
    применяется глобальная subc_phases (если не нулевые)
    """
    if len(data_syms) < Nsub:
        ds = np.concatenate((data_syms, np.zeros(Nsub - len(data_syms), dtype=complex)))
    else:
        ds = np.array(data_syms[:Nsub], dtype=complex)

    global subc_phases
    if subc_phases is not None and np.any(subc_phases != 0):
        ds = ds * np.exp(1j * subc_phases)

    X = np.zeros(Nfft, dtype=complex)
    X[subc_inds] = ds
    X[-subc_inds] = np.conj(ds)
    X[0] = X[0].real
    if Nfft % 2 == 0:
        X[Nfft//2] = X[Nfft//2].real
    x = np.fft.ifft(X)
    x = np.real(x)
    return np.concatenate((x[-Ncp:], x))

def build_preamble(reps=2):
    pilot = np.full(Nsub, (1+1j)/np.sqrt(2))
    S = ofdm_symbol(pilot)
    return np.tile(S, reps)

def build_data_td(bits):
    syms = qpsk_map(bits)
    pad = (-len(syms)) % Nsub
    if pad:
        syms = np.concatenate((syms, np.zeros(pad, dtype=complex)))
    blk = syms.reshape(-1, Nsub)
    td = [ofdm_symbol(b) for b in blk]
    return np.concatenate(td), blk.shape[0]

# -----------------------
# Синхронизация
# -----------------------
def sync_by_corr(rx, pre):
    corr = fftconvolve(rx, pre[::-1], mode='valid')
    return int(np.argmax(np.abs(corr)))

# -----------------------
# Диагностические/утилитарные функции
# -----------------------
def crest_per_symbol_and_total(td_symbols):
    sym_len = Nfft + Ncp
    n = len(td_symbols) // sym_len
    crests = []
    for i in range(n):
        s = td_symbols[i*sym_len:(i+1)*sym_len]
        crests.append(crest_factor(s))
    return np.array(crests), crest_factor(td_symbols)

def plot_constellation(syms, title):
    plt.figure(figsize=(5,5))
    plt.scatter(syms.real, syms.imag, s=20, alpha=0.6, color='navy')
    plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
    plt.title(title); plt.xlabel('I'); plt.ylabel('Q')
    plt.grid(True); plt.axis('equal')
    plt.show()

# -----------------------
# Habr method: training blocks and optimizer
# -----------------------
def make_training_blocks(n_blocks=HABR_SAMPLE_BLOCKS, seed=HABR_SEED):
    rng = np.random.RandomState(seed)
    blocks = []
    for _ in range(n_blocks):
        bits = rng.randint(0, 2, Nsub * 2)
        syms = qpsk_map(bits)
        blocks.append(syms)
    return np.array(blocks)

def optimize_phases_habr(sample_blocks=None, n_iter=HABR_MAX_ITERS, grid_size=HABR_PHASE_GRID, seed=HABR_SEED, verbose=True):
    """
    Greedy per-subcarrier grid search as in Habr article.
    Возвращает (optimized_phases_radians, best_crest_dB).
    """
    if sample_blocks is None:
        sample_blocks = make_training_blocks()
    M = sample_blocks.shape[0]
    # initial phases (Schroeder)
    phases = make_subcarrier_phases("schroeder", Nsub)
    # helper to compute concatenated TD for given phases
    def compute_total_td(phs):
        old = globals().get('subc_phases', None)
        globals()['subc_phases'] = phs
        td_list = [ofdm_symbol(sample_blocks[m]) for m in range(M)]
        td = np.concatenate(td_list)
        globals()['subc_phases'] = old
        return td

    best_td = compute_total_td(phases)
    best_crest = crest_factor(best_td)
    if verbose:
        print(f"[HABR] init crest = {best_crest:.3f} dB (Schroeder start)")

    grid = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
    for it in range(n_iter):
        if verbose:
            print(f"[HABR] Iteration {it+1}/{n_iter}")
        improved = False
        for k_idx in range(Nsub):
            cur_ph = phases[k_idx]
            best_local_phase = cur_ph
            best_local_crest = best_crest
            # evaluate grid
            for phi in grid:
                cand = phases.copy()
                cand[k_idx] = phi
                td_cand = compute_total_td(cand)
                c = crest_factor(td_cand)
                if c < best_local_crest:
                    best_local_crest = c
                    best_local_phase = phi
            if best_local_phase != cur_ph:
                phases[k_idx] = best_local_phase
                best_crest = best_local_crest
                improved = True
        if not improved:
            if verbose:
                print("[HABR] no improvement in iteration, stopping early")
            break
    if verbose:
        print(f"[HABR] final crest = {best_crest:.3f} dB")
    return phases % (2*np.pi), best_crest

# -----------------------
# Инициализация subc_phases (поддержка загрузки/сохранения)
# -----------------------
def init_phases():
    global subc_phases
    if PHASE_METHOD == "habr":
        if os.path.isfile(HABR_SAVE_FILE):
            try:
                subc_phases = np.load(HABR_SAVE_FILE)
                print(f"[HABR] loaded phases from {HABR_SAVE_FILE}")
                return
            except Exception as e:
                print("[HABR] failed load, will optimize:", e)
        # run optimization (time-consuming)
        train_blocks = make_training_blocks(n_blocks=HABR_SAMPLE_BLOCKS, seed=HABR_SEED)
        ph, c = optimize_phases_habr(sample_blocks=train_blocks,
                                     n_iter=HABR_MAX_ITERS,
                                     grid_size=HABR_PHASE_GRID,
                                     seed=HABR_SEED,
                                     verbose=True)
        subc_phases = ph
        try:
            np.save(HABR_SAVE_FILE, subc_phases)
            print(f"[HABR] saved optimized phases to {HABR_SAVE_FILE} (crest={c:.3f} dB)")
        except Exception as e:
            print("[HABR] failed save:", e)
    else:
        subc_phases = make_subcarrier_phases(PHASE_METHOD, Nsub, seed=HABR_SEED)
    print(f"[PHASE] method={PHASE_METHOD}, example degs[:8]={np.degrees(subc_phases[:8])}")

# Инициализируем фазы перед основной логикой
init_phases()

# -----------------------
# Main TX/RX workflow
# -----------------------
if __name__ == "__main__":
    preamble_td = build_preamble(reps=2)

    op = input("Режим работы — [T]ransmit или [R]eceive (по умолчанию T): ").strip().upper()
    op = "R" if op == "R" else "T"

    if op == "T":
        mode = input("Режим передачи — [F]ile или [T]ext (по умолчанию F): ").strip().upper()
        mode = "T" if mode == "T" else "F"

        if mode == "F":
            file_path = input("Путь к файлу для передачи: ").strip()
            if not os.path.isfile(file_path):
                print(f"Файл не найден: {file_path}")
                sys.exit(1)
            file_size = os.path.getsize(file_path)
            if file_size > MAX_PAYLOAD_SIZE:
                print("Файл слишком большой (>1GB).")
                sys.exit(1)
            filename = os.path.basename(file_path)
            name_bytes = filename.encode("utf-8")
            if len(name_bytes) > 255:
                print("Имя файла слишком длинное (>255 байт).")
                sys.exit(1)
            with open(file_path, "rb") as f:
                file_data = f.read()
            header = (b"F" +
                      struct.pack(">Q", file_size) +
                      struct.pack("B", len(name_bytes)) +
                      name_bytes)
            payload = header + file_data
        else:
            text = input("Введите текст для передачи: ")
            text_bytes = text.encode("utf-8")
            header = b"T" + struct.pack(">Q", len(text_bytes))
            payload = header + text_bytes

        # RS encode per RS_DATA_BYTES
        data_bytes = payload
        blocks = [data_bytes[i:i+RS_DATA_BYTES] for i in range(0, len(data_bytes), RS_DATA_BYTES)]
        if len(blocks[-1]) < RS_DATA_BYTES:
            blocks[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks[-1]))
        nblk = len(blocks)
        encoded_blocks = [rs.encode(b) for b in blocks]
        bits_blocks = [bytes_to_bits(b) for b in encoded_blocks]
        data_bits = np.concatenate(bits_blocks)
        orig_len_bits = len(data_bits)

        data_td, _ = build_data_td(data_bits)
        tx = np.concatenate((preamble_td, data_td))

        # CREST diagnostics on training set if HABR used
        if PHASE_METHOD == "habr" and os.path.isfile(HABR_SAVE_FILE):
            try:
                saved = np.load(HABR_SAVE_FILE)
                # quick train test on saved blocks if available (not stored here) - skip
            except:
                pass

        cf_before = crest_factor(tx)
        tx *= 0.9 / np.max(np.abs(tx))
        cf_after = crest_factor(tx)
        print(f"[CREST] before norm: {cf_before:.2f} dB, after norm: {cf_after:.2f} dB (PHASE_METHOD={PHASE_METHOD})")

        preroll = np.zeros(int(0.25*fs), dtype=tx.dtype)
        tx_out = np.concatenate((preroll, tx))
        wavfile.write("ofdm_acoustic_tx_habr.wav", fs, (tx_out * np.iinfo(np.int16).max).astype(np.int16))
        print("TX saved to ofdm_acoustic_tx_habr.wav (с 0.25 с преренролом)")

    else:
        src = input("Откуда демодулировать? (file/mic): ").strip().lower()
        if src == "file":
            wav_path = input("Путь к WAV-файлу для приёма: ").strip()
            _, wavd = wavfile.read(wav_path)
            sig = wavd[:,0] if wavd.ndim>1 else wavd
            rx = sig.astype(float) / np.iinfo(wavd.dtype).max
        else:
            print("Запись с микрофона 10 секунд…")
            rec = sd.rec(int(10*fs), samplerate=fs, channels=1)
            sd.wait()
            rx = rec[:,0].astype(float)
            rx /= np.max(np.abs(rx))

        sync_idx = sync_by_corr(rx, preamble_td)
        print(f"Sync index: {sync_idx}")

        symbol_len = Nfft + Ncp
        start = sync_idx + len(preamble_td)
        remaining = len(rx) - start
        nblk = remaining // symbol_len
        if nblk == 0:
            raise RuntimeError("Не удалось найти ни одного OFDM-блока после синхронизации")

        print(f"[RX] Обнаружено OFDM-блоков: {nblk}")
        needed = nblk * symbol_len
        segment = rx[start:start+needed]

        ideal_sym = preamble_td[Ncp : Ncp + Nfft]
        rx_pre1 = rx[sync_idx + Ncp : sync_idx + Ncp + Nfft]
        rx_pre2 = rx[sync_idx + symbol_len + Ncp : sync_idx + symbol_len + Ncp + Nfft]
        S_ref = np.fft.fft(ideal_sym)
        R1, R2 = np.fft.fft(rx_pre1), np.fft.fft(rx_pre2)
        Hk = ((R1[subc_inds]/S_ref[subc_inds]) + (R2[subc_inds]/S_ref[subc_inds]))/2
        Hk_mag_s = np.clip(medfilt(np.abs(Hk), 5), 1/2.0, None)
        Hk_smooth = Hk_mag_s * np.exp(1j*np.angle(Hk))

        frames = segment.reshape(nblk, Nfft+Ncp)

        # извлекаем поднесущие и компенсируем канал
        rx_subc = [np.fft.fft(frm[Ncp:])[subc_inds] / Hk_smooth for frm in frames]
        rx_subc = np.concatenate(rx_subc)

        # компенсируем фазирование
        if subc_phases is not None and np.any(subc_phases != 0):
            reps = nblk
            phases_rep = np.tile(subc_phases, reps)
            rx_syms = rx_subc * np.exp(-1j * phases_rep)
        else:
            rx_syms = rx_subc

        # демаппинг и RS-декодирование
        all_rx_bits = qpsk_demap(rx_syms)
        codeword_len_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
        total_cw_bits = codeword_len_bits * nblk
        rx_bits = all_rx_bits[:total_cw_bits]

        decoded = []
        rs_statuses = []
        print("=== RS-декодирование по блокам ===")
        for i in range(nblk):
            bts = bits_to_bytes(rx_bits[i*codeword_len_bits:(i+1)*codeword_len_bits])
            try:
                msg = rs.decode(bts)[0]
                status = "OK"
            except ReedSolomonError:
                msg = b'\x00' * RS_DATA_BYTES
                status = "ERR"
            print(f"Block {i+1}/{nblk}: RS decode = {status}")
            rs_statuses.append(status)
            decoded.append(msg)

        all_bytes = b"".join(decoded).rstrip(b"\x00")

        # парсинг header
        pos = 0
        if len(all_bytes) < 9:
            print("[RX] Недостаточно данных после RS декодирования")
            sys.exit(1)
        mode_rx = all_bytes[pos:pos+1]; pos += 1
        total_sz = struct.unpack(">Q", all_bytes[pos:pos+8])[0]; pos += 8

        if mode_rx == b"F":
            name_len = all_bytes[pos]; pos += 1
            fname = all_bytes[pos:pos+name_len].decode("utf-8",errors="ignore")
            pos += name_len
            body = all_bytes[pos:pos+total_sz]
            out_path = "rx_" + fname
            with open(out_path,"wb") as f: f.write(body)
            print(f"[RX FILE] Сохранён файл: {out_path} ({len(body)} байт)")
        else:
            text_bytes = all_bytes[pos:pos+total_sz]
            rec_text = text_bytes.decode("utf-8",errors="ignore")
            print("[RX TEXT]", rec_text)

        # BER если были data_bits в памяти (только в TX режиме)
        if 'data_bits' in locals():
            tx_bits = data_bits[:total_cw_bits]
            bit_errs = np.count_nonzero(tx_bits[:len(rx_bits)] != rx_bits[:len(tx_bits)])
            ber = bit_errs / len(tx_bits) if len(tx_bits)>0 else 0
            print(f"[BER] Bit errors: {bit_errs}/{len(tx_bits)} (BER = {ber:.2%})")
        else:
            print("[BER] пропущено (нет TX-сессии для сравнения)")

        # Визуализация
        plot_constellation(rx_syms, "RX Constellation (eq.)")
        freqs = subc_inds * fs / Nfft
        eq_gain = 1.0/np.abs(Hk_smooth)
        plt.figure(figsize=(8,4))
        plt.plot(freqs, eq_gain, "o-", color="darkorange", label="Gain")
        plt.title("Equalizer Gain vs Frequency")
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Gain")
        plt.grid(True); plt.legend(); plt.show()

