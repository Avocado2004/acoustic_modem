#!/usr/bin/env python3
"""
OFDM Acoustic Modem — модификация для пакетной передачи:
- пакет = 64 OFDM символа данных
- каждый пакет имеет свою преамбулу и заголовок (включая номер пакета)
- пауза между пакетами = PACKET_GAP_SYMS (4 символа)
- чтение с микрофона кольцевым буфером длиной 1 пакет с зазором 1 символ спереди/сзади
- минимальные правки по сравнению с оригиналом
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
fs = 48000 # sample rate
Nfft = 512 # FFT size
Ncp = 128 # cyclic prefix
df = fs / Nfft

# частотный диапазон (поднесущие)
k_low = int(math.ceil(300 / df))
k_high = int(math.floor(4300 / df))
subc_inds = np.arange(k_low, k_high + 1)
Nsub = len(subc_inds)

# FEC (Reed-Solomon)
RS_DATA_BYTES = 8
RS_PARITY_BYTES = 4
rs = RSCodec(RS_PARITY_BYTES)

MAX_PAYLOAD_SIZE = 1 << 30  # оставлено большое, оригинал выставлял 1<<30

# PHASE METHOD
PHASE_METHOD = "schroeder"

# Habr params (оставлены как раньше)
HABR_SAMPLE_BLOCKS = 200
HABR_PHASE_GRID = 36
HABR_MAX_ITERS = 2
HABR_SEED = 12345
HABR_SAVE_FILE = "habr_phases.npy"

subc_phases = None

# -----------------------
# Пакетизация (новые параметры)
# -----------------------
# число OFDM-символов в пакете данных
PKT_DATA_SYMS = 64

# пауза между пакетами в символах (4 по заданию)
PACKET_GAP_SYMS = 4

# -----------------------
# Функции конвертации
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

# QPSK mapping / demapping
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

def crest_factor(sig):
    peak = np.max(np.abs(sig))
    rms = np.sqrt(np.mean(sig**2))
    if rms == 0:
        return np.inf
    return 20 * np.log10(peak / rms)

# -----------------------
# Фазы поднесущих и OFDM
# -----------------------
def make_subcarrier_phases(method=None, N=Nsub, seed=0, habr_phases=None):
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
        k = np.arange(N)
        return np.mod(np.pi * k * (k - 1) / float(N), 2*np.pi)
    return np.zeros(N)

def ofdm_symbol(data_syms):
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
# Habr и инициализация фаз
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
    if sample_blocks is None:
        sample_blocks = make_training_blocks()
    M = sample_blocks.shape[0]
    phases = make_subcarrier_phases("schroeder", Nsub)
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
        train_blocks = make_training_blocks(n_blocks=HABR_SAMPLE_BLOCKS, seed=HABR_SEED)
        ph, c = optimize_phases_habr(sample_blocks=train_blocks, n_iter=HABR_MAX_ITERS, grid_size=HABR_PHASE_GRID, seed=HABR_SEED, verbose=True)
        subc_phases = ph
        try:
            np.save(HABR_SAVE_FILE, subc_phases)
            print(f"[HABR] saved optimized phases to {HABR_SAVE_FILE} (crest={c:.3f} dB)")
        except Exception as e:
            print("[HABR] failed save:", e)
    else:
        subc_phases = make_subcarrier_phases(PHASE_METHOD, Nsub, seed=HABR_SEED)
        print(f"[PHASE] method={PHASE_METHOD}, example degs[:8]={np.degrees(subc_phases[:8])}")

init_phases()

# -----------------------
# Синхронизация
# -----------------------
def sync_by_corr(rx, pre):
    corr = fftconvolve(rx, pre[::-1], mode='valid')
    return int(np.argmax(np.abs(corr)))

# -----------------------
# Вспомогательные: упаковка/распаковка заголовка пакета
# -----------------------
def build_packet_header(mode_byte: bytes, total_size: int, packet_index: int, filename_bytes: bytes = b''):
    # mode_byte: b'F' или b'T'
    # total_size: общий размер всего файла/текста в байтах
    # packet_index: индекс текущего пакета (64-bit unsigned)
    hdr = mode_byte + struct.pack(">Q", total_size) + struct.pack(">Q", packet_index)
    if mode_byte == b'F':
        name_bytes = filename_bytes
        if len(name_bytes) > 255:
            raise ValueError("File name too long")
        hdr += struct.pack("B", len(name_bytes)) + name_bytes
    else:
        pass
    return hdr

def parse_rx_header(all_bytes):
    pos = 0
    if len(all_bytes) < 17:
        raise ValueError("Header too short")
    mode_rx = all_bytes[pos:pos+1]; pos += 1
    total_sz = struct.unpack(">Q", all_bytes[pos:pos+8])[0]; pos += 8
    packet_index = struct.unpack(">Q", all_bytes[pos:pos+8])[0]; pos += 8
    name_len = 0
    fname = ""
    if mode_rx == b'F':
        name_len = all_bytes[pos]; pos += 1
        fname = all_bytes[pos:pos+name_len].decode("utf-8", errors="ignore")
        pos += name_len
    return mode_rx, total_sz, packet_index, fname, pos

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
                print("Файл слишком большой.")
                sys.exit(1)
            filename = os.path.basename(file_path)
            name_bytes = filename.encode("utf-8")
            with open(file_path, "rb") as f:
                file_data = f.read()
            # payload = header + file_data, но мы будем сегментировать на пакеты
            top_header = b"F" + struct.pack(">Q", file_size) + struct.pack(">Q", 0) + struct.pack("B", len(name_bytes)) + name_bytes
            global_payload = top_header + file_data
        else:
            text = input("Введите текст для передачи: ")
            text_bytes = text.encode("utf-8")
            top_header = b"T" + struct.pack(">Q", len(text_bytes)) + struct.pack(">Q", 0)
            global_payload = top_header + text_bytes

        # RS encode per RS_DATA_BYTES and then packetize into PKT_DATA_SYMS symbols per packet
        data_bytes = global_payload
        blocks = [data_bytes[i:i+RS_DATA_BYTES] for i in range(0, len(data_bytes), RS_DATA_BYTES)]
        if len(blocks[-1]) < RS_DATA_BYTES:
            blocks[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks[-1]))
        encoded_blocks = [rs.encode(b) for b in blocks]
        bits_blocks = [bytes_to_bits(b) for b in encoded_blocks]
        data_bits = np.concatenate(bits_blocks)
        # layout bits into OFDM symbols: Nsub QPSK syms per symbol -> Nsub*2 bits per OFDM symbol
        bits_per_symbol = Nsub * 2
        # количество OFDM символов всего:
        total_ofdm_syms = int(np.ceil(len(data_bits) / bits_per_symbol))
        # количество пакетов
        syms_per_packet = PKT_DATA_SYMS
        packets_count = int(np.ceil(total_ofdm_syms / syms_per_packet))
        print(f"[TX] total OFDM symbols: {total_ofdm_syms}, packets: {packets_count}")

        # разбиваем по пакетам (по символам)
        tx_signal_parts = []
        bit_ptr = 0

        # фиксированная длина поля номера пакета в битах (8 байт = 64 бита)
        pkt_index_bits_len = 8 * 8

        for pkt_idx in range(packets_count):
            needed_bits = syms_per_packet * bits_per_symbol
            chunk = data_bits[bit_ptr:bit_ptr+needed_bits]
            if len(chunk) < needed_bits:
                chunk = np.concatenate((chunk, np.zeros(needed_bits - len(chunk), dtype=int)))

            # Записываем фиксированное 8-байтное поле номера пакета в начало каждого пакета в битовом виде
            idx_bytes = struct.pack(">Q", pkt_idx)
            idx_bits = bytes_to_bits(idx_bytes)
            if len(idx_bits) != pkt_index_bits_len:
                raise RuntimeError("Unexpected pkt_index_bits_len mismatch")
            chunk[:pkt_index_bits_len] = idx_bits

            bit_ptr += needed_bits
            data_td, _ = build_data_td(chunk)
            # пакет = преамбула + data_td
            pkt_td = np.concatenate((preamble_td, data_td))
            tx_signal_parts.append(pkt_td)
            # после каждого пакета вставляем паузу в виде нулевых сэмплов длиной PACKET_GAP_SYMS символов
            gap_samples = PACKET_GAP_SYMS * (Nfft + Ncp)
            if PACKET_GAP_SYMS > 0:
                tx_signal_parts.append(np.zeros(gap_samples, dtype=float))

        tx = np.concatenate(tx_signal_parts)
        cf_before = crest_factor(tx)
        tx *= 0.9 / np.max(np.abs(tx))
        cf_after = crest_factor(tx)
        print(f"[CREST] before norm: {cf_before:.2f} dB, after norm: {cf_after:.2f} dB")
        preroll = np.zeros(int(0.25*fs), dtype=tx.dtype)
        tx_out = np.concatenate((preroll, tx))
        wavfile.write("ofdm_acoustic_tx_packets.wav", fs, (tx_out * np.iinfo(np.int16).max).astype(np.int16))
        print("TX saved to ofdm_acoustic_tx_packets.wav")

    else:
        # Receive mode (значительная часть переработана для пакетного приема)
        src = input("Откуда демодулировать? (file/mic): ").strip().lower()
        if src == "file":
            wav_path = input("Путь к WAV-файлу для приёма: ").strip()
            _, wavd = wavfile.read(wav_path)
            sig = wavd[:,0] if wavd.ndim>1 else wavd
            rx = sig.astype(float) / np.iinfo(wavd.dtype).max
        else:
            print("Запись с микрофона. Нажмите Ctrl+C чтобы остановить.")
            # Для приема делаем кольцевой буфер длиной 1 пакет + 1 символ спереди/сзади
            sym_len = Nfft + Ncp
            pkt_samples = PKT_DATA_SYMS * sym_len
            margin = sym_len  # 1 символ спереди и сзади -> мы будем сдвигать, поэтому возьмём margin = 1*sym_len
            ring_len = pkt_samples + 2*margin
            rec_seconds = int(np.ceil(ring_len / fs))  # минимум 1 сек
            rec = sd.rec(int((rec_seconds+1)*fs), samplerate=fs, channels=1)
            sd.wait()
            rx = rec[:,0].astype(float)
            if np.max(np.abs(rx)) > 0:
                rx /= np.max(np.abs(rx))

        # синхронизация по преамбуле: будем искать в потоке начало каждого пакета используя correlation
        pre = preamble_td
        sym_len = Nfft + Ncp

        # функция попытки извлечь один пакет от произвольного положения в rx
        def try_extract_packet_stream(rx_stream):
            """
            Попытаться найти и извлечь ОДИН пакет (pre + PKT_DATA_SYMS symbols) из rx_stream.
            Возвращает (packet_bytes_or_None, packet_index_estimated_or_None, start_sample) или (None,None,None) если не найден.
            """
            # корреляция по преамбуле
            idx = sync_by_corr(rx_stream, pre)
            if idx is None:
                return None, None, None
            # убедимся, что после найденной позиции есть место на весь пакет
            start = idx
            needed = len(pre) + PKT_DATA_SYMS * sym_len
            if start + needed > len(rx_stream):
                return None, None, None
            # извлекаем сегмент, пропускаем преамбулу
            data_segment = rx_stream[start + len(pre): start + needed]
            # разбиваем на символы
            nblk = PKT_DATA_SYMS * (sym_len) // sym_len
            frames = data_segment.reshape(nblk, sym_len)
            # простейшее эквалайзное оценивание: берём первые два OFDM (как ранее) для Hk
            ideal_sym = pre[Ncp : Ncp + Nfft]
            rx_pre1 = rx_stream[start + Ncp : start + Ncp + Nfft]
            rx_pre2 = rx_stream[start + sym_len + Ncp : start + sym_len + Ncp + Nfft]
            S_ref = np.fft.fft(ideal_sym)
            R1, R2 = np.fft.fft(rx_pre1), np.fft.fft(rx_pre2)
            Hk = ((R1[subc_inds]/S_ref[subc_inds]) + (R2[subc_inds]/S_ref[subc_inds]))/2
            Hk_mag_s = np.clip(medfilt(np.abs(Hk), 5), 1/2.0, None)
            Hk_smooth = Hk_mag_s * np.exp(1j*np.angle(Hk))
            rx_subc = [np.fft.fft(frm[Ncp:])[subc_inds] / Hk_smooth for frm in frames]
            rx_subc = np.concatenate(rx_subc)
            if subc_phases is not None and np.any(subc_phases != 0):
                phases_rep = np.tile(subc_phases, PKT_DATA_SYMS)
                rx_syms = rx_subc * np.exp(-1j * phases_rep)
            else:
                rx_syms = rx_subc
            all_rx_bits = qpsk_demap(rx_syms)

            # --- Новый: извлекаем номер пакета из первых 64 демодулированных бит (до RS-decoding) ---
            pkt_index_bits_len = 8 * 8
            if len(all_rx_bits) < pkt_index_bits_len:
                return None, None, None
            idx_bits = all_rx_bits[:pkt_index_bits_len]
            # преобразуем биты в байты (big-endian по байту)
            try:
                idx_bytes = bits_to_bytes(idx_bits)
                # Если bits_to_bytes вернул не 8 байт (маленькая длина), приводим/дополняем
                if len(idx_bytes) < 8:
                    idx_bytes = idx_bytes.ljust(8, b'\x00')
                pkt_index = struct.unpack(">Q", idx_bytes[:8])[0]
            except Exception:
                return None, None, None
            # ------------------------------------------------------------------------------

            # RS decode по блокам (оставляем как было)
            codeword_len_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
            total_cw_bits = codeword_len_bits * ( (len(all_rx_bits) // codeword_len_bits) )
            rx_bits = all_rx_bits[:total_cw_bits]
            decoded = []
            for i in range(0, total_cw_bits, codeword_len_bits):
                bts = bits_to_bytes(rx_bits[i:i+codeword_len_bits])
                try:
                    msg = rs.decode(bts)[0]
                except ReedSolomonError:
                    msg = b'\x00' * RS_DATA_BYTES
                decoded.append(msg)
            # собираем декодированные байты без удаления нулевого паддинга — нужно, чтобы первые 8 байт присутствовали
            all_bytes = b"".join(decoded)

            # возвращаем все байты (включая декодированные данные) и pkt_index, start
            return all_bytes, pkt_index, start

        # В режиме приёма — непрерывно читаем ring buffer и пытаемся восстановить пакеты пока не соберём все данные
        recovered_packets = {}
        total_expected_size = None
        filename = None
        mode_rx_global = None

        pos = 0
        max_tries = 5000
        tries = 0
        last_abs_start = None
        while True:
            pkt_bytes, pkt_idx, start_sample = try_extract_packet_stream(rx[pos:])
            tries += 1

            if pkt_bytes is None:
                # сдвиг на один OFDM символ, чтобы корреляция при следующем проходе не возвращала ту же позицию
                pos += sym_len
                if pos + len(preamble_td) >= len(rx):
                    pos = 0
                if tries > max_tries:
                    print("[RX] слишком много попыток, прерываю приём")
                    break
                continue

            # абсолютный индекс начала пакета в rx
            abs_start = pos + start_sample

            # защититься от повторного детекта той же преамбулы (слишком близко к предыдущему)
            if last_abs_start is not None and abs(abs_start - last_abs_start) < sym_len:
                # сдвинем указатель за найденный кусок и продолжим
                pos = (abs_start + sym_len) % max(1, len(rx))
                if pos + len(preamble_td) >= len(rx):
                    pos = 0
                # не учитываем как новая попытка получения пакета
                if tries > max_tries:
                    print("[RX] лимит попыток исчерпан, выходим")
                    break
                continue

            last_abs_start = abs_start

            # ДЕЛАЕМ МЕНЬШЕЕ СМЕЩЕНИЕ: сдвигаем pos только на один символ (sym_len),
            # чтобы не пропускать промежуточные пакеты (альтернатива полному пропуску пакета+gap).
            pos = abs_start + sym_len
            if pos + len(preamble_td) >= len(rx):
                pos = 0

            # распознавание первого заголовка чтобы узнать total_size и имя
            try:
                # parse_rx_header ожидает header, который в вашем формировании лежит в начале payload.
                mode_rx, total_sz, pkt_index_read, fname, hdr_len = parse_rx_header(pkt_bytes)
            except Exception:
                # если header не распарсился — продолжаем
                continue
            if total_expected_size is None:
                total_expected_size = total_sz
                mode_rx_global = mode_rx
                filename = fname
                print(f"[RX] ожидаемый общий размер: {total_expected_size} байт, режим: {mode_rx}, fname: {filename}")
            # сохраняем пакет по индексу (если ещё не было)
            if pkt_idx not in recovered_packets:
                recovered_packets[pkt_idx] = pkt_bytes
                print(f"[RX] получен пакет #{pkt_idx}, байт в блоке: {len(pkt_bytes)}")
            else:
                print(f"[RX] пакет #{pkt_idx} уже получен, пропускаем")
            # проверяем, собрали ли все пакеты: оценим по числу байт: ожидаемый total_expected_size + заголовок и RS padding
            if 0 in recovered_packets:
                parts = []
                i = 0
                while i in recovered_packets:
                    parts.append(recovered_packets[i])
                    i += 1
                assembled = b"".join(parts)
                # пробуем извлечь глобальный header и тело
                try:
                    mode_rx2, total_sz2, pkt_index2, fname2, hdr_len2 = parse_rx_header(assembled)
                    body = assembled[hdr_len2:hdr_len2 + total_sz2]
                except Exception:
                    body = b''
                if total_expected_size is not None and len(body) >= total_expected_size:
                    if mode_rx_global == b'F':
                        out_path = "rx_" + (filename if filename else "out.bin")
                        with open(out_path, "wb") as f:
                            f.write(body[:total_expected_size])
                        print(f"[RX FILE] Сохранён файл: {out_path} ({len(body[:total_expected_size])} байт)")
                    else:
                        rec_text = body[:total_expected_size].decode("utf-8", errors="ignore")
                        print("[RX TEXT]", rec_text)
                    break

            if tries > max_tries:
                print("[RX] лимит попыток исчерпан, выходим")
                break

        print("[RX] приём завершён (или прерван)")

    # конец main


