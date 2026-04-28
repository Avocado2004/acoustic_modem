#!/usr/bin/env python3
"""
OFDM Acoustic Modem с фазированием подписанных и реализацией методов Habr для умeньшения CREST.
Фиксированный 64-байтовый заголовок.
    
Build 64-byte header with layout:
0: version/reserved (1 byte)
1: mode (1 byte) b'T' or b'F' or b'\x00'
2..9: data length (8 bytes, big-endian unsigned)
10..13: filename length (4 bytes, big-endian unsigned)
14..45: filename (32 bytes, utf-8), padded/truncated
46..49: packet number (4 bytes, big-endian unsigned)
50..63: reserved (14 bytes), zero-filled

Note: if mode == b'T' (text), all unused fields (filename/name_len/packet_no) are zero-filled.
    
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

# подсеть в диапазоне 300..4300 Гц
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

# Habr params
HABR_SAMPLE_BLOCKS = 200
HABR_PHASE_GRID = 36
HABR_MAX_ITERS = 2
HABR_SEED = 12345
HABR_SAVE_FILE = "habr_phases.npy"

# инициализация фаз
subc_phases = None

# -----------------------
# Утилиты по работе с битами/текстом
# -----------------------
def text_to_bits(text, encoding='utf-8'):
  bs = ''.join(f"{b:08b}" for b in text.encode(encoding))
  return np.array(list(bs), dtype=int)

def bits_to_text(bits, encoding='utf-8'):
  L = (len(bits) // 8) * 8
  b = bits[:L].reshape(-1, 8)
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
  '00': (1 + 1j) / np.sqrt(2),
  '01': (-1 + 1j) / np.sqrt(2),
  '11': (-1 - 1j) / np.sqrt(2),
  '10': (1 - 1j) / np.sqrt(2),
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
# Функции фазирования
# -----------------------
def make_subcarrier_phases(method=None, N=Nsub, seed=0, habr_phases=None):
  """
  Возвращает массив фаз длины N.
  method: None, "random", "schroeder", "habr"
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
  """ Формирует OFDM временной символ с Hermitian-симметрией и CP.
  data_syms: комплексные символы длины <= Nsub
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
  pilot = np.full(Nsub, (1 + 1j) / np.sqrt(2))
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
# Habr optimizer (unchanged)
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

# инициализация фаз
init_phases()

# -----------------------
# Header: сборка и разбор 64-байтового заголовка
# -----------------------

def build_header(mode: bytes, data_len: int, filename_bytes: bytes = b'', packet_no: int = 0, version: int = 0) -> bytes:
    """
    Build 64-byte header with layout:
    0: version/reserved (1 byte)
    1: mode (1 byte) b'T' or b'F' or b'\x00'
    2..9: data length (8 bytes, big-endian unsigned)
    10..13: filename length (4 bytes, big-endian unsigned)
    14..45: filename (32 bytes, utf-8), padded/truncated
    46..49: packet number (4 bytes, big-endian unsigned)
    50..63: reserved (14 bytes), zero-filled

    Note: if mode == b'T' (text), all unused fields (filename/name_len/packet_no) are zero-filled.
    """
    # version byte first (byte 0)
    ver_b = struct.pack('B', int(version) & 0xFF)

    # mode byte (byte 1)
    if not isinstance(mode, (bytes, bytearray)) or len(mode) == 0:
        mode_b = b'\x00'
    else:
        mode_b = mode[:1]

    # For text mode ensure filename and packet_no are zeroed
    is_text = (mode_b == b'T')

    # data length: 8 bytes BE (byte 2..9)
    data_len_b = struct.pack('>Q', int(data_len) & 0xFFFFFFFFFFFFFFFF)

    # filename handling
    if is_text:
        name_len = 0
        name_field = b'\x00' * 32
        name_len_b = struct.pack('>I', 0)
    else:
        fname = filename_bytes if filename_bytes is not None else b''
        name_len = len(fname)
        name_field = (fname[:32]).ljust(32, b'\x00')
        name_len_b = struct.pack('>I', int(name_len) & 0xFFFFFFFF)

    # packet number (4 bytes BE at 46..49). Zero for text mode.
    pkt = 0 if is_text else int(packet_no) & 0xFFFFFFFF
    packet_b = struct.pack('>I', int(pkt))

    # reserved tail to fill to 64 bytes (50..63 -> 14 bytes)
    tail_reserved = b'\x00' * 14

    header = ver_b + mode_b + data_len_b + name_len_b + name_field + packet_b + tail_reserved
    assert len(header) == 64, f"header length {len(header)} != 64"
    return header

def parse_header(header64: bytes):
    """
    Parse 64-byte header produced by build_header.
    Returns dict with keys: version (int), mode (bytes), data_len (int), name_len (int), filename (str), packet_no (int).
    """
    if len(header64) != 64:
        raise ValueError(f"Header must be 64 bytes, got {len(header64)}")
    version = int(header64[0])
    mode = header64[1:2]
    data_len = struct.unpack('>Q', header64[2:10])[0]
    name_len = struct.unpack('>I', header64[10:14])[0]
    name_bytes = header64[14:46]
    packet_no = struct.unpack('>I', header64[46:50])[0]

    # If mode indicates text, ensure we treat filename/packet as empty/zero
    if mode == b'T':
        filename = ''
        name_len = 0
        packet_no = 0
    else:
        if name_len > 0:
            flen = min(name_len, 32)
            filename = name_bytes[:flen].rstrip(b'\x00').decode('utf-8', errors='ignore')
        else:
            filename = ''
    return {
        'version': version,
        'mode': mode,
        'data_len': data_len,
        'name_len': name_len,
        'filename': filename,
        'packet_no': packet_no
    }

# -----------------------
# Визуализация созвездия (гарантированно доступно)
# -----------------------
def plot_constellation(symb, title="Constellation"):
  """
  Показывает созвездие комплексных символов.
  symb: массив комплексных символов
  """
  try:
    plt.figure(figsize=(5,5))
    plt.plot(np.real(symb), np.imag(symb), 'o', markersize=2, alpha=0.6)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.axvline(0, color='grey', linewidth=0.5)
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis('equal')
    plt.show()
  except Exception as e:
    print("[PLOT] cannot show constellation:", e)
    
# --- START: soft clipping utilities for target crest factor
TARGET_CREST_DB = 4.5
CREST_TOLERANCE_DB = 0.01  # точность подбора (можно увеличить для быстродействия)
MAX_SOFTCLIP_ITERS = 40

def softclip_tanh(x, g):
    """
    Параметр g управляет "жёсткостью" клиппинга.
    Для g->0 сигнал ~ x (нет клиппинга), при больших g - сильное ограничение.
    Нормируем делением на tanh(g) чтобы пик сигнала сохранял масштабирование единицы.
    """
    if g <= 0:
        return x
    denom = np.tanh(g)
    if denom == 0:
        return x
    return np.tanh(g * x) / denom

def crest_db(sig):
    peak = np.max(np.abs(sig))
    rms = np.sqrt(np.mean(sig**2))
    if rms == 0 or peak == 0:
        return np.inf
    return 20.0 * np.log10(peak / rms)

def apply_softclip_to_target_crest(x, target_db=TARGET_CREST_DB, tol_db=CREST_TOLERANCE_DB, max_iters=MAX_SOFTCLIP_ITERS):
    """
    Ищем параметр g для softclip_tanh так, чтоб crest(sig_after) ~= target_db.
    Ищем по g методом бинарного поиска на лог. шкале: g in [g_min, g_max].
    Начинаем с g_min=1e-6 (почти линейно), g_max увеличиваем пока crest <= target (или пока не достигнем лимита).
    """
    if x.size == 0:
        return x, 0.0

    # нормируем вход для численной стабильности (будем искать g для сигнала с peak=1)
    peak_in = np.max(np.abs(x))
    if peak_in == 0:
        return x, crest_db(x)
    x_norm = x / peak_in

    # если уже подходит без клиппинга (crest <= target), ничего не делаем
    if crest_db(x_norm) <= target_db + tol_db:
        return x, crest_db(x)

    # подготовка поискового интервала
    g_lo = 1e-8
    g_hi = 1.0
    # увеличиваем g_hi пока не достигнем желаемого crest (ограничение жёстче => crest падает)
    for _ in range(30):
        y = softclip_tanh(x_norm, g_hi)
        if crest_db(y) <= target_db:
            break
        g_hi *= 2.0

    # бинпоиск по g в [g_lo, g_hi]
    best_g = g_hi
    for _ in range(max_iters):
        g_mid = 0.5 * (g_lo + g_hi)
        y_mid = softclip_tanh(x_norm, g_mid)
        c_mid = crest_db(y_mid)
        if abs(c_mid - target_db) <= tol_db:
            best_g = g_mid
            break
        # если crest слишком большой => надо сильнее ограничивать => увеличить g
        if c_mid > target_db:
            g_lo = g_mid
        else:
            g_hi = g_mid
        best_g = g_mid

    # применяем найденный параметр к оригинальному масштабу
    y_final = softclip_tanh(x / peak_in, best_g) * peak_in
    final_crest = crest_db(y_final)
    return y_final, final_crest
# --- END: soft clipping utilities


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

    # номер пакета: по умолчанию 0 (можно расширить/инкрементировать при фрагментации)
    packet_index = 0

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
      # build fixed header
      header = build_header(b'F', file_size, filename_bytes=name_bytes, packet_no=packet_index)
      payload = header + file_data
    else:
      text = input("Введите текст для передачи: ")
      text_bytes = text.encode("utf-8")
      header = build_header(b'T', len(text_bytes), filename_bytes=b'', packet_no=packet_index)
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

    # --- Controlled noise in silent regions before saving WAV ---
    # Parameters (tune as needed)
    PREROLL_SEC = 0.25 # length of leading silence (seconds)
    POSTROLL_SEC = 0.10 # length of trailing silence (seconds)
    NOISE_LEVEL = 0.001 # noise RMS relative to tx peak before final scaling (0.0 = no noise)
    NOISE_SEED = 20231107 # deterministic seed for reproducibility

    preroll_len = int(PREROLL_SEC * fs)
    postroll_len = int(POSTROLL_SEC * fs)

    preroll = np.zeros(preroll_len, dtype=tx.dtype)
    postroll = np.zeros(postroll_len, dtype=tx.dtype)

    if NOISE_LEVEL is not None and NOISE_LEVEL > 0.0:
      rnd = np.random.RandomState(NOISE_SEED)
      tx_peak = np.max(np.abs(tx)) if np.max(np.abs(tx)) > 0 else 1.0
      noise_rms = float(NOISE_LEVEL) * float(tx_peak)

      def make_noise(n_samples):
        w = rnd.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float64)
        cur_rms = np.sqrt(np.mean(w**2)) if w.size > 0 else 1.0
        if cur_rms == 0:
          return np.zeros(n_samples, dtype=np.float64)
        w = w * (noise_rms / cur_rms)
        return w

      preroll_noise = make_noise(preroll_len)
      postroll_noise = make_noise(postroll_len)

      preroll = preroll_noise.astype(tx.dtype)
      postroll = postroll_noise.astype(tx.dtype)
      print(f"[NOISE] added noise in preroll/postroll with NOISE_LEVEL={NOISE_LEVEL} (noise_rms={noise_rms:.6f})")
    else:
      print("[NOISE] no noise added (NOISE_LEVEL <= 0)")

    tx_out = np.concatenate((preroll, tx, postroll))

    # сначала применяем soft clipping, целевой crest factor = TARGET_CREST_DB dB
    tx_clipped, achieved_crest = apply_softclip_to_target_crest(tx_out, target_db=TARGET_CREST_DB)
    print(f"[SOFTCLIP] achieved crest = {achieved_crest:.3f} dB (target {TARGET_CREST_DB} dB)")

    # затем масштабируем для int16 (как и раньше, используем 1 headroom)
    max_abs = np.max(np.abs(tx_clipped))
    if max_abs == 0:
        scale_to_int16 = 1.0
    else:
        scale_to_int16 = 1 / max_abs

    tx_int16 = (tx_clipped * scale_to_int16 * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write("ofdm_acoustic_tx_with_noise.wav", fs, tx_int16)
    print("TX saved to ofdm_acoustic_tx_with_noise.wav (with controlled noise in silent regions and soft clipping)")


  else:
    src = input("Откуда демодулировать? (file/mic): ").strip().lower()
    if src == "file":
      wav_path = input("Путь к WAV-файлу для приёма: ").strip()
      _, wavd = wavfile.read(wav_path)
      sig = wavd[:,0] if wavd.ndim > 1 else wavd
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

    rx_subc = [np.fft.fft(frm[Ncp:])[subc_inds] / Hk_smooth for frm in frames]
    rx_subc = np.concatenate(rx_subc)

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

    all_bytes = b"".join(decoded)

    # parsing header (ожидаем минимум 64 байт заголовка)
    pos = 0
    if len(all_bytes) < 64:
      print("[RX] Недостаточно данных после RS декодирования для полноценного заголовка (64 байта)")
      sys.exit(1)

    header64 = all_bytes[pos:pos+64]; pos += 64
    hdr = parse_header(header64)
    mode_rx = hdr['mode']
    total_sz = hdr['data_len']
    fname = hdr['filename']
    name_len = hdr['name_len']
    packet_no = hdr['packet_no']

    # Ensure we don’t read beyond available bytes
    available = max(0, len(all_bytes) - pos)
    to_read = min(available, total_sz)

    if mode_rx == b"F":
      body = all_bytes[pos:pos+to_read]
      out_fname = fname if fname else "rx_file"
      out_path = "rx_" + out_fname
      # write exactly the bytes present (to_read)
      with open(out_path, "wb") as f:
        f.write(body)
      print(f"[RX FILE] Сохранён файл: {out_path} ({len(body)} байт) packet_no={packet_no}")
    else:
      text_bytes = all_bytes[pos:pos+to_read]
      rec_text = text_bytes.decode("utf-8", errors="ignore")
      print("[RX TEXT]", rec_text)

    # BER если были данные tx
    if 'data_bits' in locals():
      tx_bits = data_bits[:total_cw_bits]
      bit_errs = np.count_nonzero(tx_bits[:len(rx_bits)] != rx_bits[:len(tx_bits)])
      ber = bit_errs / len(tx_bits) if len(tx_bits) > 0 else 0
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


