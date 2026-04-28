#!/usr/bin/env python3
"""
OFDM Acoustic Modem — обновлённый: per-packet RX декодирование и packet_blocks в transmission header.
TX: каждый пакет содержит ровно DEFAULT_PACKET_BLOCKS OFDM блоков; при необходимости последний пакет дополняется нулями.
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
fs = 48000
Nfft = 512
Ncp = 128
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
PHASE_METHOD = "schroeder"

# Habr params (unchanged)
HABR_SAMPLE_BLOCKS = 200
HABR_PHASE_GRID = 36
HABR_MAX_ITERS = 2
HABR_SEED = 12345
HABR_SAVE_FILE = "habr_phases.npy"

subc_phases = None

# Globals shared TX/RX
DEFAULT_PACKET_BLOCKS = 75  # default OFDM blocks per packet
GAP_OFDM_SYMBOLS = 2
SYMBOL_LEN = Nfft + Ncp
GAP_SAMPLES_DEFAULT = GAP_OFDM_SYMBOLS * SYMBOL_LEN

# -----------------------
# Биты/текст/битовые утилиты
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

# QPSK map/demap (unchanged)
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

# crest utilities
def crest_factor(sig):
  peak = np.max(np.abs(sig))
  rms = np.sqrt(np.mean(sig**2))
  if rms == 0:
    return np.inf
  return 20 * np.log10(peak / rms)

# -----------------------
# Фазирование
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

# -----------------------
# OFDM symbol build
# -----------------------
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
# sync helper
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

init_phases()

# -----------------------
# Header: updated format (Transmission header contains packet_blocks at bytes 50..51)
# -----------------------
def build_header(mode: bytes, data_len: int, filename_bytes: bytes = b'', packet_no: int = 0, version: int = 0, packet_blocks: int = DEFAULT_PACKET_BLOCKS) -> bytes:
    """
    Build 64-byte Transmission header (MSB of first byte =1).
    We place packet_blocks as uint16 big-endian at hdr[50:52].
    """
    if isinstance(mode, (bytes, bytearray)) and len(mode) > 0 and mode[:1] == b'F':
        tx_type = 0b11
    elif isinstance(mode, (bytes, bytearray)) and len(mode) > 0 and mode[:1] == b'T':
        tx_type = 0b10
    else:
        tx_type = 0b00

    flags = 0x80 | (tx_type & 0x03)
    hdr = bytearray(64)
    hdr[0] = flags & 0xFF
    hdr[1] = int(version) & 0xFF
    hdr[2:10] = struct.pack('>Q', int(data_len) & 0xFFFFFFFFFFFFFFFF)

    if isinstance(mode, (bytes, bytearray)) and mode[:1] == b'T':
        fname_b = b''
        name_len = 0
    else:
        fname_b = filename_bytes if filename_bytes is not None else b''
        name_len = len(fname_b)
    hdr[10:14] = struct.pack('>I', int(name_len) & 0xFFFFFFFF)
    fname_field = (fname_b[:32]).ljust(32, b'\x00')
    hdr[14:14+32] = fname_field
    hdr[46:50] = struct.pack('>I', int(packet_no) & 0xFFFFFFFF)
    # packet_blocks stored at 50..51 (2 bytes)
    hdr[50:52] = struct.pack('>H', int(packet_blocks) & 0xFFFF)
    # remaining reserved bytes
    hdr[52:64] = b'\x00' * 12
    return bytes(hdr)

def make_packet_header_bytes(packet_no: int, tx_type: int) -> bytes:
    flags = (0x00) | (tx_type & 0x03)
    hdr = bytearray(4)
    hdr[0] = flags & 0xFF
    hdr[1] = (packet_no >> 16) & 0xFF
    hdr[2] = (packet_no >> 8) & 0xFF
    hdr[3] = packet_no & 0xFF
    return bytes(hdr)

def parse_header(header64: bytes):
    if len(header64) < 1:
        raise ValueError("Empty header data")
    b0 = header64[0]
    is_transmission = bool(b0 & 0x80)
    tx_type = b0 & 0x03
    if is_transmission:
        if len(header64) < 64:
            raise ValueError("Transmission header requires 64 bytes")
        version = int(header64[1])
        data_len = struct.unpack('>Q', header64[2:10])[0]
        name_len = struct.unpack('>I', header64[10:14])[0]
        name_bytes = header64[14:46]
        packet_no = struct.unpack('>I', header64[46:50])[0]
        # read packet_blocks from 50..51
        packet_blocks = struct.unpack('>H', header64[50:52])[0]
        if tx_type == 0b10:
            filename = ''
            name_len = 0
            packet_no = 0
        else:
            if name_len > 0:
                flen = min(name_len, 32)
                filename = name_bytes[:flen].rstrip(b'\x00').decode('utf-8', errors='ignore')
            else:
                filename = ''
        mode = b'T' if tx_type == 0b10 else (b'F' if tx_type == 0b11 else b'\x00')
        return {
            'is_transmission': True,
            'version': version,
            'tx_type': tx_type,
            'mode': mode,
            'data_len': data_len,
            'name_len': name_len,
            'filename': filename,
            'packet_no': packet_no,
            'packet_blocks': packet_blocks
        }
    else:
        if len(header64) < 4:
            raise ValueError("Packet header requires 4 bytes")
        packet_no = (header64[1] << 16) | (header64[2] << 8) | header64[3]
        mode = b'\x00'
        return {
            'is_transmission': False,
            'tx_type': tx_type,
            'mode': mode,
            'packet_no': packet_no,
            'header_size': 4
        }

# -----------------------
# bytes -> OFDM helper (module level)
# -----------------------
def bytes_to_ofdm_blocks_bytes(bstream: bytes):
    blocks_local = [bstream[i:i+RS_DATA_BYTES] for i in range(0, len(bstream), RS_DATA_BYTES)]
    if len(blocks_local[-1]) < RS_DATA_BYTES:
        blocks_local[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks_local[-1]))
    encoded = [rs.encode(b) for b in blocks_local]
    bits_blocks_local = [bytes_to_bits(b) for b in encoded]
    data_bits_local = np.concatenate(bits_blocks_local) if len(bits_blocks_local) > 0 else np.array([], dtype=int)
    td_local, nblocks_local = build_data_td(data_bits_local) if data_bits_local.size > 0 else (np.array([], dtype=float), 0)
    return td_local, nblocks_local

# -----------------------
# simulate positions (uses fixed packet_blocks)
# -----------------------
def simulate_packet_positions(total_data_len_bytes, filename_bytes, mode_is_text, packet_blocks_local,
                              preamble_len, symbol_len, gap_samples):
    """
    Returns list of packet preamble offsets (relative to concatenated packets start, without preroll).
    Uses fixed packet_blocks_local (number of OFDM blocks per packet).
    This version limits the hi bound during payload search to avoid expensive loops when total_data_len is small.
    """
    positions = []
    # For first packet header is 64, subsequent 4; but payload packing depends on RS blocks.
    # We use conservative model: compute bytes per packet by binary search via bytes_to_ofdm_blocks_bytes
    # but since we know packet_blocks_local is fixed, we can compute payload bytes that fit into that many blocks.
    # Find payload bytes per packet for first packet (header 64) and others (header 4)
    def max_payload_for_header(header_bytes):
        step = RS_DATA_BYTES
        # quick return for very small total size
        if total_data_len_bytes <= step:
            td, nblk = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * total_data_len_bytes))
            if nblk <= packet_blocks_local:
                return total_data_len_bytes
            # otherwise return 0
            return 0

        # set a safe upper bound for hi to avoid exponential blowup.
        # Rough heuristic: each RS data byte turns into ~ ( (RS_DATA_BYTES+RS_PARITY_BYTES) * 8 ) / (2*Nsub) bytes of payload per OFDM block
        # but to be conservative we choose upper bound = min(total_data_len_bytes, packet_blocks_local * RS_DATA_BYTES * 8)
        # this ensures hi cannot grow extremely large for small total_data_len_bytes.
        hi_bound = min(total_data_len_bytes, packet_blocks_local * RS_DATA_BYTES * 8)
        lo = 0
        hi = step if step < hi_bound else hi_bound

        # exponential growth but capped by hi_bound
        while True:
            td, nblk = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * hi))
            if nblk > packet_blocks_local:
                break
            if hi >= hi_bound:
                break
            hi = min(hi * 2, hi_bound)

        # binary search in steps of RS_DATA_BYTES within bounded range
        best = 0
        lo = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            mid -= (mid % step)
            td, nblk = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * mid))
            if nblk <= packet_blocks_local:
                best = mid
                lo = mid + step
            else:
                hi = mid - step
        return best

    tx_type_bits = 0b10 if mode_is_text else 0b11
    hdr_first = build_header(b'T' if mode_is_text else b'F', total_data_len_bytes, filename_bytes=filename_bytes, packet_no=0, version=0, packet_blocks=packet_blocks_local)
    remaining = total_data_len_bytes
    offset = 0
    pkt_no = 0

    # first
    best_first = max_payload_for_header(hdr_first)
    payload = min(remaining, best_first)
    td, nblk = bytes_to_ofdm_blocks_bytes(hdr_first + (b'\x00' * payload))
    packet_samples = preamble_len + nblk * symbol_len + gap_samples
    positions.append(offset)
    offset += packet_samples
    remaining -= payload
    pkt_no += 1

    # subsequent
    while remaining > 0:
        hdr = make_packet_header_bytes(pkt_no, tx_type_bits)
        best = max_payload_for_header(hdr)
        payload = min(remaining, best)
        td, nblk = bytes_to_ofdm_blocks_bytes(hdr + (b'\x00' * payload))
        packet_samples = preamble_len + nblk * symbol_len + gap_samples
        positions.append(offset)
        offset += packet_samples
        remaining -= payload
        pkt_no += 1
        if pkt_no > 1000000:
            break

    return positions

# -----------------------
# Visualization helper
# -----------------------
def plot_constellation(symb, title="Constellation"):
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

# --- soft clipping utilities (unchanged) ---
TARGET_CREST_DB = 4.5
CREST_TOLERANCE_DB = 0.01
MAX_SOFTCLIP_ITERS = 40

def softclip_tanh(x, g):
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
    if x.size == 0:
        return x, 0.0
    peak_in = np.max(np.abs(x))
    if peak_in == 0:
        return x, crest_db(x)
    x_norm = x / peak_in
    if crest_db(x_norm) <= target_db + tol_db:
        return x, crest_db(x)
    g_lo = 1e-8
    g_hi = 1.0
    for _ in range(30):
        y = softclip_tanh(x_norm, g_hi)
        if crest_db(y) <= target_db:
            break
        g_hi *= 2.0
    best_g = g_hi
    for _ in range(max_iters):
        g_mid = 0.5 * (g_lo + g_hi)
        y_mid = softclip_tanh(x_norm, g_mid)
        c_mid = crest_db(y_mid)
        if abs(c_mid - target_db) <= tol_db:
            best_g = g_mid
            break
        if c_mid > target_db:
            g_lo = g_mid
        else:
            g_hi = g_mid
        best_g = g_mid
    y_final = softclip_tanh(x / peak_in, best_g) * peak_in
    final_crest = crest_db(y_final)
    return y_final, final_crest

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
      total_data_len = file_size
      filename_bytes = name_bytes
    else:
      text = input("Введите текст для передачи: ").strip()
      text_bytes = text.encode("utf-8")
      total_data_len = len(text_bytes)
      filename_bytes = b''
      file_data = text_bytes

    # TX params
    packet_blocks = DEFAULT_PACKET_BLOCKS
    OFDM_BLOCKS_PER_PACKET = packet_blocks
    gap_ofdm_symbols = GAP_OFDM_SYMBOLS
    symbol_len_samples = SYMBOL_LEN
    gap_samples = gap_ofdm_symbols * symbol_len_samples

    PREROLL_SEC = 0.25
    POSTROLL_SEC = 0.50
    NOISE_LEVEL = 0.001
    NOISE_SEED = 20231107

    rng_global = np.random.RandomState(NOISE_SEED)

    tx_type_bits = 0b11 if mode == "F" else (0b10 if mode == "T" else 0b00)
    version = 0
    transmission_header = build_header(b'F' if mode == "F" else b'T', total_data_len, filename_bytes=filename_bytes, packet_no=0, version=version, packet_blocks=packet_blocks)
    remaining_data = file_data

    samples_packets = []
    packet_no = 0
    running_sample_offset = 0
    packet_preamble_offsets_no_preroll = []

    while True:
      is_first = (packet_no == 0)
      header_bytes = transmission_header if is_first else make_packet_header_bytes(packet_no, tx_type_bits)
      allowed_blocks = OFDM_BLOCKS_PER_PACKET
      lo = 0
      hi = len(remaining_data)
      best_sz = 0
      best_td = None
      while lo <= hi:
        mid = (lo + hi) // 2
        test_bytes = header_bytes + remaining_data[:mid]
        td_local, nblocks_local = bytes_to_ofdm_blocks_bytes(test_bytes)
        if nblocks_local <= allowed_blocks:
          best_sz = mid
          best_td = td_local
          lo = mid + 1
        else:
          hi = mid - 1
      if best_td is None:
        best_sz = 0
        td_local, nblocks_local = bytes_to_ofdm_blocks_bytes(header_bytes)
        if nblocks_local > allowed_blocks:
          samples_keep = allowed_blocks * symbol_len_samples
          td_local = td_local[:samples_keep]
        best_td = td_local

    # --- NEW: ensure each packet has exactly allowed_blocks OFDM blocks ---
      td_now = best_td
      if SYMBOL_LEN <= 0:
        nblocks_now = 0
      else:
        nblocks_now = len(td_now) // SYMBOL_LEN
      if nblocks_now < allowed_blocks:
        needed_blocks = allowed_blocks - nblocks_now
        # build filler from zeros (in bytes) that encode into needed_blocks blocks
        filler_bytes_len = needed_blocks * RS_DATA_BYTES
        td_filler, nblk_f = bytes_to_ofdm_blocks_bytes(b'\x00' * filler_bytes_len)
        # If bytes_to_ofdm_blocks_bytes produced at least needed_blocks, take exact samples
        if nblk_f >= needed_blocks and len(td_filler) >= needed_blocks * SYMBOL_LEN:
          td_now = np.concatenate((td_now, td_filler[:needed_blocks * SYMBOL_LEN]))
          nblocks_now = allowed_blocks
        else:
          # fallback: append raw zeros in time domain to reach required samples
          pad_samples = needed_blocks * SYMBOL_LEN
          td_now = np.concatenate((td_now, np.zeros(pad_samples, dtype=td_now.dtype)))
          nblocks_now = allowed_blocks
      elif nblocks_now > allowed_blocks:
        # safety trim (shouldn't happen due to binary search)
        td_now = td_now[:allowed_blocks * SYMBOL_LEN]
        nblocks_now = allowed_blocks
    # --------------------------------------------------------------------

      nblocks_sent_est = nblocks_now
      print(f"[TX-DBG] packet_no={packet_no} is_first={is_first} payload_bytes={best_sz} nblocks_est={nblocks_sent_est} header_first16={header_bytes[:16].hex()}")

      preamble = preamble_td
      packet_samples = np.concatenate((preamble, td_now)) if len(td_now) > 0 else preamble.copy()
      packet_preamble_offsets_no_preroll.append(running_sample_offset)

      def make_gap_noise(n_samps):
        if NOISE_LEVEL is None or NOISE_LEVEL <= 0:
          return np.zeros(n_samps, dtype=np.float64)
        w = rng_global.normal(loc=0.0, scale=1.0, size=n_samps).astype(np.float64)
        cur_rms = np.sqrt(np.mean(w**2)) if w.size > 0 else 1.0
        if cur_rms == 0:
          return np.zeros(n_samps, dtype=np.float64)
        pkt_peak = np.max(np.abs(packet_samples)) if np.max(np.abs(packet_samples)) > 0 else 1.0
        noise_rms = NOISE_LEVEL * pkt_peak
        w = w * (noise_rms / cur_rms)
        return w.astype(packet_samples.dtype)

      gap_noise = make_gap_noise(gap_samples)
      packet_samples = np.concatenate((packet_samples, gap_noise))

      samples_packets.append(packet_samples)

      packet_sample_count = len(packet_samples)
      print(f"[TX-DBG] packet {packet_no} sample_offset_start(no_preroll)={running_sample_offset} sample_count={packet_sample_count}")
      running_sample_offset += packet_sample_count

      remaining_data = remaining_data[best_sz:]
      packet_no += 1

      if len(remaining_data) == 0:
        break
      if packet_no > 1000000:
        print("[TX] too many packets, aborting")
        break

    tx_packets_concat = np.concatenate(samples_packets) if len(samples_packets) > 0 else np.array([], dtype=float)
    preroll_len = int(PREROLL_SEC * fs)
    postroll_len = int(POSTROLL_SEC * fs)
    preroll = np.zeros(preroll_len, dtype=tx_packets_concat.dtype)
    postroll = np.zeros(postroll_len, dtype=tx_packets_concat.dtype)
    if NOISE_LEVEL is not None and NOISE_LEVEL > 0.0:
      rnd = np.random.RandomState(NOISE_SEED)
      tx_peak = np.max(np.abs(tx_packets_concat)) if np.max(np.abs(tx_packets_concat)) > 0 else 1.0
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
      preroll = preroll_noise.astype(tx_packets_concat.dtype)
      postroll = postroll_noise.astype(tx_packets_concat.dtype)

    tx_out = np.concatenate((preroll, tx_packets_concat, postroll))
    tx_clipped, achieved_crest = apply_softclip_to_target_crest(tx_out, target_db=TARGET_CREST_DB)
    print(f"[SOFTCLIP] achieved crest = {achieved_crest:.3f} dB (target {TARGET_CREST_DB} dB)")
    max_abs = np.max(np.abs(tx_clipped)) if tx_clipped.size else 0.0
    if max_abs == 0:
        scale_to_int16 = 1.0
    else:
        scale_to_int16 = 1 / max_abs
    tx_int16 = (tx_clipped * scale_to_int16 * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write("ofdm_acoustic_tx_with_noise.wav", fs, tx_int16)
    packet_preamble_offsets_with_preroll = [preroll_len + int(x) for x in packet_preamble_offsets_no_preroll]
    print("TX saved to ofdm_acoustic_tx_with_noise.wav (packetized, gaps inserted)")
    print(f"[TX-DBG] total_packets={packet_no} total_samples={len(tx_packets_concat)} preroll={preroll_len} postroll={postroll_len}")
    print(f"[TX-DBG] transmission_header_hex={transmission_header.hex()[:256]}")
    print("[TX-DBG] preamble positions (sample indices in WAV, including preroll):")
    for i, pos in enumerate(packet_preamble_offsets_with_preroll):
      print(f"  packet {i}: preamble_sample_index={pos}")

  else:
    src = input("Откуда демодулировать? (file/mic): ").strip().lower()
    if src == "file":
      wav_path = input("Путь к WAV-файлу для приёма: ").strip()
      _, wavd = wavfile.read(wav_path)
      sig = wavd[:,0] if wavd.ndim > 1 else wavd
      rx = sig.astype(float) / np.iinfo(wavd.dtype).max
      
      
    else:
      # Live MIC on-the-fly reception
      print("Live MIC RX: starting input stream and on-the-fly decoding")

      # Config
      frames_per_buffer = 2048
      min_corr_len = len(preamble_td)
      max_buffer_len_sec = 60.0
      max_buffer_len = int(max_buffer_len_sec * fs)
      first_packet_wait_timeout = 20.0
      inter_packet_timeout = 20.0

      rx_buf = np.zeros(0, dtype=np.float64)
      rx_lock = None
      try:
        import threading
        rx_lock = threading.Lock()
      except Exception:
        rx_lock = None

      def append_to_buffer(arr):
        global rx_buf
        if rx_lock is not None:
            rx_lock.acquire()
        try:
            rx_buf = np.concatenate((rx_buf, arr))
            if len(rx_buf) > max_buffer_len:
                rx_buf = rx_buf[-max_buffer_len:]
        finally:
            if rx_lock is not None:
                rx_lock.release()

      def audio_callback(indata, frames, time_info, status):
        if status:
          print("[SD] status:", status)
        arr = indata[:,0].astype(np.float64)
        append_to_buffer(arr)

      stream = sd.InputStream(samplerate=fs, channels=1, blocksize=frames_per_buffer, callback=audio_callback)
      stream.start()
      print("[LIVE] stream started, waiting for preamble...")

      try:
        import time
        start_time = time.time()
        sync_idx = None
        local_abs_corr = None
        # first: wait for candidate preamble and attempt first-packet decode
        while True:
          time.sleep(0.05)
          if rx_lock is not None:
            rx_lock.acquire()
          try:
            buf = rx_buf.copy()
          finally:
            if rx_lock is not None:
              rx_lock.release()
          if len(buf) < min_corr_len:
            if time.time() - start_time > first_packet_wait_timeout:
              print("[LIVE] timeout waiting for first preamble")
              stream.stop()
              stream.close()
              sys.exit(1)
            continue

          corr = fftconvolve(buf, preamble_td[::-1], mode='valid')
          abs_corr = np.abs(corr)
          if abs_corr.size == 0:
            continue
          local_abs_corr = abs_corr
          from scipy.signal import find_peaks
          peak_val = np.max(abs_corr)
          thr = 0.45 * peak_val if peak_val != 0 else 0.0
          peaks, props = find_peaks(abs_corr, height=thr, distance=len(preamble_td)//2)
          peak_heights = props['peak_heights'] if 'peak_heights' in props else abs_corr[peaks]
          candidates = sorted(zip(peaks, peak_heights), key=lambda x: -x[1])
          if len(candidates) == 0:
            continue

          # try to find a candidate that decodes as transmission header
          found = False
          for cand_idx, cand_h in candidates[:12]:
            cand_sync = int(cand_idx)
            # check if enough samples for two preamble OFDMs + small payload
            min_need = cand_sync + len(preamble_td) + SYMBOL_LEN
            if len(buf) < min_need:
              continue
            # attempt fast header check (like in file-based code)
            try:
              rx_pre1 = buf[cand_sync + Ncp : cand_sync + Ncp + Nfft]
              rx_pre2 = buf[cand_sync + SYMBOL_LEN + Ncp : cand_sync + SYMBOL_LEN + Ncp + Nfft]
              S_ref = np.fft.fft(preamble_td[Ncp : Ncp + Nfft])
              R1t, R2t = np.fft.fft(rx_pre1), np.fft.fft(rx_pre2)
              Hk_try = ((R1t[subc_inds]/S_ref[subc_inds]) + (R2t[subc_inds]/S_ref[subc_inds]))/2
              Hk_mag_s_try = np.clip(medfilt(np.abs(Hk_try), 5), 1/2.0, None)
              Hk_smooth_try = Hk_mag_s_try * np.exp(1j*np.angle(Hk_try))
            except Exception:
              continue

            # attempt minimal decode of first codeword
            available_after = len(buf) - (cand_sync + len(preamble_td))
            nblk = available_after // SYMBOL_LEN
            if nblk <= 0:
              continue
            try_nblk = min(6, nblk)
            with_sample_end = cand_sync + len(preamble_td) + try_nblk * SYMBOL_LEN
            if len(buf) < with_sample_end:
              continue
            seg_try = buf[cand_sync + len(preamble_td) : with_sample_end]
            frames_try = seg_try.reshape(-1, SYMBOL_LEN)
            try:
              rx_subc_try = np.concatenate([np.fft.fft(frm[Ncp:])[subc_inds] / Hk_smooth_try for frm in frames_try])
            except Exception:
              continue
            if subc_phases is not None and np.any(subc_phases != 0):
              rx_syms_try = rx_subc_try * np.tile(np.exp(-1j * subc_phases), try_nblk)
            else:
              rx_syms_try = rx_subc_try
            bits_try = qpsk_demap(rx_syms_try)
            cw_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
            if len(bits_try) < cw_bits:
              continue
            bts0 = bits_to_bytes(bits_try[:cw_bits])
            try:
              msg0 = rs.decode(bts0)[0]
              # header detection: MSB set or our header rule
              if len(msg0) > 0 and (msg0[0] & 0x80):
                sync_idx = cand_sync
                found = True
                break
            except Exception:
              continue

          if found:
            print(f"[LIVE] initial sync at buffer index {sync_idx}")
            break

        # at this point we have sync_idx and local buffer; wait until full first-packet samples present
        while True:
          if rx_lock is not None:
            rx_lock.acquire()
          try:
            buf = rx_buf.copy()
          finally:
            if rx_lock is not None:
              rx_lock.release()
          if sync_idx is None:
            print("[LIVE] lost sync")
            stream.stop(); stream.close(); sys.exit(1)
          # compute Hk from preamble (needs two preamble symbols fully present)
          if len(buf) < sync_idx + len(preamble_td) + Nfft + SYMBOL_LEN + Nfft:
            time.sleep(0.05); continue
          try:
            rx_pre1 = buf[sync_idx + Ncp : sync_idx + Ncp + Nfft]
            rx_pre2 = buf[sync_idx + SYMBOL_LEN + Ncp : sync_idx + SYMBOL_LEN + Ncp + Nfft]
            S_ref = np.fft.fft(preamble_td[Ncp : Ncp + Nfft])
            R1, R2 = np.fft.fft(rx_pre1), np.fft.fft(rx_pre2)
            Hk = ((R1[subc_inds]/S_ref[subc_inds]) + (R2[subc_inds]/S_ref[subc_inds]))/2
            Hk_mag_s = np.clip(medfilt(np.abs(Hk), 5), 1/2.0, None)
            Hk_smooth = Hk_mag_s * np.exp(1j*np.angle(Hk))
          except Exception:
            time.sleep(0.05); continue

          # estimate how many OFDM blocks we currently have after preamble
          available_after = len(buf) - (sync_idx + len(preamble_td))
          nblk = available_after // SYMBOL_LEN
          if nblk <= 0:
            time.sleep(0.05); continue

          # read as many blocks as present (first we need enough to decode header region)
          nblk_read = nblk
          seg = buf[sync_idx + len(preamble_td) : sync_idx + len(preamble_td) + nblk_read * SYMBOL_LEN]
          frames = seg.reshape(-1, SYMBOL_LEN)
          try:
            rx_subc = np.concatenate([np.fft.fft(frm[Ncp:])[subc_inds] / Hk_smooth for frm in frames])
          except Exception:
            time.sleep(0.05); continue
          if subc_phases is not None and np.any(subc_phases != 0):
            rx_syms = rx_subc * np.tile(np.exp(-1j * subc_phases), nblk_read)
          else:
            rx_syms = rx_subc
          all_rx_bits = qpsk_demap(rx_syms)
          cw_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
          total_cw_bits = cw_bits * nblk_read
          rx_bits = all_rx_bits[:total_cw_bits]

          # RS decode blocks
          decoded_blocks = []
          for i in range(nblk_read):
            bstart = i * cw_bits
            bbits = rx_bits[bstart:bstart+cw_bits]
            if len(bbits) < cw_bits:
              bbits = np.concatenate((bbits, np.zeros(cw_bits - len(bbits), dtype=int)))
            bts = bits_to_bytes(bbits)
            try:
              msg = rs.decode(bts)[0]
            except Exception:
              msg = b'\x00' * RS_DATA_BYTES
            decoded_blocks.append(msg)
          all_bytes = b"".join(decoded_blocks)
          # need at least 64 bytes to parse transmission header
          if len(all_bytes) < 64:
            # wait more samples
            time.sleep(0.05); continue

          header64 = all_bytes[:64]
          try:
            hdr = parse_header(header64)
          except Exception as e:
            print("[LIVE] parse_header failed on first packet:", e)
            time.sleep(0.05); continue
          mode_rx = hdr['mode']
          total_sz = hdr['data_len']
          fname = hdr['filename']
          packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)
          print(f"[LIVE] got transmission header: data_len={total_sz} filename={fname} packet_blocks={packet_blocks_from_hdr}")
          # compute expected packet preamble absolute indices relative to this buffer start
          positions_no_preroll = simulate_packet_positions(total_sz, fname.encode('utf-8') if fname else b'', (mode_rx == b'T'),
                                                           packet_blocks_from_hdr, preamble_len=len(preamble_td),
                                                           symbol_len=SYMBOL_LEN, gap_samples=GAP_SAMPLES_DEFAULT)
          if len(positions_no_preroll) == 0:
            print("[LIVE] simulate_packet_positions returned no positions"); stream.stop(); stream.close(); sys.exit(1)
          base_abs = sync_idx - positions_no_preroll[0]
          expected_abs = [base_abs + int(x) for x in positions_no_preroll]
          print("[LIVE] expected preamble absolute indices (buffer coords):", expected_abs)
          # now process packets in sequence
          assembled_chunks = []
          bytes_collected = 0
          expected_total = total_sz
          pkt_index = 0
          # for subsequent processing we require that buffer keeps being appended; we'll wait when needed
          while pkt_index < len(expected_abs):
            target_pref = expected_abs[pkt_index]
            # wait until enough samples are in buffer for the whole packet (preamble + packet_blocks * SYMBOL_LEN)
            need_samples = target_pref + len(preamble_td) + packet_blocks_from_hdr * SYMBOL_LEN
            waited = 0.0
            while True:
              if rx_lock is not None:
                rx_lock.acquire()
              try:
                cur_len = len(rx_buf)
              finally:
                if rx_lock is not None:
                  rx_lock.release()
              if cur_len >= need_samples:
                break
              time.sleep(0.05)
              waited += 0.05
              if waited > inter_packet_timeout:
                print(f"[LIVE] timeout waiting for packet {pkt_index} samples"); stream.stop(); stream.close(); sys.exit(1)
            # attempt decode at local candidates near target_pref (reuse decode_packet_at_candidate logic)
            # build local abs_corr if needed
            if local_abs_corr is None:
              if rx_lock is not None:
                rx_lock.acquire()
              try:
                temp_buf = rx_buf.copy()
              finally:
                if rx_lock is not None:
                  rx_lock.release()
              local_abs_corr = np.abs(fftconvolve(temp_buf, preamble_td[::-1], mode='valid'))
            # search window +/- Ncp
            wlo = max(0, target_pref - Ncp)
            whi = min(len(local_abs_corr)-1, target_pref + Ncp)
            window = local_abs_corr[wlo:whi+1] if whi >= wlo else np.array([])
            candidate_positions = []
            if window.size > 0:
              local_rel = np.argsort(window)[::-1]
              for r in local_rel[:8]:
                candidate_positions.append(wlo + int(r))
            grid = list(range(max(0, target_pref - Ncp), min(len(local_abs_corr), target_pref + Ncp + 1)))
            combined = []
            for p in candidate_positions + grid:
              if p not in combined:
                combined.append(p)

            # --- Diagnostic candidate loop: prints Hk stats, statuses, first bytes; saves WAV optionally ---
            debug_print = True
            save_problem_wav = False  # set True to save problematic segments
            problem_count = 0

            best_bytes = b''
            best_rs_ok = -1
            best_used = None

            # capture buffer for decoding
            if rx_lock is not None:
              rx_lock.acquire()
            try:
              cur_buf = rx_buf.copy()
              cur_abs_corr = local_abs_corr.copy()
            finally:
              if rx_lock is not None:
                rx_lock.release()

            for cand in combined:
              if debug_print:
                peak_val_here = cur_abs_corr[cand] if cand < len(cur_abs_corr) else 0.0
                print(f"[DIAG] candidate={cand} peak={peak_val_here:.6g}")
                
              # check preamble samples presence
              if len(cur_buf) < cand + len(preamble_td):
                if debug_print:
                  print(f"  [SKIP] not enough samples for preamble at cand (have {len(cur_buf)}, need {cand+len(preamble_td)})")
                continue

              # estimate Hk
              try:
                pre1 = cur_buf[cand + Ncp : cand + Ncp + Nfft]
                pre2 = cur_buf[cand + SYMBOL_LEN + Ncp : cand + SYMBOL_LEN + Ncp + Nfft]
                S_ref = np.fft.fft(preamble_td[Ncp : Ncp + Nfft])
                R1t, R2t = np.fft.fft(pre1), np.fft.fft(pre2)
                Hk_est = ((R1t[subc_inds]/S_ref[subc_inds]) + (R2t[subc_inds]/S_ref[subc_inds]))/2
                Hk_amp = np.abs(Hk_est)
                Hk_phase = np.angle(Hk_est)
                Hk_stats = (Hk_amp.min(), np.median(Hk_amp), Hk_amp.max(), np.std(Hk_amp))
                if debug_print:
                  print(f"  Hk_amp min/med/max/std = {Hk_stats[0]:.3g}/{Hk_stats[1]:.3g}/{Hk_stats[2]:.3g}/{Hk_stats[3]:.3g}")
              except Exception as e:
                if debug_print:
                  print("  [ERR] Hk estimation failed:", e)
                continue

              # how many OFDM blocks available after preamble
              pkt_data_start = cand + len(preamble_td)
              available_after = len(cur_buf) - pkt_data_start
              nblk_avail = available_after // SYMBOL_LEN
              pkt_payload_samples = packet_blocks_from_hdr * SYMBOL_LEN
              if debug_print:
                print(f"  available_after_samples={available_after} => nblk_avail={nblk_avail} (need {packet_blocks_from_hdr})")

              if nblk_avail < 1:
                if debug_print:
                  print("  [SKIP] no OFDM blocks available after preamble")
                continue

              # choose n_blocks_used = min(nblk_avail, packet_blocks_from_hdr)
              n_blocks_used = min(nblk_avail, packet_blocks_from_hdr)
              seg = cur_buf[pkt_data_start : pkt_data_start + n_blocks_used * SYMBOL_LEN]
              if len(seg) < n_blocks_used * SYMBOL_LEN:
                if debug_print:
                  print("  [SKIP] segment too short")
                continue

              # form frames and equalize
              try:
                frames_local = seg.reshape(n_blocks_used, SYMBOL_LEN)
                Hk_mask = np.clip(medfilt(Hk_amp,5), 1/2.0, None) * np.exp(1j*np.angle(Hk_est))
                rx_subc_pkt = np.concatenate([np.fft.fft(fr[Ncp:])[subc_inds] / Hk_mask for fr in frames_local])
              except Exception:
                # fallback simpler: use Hk_est directly
                try:
                  frames_local = seg.reshape(n_blocks_used, SYMBOL_LEN)
                  rx_subc_pkt = np.concatenate([np.fft.fft(fr[Ncp:])[subc_inds] / Hk_est for fr in frames_local])
                except Exception as e2:
                  if debug_print:
                    print("  [ERR] equalization failed:", e2)
                  continue

              if subc_phases is not None and np.any(subc_phases != 0):
                rx_syms_pkt = rx_subc_pkt * np.tile(np.exp(-1j * subc_phases), n_blocks_used)
              else:
                rx_syms_pkt = rx_subc_pkt

              bits_pkt = qpsk_demap(rx_syms_pkt)
              cw_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
              n_cw = len(bits_pkt) // cw_bits
              if debug_print:
                print(f"  n_blocks_used={n_blocks_used} => n_cw={n_cw} codewords")

              rs_ok = 0
              per_block_status = []
              decoded_blocks = []
              for ci in range(n_cw):
                bstart = ci * cw_bits
                bbits = bits_pkt[bstart:bstart+cw_bits]
                if len(bbits) < cw_bits:
                  bbits = np.concatenate((bbits, np.zeros(cw_bits - len(bbits), dtype=int)))
                bts = bits_to_bytes(bbits)
                try:
                  msg = rs.decode(bts)[0]
                  per_block_status.append("OK")
                  rs_ok += 1
                except Exception:
                  msg = b'\x00' * RS_DATA_BYTES
                  per_block_status.append("ERR")
                decoded_blocks.append(msg)

              first_bytes_hex = decoded_blocks[0][:16].hex() if len(decoded_blocks) > 0 else ""
              if debug_print:
                print(f"  rs_ok={rs_ok}/{n_cw} statuses={per_block_status[:12]} first_block16={first_bytes_hex}")

              # optionally save problematic segment to WAV for offline analysis
              if save_problem_wav and rs_ok < max(1, int(0.5*n_cw)):
                try:
                  import soundfile as sf
                  wav_name = f"problem_pkt_cand_{problem_count}_pos_{cand}.wav"
                  sf.write(wav_name, seg, fs)
                  if debug_print:
                    print(f"  saved segment to {wav_name}")
                  problem_count += 1
                except Exception:
                  pass

              # decide best candidate
              if rs_ok > best_rs_ok:
                best_rs_ok = rs_ok
                best_bytes = b"".join(decoded_blocks)
                best_used = cand
                if rs_ok >= n_cw * 0.9 and n_cw > 0:
                  if debug_print:
                    print("  [ACCEPT] good candidate; breaking")
                  break

            # end candidate loop
            print(f"[DIAG] best_used={best_used} best_rs_ok={best_rs_ok} n_cw={n_cw if 'n_cw' in locals() else 'NA'}")

            if best_rs_ok <= 0:
              print(f"[LIVE] failed to decode packet {pkt_index} at expected pos {target_pref}")
              stream.stop(); stream.close(); sys.exit(1)
            # extract payload: first packet special-case header; subsequent packets strip 4-byte packet header
            if pkt_index == 0:
              if len(best_bytes) < 64:
                print("[LIVE] first packet decoded <64 bytes, aborting"); stream.stop(); stream.close(); sys.exit(1)
              payload_chunk = best_bytes[64:]
            else:
              payload_chunk = best_bytes[4:] if len(best_bytes) >= 4 else b''
            assembled_chunks.append(payload_chunk)
            bytes_collected += len(payload_chunk)
            print(f"[LIVE] pkt{pkt_index} decoded RS_ok={best_rs_ok} used_pre={best_used} collected {bytes_collected}/{expected_total}")
            if bytes_collected >= expected_total:
              break
            pkt_index += 1

          # done with all needed packets
          assembled = b"".join(assembled_chunks)[:expected_total]
          if mode_rx == b"F":
            out_fname = fname if fname else "rx_file"
            out_path = "rx_" + out_fname
            with open(out_path, "wb") as f:
              f.write(assembled)
            print(f"[LIVE RX FILE] saved: {out_path} ({len(assembled)} bytes)")
          else:
            rec_text = assembled.decode("utf-8", errors="ignore")
            print("[LIVE RX TEXT]", rec_text)

          stream.stop()
          stream.close()
          break

      except KeyboardInterrupt:
        print("[LIVE] interrupted by user")
        try:
          stream.stop(); stream.close()
        except Exception:
          pass
        sys.exit(1)

