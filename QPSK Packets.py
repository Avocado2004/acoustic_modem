#!/usr/bin/env python3
"""
OFDM Acoustic Modem — обновлённый: per-packet RX декодирование и packet_blocks в transmission header.
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
    """
    positions = []
    # For first packet header is 64, subsequent 4; but payload packing depends on RS blocks.
    # We use conservative model: compute bytes per packet by binary search via bytes_to_ofdm_blocks_bytes
    # but since we know packet_blocks_local is fixed, we can compute payload bytes that fit into that many blocks.
    # Find payload bytes per packet for first packet (header 64) and others (header 4)
    def max_payload_for_header(header_bytes):
        # search in RS_DATA_BYTES steps
        step = RS_DATA_BYTES
        lo = 0
        hi = step
        # exponential growth
        while True:
            td, nblk = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * hi))
            if nblk > packet_blocks_local:
                break
            hi *= 2
            if hi > total_data_len_bytes + step:
                hi = min(hi, total_data_len_bytes)
                break
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
    POSTROLL_SEC = 0.10
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

      nblocks_sent_est = (len(best_td) // (Nfft + Ncp)) if (Nfft + Ncp) > 0 else None
      print(f"[TX-DBG] packet_no={packet_no} is_first={is_first} payload_bytes={best_sz} nblocks_est={nblocks_sent_est} header_first16={header_bytes[:16].hex()}")

      preamble = preamble_td
      packet_samples = np.concatenate((preamble, best_td)) if len(best_td) > 0 else preamble.copy()
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
      print("Запись с микрофона 10 секунд…")
      rec = sd.rec(int(10*fs), samplerate=fs, channels=1)
      sd.wait()
      rx = rec[:,0].astype(float)
      rx /= np.max(np.abs(rx))

    # initial correlation and candidate selection (unchanged)
    corr = fftconvolve(rx, preamble_td[::-1], mode='valid')
    abs_corr = np.abs(corr)
    if abs_corr.size == 0:
        print("[RX-ERR] correlation empty, cannot find preambles")
        sync_idx = sync_by_corr(rx, preamble_td)
    else:
        from scipy.signal import find_peaks
        peak_val = np.max(abs_corr)
        threshold = 0.5 * peak_val if peak_val != 0 else 0.0
        peaks, props = find_peaks(abs_corr, height=threshold, distance=len(preamble_td)//2)
        peak_heights = props['peak_heights'] if 'peak_heights' in props else abs_corr[peaks]
        candidates = sorted(zip(peaks, peak_heights), key=lambda x: -x[1])
        print(f"[RX-DBG] found {len(candidates)} preamble peak candidates (threshold={threshold:.6g})")
        for i,(p,h) in enumerate(candidates):
            print(f"[RX-DBG] candidate {i}: corr_index={p} sample_index_in_rx={p} height={h:.6g}")

        sync_idx = None
        max_try = min(10, len(candidates))
        for try_i in range(max_try):
            cand_idx, cand_h = candidates[try_i]
            cand_sync = int(cand_idx)
            start_try = cand_sync + len(preamble_td)
            remaining_try = len(rx) - start_try
            nblk_try = remaining_try // SYMBOL_LEN
            if nblk_try <= 0:
                continue
            try:
                rx_pre1_try = rx[cand_sync + Ncp : cand_sync + Ncp + Nfft]
                rx_pre2_try = rx[cand_sync + SYMBOL_LEN + Ncp : cand_sync + SYMBOL_LEN + Ncp + Nfft]
                S_ref = np.fft.fft(preamble_td[Ncp : Ncp + Nfft])
                R1t, R2t = np.fft.fft(rx_pre1_try), np.fft.fft(rx_pre2_try)
                Hk_try = ((R1t[subc_inds]/S_ref[subc_inds]) + (R2t[subc_inds]/S_ref[subc_inds]))/2
                Hk_mag_s_try = np.clip(medfilt(np.abs(Hk_try), 5), 1/2.0, None)
                Hk_smooth_try = Hk_mag_s_try * np.exp(1j*np.angle(Hk_try))
            except Exception:
                continue

            try_nblk = min(8, nblk_try)
            seg_try = rx[start_try : start_try + try_nblk * SYMBOL_LEN]
            frames_try = seg_try.reshape(-1, SYMBOL_LEN)
            rx_subc_try = np.concatenate([np.fft.fft(frm[Ncp:])[subc_inds] / Hk_smooth_try for frm in frames_try])
            if subc_phases is not None and np.any(subc_phases != 0):
                phases_rep = np.tile(subc_phases, try_nblk)
                rx_syms_try = rx_subc_try * np.exp(-1j * phases_rep)
            else:
                rx_syms_try = rx_subc_try
            bits_try = qpsk_demap(rx_syms_try)
            cw_bits = ((RS_DATA_BYTES+RS_PARITY_BYTES)*8)
            if len(bits_try) < cw_bits:
                continue
            bts0 = bits_to_bytes(bits_try[:cw_bits])
            try:
                msg0 = rs.decode(bts0)[0]
                if len(msg0) > 0 and (msg0[0] & 0x80):
                    sync_idx = cand_sync
                    print(f"[RX-DBG] selected sync candidate at {sync_idx} (peak height {cand_h:.6g})")
                    break
            except Exception:
                continue

        if sync_idx is None:
            if len(candidates) > 0:
                sync_idx = int(candidates[0][0])
                print(f"[RX-DBG] no candidate passed header check; falling back to strongest peak {sync_idx}")
            else:
                sync_idx = sync_by_corr(rx, preamble_td)
                print(f"[RX-DBG] no peaks found; fallback sync index {sync_idx}")

    # parse first packet region enough to decode all first-packet codewords
    symbol_len = SYMBOL_LEN
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

    frames = segment.reshape(nblk, SYMBOL_LEN)
    rx_subc = [np.fft.fft(frm[Ncp:])[subc_inds] / Hk_smooth for frm in frames]
    rx_subc = np.concatenate(rx_subc)

    if subc_phases is not None and np.any(subc_phases != 0):
      reps = nblk
      phases_rep = np.tile(subc_phases, reps)
      rx_syms = rx_subc * np.exp(-1j * phases_rep)
    else:
      rx_syms = rx_subc

    all_rx_bits = qpsk_demap(rx_syms)
    codeword_len_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
    total_cw_bits = codeword_len_bits * nblk
    rx_bits = all_rx_bits[:total_cw_bits]

    decoded = []
    rs_statuses = []
    print("=== RS-декодирование по блокам (first packet region) ===")
    for i in range(nblk):
      start_bit = i * codeword_len_bits
      end_bit = (i+1) * codeword_len_bits
      block_bits = rx_bits[start_bit:end_bit]
      if len(block_bits) < codeword_len_bits:
        pad_bits = np.zeros(codeword_len_bits - len(block_bits), dtype=int)
        block_bits = np.concatenate((block_bits, pad_bits))
      bts = bits_to_bytes(block_bits)
      try:
        msg = rs.decode(bts)[0]
        status = "OK"
      except ReedSolomonError:
        msg = b'\x00' * RS_DATA_BYTES
        status = "ERR"
      print(f"[RX-DBG] Block {i+1}/{nblk}: RS decode = {status} data_first4={msg[:4].hex()}")
      rs_statuses.append(status)
      decoded.append(msg)

    all_bytes = b"".join(decoded)
    print(f"[RX-DEBUG] decoded_blocks={len(decoded)}, total_data_bytes={len(all_bytes)}")
    if len(all_bytes) >= 64:
      print("[RX-DEBUG] received header (hex) first 64 bytes:", all_bytes[:64].hex())
    else:
      print("[RX-DEBUG] received <64 bytes after decode:", all_bytes.hex())

    def find_transmission_header_offset(buf, max_scan=1024):
        for off in range(0, min(len(buf)-63, max_scan)):
            if buf[off] & 0x80:
                return off
        return None

    off = find_transmission_header_offset(all_bytes, max_scan=512)
    print(f"[RX-DBG] found_tx_hdr_offset={off}")
    if off is not None and off != 0:
        print("[RX-DBG] candidate header hex at offset:", all_bytes[off:off+64].hex()[:256])

    if len(all_bytes) < 64:
      print("[RX] Недостаточно данных после RS декодирования для полноценного заголовка (64 байта)")
      sys.exit(1)

    header64 = all_bytes[:64]
    try:
      hdr = parse_header(header64)
    except Exception as e:
      print("[RX-ERR] parse_header failed:", e)
      print("[RX-ERR] header64_hex:", header64.hex())
      sys.exit(1)

    mode_rx = hdr['mode']
    total_sz = hdr['data_len']
    fname = hdr['filename']
    packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)
    print(f"[RX-INFO] transmission header: data_len={total_sz} filename={fname} packet_blocks={packet_blocks_from_hdr}")

    # simulate expected packet preamble offsets (relative to concatenated packets)
    positions_no_preroll = simulate_packet_positions(total_sz, fname.encode('utf-8') if fname else b'', (mode_rx == b'T'),
                                                     packet_blocks_from_hdr, preamble_len=len(preamble_td), symbol_len=SYMBOL_LEN,
                                                     gap_samples=GAP_SAMPLES_DEFAULT)
    # map to absolute indices in rx
    if len(positions_no_preroll) == 0:
      print("[RX-ERR] simulate_packet_positions returned no positions")
      sys.exit(1)
    base_abs = sync_idx - positions_no_preroll[0]
    expected_abs = [base_abs + int(x) for x in positions_no_preroll]
    print("[RX-DBG] expected preamble absolute indices:", expected_abs)

    # per-packet decoding loop
    def decode_packet_at_candidate(pref_abs, packet_blocks_expected):
        """
        Try to decode packet whose preamble is near pref_abs.
        Search window +/- Ncp for best local peak (and try shifts if RS fails).
        Returns tuple (decoded_bytes, rs_ok_count, used_preamble_abs)
        """
        best_result = (b'', 0, None)
        # search local peaks in window first (prefer actual correlation peaks)
        wlo = max(0, pref_abs - Ncp)
        whi = min(len(abs_corr)-1, pref_abs + Ncp)
        window = abs_corr[wlo:whi+1] if whi >= wlo else np.array([])
        candidate_positions = []
        if window.size > 0:
            local_rel = np.argsort(window)[::-1]  # descending by peak height
            # take top few candidates but keep order deterministic
            for r in local_rel[:6]:
                candidate_positions.append(wlo + int(r))
        # fallback: try full ±Ncp grid
        grid = list(range(max(0, pref_abs - Ncp), min(len(abs_corr), pref_abs + Ncp + 1)))
        # combine unique candidates preserving order
        combined = []
        for p in candidate_positions + grid:
            if p not in combined:
                combined.append(p)
        # attempt decode for each candidate position, track best by RS_OK
        for cand in combined:
            # estimate Hk from two preamble symbols at cand
            try:
                pre1 = rx[cand + Ncp : cand + Ncp + Nfft]
                pre2 = rx[cand + SYMBOL_LEN + Ncp : cand + SYMBOL_LEN + Ncp + Nfft]
                S_ref = np.fft.fft(preamble_td[Ncp : Ncp + Nfft])
                R1t, R2t = np.fft.fft(pre1), np.fft.fft(pre2)
                Hk_est = ((R1t[subc_inds]/S_ref[subc_inds]) + (R2t[subc_inds]/S_ref[subc_inds]))/2
                Hk_mag = np.clip(medfilt(np.abs(Hk_est), 5), 1/2.0, None)
                Hk_s = Hk_mag * np.exp(1j*np.angle(Hk_est))
            except Exception:
                continue
            # collect packet frames
            pkt_data_start = cand + len(preamble_td)
            pkt_payload_samples = packet_blocks_expected * SYMBOL_LEN
            seg = rx[pkt_data_start : pkt_data_start + pkt_payload_samples]
            if len(seg) < pkt_payload_samples:
                # not enough samples, skip
                continue
            frames = seg.reshape(packet_blocks_expected, SYMBOL_LEN)
            try:
                rx_subc_pkt = np.concatenate([np.fft.fft(fr[Ncp:])[subc_inds] / Hk_s for fr in frames])
            except Exception:
                continue
            if subc_phases is not None and np.any(subc_phases != 0):
                rx_syms_pkt = rx_subc_pkt * np.tile(np.exp(-1j * subc_phases), packet_blocks_expected)
            else:
                rx_syms_pkt = rx_subc_pkt
            bits_pkt = qpsk_demap(rx_syms_pkt)
            cw_bits = (RS_DATA_BYTES + RS_PARITY_BYTES) * 8
            n_cw = len(bits_pkt) // cw_bits
            rs_ok = 0
            decoded_blocks = []
            for ci in range(n_cw):
                bstart = ci * cw_bits
                bbits = bits_pkt[bstart:bstart+cw_bits]
                if len(bbits) < cw_bits:
                    bbits = np.concatenate((bbits, np.zeros(cw_bits - len(bbits), dtype=int)))
                bts = bits_to_bytes(bbits)
                try:
                    msg = rs.decode(bts)[0]
                    rs_ok += 1
                except ReedSolomonError:
                    msg = b'\x00' * RS_DATA_BYTES
                decoded_blocks.append(msg)
            # choose candidate with highest rs_ok
            if rs_ok > best_result[1]:
                best_result = (b"".join(decoded_blocks), rs_ok, cand)
                # early accept high success
                if rs_ok >= n_cw * 0.9:
                    break
        return best_result

    assembled_chunks = []
    bytes_collected = 0
    expected_total = total_sz
    packet_blocks_val = packet_blocks_from_hdr
    hdr_first_bytes = header64

    for pkt_idx, pref in enumerate(expected_abs):
        # decode each packet separately
        decoded_bytes, rs_ok_count, used_preamble = decode_packet_at_candidate(pref, packet_blocks_val)
        print(f"[RX-DBG] pkt{pkt_idx}: pref={pref} used_pre={used_preamble} RS_OK={rs_ok_count} bytes={len(decoded_bytes)}")
        if pkt_idx == 0:
            # first packet contains transmission header at its beginning
            if len(decoded_bytes) < 64:
                print("[RX-ERR] first packet decoded <64 bytes, aborting")
                sys.exit(1)
            # first 64 bytes are transmission header area (may include padding from RS)
            first_hdr_bytes = decoded_bytes[:64]
            try:
                hdr0 = parse_header(first_hdr_bytes)
            except Exception as e:
                print("[RX-ERR] parse_header(first) failed:", e)
                hdr0 = hdr
            # payload part after header in first packet:
            payload_chunk = decoded_bytes[64:]
            assembled_chunks.append(payload_chunk)
            bytes_collected += len(payload_chunk)
        else:
            # subsequent packets: first 4 bytes are packet header (within first RS data bytes)
            if len(decoded_bytes) >= 4:
                packet_header = decoded_bytes[:4]
                payload_chunk = decoded_bytes[4:]
                assembled_chunks.append(payload_chunk)
                bytes_collected += len(payload_chunk)
            else:
                assembled_chunks.append(b'')
        print(f"[RX-INFO] collected {bytes_collected}/{expected_total} bytes")
        if bytes_collected >= expected_total:
            break

    assembled = b"".join(assembled_chunks)[:expected_total]
    if mode_rx == b"F":
      out_fname = fname if fname else "rx_file"
      out_path = "rx_" + out_fname
      with open(out_path, "wb") as f:
        f.write(assembled)
      print(f"[RX FILE] Сохранён файл: {out_path} ({len(assembled)} байт)")
    else:
      rec_text = assembled.decode("utf-8", errors="ignore")
      print("[RX TEXT]", rec_text)

    # Visualization
    try:
      plot_constellation(rx_syms, "RX Constellation (eq.)")
      freqs = subc_inds * fs / Nfft
      eq_gain = 1.0/np.abs(Hk_smooth)
      plt.figure(figsize=(8,4))
      plt.plot(freqs, eq_gain, "o-", color="darkorange", label="Gain")
      plt.title("Equalizer Gain vs Frequency")
      plt.xlabel("Frequency (Hz)"); plt.ylabel("Gain")
      plt.grid(True); plt.legend(); plt.show()
    except Exception:
      pass


