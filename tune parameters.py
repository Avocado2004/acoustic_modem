#!/usr/bin/env python3
"""
OFDM Acoustic Modem — обновлённый: per-packet RX декодирование и packet_blocks в transmission header.
TX: каждый пакет содержит ровно DEFAULT_PACKET_BLOCKS OFDM блоков; при необходимости последний пакет дополняется нулями.

Изменения: преамбула теперь ZC, ZC, (gap), pilot, pilot, payload.
ZC используется для точной синхронизации и оценки FOE (frequency offset error).
Пилоты остаются для оценки частотно-зависимого канала (Hk).
Все остальные части кода оставлены без изменений по логике.
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

# -----------------------
# Zadoff-Chu generator (frequency-domain placement)
# -----------------------
def zc_root_sequence(u: int, L: int):
  """
  Generate Zadoff-Chu sequence length L, root u.
  Standard definition for prime-friendly lengths; here we use general formula:
  x[n] = exp(-j*pi*u*n*(n+1)/L)  (n = 0..L-1)
  Return complex array length L.
  """
  n = np.arange(L)
  z = np.exp(-1j * np.pi * u * n * (n + 1) / float(L))
  # normalize energy to 1
  z = z / np.sqrt(np.mean(np.abs(z)**2))
  return z

# -----------------------
# build_preamble: ZC, ZC, (gap), pilot, pilot
# -----------------------
def build_preamble(reps=1, zc_root=1):
  """
  New preamble structure:
  - ZC OFDM symbol (frequency-domain placed on subc_inds)
  - ZC OFDM symbol (repeat for FOE estimation)
  - then two pilot OFDM symbols (as before) for per-subcarrier channel estimation
  The function returns concatenated time-domain samples (with CP).
  """
  # ZC in frequency domain on subc_inds (length Nsub)
  zc_seq = zc_root_sequence(zc_root, Nsub)
  # place zc_seq on subc_inds (ofdm_symbol expects data_syms length Nsub)
  S_zc = ofdm_symbol(zc_seq)
  pilot = np.full(Nsub, (1 + 1j) / np.sqrt(2))
  S_pilot = ofdm_symbol(pilot)
  # structure: ZC, ZC, pilot, pilot (repeat parameter not used for ZC here)
  return np.concatenate((S_zc, S_zc, S_pilot, S_pilot))

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

def decode_packet_at_candidate(pref_abs, packet_blocks_expected, packet_idx=0, bytes_before_packet=0, expected_total=0):
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
        local_rel = np.argsort(window)[::-1]   # preserved intent
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
        # estimate FOE & Hk from preamble at cand
        try:
            # ZC at cand and cand+SYMBOL_LEN, pilots at cand+2*SYMBOL_LEN and +3*SYMBOL_LEN
            # FOE estimation using ZC pair:
            rx_zc1 = rx[cand + Ncp : cand + Ncp + Nfft]
            rx_zc2 = rx[cand + SYMBOL_LEN + Ncp : cand + SYMBOL_LEN + Ncp + Nfft]
            cross = np.vdot(rx_zc1, rx_zc2)
            delta_phi = np.angle(cross)
            T_between = SYMBOL_LEN / float(fs)
            f_err_loc = delta_phi / (2.0 * np.pi * T_between)
            # pilot positions for Hk
            pilot1_start = cand + 2*SYMBOL_LEN
            rx_pre1 = rx[pilot1_start + Ncp : pilot1_start + Ncp + Nfft]
            pilot2_start = pilot1_start + SYMBOL_LEN
            rx_pre2 = rx[pilot2_start + Ncp : pilot2_start + Ncp + Nfft]
            # FOE compensate pilots before FFT
            t1_offset = (pilot1_start) / float(fs)
            t2_offset = (pilot2_start) / float(fs)
            time_vec = np.arange(Nfft) / float(fs)
            rx_pre1_corr = rx_pre1 * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t1_offset + time_vec))
            rx_pre2_corr = rx_pre2 * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t2_offset + time_vec))
            S_ref = np.fft.fft(preamble_td[2*SYMBOL_LEN + Ncp : 2*SYMBOL_LEN + Ncp + Nfft])
            R1t, R2t = np.fft.fft(rx_pre1_corr), np.fft.fft(rx_pre2_corr)
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
        # apply FOE compensation per frame and equalize
        rx_subc_pkt = []
        for idxf, fr in enumerate(frames):
            frame_start_abs = pkt_data_start + idxf * SYMBOL_LEN
            t_frame_start = frame_start_abs / float(fs)
            useful = fr[Ncp:]
            tv = t_frame_start + np.arange(Nfft) / float(fs)
            useful_corr = useful * np.exp(-1j * 2.0 * np.pi * f_err_loc * tv)
            try:
                F = np.fft.fft(useful_corr)
                subc = F[subc_inds] / Hk_s
            except Exception:
                subc = np.zeros(Nsub, dtype=complex)
            rx_subc_pkt.append(subc)
        try:
            rx_subc_pkt = np.concatenate(rx_subc_pkt)
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
            cw_in_pkt = bts.hex()
            try:
                msg = rs.decode(bts)[0]
                rs_ok += 1
                # optionally verbose per-block: uncomment next line for very detailed log
                # print(f"[RX-DBG-PKT] cw_idx={ci} RS_OK cw_in={cw_in_pkt} data_first4={msg[:4].hex()}")
            except Exception as e:
                msg = b'\x00' * RS_DATA_BYTES
                # compute global byte offset of this codeword inside assembled payload
                header_len = 64 if packet_idx == 0 else 4
                # local payload offset: first byte of this cw relative to start of packet payload (after header drop)
                local_payload_offset = ci * RS_DATA_BYTES - header_len
                if local_payload_offset < 0:
                    # this codeword overlaps packet header area
                    location = "inside packet header"
                    global_offset = bytes_before_packet + 0
                else:
                    global_offset = bytes_before_packet + local_payload_offset
                    location = ("useful area" if (0 <= global_offset < expected_total) else "padding/overflow")
                # log failing codeword input for diagnostics including whether it hits useful payload
                #print(f"[RX-DBG-PKT] cw_idx={ci} RS_FAIL cw_in={cw_in_pkt} exc={e} global_offset={global_offset} location={location}")
            decoded_blocks.append(msg)
        # choose candidate with highest rs_ok
        if rs_ok > best_result[1]:
            best_result = (b"".join(decoded_blocks), rs_ok, cand)
            # Save last successful equalized symbols and equalizer for plotting/debug
            try:
                # init accumulation lists if absent
                if 'rx_syms_list' not in globals():
                    globals()['rx_syms_list'] = []
                if 'Hk_smooth_list' not in globals():
                    globals()['Hk_smooth_list'] = []
                # store small slice to limit memory
                rx_slice = rx_syms_pkt[:4096].copy() if isinstance(rx_syms_pkt, np.ndarray) else np.array([], dtype=complex)
                globals()['rx_syms_list'].append(rx_slice)
                globals()['Hk_smooth_list'].append(Hk_s.copy() if isinstance(Hk_s, np.ndarray) else np.array([], dtype=complex))
                globals()['last_packet_used_pre'] = cand
                globals()['last_packet_rs_ok'] = rs_ok
            except Exception:
                pass
            # early accept high success
            if rs_ok >= n_cw * 0.9:
                break
    return best_result

    


# -----------------------
# Main TX/RX workflow
# -----------------------
if __name__ == "__main__":
  # build_preamble now returns ZC,ZC,pilot,pilot
  preamble_td = build_preamble()

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
      # --- RECORD FROM MICROPHONE: streaming buffer + incremental preamble search ---
      print("Запись с микрофона (stream) — ждите...")

      # Thread-safe growing buffer (callback appends pieces to buf_list)
      buf_list = []               # list of numpy arrays appended by callback
      buf_len_holder = [0]        # mutable container to hold total length without nonlocal
      # total number of samples removed from the logical start of the recording
      globals()['base_offset'] = 0
      stream_closed = False

      # callback appends frames (float32) to buf_list
      def cb(indata, frames, time, status):
        if status:
          print("[MIC-STAT]", status)
        arr = indata[:,0].astype(np.float64, copy=True)
        buf_list.append(arr)
        buf_len_holder[0] += arr.size

      # start stream
      stream = sd.InputStream(samplerate=fs, channels=1, callback=cb, dtype='float32')
      stream.start()

      try:
        # helper to get contiguous view of buffer without destroying original pieces
        def get_buffer_array():
          if len(buf_list) == 0:
            return np.array([], dtype=np.float64)
          if len(buf_list) == 1:
            return buf_list[0]
          return np.concatenate(buf_list)

        # helper to drop consumed prefix up to index 'consumed'
        def drop_prefix(consumed):
          if consumed <= 0:
            return
          remaining = consumed
          new_list = []
          for piece in buf_list:
            if remaining >= piece.size:
              remaining -= piece.size
              continue
            else:
              if remaining > 0:
                new_piece = piece[remaining:].copy()
                new_list.append(new_piece)
                remaining = 0
              else:
                new_list.append(piece)
          buf_list[:] = new_list
          buf_len_holder[0] = sum(p.size for p in buf_list)
          # update global base offset so absolute indices remain consistent
          globals()['base_offset'] = globals().get('base_offset', 0) + consumed
        # parameters
        PREAMBLE_LEN = len(preamble_td)
        # ZC halves positions inside preamble (with CP)
        zc1_td = preamble_td[0:SYMBOL_LEN]
        zc2_td = preamble_td[SYMBOL_LEN:2*SYMBOL_LEN]

        first_preamble_abs = None
        first_packet_decoded = False
        decoded_packet_indices = set()
        packet_blocks_val = DEFAULT_PACKET_BLOCKS

        print("[MIC] streaming loop started, waiting for preamble detection...")
        import time as _time
        search_step_sleep = 0.05

        while True:
          # ensure enough samples to search for zc1
          if buf_len_holder[0] < PREAMBLE_LEN:
            _time.sleep(search_step_sleep)
            continue

          rx_stream = get_buffer_array()

          # search for first preamble if not found yet
          if first_preamble_abs is None:
            try:
              corr1 = fftconvolve(rx_stream, zc1_td[::-1], mode='valid')
            except Exception:
              corr1 = np.abs(rx_stream)
            abs_corr1 = np.abs(corr1)
            if abs_corr1.size == 0:
              _time.sleep(search_step_sleep)
              continue

            from scipy.signal import find_peaks
            peak_val1 = np.max(abs_corr1)
            if peak_val1 == 0:
              _time.sleep(search_step_sleep)
              continue
            thr1 = 0.5 * peak_val1
            peaks1, props1 = find_peaks(abs_corr1, height=thr1, distance=SYMBOL_LEN//2)
            if len(peaks1) == 0:
              peaks1, props1 = find_peaks(abs_corr1, height=0.25*peak_val1, distance=SYMBOL_LEN//2)
            peak_heights1 = props1['peak_heights'] if 'peak_heights' in props1 else abs_corr1[peaks1]
            candidates1 = sorted(zip(peaks1, peak_heights1), key=lambda x: -x[1])

            found = False
            for cand_idx, cand_h in candidates1[:12]:
               # candidate start of first ZC half (index into corr1 / rx_stream)
               zc1_start = int(cand_idx)
               zc2_expected_pos = zc1_start + SYMBOL_LEN

               # --- 1) ensure enough samples for zc2 check ---
               if zc2_expected_pos + SYMBOL_LEN > rx_stream.size:
                   # not enough samples yet to confirm second half
                   print(f"[MIC-DBG-CAND] cand {zc1_start} skipped: insufficient samples for zc2 (need {zc2_expected_pos+SYMBOL_LEN}, have {rx_stream.size})")
                   continue

               # --- 2) robust local noise estimation around zc1 candidate (for corr thresholding) ---
               win_lo = max(0, zc1_start - 4*SYMBOL_LEN)
               win_hi = min(len(abs_corr1), zc1_start + 4*SYMBOL_LEN)
               local_corr_window = abs_corr1[win_lo:win_hi] if win_hi > win_lo else abs_corr1
               if local_corr_window.size == 0:
                   continue
               noise_floor = np.median(local_corr_window)
               mad = np.median(np.abs(local_corr_window - noise_floor))
               noise_rms = 1.4826 * mad if mad > 0 else np.std(local_corr_window)
               # soften primary threshold slightly to allow borderline candidates in low-SNR mic input
               global_rel_thr = 0.35
               k_noise = 6.0
               thr1_local = max(global_rel_thr * peak_val1, noise_floor + k_noise * noise_rms)
               thr1_local *= 0.8   # reduce by 20% to avoid losing real preambles at low SNR
               if cand_h < thr1_local:
                   print(f"[MIC-DBG-CHECK] cand {zc1_start} rejected: cand_h {cand_h:.6g} < thr1_local {thr1_local:.6g}")
                   continue

               # --- NEW: correlate whole preamble block (ZC1,ZC2,PILOT1,PILOT2) in one pass ---
               # full preamble length in samples (4 OFDM symbols with CP presumed present in preamble_td)
               FULL_PREAMBLE_LEN = 4 * SYMBOL_LEN
               if zc1_start + FULL_PREAMBLE_LEN > rx_stream.size:
                   # not enough samples yet to check full preamble
                   print(f"[MIC-DBG-CAND] cand {zc1_start} skipped: insufficient samples for full preamble (need {zc1_start+FULL_PREAMBLE_LEN}, have {rx_stream.size})")
                   continue
 
               try:
                   seg_full = rx_stream[zc1_start:zc1_start+FULL_PREAMBLE_LEN]
                   pre_full_td = preamble_td[0:FULL_PREAMBLE_LEN]
                   # complex correlation preserving phase across entire preamble
 
                   c_full = np.vdot(seg_full, pre_full_td)
                   mag_full = np.abs(c_full)
               except Exception as e:
                   print(f"[MIC-DBG-CAND] cand {zc1_start} full-corr failed: {e}")
                   continue
 
               # robust local noise estimate around full preamble interval
               win_lo_full = max(0, zc1_start - 6*SYMBOL_LEN)
               win_hi_full = min(len(abs_corr1), zc1_start + FULL_PREAMBLE_LEN + 6*SYMBOL_LEN)
               local_full = np.abs(rx_stream[win_lo_full:win_hi_full]) if win_hi_full > win_lo_full else np.abs(rx_stream)
               noise_floor_full = np.median(local_full) if local_full.size>0 else 0.0
               mad_full = np.median(np.abs(local_full - noise_floor_full)) if local_full.size>0 else 0.0
               noise_rms_full = 1.4826 * mad_full if mad_full > 0 else (np.std(local_full) if local_full.size>0 else 0.0)
 
               # absolute threshold for full-correlation (tuned for mic input)
               full_rel_thr = 0.7              # fraction of peak (coarse)
               k_noise_full = 14              # noise multiplier
               thr_full = max(full_rel_thr * peak_val1 * FULL_PREAMBLE_LEN / float(SYMBOL_LEN), noise_floor_full + k_noise_full * noise_rms_full)
               thr_full *= 0.65   # relax slightly for mic input
 
               # relative fallback: also compute simpler half/pilot correlations to allow soft acceptance
               # compute ZC halves and pilot mags quickly for fallback decisions
               try:
                   seg_zc1 = rx_stream[zc1_start:zc1_start+SYMBOL_LEN]
                   seg_zc2 = rx_stream[zc1_start+SYMBOL_LEN:zc1_start+2*SYMBOL_LEN]
                   seg_p1  = rx_stream[zc1_start+2*SYMBOL_LEN:zc1_start+3*SYMBOL_LEN]
                   seg_p2  = rx_stream[zc1_start+3*SYMBOL_LEN:zc1_start+4*SYMBOL_LEN]
                   c1 = np.vdot(seg_zc1, zc1_td)
                   c2 = np.vdot(seg_zc2, zc2_td)
                   cp1 = np.vdot(seg_p1, preamble_td[2*SYMBOL_LEN:3*SYMBOL_LEN])
                   cp2 = np.vdot(seg_p2, preamble_td[3*SYMBOL_LEN:4*SYMBOL_LEN])
                   mag1 = np.abs(c1); mag2 = np.abs(c2); mag_p1 = np.abs(cp1); mag_p2 = np.abs(cp2)
               except Exception:
                   mag1 = mag2 = mag_p1 = mag_p2 = 0.0
 
               # accept if full-correlation is strong
               passed_full = (mag_full >= thr_full)
 
               # fallback accept if combined relative evidence from halves+pilots is strong
               rel_thresh_zc = 0.25   # require each ZC half magnitude >= 25% of max peak
               rel_thresh_p  = 0.20   # pilots relative threshold
               passed_rel_parts = ((mag1 >= rel_thresh_zc * peak_val1) and (mag2 >= rel_thresh_zc * peak_val1)) or ((mag_p1 >= rel_thresh_p * peak_val1) and (mag_p2 >= rel_thresh_p * peak_val1))
 
               if not (passed_full or passed_rel_parts):
                   print(f"[MIC-DBG-CHECK] cand {zc1_start} rejected: mag_full {mag_full:.6g} < thr_full {thr_full:.6g} and parts weak (z1={mag1:.6g},z2={mag2:.6g},p1={mag_p1:.6g},p2={mag_p2:.6g})")
                   continue
 
               # estimate FOE from full correlation phase across full preamble for better robustness
               delta_phi_full = np.angle(c_full)
               T_full = (3.0 * SYMBOL_LEN) / float(fs)   # effective spacing across parts; empirical
               f_err_loc = delta_phi_full / (2.0 * np.pi * T_full)
               f_err_max = 400.0
               if abs(f_err_loc) > f_err_max:
                   print(f"[MIC-DBG-CHECK] cand {zc1_start} rejected: estimated FOE from full corr {f_err_loc:.1f} Hz exceeds {f_err_max} Hz")
                   continue
 
               # accept candidate
               # accept candidate
               first_preamble_abs = zc1_start
               print(f"[MIC-DBG] FIRST PREAMBLE FOUND at buffer_index={first_preamble_abs} mag_full={mag_full:.6g}  thr_full={thr_full:.6g} f_err={f_err_loc:.2f}Hz")
               # record absolute position of first preamble in global coordinates
               # current rx_stream starts at globals()['base_offset'] samples from original recording start
               first_global = globals().get('base_offset', 0) + first_preamble_abs
               globals()['first_preamble_global'] = int(first_global)
               # drop prefix up to first_preamble_abs so remaining buffer begins at the preamble
               if first_preamble_abs > 0:
                 drop_prefix(first_preamble_abs)
               # update globals rx and abs_corr to reflect trimmed buffer
               globals()['rx'] = get_buffer_array()
               try:
                 globals()['abs_corr'] = np.abs(fftconvolve(globals()['rx'], preamble_td[::-1], mode='valid'))
               except Exception:
                 globals()['abs_corr'] = np.abs(globals()['rx']) if globals()['rx'].size > 0 else np.array([])
               # set local sync_idx: position of first preamble in the local (trimmed) buffer is zero
               sync_idx = 0
               globals()['sync_idx'] = sync_idx
               print(f"[MIC-DBG] registered first_preamble_global={globals().get('first_preamble_global')} base_offset={globals().get('base_offset')}")
               found = True
               break

 
               # Accept candidate if any strong condition met:
               #  - absolute strong correlation on second half, or
               #  - relative strength vs first half, or
               #  - slightly below thresholds but excellent phase consistency
               passed_abs = (mag2 >= thr2_local)
               passed_rel = (mag2 >= rel_cand * mag1)

               # phase consistency (already computed later); precompute rough delta_phi and f_err for soft acceptance
               delta_phi_try = np.angle(c1 * np.conj(c2)) if ('c1' in locals() and 'c2' in locals()) else 0.0
               T_between = SYMBOL_LEN / float(fs)
               f_err_try = delta_phi_try / (2.0 * np.pi * T_between)
               f_err_strict = 100.0  # Hz — tighter criterion for soft acceptance
 
               # soft accept if mag2 is slightly below thresholds but phase error is tiny and mag2 above min_abs_mag2
               soft_margin = 0.8
               passed_soft = (mag2 >= soft_margin * thr2_local and abs(f_err_try) <= f_err_strict and mag2 >= min_abs_mag2)
 
               if not (passed_abs or passed_rel or passed_soft):
                   print(f"[MIC-DBG-CHECK] cand {zc1_start} rejected: mag2 {mag2:.6g} thr2_local {thr2_local:.6g} rel_req {rel_cand*mag1:.6g} f_err={f_err_try:.2f}Hz")
                   continue

               # --- 5) phase consistency / FOE check between ZC halves ---
               delta_phi = np.angle(c1 * np.conj(c2))
               T_between = SYMBOL_LEN / float(fs)
               f_err_loc = delta_phi / (2.0 * np.pi * T_between)
               f_err_max = 200.0
               if abs(f_err_loc) > f_err_max:
                   print(f"[MIC-DBG-CHECK] cand {zc1_start} rejected: estimated FOE {f_err_loc:.1f} Hz exceeds {f_err_max} Hz")
                   continue

               # candidate passed all checks -- accept
               first_preamble_abs = zc1_start
               print(f"[MIC-DBG] FIRST PREAMBLE FOUND at buffer_index={first_preamble_abs} cand_h={cand_h:.6g} mag2={mag2:.6g} f_err={f_err_loc:.2f}Hz")
               found = True
               break

            if not found:
              _time.sleep(search_step_sleep)
              continue

            # Wait for enough samples for full first packet + one extra OFDM symbol
            required_samples = PREAMBLE_LEN + packet_blocks_val * SYMBOL_LEN + SYMBOL_LEN
            while buf_len_holder[0] < first_preamble_abs + required_samples:
              need = first_preamble_abs + required_samples - buf_len_holder[0]
              print(f"[MIC-DBG] waiting for samples: need={need}, have={buf_len_holder[0]}, waiting...")
              _time.sleep(search_step_sleep)
              rx_stream = get_buffer_array()

            # prepare globals expected by decode_packet_at_candidate
            try:
              globals()['abs_corr'] = np.abs(fftconvolve(rx_stream, preamble_td[::-1], mode='valid'))
            except Exception:
              globals()['abs_corr'] = np.abs(rx_stream)
            globals()['rx'] = rx_stream

            # call existing per-packet decoder
            # decode helper may be defined later in file; wait briefly until it appears in globals
            wait_start = _time.time()
            while 'decode_packet_at_candidate' not in globals():
              if _time.time() - wait_start > 5.0:
                print("[MIC-ERR] decode_packet_at_candidate not available after 5s, aborting this attempt")
                break
              _time.sleep(0.05)

            if 'decode_packet_at_candidate' not in globals():
              # fallback: treat as failure so loop will continue listening for next candidate
              print("[MIC-ERR] decode_packet_at_candidate still missing; will continue listening")
              drop_prefix(first_preamble_abs + SYMBOL_LEN)
              first_preamble_abs = None
              continue

            # call helper via globals to avoid NameError regardless of definition order
            pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = globals()['decode_packet_at_candidate'](first_preamble_abs, packet_blocks_val, packet_idx=0, bytes_before_packet=0, expected_total=0)

            if pkt0_decoded is None or len(pkt0_decoded) < 64:
              print("[MIC-ERR] failed to decode first packet after detection; will continue listening and try next candidate")
              drop_prefix(first_preamble_abs + SYMBOL_LEN)
              first_preamble_abs = None
              continue

            header64 = pkt0_decoded[:64]
            try:
              hdr = parse_header(header64)
            except Exception as e:
              print("[MIC-ERR] parse_header failed for first packet:", e)
              hdr = None

            print(f"[MIC-DBG] first-packet decoded: used_pre={pkt0_used_pre} RS_OK={pkt0_rs_ok} header_hex={header64.hex()[:128]}")

            # Drop prefix up to first_preamble_abs to avoid re-detection of same samples
            if first_preamble_abs > 0:
              drop_prefix(first_preamble_abs)

            # update globals with trimmed buffer for the rest of pipeline
            globals()['rx'] = get_buffer_array()
            try:
              globals()['abs_corr'] = np.abs(fftconvolve(globals()['rx'], preamble_td[::-1], mode='valid'))
            except Exception:
              globals()['abs_corr'] = np.abs(globals()['rx']) if globals()['rx'].size > 0 else np.array([])

            # fill header-derived params if available
            if hdr is not None:
              mode_rx = hdr['mode']
              total_sz = hdr['data_len']
              fname = hdr['filename']
              packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)
              packet_blocks_val = packet_blocks_from_hdr
              print(f"[MIC-INFO] transmission header: data_len={total_sz} filename={fname} packet_blocks={packet_blocks_from_hdr}")
            else:
              packet_blocks_from_hdr = DEFAULT_PACKET_BLOCKS

            decoded_packet_indices.add(0)
            first_packet_decoded = True
            break

          else:
            _time.sleep(search_step_sleep)
            continue

        # end streaming loop
      finally:
        try:
          stream.stop()
          stream.close()
        except Exception:
          pass

      # ensure globals rx and abs_corr are set for rest of pipeline
      if 'rx' not in globals() or globals().get('rx', None) is None:
        globals()['rx'] = get_buffer_array()
      if 'abs_corr' not in globals() or globals().get('abs_corr', None) is None:
        try:
          globals()['abs_corr'] = np.abs(fftconvolve(globals()['rx'], preamble_td[::-1], mode='valid'))
        except Exception:
          globals()['abs_corr'] = np.abs(globals()['rx']) if globals()['rx'].size > 0 else np.array([])

      # set initial sync_idx for the rest of existing logic
      if 'last_packet_used_pre' in globals():
        sync_idx = int(globals().get('last_packet_used_pre', 0))
      else:
        ac = globals().get('abs_corr', np.array([]))
        if ac.size > 0:
          sync_idx = int(np.argmax(ac))
        else:
          sync_idx = 0

      print(f"[MIC-DBG] initial sync_idx set to {sync_idx} (ready to continue packet decoding loop)")

    # initial correlation and candidate selection (unchanged)
    corr = fftconvolve(rx, preamble_td[::-1], mode='valid')
    abs_corr = np.abs(corr)
    if abs_corr.size == 0:
        print("[RX-ERR] correlation empty, cannot find preambles")
        sync_idx = sync_by_corr(rx, preamble_td)
    else:
        from scipy.signal import find_peaks
        peak_val = np.max(abs_corr)
        threshold = 0.9 * peak_val if peak_val != 0 else 0.0
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
                # For initial candidate check we still estimate Hk using pilot positions (which are after ZC pair)
                # derive indices of pilot symbols in preamble_td:
                # preamble_td layout: [ZC (with CP)], [ZC], [pilot], [pilot]
                # ideal pilot symbol (time domain without CP) is at position:
                # offset = 2*(Nfft+Ncp) so pilot1 start = cand_sync + 2*SYMBOL_LEN
                pilot1_start = cand_sync + 2*SYMBOL_LEN
                rx_pre1_try = rx[pilot1_start + Ncp : pilot1_start + Ncp + Nfft]
                pilot2_start = pilot1_start + SYMBOL_LEN
                rx_pre2_try = rx[pilot2_start + Ncp : pilot2_start + Ncp + Nfft]
                S_ref = np.fft.fft(preamble_td[2*SYMBOL_LEN + Ncp : 2*SYMBOL_LEN + Ncp + Nfft])  # ideal pilot symbol
                R1t, R2t = np.fft.fft(rx_pre1_try), np.fft.fft(rx_pre2_try)
                Hk_try = ((R1t[subc_inds]/S_ref[subc_inds]) + (R2t[subc_inds]/S_ref[subc_inds]))/2
                Hk_mag_s_try = np.clip(medfilt(np.abs(Hk_try), 5), 1/2.0, None)
                Hk_smooth_try = Hk_mag_s_try * np.exp(1j*np.angle(Hk_try))
            except Exception:
                continue

            try_nblk = min(8, nblk_try)
            seg_try = rx[start_try : start_try + try_nblk * SYMBOL_LEN]
            if len(seg_try) < try_nblk * SYMBOL_LEN:
                continue
            frames_try = seg_try.reshape(-1, SYMBOL_LEN)
            # quick demap with current Hk_smooth_try
            try:
                rx_subc_try = np.concatenate([np.fft.fft(frm[Ncp:])[subc_inds] / Hk_smooth_try for frm in frames_try])
            except Exception:
                continue
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

    # Decode first packet only (local per-packet decode) to obtain transmission header.
    # We will not perform a global RS decode of the whole region here.
    # Use decode_packet_at_candidate around sync_idx to get first packet's decoded bytes.
        # per-packet decoding loop
        # try center at sync_idx (preamble detected there). packet_blocks default used as expected blocks for first attempt.
    # before decoding packet 0, bytes_before_packet == 0
    pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = decode_packet_at_candidate(sync_idx, DEFAULT_PACKET_BLOCKS, packet_idx=0, bytes_before_packet=0, expected_total=0)
    if pkt0_decoded is None or len(pkt0_decoded) < 64:
      print("[RX-ERR] unable to decode first packet header using per-packet decoder")
      sys.exit(1)
    header64 = pkt0_decoded[:64]
    try:
      hdr = parse_header(header64)
    except Exception as e:
      print("[RX-ERR] parse_header failed for first packet:", e)
      print("[RX-ERR] header64_hex:", header64.hex())
      sys.exit(1)
    # report what we obtained
    print(f"[RX-DBG] first-packet local decode: used_pre={pkt0_used_pre} RS_OK={pkt0_rs_ok} header_hex={header64.hex()[:256]}")


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

    
    assembled_chunks = []
    bytes_collected = 0
    expected_total = total_sz
    packet_blocks_val = packet_blocks_from_hdr
    hdr_first_bytes = header64

    bytes_collected = 0
    for pkt_idx, pref in enumerate(expected_abs):
        # decode each packet separately; provide packet index, bytes already collected, and expected total for debug
        decoded_bytes, rs_ok_count, used_preamble = decode_packet_at_candidate(pref, packet_blocks_val, packet_idx=pkt_idx, bytes_before_packet=bytes_collected, expected_total=expected_total)
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

    # Visualization (accumulated across packets if available)
    try:
        import matplotlib
        try:
            matplotlib.pyplot.ion()
        except Exception:
            pass

        figs_shown = 0

        # Constellation plot: prefer accumulated list, otherwise single snapshot
        if 'rx_syms_list' in globals() and isinstance(globals()['rx_syms_list'], list) and len(globals()['rx_syms_list']) > 0:
            # take up to first M symbols from each packet, concatenate for plot
            M = 2048
            concat_list = [r[:M] for r in globals()['rx_syms_list'] if isinstance(r, np.ndarray) and r.size > 0]
            if len(concat_list) > 0:
                concat = np.concatenate(concat_list)
                plt.figure(figsize=(5,5))
                plt.plot(np.real(concat), np.imag(concat), 'o', markersize=2, alpha=0.4)
                plt.axhline(0, color='grey', linewidth=0.5); plt.axvline(0, color='grey', linewidth=0.5)
                plt.title(f"RX Constellation (combined from {len(concat_list)} packets)")
                plt.xlabel("In-phase"); plt.ylabel("Quadrature"); plt.grid(True); plt.axis('equal')
                figs_shown += 1
            else:
                print("[PLOT] rx_syms_list present but empty entries, skipping constellation")
        elif 'rx_syms' in globals() and isinstance(rx_syms, np.ndarray) and rx_syms.size > 0:
            plt.figure(figsize=(5,5))
            plt.plot(np.real(rx_syms), np.imag(rx_syms), 'o', markersize=2, alpha=0.6)
            plt.axhline(0, color='grey', linewidth=0.5); plt.axvline(0, color='grey', linewidth=0.5)
            plt.title("RX Constellation (last packet)")
            plt.xlabel("In-phase"); plt.ylabel("Quadrature"); plt.grid(True); plt.axis('equal')
            figs_shown += 1
        else:
            print("[PLOT] no rx_syms available, skipping constellation")

        # Equalizer gain vs frequency: average over accumulated Hk_smooth_list if present
        if 'Hk_smooth_list' in globals() and isinstance(globals()['Hk_smooth_list'], list) and len(globals()['Hk_smooth_list']) > 0:
            Hlist = [h for h in globals()['Hk_smooth_list'] if isinstance(h, np.ndarray) and h.size == len(subc_inds)]
            if len(Hlist) > 0:
                Hstack = np.vstack(Hlist)
                Hmedian = np.median(Hstack, axis=0)
                freqs = subc_inds * fs / float(Nfft)
                with np.errstate(divide='ignore', invalid='ignore'):
                    eq_gain = 1.0 / np.abs(Hmedian)
                    eq_gain = np.clip(eq_gain, 0, np.percentile(eq_gain[np.isfinite(eq_gain)], 99) * 1.5 if np.any(np.isfinite(eq_gain)) else 1.0)
                plt.figure(figsize=(8,4))
                plt.plot(freqs, eq_gain, "o-", color="darkorange", label="Median Gain")
                plt.title(f"Equalizer Gain vs Frequency (median of {Hstack.shape[0]} packets)")
                plt.xlabel("Frequency (Hz)"); plt.ylabel("Gain"); plt.grid(True); plt.legend()
                figs_shown += 1
            else:
                print("[PLOT] Hk_smooth_list present but incompatible entries, skipping equalizer gain")
        elif 'Hk_smooth' in globals() and isinstance(Hk_smooth, np.ndarray) and Hk_smooth.size > 0:
            freqs = subc_inds * fs / float(Nfft)
            with np.errstate(divide='ignore', invalid='ignore'):
                eq_gain = 1.0/np.abs(Hk_smooth)
                eq_gain = np.clip(eq_gain, 0, np.percentile(eq_gain[np.isfinite(eq_gain)], 99) * 1.5 if np.any(np.isfinite(eq_gain)) else 1.0)
            plt.figure(figsize=(8,4))
            plt.plot(freqs, eq_gain, "o-", color="darkorange", label="Gain")
            plt.title("Equalizer Gain vs Frequency (last packet)")
            plt.xlabel("Frequency (Hz)"); plt.ylabel("Gain"); plt.grid(True); plt.legend()
            figs_shown += 1
        else:
            print("[PLOT] no Hk_smooth available, skipping equalizer gain")

        if figs_shown > 0:
            plt.show(block=True)
        else:
            print("[PLOT] no figures to show")
    except Exception as e:
        print("[PLOT] failed to produce plots:", e)

