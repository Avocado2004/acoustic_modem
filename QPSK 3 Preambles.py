#!/usr/bin/env python3
"""
OFDM Acoustic Modem — transmission MSB marking change
Изменения:
- В transmission header первые четыре старших бита первого байта устанавливаются в 1 (mask 0xF0).
- В приёмнике проверка, является ли заголовок Transmission, теперь делается как (flags & 0xF0) != 0.
- Везде, где раньше проверялось (flags & 0x80), заменено на проверку верхних 4 бит.
- Логика короткого 4-байтового packet header не изменена (верхние 4 бита должны быть 0).
- Изменения применены и к веткам file/live приёма.
Остальная логика сохранена.
Дополнительно:
- Плоты ограничены от синхронизации до последнего полезного символа.
- При RS FAIL после синхронизации печатается частота поднесущей, соответствующей нерасшифрованному блокy.
Добавлено:
- CRC32 рассчитанный по передаваемым данным кладётся в Transmission Header (hdr[52:56]).
- На приёме после сборки всех байт вычисляется CRC32 и сравнивается с полем в заголовке — печатается статус.

Дополнительные изменения по запросу:
- Поднята верхняя граница поднесущих так, чтобы получилось ровно 48 поднесущих (Nsub = 48).
- Добавлены проверки и диагностические сообщения, согласующие RS кодирование/распаковку с BITS_PER_OFDM_SYMBOL = 96.
"""

import os
import sys
import math
import struct
import zlib
import numpy as np
from scipy.signal import fftconvolve, medfilt, correlate, find_peaks
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
from reedsolo import RSCodec
import threading
import time
from collections import deque

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

# Поднесущие: установить так, чтобы было ровно 48 поднесущих в диапазоне начинающемся от ~300 Hz
k_low = int(math.ceil(300 / df))
REQUIRED_NSUB = 48
k_high = k_low + REQUIRED_NSUB - 1
subc_inds = np.arange(k_low, k_high + 1)
Nsub = len(subc_inds)

f_low_hz = k_low * df
f_high_hz = k_high * df
print(f"[CFG] df={df:.3f} Hz, k_low={k_low} -> {f_low_hz:.1f} Hz, k_high={k_high} -> {f_high_hz:.1f} Hz, Nsub={Nsub}")

# FEC (Reed-Solomon)
RS_DATA_BYTES = 8
RS_PARITY_BYTES = 4
rs = RSCodec(RS_PARITY_BYTES)

RS_CW_BYTES = RS_DATA_BYTES + RS_PARITY_BYTES
RS_CW_BITS = RS_CW_BYTES * 8

MAX_PAYLOAD_SIZE = 1 << 30

# Check compatibility: BITS_PER_OFDM_SYMBOL should equal RS_CW_BITS for 1 RS cw per OFDM symbol
BITS_PER_OFDM_SYMBOL = Nsub * 2  # QPSK: 2 bits per subcarrier symbol
if BITS_PER_OFDM_SYMBOL != RS_CW_BITS:
    print(f"[WARN] bits per OFDM symbol = {BITS_PER_OFDM_SYMBOL}, RS cw bits = {RS_CW_BITS}. Expected equality for 1 cw/symbol.")
else:
    print(f"[CFG] One OFDM symbol carries exactly one RS codeword ({RS_CW_BYTES} bytes, {RS_CW_BITS} bits).")

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

# limit diagnostic RS_FAIL prints globally per receive to avoid flood
_MAX_RS_FAIL_PRINTS_GLOBAL = 1

# --- PACKET-LEVEL AGC (insert after cand chosen, before pkt_data_start computation) ---
# параметры AGC
TARGET_RMS = 0.5    # целевой RMS для полезной части пакета (подберите экспериментально)
MIN_RMS = 1e-12
# параметры символьного AGC
SYMBOL_TARGET_RMS = 0.4  # можно совпадать с TARGET_RMS или вынести отдельно
AGC_ALPHA = 0.15           # экспоненциальный коэффициент для скользящего оценщика

# target RMS (амплитуда) для каждого OFDM-символа на стороне TX
SYMBOL_TX_TARGET = 1.0

# -----------------------
# Active Constellation Extension (ACE) – simplified implementation
# -----------------------
ACE_MAX_ITERS = 5
ACE_PEAK_THRESHOLD = 1.01
ACE_STEP = 0.3
ACE_ALLOW_EXPANSION = True

def ace_reduce_peaks(ds, Nfft_local, subc_inds_local):
  try:
    if ds is None or len(ds) == 0:
      return ds
    X = np.zeros(Nfft_local, dtype=complex)
    X[subc_inds_local] = ds
    X[-subc_inds_local] = np.conj(ds)
    for it in range(ACE_MAX_ITERS):
      x_td = np.fft.ifft(X)
      x_td_real = np.real(x_td)
      mag = np.abs(x_td_real)
      mean_mag = np.mean(mag) if mag.size > 0 else 0.0
      if mean_mag <= 0:
        break
      thresh = ACE_PEAK_THRESHOLD * mean_mag
      peak_idx = np.where(mag > thresh)[0]
      if peak_idx.size == 0:
        break
      corr_td = np.zeros_like(x_td, dtype=complex)
      exceed = mag[peak_idx] - thresh
      corr_td[peak_idx] = - (exceed / (mag[peak_idx] + 1e-12)) * x_td[peak_idx] * ACE_STEP
      Corr_fd = np.fft.fft(corr_td)
      Corr_proj = np.zeros_like(Corr_fd)
      Corr_proj[subc_inds_local] = Corr_fd[subc_inds_local]
      Corr_proj[-subc_inds_local] = Corr_fd[-subc_inds_local]
      X_new = X + Corr_proj
      ds_cand = X_new[subc_inds_local].copy()
      if ACE_ALLOW_EXPANSION:
        phases = np.angle(ds)
        mags_old = np.abs(ds)
        mags_cand = np.abs(ds_cand)
        mags_new = np.maximum(mags_old, mags_cand)
        mags_new = np.minimum(mags_new, mags_old * (1.0 + ACE_STEP))
        ds = mags_new * np.exp(1j * phases)
        X = np.zeros(Nfft_local, dtype=complex)
        X[subc_inds_local] = ds
        X[-subc_inds_local] = np.conj(ds)
      else:
        ds = ds_cand
        X = X_new
    return ds
  except Exception:
    return ds

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

# QPSK map/demap
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

  try:
    eps = 1e-12
    cur_rms = np.sqrt(np.mean(np.abs(ds)**2)) if ds.size > 0 else 0.0
    if cur_rms < eps:
      cur_rms = eps
    ds = ds / cur_rms * SYMBOL_TX_TARGET
  except Exception:
    pass

  ds = ace_reduce_peaks(ds, Nfft, subc_inds)

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
  n = np.arange(L)
  z = np.exp(-1j * np.pi * u * n * (n + 1) / float(L))
  z = z / np.sqrt(np.mean(np.abs(z)**2))
  return z

# -----------------------
# build_preamble: ZC, ZC, pilot, pilot
# -----------------------
def build_preamble(reps=1, zc_root=1):
  zc_seq = zc_root_sequence(zc_root, Nsub)
  S_zc = ofdm_symbol(zc_seq)
  pilot = np.full(Nsub, (1 + 1j) / np.sqrt(2))
  S_pilot = ofdm_symbol(pilot)
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
# - изменение: устанавливаем верхние 4 бита первого байта в единицы при формировании Transmission header
# - добавлено: поле CRC32 в hdr[52:56] (big-endian)
# -----------------------
def build_header(mode: bytes, data_len: int, filename_bytes: bytes = b'', packet_no: int = 0, version: int = 0, packet_blocks: int = DEFAULT_PACKET_BLOCKS, crc32: int = 0) -> bytes:
    """
    Build 64-byte Transmission header.
    Now set top 4 bits of hdr[0] to 1 to mark Transmission header robustly.
    CRC32 (4 bytes, big-endian) is stored at hdr[52:56].
    """
    if isinstance(mode, (bytes, bytearray)) and len(mode) > 0 and mode[:1] == b'F':
        tx_type = 0b11
    elif isinstance(mode, (bytes, bytearray)) and len(mode) > 0 and mode[:1] == b'T':
        tx_type = 0b10
    else:
        tx_type = 0b00

    # set top 4 bits to 1 (0xF0) and keep tx_type low bits
    flags = 0xF0 | (tx_type & 0x03)
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
    # store CRC32 at 52..55 (big-endian). If crc32 is 0 default, stores zero.
    hdr[52:56] = struct.pack('>I', int(crc32) & 0xFFFFFFFF)
    # remaining bytes (56..63) keep zero
    hdr[56:64] = b'\x00' * 8
    return bytes(hdr)

def make_packet_header_bytes(packet_no: int, tx_type: int) -> bytes:
    # keep top 4 bits zero for short packet header
    flags = (0x00) | (tx_type & 0x03)
    hdr = bytearray(4)
    hdr[0] = flags & 0xFF
    hdr[1] = (packet_no >> 16) & 0xFF
    hdr[2] = (packet_no >> 8) & 0xFF
    hdr[3] = packet_no & 0xFF
    return bytes(hdr)

def parse_header(header64: bytes):
    """
    Modified parsing:
    Transmission header if upper nibble of header64[0] != 0 (i.e., any of top 4 bits set).
    Also extracts CRC32 from hdr[52:56] if present.
    """
    if len(header64) < 1:
        raise ValueError("Empty header data")
    b0 = header64[0]
    is_transmission = bool(b0 & 0xF0)
    tx_type = b0 & 0x03
    if is_transmission:
        if len(header64) < 64:
            raise ValueError("Transmission header requires 64 bytes")
        version = int(header64[1])
        data_len = struct.unpack('>Q', header64[2:10])[0]
        name_len = struct.unpack('>I', header64[10:14])[0]
        name_bytes = header64[14:46]
        packet_no = struct.unpack('>I', header64[46:50])[0]
        packet_blocks = struct.unpack('>H', header64[50:52])[0]
        crc32_val = struct.unpack('>I', header64[52:56])[0]
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
            'packet_blocks': packet_blocks,
            'crc32': crc32_val
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
# остальные вспомогательные функции
# -----------------------
def bytes_to_ofdm_blocks_bytes(bstream: bytes):
    # разбиваем на RS DATA байты и кодируем каждую кодовую единицу
    blocks_local = [bstream[i:i+RS_DATA_BYTES] for i in range(0, len(bstream), RS_DATA_BYTES)]
    if len(blocks_local[-1]) < RS_DATA_BYTES:
        blocks_local[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks_local[-1]))
    encoded = [rs.encode(b) for b in blocks_local]   # каждый элемент — RS_CW_BYTES bytes
    bits_blocks_local = [bytes_to_bits(b) for b in encoded]  # каждый — RS_CW_BITS бит

    data_bits_local = np.concatenate(bits_blocks_local) if len(bits_blocks_local) > 0 else np.array([], dtype=int)
    td_local, nblocks_local = build_data_td(data_bits_local) if data_bits_local.size > 0 else (np.array([], dtype=float), 0)
    return td_local, nblocks_local

def simulate_packet_positions(total_data_len_bytes, filename_bytes, mode_is_text, packet_blocks_local,
                              preamble_len, symbol_len, gap_samples):
    positions = []
    def max_payload_for_header(header_bytes):
        step = RS_DATA_BYTES
        if total_data_len_bytes <= step:
            td, nblk = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * total_data_len_bytes))
            if nblk <= packet_blocks_local:
                return total_data_len_bytes
            return 0
        hi_bound = min(total_data_len_bytes, packet_blocks_local * RS_DATA_BYTES * 8)
        lo = 0
        hi = step if step < hi_bound else hi_bound
        while True:
            td, nblk = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * hi))
            if nblk > packet_blocks_local:
                break
            if hi >= hi_bound:
                break
            hi = min(hi * 2, hi_bound)
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
    # first packet will contain three consecutive 64-byte transmission headers
    single_hdr = build_header(b'T' if mode_is_text else b'F', total_data_len_bytes, filename_bytes=filename_bytes, packet_no=0, version=0, packet_blocks=packet_blocks_local)
    hdr_first = single_hdr + single_hdr + single_hdr
    remaining = total_data_len_bytes
    offset = 0
    pkt_no = 0

    best_first = max_payload_for_header(hdr_first)
    payload = min(remaining, best_first)
    td, nblk = bytes_to_ofdm_blocks_bytes(hdr_first + (b'\x00' * payload))
    packet_samples = preamble_len + nblk * symbol_len + gap_samples
    positions.append(offset)
    offset += packet_samples
    remaining -= payload
    pkt_no += 1

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

# soft clipping utilities
TARGET_CREST_DB = 6
CREST_TOLERANCE_DB = 0.1
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
# decode_packet_at_candidate
# -----------------------
def _try_alternate_demaps_and_rs(rx_syms_pkt, n_cw, packet_idx, bytes_before_packet):
    transforms = [
        lambda x: x,
        lambda x: np.conj(x),
        lambda x: -x,
        lambda x: x * 1j,
        lambda x: x * -1j,
    ]
    best = (None, 0)
    for tr in transforms:
        try:
            rx_t = tr(rx_syms_pkt)
        except Exception:
            continue
        bits_t = qpsk_demap(rx_t)
        cw_bits = RS_CW_BITS
        n_cw_t = len(bits_t) // cw_bits
        rs_ok_t = 0
        decoded_blocks_t = []
        for ci in range(min(n_cw_t, n_cw)):
            bstart = ci * cw_bits
            bbits = bits_t[bstart:bstart+cw_bits]
            if len(bbits) < cw_bits:
                bbits = np.concatenate((bbits, np.zeros(cw_bits - len(bbits), dtype=int)))
            bts = bits_to_bytes(bbits)
            try:
                msg = rs.decode(bts)[0]
                rs_ok_t += 1
            except Exception:
                msg = b'\x00' * RS_DATA_BYTES
            decoded_blocks_t.append(msg)
        if rs_ok_t > best[1]:
            best = (b"".join(decoded_blocks_t), rs_ok_t)
    return best

def decode_packet_at_candidate(pref_abs, packet_blocks_expected, packet_idx=0, bytes_before_packet=0, expected_total=0):
    best_result = (b'', 0, None)
    global abs_corr, rx
    if 'abs_corr' not in globals() or 'rx' not in globals():
        return best_result

    if '_rs_fail_prints_count' not in globals():
        globals()['_rs_fail_prints_count'] = 0

    wlo = max(0, pref_abs - Ncp)
    whi = min(len(abs_corr)-1, pref_abs + Ncp)
    window = abs_corr[wlo:whi+1] if whi >= wlo else np.array([])
    candidate_positions = []
    if window.size > 0:
      local_rel = np.argsort(window)[::-1]
      for r in local_rel[:6]:
        candidate_positions.append(wlo + int(r))
    grid = list(range(max(0, pref_abs - Ncp), min(len(abs_corr), pref_abs + Ncp + 1)))
    combined = []
    for p in candidate_positions + grid:
      if p not in combined:
        combined.append(p)

    try:
        zc_seq_ideal = zc_root_sequence(1, Nsub)
        S_zc_fd = np.zeros(Nfft, dtype=complex)
        S_zc_fd[subc_inds] = zc_seq_ideal
        S_zc_fd[-subc_inds] = np.conj(zc_seq_ideal)
    except Exception:
        S_zc_fd = None

    for cand in combined:
        try:
            rx_zc1 = rx[cand + Ncp : cand + Ncp + Nfft]
            rx_zc2 = rx[cand + SYMBOL_LEN + Ncp : cand + SYMBOL_LEN + Ncp + Nfft]
            cross = np.vdot(rx_zc1, rx_zc2)
            delta_phi = np.angle(cross)
            T_between = SYMBOL_LEN / float(fs)
            f_err_loc = delta_phi / (2.0 * np.pi * T_between)
            pilot1_start = cand + 2*SYMBOL_LEN
            rx_pre1 = rx[pilot1_start + Ncp : pilot1_start + Ncp + Nfft]
            pilot2_start = pilot1_start + SYMBOL_LEN
            rx_pre2 = rx[pilot2_start + Ncp : pilot2_start + Ncp + Nfft]
            t1_offset = (pilot1_start) / float(fs)
            t2_offset = (pilot2_start) / float(fs)
            time_vec = np.arange(Nfft) / float(fs)
            rx_pre1_corr = rx_pre1 * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t1_offset + time_vec))
            rx_pre2_corr = rx_pre2 * np.exp(-1j * 2.0 * np.pi * f_err_loc * (t2_offset + time_vec))
            R1t = np.fft.fft(rx_pre1_corr)
            R2t = np.fft.fft(rx_pre2_corr)
            phi_est = 0.0
            if S_zc_fd is not None:
                try:
                    Rzc = np.fft.fft(rx_zc1)
                    phi_est = np.angle(np.vdot(S_zc_fd, Rzc))
                except Exception:
                    phi_est = 0.0
            if phi_est != 0.0:
                R1t = R1t * np.exp(-1j * phi_est)
                R2t = R2t * np.exp(-1j * phi_est)
            S_ref = np.fft.fft(preamble_td[2*SYMBOL_LEN + Ncp : 2*SYMBOL_LEN + Ncp + Nfft])
            Hk_est = ((R1t[subc_inds]/S_ref[subc_inds]) + (R2t[subc_inds]/S_ref[subc_inds]))/2
            Hk_mag = np.clip(medfilt(np.abs(Hk_est), 5), 1/2.0, None)
            Hk_s = Hk_mag * np.exp(1j*np.angle(Hk_est))
        except Exception:
            continue
        

        try:
            pre_segment = rx[cand : cand + len(preamble_td)]
            pre_rms = np.sqrt(np.mean(pre_segment**2)) if pre_segment.size > 0 else MIN_RMS
            if pre_rms < MIN_RMS:
                pre_rms = MIN_RMS
            packet_gain = TARGET_RMS / pre_rms
        except Exception:
            packet_gain = 1.0

        pkt_data_start = cand + len(preamble_td)
        pkt_payload_samples = packet_blocks_expected * SYMBOL_LEN
        seg = packet_gain * rx[pkt_data_start : pkt_data_start + pkt_payload_samples]
        if len(seg) < pkt_payload_samples:
            continue
        frames = seg.reshape(packet_blocks_expected, SYMBOL_LEN)
        rx_subc_pkt = []
        for idxf, fr in enumerate(frames):
            frame_start_abs = pkt_data_start + idxf * SYMBOL_LEN
            t_frame_start = frame_start_abs / float(fs)
            useful = fr[Ncp:]
            tv = t_frame_start + np.arange(Nfft) / float(fs)
            useful_corr = useful * np.exp(-1j * 2.0 * np.pi * f_err_loc * tv)

            cur_rms = np.sqrt(np.mean(np.abs(useful)**2)) if useful.size > 0 else 1e-12
            last_agc = globals().get('last_agc_rms', cur_rms)
            est_rms = (1.0 - AGC_ALPHA) * last_agc + AGC_ALPHA * cur_rms
            globals()['last_agc_rms'] = est_rms
            if est_rms < 1e-12:
                est_rms = 1e-12
            gain_sym = SYMBOL_TARGET_RMS / est_rms
            useful = useful * gain_sym

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
        try:
            if 'phi_est' in locals() and phi_est != 0.0:
                rx_subc_pkt = rx_subc_pkt * np.exp(-1j * phi_est)
        except Exception:
            pass
        if subc_phases is not None and np.any(subc_phases != 0):
            rx_syms_pkt = rx_subc_pkt * np.tile(np.exp(-1j * subc_phases), packet_blocks_expected)
        else:
            rx_syms_pkt = rx_subc_pkt

        bits_pkt = qpsk_demap(rx_syms_pkt)
        cw_bits = RS_CW_BITS
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
                decoded_ok = True
            except Exception as e:
                msg = b'\x00' * RS_DATA_BYTES
                decoded_ok = False
                header_len = 64 if packet_idx == 0 else 4
                local_payload_offset = ci * RS_DATA_BYTES - header_len
                if local_payload_offset < 0:
                    global_offset = bytes_before_packet + 0
                else:
                    global_offset = bytes_before_packet + local_payload_offset
                do_print = False
                if packet_idx == 0:
                    if globals()['_rs_fail_prints_count'] < _MAX_RS_FAIL_PRINTS_GLOBAL:
                        do_print = True
                else:
                    if globals()['_rs_fail_prints_count'] < int(_MAX_RS_FAIL_PRINTS_GLOBAL * 0.25):
                        do_print = True
                if do_print:
                    globals()['_rs_fail_prints_count'] += 1
                    try:
                        symbols_per_cw = (cw_bits) // 2
                        s_start = ci * symbols_per_cw
                        subc_idx = int(s_start % Nsub)
                        freq_hz = subc_inds[subc_idx] * fs / float(Nfft)
                        freq_msg = f"{freq_hz:.1f} Hz (subcarrier idx {subc_inds[subc_idx]})"
                    except Exception:
                        freq_msg = "unknown"
                    if packet_idx == 0:
                        print(f"[RX-DBG-PKT] pkt{packet_idx} cw_idx={ci} RS_FAIL global_offset={global_offset} cw_in={bts.hex()} exc={e} freq={freq_msg}")
                    else:
                        if globals().get('post_sync', False):
                            print(f"[RX-DBG-PKT] pkt{packet_idx} cw_idx={ci} RS_FAIL global_offset={global_offset} freq={freq_msg}")
                        else:
                            print(f"[RX-DBG-PKT] pkt{packet_idx} cw_idx={ci} RS_FAIL global_offset={global_offset}")
            # store per-codeword result for packet_idx==0 to allow voting across 3 header copies
            decoded_blocks.append((msg, decoded_ok) if packet_idx == 0 else msg)
 

        # If this is the first packet, we want to analyze the first 3*64=192 bytes (covered by cw_count_needed codewords)
        if packet_idx == 0:
            # decoded_blocks is list of tuples (msg, ok)
            # build raw concatenation for fallback/alternate demap
            assembled_raw = b"".join([b for (b,ok) in decoded_blocks])
            # try alternate demap if header nibble check fails
            if len(assembled_raw) >= 1 and not ((assembled_raw[0] & 0xF0) != 0):
                alt_decoded, alt_rs_ok = _try_alternate_demaps_and_rs(rx_syms_pkt[:Nsub * 4], n_cw, packet_idx, bytes_before_packet)
                if alt_decoded is not None and alt_rs_ok > rs_ok:
                    header_len_bytes = min(len(alt_decoded), len(assembled_raw))
                    assembled_raw = alt_decoded + assembled_raw[header_len_bytes:]
                    # mark everything from alt_decoded as good
                    decoded_blocks = []
                    for i in range(0, len(assembled_raw), RS_DATA_BYTES):
                        chunk = assembled_raw[i:i+RS_DATA_BYTES]
                        if len(chunk) < RS_DATA_BYTES:
                            chunk = chunk.ljust(RS_DATA_BYTES, b'\x00')
                        decoded_blocks.append((chunk, True))
                    rs_ok = alt_rs_ok
                    if globals()['_rs_fail_prints_count'] < 5:
                        print("[RX-DBG-PKT] pkt0: header improved using alternate demap (alt_rs_ok increased)")

            # Now compute per-64-byte-copy score: for each header copy offset 0,64,128 count how many RS codewords covering that 64 bytes decoded_ok==True
            # number of cw covering 192 bytes:
            expected_hdr_len = 3 * 64
            cw_per_hdr = (64 + RS_DATA_BYTES - 1) // RS_DATA_BYTES
            # prepare flags list
            decoded_flags = [1 if ok else 0 for (b,ok) in decoded_blocks]
            scores = []
            for copy_idx, off in enumerate((0,64,128)):
                # cw indices that intersect this 64-byte window:
                start_cw = (off) // RS_DATA_BYTES
                end_cw = (off + 64 - 1) // RS_DATA_BYTES
                score = 0
                for ci in range(start_cw, end_cw+1):
                    if ci < len(decoded_flags) and decoded_flags[ci]:
                        score += 1
                scores.append(score)
            # choose best copy (max score). tie-breaker: smallest offset (leftmost)
            best_copy = int(np.argmax(scores))
            header_offset_in_pkt0 = best_copy * 64
            # reconstruct merged 192-bytes header area by taking bytes from codewords decoded correctly where available
            header_by_cw = bytearray(expected_hdr_len)
            for cw_idx in range(min(len(decoded_blocks), (expected_hdr_len + RS_DATA_BYTES -1)//RS_DATA_BYTES)):
                msg_bytes, ok_flag = decoded_blocks[cw_idx]
                if ok_flag:
                    start_b = cw_idx * RS_DATA_BYTES
                    header_by_cw[start_b:start_b+RS_DATA_BYTES] = msg_bytes[:RS_DATA_BYTES]
            # assembled_return: merged header-by-cw (192 bytes) + remaining assembled_raw tail
            candidate_return_bytes = bytes(header_by_cw) + assembled_raw[expected_hdr_len:]
            # make assembled_try to be used for header parsing attempts
            assembled_try = candidate_return_bytes[:expected_hdr_len]
            # debug
            print(f"[RX-DBG-PKT] pkt0 header copy scores={scores} selected_offset={header_offset_in_pkt0}")
            # also expose header_offset and candidate bytes for caller via globals
            globals()['pkt0_header_offset'] = int(header_offset_in_pkt0)
            globals()['pkt0_header_bytes'] = bytes(header_by_cw[:64])  # primary canonical 1st copy bytes (merged)

        if rs_ok > best_result[1]:
            # for pkt0 return candidate_return_bytes if built, otherwise fallback to raw concatenation of decoded_blocks bytes
            if packet_idx == 0 and 'candidate_return_bytes' in locals():
                ret_bytes = candidate_return_bytes
            else:
                # decoded_blocks may be list of tuples or bytes; normalize
                if packet_idx == 0:
                    ret_bytes = b"".join([b for (b,ok) in decoded_blocks])
                else:
                    ret_bytes = b"".join(decoded_blocks)
            best_result = (ret_bytes, rs_ok, cand)
            try:
                if 'rx_syms_list' not in globals():
                    globals()['rx_syms_list'] = []
                if 'Hk_smooth_list' not in globals():
                    globals()['Hk_smooth_list'] = []
                rx_slice = rx_syms_pkt[:4096].copy() if isinstance(rx_syms_pkt, np.ndarray) else np.array([], dtype=complex)
                globals()['rx_syms_list'].append(rx_slice)
                globals()['Hk_smooth_list'].append(Hk_s.copy() if isinstance(Hk_s, np.ndarray) else np.array([], dtype=complex))
                globals()['last_packet_used_pre'] = cand
                globals()['last_packet_end_sample'] = int(cand + len(preamble_td) + packet_blocks_expected * SYMBOL_LEN)
                globals()['last_packet_rs_ok'] = rs_ok
            except Exception:
                pass
            if rs_ok >= n_cw * 0.9:
                break
    return best_result

# -----------------------
# Live receive and main flow
# -----------------------
def live_receive_and_process():
    CHUNK = 1024
    RECORD_CHANNELS = 1
    BUFFER_LOCK = threading.Lock()
    ring = deque()
    total_samples_in_buffer = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal total_samples_in_buffer
        if status:
            pass
        s = indata[:, 0].astype(np.float64)
        with BUFFER_LOCK:
            ring.append(s)
            total_samples_in_buffer += s.size

    stream = sd.InputStream(samplerate=fs, channels=RECORD_CHANNELS, blocksize=CHUNK, callback=audio_callback)
    stream.start()
    print("[LIVE] microphone stream started, buffering...")

    def read_buffer_snapshot():
        with BUFFER_LOCK:
            if not ring:
                return np.array([], dtype=float)
            arr = np.concatenate(list(ring))
            return arr.copy()

    def wait_for_samples(n_needed, timeout=None):
        start = time.time()
        while True:
            with BUFFER_LOCK:
                cur = total_samples_in_buffer
            if cur >= n_needed:
                return True
            if timeout is not None and (time.time() - start) > timeout:
                return False
            time.sleep(0.01)

    preamble_td_local = preamble_td
    pre_len = len(preamble_td_local)

    try:
        print("[LIVE] searching first preamble (continuous scan)")
        found_first = False
        first_sync_abs = None

        while not found_first:
            buf = read_buffer_snapshot()
            if buf.size < pre_len:
                time.sleep(0.02)
                continue

            corr_full = correlate(buf, preamble_td_local, mode='valid')
            energy = np.convolve(buf * buf, np.ones(pre_len)[::-1], mode='valid')
            denom = np.sqrt(energy * np.sum(preamble_td_local * preamble_td_local))
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_corr = np.abs(corr_full) / denom
                norm_corr[~np.isfinite(norm_corr)] = 0.0
            peak_idx = int(np.argmax(norm_corr))
            peak_val = float(norm_corr[peak_idx])

            if peak_val < 0.35:
                time.sleep(0.02)
                continue

            cand_abs = peak_idx
            packet_blocks_guess = DEFAULT_PACKET_BLOCKS
            needed_total = cand_abs + pre_len + packet_blocks_guess * SYMBOL_LEN + SYMBOL_LEN

            if not wait_for_samples(needed_total, timeout=10.0):
                time.sleep(0.05)
                continue

            refine_center = max(0, cand_abs - SYMBOL_LEN)
            refine_window_len = pre_len + 2 * SYMBOL_LEN
            buf2 = read_buffer_snapshot()
            if refine_center + refine_window_len > buf2.size:
                time.sleep(0.02)
                continue
            local_segment = buf2[refine_center: refine_center + refine_window_len]
            corr_local = correlate(local_segment, preamble_td_local, mode='valid')
            energy_local = np.convolve(local_segment * local_segment, np.ones(pre_len)[::-1], mode='valid')
            denom_local = np.sqrt(energy_local * np.sum(preamble_td_local * preamble_td_local))
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_local = np.abs(corr_local) / denom_local
                norm_local[~np.isfinite(norm_local)] = 0.0
            loc_peak = int(np.argmax(norm_local))
            refined_abs = refine_center + loc_peak
            print(f"[LIVE] first preamble candidate found: peak={peak_val:.3f} cand={cand_abs} refined={refined_abs}")

            needed_total_refined = refined_abs + pre_len + packet_blocks_guess * SYMBOL_LEN + SYMBOL_LEN
            if not wait_for_samples(needed_total_refined, timeout=10.0):
                print("[LIVE] waiting for more samples for refined packet timed out, continue scanning")
                time.sleep(0.05)
                continue

            first_sync_abs = refined_abs
            found_first = True

        if first_sync_abs is None:
            raise RuntimeError("no preamble found in live stream")

        print(f"[LIVE] attempting decode of first packet at abs={first_sync_abs}")

        buf_all = read_buffer_snapshot()
        abs_corr = np.abs(correlate(buf_all, preamble_td_local, mode='valid'))

        globals_backup = {}
        for name in ('rx', 'abs_corr'):
            globals_backup[name] = globals().get(name, None)
            globals()[name] = buf_all

        try:
            pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = decode_packet_at_candidate(first_sync_abs, DEFAULT_PACKET_BLOCKS, packet_idx=0, bytes_before_packet=0, expected_total=0)
            if pkt0_decoded is None or len(pkt0_decoded) < 64:
                print("[LIVE] failed to decode first packet from live buffer")
                globals()['rx'] = globals_backup['rx']
                globals()['abs_corr'] = globals_backup['abs_corr']
                stream.stop()
                stream.close()
                return
            header64 = pkt0_decoded[:64]
            try:
                hdr = parse_header(header64)
            except Exception as e:
                print("[LIVE] parse_header failed:", e)
                stream.stop()
                stream.close()
                return
            print(f"[LIVE] first packet decoded: RS_OK={pkt0_rs_ok} header={header64[:16].hex()}")

            globals()['post_sync'] = True
            globals()['sync_sample_abs'] = int(pkt0_used_pre) if pkt0_used_pre is not None else int(first_sync_abs)

            mode_rx = hdr['mode']
            total_sz = hdr['data_len']
            fname = hdr['filename']
            packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)
            print(f"[LIVE] transmission header: data_len={total_sz} filename={fname} packet_blocks={packet_blocks_from_hdr}")

            positions_no_preroll = simulate_packet_positions(total_sz, fname.encode('utf-8') if fname else b'', (mode_rx == b'T'),
                                                             packet_blocks_from_hdr, preamble_len=len(preamble_td_local), symbol_len=SYMBOL_LEN,
                                                             gap_samples=GAP_SAMPLES_DEFAULT)
            if len(positions_no_preroll) == 0:
                print("[LIVE] simulate_packet_positions returned no positions")
                stream.stop()
                stream.close()
                return

            base_abs = first_sync_abs - positions_no_preroll[0]
            expected_abs = [base_abs + int(x) for x in positions_no_preroll]
            print("[LIVE] expected preamble absolute indices (in buffer snapshot coords):", expected_abs)

            assembled_chunks = []
            bytes_collected = 0
            expected_total = total_sz
            packet_blocks_val = packet_blocks_from_hdr

            for pkt_idx, pref in enumerate(expected_abs):
                pref_backoff = max(0, pref - SYMBOL_LEN)
                needed_for_pkt = pref_backoff + SYMBOL_LEN + pre_len + packet_blocks_val * SYMBOL_LEN + SYMBOL_LEN
                if not wait_for_samples(needed_for_pkt, timeout=20.0):
                    print(f"[LIVE] timeout waiting for packet {pkt_idx} samples (needed {needed_for_pkt}), aborting")
                    break

                buf_now = read_buffer_snapshot()
                abs_corr = np.abs(correlate(buf_now, preamble_td_local, mode='valid'))

                refine_center = pref_backoff
                refine_window_len = pre_len + 2 * SYMBOL_LEN
                if refine_center + refine_window_len > buf_now.size:
                    if not wait_for_samples(refine_center + refine_window_len, timeout=5.0):
                        print("[LIVE] not enough samples for refine, abort packet")
                        break
                    buf_now = read_buffer_snapshot()
                    abs_corr = np.abs(correlate(buf_now, preamble_td_local, mode='valid'))

                local_segment = buf_now[refine_center: refine_center + refine_window_len]
                corr_local = correlate(local_segment, preamble_td_local, mode='valid')
                energy_local = np.convolve(local_segment * local_segment, np.ones(pre_len)[::-1], mode='valid')
                denom_local = np.sqrt(energy_local * np.sum(preamble_td_local * preamble_td_local))
                with np.errstate(divide='ignore', invalid='ignore'):
                    norm_local = np.abs(corr_local) / denom_local
                    norm_local[~np.isfinite(norm_local)] = 0.0
                loc_peak = int(np.argmax(norm_local))
                refined_abs = refine_center + loc_peak
                print(f"[LIVE] pkt{pkt_idx} predicted={pref} backoff={pref_backoff} refined={refined_abs} peak={float(norm_local[loc_peak]):.3f}")

                globals()['rx'] = buf_now
                globals()['abs_corr'] = abs_corr
                decoded_bytes, rs_ok_count, used_preamble = decode_packet_at_candidate(refined_abs, packet_blocks_val, packet_idx=pkt_idx, bytes_before_packet=bytes_collected, expected_total=expected_total)
                print(f"[LIVE] pkt{pkt_idx}: refined={refined_abs} used_pre={used_preamble} RS_OK={rs_ok_count} bytes={len(decoded_bytes)}")

                if pkt_idx == 0:
                    if len(decoded_bytes) < 64:
                        print("[LIVE] first packet decoded <64 bytes, abort")
                        break
                    # payload always starts after the three headers (3 * 64 = 192 bytes).
                    # chosen header offset is used only to parse fields, not to determine payload start.
                    payload_start = 3 * 64
                    if payload_start >= len(decoded_bytes):
                        payload_chunk = b''
                    else:
                        payload_chunk = decoded_bytes[payload_start:]
                    need = max(0, expected_total - bytes_collected)
                    if need <= 0:
                        payload_chunk_take = b''
                    else:
                        payload_chunk_take = payload_chunk[:need]
                    assembled_chunks.append(payload_chunk_take)
                    bytes_collected += len(payload_chunk_take)
                else:
                    if len(decoded_bytes) >= 4:
                        payload_chunk = decoded_bytes[4:]
                        assembled_chunks.append(payload_chunk)
                        bytes_collected += len(payload_chunk)
                    else:
                        assembled_chunks.append(b'')
                print(f"[LIVE] collected {bytes_collected}/{expected_total} bytes")
                if bytes_collected >= expected_total:
                    print("[LIVE] collected all expected bytes")
                    break

            assembled = b"".join(assembled_chunks)[:expected_total]

            try:
                hdr_crc = hdr.get('crc32', None) if isinstance(hdr, dict) else None
                if hdr_crc is not None:
                    calc_crc = zlib.crc32(assembled) & 0xFFFFFFFF
                    if calc_crc == int(hdr_crc):
                        print(f"[RX-CRC] OK: CRC32 matched (0x{calc_crc:08X})")
                    else:
                        print(f"[RX-CRC] MISMATCH: received 0x{int(hdr_crc):08X}, calculated 0x{calc_crc:08X}")
                else:
                    print("[RX-CRC] no CRC32 present in header, skipping check")
            except Exception as e:
                print("[RX-CRC] crc check failed:", e)

            if mode_rx == b"F":
                out_fname = fname if fname else "rx_file"
                out_path = "rx_" + out_fname
                with open(out_path, "wb") as f:
                    f.write(assembled)
                print(f"[RX FILE] Saved: {out_path} ({len(assembled)} bytes)")
            else:
                rec_text = assembled.decode("utf-8", errors="ignore")
                print("[RX TEXT]", rec_text)

            try:
                sync_start = globals().get('sync_sample_abs', None)
                last_end = globals().get('last_packet_end_sample', None)
                if sync_start is None:
                    print("[PLOT] sync_sample_abs unknown, skipping focused plots")
                    plot_slice = None
                else:
                    if last_end is None:
                        last_end = len(buf_all)
                    buf_now_for_plot = read_buffer_snapshot()
                    s0 = max(0, int(sync_start))
                    s1 = min(len(buf_now_for_plot), int(last_end))
                    if s1 <= s0:
                        print("[PLOT] invalid plot slice (empty), skipping")
                        plot_slice = None
                    else:
                        plot_slice = buf_now_for_plot[s0:s1]
                        print(f"[PLOT] plotting samples from {s0} to {s1} (len={len(plot_slice)})")
                figs_shown = 0
                if plot_slice is not None:
                    if 'rx_syms_list' in globals() and isinstance(globals()['rx_syms_list'], list) and len(globals()['rx_syms_list']) > 0:
                        M = 2048
                        concat_list = [r[:M] for r in globals()['rx_syms_list'] if isinstance(r, np.ndarray) and r.size > 0]
                        if len(concat_list) > 0:
                            concat = np.concatenate(concat_list)
                            plt.figure(figsize=(5,5))
                            plt.plot(np.real(concat), np.imag(concat), 'o', markersize=2, alpha=0.4)
                            plt.axhline(0, color='grey', linewidth=0.5); plt.axvline(0, color='grey', linewidth=0.5)
                            plt.title(f"RX Constellation (combined from {len(concat_list)} packets, post-sync)")
                            plt.xlabel("In-phase"); plt.ylabel("Quadrature"); plt.grid(True); plt.axis('equal')
                            figs_shown += 1
                    else:
                        plt.figure(figsize=(8,3))
                        tvec = np.arange(len(plot_slice)) / float(fs)
                        plt.plot(tvec, plot_slice)
                        plt.title("Time-domain slice (post-sync)")
                        plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True)
                        figs_shown += 1
                else:
                    print("[PLOT] no focused slice available, skipping plots")

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
                        plt.plot(freqs, eq_gain, "o-", color="darkorange", label="Median Gain (post-sync)")
                        plt.title(f"Equalizer Gain vs Frequency (median of {Hstack.shape[0]} packets)")
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

        finally:
            globals()['rx'] = globals_backup['rx']
            globals()['abs_corr'] = globals_backup['abs_corr']

    except Exception as e:
        print("[LIVE] exception during live receive:", e)
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        print("[LIVE] microphone stream stopped")

# -----------------------
# Main TX/RX workflow
# -----------------------
if __name__ == "__main__":
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
    NOISE_LEVEL = 0.0001
    NOISE_SEED = 20231107

    rng_global = np.random.RandomState(NOISE_SEED)

    tx_type_bits = 0b11 if mode == "F" else (0b10 if mode == "T" else 0b00)
    version = 0

    # compute CRC32 over payload bytes (file_data / text bytes)
    crc_val = zlib.crc32(file_data) & 0xFFFFFFFF
    transmission_header = build_header(b'F' if mode == "F" else b'T', total_data_len, filename_bytes=filename_bytes, packet_no=0, version=version, packet_blocks=packet_blocks, crc32=crc_val)
    remaining_data = file_data

    samples_packets = []
    packet_no = 0
    running_sample_offset = 0
    packet_preamble_offsets_no_preroll = []

    while True:
      is_first = (packet_no == 0)
      if is_first:
        # first packet: three consecutive 64-byte transmission headers
        header_bytes = transmission_header + transmission_header + transmission_header
      else:
        header_bytes = make_packet_header_bytes(packet_no, tx_type_bits)
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

      td_now = best_td
      if SYMBOL_LEN <= 0:
        nblocks_now = 0
      else:
        nblocks_now = len(td_now) // SYMBOL_LEN
      if nblocks_now < allowed_blocks:
        needed_blocks = allowed_blocks - nblocks_now
        filler_bytes_len = needed_blocks * RS_DATA_BYTES
        td_filler, nblk_f = bytes_to_ofdm_blocks_bytes(b'\x00' * filler_bytes_len)
        if nblk_f >= needed_blocks and len(td_filler) >= needed_blocks * SYMBOL_LEN:
          td_now = np.concatenate((td_now, td_filler[:needed_blocks * SYMBOL_LEN]))
          nblocks_now = allowed_blocks
        else:
          pad_samples = needed_blocks * SYMBOL_LEN
          td_now = np.concatenate((td_now, np.zeros(pad_samples, dtype=td_now.dtype)))
          nblocks_now = allowed_blocks
      elif nblocks_now > allowed_blocks:
        td_now = td_now[:allowed_blocks * SYMBOL_LEN]
        nblocks_now = allowed_blocks

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
      live_receive_and_process()
      sys.exit(0)

    corr = fftconvolve(rx, preamble_td[::-1], mode='valid')
    abs_corr = np.abs(corr)
    if abs_corr.size == 0:
        print("[RX-ERR] correlation empty, cannot find preambles")
        sync_idx = sync_by_corr(rx, preamble_td)
    else:
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
                pilot1_start = cand_sync + 2*SYMBOL_LEN
                rx_pre1_try = rx[pilot1_start + Ncp : pilot1_start + Ncp + Nfft]
                pilot2_start = pilot1_start + SYMBOL_LEN
                rx_pre2_try = rx[pilot2_start + Ncp : pilot2_start + Ncp + Nfft]
                S_ref = np.fft.fft(preamble_td[2*SYMBOL_LEN + Ncp : 2*SYMBOL_LEN + Ncp + Nfft])
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
            cw_bits = RS_CW_BITS
            if len(bits_try) < cw_bits:
                continue
            bts0 = bits_to_bytes(bits_try[:cw_bits])
            try:
                msg0 = rs.decode(bts0)[0]
                if len(msg0) > 0 and (msg0[0] & 0xF0):
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

    pkt0_decoded, pkt0_rs_ok, pkt0_used_pre = decode_packet_at_candidate(sync_idx, DEFAULT_PACKET_BLOCKS, packet_idx=0, bytes_before_packet=0, expected_total=0)
    if pkt0_decoded is None or len(pkt0_decoded) < 64:
      print("[RX-ERR] unable to decode first packet header using per-packet decoder")
      sys.exit(1)
    # if decode_packet_at_candidate returned merged triple-header, select the best copy using pkt0_header_offset (set inside decode)
    hdr_offset = globals().get('pkt0_header_offset', None)
    if hdr_offset is None:
      # fallback: try offsets 0,64,128 in order and pick first valid parse
      found = False
      for off in (0,64,128):
        if len(pkt0_decoded) >= off + 64:
          try:
            ph = parse_header(pkt0_decoded[off:off+64])
            hdr = ph
            hdr_offset = off
            found = True
            break
          except Exception:
            continue
      if not found:
        # final fallback: try start
        try:
          hdr = parse_header(pkt0_decoded[:64])
          hdr_offset = 0
        except Exception as e:
          print("[RX-ERR] parse_header failed for first packet:", e)
          print("[RX-ERR] header64_hex:", pkt0_decoded[:64].hex())
          sys.exit(1)
    else:
      # parse header at hdr_offset
      try:
        hdr = parse_header(pkt0_decoded[hdr_offset:hdr_offset+64])
      except Exception as e:
        # if parsing fails, attempt fallbacks
        parsed_ok = False
        for off in (0,64,128):
          try:
            ph = parse_header(pkt0_decoded[off:off+64])
            hdr = ph
            hdr_offset = off
            parsed_ok = True
            break
          except Exception:
            continue
        if not parsed_ok:
          print("[RX-ERR] parse_header failed for first packet (all offsets)", e)
          print("[RX-ERR] pkt0_decoded_hex:", pkt0_decoded[:192].hex())
          sys.exit(1)
    print(f"[RX-DBG] first-packet local decode: used_pre={pkt0_used_pre} RS_OK={pkt0_rs_ok} header_offset={hdr_offset} header_hex={pkt0_decoded[hdr_offset:hdr_offset+64].hex()[:256]}")
    # expose chosen offset to downstream code (file-mode loop)
    globals()['pkt0_header_offset'] = int(hdr_offset)
    globals()['post_sync'] = True
    globals()['sync_sample_abs'] = int(pkt0_used_pre) if pkt0_used_pre is not None else int(sync_idx)

    mode_rx = hdr['mode']
    total_sz = hdr['data_len']
    fname = hdr['filename']
    packet_blocks_from_hdr = hdr.get('packet_blocks', DEFAULT_PACKET_BLOCKS)
    print(f"[RX-INFO] transmission header: data_len={total_sz} filename={fname} packet_blocks={packet_blocks_from_hdr}")

    positions_no_preroll = simulate_packet_positions(total_sz, fname.encode('utf-8') if fname else b'', (mode_rx == b'T'),
                                                     packet_blocks_from_hdr, preamble_len=len(preamble_td), symbol_len=SYMBOL_LEN,
                                                     gap_samples=GAP_SAMPLES_DEFAULT)
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
        decoded_bytes, rs_ok_count, used_preamble = decode_packet_at_candidate(pref, packet_blocks_val, packet_idx=pkt_idx, bytes_before_packet=bytes_collected, expected_total=expected_total)
        print(f"[RX-DBG] pkt{pkt_idx}: pref={pref} used_pre={used_preamble} RS_OK={rs_ok_count} bytes={len(decoded_bytes)}")
        if pkt_idx == 0:
            if len(decoded_bytes) < 64:
                print("[RX-ERR] first packet decoded <64 bytes, aborting")
                sys.exit(1)
            # header fields already parsed earlier using chosen header offset;
            # payload always starts after all three 64-byte header copies
            payload_start = 3 * 64
            if payload_start >= len(decoded_bytes):
                payload_chunk = b''
            else:
                payload_chunk = decoded_bytes[payload_start:]
            need = max(0, expected_total - bytes_collected)
            if need <= 0:
                payload_chunk = b''
            else:
                payload_chunk = payload_chunk[:need]
            assembled_chunks.append(payload_chunk)
            bytes_collected += len(payload_chunk)
        else:
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

    try:
        hdr_crc = hdr.get('crc32', None) if isinstance(hdr, dict) else None
        if hdr_crc is not None:
            calc_crc = zlib.crc32(assembled) & 0xFFFFFFFF
            if calc_crc == int(hdr_crc):
                print(f"[RX-CRC] OK: CRC32 matched (0x{calc_crc:08X})")
            else:
                print(f"[RX-CRC] MISMATCH: received 0x{int(hdr_crc):08X}, calculated 0x{calc_crc:08X}")
        else:
            print("[RX-CRC] no CRC32 present in header, skipping check")
    except Exception as e:
        print("[RX-CRC] crc check failed:", e)

    if mode_rx == b"F":
      out_fname = fname
      out_path = "rx_" + out_fname
      with open(out_path, "wb") as f:
        f.write(assembled)
      print(f"[RX FILE] Сохранён файл: {out_path} ({len(assembled)} байт)")
    else:
      rec_text = assembled.decode("utf-8", errors="ignore")
      print("[RX TEXT]", rec_text)

    try:
        sync_start = globals().get('sync_sample_abs', None)
        last_end = globals().get('last_packet_end_sample', None)
        if sync_start is None:
            print("[PLOT] sync_sample_abs unknown, skipping focused plots")
            plot_slice = None
        else:
            if last_end is None:
                last_end = len(rx)
            s0 = max(0, int(sync_start))
            s1 = min(len(rx), int(last_end))
            if s1 <= s0:
                print("[PLOT] invalid plot slice (empty), skipping")
                plot_slice = None
            else:
                plot_slice = rx[s0:s1]
                                
                print(f"[PLOT] plotting samples from {s0} to {s1} (len={len(plot_slice)})")

        figs_shown = 0
        if plot_slice is not None:
            if 'rx_syms_list' in globals() and isinstance(globals()['rx_syms_list'], list) and len(globals()['rx_syms_list']) > 0:
                M = 2048
                concat_list = [r[:M] for r in globals()['rx_syms_list'] if isinstance(r, np.ndarray) and r.size > 0]
                if len(concat_list) > 0:
                    concat = np.concatenate(concat_list)
                    plt.figure(figsize=(5,5))
                    plt.plot(np.real(concat), np.imag(concat), 'o', markersize=2, alpha=0.4)
                    plt.axhline(0, color='grey', linewidth=0.5); plt.axvline(0, color='grey', linewidth=0.5)
                    plt.title(f"RX Constellation (combined post-sync)")
                    plt.xlabel("In-phase"); plt.ylabel("Quadrature"); plt.grid(True); plt.axis('equal')
                    figs_shown += 1
            else:
                plt.figure(figsize=(8,3))
                tvec = np.arange(len(plot_slice)) / float(fs)
                plt.plot(tvec, plot_slice)
                plt.title("Time-domain slice (post-sync)")
                plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True)
                figs_shown += 1
        else:
            print("[PLOT] no focused slice available, skipping plots")

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
                plt.plot(freqs, eq_gain, "o-", color="darkorange", label="Median Gain (post-sync)")
                plt.title(f"Equalizer Gain vs Frequency (median of {Hstack.shape[0]} packets)")
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

# end of file


