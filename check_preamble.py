#!/usr/bin/env python3
# check_preamble_with_modem.py
# Usage: python check_preamble_with_modem.py /path/to/test_modem_simple.py
import sys, os, json, numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, find_peaks

if len(sys.argv) < 2:
    print("Usage: python check_preamble_with_modem.py /path/to/your_modem_script.py")
    sys.exit(1)

modem_path = sys.argv[1]
if not os.path.isfile(modem_path):
    print("Modem script not found:", modem_path)
    sys.exit(1)

# load modem module dynamically to get build_preamble, Nfft, Ncp, fs
import importlib.util
spec = importlib.util.spec_from_file_location("user_modem", modem_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# check required symbols
for name in ("build_preamble","Nfft","Ncp","fs"):
    if not hasattr(mod, name):
        print(f"Modem script missing required symbol: {name}")
        sys.exit(1)

preamble_td = mod.build_preamble(reps=2)
Nfft = int(mod.Nfft)
Ncp = int(mod.Ncp)
fs = int(mod.fs)
symbol_len = Nfft + Ncp

wav_name = "ofdm_acoustic_tx_with_noise.wav"
if not os.path.isfile(wav_name):
    print("WAV not found in CWD:", wav_name)
    sys.exit(1)

sr, wav = wavfile.read(wav_name)
if sr != fs:
    print(f"[WARN] WAV sample rate {sr} != expected {fs}")

sig = wav.astype(float)
if sig.ndim > 1:
    sig = sig[:,0]
# normalize to [-1,1]
if np.issubdtype(wav.dtype, np.integer):
    sig = sig / float(np.iinfo(wav.dtype).max)
else:
    maxabs = np.max(np.abs(sig)) if np.max(np.abs(sig))>0 else 1.0
    sig = sig / maxabs

print(f"Loaded WAV {wav_name}: samples={len(sig)}, sr={sr}")
print(f"Using preamble length {len(preamble_td)} samples (reps=2). Symbol length = {symbol_len}")

# compute correlation
print("Computing correlation (this may take a moment)...")
corr = fftconvolve(sig, preamble_td[::-1], mode='valid')
abs_corr = np.abs(corr)

# find peaks — tune distance to avoid duplicate adjacent peaks
min_distance = max(1, int(symbol_len*0.6))
peaks, props = find_peaks(abs_corr, distance=min_distance, height=np.median(abs_corr) + 3.0*(np.median(np.abs(abs_corr-np.median(abs_corr)))+1e-12))
heights = props.get("peak_heights", np.array([]))

# if no peaks found with that threshold, relax threshold and search again
if peaks.size == 0:
    thr = np.median(abs_corr) + 1.5*(np.median(np.abs(abs_corr-np.median(abs_corr)))+1e-12)
    peaks, props = find_peaks(abs_corr, distance=min_distance, height=thr)
    heights = props.get("peak_heights", np.array([]))

print(f"Found {len(peaks)} peaks (distance={min_distance})")

# sort peaks by height descending for reporting, but also provide chronological listing
order_by_height = np.argsort(heights)[::-1] if heights.size>0 else np.array([], dtype=int)
chron = np.argsort(peaks)

top_show = min(20, len(peaks))
print("\nTop peaks by height (rank, sample_index, seconds, height):")
for i in range(top_show):
    idx = peaks[order_by_height[i]]
    print(f"{i+1:2d}. {idx:8d}  {idx/float(fs):8.3f}s   height={heights[order_by_height[i]]:.6g}")

print("\nFirst 20 peaks in time order (index, seconds, height):")
for i in chron[:20]:
    idx = peaks[i]
    h = heights[i] if heights.size>0 else float(abs_corr[idx])
    print(f"{idx:8d}  {idx/float(fs):8.3f}s  height={h:.6g}")

# compute deltas
if len(peaks) >= 2:
    sorted_peaks = np.sort(peaks)
    deltas = np.diff(sorted_peaks)
    print("\nDeltas between consecutive peaks (first 20):")
    for d in deltas[:20]:
        print(f"{d:6d} samples, {d/float(symbol_len):6.3f} OFDM symbols")
else:
    deltas = []

# Save results
out = {
    "wav": wav_name,
    "fs": fs,
    "Nfft": Nfft,
    "Ncp": Ncp,
    "symbol_len": symbol_len,
    "preamble_len": len(preamble_td),
    "peaks": [{"sample": int(int(peaks[i])), "height": float(heights[i]) if heights.size>0 else float(abs_corr[peaks[i]])} for i in range(len(peaks))],
    "deltas_first": [int(x) for x in deltas[:50]]
}
with open("found_preamble_peaks.json","w") as f:
    json.dump(out, f, indent=2)
print("\nSaved found_preamble_peaks.json")

# If tx_header_map.json exists, compare closest expected headers
if os.path.isfile("tx_header_map.json"):
    with open("tx_header_map.json","r") as f:
        txmap = json.load(f)
    preroll_len = int(0.25 * fs)  # default used in your code
    print("\nComparing to tx_header_map.json entries (if present):")
    for entry in txmap:
        expected = preroll_len + int(entry["start_sample"])
        # find nearest peak
        if len(peaks)==0:
            print(f"packet {entry['packet_no']}: expected_abs={expected}  -> no peaks found")
            continue
        diffs = np.abs(np.array(peaks) - expected)
        idx = np.argmin(diffs)
        print(f"packet {entry['packet_no']}: expected_abs={expected}, nearest_peak={int(peaks[idx])}, delta={int(diffs[idx])} samples")
else:
    print("\nNo tx_header_map.json found; generate TX with header map to compare.")

print("\nDone.")

