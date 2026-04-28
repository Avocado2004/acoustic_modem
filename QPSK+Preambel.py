import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import math

# ------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------
fs        = 48000           # sampling rate, Hz
Nfft      = 512             # FFT size
Ncp       = 64              # cyclic prefix length
f_low     = 300             # bandpass low cutoff, Hz
f_high    = 4300            # bandpass high cutoff, Hz
df        = fs / Nfft
k_low     = int(math.ceil(f_low / df))
k_high    = int(math.floor(f_high / df))
subc_inds = np.arange(k_low, k_high + 1)
Nsub      = len(subc_inds)

# ------------------------------------------------------------------------
# Text ↔ Bits
# ------------------------------------------------------------------------
def text_to_bits(text, encoding='utf-8'):
    bs = ''.join(f'{b:08b}' for b in text.encode(encoding))
    return np.array(list(bs), dtype=int)

def bits_to_text(bits, encoding='utf-8'):
    L = (len(bits)//8)*8
    b = bits[:L].reshape(-1, 8)
    data = bytes(int(''.join(str(x) for x in row), 2) for row in b)
    return data.decode(encoding, errors='ignore')

# ------------------------------------------------------------------------
# QPSK Mapper / Demapper
# ------------------------------------------------------------------------
def qpsk_map(bits):
    M = {
        '00': (1+1j)/np.sqrt(2),
        '01': (-1+1j)/np.sqrt(2),
        '11': (-1-1j)/np.sqrt(2),
        '10': (1-1j)/np.sqrt(2),
    }
    if len(bits)%2:
        bits = np.append(bits, 0)
    syms = [ M[f"{bits[i]}{bits[i+1]}"]
             for i in range(0, len(bits), 2) ]
    return np.array(syms)

def qpsk_demap(syms):
    bits = []
    for s in syms:
        b0 = 0 if s.real > 0 else 1
        b1 = 0 if s.imag > 0 else 1
        bits += [b1, b0]
    return np.array(bits, dtype=int)

# ------------------------------------------------------------------------
# OFDM symbol + preamble
# ------------------------------------------------------------------------
def ofdm_symbol(data_syms):
    X = np.zeros(Nfft, dtype=complex)
    X[subc_inds]  = data_syms
    X[-subc_inds] = np.conj(data_syms)
    x = np.fft.ifft(X)
    return np.concatenate((x[-Ncp:], x))

def build_preamble(reps=2):
    pilot = np.full(Nsub, (1+1j)/np.sqrt(2))
    S = ofdm_symbol(pilot)
    return np.tile(S, reps)

def build_data_td(bits):
    syms = qpsk_map(bits)
    pad  = (-len(syms)) % Nsub
    if pad:
        syms = np.concatenate((syms, np.zeros(pad, dtype=complex)))
    blk = syms.reshape(-1, Nsub)
    td  = [ofdm_symbol(b) for b in blk]
    return np.concatenate(td), blk.shape[0]

# ------------------------------------------------------------------------
# Direct corr‐based synchronization
# ------------------------------------------------------------------------
def sync_by_corr(rx, pre):
    # full cross-correlation
    corr = signal.fftconvolve(rx, pre[::-1], mode='valid')
    idx  = np.argmax(np.abs(corr))
    return idx

# ------------------------------------------------------------------------
# Plot constellation
# ------------------------------------------------------------------------
def plot_constellation(syms, title):
    plt.figure(figsize=(5,5))
    plt.scatter(syms.real, syms.imag, s=20, alpha=0.6, color='navy')
    plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
    plt.title(title); plt.xlabel('I'); plt.ylabel('Q')
    plt.grid(True); plt.axis('equal')
    plt.show()

# ------------------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Input
    text      = input("Введите текст для передачи: ")
    data_bits = text_to_bits(text)
    orig_len  = len(data_bits)

    # 2) Build preamble + data
    preamble_td = build_preamble(reps=2)
    data_td, nblk = build_data_td(data_bits)
    tx = np.concatenate((preamble_td, data_td))

    # 3) FIR bandpass – zero phase
    numtaps = 129
    taps    = signal.firwin(numtaps, [f_low, f_high], fs=fs, pass_zero=False)
    tx_f = signal.filtfilt(taps, [1.0], tx)

    # 4) Normalize ±0.9
    tx_n = tx_f * (0.9 / np.max(np.abs(tx_f)))

    # 5) Save WAV
    
    # В WAV сохраняем только действительную часть, отбрасываем возможные малые мнимые шумы
    real_tx = np.real(tx_n)
    wavfile.write("ofdm_acoustic_tx.wav",
                  fs,
                  (real_tx * np.iinfo(np.int16).max).astype(np.int16))
    """
    wavfile.write("ofdm_acoustic_tx.wav",
                  fs,
                  (tx_n * np.iinfo(np.int16).max).astype(np.int16))
                  """
    print("TX saved to ofdm_acoustic_tx.wav")

    # 6) Receive
    src = input("Откуда демодулировать? (file/mic): ").strip().lower()
    if src == 'file':
        _, wavd = wavfile.read("ofdm_acoustic_tx.wav")
        sig    = wavd[:,0] if wavd.ndim>1 else wavd
        rx     = sig.astype(float) / np.iinfo(wavd.dtype).max
    else:
        print("Запись с микрофона 10 секунд...")
        rec = sd.rec(int(10*fs), samplerate=fs, channels=1)
        sd.wait()
        rx = rec[:,0].astype(float)
        rx /= np.max(np.abs(rx))

    # 7) Optional RX filtering
    rx = signal.filtfilt(taps, [1.0], rx)

    # 8) Find preamble
    sync_idx = sync_by_corr(rx, preamble_td)
    print(f"Sync index: {sync_idx}")

    # 9) Extract data segment
    start = sync_idx + len(preamble_td)
    needed = nblk*(Nfft+Ncp)
    segment = rx[start:start+needed]
    if len(segment) < needed:
        segment = np.concatenate((segment, np.zeros(needed-len(segment))))

    # 10) Chop into OFDM frames, FFT + equalize
    #    channel estimation on first preamble symbol
    pre1 = tx_n[:Nfft+Ncp][Ncp:]           # ideal first symbol from TX
    # but we want rx version:
    rx_pre1 = rx[sync_idx+Ncp:sync_idx+Ncp+Nfft]
    H_est   = np.fft.fft(rx_pre1) / np.fft.fft(pre1)

    frames = segment.reshape(nblk, Nfft+Ncp)
    rx_syms = []
    for frm in frames:
        Xf = np.fft.fft(frm[Ncp:])
        eq = Xf[subc_inds] / H_est[subc_inds]
        rx_syms.append(eq)
    rx_syms = np.concatenate(rx_syms)

    # 11) Demap → bits → text
    rx_bits  = qpsk_demap(rx_syms)[:orig_len]
    rec_text = bits_to_text(rx_bits)
    print("Восстановленный текст:", rec_text)

    # 12) Plot
    plot_constellation(rx_syms, "RX Constellation (equalized)")

