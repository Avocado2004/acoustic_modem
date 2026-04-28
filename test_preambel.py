import time
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, lfilter, correlate

# === Параметры ===
fs            = 44100
order_golay   = 10        # длина A, B = 2**order
f_low, f_high = 300, 5000
threshold     = 2000      # порог по сумме квадратов корреляций
agc_target    = 0.9
block_dur     = 0.05      # 50 ms
max_listen    = 10

# === Генерация Golay ===
def generate_golay(order):
    A = np.array([1], dtype=int)
    B = np.array([1], dtype=int)
    for _ in range(order):
        A, B = np.concatenate((A,  B)), np.concatenate((A, -B))
    return A.astype(float), B.astype(float)

# === БПФ (в одну сторону) ===
def design_bpf(fs, f_low, f_high, order=6):
    nyq = fs/2
    return butter(order, [f_low/nyq, f_high/nyq], btype='band')

b_bpf, a_bpf = design_bpf(fs, f_low, f_high)

# === AGC ===
def apply_agc(sig, target=0.9, eps=1e-12):
    peak = np.max(np.abs(sig))
    return sig if peak<eps else sig * (target/peak)

# === Детектор Golay ===
def detect_golay(win, A, B):
    # к данным предварительно применён BPF и AGC
    cA = correlate(win, A, mode='valid')
    cB = correlate(win, B, mode='valid')
    metric = cA**2 + cB**2
    idx = np.argmax(metric)
    return idx, metric[idx]

# === Подготовка преамбулы для проигрывания ===
A, B = generate_golay(order_golay)
# Отфильтруем и нормируем A и B
A_f = lfilter(b_bpf, a_bpf, A)
B_f = lfilter(b_bpf, a_bpf, B)
A_f /= np.max(np.abs(A_f))
B_f /= np.max(np.abs(B_f))
# Для воспроизведения склеиваем A_f + B_f
preamble = np.concatenate((A_f, B_f))
sf.write("golay_preamble.wav", preamble, fs)
print("✅ Golay-pre saved as golay_preamble.wav")
print(f"Threshold = {threshold}, block = {block_dur*1e3:.0f} ms")

# длина окна равна длине A Ф и B Ф, но для detection достаточно A+B
len_win = len(preamble)
buffer = np.zeros(0, dtype=float)
total = 0
t0 = time.time()

with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
    while True:
        block, _ = stream.read(int(block_dur*fs))
        block = block[:,0].astype(float)
        total += len(block)

        # обновляем кольцевой буфер
        buffer = np.concatenate((buffer, block))
        if len(buffer) > len_win:
            buffer = buffer[-len_win:]

        # BPF→AGC
        buf_f = lfilter(b_bpf, a_bpf, buffer)
        buf_f = apply_agc(buf_f, target=agc_target)

        # детект
        idx, m = detect_golay(buf_f, A_f, B_f)
        print(f"\rPeak={m:.0f}", end='')  # капелька отладки

        if m > threshold:
            samp  = total - len(buffer) + idx
            t_sec = samp/fs
            print(f"\n✅ Detected at {t_sec:.3f}s (sample {samp}), metric={m:.0f}")
            break

        if time.time()-t0 > max_listen:
            print("\n⚠️ Timeout, not found")
            break

