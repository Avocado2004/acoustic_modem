#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-carrier DQPSK-модем с полосой 300–5000 Гц, 6-порядковым BPF,
дифференциальной фазовой манипуляцией (DQPSK), коррекцией ошибок
Reed-Solomon, автоматическим подбором M и визуализацией спектра.
"""

import numpy as np
import wave, struct, math
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ---- Подключаем Reed-Solomon ----
try:
    from reedsolo import RSCodec, ReedSolomonError
except ImportError:
    print("Требуется библиотека reedsolo. Установите: pip install reedsolo")
    exit(1)

# ========== Параметры полосы и фильтра ==========
f_low   = 300
f_high  = 5000
band    = f_high - f_low
fs      = 10 * band
order   = 6

# ========== Вспомогательные функции ==========
def design_bpf(fs, f1, f2, order=6):
    nyq = fs/2
    return butter(order, [f1/nyq, f2/nyq], btype='band')

def plot_spectrum(x, fs, title="Spectrum"):
    N    = len(x)
    w    = x * np.hanning(N)
    F    = np.fft.rfft(w)
    freq = np.fft.rfftfreq(N, 1/fs)
    mag  = np.abs(F)/N
    plt.figure(figsize=(8,4))
    plt.plot(freq, 20*np.log10(mag + 1e-12), 'C0')
    plt.title(title)
    plt.xlabel("Частота, Гц")
    plt.ylabel("Уровень, dB")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def bytes_to_bits(data: bytes) -> str:
    return ''.join(f"{b:08b}" for b in data)

def bits_to_bytes(bstr: str) -> bytes:
    return bytes(int(bstr[i:i+8], 2) for i in range(0, len(bstr), 8))

# ========== Модуляция DQPSK ==========
def modulate_multi_dqpsk(bit_string: str, M: int) -> np.ndarray:
    pad = (-len(bit_string)) % (2*M)
    bit_string += '0' * pad

    Rs    = band / M
    Ts    = 1 / Rs
    Nsym  = int(fs * Ts)
    t     = np.arange(Nsym) / fs

    fcs = [f_low + (i + 0.5)*Rs for i in range(M)]
    map_dphi = {
        "00": 0.0,
        "01":  math.pi/2,
        "11":  math.pi,
        "10": -math.pi/2,
    }

    phases = np.zeros(M)
    nsym   = len(bit_string)//(2*M)
    out    = np.zeros(nsym * Nsym, dtype=np.float32)

    for n in range(nsym):
        buf = np.zeros(Nsym, dtype=np.float32)
        for m, fc in enumerate(fcs):
            bits = bit_string[2*M*n + 2*m : 2*M*n + 2*m + 2]
            dphi = map_dphi[bits]
            phases[m] = (phases[m] + dphi + math.pi) % (2*math.pi) - math.pi
            buf += np.cos(2*math.pi*fc*t + phases[m])
        out[n*Nsym:(n+1)*Nsym] = buf

    b, a = design_bpf(fs, f_low, f_high, order)
    filtered = filtfilt(b, a, out)
    return filtered / (np.max(np.abs(filtered)) or 1.0)

# ========== Демодуляция DQPSK ==========
def demodulate_multi_dqpsk(rx: np.ndarray, M: int) -> str:
    b, a = design_bpf(fs, f_low, f_high, order)
    rx_f  = filtfilt(b, a, rx)

    Rs    = band / M
    Ts    = 1 / Rs
    Nsym  = int(fs * Ts)
    t     = np.arange(Nsym) / fs
    fcs   = [f_low + (i + 0.5)*Rs for i in range(M)]
    nsym  = len(rx_f)//Nsym

    Z = np.zeros((nsym, M), dtype=complex)
    for n in range(nsym):
        seg = rx_f[n*Nsym:(n+1)*Nsym]
        for m, fc in enumerate(fcs):
            Ir = np.dot(seg, np.cos(2*math.pi*fc*t))
            Qr = -np.dot(seg, np.sin(2*math.pi*fc*t))
            Z[n, m] = Ir + 1j*Qr

    bits_out = []
    prev = np.ones(M, dtype=complex)
    for n in range(nsym):
        diff = Z[n] * np.conj(prev)
        prev = Z[n]
        for phi in np.angle(diff):
            if   phi >=  3*math.pi/4 or phi < -3*math.pi/4: bits_out += ["1","1"]
            elif phi >=  math.pi/4:                        bits_out += ["0","1"]
            elif phi >= -math.pi/4:                        bits_out += ["0","0"]
            else:                                          bits_out += ["1","0"]
    return ''.join(bits_out)

# ========== AWGN для тестов ==========
def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    P     = np.var(x)
    sigma = np.sqrt(P / (2 * 10**(snr_db/10)))
    return x + sigma * np.random.randn(*x.shape)

def eval_ber(M: int, snr_db: float, trials: int, bits_per_trial: int) -> float:
    errs = 0
    total = 0
    for _ in range(trials):
        rand_bits = ''.join(np.random.choice(["0","1"], bits_per_trial))
        tx = modulate_multi_dqpsk(rand_bits, M)
        rx = add_awgn(tx, snr_db)
        rec = demodulate_multi_dqpsk(rx, M)[:bits_per_trial]
        errs += sum(b1 != b2 for b1, b2 in zip(rand_bits, rec))
        total += bits_per_trial
    return errs / total

def auto_select_M(M_list, snr_db, trials, bits_per_trial):
    bers = []
    for M in M_list:
        ber = eval_ber(M, snr_db, trials, bits_per_trial)
        print(f"M={M:>3} → BER={ber:.2e}")
        bers.append(ber)
    best = M_list[int(np.argmin(bers))]
    print(f"Оптимальное M = {best} (минимум BER={min(bers):.2e})")
    plt.figure(figsize=(6,4))
    plt.semilogy(M_list, bers, '-o')
    plt.title(f"BER vs M @ SNR={snr_db}dB")
    plt.xlabel("M (число поднесущих)")
    plt.ylabel("BER")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return best

# ========== WAV I/O ==========
def save_wave(fn, x):
    mx = np.max(np.abs(x)) or 1.0
    sig = (x/mx*32767).astype(np.int16)
    with wave.open(fn, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(sig.tobytes())

def load_wave(fn):
    with wave.open(fn,'rb') as wf:
        n    = wf.getnframes()
        data = wf.readframes(n)
    vals = struct.unpack('<'+'h'*n, data)
    return np.array(vals, dtype=np.float32) / 32767.0

# ========== Главная ==========
if __name__ == "__main__":
    WAV_FILE = "dqpsk_multi_rs.wav"

    # Авто-подбор M?
    if input("Автоматический подбор M? [y/N]: ").lower().startswith('y'):
        Ms       = list(map(int, input("Список M [10 20 40 60 80]: ").split() or [10,20,40,60,80]))
        SNR      = float(input("SNR для теста, дБ [15]: ") or "15")
        trials   = int(input("Испытаний [20]: ") or "20")
        bits_len = int(input("Бит/итерация [1000]: ") or "1000")
        M = auto_select_M(Ms, SNR, trials, bits_len)
    else:
        try:
            M = int(input("Число поднесущих [50]: ") or "50")
            assert M>0
        except:
            M = 50

    # Использовать RS-кодирование?
    use_rs = input("Коррекция ошибок RS? [y/N]: ").lower().startswith('y')
    if use_rs:
        nsym = int(input("Избыточных байт RS [32]: ") or "32")
        rs   = RSCodec(nsym)

    text = input("Введите текст: ")
    if use_rs:
        data_bytes    = text.encode('utf-8')
        encoded_bytes = rs.encode(data_bytes)
        bits          = bytes_to_bits(encoded_bytes)
        print(f"Входных байт: {len(data_bytes)}, с RS: {len(encoded_bytes)}")
    else:
        bits = text.encode('utf-8').hex()  # если не RS, просто raw bits
        bits = ''.join(f"{int(bits[i:i+2],16):08b}" for i in range(0,len(bits),2))

    print(f"Генерируем {len(bits)} бит, M={M}")
    tx = modulate_multi_dqpsk(bits, M)
    print(f"Сгенерировано {len(tx)} сэмплов")

    plot_spectrum(tx, fs, title="DQPSK Multi-Carrier Spectrum")

    save_wave(WAV_FILE, tx)
    print("Сигнал сохранён в", WAV_FILE)
    rx = load_wave(WAV_FILE)
    print(f"Прочитано {len(rx)} сэмплов")

    rec_bits = demodulate_multi_dqpsk(rx, M)[:len(bits)]

    if use_rs:
        # восстанавливаем байты и RS-декодируем
        rec_bytes = bits_to_bytes(rec_bits[:len(encoded_bytes)*8])
        try:
            decoded = rs.decode(rec_bytes)
            # если RS.decode вернул tuple, берем первый элемент
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            rec_text = decoded.decode('utf-8', errors='replace')
            status = "OK"
        except ReedSolomonError:
            rec_text = ""
            status   = "RS decode error"
    else:
        # без RS — просто текст
        rec_bytes = bits_to_bytes(rec_bits)
        rec_text  = rec_bytes.decode('utf-8', errors='replace')
        status    = "OK" if rec_text == text else "ERROR"

    print("Восстановленный текст:")
    print(rec_text)
    print("Статус:", status)

