#!/usr/bin/env python3
"""
OFDM QPSK Modulation/Demodulation with Timing Synchronization via Cross-Correlation

1. Преобразуем текст в биты.
2. Модулируем в OFDM+QPSK с преамбулой.
3. По кросс-корреляции находим задержку старта преамбулы.
4. По выровненному сигналу оцениваем фазовый сдвиг и демодулируем.
5. Логируем каждый шаг.
"""

import numpy as np
import wave
import struct
import matplotlib.pyplot as plt

# 📌 Параметры
TEXT             = ("Lorem ipsum dolor sit amet consectetur adipiscing elit. "
                    "Quisque faucibus ex sapien vitae pellentesque sem placerat. "
                    "In id cursus mi pretium tellus duis convallis. Tempus leo eu "
                    "aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus "
                    "nec metus bibendum egestas. Iaculis massa nisl malesuada "
                    "lacinia integer nunc posuere. Ut hendrerit semper vel class "
                    "aptent taciti sociosqu. Ad litora torquent per conubia nostra "
                    "inceptos himenaeos.")
FILENAME         = "output.wav"
SYMBOL_DURATION  = 0.02                # секунда на OFDM символ
SAMPLE_RATE      = 44100              # Гц
FREQ_START       = 300                # Нижняя грань OFDM (Гц)
FREQ_END         = 5000               # Верхняя грань OFDM (Гц)
SUBCARRIER_COUNT = int((FREQ_END - FREQ_START) * SYMBOL_DURATION)
LOG_FILE         = "ofdm_demodulation.log"
MOD_LOG_FILE     = "modulation_log.txt"


def text_to_bits(text):
    return ''.join(f"{ord(c):08b}" for c in text)


def bits_to_text(bits):
    l = len(bits) - (len(bits) % 8)
    chars = [bits[i:i+8] for i in range(0, l, 8)]
    try:
        return "".join(chr(int(b, 2)) for b in chars)
    except:
        return ""


def modulate(bits):
    # 1) Выравниваем длину битов на чётное число
    if len(bits) % 2:
        bits += '0'

    # 2) Группируем по 2 бита → QPSK
    pairs   = [bits[i:i+2] for i in range(0, len(bits), 2)]
    mapping = {'00': 1+1j, '01': -1+1j, '11': -1-1j, '10': 1-1j}
    symbols = np.array([mapping[p] for p in pairs])

    # 3) Генерируем преамбулу той же длины
    patterns    = ['00','01','11','10']
    pre_bits    = (patterns * ((SUBCARRIER_COUNT + 3)//4))[:SUBCARRIER_COUNT]
    pre_symbols = np.array([mapping[b] for b in pre_bits])

    # 4) Лог модуляции
    with open(MOD_LOG_FILE, 'w') as flog:
        flog.write("=== Modulation Log ===\n")
        for idx, p in enumerate(pairs):
            flog.write(f"Sym {idx:4d}: bits={p} → {mapping[p]}\n")

    # 5) Разбиваем на блоки: сначала преамбула, затем данные
    chunks = [pre_symbols]
    for i in range(0, len(symbols), SUBCARRIER_COUNT):
        blk = symbols[i:i+SUBCARRIER_COUNT]
        if len(blk) < SUBCARRIER_COUNT:
            blk = np.pad(blk, (0, SUBCARRIER_COUNT - len(blk)))
        chunks.append(blk)

    # 6) IFFT каждого блока и ресемплирование на SYMBOL_DURATION
    N     = int(SAMPLE_RATE * SYMBOL_DURATION)
    t     = np.linspace(0, SYMBOL_DURATION, N, endpoint=False)
    grid  = np.linspace(0, SYMBOL_DURATION, SUBCARRIER_COUNT, endpoint=False)
    out   = []

    for chunk in chunks:
        # ИКПФТ длины SUBCARRIER_COUNT
        td_small = np.fft.ifft(chunk, n=SUBCARRIER_COUNT)
        # Ресемплируем реальную часть на N точек
        res = np.interp(t, grid, np.real(td_small))
        out.extend(res)

    return np.array(out, dtype=np.float32)


def demodulate(signal):
    N = int(SAMPLE_RATE * SYMBOL_DURATION)

    # 1) Реконструируем ту же преамбулу во временной области
    patterns    = ['00','01','11','10']
    mapping     = {'00': 1+1j, '01': -1+1j, '11': -1-1j, '10': 1-1j}
    pre_bits    = (patterns * ((SUBCARRIER_COUNT + 3)//4))[:SUBCARRIER_COUNT]
    pre_symbols = np.array([mapping[b] for b in pre_bits])

    t     = np.linspace(0, SYMBOL_DURATION, N, endpoint=False)
    grid  = np.linspace(0, SYMBOL_DURATION, SUBCARRIER_COUNT, endpoint=False)
    td_pre_small  = np.fft.ifft(pre_symbols, n=SUBCARRIER_COUNT)
    known_pre_td  = np.interp(t, grid, np.real(td_pre_small))

    # 2) Точная временная синхронизация через корреляцию
    corr   = np.abs(np.correlate(signal, known_pre_td, mode='valid'))
    offset = int(np.argmax(corr))

    # 3) Подготовка логирования
    with open(LOG_FILE, 'w') as flog:
        flog.write("=== Timing Sync ===\n")
        flog.write(f"Sample offset = {offset}\n\n")

    # 4) Выровненный сигнал
    aligned = signal[offset:]

    # 5) Оценка фазового сдвига по преамбуле
    pre_blk = aligned[:N]
    t_s     = np.linspace(0, SYMBOL_DURATION, N, endpoint=False)
    t_o     = np.linspace(0, SYMBOL_DURATION, SUBCARRIER_COUNT, endpoint=False)
    pre_rs  = np.interp(t_o, t_s, pre_blk)
    pre_fd  = np.fft.fft(pre_rs)

    best_phi, min_err = 0.0, SUBCARRIER_COUNT + 1
    with open(LOG_FILE, 'a') as flog:
        flog.write("=== Phase Search ===\n")
        for phi in np.linspace(0, 2*np.pi, 72, endpoint=False):
            rot = pre_fd * np.exp(-1j * phi)
            dec_pairs = [
                ('0' if v.real >= 0 else '1') +
                ('0' if v.imag >= 0 else '1')
                for v in rot
            ]
            errs = sum(dp != ep for dp, ep in zip(dec_pairs, pre_bits))
            flog.write(f"phi={phi:.3f} rad → errors={errs}/{SUBCARRIER_COUNT}\n")
            if errs < min_err:
                min_err, best_phi = errs, phi

        flog.write(f"\nSelected phase_offset = {best_phi:.3f} rad "
                   f"(preamble errors: {min_err}/{SUBCARRIER_COUNT})\n\n")

        # 6) Демодуляция всех последующих блоков
        flog.write("=== Data Demodulation ===\n")
        decoded = []
        num_sym = len(aligned) // N
        for blk_idx in range(1, num_sym):
            blk = aligned[blk_idx*N:(blk_idx+1)*N]
            rs  = np.interp(t_o, t_s, blk)
            fd  = np.fft.fft(rs) * np.exp(-1j * best_phi)
            for v in fd:
                b1 = '0' if v.real >= 0 else '1'
                b2 = '0' if v.imag >= 0 else '1'
                decoded.append(b1 + b2)
                flog.write(f"blk={blk_idx}, Re={v.real:+.2f}, Im={v.imag:+.2f} → {b1}{b2}\n")

    return ''.join(decoded)


def save_wave(fn, sig):
    mx   = np.max(np.abs(sig)) or 1.0
    data = np.int16(sig/mx * 32767)
    with wave.open(fn, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack('<' + 'h'*len(data), *data))


def load_wave(fn):
    with wave.open(fn, 'r') as wf:
        frames = wf.readframes(wf.getnframes())
        vals   = struct.unpack('<' + 'h'*wf.getnframes(), frames)
        return np.array(vals, dtype=np.float32) / 32767


def plot_spectrum(sig, fn="ofdm_spectrum.png"):
    sp = np.fft.rfft(sig)
    fr = np.fft.rfftfreq(len(sig), 1/SAMPLE_RATE)
    plt.figure(figsize=(8,3))
    plt.plot(fr, np.abs(sp))
    plt.title("OFDM Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


if __name__ == "__main__":
    # 1) Текст → биты → OFDM модуляция
    bits     = text_to_bits(TEXT)
    tx_signal= modulate(bits)

    # 2) Прямая демодуляция для оценки BER
    rec_direct = demodulate(tx_signal)
    L          = min(len(rec_direct), len(bits)//2)
    errs       = sum(1 for i in range(L)
                     if rec_direct[i] != bits[2*i:2*i+2])
    print(f"Direct BER = {errs}/{L} symbols → {errs/L:.4f}")

    # 3) Сохраняем WAV и строим спектр
    save_wave(FILENAME, tx_signal)
    plot_spectrum(tx_signal)

    # 4) Демодулируем из WAV, восстанавливаем текст
    rx      = load_wave(FILENAME)
    rec_wav = demodulate(rx)
    restored= bits_to_text(rec_wav)
    print("\nRestored text via WAV:\n")
    print(restored)

