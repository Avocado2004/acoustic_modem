import numpy as np
import wave
import struct
import matplotlib.pyplot as plt

# 📌 Параметры
TEXT = "Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos."
FILENAME = "output.wav"
SYMBOL_DURATION = 0.02
SAMPLE_RATE = 44100
SUBCARRIER_COUNT = 96
FREQ_START = 500
DELTA_F = 1 / SYMBOL_DURATION  # 50 Hz
LOG_FILE = "ofdm_demodulation.log"

# ✅ Ортогональные частоты
freqs = [FREQ_START + i * DELTA_F for i in range(SUBCARRIER_COUNT)]

# 📤 Текст → биты
def text_to_bits(text):
    print(f"🔤 Исходный текст: {text[:60]}...")
    bits = ''.join(f'{ord(c):08b}' for c in text)
    print(f"📤 Биты: {bits[:64]}...")
    return bits

# 📥 Биты → текст
def bits_to_text(bits):
    bits = bits[:len(bits) - (len(bits) % 8)]
    print(f"📏 Всего битов: {len(bits)} (остаток: {len(bits) % 8})")
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    try:
        text = ''.join(chr(int(c, 2)) for c in chars)
        print(f"📥 Восстановленный текст: {text[:60]}...")
        return text
    except Exception as e:
        print(f"❌ Ошибка восстановления текста: {e}")
        return ""

# 📡 Модуляция OFDM
def modulate(bits):
    print("📡 Модулируем OFDM-сигнал...")
    bits_per_symbol = SUBCARRIER_COUNT
    bit_chunks = [bits[i:i+bits_per_symbol] for i in range(0, len(bits), bits_per_symbol)]
    signal = []
    t = np.linspace(0, SYMBOL_DURATION, int(SAMPLE_RATE * SYMBOL_DURATION), endpoint=False)

    for chunk in bit_chunks:
        symbol = np.zeros_like(t)
        for i, bit in enumerate(chunk):
            if bit == '1':
                symbol += np.sin(2 * np.pi * freqs[i] * t)
        signal.extend(symbol)

    signal = np.array(signal)
    print(f"📈 Длина сигнала: {len(signal)}")
    return signal

# 📶 Демодуляция OFDM
def demodulate(signal):
    print("📶 Демодулируем OFDM-сигнал...")
    samples_per_symbol = int(SAMPLE_RATE * SYMBOL_DURATION)
    num_symbols = len(signal) // samples_per_symbol
    bits = ''
    t = np.linspace(0, SYMBOL_DURATION, samples_per_symbol, endpoint=False)

    with open(LOG_FILE, 'w') as log:
        for i in range(num_symbols):
            symbol = signal[i * samples_per_symbol:(i + 1) * samples_per_symbol]
            norm = np.max(np.abs(symbol))
            symbol = symbol / norm if norm != 0 else symbol
            chunk = ''
            for j, f in enumerate(freqs):
                ref = np.sin(2 * np.pi * f * t)
                power = np.dot(symbol, ref)
                log.write(f"Symbol {i}, Subcarrier {j} ({int(f)} Hz): Power = {power:.4f}\n")
                chunk += '1' if power > 0.5 else '0'
            bits += chunk

    print(f"📤 Извлечённые биты: {bits[:64]}...")
    print(f"📄 Лог мощности сохранён в {LOG_FILE}")
    return bits

# 💾 Сохранение WAV
def save_wave(filename, signal):
    print(f"💾 Сохраняем в {filename}...")
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        max_val = np.max(np.abs(signal))
        scaled = np.int16(signal / max_val * 32767) if max_val != 0 else np.zeros_like(signal, dtype=np.int16)
        wf.writeframes(struct.pack('<' + 'h' * len(scaled), *scaled))

# 📂 Загрузка WAV
def load_wave(filename):
    print(f"📂 Загружаем из {filename}...")
    with wave.open(filename, 'r') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = struct.unpack('<' + 'h' * wf.getnframes(), frames)
        return np.array(samples, dtype=np.float32) / 32767

# 📊 Визуализация спектра
def plot_spectrum(signal):
    print("📊 Строим спектр сигнала...")
    fft = np.fft.rfft(signal)
    freqs_fft = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs_fft, np.abs(fft))
    plt.title("OFDM Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ofdm_spectrum_visualization.png")
    print("✅ Спектр сохранён в ofdm_spectrum_visualization.png")

# 🚀 Основной процесс
bits = text_to_bits(TEXT)
signal = modulate(bits)
save_wave(FILENAME, signal)
plot_spectrum(signal)
loaded_signal = load_wave(FILENAME)
recovered_bits = demodulate(loaded_signal)
restored_text = bits_to_text(recovered_bits)

# 🧾 Вывод полного текста
print("\n📜 Восстановленный текст:\n")
print(restored_text)



