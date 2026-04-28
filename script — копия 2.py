import numpy as np
import wave
import struct

# Параметры
TEXT = "Hello"
FILENAME = "output.wav"
SYMBOL_DURATION = 0.01
SAMPLE_RATE = 44100
BITS_PER_SYMBOL = 6

def text_to_bits(text):
    print(f"🔤 Исходный текст: {text}")
    bits = ''.join(f'{ord(c):08b}' for c in text)
    print(f"📤 Биты: {bits}")
    return bits

def bits_to_text(bits):
    bits = bits[:len(bits) - (len(bits) % 8)]
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    text = ''.join(chr(int(c, 2)) for c in chars)
    print(f"📥 Восстановленный текст: {text}")
    return text

def modulate(bits):
    print("📡 Модуляция...")
    bit_chunks = [bits[i:i+BITS_PER_SYMBOL] for i in range(0, len(bits), BITS_PER_SYMBOL)]
    signal = []

    for chunk in bit_chunks:
        freqs = [1000 + i * 100 for i, bit in enumerate(chunk) if bit == '1']
        t = np.linspace(0, SYMBOL_DURATION, int(SAMPLE_RATE * SYMBOL_DURATION), endpoint=False)
        symbol = sum(np.sin(2 * np.pi * f * t) for f in freqs)
        signal.extend(symbol)

    signal = np.array(signal)
    print(f"📈 Длина сигнала: {len(signal)}")
    return signal

def demodulate(signal):
    print("📶 Демодуляция...")
    samples_per_symbol = int(SAMPLE_RATE * SYMBOL_DURATION)
    num_symbols = len(signal) // samples_per_symbol
    bits = ''

    for i in range(num_symbols):
        symbol = signal[i * samples_per_symbol:(i + 1) * samples_per_symbol]
        t = np.linspace(0, SYMBOL_DURATION, samples_per_symbol, endpoint=False)
        chunk = ''
        for j in range(BITS_PER_SYMBOL):
            f = 1000 + j * 100
            ref = np.sin(2 * np.pi * f * t)
            power = np.dot(symbol, ref)
            chunk += '1' if power > 30 else '0'
        bits += chunk

    print(f"📤 Извлечённые биты: {bits}")
    return bits

def save_wave(filename, signal):
    print(f"💾 Сохраняем в {filename}...")
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        max_val = np.max(np.abs(signal))
        scaled = np.int16(signal / max_val * 32767) if max_val != 0 else np.zeros_like(signal, dtype=np.int16)
        wf.writeframes(struct.pack('<' + 'h' * len(scaled), *scaled))

def load_wave(filename):
    print(f"📂 Загружаем из {filename}...")
    with wave.open(filename, 'r') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = struct.unpack('<' + 'h' * wf.getnframes(), frames)
        return np.array(samples, dtype=np.float32) / 32767

# 🚀 Основной процесс
bits = text_to_bits(TEXT)
signal = modulate(bits)
save_wave(FILENAME, signal)
loaded_signal = load_wave(FILENAME)
recovered_bits = demodulate(loaded_signal)
bits_to_text(recovered_bits)

