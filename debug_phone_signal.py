"""
Скрипт для диагностики проблемы с приемом сигнала с телефона.
Записывает сигнал с микрофона и анализирует спектр.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modem_config import fs, Nfft, Ncp, Nsub, subc_inds, f_low_hz, f_high_hz, SYMBOL_LEN
from modem_modulation import build_preamble

def analyze_signal_file(wav_path):
    """Анализ WAV-файла с записью сигнала с телефона."""
    print(f"[DEBUG] Анализ файла: {wav_path}")
    print(f"[DEBUG] Частота дискретизации системы: {fs} Hz")
    print(f"[DEBUG] Диапазон поднесущих: {f_low_hz:.1f} - {f_high_hz:.1f} Hz")
    
    # Читаем файл
    sample_rate, data = wavfile.read(wav_path)
    print(f"[DEBUG] Частота дискретизации файла: {sample_rate} Hz")
    
    # Нормализация
    if data.dtype == np.int16:
        data = data.astype(float) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(float) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        data = data.astype(float)
    
    if data.ndim > 1:
        data = data[:, 0]
    
    print(f"[DEBUG] Длина сигнала: {len(data)} samples ({len(data)/sample_rate:.2f} sec)")
    print(f"[DEBUG] RMS сигнала: {np.sqrt(np.mean(data**2)):.6f}")
    print(f"[DEBUG] Min/Max: {np.min(data):.6f} / {np.max(data):.6f}")
    
    # Если частота дискретизации файла отличается от системной
    if sample_rate != fs:
        print(f"[DEBUG] ВНИМАНИЕ: Частота дискретизации файла ({sample_rate}) отличается от системной ({fs})!")
        print(f"[DEBUG] Это может вызывать сдвиг частот поднесущих.")
        # Пересчитываем ожидаемые частоты
        freq_ratio = fs / sample_rate
        expected_low = f_low_hz * freq_ratio
        expected_high = f_high_hz * freq_ratio
        print(f"[DEBUG] Ожидаемый диапазон поднесущих с учетом сдвига: {expected_low:.1f} - {expected_high:.1f} Hz")
    
    # Строим спектр
    n_fft = 1024
    spectrum = np.abs(np.fft.fft(data[:n_fft])) / n_fft
    freqs = np.fft.fftfreq(n_fft, 1.0/sample_rate)
    
    # Ищем пики в диапазоне поднесущих
    if sample_rate != fs:
        freq_ratio = fs / sample_rate
        search_low = f_low_hz * freq_ratio - 500
        search_high = f_high_hz * freq_ratio + 500
    else:
        search_low = f_low_hz - 500
        search_high = f_high_hz + 500
    
    mask = (freqs >= search_low) & (freqs <= search_high)
    if np.any(mask):
        spec_range = spectrum[mask]
        freq_range = freqs[mask]
        peak_idx = np.argmax(spec_range)
        peak_freq = freq_range[peak_idx]
        peak_mag = spec_range[peak_idx]
        print(f"[DEBUG] Пик в диапазоне поднесущих: {peak_freq:.1f} Hz, амплитуда: {peak_mag:.6f}")
    
    # Сохраняем спектрограмму
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(freqs[:n_fft//2], 20*np.log10(spectrum[:n_fft//2] + 1e-12))
    plt.axvline(f_low_hz, color='r', linestyle='--', label=f'Low: {f_low_hz:.1f} Hz')
    plt.axvline(f_high_hz, color='g', linestyle='--', label=f'High: {f_high_hz:.1f} Hz')
    if sample_rate != fs:
        plt.axvline(expected_low, color='r', linestyle=':', label=f'Expected Low (shifted): {expected_low:.1f} Hz')
        plt.axvline(expected_high, color='g', linestyle=':', label=f'Expected High (shifted): {expected_high:.1f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Spectrum of received signal')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.specgram(data, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='jet')
    plt.axhline(f_low_hz, color='r', linestyle='--', label=f'Low: {f_low_hz:.1f} Hz')
    plt.axhline(f_high_hz, color='g', linestyle='--', label=f'High: {f_high_hz:.1f} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.colorbar(label='Power (dB)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('debug_spectrum.png')
    print(f"[DEBUG] Спектрограмма сохранена в debug_spectrum.png")
    
    # Сохраняем данные для дальнейшего анализа
    np.savez('debug_signal.npz', data=data, sample_rate=sample_rate, fs=fs)
    print(f"[DEBUG] Данные сохранены в debug_signal.npz")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        print("Использование: python debug_phone_signal.py <wav_file>")
        print("Или запустите без аргументов для записи с микрофона...")
        # Можно добавить запись с микрофона
        sys.exit(1)
    
    analyze_signal_file(wav_path)
