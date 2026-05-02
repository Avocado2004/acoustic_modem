"""
Скрипт для диагностики проблемы приема сигнала с телефона.
Записывает сигнал с микрофона и анализирует его спектр.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os
import time

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modem_config import fs, Nfft, Ncp, Nsub, subc_inds, f_low_hz, f_high_hz, SYMBOL_LEN
from modem_modulation import build_preamble

def record_and_analyze(duration=10):
    """Запись сигнала с микрофона и анализ."""
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] sounddevice not installed. Install: pip install sounddevice")
        return
    
    print(f"[DEBUG] Запись с микрофона {duration} секунд...")
    print(f"[DEBUG] Частота дискретизации: {fs} Hz")
    print(f"[DEBUG] Диапазон поднесущих: {f_low_hz:.1f} - {f_high_hz:.1f} Hz")
    print(f"[DEBUG] Нажмите Enter, когда будете готовы воспроизвести сигнал с телефона...")
    input()
    
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    recording = recording.flatten()
    
    print(f"[DEBUG] Записано {len(recording)} samples")
    print(f"[DEBUG] RMS записи: {np.sqrt(np.mean(recording**2)):.6f}")
    
    # Сохраняем запись
    wavfile.write('phone_recording.wav', fs, (recording * 32767).astype(np.int16))
    print(f"[DEBUG] Запись сохранена в phone_recording.wav")
    
    # Анализируем спектр
    analyze_spectrum(recording, fs)

def analyze_spectrum(data, sample_rate):
    """Анализ спектра сигнала."""
    print(f"\n[DEBUG] Анализ спектра...")
    print(f"[DEBUG] Частота дискретизации: {sample_rate} Hz")
    print(f"[DEBUG] Длина сигнала: {len(data)} samples ({len(data)/sample_rate:.2f} sec)")
    
    # Нормализация
    data = data.astype(float)
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    
    # Строим спектрограмму
    n_fft = 1024
    spectrum = np.abs(np.fft.fft(data[:n_fft])) / n_fft
    freqs = np.fft.fftfreq(n_fft, 1.0/sample_rate)
    
    # Ищем пики в диапазоне поднесущих
    mask = (freqs >= f_low_hz - 500) & (freqs <= f_high_hz + 500)
    if np.any(mask):
        spec_range = spectrum[mask]
        freq_range = freqs[mask]
        peak_idx = np.argmax(spec_range)
        peak_freq = freq_range[peak_idx]
        peak_mag = spec_range[peak_idx]
        print(f"[DEBUG] Пик в диапазоне поднесущих: {peak_freq:.1f} Hz, амплитуда: {peak_mag:.6f}")
        print(f"[DEBUG] Ожидаемый диапазон: {f_low_hz:.1f} - {f_high_hz:.1f} Hz")
        
        # Проверяем сдвиг частоты
        freq_shift = peak_freq - f_low_hz
        if abs(freq_shift) > 200:
            print(f"[DEBUG] ВНИМАНИЕ: Обнаружен сдвиг частоты {freq_shift:.1f} Hz!")
            print(f"[DEBUG] Возможная причина: частота дискретизации телефона отличается от {fs} Hz")
            # Предполагаемая частота дискретизации телефона
            assumed_fs_phone = sample_rate * (peak_freq / f_low_hz)
            print(f"[DEBUG] Предполагаемая частота дискретизации телефона: {assumed_fs_phone:.0f} Hz")
    
    # Строим спектрограмму
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(freqs[:n_fft//2], 20*np.log10(spectrum[:n_fft//2] + 1e-12))
    plt.axvline(f_low_hz, color='r', linestyle='--', label=f'Low: {f_low_hz:.1f} Hz')
    plt.axvline(f_high_hz, color='g', linestyle='--', label=f'High: {f_high_hz:.1f} Hz')
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
    
    # Проверяем амплитуду в диапазоне поднесущих
    print(f"\n[DEBUG] Проверка амплитуды в диапазоне поднесущих:")
    print(f"[DEBUG] Если амплитуда низкая, возможные причины:")
    print(f"[DEBUG]  1. Телефон имеет плохую АЧХ в этом диапазоне")
    print(f"[DEBUG]  2. Громкость телефона слишком низкая")
    print(f"[DEBUG]  3. Микрофон слишком далеко или близко (эффект ближнего поля)")
    print(f"[DEBUG]  4. Программная обработка звука на телефоне (эквалайзер, шумоподавление)")

def analyze_wav_file(wav_path):
    """Анализ WAV-файла с записью сигнала с телефона."""
    print(f"[DEBUG] Анализ файла: {wav_path}")
    
    if not os.path.exists(wav_path):
        print(f"[ERROR] Файл не найден: {wav_path}")
        return
    
    sample_rate, data = wavfile.read(wav_path)
    print(f"[DEBUG] Частота дискретизации файла: {sample_rate} Hz")
    print(f"[DEBUG] Частота дискретизации системы: {fs} Hz")
    
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
    
    analyze_spectrum(data, sample_rate)

if __name__ == "__main__":
    print("=" * 60)
    print("Диагностика проблемы приема сигнала с телефона")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Анализ существующего WAV-файла
        wav_path = sys.argv[1]
        analyze_wav_file(wav_path)
    else:
        # Запись с микрофона
        print("\nВыберите режим:")
        print("1 - Запись с микрофона")
        print("2 - Анализ WAV-файла")
        choice = input("Ваш выбор (1/2): ")
        
        if choice == "1":
            duration = input("Длительность записи в секундах (по умолчанию 10): ")
            duration = int(duration) if duration.isdigit() else 10
            record_and_analyze(duration)
        elif choice == "2":
            wav_path = input("Путь к WAV-файлу: ")
            analyze_wav_file(wav_path)
        else:
            print("Неверный выбор")
