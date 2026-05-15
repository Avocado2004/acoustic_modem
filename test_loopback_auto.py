#!/usr/bin/env python3
"""
Автоматизированный loopback-тест для Acoustic Modem.

Симулирует передачу данных через канал с искажениями:
1. Генерирует случайный текст
2. Модулирует в OFDM сигнал
3. Применяет искажения канала (AWGN, multipath, АЧХ, затухание)
4. Демодулирует из искажённого сигнала
5. Сравнивает результат с оригиналом

Запуск: python test_loopback_auto.py
"""

import sys
import os
import random
import string
import numpy as np

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_loopback_test(percent=50.0, data_size=256, modulation="QPSK", seed=42):
    """
    Запуск loopback-теста с заданными параметрами.

    Параметры
    ----------
    percent : float
        Процент искажений канала (0-100)
    data_size : int
        Размер данных в байтах
    modulation : str
        Тип модуляции ("QPSK" или "BPSK")
    seed : int
        Seed для генератора случайных чисел

    Возвращает
    -------
    bool
        True если приём успешен и данные совпадают
    """
    import modem_config
    from modem_config import set_modulation, init_phases, fs, wavfile
    from modem_tx import transmit_text
    from signal_processor import receive_from_file
    from channel_simulator import ChannelSimulator

    print(f"\n{'='*60}")
    print(f"[TEST] Loopback test: percent={percent}%, data_size={data_size}B, mod={modulation}")
    print(f"{'='*60}")

    # Устанавливаем модуляцию
    set_modulation(modulation)
    init_phases()

    # Генерируем случайный текст
    rng = np.random.RandomState(seed)
    random.seed(seed)
    random_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(data_size))
    original_data = random_text.encode('utf-8')
    print(f"[TEST] Сгенерирован текст ({len(original_data)} байт): {random_text[:30]}...")

    # Сохраняем оригинальный текст
    temp_original = "test_loop_original.txt"
    with open(temp_original, "w", encoding='utf-8') as f:
        f.write(random_text)

    # Передаём текст
    print(f"\n[TEST] === ПЕРЕДАЧА ===")
    transmit_text(random_text)
    print(f"[TEST] Передача завершена")

    # Загружаем WAV-файл
    wav_filename = "ofdm_acoustic_tx_with_noise.wav"
    if not os.path.exists(wav_filename):
        print(f"[TEST-ERR] WAV-файл не найден: {wav_filename}")
        return False

    fs_wav, sig = wavfile.read(wav_filename)
    print(f"[TEST] Загружен WAV: {len(sig)} samples, dtype={sig.dtype}")

    # Применяем искажения канала
    print(f"\n[TEST] === ИСКАЖЕНИЯ КАНАЛА ({percent}%) ===")
    sim = ChannelSimulator(fs=fs_wav, seed=seed)
    sim.reset_channel_state()

    # Преобразуем к float64
    if np.issubdtype(sig.dtype, np.integer):
        sig_float = sig.astype(np.float64) / np.iinfo(sig.dtype).max
    else:
        sig_float = sig.astype(np.float64)

    print(f"[TEST] Исходный RMS: {np.sqrt(np.mean(sig_float**2)):.6f}")

    # 1. AWGN
    snr_db = 30.0 - (30.0 - 15.0) * (percent / 100.0)
    sig_float = sim.add_awgn(sig_float, snr_db)
    print(f"[TEST] После AWGN ({snr_db:.1f} dB): RMS={np.sqrt(np.mean(sig_float**2)):.6f}")

    # 2. Multipath
    if percent > 0:
        direct_gain = 1.0 - (1.0 - 0.6) * (percent / 100.0)
        mp1 = 0.3 if percent > 30 else 0.3 * (percent / 30.0)
        mp2 = 0.1 if percent > 60 else (0.1 * (percent / 60.0) if percent > 0 else 0.0)
        delays_ms = [0.0, 2.0, 5.0]
        gains = [direct_gain, mp1, mp2]
        sig_float = sim.apply_multipath(sig_float, delays_ms, gains)
        print(f"[TEST] После multipath: RMS={np.sqrt(np.mean(sig_float**2)):.6f}")

    # 3. Frequency offset
    freq_offset = 5.0 * (percent / 100.0)
    if freq_offset > 0:
        sig_float = sim.apply_frequency_offset(sig_float, freq_offset)
        print(f"[TEST] После freq offset ({freq_offset:.2f} Hz): RMS={np.sqrt(np.mean(sig_float**2)):.6f}")

    # 4. Random frequency response
    sig_float = sim.apply_random_frequency_response(sig_float, percent=percent)
    print(f"[TEST] После АЧХ: RMS={np.sqrt(np.mean(sig_float**2)):.6f}")

    # 5. Distance attenuation
    sig_float = sim.apply_distance_attenuation(sig_float, percent=percent)
    print(f"[TEST] После затухания: RMS={np.sqrt(np.mean(sig_float**2)):.6f}")

    # Нормализуем
    max_val = np.max(np.abs(sig_float))
    if max_val > 1.0:
        sig_float = sig_float / max_val

    # Сохраняем искажённый сигнал
    distorted_int16 = (sig_float * 32767).astype(np.int16)
    wavfile.write(wav_filename, fs_wav, distorted_int16)
    print(f"[TEST] Искажённый сигнал сохранён в {wav_filename}")

    # Принимаем
    print(f"\n[TEST] === ПРИЁМ ===")
    received_file = "rx_text.txt"
    if os.path.exists(received_file):
        os.remove(received_file)

    success = receive_from_file(wav_filename)

    if not success:
        print(f"\n[TEST-ERR] ❌ Приём не удался!")
        return False

    # Сравниваем
    if os.path.exists(received_file):
        with open(received_file, "r", encoding='utf-8') as f:
            received_text = f.read()

        if random_text == received_text:
            print(f"\n[TEST] ✅ УСПЕХ: Текст совпадает полностью!")
            return True
        else:
            print(f"\n[TEST-ERR] ❌ Текст отличается!")
            min_len = min(len(random_text), len(received_text))
            for i in range(min_len):
                if random_text[i] != received_text[i]:
                    print(f"[TEST] Первое отличие в позиции {i}: ориг='{random_text[i]}', принято='{received_text[i]}'")
                    break
            if len(random_text) != len(received_text):
                print(f"[TEST] Длина: ориг={len(random_text)}, принято={len(received_text)}")
            return False
    else:
        print(f"\n[TEST-ERR] ❌ Файл {received_file} не найден!")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Автоматизированный loopback-тест Acoustic Modem")
    print("=" * 60)

    # Тест 1: Без искажений (0%)
    result1 = run_loopback_test(percent=0.0, data_size=256, modulation="QPSK", seed=42)
    print(f"\n{'='*60}")
    print(f"Тест 1 (0% искажений): {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"{'='*60}")

    # Тест 2: Умеренные искажения (30%)
    result2 = run_loopback_test(percent=30.0, data_size=256, modulation="QPSK", seed=123)
    print(f"\n{'='*60}")
    print(f"Тест 2 (30% искажений): {'✅ PASS' if result2 else '❌ FAIL'}")
    print(f"{'='*60}")

    # Тест 3: Сильные искажения (70%)
    result3 = run_loopback_test(percent=70.0, data_size=256, modulation="QPSK", seed=456)
    print(f"\n{'='*60}")
    print(f"Тест 3 (70% искажений): {'✅ PASS' if result3 else '❌ FAIL'}")
    print(f"{'='*60}")

    # Итог
    all_passed = result1 and result2 and result3
    print(f"\n{'='*60}")
    print(f"ИТОГО: {'✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if all_passed else '❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ'}")
    print(f"  Тест 1 (0%):  {'PASS' if result1 else 'FAIL'}")
    print(f"  Тест 2 (30%): {'PASS' if result2 else 'FAIL'}")
    print(f"  Тест 3 (70%): {'PASS' if result3 else 'FAIL'}")
    print(f"{'='*60}")

    sys.exit(0 if all_passed else 1)
