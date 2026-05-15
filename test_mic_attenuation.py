#!/usr/bin/env python3
"""
Тест микрофонного приёма с ослаблением сигнала.

Симулирует микрофонный приём путём ослабления сигнала
(как при прохождении через динамик -> воздух -> микрофон).

Запуск: python test_mic_attenuation.py
"""

import sys
import os
import random
import string
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_test(attenuation, data_size=128, seed=42):
    """
    Тест с заданным ослаблением сигнала.

    Параметры
    ----------
    attenuation : float
        Коэффициент ослабления (1.0 = без ослабления, 0.1 = 10x ослабление)
    data_size : int
        Размер данных в байтах
    seed : int
        Seed для генератора случайных чисел

    Возвращает
    -------
    bool
        True если приём успешен
    """
    import modem_config
    from modem_config import set_modulation, init_phases, fs, wavfile
    from modem_tx import transmit_text
    from signal_processor import receive_from_file

    print(f"\n{'='*60}")
    print(f"[TEST] Ослабление: {attenuation}x (attenuation={attenuation})")
    print(f"{'='*60}")

    set_modulation("QPSK")
    init_phases()

    random.seed(seed)
    test_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(data_size))

    transmit_text(test_text)

    wav_filename = "ofdm_acoustic_tx_with_noise.wav"
    if not os.path.exists(wav_filename):
        print("[ERR] WAV-файл не найден")
        return False

    fs_wav, sig = wavfile.read(wav_filename)

    if np.issubdtype(sig.dtype, np.integer):
        sig_float = sig.astype(np.float64) / np.iinfo(sig.dtype).max
    else:
        sig_float = sig.astype(np.float64)

    print(f"[TEST] Исходный RMS: {np.sqrt(np.mean(sig_float**2)):.6f}")

    # Ослабляем сигнал
    sig_float *= attenuation

    print(f"[TEST] После ослабления RMS: {np.sqrt(np.mean(sig_float**2)):.6f}")

    # Сохраняем
    max_val = np.max(np.abs(sig_float))
    if max_val > 1.0:
        sig_float = sig_float / max_val

    distorted_int16 = (sig_float * 32767).astype(np.int16)
    wavfile.write(wav_filename, fs_wav, distorted_int16)

    # Принимаем
    received_file = "rx_text.txt"
    if os.path.exists(received_file):
        os.remove(received_file)

    success = receive_from_file(wav_filename)

    if not success:
        print(f"\n[TEST] ❌ FAIL: attenuation={attenuation} — приём не удался")
        return False

    if os.path.exists(received_file):
        with open(received_file, "r", encoding='utf-8') as f:
            received_text = f.read()

        if test_text == received_text:
            print(f"\n[TEST] ✅ PASS: attenuation={attenuation}")
            return True
        else:
            print(f"\n[TEST] ❌ FAIL: attenuation={attenuation} — текст отличается")
            print(f"  Оригинал: {test_text[:30]}...")
            print(f"  Принято:  {received_text[:30]}...")
            return False
    else:
        print(f"\n[TEST] ❌ FAIL: attenuation={attenuation} — файл не найден")
        return False


def main():
    print("=" * 60)
    print("Тест микрофонного приёма с ослаблением сигнала")
    print("=" * 60)

    # Подготавливаем сигнал один раз
    import modem_config
    from modem_config import set_modulation, init_phases, fs, wavfile
    from modem_tx import transmit_text
    from signal_processor import receive_from_file

    set_modulation("QPSK")
    init_phases()

    random.seed(42)
    test_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(128))

    transmit_text(test_text)

    wav_filename = "ofdm_acoustic_tx_with_noise.wav"
    fs_wav, sig = wavfile.read(wav_filename)

    if np.issubdtype(sig.dtype, np.integer):
        sig_float = sig.astype(np.float64) / np.iinfo(sig.dtype).max
    else:
        sig_float = sig.astype(np.float64)

    print(f"[INFO] Сигнал подготовлен: {len(sig_float)} samples")

    # Тесты с разным ослаблением
    attenuations = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]
    results = []

    for att in attenuations:
        print(f"\n{'='*60}")
        print(f"[TEST] Ослабление: {att}x")
        print(f"{'='*60}")

        # Ослабляем
        attenuated = sig_float * att

        # Сохраняем
        max_val = np.max(np.abs(attenuated))
        if max_val > 1.0:
            attenuated = attenuated / max_val

        distorted_int16 = (attenuated * 32767).astype(np.int16)
        wavfile.write(wav_filename, fs_wav, distorted_int16)

        # Принимаем
        received_file = "rx_text.txt"
        if os.path.exists(received_file):
            os.remove(received_file)

        success = receive_from_file(wav_filename)

        if success and os.path.exists(received_file):
            with open(received_file, "r", encoding='utf-8') as f:
                received_text = f.read()

            if test_text == received_text:
                results.append((att, True))
                print(f"\n[RESULT] ✅ PASS: attenuation={att}")
            else:
                results.append((att, False))
                print(f"\n[RESULT] ❌ FAIL: attenuation={att} — текст отличается")
        else:
            results.append((att, False))
            print(f"\n[RESULT] ❌ FAIL: attenuation={att} — приём не удался")

    # Итог
    print(f"\n{'='*60}")
    print("ИТОГО:")
    print(f"{'='*60}")
    passed_count = sum(1 for _, p in results if p)
    failed_count = sum(1 for _, p in results if not p)
    print(f"  Пройдено: {passed_count}/{len(results)}")
    print(f"  Провалено: {failed_count}/{len(results)}")
    print()
    for att, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] attenuation={att}")
    print(f"{'='*60}")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
