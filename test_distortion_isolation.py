#!/usr/bin/env python3
"""
Изолированный тест каждого типа искажения.

Применяет только ОДНО искажение к сигналу и проверяет,
проходит ли демодуляция. Это позволяет найти проблемное искажение.

Запуск: python test_distortion_isolation.py
"""

import sys
import os
import random
import string
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def prepare_signal(data_size=128, seed=42):
    """
    Подготовка сигнала: генерация текста, передача, загрузка WAV.

    Возвращает (sig_float, fs_wav, test_text) или (None, None, None) при ошибке.
    """
    import modem_config
    from modem_config import set_modulation, init_phases, fs, wavfile
    from modem_tx import transmit_text

    set_modulation("QPSK")
    init_phases()

    random.seed(seed)
    test_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(data_size))

    transmit_text(test_text)

    wav_filename = "ofdm_acoustic_tx_with_noise.wav"
    if not os.path.exists(wav_filename):
        print("[ERR] WAV-файл не найден")
        return None, None, None

    fs_wav, sig = wavfile.read(wav_filename)

    if np.issubdtype(sig.dtype, np.integer):
        sig_float = sig.astype(np.float64) / np.iinfo(sig.dtype).max
    else:
        sig_float = sig.astype(np.float64)

    return sig_float, fs_wav, test_text


def apply_and_save(sig_float, fs_wav, distorted):
    """Сохранение искажённого сигнала в WAV-файл."""
    from modem_config import wavfile

    max_val = np.max(np.abs(distorted))
    if max_val > 1.0:
        distorted = distorted / max_val

    distorted_int16 = (distorted * 32767).astype(np.int16)
    wavfile.write("ofdm_acoustic_tx_with_noise.wav", fs_wav, distorted_int16)


def receive_and_check(test_text):
    """Приём и проверка текста. Возвращает True если совпадает."""
    from signal_processor import receive_from_file

    received_file = "rx_text.txt"
    if os.path.exists(received_file):
        os.remove(received_file)

    success = receive_from_file("ofdm_acoustic_tx_with_noise.wav")

    if not success:
        return False

    if os.path.exists(received_file):
        with open(received_file, "r", encoding='utf-8') as f:
            received_text = f.read()
        return test_text == received_text

    return False


def main():
    from channel_simulator import ChannelSimulator

    print("=" * 60)
    print("Изолированный тест искажений")
    print("=" * 60)

    # Подготавливаем сигнал один раз
    sig_float, fs_wav, test_text = prepare_signal(data_size=128, seed=42)
    if sig_float is None:
        print("[ERR] Не удалось подготовить сигнал")
        sys.exit(1)

    print(f"[INFO] Сигнал подготовлен: {len(sig_float)} samples, RMS={np.sqrt(np.mean(sig_float**2)):.6f}")
    print(f"[INFO] Текст: {test_text[:30]}...")

    # Определяем тесты — каждое искажение отдельно
    tests = [
        # 1. Без искажений (baseline)
        ("Без искажений", lambda s, f: s.copy()),

        # 2. Только AWGN (SNR=20dB)
        ("AWGN SNR=20dB", lambda s, f: ChannelSimulator(fs=f, seed=42).add_awgn(s.copy(), 20.0)),

        # 3. Только AWGN (SNR=15dB)
        ("AWGN SNR=15dB", lambda s, f: ChannelSimulator(fs=f, seed=42).add_awgn(s.copy(), 15.0)),

        # 4. Только AWGN (SNR=10dB)
        ("AWGN SNR=10dB", lambda s, f: ChannelSimulator(fs=f, seed=42).add_awgn(s.copy(), 10.0)),

        # 5. Только multipath (2 луча: 0ms + 2ms)
        ("Multipath [0,2]ms gains=[1.0,0.3]",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_multipath(
             s.copy(), delays_ms=[0.0, 2.0], gains=[1.0, 0.3])),

        # 6. Только multipath (3 луча)
        ("Multipath [0,2,5]ms gains=[0.6,0.3,0.1]",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_multipath(
             s.copy(), delays_ms=[0.0, 2.0, 5.0], gains=[0.6, 0.3, 0.1])),

        # 7. Только frequency offset (2 Hz)
        ("Freq offset 2Hz",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_frequency_offset(s.copy(), 2.0)),

        # 8. Только frequency offset (5 Hz)
        ("Freq offset 5Hz",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_frequency_offset(s.copy(), 5.0)),

        # 9. Только random frequency response (30%)
        ("Random FR 30%",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_random_frequency_response(
             s.copy(), percent=30.0)),

        # 10. Только random frequency response (70%)
        ("Random FR 70%",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_random_frequency_response(
             s.copy(), percent=70.0)),

        # 11. Только random frequency response (100%)
        ("Random FR 100%",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_random_frequency_response(
             s.copy(), percent=100.0)),

        # 12. Только distance attenuation (30%)
        ("Distance attenuation 30%",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_distance_attenuation(
             s.copy(), percent=30.0)),

        # 13. Только distance attenuation (70%)
        ("Distance attenuation 70%",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_distance_attenuation(
             s.copy(), percent=70.0)),

        # 14. Только distance attenuation (100%)
        ("Distance attenuation 100%",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_distance_attenuation(
             s.copy(), percent=100.0)),

        # 15. Только soft clipping (threshold=0.8)
        ("Soft clipping tanh 0.8",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_soft_clipping(
             s.copy(), threshold=0.8, clip_type='tanh')),

        # 16. Только hard clipping (threshold=0.8)
        ("Hard clipping 0.8",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_hard_clipping(
             s.copy(), threshold=0.8)),

        # 17. Только phase jitter (0.1 rad)
        ("Phase jitter 0.1 rad",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_phase_jitter(
             s.copy(), jitter_std_rad=0.1)),

        # 18. Только phase jitter (0.3 rad)
        ("Phase jitter 0.3 rad",
         lambda s, f: ChannelSimulator(fs=f, seed=42).apply_phase_jitter(
             s.copy(), jitter_std_rad=0.3)),
    ]

    results = []
    for name, func in tests:
        try:
            distorted = func(sig_float, fs_wav)
            apply_and_save(sig_float, fs_wav, distorted)
            passed = receive_and_check(test_text)
            results.append((name, passed))
            status = "PASS" if passed else "FAIL"
            print(f"\n[RESULT] {status}: {name}")
        except Exception as e:
            results.append((name, False))
            print(f"\n[RESULT] ERROR: {name} — {e}")
            import traceback
            traceback.print_exc()

    # Итог
    print(f"\n{'='*60}")
    print("ИТОГО:")
    print(f"{'='*60}")
    passed_count = sum(1 for _, p in results if p)
    failed_count = sum(1 for _, p in results if not p)
    print(f"  Пройдено: {passed_count}/{len(results)}")
    print(f"  Провалено: {failed_count}/{len(results)}")
    print()
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print(f"{'='*60}")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
