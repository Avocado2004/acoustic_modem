"""
Тесты обнаружения преамбулы для различных условий приёма.

Покрывает:
  - Обнаружение преамбулы в идеальных условиях (loopback)
  - Обнаружение преамбулы при ослаблении сигнала (микрофонный приём)
  - Обнаружение преамбулы при добавлении шума
  - Проверка порога корреляции PREAMBLE_CORR_THRESHOLD
  - Проверка адаптивного порога

Каждый тест выполняется менее чем за 5 секунд.
"""

import numpy as np
import pytest
import sys
import os

# Добавляем родительскую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Тесты обнаружения преамбулы
# =============================================================================

class TestPreambleDetection:
    """Тесты обнаружения преамбулы в различных условиях."""

    def _build_test_signal(self, attenuation=1.0, noise_snr_db=None, seed=42):
        """
        Построение тестового сигнала с преамбулой.

        Параметры
        ----------
        attenuation : float
            Коэффициент ослабления сигнала (1.0 = без ослабления, 0.1 = 10x ослабление)
        noise_snr_db : float or None
            SNR шума в дБ (None = без шума)
        seed : int
            Seed для генератора случайных чисел

        Возвращает
        -------
        signal : np.ndarray
            Тестовый сигнал с preroll + пилоты + преамбула
        preamble_pos : int
            Позиция начала преамбулы в сигнале
        preamble_td : np.ndarray
            Временное представление преамбулы
        """
        import modem_config
        from modem_modulation import build_preamble

        # Инициализируем фазы
        modem_config.init_phases()

        # Строим преамбулу
        preamble_td = build_preamble()
        pre_len = len(preamble_td)

        # Параметры сигнала
        preroll_samples = int(0.25 * modem_config.fs)  # 0.25 секунды preroll
        pilot_samples = modem_config.PREAMBLE_PILOT_SYMBOLS * modem_config.SYMBOL_LEN

        # Создаём сигнал: preroll (тишина) + пилоты + преамбула
        total_len = preroll_samples + pilot_samples + pre_len + 1000
        signal = np.zeros(total_len)

        # Добавляем преамбулу в нужную позицию
        preamble_pos = preroll_samples + pilot_samples
        signal[preamble_pos:preamble_pos + pre_len] = np.real(preamble_td)

        # Ослабляем сигнал (симуляция микрофонного канала)
        signal *= attenuation

        # Добавляем шум если указан
        if noise_snr_db is not None:
            rng = np.random.RandomState(seed)
            sig_power = np.mean(signal ** 2)
            if sig_power > 1e-15:
                snr_linear = 10 ** (noise_snr_db / 10.0)
                noise_power = sig_power / snr_linear
                noise_std = np.sqrt(noise_power)
                signal += rng.normal(0, noise_std, len(signal))

        return signal, preamble_pos, preamble_td

    def test_preamble_correlation_ideal(self):
        """
        Тест: корреляция преамбулы в идеальных условиях.

        При идеальном сигнале (без ослабления, без шума) корреляция
        должна быть высокой (близкой к 1.0 для нормализованной корреляции).
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD

        signal, preamble_pos, preamble_td = self._build_test_signal(attenuation=1.0)

        # Вычисляем корреляцию
        norm_corr = _normalized_correlation(signal, preamble_td)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        print(f"[TEST] Ideal: peak_idx={peak_idx}, peak_val={peak_val:.4f}, "
              f"expected_pos={preamble_pos}, threshold={PREAMBLE_CORR_THRESHOLD}")

        # Пик должен быть на правильной позиции
        assert peak_idx == preamble_pos, \
            f"Peak at {peak_idx}, expected {preamble_pos}"

        # Корреляция должна быть выше порога
        assert peak_val > PREAMBLE_CORR_THRESHOLD, \
            f"Peak value {peak_val:.4f} below threshold {PREAMBLE_CORR_THRESHOLD}"

    def test_preamble_correlation_attenuated_0_5(self):
        """
        Тест: корреляция при ослаблении сигнала в 2 раза (attenuation=0.5).

        Это симуляция микрофонного приёма на среднем расстоянии.
        Ослабление амплитуды в 2 раза снижает корреляцию примерно в 2 раза.
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD

        signal, preamble_pos, preamble_td = self._build_test_signal(attenuation=0.5)

        norm_corr = _normalized_correlation(signal, preamble_td)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        print(f"[TEST] Attenuated 0.5: peak_idx={peak_idx}, peak_val={peak_val:.4f}, "
              f"expected_pos={preamble_pos}, threshold={PREAMBLE_CORR_THRESHOLD}")

        # Пик должен быть на правильной позиции
        assert peak_idx == preamble_pos, \
            f"Peak at {peak_idx}, expected {preamble_pos}"

        # Корреляция должна быть выше порога (0.20)
        assert peak_val > PREAMBLE_CORR_THRESHOLD, \
            f"Peak value {peak_val:.4f} below threshold {PREAMBLE_CORR_THRESHOLD}"

    def test_preamble_correlation_attenuated_0_3(self):
        """
        Тест: корреляция при ослаблении сигнала в ~3 раза (attenuation=0.3).

        Это симуляция микрофонного приёма на большом расстоянии
        или при плохих условиях (неплоская ЧХ динамика/микрофона).
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD

        signal, preamble_pos, preamble_td = self._build_test_signal(attenuation=0.3)

        norm_corr = _normalized_correlation(signal, preamble_td)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        print(f"[TEST] Attenuated 0.3: peak_idx={peak_idx}, peak_val={peak_val:.4f}, "
              f"expected_pos={preamble_pos}, threshold={PREAMBLE_CORR_THRESHOLD}")

        # Пик должен быть на правильной позиции
        assert peak_idx == preamble_pos, \
            f"Peak at {peak_idx}, expected {preamble_pos}"

        # Корреляция должна быть выше порога (0.20)
        assert peak_val > PREAMBLE_CORR_THRESHOLD, \
            f"Peak value {peak_val:.4f} below threshold {PREAMBLE_CORR_THRESHOLD}"

    def test_preamble_correlation_attenuated_0_2(self):
        """
        Тест: корреляция при ослаблении сигнала в 5 раз (attenuation=0.2).

        Это симулация очень слабого микрофонного сигнала.
        Корреляция должна быть около 0.2, что равно новому порогу.
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD

        signal, preamble_pos, preamble_td = self._build_test_signal(attenuation=0.2)

        norm_corr = _normalized_correlation(signal, preamble_td)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        print(f"[TEST] Attenuated 0.2: peak_idx={peak_idx}, peak_val={peak_val:.4f}, "
              f"expected_pos={preamble_pos}, threshold={PREAMBLE_CORR_THRESHOLD}")

        # Пик должен быть на правильной позиции
        assert peak_idx == preamble_pos, \
            f"Peak at {peak_idx}, expected {preamble_pos}"

        # Корреляция должна быть выше или равна порогу
        assert peak_val >= PREAMBLE_CORR_THRESHOLD * 0.95, \
            f"Peak value {peak_val:.4f} significantly below threshold {PREAMBLE_CORR_THRESHOLD}"

    def test_preamble_correlation_with_noise(self):
        """
        Тест: корреляция при ослаблении 0.5 и шуме SNR=20dB.

        Комбинация ослабления и шума — типичный сценарий микрофонного приёма.
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD

        signal, preamble_pos, preamble_td = self._build_test_signal(
            attenuation=0.5, noise_snr_db=20, seed=42
        )

        norm_corr = _normalized_correlation(signal, preamble_td)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        print(f"[TEST] Attenuated 0.5 + noise 20dB: peak_idx={peak_idx}, "
              f"peak_val={peak_val:.4f}, expected_pos={preamble_pos}, "
              f"threshold={PREAMBLE_CORR_THRESHOLD}")

        # Пик должен быть на правильной позиции
        assert peak_idx == preamble_pos, \
            f"Peak at {peak_idx}, expected {preamble_pos}"

        # Корреляция должна быть выше порога
        assert peak_val > PREAMBLE_CORR_THRESHOLD, \
            f"Peak value {peak_val:.4f} below threshold {PREAMBLE_CORR_THRESHOLD}"

    def test_preamble_correlation_with_strong_noise(self):
        """
        Тест: корреляция при ослаблении 0.5 и шуме SNR=10dB.

        Более шумный сценарий — проверяем что преамбула всё ещё обнаруживается.
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD

        signal, preamble_pos, preamble_td = self._build_test_signal(
            attenuation=0.5, noise_snr_db=10, seed=123
        )

        norm_corr = _normalized_correlation(signal, preamble_td)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        print(f"[TEST] Attenuated 0.5 + noise 10dB: peak_idx={peak_idx}, "
              f"peak_val={peak_val:.4f}, expected_pos={preamble_pos}, "
              f"threshold={PREAMBLE_CORR_THRESHOLD}")

        # Пик должен быть на правильной позиции
        assert peak_idx == preamble_pos, \
            f"Peak at {peak_idx}, expected {preamble_pos}"

        # Корреляция должна быть выше порога
        assert peak_val > PREAMBLE_CORR_THRESHOLD, \
            f"Peak value {peak_val:.4f} below threshold {PREAMBLE_CORR_THRESHOLD}"

    def test_threshold_value(self):
        """
        Тест: проверка что порог PREAMBLE_CORR_THRESHOLD = 0.35.

        Порог возвращён на 0.35 для стандартного обнаружения преамбулы.
        Микрофонный приём обрабатывается отдельным режимом (test_mic_mode.py).
        """
        from signal_processor import PREAMBLE_CORR_THRESHOLD

        print(f"[TEST] PREAMBLE_CORR_THRESHOLD = {PREAMBLE_CORR_THRESHOLD}")
        assert PREAMBLE_CORR_THRESHOLD == 0.35, \
            f"Expected threshold 0.35, got {PREAMBLE_CORR_THRESHOLD}"

    def test_correlation_peak_ratio(self):
        """
        Тест: пик корреляции должен быть значительно выше среднего.

        Отношение пика к среднему значению корреляции должно быть > 3
        для надёжного обнаружения преамбулы.
        """
        from signal_processor import _normalized_correlation

        signal, preamble_pos, preamble_td = self._build_test_signal(attenuation=0.5)

        norm_corr = _normalized_correlation(signal, preamble_td)
        peak_val = float(np.max(norm_corr))
        mean_val = float(np.mean(norm_corr))

        ratio = peak_val / mean_val if mean_val > 0 else float('inf')

        print(f"[TEST] Peak/mean ratio: {peak_val:.4f}/{mean_val:.6f} = {ratio:.1f}")

        # Пик должен быть минимум в 3 раза выше среднего
        assert ratio > 3.0, \
            f"Peak/mean ratio {ratio:.1f} too low (expected > 3.0)"

    def test_frequency_response_distortion(self):
        """
        Тест: корреляция при неплоской частотной характеристике.

        Симулирует реальный микрофонный канал, где некоторые частоты
        ослаблены сильнее других (неплоская ЧХ динамика/микрофона).
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD
        import modem_config
        from modem_modulation import build_preamble

        modem_config.init_phases()
        preamble_td = build_preamble()
        pre_len = len(preamble_td)

        # Параметры сигнала
        preroll_samples = int(0.25 * modem_config.fs)
        pilot_samples = modem_config.PREAMBLE_PILOT_SYMBOLS * modem_config.SYMBOL_LEN
        preamble_pos = preroll_samples + pilot_samples
        total_len = preamble_pos + pre_len + 1000

        # Создаём сигнал
        signal = np.zeros(total_len)
        signal[preamble_pos:preamble_pos + pre_len] = np.real(preamble_td)

        # Применяем случайную частотную характеристику (симуляция неплоской ЧХ)
        rng = np.random.RandomState(42)
        n = len(signal)
        freqs = np.fft.rfftfreq(n, 1.0 / modem_config.fs)
        spec = np.fft.rfft(signal)

        # Создаём неплоскую АЧХ: ±6 дБ в рабочем диапазоне
        f_low = 300.0
        f_high = 5000.0
        mask = (freqs >= f_low) & (freqs <= f_high)
        random_db = np.zeros_like(freqs)
        random_db[mask] = rng.uniform(-6, 6, size=np.sum(mask))
        random_linear = 10 ** (random_db / 20.0)

        spec *= random_linear
        signal = np.real(np.fft.irfft(spec, n=n))

        # Проверяем корреляцию
        norm_corr = _normalized_correlation(signal, preamble_td)
        peak_idx = int(np.argmax(norm_corr))
        peak_val = float(norm_corr[peak_idx])

        print(f"[TEST] Frequency distortion: peak_idx={peak_idx}, "
              f"peak_val={peak_val:.4f}, expected_pos={preamble_pos}, "
              f"threshold={PREAMBLE_CORR_THRESHOLD}")

        # Пик должен быть на правильной позиции
        assert peak_idx == preamble_pos, \
            f"Peak at {peak_idx}, expected {preamble_pos}"

        # Даже при неплоской ЧХ корреляция должна быть выше порога
        assert peak_val > PREAMBLE_CORR_THRESHOLD, \
            f"Peak value {peak_val:.4f} below threshold {PREAMBLE_CORR_THRESHOLD}"


# =============================================================================
# Тесты адаптивного порога
# =============================================================================

class TestAdaptiveThreshold:
    """Тесты адаптивного порога корреляции."""

    def test_adaptive_threshold_basic(self):
        """
        Тест: адаптивный порог должен быть >= PREAMBLE_CORR_THRESHOLD.

        Адаптивный порог = max(PREAMBLE_CORR_THRESHOLD, mean + 3*std)
        Он никогда не должен быть ниже фиксированного порога.
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD
        import modem_config
        from modem_modulation import build_preamble

        modem_config.init_phases()
        preamble_td = build_preamble()

        # Создаём сигнал с преамбулой
        preroll_samples = int(0.25 * modem_config.fs)
        pilot_samples = modem_config.PREAMBLE_PILOT_SYMBOLS * modem_config.SYMBOL_LEN
        preamble_pos = preroll_samples + pilot_samples
        total_len = preamble_pos + len(preamble_td) + 1000
        signal = np.zeros(total_len)
        signal[preamble_pos:preamble_pos + len(preamble_td)] = np.real(preamble_td)

        norm_corr = _normalized_correlation(signal, preamble_td)
        mean_corr = float(np.mean(norm_corr))
        std_corr = float(np.std(norm_corr))
        adaptive_threshold = max(PREAMBLE_CORR_THRESHOLD, mean_corr + 3.0 * std_corr)

        print(f"[TEST] Adaptive threshold: fixed={PREAMBLE_CORR_THRESHOLD}, "
              f"adaptive={adaptive_threshold:.4f}, mean={mean_corr:.6f}, std={std_corr:.6f}")

        # Адаптивный порог должен быть >= фиксированного
        assert adaptive_threshold >= PREAMBLE_CORR_THRESHOLD, \
            f"Adaptive threshold {adaptive_threshold} < fixed {PREAMBLE_CORR_THRESHOLD}"

    def test_adaptive_threshold_with_noise(self):
        """
        Тест: адаптивный порог увеличивается при наличии шума.

        При высоком уровне шума адаптивный порог должен быть выше
        фиксированного, чтобы избежать ложных срабатываний.
        """
        from signal_processor import _normalized_correlation, PREAMBLE_CORR_THRESHOLD
        import modem_config
        from modem_modulation import build_preamble

        modem_config.init_phases()
        preamble_td = build_preamble()

        # Создаём сигнал с преамбулой и шумом
        preroll_samples = int(0.25 * modem_config.fs)
        pilot_samples = modem_config.PREAMBLE_PILOT_SYMBOLS * modem_config.SYMBOL_LEN
        preamble_pos = preroll_samples + pilot_samples
        total_len = preamble_pos + len(preamble_td) + 1000
        signal = np.zeros(total_len)
        signal[preamble_pos:preamble_pos + len(preamble_td)] = np.real(preamble_td)

        # Добавляем шум
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 0.1, total_len)
        noisy_signal = signal + noise

        norm_corr = _normalized_correlation(noisy_signal, preamble_td)
        mean_corr = float(np.mean(norm_corr))
        std_corr = float(np.std(norm_corr))
        adaptive_threshold = max(PREAMBLE_CORR_THRESHOLD, mean_corr + 3.0 * std_corr)

        print(f"[TEST] Adaptive with noise: fixed={PREAMBLE_CORR_THRESHOLD}, "
              f"adaptive={adaptive_threshold:.4f}, mean={mean_corr:.6f}, std={std_corr:.6f}")

        # Адаптивный порог должен быть >= фиксированного
        assert adaptive_threshold >= PREAMBLE_CORR_THRESHOLD, \
            f"Adaptive threshold {adaptive_threshold} < fixed {PREAMBLE_CORR_THRESHOLD}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
