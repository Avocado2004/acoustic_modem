"""
Тесты для новых искажений: затухание с расстоянием и случайная АЧХ.
"""

import numpy as np
import pytest
from channel_simulator import ChannelSimulator


class TestDistanceAttenuation:
    """Тесты для функции apply_distance_attenuation."""
    
    def test_zero_percent_distance(self):
        """При 0% расстояние 1 м, затухание минимальное."""
        fs = 48000
        sim = ChannelSimulator(fs=fs, seed=42)
        sig = np.ones(1000, dtype=np.float64)
        result = sim.apply_distance_attenuation(sig, percent=0.0)
        # При 0% расстояние 1 м, геометрическое затухание (1/1)^2 = 1
        # Сигнал должен остаться практически без изменений
        assert np.allclose(result, sig, rtol=1e-10), "При 0% сигнал не должен меняться"
    
    def test_hundred_percent_distance(self):
        """При 100% расстояние 5 м, затухание максимальное."""
        fs = 48000
        sim = ChannelSimulator(fs=fs, seed=42)
        sig = np.ones(1000, dtype=np.float64)
        result = sim.apply_distance_attenuation(sig, percent=100.0)
        # При 100% расстояние 5 м, геометрическое затухание (1/5)^2 = 0.04
        # Проверяем, что уровень сигнала уменьшился
        rms_original = np.sqrt(np.mean(sig**2))
        rms_result = np.sqrt(np.mean(result**2))
        assert rms_result < rms_original, "Уровень сигнала должен уменьшиться"
        # Проверяем, что затухание примерно соответствует расстоянию 5 м
        expected_attenuation = (1.0 / 5.0) ** 2  # геометрическое затухание
        actual_ratio = rms_result / rms_original
        assert abs(actual_ratio - expected_attenuation) < 0.1, \
            f"Ожидаемое затухание {expected_attenuation:.3f}, получено {actual_ratio:.3f}"
    
    def test_distance_level_reduction(self):
        """Проверка уменьшения уровня сигнала с расстоянием."""
        fs = 48000
        sim = ChannelSimulator(fs=fs, seed=42)
        sig = np.random.randn(2000).astype(np.float64)
        sig = sig / np.max(np.abs(sig))  # нормализуем
        
        # Тестируем разные проценты
        for percent, expected_distance in [(0.0, 1.0), (50.0, 3.0), (100.0, 5.0)]:
            result = sim.apply_distance_attenuation(sig, percent=percent)
            rms_original = np.sqrt(np.mean(sig**2))
            rms_result = np.sqrt(np.mean(result**2))
            expected_geom = (1.0 / expected_distance) ** 2
            actual_ratio = rms_result / rms_original
            print(f"[TEST] percent={percent}%, distance={expected_distance} m, "
                  f"geom_factor={expected_geom:.4f}, actual_ratio={actual_ratio:.4f}")
            # Проверяем, что уровень уменьшается с ростом расстояния
            if percent > 0:
                assert rms_result < rms_original, \
                    f"При {percent}% уровень должен уменьшиться"
    
    def test_frequency_dependent_attenuation(self):
        """Проверка частотно-зависимого затухания: высокие частоты затухают сильнее."""
        fs = 48000
        sim = ChannelSimulator(fs=fs, seed=42)
        # Создаем сигнал с двумя частотами: низкой и высокой
        t = np.arange(0, 1.0, 1.0/fs)
        low_freq = 500.0   # низкая частота
        high_freq = 4000.0  # высокая частота
        sig = (np.sin(2 * np.pi * low_freq * t) + 
                np.sin(2 * np.pi * high_freq * t)).astype(np.float64)
        
        result = sim.apply_distance_attenuation(sig, percent=100.0)
        
        # Анализируем спектр до и после
        freqs = np.fft.rfftfreq(len(sig), 1.0/fs)
        spec_orig = np.abs(np.fft.rfft(sig))
        spec_result = np.abs(np.fft.rfft(result))
        
        # Находим индексы для низкой и высокой частот
        low_idx = np.argmin(np.abs(freqs - low_freq))
        high_idx = np.argmin(np.abs(freqs - high_freq))
        
        low_attenuation = spec_result[low_idx] / spec_orig[low_idx]
        high_attenuation = spec_result[high_idx] / spec_orig[high_idx]
        
        print(f"[TEST] low_freq attenuation: {low_attenuation:.3f}, "
              f"high_freq attenuation: {high_attenuation:.3f}")
        # Высокие частоты должны затухать сильнее
        assert high_attenuation < low_attenuation, \
            "Высокие частоты должны затухать сильнее низких"


class TestRandomFrequencyResponse:
    """Тесты для функции apply_random_frequency_response."""
    
    def test_zero_percent_flat_response(self):
        """При 0% АЧХ должна быть ровной (коэффициент 1.0)."""
        fs = 48000
        sim = ChannelSimulator(fs=fs, seed=42)
        sig = np.ones(1000, dtype=np.float64)
        result = sim.apply_random_frequency_response(sig, percent=0.0)
        # При 0% сигнал не должен меняться
        assert np.allclose(result, sig, rtol=1e-10), "При 0% АЧХ должна быть ровной"
    
    def test_hundred_percent_random_response(self):
        """При 100% АЧХ случайная, но в пределах ±12 дБ."""
        fs = 48000
        sim = ChannelSimulator(fs=fs, seed=42)
        sig = np.random.randn(2000).astype(np.float64)
        
        result = sim.apply_random_frequency_response(sig, percent=100.0)
        
        # Проверяем, что АЧХ сохранилась для повторного вызова
        result2 = sim.apply_random_frequency_response(sig, percent=100.0)
        # Второй вызов должен использовать ту же АЧХ (постоянство в сеансе)
        assert np.allclose(result, result2), \
            "При повторном вызове должна использоваться та же АЧХ"
        
        # Проверяем границы АЧХ в дБ
        freqs = np.fft.rfftfreq(len(sig), 1.0/fs)
        spec_orig = np.fft.rfft(sig)
        spec_result = np.fft.rfft(result)
        # АЧХ = spec_result / spec_orig (отношение спектров)
        with np.errstate(divide='ignore', invalid='ignore'):
            freq_response_db = 20 * np.log10(np.abs(spec_result / (spec_orig + 1e-15)))
        
        # Проверяем, что в полосе 300-5000 Гц АЧХ в пределах ±12 дБ
        mask = (freqs >= 300.0) & (freqs <= 5000.0)
        if np.any(mask):
            max_db = np.max(freq_response_db[mask])
            min_db = np.min(freq_response_db[mask])
            print(f"[TEST] АЧХ диапазон в полосе: {min_db:.2f}..{max_db:.2f} дБ")
            assert max_db <= 12.0 + 0.1, f"АЧХ превышает +12 дБ: {max_db:.2f} дБ"
            assert min_db >= -12.0 - 0.1, f"АЧХ ниже -12 дБ: {min_db:.2f} дБ"
    
    def test_response_persists_for_session(self):
        """Проверка, что АЧХ постоянна в течение сеанса."""
        fs = 48000
        sim = ChannelSimulator(fs=fs, seed=42)
        sig1 = np.random.randn(1000).astype(np.float64)
        sig2 = np.random.randn(1000).astype(np.float64)
        
        # Первый вызов генерирует АЧХ
        result1 = sim.apply_random_frequency_response(sig1, percent=50.0)
        # Второй вызов должен использовать ту же АЧХ
        result2 = sim.apply_random_frequency_response(sig2, percent=50.0)
        
        # Проверяем, что АЧХ не сбрасывается
        assert sim._current_freq_response is not None, "АЧХ должна быть сохранена"
        
        # После сброса должна генерироваться новая АЧХ
        sim.reset_channel_state()
        assert sim._current_freq_response is None, "После сброса АЧХ должна быть None"
        result3 = sim.apply_random_frequency_response(sig1, percent=50.0)
        # result3 может отличаться от result1, так как АЧХ новая
        print("[TEST] АЧХ успешно сброшена и сгенерирована новая")
    
    def test_response_range_scaling(self):
        """Проверка масштабирования диапазона АЧХ от процента."""
        fs = 48000
        sig = np.random.randn(2000).astype(np.float64)
        
        # При 0% диапазон 0 дБ
        sim0 = ChannelSimulator(fs=fs, seed=42)
        sim0.apply_random_frequency_response(sig, percent=0.0)
        # При 50% диапазон до 6 дБ
        sim50 = ChannelSimulator(fs=fs, seed=42)
        result50 = sim50.apply_random_frequency_response(sig, percent=50.0)
        # При 100% диапазон до 12 дБ
        sim100 = ChannelSimulator(fs=fs, seed=42)
        result100 = sim100.apply_random_frequency_response(sig, percent=100.0)
        
        # Проверяем, что при 100% диапазон шире, чем при 50%
        freqs = np.fft.rfftfreq(len(sig), 1.0/fs)
        mask = (freqs >= 300.0) & (freqs <= 5000.0)
        
        spec_orig = np.fft.rfft(sig)
        spec_50 = np.fft.rfft(result50)
        spec_100 = np.fft.rfft(result100)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            resp_50_db = 20 * np.log10(np.abs(spec_50 / (spec_orig + 1e-15)))
            resp_100_db = 20 * np.log10(np.abs(spec_100 / (spec_orig + 1e-15)))
        
        if np.any(mask):
            range_50 = np.max(resp_50_db[mask]) - np.min(resp_50_db[mask])
            range_100 = np.max(resp_100_db[mask]) - np.min(resp_100_db[mask])
            print(f"[TEST] Диапазон АЧХ: 50%={range_50:.2f} дБ, 100%={range_100:.2f} дБ")
            assert range_100 > range_50, \
                "При 100% диапазон АЧХ должен быть шире, чем при 50%"


class TestIntegrationWithModemCLI:
    """Интеграционные тесты с modem_cli.py."""
    
    def test_apply_channel_distortions_includes_new_effects(self):
        """Проверка, что apply_channel_distortions вызывает новые методы."""
        from modem_cli import apply_channel_distortions
        from unittest.mock import patch, MagicMock
        import numpy as np
        
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        
        # Патчим методы ChannelSimulator
        with patch.object(ChannelSimulator, 'apply_random_frequency_response') as mock_freq, \
             patch.object(ChannelSimulator, 'apply_distance_attenuation') as mock_dist:
            
            mock_freq.return_value = sig.copy()
            mock_dist.return_value = sig.copy()
            
            # Вызываем с процентом > 0
            result = apply_channel_distortions(sig, fs, 50.0)
            
            # Проверяем, что новые методы были вызваны
            assert mock_freq.called, "apply_random_frequency_response не был вызван"
            assert mock_dist.called, "apply_distance_attenuation не был вызван"
            
            # Проверяем, что параметр percent передается
            call_args_freq = mock_freq.call_args
            # Проверяем, что percent передан (должен быть в kwargs или args)
            if call_args_freq[1] and 'percent' in call_args_freq[1]:
                assert call_args_freq[1]['percent'] == 50.0, "percent не передан в apply_random_frequency_response"
            elif call_args_freq[0]:
                # percent должен быть вторым аргументом
                assert call_args_freq[0][1] == 50.0, "percent не передан в apply_random_frequency_response"
            
            print("[TEST] Новые искажения успешно интегрированы в apply_channel_distortions")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
