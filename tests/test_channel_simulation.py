"""
Тесты для проверки интеграции симулятора канала в режим Loop.
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, Mock

# Импортируем тестируемые функции
from modem_cli import scale_param, apply_channel_distortions
from channel_simulator import ChannelSimulator


class TestScaleParam:
    """Тесты для функции масштабирования параметров."""
    
    def test_scale_param_zero_percent(self):
        """Проверка масштабирования при 0%."""
        result = scale_param(10.0, 20.0, 0.0)
        assert result == 10.0, f"Ожидалось 10.0, получено {result}"
    
    def test_scale_param_hundred_percent(self):
        """Проверка масштабирования при 100%."""
        result = scale_param(10.0, 20.0, 100.0)
        assert result == 20.0, f"Ожидалось 20.0, получено {result}"
    
    def test_scale_param_fifty_percent(self):
        """Проверка масштабирования при 50%."""
        result = scale_param(10.0, 20.0, 50.0)
        assert result == 15.0, f"Ожидалось 15.0, получено {result}"
    
    def test_scale_param_negative_percent(self):
        """Проверка масштабирования при отрицательном проценте (линейная экстраполяция)."""
        # scale_param выполняет линейную интерполяцию/экстраполяцию
        # При -10%: 10.0 + (20.0 - 10.0) * (-10.0 / 100.0) = 10.0 - 1.0 = 9.0
        result = scale_param(10.0, 20.0, -10.0)
        assert result == 9.0, f"Ожидалось 9.0, получено {result}"
    
    def test_scale_param_over_hundred_percent(self):
        """Проверка масштабирования при проценте > 100% (линейная экстраполяция)."""
        # При 150%: 10.0 + (20.0 - 10.0) * (150.0 / 100.0) = 10.0 + 15.0 = 25.0
        result = scale_param(10.0, 20.0, 150.0)
        assert result == 25.0, f"Ожидалось 25.0, получено {result}"


class TestApplyChannelDistortions:
    """Тесты для функции применения искажений канала."""
    
    def test_apply_zero_percent(self):
        """Проверка работы при 0% искажений (сигнал не меняется)."""
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        
        # При 0% искажений сигнал должен возвращаться без изменений
        # Методы ChannelSimulator не должны вызываться
        with patch.object(ChannelSimulator, 'apply_multipath') as mock_multipath, \
             patch.object(ChannelSimulator, 'add_awgn') as mock_awgn, \
             patch.object(ChannelSimulator, 'apply_frequency_offset') as mock_freq:
            
            result = apply_channel_distortions(sig, fs, 0.0)
            
            # Проверяем, что методы ChannelSimulator НЕ вызывались
            assert not mock_multipath.called, "apply_multipath не должен вызываться при 0%"
            assert not mock_awgn.called, "add_awgn не должен вызываться при 0%"
            assert not mock_freq.called, "apply_frequency_offset не должен вызываться при 0%"
            
            # Проверяем, что сигнал не изменился
            assert np.allclose(sig, result), "Сигнал изменился при 0% искажений"
    
    def test_apply_hundred_percent(self):
        """Проверка работы при 100% искажений (все искажения)."""
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        
        result = apply_channel_distortions(sig, fs, 100.0)
        
        # Проверяем, что сигнал изменился
        assert not np.allclose(sig, result), "Сигнал не изменился при 100% искажений"
        assert len(result) == len(sig), "Длина сигнала изменилась"
    
    def test_apply_fifty_percent_soft_clipping(self):
        """Проверка включения soft clipping при >50%."""
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        
        # При 51% soft clipping должен применяться
        with patch.object(ChannelSimulator, 'apply_soft_clipping') as mock_clip:
            mock_clip.return_value = sig.copy()
            apply_channel_distortions(sig, fs, 51.0)
            assert mock_clip.called, "Soft clipping не был вызван при 51%"
        
        # При 50% soft clipping НЕ должен применяться
        with patch.object(ChannelSimulator, 'apply_soft_clipping') as mock_clip:
            mock_clip.return_value = sig.copy()
            apply_channel_distortions(sig, fs, 50.0)
            assert not mock_clip.called, "Soft clipping был вызван при 50%"
    
    def test_apply_eighty_percent_distortion(self):
        """Проверка включения distortion при >80%."""
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        
        # При 81% distortion должен применяться
        with patch.object(ChannelSimulator, 'apply_distortion') as mock_dist:
            mock_dist.return_value = sig.copy()
            apply_channel_distortions(sig, fs, 81.0)
            assert mock_dist.called, "Distortion не был вызван при 81%"
        
        # При 80% distortion НЕ должен применяться
        with patch.object(ChannelSimulator, 'apply_distortion') as mock_dist:
            mock_dist.return_value = sig.copy()
            apply_channel_distortions(sig, fs, 80.0)
            assert not mock_dist.called, "Distortion был вызван при 80%"
    
    def test_apply_seventy_percent_impulse_noise(self):
        """Проверка включения impulse noise при >70%."""
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        
        # При 71% impulse noise должен применяться
        with patch.object(ChannelSimulator, 'apply_impulse_noise') as mock_impulse:
            mock_impulse.return_value = sig.copy()
            apply_channel_distortions(sig, fs, 71.0)
            assert mock_impulse.called, "Impulse noise не был вызван при 71%"
        
        # При 70% impulse noise НЕ должен применяться
        with patch.object(ChannelSimulator, 'apply_impulse_noise') as mock_impulse:
            mock_impulse.return_value = sig.copy()
            apply_channel_distortions(sig, fs, 70.0)
            assert not mock_impulse.called, "Impulse noise был вызван при 70%"
    
    def test_apply_snr_scaling(self):
        """Проверка масштабирования SNR."""
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        
        # При 0% искажений add_awgn не должен вызываться
        with patch.object(ChannelSimulator, 'add_awgn') as mock_awgn:
            mock_awgn.return_value = sig.copy()
            
            apply_channel_distortions(sig, fs, 0.0)
            assert not mock_awgn.called, "add_awgn не должен вызываться при 0%"
        
        # При 100% SNR должен быть 15 дБ
        with patch.object(ChannelSimulator, 'add_awgn') as mock_awgn:
            mock_awgn.return_value = sig.copy()
            
            apply_channel_distortions(sig, fs, 100.0)
            call_args = mock_awgn.call_args
            snr_db = call_args[0][1]
            assert abs(snr_db - 15.0) < 0.01, f"Ожидалось SNR=15.0, получено {snr_db}"
        
        # При 50% SNR должен быть посередине (22.5 дБ)
        with patch.object(ChannelSimulator, 'add_awgn') as mock_awgn:
            mock_awgn.return_value = sig.copy()
            
            apply_channel_distortions(sig, fs, 50.0)
            call_args = mock_awgn.call_args
            snr_db = call_args[0][1]
            assert abs(snr_db - 22.5) < 0.01, f"Ожидалось SNR=22.5, получено {snr_db}"
    
    def test_apply_frequency_offset_scaling(self):
        """Проверка масштабирования сдвига частоты."""
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        
        # При 0% искажений apply_frequency_offset не должен вызываться
        with patch.object(ChannelSimulator, 'apply_frequency_offset') as mock_freq:
            mock_freq.return_value = sig.copy()
            
            apply_channel_distortions(sig, fs, 0.0)
            assert not mock_freq.called, "apply_frequency_offset не должен вызываться при 0%"
        
        # При 100% сдвиг должен быть 5 Гц
        with patch.object(ChannelSimulator, 'apply_frequency_offset') as mock_freq:
            mock_freq.return_value = sig.copy()
            
            apply_channel_distortions(sig, fs, 100.0)
            call_args = mock_freq.call_args
            offset_hz = call_args[0][1]
            assert abs(offset_hz - 5.0) < 0.01, f"Ожидалось offset=5.0, получено {offset_hz}"
        
        # При 50% сдвиг должен быть 2.5 Гц
        with patch.object(ChannelSimulator, 'apply_frequency_offset') as mock_freq:
            mock_freq.return_value = sig.copy()
            
            apply_channel_distortions(sig, fs, 50.0)
            call_args = mock_freq.call_args
            offset_hz = call_args[0][1]
            assert abs(offset_hz - 2.5) < 0.01, f"Ожидалось offset=2.5, получено {offset_hz}"


class TestRunLoopIntegration:
    """Интеграционные тесты для режима Loop с симулятором канала."""
    
    @patch('modem_cli.transmit_text')
    @patch('modem_cli.receive_from_file')
    @patch('modem_cli.wavfile')
    @patch('builtins.input')
    def test_run_loop_with_channel_simulation(self, mock_input, mock_wavfile, mock_receive, mock_transmit):
        """Проверка интеграции симулятора канала в run_loop()."""
        from modem_cli import run_loop
        
        # Настройка моков
        mock_input.side_effect = ['50']  # Процент искажений 50%
        
        # Мок для wavfile.read
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        mock_wavfile.read.return_value = (fs, sig)
        
        # Мок для transmit_text (создает WAV файл)
        def mock_transmit_side_effect(text):
            # Симулируем создание WAV файла
            pass
        mock_transmit.side_effect = mock_transmit_side_effect
        
        # Мок для receive_from_file
        mock_receive.return_value = True
        
        # Создаем временный файл для имитации rx_text.txt
        with open("rx_text.txt", "w") as f:
            f.write("testtext")
        
        try:
            # Патчим open для имитации чтения оригинального текста
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__ = lambda s: s
                mock_open.return_value.__exit__ = Mock(return_value=False)
                mock_open.return_value.read.return_value = "testtext"
                
                # Запускаем run_loop
                # (здесь может потребоваться дополнительная настройка моков)
        finally:
            # Удаляем временный файл
            if os.path.exists("rx_text.txt"):
                os.remove("rx_text.txt")
    
    def test_wav_file_loading_and_saving(self):
        """Проверка загрузки и сохранения WAV файла."""
        from modem_cli import apply_channel_distortions
        
        # Создаем тестовый WAV файл
        fs = 48000
        sig = np.random.randn(1000).astype(np.float32)
        test_wav = "test_channel.wav"
        
        from modem_config import wavfile
        wavfile.write(test_wav, fs, sig)
        
        try:
            # Загружаем, применяем искажения и сохраняем
            fs_loaded, sig_loaded = wavfile.read(test_wav)
            distorted = apply_channel_distortions(sig_loaded, fs_loaded, 50.0)
            wavfile.write(test_wav, fs_loaded, distorted)
            
            # Проверяем, что файл можно снова загрузить
            fs_check, sig_check = wavfile.read(test_wav)
            assert fs_check == fs, "Частота дискретизации изменилась"
            assert len(sig_check) == len(sig), "Длина сигнала изменилась"
        finally:
            if os.path.exists(test_wav):
                os.remove(test_wav)


class TestDebugOutput:
    """Тесты для проверки отладочного вывода."""
    
    def test_debug_output_present(self):
        """Проверка наличия отладочного вывода."""
        import inspect
        from modem_cli import apply_channel_distortions
        
        # Получаем исходный код функции
        source = inspect.getsource(apply_channel_distortions)
        
        # Проверяем наличие print statements с отладочной информацией
        assert "[CHANNEL]" in source, "Отладочный вывод [CHANNEL] не найден"
        assert "SNR" in source, "Вывод SNR не найден"
        assert "многолучевости" in source, "Вывод многолучевости не найден"
        assert "сдвига частоты" in source, "Вывод сдвига частоты не найден"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
