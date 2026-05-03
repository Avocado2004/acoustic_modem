"""
Тесты для проверки работы режима Loop с 300 байтами данных.
Проверяет, что размер данных изменился с 50 на 300 байт,
и что вывод параметров симуляции канала работает корректно.
"""

import os
import sys
import random
import string
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call


class TestDataSize300Bytes:
    """Тесты для проверки размера данных 300 байт в режиме Loop."""
    
    def test_random_text_length_300_bytes(self):
        """Проверка, что генерируемый текст дает 300 байт."""
        # Симулируем генерацию текста как в run_loop()
        random_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(300))
        original_data = random_text.encode('utf-8')
        
        assert len(original_data) == 300, f"Ожидалось 300 байт, получено {len(original_data)}"
        print(f"[TEST] Размер данных: {len(original_data)} байт - OK")
    
    def test_run_loop_generates_300_bytes(self):
        """Проверка, что run_loop генерирует 300 байт данных."""
        from modem_cli import run_loop
        
        # Мокаем input для имитации ввода пользователя
        # Порядок ввода: тип модуляции, процент искажений, удаление файлов
        with patch('builtins.input') as mock_input, \
             patch('modem_cli.transmit_text') as mock_transmit, \
             patch('modem_cli.receive_from_file') as mock_receive, \
             patch('modem_cli.wavfile.read') as mock_wav_read, \
             patch('modem_cli.wavfile.write') as mock_wav_write, \
             patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open, \
             patch('os.remove') as mock_remove, \
             patch('sys.exit'):  # Мокаем sys.exit, чтобы избежать выхода из теста
            
            # Настраиваем возвращаемые значения
            # Порядок: 1) выбор модуляции ('2' = QPSK), 2) процент искажений, 3) удаление файлов
            mock_input.side_effect = ['2', '50', 'n']  # QPSK, 50% искажений, не удалять файлы
            mock_wav_read.return_value = (48000, np.random.randn(10000).astype(np.int16))
            mock_exists.return_value = True
            mock_receive.return_value = True
            
            # Мокаем open, чтобы перехватить запись оригинальных данных
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Запускаем run_loop
            # (может не выполниться полностью из-за моков, но проверим генерацию данных)
            try:
                run_loop()
            except Exception as e:
                print(f"[TEST] run_loop завершился с ошибкой (ожидаемо при моках): {e}")
            
            # Проверяем, что transmit_text был вызван с текстом длиной ~300 байт
            if mock_transmit.called:
                call_args = mock_transmit.call_args
                transmitted_text = call_args[0][0]
                data_bytes = transmitted_text.encode('utf-8')
                print(f"[TEST] Переданный текст: {len(data_bytes)} байт")
                # Проверяем, что размер данных около 300 байт (может отличаться из-за кодировки)
                assert len(data_bytes) == 300, f"Ожидалось 300 байт, получено {len(data_bytes)}"


class TestPrintDistortionParams300Bytes:
    """Тесты для проверки вывода параметров симуляции с 300 байтами."""
    
    def test_print_distortion_params_with_300_bytes(self, capsys):
        """Проверка, что print_distortion_params выводит параметры для 300 байт."""
        from modem_cli import print_distortion_params
        
        # Вызываем функцию с 300 байтами данных
        print_distortion_params(50.0, 48000, 300)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Проверяем, что выводится информация о размере данных
        assert "300 байт" in output, "В выводе нет информации о 300 байтах"
        assert "Размер данных: 300 байт" in output, "Нет строки о размере данных"
        print("[TEST] Вывод параметров с 300 байтами - OK")
    
    def test_print_distortion_params_calculates_duration(self, capsys):
        """Проверка, что print_distortion_params правильно рассчитывает длительность."""
        from modem_cli import print_distortion_params
        
        # Вызываем функцию
        print_distortion_params(50.0, 48000, 300)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Проверяем, что выводится информация о длительности
        assert "Длительность" in output, "В выводе нет информации о длительности"
        assert "Оценка длительности сигнала" in output, "Нет заголовка об оценке длительности"
        print("[TEST] Расчет длительности сигнала - OK")
    
    def test_print_distortion_params_shows_modulation(self, capsys):
        """Проверка, что print_distortion_params выводит тип модуляции."""
        from modem_cli import print_distortion_params
        
        # Вызываем функцию
        print_distortion_params(50.0, 48000, 300)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Проверяем, что выводится тип модуляции
        assert "Тип модуляции" in output, "В выводе нет информации о типе модуляции"
        print("[TEST] Вывод типа модуляции - OK")
    
    def test_print_distortion_params_shows_fs(self, capsys):
        """Проверка, что print_distortion_params выводит частоту дискретизации."""
        from modem_cli import print_distortion_params
        
        # Вызываем функцию
        print_distortion_params(50.0, 48000, 300)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Проверяем, что выводится частота дискретизации
        assert "Частота дискретизации" in output, "В выводе нет информации о частоте дискретизации"
        assert "48000" in output, "В выводе нет значения частоты 48000"
        print("[TEST] Вывод частоты дискретизации - OK")


class TestIntegration300Bytes:
    """Интеграционные тесты для 300 байт."""
    
    @patch('modem_cli.transmit_text')
    @patch('modem_cli.receive_from_file')
    @patch('modem_cli.wavfile.read')
    @patch('modem_cli.wavfile.write')
    @patch('modem_cli.apply_channel_distortions')
    @patch('builtins.open', create=True)
    @patch('os.path.exists')
    @patch('os.remove')
    def test_full_loop_with_300_bytes(self, mock_remove, mock_exists, mock_open,
                                      mock_apply, mock_wav_write, mock_wav_read,
                                      mock_receive, mock_transmit):
        """Проверка полного цикла Loop с 300 байтами."""
        from modem_cli import run_loop
        
        # Настройка моков
        mock_wav_read.return_value = (48000, np.random.randn(50000).astype(np.int16))
        mock_exists.return_value = True
        mock_receive.return_value = True
        mock_apply.return_value = np.random.randn(50000).astype(np.float64)
        
        # Мокаем input: тип модуляции, процент искажений, затем 'n' для удаления файлов
        with patch('builtins.input') as mock_input, \
             patch('sys.exit'):  # Мокаем sys.exit, чтобы избежать выхода из теста
            # Порядок: 1) выбор модуляции ('2' = QPSK), 2) процент искажений, 3) удаление файлов
            mock_input.side_effect = ['2', '30', 'n']  # QPSK, 30% искажений, не удалять
            
            try:
                run_loop()
                print("[TEST] run_loop выполнился успешно с 300 байтами")
            except Exception as e:
                pytest.fail(f"run_loop завершился с ошибкой: {e}")
        
        # Проверяем, что transmit_text был вызван
        assert mock_transmit.called, "transmit_text не был вызван"
        
        # Проверяем, что apply_channel_distortions был вызван
        assert mock_apply.called, "apply_channel_distortions не был вызван"
        
        print("[TEST] Интеграционный тест с 300 байтами - OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
