"""
Тесты для модуля modem_cli.
Проверяют функции CLI: run_transmit, run_receive, main.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modem_cli import run_transmit, run_receive, run_loop, main


class TestRunTransmit:
    """Тесты для run_transmit."""
    
    @patch('modem_cli.transmit_text')
    def test_transmit_text_mode(self, mock_transmit_text):
        """Тест режима передачи текста."""
        mock_transmit_text.return_value = True
        
        # Добавляем выбор модуляции (Q - QPSK по умолчанию)
        with patch('builtins.input', side_effect=['T', 'Q', 'Test text']):
            run_transmit()
            mock_transmit_text.assert_called_once_with('Test text')
    
    @patch('modem_cli.transmit_file')
    def test_transmit_file_mode(self, mock_transmit_file):
        """Тест режима передачи файла."""
        mock_transmit_file.return_value = True
        
        with patch('builtins.input', side_effect=['F', 'Q', '/path/to/file']):
            run_transmit()
            mock_transmit_file.assert_called_once_with('/path/to/file')
    
    @patch('modem_cli.transmit_file')
    def test_transmit_default_file_mode(self, mock_transmit_file):
        """Тест режима передачи файла по умолчанию."""
        mock_transmit_file.return_value = True
        
        # Если введено что-то другое, должно быть F
        with patch('builtins.input', side_effect=['X', 'Q', '/path/to/file']):
            run_transmit()
            mock_transmit_file.assert_called_once_with('/path/to/file')
    
    @patch('modem_cli.transmit_text')
    def test_transmit_bpsk_mode(self, mock_transmit_text):
        """Тест передачи текста с модуляцией BPSK."""
        mock_transmit_text.return_value = True
        
        with patch('builtins.input', side_effect=['T', 'B', 'Test text']):
            run_transmit()
            mock_transmit_text.assert_called_once_with('Test text')


class TestRunReceive:
    """Тесты для run_receive."""
    
    @patch('modem_cli.receive_from_file')
    def test_receive_from_file(self, mock_receive_file):
        """Тест приема из файла."""
        mock_receive_file.return_value = True
        
        with patch('builtins.input', side_effect=['file', 'test.wav']):
            run_receive()
            mock_receive_file.assert_called_once_with('test.wav')
    
    @patch('modem_cli.receive_from_file')
    @patch('modem_cli.live_receive_and_process')
    def test_receive_from_mic(self, mock_live, mock_receive_file):
        """Тест приема с микрофона."""
        with patch('builtins.input', side_effect=['mic']):
            with pytest.raises(SystemExit) as exc_info:
                run_receive()
            mock_live.assert_called_once()
            mock_receive_file.assert_not_called()
            assert exc_info.value.code == 0
    
    @patch('modem_cli.receive_from_file')
    def test_receive_file_failure(self, mock_receive_file):
        """Тест неудачного приема из файла."""
        mock_receive_file.return_value = False
        
        with patch('builtins.input', side_effect=['file', 'test.wav']):
            with pytest.raises(SystemExit):
                run_receive()


@pytest.mark.skip(reason="run_loop выполняет реальные операции ввода-вывода")
class TestRunLoop:
    """Тесты для run_loop. Пропущены, так как требуют реальных операций."""
    pass


class TestMain:
    """Тесты для main."""
    
    @patch('modem_cli.run_transmit')
    @patch('modem_cli.init_phases')
    @patch('modem_cli.build_preamble')
    def test_main_transmit_mode(self, mock_build_preamble, mock_init_phases, mock_run_transmit):
        """Тест main в режиме передачи."""
        with patch('builtins.input', side_effect=['T', 'Q']):
            main()
            mock_init_phases.assert_called_once()
            mock_build_preamble.assert_called_once()
            mock_run_transmit.assert_called_once()
    
    @patch('modem_cli.run_receive')
    @patch('modem_cli.init_phases')
    @patch('modem_cli.build_preamble')
    def test_main_receive_mode(self, mock_build_preamble, mock_init_phases, mock_run_receive):
        """Тест main в режиме приема."""
        with patch('builtins.input', side_effect=['R']):
            main()
            mock_init_phases.assert_called_once()
            mock_build_preamble.assert_called_once()
            mock_run_receive.assert_called_once()
    
    @patch('modem_cli.run_loop')
    @patch('modem_cli.init_phases')
    @patch('modem_cli.build_preamble')
    def test_main_loop_mode(self, mock_build_preamble, mock_init_phases, mock_run_loop):
        """Тест main в режиме Loop."""
        with patch('builtins.input', side_effect=['L']):
            main()
            mock_init_phases.assert_called_once()
            mock_build_preamble.assert_called_once()
            mock_run_loop.assert_called_once()
    
    @patch('modem_cli.run_transmit')
    @patch('modem_cli.init_phases')
    @patch('modem_cli.build_preamble')
    def test_main_default_mode(self, mock_build_preamble, mock_init_phases, mock_run_transmit):
        """Тест main с режимом по умолчанию."""
        with patch('builtins.input', side_effect=['']):
            main()
            mock_init_phases.assert_called_once()
            mock_build_preamble.assert_called_once()
            mock_run_transmit.assert_called_once()
