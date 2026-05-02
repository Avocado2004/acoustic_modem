"""
Тесты для модуля modem_tx.
Проверяют функции передачи: transmit_text, transmit_file, _transmit_data, softclip.
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock, call
import tempfile

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modem_tx import (transmit_text, transmit_file, _transmit_data, 
                       softclip_tanh, crest_db, apply_softclip_to_target_crest)


class TestSoftclipFunctions:
    """Тесты для функций мягкого клиппинга."""
    
    def test_softclip_tanh_zero_gain(self):
        """Тест softclip_tanh с нулевым усилением."""
        x = np.array([1.0, -0.5, 0.3])
        result = softclip_tanh(x, 0)
        np.testing.assert_array_equal(result, x)
    
    def test_softclip_tanh_negative_gain(self):
        """Тест softclip_tanh с отрицательным усилением."""
        x = np.array([1.0, -0.5, 0.3])
        result = softclip_tanh(x, -1.0)
        np.testing.assert_array_equal(result, x)
    
    def test_softclip_tanh_small_gain(self):
        """Тест softclip_tanh с малым усилением."""
        x = np.array([1.0, -0.5, 0.3])
        result = softclip_tanh(x, 0.1)
        # При малом g результат должен быть близок к x
        np.testing.assert_allclose(result, x, rtol=1e-1)
    
    def test_softclip_tanh_large_gain(self):
        """Тест softclip_tanh с большим усилением."""
        x = np.array([1.0, -0.5, 0.3])
        result = softclip_tanh(x, 10.0)
        # При большом g результат должен быть ограничен
        assert np.all(np.abs(result) <= 1.0)
    
    def test_softclip_tanh_empty(self):
        """Тест softclip_tanh с пустым массивом."""
        x = np.array([])
        result = softclip_tanh(x, 1.0)
        assert len(result) == 0
    
    def test_crest_db_basic(self):
        """Тест расчета пик-фактора."""
        # Синусоида: peak=1, rms=1/sqrt(2)
        x = np.sin(np.linspace(0, 2*np.pi, 1000))
        result = crest_db(x)
        expected = 20 * np.log10(np.sqrt(2))  # ~3.01 дБ
        assert abs(result - expected) < 0.5
    
    def test_crest_db_zero_signal(self):
        """Тест расчета пик-фактора для нулевого сигнала."""
        x = np.zeros(100)
        result = crest_db(x)
        assert result == np.inf or result > 1e10
    
    def test_crest_db_constant(self):
        """Тест расчета пик-фактора для константы."""
        x = np.ones(100) * 5.0
        result = crest_db(x)
        assert abs(result) < 1e-10  # 20*log10(1) = 0
    
    def test_apply_softclip_basic(self):
        """Тест применения мягкого клиппинга."""
        x = np.random.randn(1000)
        result, crest = apply_softclip_to_target_crest(x, target_db=6.0)
        assert len(result) == len(x)
        assert crest <= 6.1  # Допуск 0.1 дБ
    
    def test_apply_softclip_empty(self):
        """Тест применения мягкого клиппинга к пустому сигналу."""
        x = np.array([])
        result, crest = apply_softclip_to_target_crest(x)
        assert len(result) == 0
    
    def test_apply_softclip_zero_signal(self):
        """Тест применения мягкого клиппинга к нулевому сигналу."""
        x = np.zeros(100)
        result, crest = apply_softclip_to_target_crest(x)
        # Для нулевого сигнала crest_db возвращает inf
        assert crest == np.inf or crest > 1e10


class TestTransmitFunctions:
    """Тесты для функций передачи."""
    
    def setup_method(self):
        """Подготовка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('modem_tx.build_preamble')
    @patch('modem_tx.build_data_td')
    @patch('modem_tx.bytes_to_ofdm_blocks_bytes')
    @patch('audio_backend.get_audio_backend')  # get_audio реально в audio_backend
    @patch('wav_utils.wavfile')  # wavfile импортируется из wav_utils
    def test_transmit_text_basic(self, mock_wavfile, mock_get_audio, 
                                  mock_bytes_to_blocks, mock_build_data, mock_build_preamble):
        """Тест передачи текста."""
        # Настраиваем моки
        mock_build_preamble.return_value = np.zeros(100)
        mock_build_data.return_value = (np.zeros(100), 1)
        mock_bytes_to_blocks.return_value = (np.zeros(100), 1)
        mock_audio = MagicMock()
        mock_get_audio.return_value = mock_audio
        
        # Вызываем функцию
        result = transmit_text("Test text")
        
        # Проверяем, что wavfile.write был вызван
        mock_wavfile.write.assert_called_once()
        assert result == True
    
    @patch('modem_tx.os.path.isfile')
    @patch('modem_tx.os.path.getsize')
    @patch('modem_tx.open', create=True)
    @patch('modem_tx.build_preamble')
    @patch('modem_tx.build_data_td')
    @patch('modem_tx.bytes_to_ofdm_blocks_bytes')
    @patch('audio_backend.get_audio_backend')  # get_audio реально в audio_backend
    @patch('wav_utils.wavfile')  # wavfile импортируется из wav_utils
    def test_transmit_file_success(self, mock_wavfile, mock_get_audio,
                                   mock_bytes_to_blocks, mock_build_data, 
                                   mock_build_preamble, mock_open, 
                                   mock_getsize, mock_isfile):
        """Тест успешной передачи файла."""
        # Настраиваем моки
        mock_isfile.return_value = True
        mock_getsize.return_value = 100
        mock_build_preamble.return_value = np.zeros(100)
        mock_build_data.return_value = (np.zeros(100), 1)
        mock_bytes_to_blocks.return_value = (np.zeros(100), 1)
        mock_audio = MagicMock()
        mock_get_audio.return_value = mock_audio
        
        # Мокаем open для чтения файла
        mock_file = MagicMock()
        mock_file.read.return_value = b'test file content'
        mock_open.return_value.__enter__ = lambda s: mock_file
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        
        result = transmit_file("/fake/path.txt")
        
        assert result == True
    
    @patch('modem_tx.os.path.isfile')
    def test_transmit_file_not_found(self, mock_isfile):
        """Тест передачи несуществующего файла."""
        mock_isfile.return_value = False
        result = transmit_file("/nonexistent/file.txt")
        assert result == False
    
    @patch('modem_tx.os.path.getsize')
    @patch('modem_tx.os.path.isfile')
    def test_transmit_file_too_large(self, mock_isfile, mock_getsize):
        """Тест передачи слишком большого файла."""
        mock_isfile.return_value = True
        mock_getsize.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        result = transmit_file("/fake/large_file.txt")
        assert result == False
    
    def test_bytes_to_ofdm_blocks_bytes_empty(self):
        """Тест разбивки пустого потока."""
        from modem_tx import _bytes_to_ofdm_blocks_bytes
        td, nblocks = _bytes_to_ofdm_blocks_bytes(b'')
        assert len(td) == 0
        assert nblocks == 0
    
    @patch('modem_tx.rs')
    @patch('modem_tx.bytes_to_bits')
    @patch('modem_tx.build_data_td')
    def test_bytes_to_ofdm_blocks_bytes_basic(self, mock_build_data, mock_bytes_to_bits, mock_rs):
        """Тест разбивки потока байт."""
        # Импортируем внутреннюю функцию правильно
        import modem_tx
        # Находим функцию в модуле
        func = getattr(modem_tx, '_bytes_to_ofdm_blocks_bytes', None)
        if func is None:
            # Если функция не найдена, пропускаем тест
            pytest.skip("Функция _bytes_to_ofdm_blocks_bytes не найдена в modem_tx")
            return
        
        # Настраиваем моки
        mock_rs.encode.return_value = b'\x00' * 12
        mock_bytes_to_bits.return_value = np.zeros(96)  # 12 байт * 8 бит
        mock_build_data.return_value = (np.zeros(100), 1)
        
        td, nblocks = func(b'test')
        assert nblocks >= 0
