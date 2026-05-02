"""
Тесты для модуля modem_rx.
Проверяют функции приема: receive_from_file, live_receive_and_process, decode_packet_at_candidate.
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock, call
import tempfile

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modem_rx import (receive_from_file, live_receive_and_process, decode_packet_at_candidate,
                      qpsk_demap, bpsk_demap)
from modem_config import init_phases, subc_phases


class TestDecodePacket:
    """Тесты для decode_packet_at_candidate."""
    
    def setup_method(self):
        """Подготовка перед каждым тестом."""
        # Инициализируем фазы
        init_phases()
        # Устанавливаем глобальные переменные для тестов
        import modem_rx
        modem_rx.rx = np.random.randn(10000) * 0.1
        modem_rx.abs_corr = np.random.rand(10000)
        modem_rx.preamble_td = np.zeros(512)
        modem_rx.last_agc_rms = 0.1
        modem_rx.post_sync = False
        modem_rx.sync_sample_abs = None
        modem_rx.last_packet_end_sample = None
        modem_rx.last_packet_rs_ok = None
        modem_rx.pkt0_header_offset = None
        modem_rx.pkt0_header_bytes = None
        modem_rx.rx_syms_list = []
        modem_rx.Hk_smooth_list = []
        modem_rx.last_packet_used_pre = None
        modem_rx._rs_fail_prints_count = 0
    
    @patch('modem_rx.parse_header')
    @patch('modem_rx.bytes_to_bits')
    @patch('modem_rx.bits_to_bytes')
    @patch('modem_rx.rs')
    @patch('modem_rx.qpsk_demap')
    @patch('modem_rx.ofdm_symbol')
    @patch('modem_rx.build_preamble')
    @patch('modem_rx.sync_by_corr')
    def test_decode_packet_basic(self, mock_sync, mock_build_preamble, 
                               mock_ofdm_symbol, mock_qpsk_demap,
                               mock_rs, mock_bits_to_bytes, mock_bytes_to_bits,
                               mock_parse_header):
        """Тест базового декодирования пакета."""
        # Настраиваем моки
        mock_sync.return_value = 1000
        mock_build_preamble.return_value = np.zeros(512)
        mock_ofdm_symbol.return_value = np.zeros(512)
        mock_qpsk_demap.return_value = np.zeros(96)  # 12 байт * 8 бит
        mock_rs.decode.return_value = (b'\x00' * 12, True)  # Успешное декодирование
        mock_bits_to_bytes.return_value = b'\x00' * 12
        mock_bytes_to_bits.return_value = np.zeros(96)
        mock_parse_header.return_value = {
            'is_transmission': True,
            'version': 0,
            'tx_type': 0b10,
            'mode': b'T',
            'modulation': 'QPSK',
            'data_len': 50,
            'name_len': 0,
            'filename': '',
            'packet_no': 0,
            'packet_blocks': 75,
            'crc32': 0
        }
        
        result, rs_ok, used_pre = decode_packet_at_candidate(1000, 75, packet_idx=0, 
                                                         bytes_before_packet=0, expected_total=50)
        
        assert isinstance(result, bytes)
        assert rs_ok >= 0
    
    def test_decode_packet_invalid_sync(self):
        """Тест декодирования с невалидным sync."""
        import modem_rx
        modem_rx.rx = None  # Имитируем отсутствие данных
        
        result, rs_ok, used_pre = decode_packet_at_candidate(1000, 75, packet_idx=0)
        
        assert result == b''
        assert rs_ok == 0
        assert used_pre is None


class TestReceiveFromFile:
    """Тесты для receive_from_file."""
    
    def setup_method(self):
        """Подготовка перед каждым тестом."""
        init_phases()
        # Создаем временный WAV файл
        import wave
        import struct
        
        self.temp_wav = tempfile.mktemp(suffix='.wav')
        
        # Создаем простой WAV файл
        with wave.open(self.temp_wav, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(48000)
            # Записываем немного тишины
            frames = b'\x00\x00' * 48000  # 1 секунда тишины
            wav_file.writeframes(frames)
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if os.path.exists(self.temp_wav):
            os.remove(self.temp_wav)
    
    @patch('modem_rx.build_preamble')
    @patch('modem_rx.decode_packet_at_candidate')
    @patch('modem_rx.parse_header')
    @patch('modem_rx.simulate_packet_positions')
    @patch('wav_utils.wavfile')  # wavfile импортируется из wav_utils внутри функции
    def test_receive_from_file_basic(self, mock_wavfile, mock_simulate, 
                                    mock_parse_header, mock_decode, mock_build_preamble):
        """Тест базового приема из файла."""
        # Настраиваем моки
        mock_wavfile.read.return_value = (48000, np.zeros(48000, dtype=np.int16))
        mock_build_preamble.return_value = np.zeros(512)
        mock_decode.return_value = (b'\x00' * 64, 1, 1000)  # Успешное декодирование заголовка
        mock_parse_header.return_value = {
            'is_transmission': True,
            'version': 0,
            'tx_type': 0b10,
            'mode': b'T',
            'modulation': 'QPSK',
            'data_len': 50,
            'name_len': 0,
            'filename': '',
            'packet_no': 0,
            'packet_blocks': 75,
            'crc32': 0
        }
        mock_simulate.return_value = [0]  # Одна позиция
        
        result = receive_from_file(self.temp_wav)
        
        # Проверяем, что функция была вызвана
        mock_wavfile.read.assert_called_once()
        assert isinstance(result, bool)


class TestLiveReceive:
    """Тесты для live_receive_and_process."""
    
    @patch('audio_backend.get_audio_backend')  # get_audio реально в audio_backend
    @patch('modem_config.init_phases')  # init_phases находится в modem_config
    def test_live_receive_basic(self, mock_init_phases, mock_get_audio):
        """Тест базового живого приема."""
        # Настраиваем моки
        mock_audio = MagicMock()
        mock_get_audio.return_value = mock_audio
        
        # Имитируем остановку после небольшой задержки
        def side_effect(*args, **kwargs):
            import time
            time.sleep(0.1)
            raise KeyboardInterrupt()
        
        mock_audio.start_stream.side_effect = side_effect
        
        # Проверяем, что функция обрабатывает KeyboardInterrupt
        try:
            live_receive_and_process()
        except KeyboardInterrupt:
            pass  # Ожидаемое поведение
        
        mock_init_phases.assert_called_once()


class TestDemappingFunctions:
    """Тесты для функций демаппинга."""
    
    def test_qpsk_demap_basic(self):
        """Тест базового демаппинга QPSK."""
        # Символы QPSK: (1+1j)/sqrt(2), (-1+1j)/sqrt(2), и т.д.
        syms = np.array([(1+1j)/np.sqrt(2), (-1+1j)/np.sqrt(2), 
                         (-1-1j)/np.sqrt(2), (1-1j)/np.sqrt(2)])
        bits = qpsk_demap(syms)
        
        assert len(bits) == 8  # 4 символа * 2 бита
        assert all(bit in (0, 1) for bit in bits)
    
    def test_qpsk_demap_empty(self):
        """Тест демаппинга пустого массива QPSK."""
        syms = np.array([], dtype=complex)
        bits = qpsk_demap(syms)
        
        assert len(bits) == 0
    
    def test_bpsk_demap_basic(self):
        """Тест базового демаппинга BPSK."""
        # Символы BPSK: 1.0 -> 0, -1.0 -> 1
        syms = np.array([1.0, -1.0, 1.0, -1.0])
        bits = bpsk_demap(syms)
        
        assert len(bits) == 4
        assert list(bits) == [0, 1, 0, 1]
    
    def test_bpsk_demap_empty(self):
        """Тест демаппинга пустого массива BPSK."""
        syms = np.array([], dtype=complex)
        bits = bpsk_demap(syms)
        
        assert len(bits) == 0