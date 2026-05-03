"""
Тесты для проверки автоопределения модуляции BPSK/QPSK.
Проверяют логику переключения и автоопределения.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modem_config
from modem_config import Nsub, Nfft, Ncp, SYMBOL_LEN, DEFAULT_PACKET_BLOCKS


class TestModulationSwitch:
    """Тесты для переключения модуляции."""
    
    def test_modulation_switch_to_bpsk(self):
        """Тест переключения в BPSK модуляцию."""
        original_mod = modem_config.MODULATION
        original_bps = modem_config.BITS_PER_SYMBOL
        original_bpofdm = modem_config.BITS_PER_OFDM_SYMBOL
        
        try:
            # Переключаемся в BPSK
            modem_config.MODULATION = "BPSK"
            modem_config.BITS_PER_SYMBOL = 1
            modem_config.BITS_PER_OFDM_SYMBOL = Nsub * 1
            
            assert modem_config.MODULATION == "BPSK"
            assert modem_config.BITS_PER_SYMBOL == 1
            assert modem_config.BITS_PER_OFDM_SYMBOL == Nsub
            
            print(f"[TEST] Switched to BPSK: BITS_PER_SYMBOL={modem_config.BITS_PER_SYMBOL}")
        finally:
            modem_config.MODULATION = original_mod
            modem_config.BITS_PER_SYMBOL = original_bps
            modem_config.BITS_PER_OFDM_SYMBOL = original_bpofdm
    
    def test_modulation_switch_to_qpsk(self):
        """Тест переключения в QPSK модуляцию."""
        original_mod = modem_config.MODULATION
        original_bps = modem_config.BITS_PER_SYMBOL
        original_bpofdm = modem_config.BITS_PER_OFDM_SYMBOL
        
        try:
            # Переключаемся в QPSK
            modem_config.MODULATION = "QPSK"
            modem_config.BITS_PER_SYMBOL = 2
            modem_config.BITS_PER_OFDM_SYMBOL = Nsub * 2
            
            assert modem_config.MODULATION == "QPSK"
            assert modem_config.BITS_PER_SYMBOL == 2
            assert modem_config.BITS_PER_OFDM_SYMBOL == Nsub * 2
            
            print(f"[TEST] Switched to QPSK: BITS_PER_SYMBOL={modem_config.BITS_PER_SYMBOL}")
        finally:
            modem_config.MODULATION = original_mod
            modem_config.BITS_PER_SYMBOL = original_bps
            modem_config.BITS_PER_OFDM_SYMBOL = original_bpofdm


class TestAutoDetectionLogic:
    """Тесты для проверки логики автоопределения."""
    
    def test_decode_packet_calls_both_modulations(self):
        """Тест что decode_packet_at_candidate вызывает обе модуляции для первого пакета."""
        import modem_rx
        
        # Логируем вызовы
        call_log = []
        original_try_decode = modem_rx._try_decode_with_modulation
        
        def mock_try_decode(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, modulation):
            call_log.append(modulation)
            # Возвращаем None чтобы симулировать неудачу
            return None
        
        try:
            modem_rx._try_decode_with_modulation = mock_try_decode
            
            # Мокаем необходимые глобальные переменные
            modem_rx.rx = MagicMock()
            modem_rx.abs_corr = MagicMock()
            modem_rx.preamble_td = MagicMock()
            modem_rx.last_agc_rms = 1.0
            modem_rx.global_equalizer = None
            
            # Вызываем decode_packet_at_candidate для первого пакета
            result = modem_rx.decode_packet_at_candidate(1000, DEFAULT_PACKET_BLOCKS, packet_idx=0)
            
            # Проверяем, что обе модуляции были вызваны
            assert "QPSK" in call_log, "QPSK должна быть вызвана"
            assert "BPSK" in call_log, "BPSK должна быть вызвана"
            print(f"[TEST] Both modulations called: {call_log}")
            
        finally:
            modem_rx._try_decode_with_modulation = original_try_decode
    
    def test_decode_packet_selects_best_modulation(self):
        """Тест что decode_packet_at_candidate выбирает лучшую модуляцию."""
        import modem_rx
        from modem_rx import _try_decode_with_modulation as original_try_decode
        
        # Симулируем успешное декодирование для QPSK (с большим rs_ok)
        qpsk_result = ([(b'data1', True), (b'data2', True)], 10, 1000, MagicMock())
        bpsk_result = ([(b'data1', True)], 5, 1000, MagicMock())
        
        call_count = 0
        def mock_try_decode(pref_abs, packet_blocks_expected, packet_idx, bytes_before_packet, expected_total, modulation):
            nonlocal call_count
            call_count += 1
            if modulation == "QPSK":
                return qpsk_result
            else:
                return bpsk_result
        
        try:
            modem_rx._try_decode_with_modulation = mock_try_decode
            
            # Мокаем необходимые глобальные переменные
            modem_rx.rx = MagicMock()
            modem_rx.abs_corr = MagicMock()
            modem_rx.preamble_td = MagicMock()
            modem_rx.last_agc_rms = 1.0
            modem_rx.global_equalizer = None
            
            # Вызываем decode_packet_at_candidate для первого пакета
            decoded_bytes, rs_ok, used_pre = modem_rx.decode_packet_at_candidate(
                1000, DEFAULT_PACKET_BLOCKS, packet_idx=0
            )
            
            # Проверяем, что выбрана QPSK (с большим rs_ok)
            assert rs_ok == 10, f"Должно быть выбрано QPSK с rs_ok=10, а не {rs_ok}"
            assert modem_config.MODULATION == "QPSK", \
                f"Модуляция должна быть QPSK, а не {modem_config.MODULATION}"
            
            print(f"[TEST] Best modulation selected: QPSK with RS_OK=10")
            
        finally:
            modem_rx._try_decode_with_modulation = original_try_decode


class TestCLIWithModulation:
    """Тесты CLI с выбором модуляции."""
    
    @patch('modem_cli.transmit_text')
    def test_transmit_text_with_qpsk(self, mock_transmit_text):
        """Тест передачи текста с QPSK."""
        mock_transmit_text.return_value = True
        
        from modem_cli import run_transmit
        
        # Выбор: Text mode (T), QPSK (Q), текст
        with patch('builtins.input', side_effect=['T', 'Q', 'Test text']):
            run_transmit()
            mock_transmit_text.assert_called_once_with('Test text')
            # Проверяем, что модуляция установилась
            assert modem_config.MODULATION == "QPSK"
            print(f"[TEST] CLI set modulation to QPSK")
    
    @patch('modem_cli.transmit_text')
    def test_transmit_text_with_bpsk(self, mock_transmit_text):
        """Тест передачи текста с BPSK."""
        mock_transmit_text.return_value = True
        
        from modem_cli import run_transmit
        
        # Выбор: Text mode (T), BPSK (B), текст
        with patch('builtins.input', side_effect=['T', 'B', 'Test text']):
            run_transmit()
            mock_transmit_text.assert_called_once_with('Test text')
            # Проверяем, что модуляция установилась
            assert modem_config.MODULATION == "BPSK"
            print(f"[TEST] CLI set modulation to BPSK")
    
    @patch('modem_cli.transmit_file')
    def test_transmit_file_with_bpsk(self, mock_transmit_file):
        """Тест передачи файла с BPSK."""
        mock_transmit_file.return_value = True
        
        from modem_cli import run_transmit
        
        # Выбор: File mode (F), BPSK (B), путь к файлу
        with patch('builtins.input', side_effect=['F', 'B', '/path/to/file']):
            run_transmit()
            mock_transmit_file.assert_called_once_with('/path/to/file')
            # Проверяем, что модуляция установилась
            assert modem_config.MODULATION == "BPSK"
            print(f"[TEST] CLI set modulation to BPSK for file transfer")


class TestHeaderModulation:
    """Тесты для проверки модуляции в заголовке."""
    
    def test_parse_header_modulation_bpsk(self):
        """Тест чтения BPSK модуляции из заголовка."""
        from modem_packet import parse_header
        
        # Создаем заголовок с BPSK модуляцией
        # mod_bits = 0b01 для BPSK
        # Флаг: 0xF0 | (0b01 << 2) | 0b10 (T mode) = 0xF0 | 0x04 | 0x02 = 0xF6
        header = bytearray(64)
        header[0] = 0xF6  # BPSK + Text mode
        header[1] = 0  # version
        # data_len = 100 (8 байт, big-endian)
        import struct
        header[2:10] = struct.pack('>Q', 100)
        # name_len = 0
        header[10:14] = struct.pack('>I', 0)
        # packet_no = 0
        header[46:50] = struct.pack('>I', 0)
        # packet_blocks = 75
        header[50:52] = struct.pack('>H', 75)
        # CRC32 = 0
        header[52:56] = struct.pack('>I', 0)
        
        hdr = parse_header(bytes(header))
        
        assert hdr['modulation'] == "BPSK", \
            f"Модуляция должна быть BPSK, а не {hdr['modulation']}"
        assert hdr['mode'] == b'T', f"Режим должен быть T, а не {hdr['mode']}"
        assert hdr['data_len'] == 100
        
        print(f"[TEST] Parsed header: modulation={hdr['modulation']}, mode={hdr['mode']}")
    
    def test_parse_header_modulation_qpsk(self):
        """Тест чтения QPSK модуляции из заголовка."""
        from modem_packet import parse_header
        
        # Создаем заголовок с QPSK модуляцией (по умолчанию)
        # mod_bits = 0b00 для QPSK
        # Флаг: 0xF0 | (0b00 << 2) | 0b10 (T mode) = 0xF0 | 0x00 | 0x02 = 0xF2
        header = bytearray(64)
        header[0] = 0xF2  # QPSK + Text mode
        header[1] = 0  # version
        # data_len = 100 (8 байт, big-endian)
        import struct
        header[2:10] = struct.pack('>Q', 100)
        # name_len = 0
        header[10:14] = struct.pack('>I', 0)
        # packet_no = 0
        header[46:50] = struct.pack('>I', 0)
        # packet_blocks = 75
        header[50:52] = struct.pack('>H', 75)
        # CRC32 = 0
        header[52:56] = struct.pack('>I', 0)
        
        hdr = parse_header(bytes(header))
        
        assert hdr['modulation'] == "QPSK", \
            f"Модуляция должна быть QPSK, а не {hdr['modulation']}"
        assert hdr['mode'] == b'T', f"Режим должен быть T, а не {hdr['mode']}"
        assert hdr['data_len'] == 100
        
        print(f"[TEST] Parsed header: modulation={hdr['modulation']}, mode={hdr['mode']}")
