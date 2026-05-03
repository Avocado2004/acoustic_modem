"""
Тесты для проверки валидации заголовка (is_header_valid).
"""
import unittest
import struct
from modem_packet import is_header_valid, build_header, DEFAULT_PACKET_BLOCKS

class TestHeaderValidation(unittest.TestCase):
    """Тесты для функции is_header_valid."""
    
    def test_valid_header(self):
        """Проверка валидного заголовка."""
        # Создаем нормальный заголовок
        hdr = bytearray(64)
        hdr[0] = 0xF0 | (0b00 << 2) | 0b10  # QPSK, режим T
        hdr[1] = 1  # version
        data_len = 100
        hdr[2:10] = struct.pack('>Q', data_len)
        # Устанавливаем ненулевой CRC
        hdr[52:56] = struct.pack('>I', 0x12345678)
        # packet_blocks
        hdr[50:52] = struct.pack('>H', 5)
        
        self.assertTrue(is_header_valid(bytes(hdr)))
    
    def test_null_header(self):
        """Проверка нулевого заголовка."""
        hdr = b'\x00' * 64
        self.assertFalse(is_header_valid(hdr))
    
    def test_short_header(self):
        """Проверка короткого заголовка."""
        hdr = b'\x00' * 3
        self.assertFalse(is_header_valid(hdr))
    
    def test_zero_crc(self):
        """Проверка заголовка с нулевым CRC."""
        hdr = bytearray(64)
        hdr[0] = 0xF0 | (0b00 << 2) | 0b10  # QPSK, режим T
        hdr[1] = 1
        data_len = 100
        hdr[2:10] = struct.pack('>Q', data_len)
        # CRC = 0
        hdr[52:56] = struct.pack('>I', 0)
        hdr[50:52] = struct.pack('>H', 5)
        
        self.assertFalse(is_header_valid(bytes(hdr)))
    
    def test_zero_data_and_blocks(self):
        """Проверка заголовка с нулевой длиной данных и нулевыми блоками."""
        hdr = bytearray(64)
        hdr[0] = 0xF0 | (0b00 << 2) | 0b10
        hdr[1] = 1
        # data_len = 0
        hdr[2:10] = struct.pack('>Q', 0)
        # packet_blocks = 0
        hdr[50:52] = struct.pack('>H', 0)
        # Ненулевой CRC
        hdr[52:56] = struct.pack('>I', 0xDEADBEEF)
        
        self.assertFalse(is_header_valid(bytes(hdr)))
    
    def test_invalid_tx_type(self):
        """Проверка заголовка с невалидным типом передачи."""
        hdr = bytearray(64)
        hdr[0] = 0xF0 | (0b00 << 2) | 0b01  # Невалидный tx_type
        hdr[1] = 1
        hdr[2:10] = struct.pack('>Q', 100)
        hdr[50:52] = struct.pack('>H', 5)
        hdr[52:56] = struct.pack('>I', 0x12345678)
        
        self.assertFalse(is_header_valid(bytes(hdr)))


class TestBuildHeaderIntegration(unittest.TestCase):
    """Интеграционные тесты с build_header."""
    
    def test_build_header_not_null(self):
        """Проверка, что созданный заголовок не нулевой."""
        hdr = build_header(b'T', 100, packet_blocks=5, crc32=0x12345678)
        self.assertTrue(is_header_valid(hdr))
    
    def test_build_header_default_crc_zero(self):
        """Проверка, что заголовок с CRC=0 (по умолчанию) не проходит валидацию."""
        hdr = build_header(b'T', 100, packet_blocks=5)  # crc32=0 по умолчанию
        # По новой логике такой заголовок должен быть невалидным
        self.assertFalse(is_header_valid(hdr))


if __name__ == '__main__':
    unittest.main()
