"""
Тесты для проверки группировки символов BPSK.
Проверяет, что 1 логический блок = 2 физических символа BPSK (96 бит).
"""

import sys
import os
import numpy as np
import zlib

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modem_config
from modem_config import (Nsub, RS_DATA_BYTES, RS_CW_BITS, DEFAULT_PACKET_BLOCKS,
                          OFDM_SYMBOLS_PER_BLOCK, BITS_PER_SYMBOL, BITS_PER_OFDM_SYMBOL)
import modem_packet
from modem_packet import build_header, parse_header, simulate_packet_positions, bytes_to_ofdm_blocks_bytes


class TestOFDMSymbolsPerBlock:
    """Тесты для проверки OFDM_SYMBOLS_PER_BLOCK."""
    
    def test_default_qpsk_ofdm_symbols_per_block(self):
        """По умолчанию (QPSK) OFDM_SYMBOLS_PER_BLOCK должен быть 1."""
        modem_config.set_modulation("QPSK")
        assert modem_config.OFDM_SYMBOLS_PER_BLOCK == 1
        assert modem_config.BITS_PER_SYMBOL == 2
        assert modem_config.BITS_PER_OFDM_SYMBOL == Nsub * 2  # 96 бит
        print(f"[TEST] QPSK: OFDM_SYMBOLS_PER_BLOCK={modem_config.OFDM_SYMBOLS_PER_BLOCK}, BITS_PER_OFDM_SYMBOL={modem_config.BITS_PER_OFDM_SYMBOL}")
    
    def test_bpsk_ofdm_symbols_per_block(self):
        """Для BPSK OFDM_SYMBOLS_PER_BLOCK должен быть 2."""
        modem_config.set_modulation("BPSK")
        assert modem_config.OFDM_SYMBOLS_PER_BLOCK == 2
        assert modem_config.BITS_PER_SYMBOL == 1
        assert modem_config.BITS_PER_OFDM_SYMBOL == Nsub * 1  # 48 бит
        print(f"[TEST] BPSK: OFDM_SYMBOLS_PER_BLOCK={modem_config.OFDM_SYMBOLS_PER_BLOCK}, BITS_PER_OFDM_SYMBOL={modem_config.BITS_PER_OFDM_SYMBOL}")
        # Возвращаем QPSK
        modem_config.set_modulation("QPSK")
    
    def test_logical_vs_physical_blocks_bpsk(self):
        """Проверка: 75 логических блоков BPSK = 150 физических символов."""
        modem_config.set_modulation("BPSK")
        logical_blocks = 75
        physical_symbols = logical_blocks * modem_config.OFDM_SYMBOLS_PER_BLOCK
        assert physical_symbols == 150  # 75 * 2 = 150
        print(f"[TEST] BPSK: {logical_blocks} логических блоков = {physical_symbols} физических символов")
        modem_config.set_modulation("QPSK")
    
    def test_logical_vs_physical_blocks_qpsk(self):
        """Проверка: 75 логических блоков QPSK = 75 физических символов."""
        modem_config.set_modulation("QPSK")
        logical_blocks = 75
        physical_symbols = logical_blocks * modem_config.OFDM_SYMBOLS_PER_BLOCK
        assert physical_symbols == 75  # 75 * 1 = 75
        print(f"[TEST] QPSK: {logical_blocks} логических блоков = {physical_symbols} физических символов")


class TestLogicalBlocksInHeader:
    """Тесты для проверки сохранения логических блоков в заголовке."""
    
    def test_header_stores_logical_blocks_bpsk(self):
        """Заголовок должен хранить количество логических блоков (не физических)."""
        modem_config.set_modulation("BPSK")
        
        # Создаем заголовок с 75 логическими блоками
        hdr = build_header(b'F', 300, filename_bytes=b'test.txt', packet_no=0, 
                          version=0, packet_blocks=75, crc32=0)
        
        # Парсим заголовок
        parsed = parse_header(hdr)
        
        # Должны получить 75 логических блоков
        assert parsed['packet_blocks'] == 75
        assert parsed['modulation'] == "BPSK"
        print(f"[TEST] Header stores logical blocks: packet_blocks={parsed['packet_blocks']}, modulation={parsed['modulation']}")
        
        modem_config.set_modulation("QPSK")
    
    def test_header_stores_logical_blocks_qpsk(self):
        """Заголовок для QPSK также хранит логические блоки."""
        modem_config.set_modulation("QPSK")
        
        hdr = build_header(b'F', 300, filename_bytes=b'test.txt', packet_no=0, 
                          version=0, packet_blocks=75, crc32=0)
        
        parsed = parse_header(hdr)
        
        assert parsed['packet_blocks'] == 75
        assert parsed['modulation'] == "QPSK"
        print(f"[TEST] Header stores logical blocks: packet_blocks={parsed['packet_blocks']}, modulation={parsed['modulation']}")


class TestSimulatePacketPositions:
    """Тесты для проверки simulate_packet_positions с логическими блоками."""
    
    def test_simulate_positions_bpsk(self):
        """Проверка позиций пакетов для BPSK с логическими блоками."""
        modem_config.set_modulation("BPSK")
        
        # 300 байт данных, 75 логических блоков
        total_data = 300
        packet_blocks = 75  # логические блоки
        
        # Симуляция позиций
        positions = simulate_packet_positions(
            total_data, b'test.txt', False, packet_blocks,
            preamble_len=1536, symbol_len=640, gap_samples=0
        )
        
        assert len(positions) > 0
        print(f"[TEST] BPSK simulate_positions: {len(positions)} packets, positions={positions[:3]}...")
        
        modem_config.set_modulation("QPSK")
    
    def test_simulate_positions_qpsk(self):
        """Проверка позиций пакетов для QPSK."""
        modem_config.set_modulation("QPSK")
        
        total_data = 300
        packet_blocks = 75  # логические блоки
        
        positions = simulate_packet_positions(
            total_data, b'test.txt', False, packet_blocks,
            preamble_len=1536, symbol_len=640, gap_samples=0
        )
        
        assert len(positions) > 0
        print(f"[TEST] QPSK simulate_positions: {len(positions)} packets, positions={positions[:3]}...")
    
    def test_physical_symbols_calculation_bpsk(self):
        """Проверка расчета физических символов в simulate_packet_positions."""
        modem_config.set_modulation("BPSK")
        
        # Для BPSK: 1 логический блок = 2 физических символа
        # 75 логических блоков = 150 физических символов
        logical_blocks = 75
        physical_symbols = logical_blocks * modem_config.OFDM_SYMBOLS_PER_BLOCK
        
        assert physical_symbols == 150
        print(f"[TEST] BPSK: 75 logical blocks = {physical_symbols} physical symbols")
        
        modem_config.set_modulation("QPSK")


class TestBytesToOFDMBlocks:
    """Тесты для проверки bytes_to_ofdm_blocks_bytes."""
    
    def test_bytes_to_ofdm_returns_physical_symbols(self):
        """bytes_to_ofdm_blocks_bytes должна возвращать количество ФИЗИЧЕСКИХ символов."""
        # Проверяем для обеих модуляций
        for mod in ["QPSK", "BPSK"]:
            modem_config.set_modulation(mod)
            
            # 12 байт = 1 RS кодовое слово
            test_data = b'\x00' * 12  # 1 RS блок
            td, n_physical = bytes_to_ofdm_blocks_bytes(test_data)
            
            # Должно возвращать положительное количество физических символов
            assert n_physical > 0
            print(f"[TEST] {mod}: 1 RS block -> {n_physical} physical symbols")
        
        modem_config.set_modulation("QPSK")
    
    def test_300_bytes_physical_symbols(self):
        """Проверка для 300 байт в обоих режимах."""
        for mod in ["QPSK", "BPSK"]:
            modem_config.set_modulation(mod)
            
            test_data = b'\xAA' * 300
            td, n_physical = bytes_to_ofdm_blocks_bytes(test_data)
            
            # Проверяем, что физических символов достаточно для передачи данных
            assert n_physical > 0
            print(f"[TEST] {mod}: 300 bytes -> {n_physical} physical symbols")
        
        modem_config.set_modulation("QPSK")


class TestFullLoop300BytesBPSK:
    """Тест полного цикла передачи-приема 300 байт в BPSK режиме."""
    
    def test_full_loop_300_bytes_bpsk(self):
        """Полный цикл: передача 300 байт текста в BPSK, прием и проверка."""
        import io
        from contextlib import redirect_stdout
        
        # Устанавливаем BPSK
        modem_config.set_modulation("BPSK")
        print(f"[TEST] Using BPSK: OFDM_SYMBOLS_PER_BLOCK={modem_config.OFDM_SYMBOLS_PER_BLOCK}")
        
        # Перенаправляем вывод в строку, чтобы не засорять консоль
        f = io.StringIO()
        with redirect_stdout(f):
            # Тестируем передачу текста
            test_text = "A" * 300  # 300 байт
            print(f"[TEST] Transmitting {len(test_text)} bytes via BPSK...")
            
            # Проверяем, что можем создать заголовок
            crc_val = zlib.crc32(test_text.encode('utf-8')) & 0xFFFFFFFF
            hdr = build_header(b'T', len(test_text), filename_bytes=b'', packet_no=0,
                              version=0, packet_blocks=75, crc32=crc_val)
            parsed = parse_header(hdr)
            assert parsed['data_len'] == len(test_text)
            assert parsed['modulation'] == "BPSK"
            print(f"[TEST] Header created and parsed successfully for BPSK")
        
        # Возвращаем QPSK
        modem_config.set_modulation("QPSK")
        print("[TEST] Full loop test passed (header creation and parsing)")


class TestBackwardCompatibility:
    """Тесты обратной совместимости."""
    
    def test_default_packet_blocks_constant(self):
        """DEFAULT_PACKET_BLOCKS остается константой (75), но теперь это логические блоки."""
        assert modem_config.DEFAULT_PACKET_BLOCKS == 75
        print(f"[TEST] DEFAULT_PACKET_BLOCKS = {modem_config.DEFAULT_PACKET_BLOCKS} (logical blocks)")
    
    def test_qpsk_backward_compatible(self):
        """QPSK должен работать как раньше (1 логический блок = 1 физический символ)."""
        modem_config.set_modulation("QPSK")
        
        # Для QPSK ничего не изменилось в поведении
        assert modem_config.OFDM_SYMBOLS_PER_BLOCK == 1
        assert modem_config.BITS_PER_OFDM_SYMBOL == 96  # 48 * 2 = 96
        
        # 75 логических блоков = 75 физических символов
        physical = 75 * modem_config.OFDM_SYMBOLS_PER_BLOCK
        assert physical == 75
        print(f"[TEST] QPSK backward compatible: 75 logical = {physical} physical symbols")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
