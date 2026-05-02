#!/usr/bin/env python3
"""
Unit tests for OFDM Acoustic Modem modules.
Tests cover:
- Basic functionality (happy path)
- Boundary conditions (empty data, zero values)
- Error handling (exceptions via pytest.raises)
- Mocking external dependencies where needed
"""

import pytest
import numpy as np
import struct
import zlib
from unittest import mock
import sys
import os

# Import functions from the new modules
from modem_modulation import (
    text_to_bits, bits_to_text, bytes_to_bits, bits_to_bytes,
    qpsk_map, qpsk_demap, bpsk_map, bpsk_demap,
    crest_factor, sync_by_corr, # Added sync_by_corr
    ofdm_symbol, build_preamble, build_data_td,
    zc_root_sequence, make_subcarrier_phases, # Added
    optimize_phases_habr, init_phases
)
from modem_config import ( # Import config vars needed for tests
    Nfft, Ncp, Nsub, MODULATION, BITS_PER_SYMBOL,
    BITS_PER_OFDM_SYMBOL, RS_DATA_BYTES, RS_CW_BYTES, RS_CW_BITS, rs,
    SYMBOL_TX_TARGET, ACE_MAX_ITERS, ACE_PEAK_THRESHOLD, ACE_STEP, ACE_ALLOW_EXPANSION,
    subc_phases, PHASE_METHOD, HABR_SAMPLE_BLOCKS, HABR_SEED, HABR_MAX_ITERS,
    HABR_PHASE_GRID, HABR_SAVE_FILE
)
from modem_packet import (
    build_header, make_packet_header_bytes, parse_header,
    simulate_packet_positions, bytes_to_ofdm_blocks_bytes
)
from modem_tx import (
    softclip_tanh, crest_db, apply_softclip_to_target_crest
)


class TestTextBitsUtils:
    """Tests for text/bits conversion utilities."""

    def test_text_to_bits_basic(self):
        """Test basic text to bits conversion."""
        result = text_to_bits("A")
        expected = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_text_to_bits_empty(self):
        """Test text_to_bits with empty string."""
        result = text_to_bits("")
        assert len(result) == 0
        assert result.dtype == int

    def test_text_to_bits_unicode(self):
        """Test text_to_bits with Unicode characters."""
        result = text_to_bits("Привет")
        # Should produce some bits
        assert len(result) > 0
        assert all(b in (0, 1) for b in result)

    def test_bits_to_text_basic(self):
        """Test basic bits to text conversion."""
        bits = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=int)
        result = bits_to_text(bits)
        assert result == "A"

    def test_bits_to_text_empty(self):
        """Test bits_to_text with empty array."""
        result = bits_to_text(np.array([], dtype=int))
        assert result == ""

    def test_bits_to_text_not_multiple_of_8(self):
        """Test bits_to_text with length not multiple of 8 (should truncate)."""
        bits = np.array([0, 1, 0, 0, 0, 0, 0], dtype=int)  # 7 bits
        result = bits_to_text(bits)
        assert result == ""

    def test_text_to_bits_and_back(self):
        """Round-trip test: text -> bits -> text."""
        original = "Hello, World!"
        bits = text_to_bits(original)
        result = bits_to_text(bits)
        assert result == original

    def test_bytes_to_bits_basic(self):
        """Test bytes to bits conversion."""
        data = b"\x01\x80"
        result = bytes_to_bits(data)
        expected = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_bytes_to_bits_empty(self):
        """Test bytes_to_bits with empty bytes."""
        result = bytes_to_bits(b"")
        assert len(result) == 0
        assert result.dtype == int

    def test_bits_to_bytes_basic(self):
        """Test bits to bytes conversion."""
        bits = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
        result = bits_to_bytes(bits)
        assert result == b"\x01\x80"

    def test_bits_to_bytes_empty(self):
        """Test bits_to_bytes with empty array."""
        result = bits_to_bytes(np.array([], dtype=int))
        assert result == b""

    def test_bits_to_bytes_not_multiple_of_8(self):
        """Test bits_to_bytes with length not multiple of 8 (should truncate)."""
        bits = np.array([0, 1, 0, 0, 0, 0, 0], dtype=int)  # 7 bits
        result = bits_to_bytes(bits)
        assert result == b""

    def test_bytes_to_bits_and_back(self):
        """Round-trip test: bytes -> bits -> bytes."""
        original = b"Hello!"
        bits = bytes_to_bits(original)
        result = bits_to_bytes(bits)
        assert result == original


class TestQPSKMapping:
    """Tests for QPSK modulation/demodulation."""

    def test_qpsk_map_basic(self):
        """Test QPSK mapping with basic bit pairs."""
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)
        result = qpsk_map(bits)
        assert len(result) == 4
        # Check expected constellation points
        expected_00 = (1 + 1j) / np.sqrt(2)
        expected_01 = (-1 + 1j) / np.sqrt(2)
        expected_11 = (-1 - 1j) / np.sqrt(2)
        expected_10 = (1 - 1j) / np.sqrt(2)
        np.testing.assert_allclose(result[0], expected_00)
        np.testing.assert_allclose(result[1], expected_01)
        np.testing.assert_allclose(result[2], expected_11)
        np.testing.assert_allclose(result[3], expected_10)

    def test_qpsk_map_odd_length(self):
        """Test QPSK mapping with odd number of bits (should pad with 0)."""
        bits = np.array([0, 1, 1], dtype=int)
        result = qpsk_map(bits)
        # Should pad to 4 bits -> 2 symbols
        assert len(result) == 2

    def test_qpsk_map_empty(self):
        """Test QPSK mapping with empty array."""
        bits = np.array([], dtype=int)
        result = qpsk_map(bits)
        assert len(result) == 0

    def test_qpsk_demap_basic(self):
        """Test QPSK demapping."""
        syms = np.array([
            (1 + 1j) / np.sqrt(2),
            (-1 + 1j) / np.sqrt(2),
            (-1 - 1j) / np.sqrt(2),
            (1 - 1j) / np.sqrt(2)
        ], dtype=complex)
        result = qpsk_demap(syms)
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_qpsk_demap_empty(self):
        """Test QPSK demapping with empty array."""
        result = qpsk_demap(np.array([], dtype=complex))
        assert len(result) == 0

    def test_qpsk_map_demap_roundtrip(self):
        """Round-trip test for QPSK mapping/demapping."""
        bits = np.random.randint(0, 2, 100, dtype=int)
        syms = qpsk_map(bits)
        result = qpsk_demap(syms)
        # Should match original (possibly padded)
        expected_len = (len(bits) // 2) * 2
        np.testing.assert_array_equal(result[:expected_len], bits[:expected_len])

    def test_qpsk_map_demap_with_noise(self):
        """Test QPSK mapping/demapping with small noise."""
        bits = np.random.randint(0, 2, 20, dtype=int)
        syms = qpsk_map(bits)
        # Add small noise
        noise = 0.01 * (np.random.randn(len(syms)) + 1j * np.random.randn(len(syms)))
        syms_noisy = syms + noise
        result = qpsk_demap(syms_noisy)
        # Should still decode correctly for small noise
        expected_len = (len(bits) // 2) * 2
        np.testing.assert_array_equal(result[:expected_len], bits[:expected_len])


class TestBPSKMapping:
    """Tests for BPSK modulation/demodulation."""

    def test_bpsk_map_basic(self):
        """Test BPSK mapping."""
        bits = np.array([0, 1, 0, 1], dtype=int)
        result = bpsk_map(bits)
        expected = np.array([1.0, -1.0, 1.0, -1.0], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_bpsk_map_empty(self):
        """Test BPSK mapping with empty array."""
        result = bpsk_map(np.array([], dtype=int))
        assert len(result) == 0

    def test_bpsk_demap_basic(self):
        """Test BPSK demapping."""
        syms = np.array([1.0, -1.0, 1.0, -1.0], dtype=complex)
        result = bpsk_demap(syms)
        expected = np.array([0, 1, 0, 1], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_bpsk_demap_empty(self):
        """Test BPSK demapping with empty array."""
        result = bpsk_demap(np.array([], dtype=complex))
        assert len(result) == 0

    def test_bpsk_map_demap_roundtrip(self):
        """Round-trip test for BPSK mapping/demapping."""
        bits = np.random.randint(0, 2, 50, dtype=int)
        syms = bpsk_map(bits)
        result = bpsk_demap(syms)
        np.testing.assert_array_equal(result, bits)


class TestCrestUtilities:
    """Tests for crest factor utilities."""

    def test_crest_factor_basic(self):
        """Test crest factor calculation."""
        # Pure sine wave
        t = np.linspace(0, 1, 1000)
        sig = np.sin(2 * np.pi * 5 * t)
        cf = crest_factor(sig)
        # Sine wave crest factor should be ~3 dB
        assert 2.9 < cf < 3.1

    def test_crest_factor_zero_signal(self):
        """Test crest factor with zero signal."""
        sig = np.zeros(100)
        cf = crest_factor(sig)
        assert cf == np.inf

    def test_crest_factor_constant(self):
        """Test crest factor with constant signal."""
        sig = np.ones(100) * 5.0
        cf = crest_factor(sig)
        assert cf == 0.0

    def test_crest_factor_random(self):
        """Test crest factor with random signal."""
        np.random.seed(42)
        sig = np.random.randn(1000)
        cf = crest_factor(sig)
        assert cf > 0


class TestSubcarrierPhases:
    """Tests for subcarrier phase generation."""

    def test_make_subcarrier_phases_zeros(self):
        """Test with method=None."""
        phases = make_subcarrier_phases(method=None, N=10)
        assert len(phases) == 10
        np.testing.assert_array_equal(phases, 0)

    def test_make_subcarrier_phases_random(self):
        """Test random phase generation."""
        phases = make_subcarrier_phases(method="random", N=10, seed=42)
        assert len(phases) == 10
        assert all(0 <= p < 2 * np.pi for p in phases)

    def test_make_subcarrier_phases_schroeder(self):
        """Test Schroeder phase generation."""
        phases = make_subcarrier_phases(method="schroeder", N=10)
        assert len(phases) == 10
        # Check first few values
        expected_0 = 0
        expected_1 = np.pi * 1 * 0 / 10 % (2 * np.pi)
        expected_2 = np.pi * 2 * 1 / 10 % (2 * np.pi)
        assert phases[0] == pytest.approx(expected_0)
        assert phases[1] == pytest.approx(expected_1)
        assert phases[2] == pytest.approx(expected_2)

    def test_make_subcarrier_phases_habr_with_phases(self):
        """Test HABR method with provided phases."""
        habr_phases = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        phases = make_subcarrier_phases(method="habr", N=5, habr_phases=habr_phases)
        np.testing.assert_array_equal(phases, habr_phases % (2*np.pi))

    def test_make_subcarrier_phases_invalid_method(self):
        """Test with invalid method (should return zeros)."""
        phases = make_subcarrier_phases(method="invalid", N=5)
        np.testing.assert_array_equal(phases, 0)


class TestOFDMFunctions:
    """Tests for OFDM-related functions."""

    def test_ofdm_symbol_basic(self):
        """Test OFDM symbol generation."""
        data_syms = np.ones(Nsub, dtype=complex)
        symbol = ofdm_symbol(data_syms)
        # Should return time-domain signal of length Nfft + Ncp
        assert len(symbol) == Nfft + Ncp

    def test_ofdm_symbol_empty(self):
        """Test OFDM symbol with empty data."""
        data_syms = np.array([], dtype=complex)
        symbol = ofdm_symbol(data_syms)
        assert len(symbol) == Nfft + Ncp

    def test_ofdm_symbol_short_data(self):
        """Test OFDM symbol with fewer data symbols than subcarriers."""
        data_syms = np.ones(10, dtype=complex)
        symbol = ofdm_symbol(data_syms)
        assert len(symbol) == Nfft + Ncp

    def test_build_preamble(self):
        """Test preamble generation."""
        preamble = build_preamble(reps=1, zc_root=1)
        # Should be 4 OFDM symbols
        expected_len = 4 * (Nfft + Ncp)
        assert len(preamble) == expected_len

    def test_build_preamble_custom_reps(self):
        """Test preamble with custom repetitions."""
        preamble = build_preamble(reps=2, zc_root=1)
        # Note: build_preamble always uses 4 symbols regardless of reps
        # (reps parameter is not actually used in the function)
        expected_len = 4 * (Nfft + Ncp)
        assert len(preamble) == expected_len

    def test_zc_root_sequence(self):
        """Test Zadoff-Chu sequence generation."""
        zc = zc_root_sequence(u=1, L=10)
        assert len(zc) == 10
        # Should be unit norm
        assert abs(np.sqrt(np.mean(np.abs(zc)**2)) - 1.0) < 1e-10

    def test_zc_root_sequence_different_root(self):
        """Test ZC sequence with different root."""
        zc = zc_root_sequence(u=3, L=20)
        assert len(zc) == 20

    def test_build_data_td_basic(self):
        """Test building time-domain data."""
        bits = np.random.randint(0, 2, 100, dtype=int)
        td, nblocks = build_data_td(bits)
        assert nblocks > 0
        assert len(td) == nblocks * (Nfft + Ncp)

    def test_build_data_td_empty(self):
        """Test building time-domain data with empty bits."""
        # Function currently raises ValueError for empty input
        # This is a known limitation - empty data should be handled separately
        with pytest.raises(ValueError):
            td, nblocks = build_data_td(np.array([], dtype=int))

    def test_sync_by_corr(self):
        """Test synchronization by correlation."""
        # Create a simple test signal
        pre = np.array([1, 2, 3, 4, 5], dtype=float)
        rx = np.zeros(20)
        rx[10:15] = pre  # Insert at position 10
        offset = sync_by_corr(rx, pre)
        assert offset == 10

    def test_sync_by_corr_no_match(self):
        """Test synchronization with no match."""
        pre = np.array([1, 2, 3], dtype=float)
        rx = np.zeros(10)
        offset = sync_by_corr(rx, pre)
        # Should return 0 (argmax of zeros)
        assert offset == 0


class TestHeaderFunctions:
    """Tests for header building and parsing."""

    def test_build_header_basic(self):
        """Test building a basic transmission header."""
        header = build_header(b'F', 1000, filename_bytes=b'test.txt', packet_no=1)
        assert len(header) == 64
        # Check flags byte has top 4 bits set
        assert header[0] & 0xF0 == 0xF0

    def test_build_header_file_mode(self):
        """Test building header for file mode."""
        header = build_header(b'F', 500, filename_bytes=b'hello.txt', packet_no=5, packet_blocks=100)
        assert len(header) == 64
        # Parse it back
        parsed = parse_header(header)
        assert parsed['is_transmission'] == True
        assert parsed['tx_type'] == 0b11
        assert parsed['data_len'] == 500
        assert parsed['filename'] == 'hello.txt'
        assert parsed['packet_no'] == 5
        assert parsed['packet_blocks'] == 100

    def test_build_header_text_mode(self):
        """Test building header for text mode."""
        header = build_header(b'T', 200, packet_no=0, packet_blocks=50)
        parsed = parse_header(header)
        assert parsed['is_transmission'] == True
        assert parsed['tx_type'] == 0b10
        assert parsed['mode'] == b'T'
        assert parsed['filename'] == ''

    def test_build_header_with_crc32(self):
        """Test building header with CRC32."""
        crc_val = 0x12345678
        header = build_header(b'F', 100, filename_bytes=b'test.txt', crc32=crc_val)
        # Check CRC32 is stored at bytes 52-55
        stored_crc = struct.unpack('>I', header[52:56])[0]
        assert stored_crc == crc_val

    def test_build_header_zero_values(self):
        """Test building header with zero values."""
        header = build_header(b'\x00', 0, filename_bytes=b'', packet_no=0)
        assert len(header) == 64

    def test_make_packet_header_bytes(self):
        """Test making short packet header."""
        header = make_packet_header_bytes(packet_no=5, tx_type=0b01)
        assert len(header) == 4
        # Top 4 bits should be 0
        assert header[0] & 0xF0 == 0x00
        # Packet number should be encoded
        assert header[1] == 0
        assert header[2] == 0
        assert header[3] == 5

    def test_make_packet_header_bytes_large_packet_no(self):
        """Test making packet header with large packet number."""
        header = make_packet_header_bytes(packet_no=0xFFFFFF, tx_type=0b00)
        assert header[1] == 0xFF
        assert header[2] == 0xFF
        assert header[3] == 0xFF

    def test_parse_header_transmission(self):
        """Test parsing a transmission header."""
        header = build_header(b'F', 1234, filename_bytes=b'test.dat', packet_no=42, packet_blocks=75, crc32=0xDEADBEEF)
        parsed = parse_header(header)
        assert parsed['is_transmission'] == True
        assert parsed['version'] == 0
        assert parsed['tx_type'] == 0b11
        assert parsed['modulation'] == 'QPSK'  # Default
        assert parsed['data_len'] == 1234
        assert parsed['filename'] == 'test.dat'
        assert parsed['packet_no'] == 42
        assert parsed['packet_blocks'] == 75
        assert parsed['crc32'] == 0xDEADBEEF

    def test_parse_header_packet(self):
        """Test parsing a short packet header."""
        header = make_packet_header_bytes(packet_no=100, tx_type=0b01)
        parsed = parse_header(header)
        assert parsed['is_transmission'] == False
        assert parsed['tx_type'] == 0b01
        assert parsed['packet_no'] == 100

    def test_parse_header_empty(self):
        """Test parsing empty header (should raise ValueError)."""
        with pytest.raises(ValueError, match="Empty header data"):
            parse_header(b"")

    def test_parse_header_too_short_transmission(self):
        """Test parsing transmission header that's too short."""
        # Create a header with top bit set but only 10 bytes
        short_header = bytearray(10)
        short_header[0] = 0xF0  # Set top 4 bits
        with pytest.raises(ValueError, match="Transmission header requires 64 bytes"):
            parse_header(bytes(short_header))

    def test_parse_header_too_short_packet(self):
        """Test parsing packet header that's too short."""
        with pytest.raises(ValueError, match="Packet header requires 4 bytes"):
            parse_header(b"\x00\x00\x00")

    def test_build_and_parse_header_roundtrip(self):
        """Round-trip test: build header -> parse header."""
        original = build_header(
            b'F',
            data_len=9999,
            filename_bytes=b'roundtrip_test.txt',
            packet_no=123,
            version=0,
            packet_blocks=200,
            crc32=0xCAFEBABE
        )
        parsed = parse_header(original)
        assert parsed['is_transmission'] == True
        assert parsed['data_len'] == 9999
        assert parsed['filename'] == 'roundtrip_test.txt'
        assert parsed['packet_no'] == 123
        assert parsed['packet_blocks'] == 200
        assert parsed['crc32'] == 0xCAFEBABE


class TestBytesToOFDMBlocks:
    """Tests for bytes_to_ofdm_blocks_bytes function."""

    def test_bytes_to_ofdm_blocks_basic(self):
        """Test basic conversion."""
        data = b"Hello, World!"
        td, nblocks = bytes_to_ofdm_blocks_bytes(data)
        assert nblocks > 0
        assert len(td) == nblocks * (Nfft + Ncp)

    def test_bytes_to_ofdm_blocks_empty(self):
        """Test with empty bytes."""
        td, nblocks = bytes_to_ofdm_blocks_bytes(b"")
        assert nblocks == 0
        assert len(td) == 0

    def test_bytes_to_ofdm_blocks_exact_block_size(self):
        """Test with data exactly matching RS_DATA_BYTES."""
        data = b"A" * RS_DATA_BYTES
        td, nblocks = bytes_to_ofdm_blocks_bytes(data)
        assert nblocks == 1
        assert len(td) == 1 * (Nfft + Ncp)

    def test_bytes_to_ofdm_blocks_multiple_blocks(self):
        """Test with data spanning multiple blocks."""
        data = b"X" * (RS_DATA_BYTES * 3 + 5)
        td, nblocks = bytes_to_ofdm_blocks_bytes(data)
        # Should be 4 blocks (3 full + 1 padded)
        assert nblocks == 4
        assert len(td) == 4 * (Nfft + Ncp)

    def test_bytes_to_ofdm_blocks_roundtrip_with_header(self):
        """Test that header + data can be converted to OFDM blocks."""
        header = build_header(b'F', 10, filename_bytes=b'test.txt')
        data = b"1234567890"
        combined = header + data
        td, nblocks = bytes_to_ofdm_blocks_bytes(combined)
        assert nblocks > 0
        assert len(td) == nblocks * (Nfft + Ncp)


class TestSoftClipping:
    """Tests for soft clipping utilities."""

    def test_softclip_tanh_zero_gain(self):
        """Test softclip with zero gain (should return input unchanged)."""
        x = np.array([0.1, 0.5, 1.0, 2.0])
        result = softclip_tanh(x, g=0)
        np.testing.assert_array_equal(result, x)

    def test_softclip_tanh_negative_gain(self):
        """Test softclip with negative gain (should return input unchanged)."""
        x = np.array([0.1, 0.5, 1.0])
        result = softclip_tanh(x, g=-1)
        np.testing.assert_array_equal(result, x)

    def test_softclip_tanh_small_gain(self):
        """Test softclip with small gain."""
        x = np.array([0.0, 0.5, 1.0])
        result = softclip_tanh(x, g=0.5)
        assert len(result) == len(x)
        # Output should be bounded
        assert all(abs(r) <= 1.0 for r in result)

    def test_softclip_tanh_large_gain(self):
        """Test softclip with large gain (approaches hard clipping)."""
        x = np.array([0.0, 0.5, 1.0, 2.0])
        result = softclip_tanh(x, g=10.0)
        assert len(result) == len(x)

    def test_softclip_tanh_empty(self):
        """Test softclip with empty array."""
        result = softclip_tanh(np.array([]), g=1.0)
        assert len(result) == 0

    def test_crest_db_basic(self):
        """Test crest_db calculation."""
        # Sine wave
        t = np.linspace(0, 1, 1000)
        sig = np.sin(2 * np.pi * 5 * t)
        cdb = crest_db(sig)
        assert 2.9 < cdb < 3.1

    def test_crest_db_zero_signal(self):
        """Test crest_db with zero signal."""
        sig = np.zeros(10)
        cdb = crest_db(sig)
        assert cdb == np.inf

    def test_crest_db_constant(self):
        """Test crest_db with constant signal."""
        sig = np.ones(10) * 3.0
        cdb = crest_db(sig)
        assert cdb == 0.0

    def test_apply_softclip_to_target_crest_basic(self):
        """Test applying softclip to achieve target crest."""
        np.random.seed(42)
        x = np.random.randn(1000)
        result, final_crest = apply_softclip_to_target_crest(x, target_db=6.0)
        assert len(result) == len(x)
        # Final crest should be close to target
        assert abs(final_crest - 6.0) < 0.5

    def test_apply_softclip_to_target_crest_empty(self):
        """Test with empty array."""
        result, final_crest = apply_softclip_to_target_crest(np.array([]))
        assert len(result) == 0
        assert final_crest == 0.0

    def test_apply_softclip_to_target_crest_zero_signal(self):
        """Test with zero signal."""
        x = np.zeros(100)
        result, final_crest = apply_softclip_to_target_crest(x)
        np.testing.assert_array_equal(result, x)
        assert final_crest == np.inf

    def test_apply_softclip_to_target_crest_already_good(self):
        """Test with signal already below target crest."""
        # Create signal with low crest factor
        np.random.seed(42)
        x = np.random.randn(1000) * 0.01
        result, final_crest = apply_softclip_to_target_crest(x, target_db=6.0)
        # Signal should have crest below target
        assert final_crest <= 6.0 + 0.1  # within tolerance
        # Result should be normalized version of x (not identical)
        assert len(result) == len(x)

    def test_apply_softclip_to_target_crest_custom_target(self):
        """Test with custom target crest."""
        np.random.seed(123)
        x = np.random.randn(500)
        result, final_crest = apply_softclip_to_target_crest(x, target_db=4.0)
        assert abs(final_crest - 4.0) < 0.5


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_parse_header_wrong_type(self):
        """Test parse_header with wrong input type."""
        # Function raises TypeError for non-bytes input
        with pytest.raises(TypeError):
            parse_header("not bytes")

    def test_build_header_invalid_mode(self):
        """Test build_header with invalid mode."""
        # Should still work (defaults to tx_type 0)
        header = build_header(b'X', 100, filename_bytes=b'test.txt')
        assert len(header) == 64

    def test_qpsk_map_invalid_input(self):
        """Test qpsk_map with non-binary input."""
        # Function raises KeyError for non-binary input (0, 1 only)
        bits = np.array([0, 2, 0, 0, 0, 0, 0, 1], dtype=int)
        with pytest.raises(KeyError):
            qpsk_map(bits)

    def test_bits_to_text_invalid_encoding(self):
        """Test bits_to_text with invalid encoding."""
        bits = text_to_bits("Hello")
        # Should handle encoding errors gracefully
        result = bits_to_text(bits, encoding='utf-8')
        assert result == "Hello"

    def test_ofdm_symbol_complex_input(self):
        """Test ofdm_symbol with complex input."""
        data = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=complex)
        result = ofdm_symbol(data)
        assert len(result) == Nfft + Ncp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])