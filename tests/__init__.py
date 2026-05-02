"""
Unit tests for Acoustic Modem project.

This package contains comprehensive unit tests for the OFDM acoustic modem implementation.
Tests cover all major functionality including:

- Text/bits conversion utilities
- QPSK and BPSK modulation/demodulation
- OFDM symbol generation and processing
- Header building and parsing
- Soft clipping and crest factor calculations
- Audio signal generation
- Error handling and edge cases

Usage:
    pytest tests/                    # Run all tests
    pytest tests/test_modem_unit.py   # Run specific test file
    pytest tests/test_modem_unit.py::TestQPSKMapping::test_qpsk_map_basic  # Run specific test

Requirements:
    pytest
    numpy
"""

__version__ = "1.0.0"