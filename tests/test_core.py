"""
Быстрые тесты для Acoustic Modem.
Покрывают основные модули: config, encoder, signal_utils, channel_simulator.
Каждый тест выполняется менее чем за 1 секунду.
"""

import numpy as np
import pytest
import sys
import os

# Добавляем родительскую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Тесты конфигурации (modem_config)
# =============================================================================

class TestModemConfig:
    """Тесты модуля конфигурации modem_config."""
    
    def test_default_modulation_is_qpsk(self):
        """Проверка что модуляция по умолчанию - QPSK."""
        import modem_config
        assert modem_config.MODULATION == "QPSK"
    
    def test_set_modulation_bpsk(self):
        """Проверка переключения на BPSK."""
        import modem_config
        result = modem_config.set_modulation("BPSK")
        assert result is True
        assert modem_config.MODULATION == "BPSK"
        assert modem_config.BITS_PER_SYMBOL == 1
        assert modem_config.OFDM_SYMBOLS_PER_BLOCK == 2
        # Восстанавливаем QPSK
        modem_config.set_modulation("QPSK")
    
    def test_set_modulation_qpsk(self):
        """Проверка переключения на QPSK."""
        import modem_config
        modem_config.set_modulation("BPSK")  # Сначала ставим BPSK
        result = modem_config.set_modulation("QPSK")
        assert result is True
        assert modem_config.MODULATION == "QPSK"
        assert modem_config.BITS_PER_SYMBOL == 2
        assert modem_config.OFDM_SYMBOLS_PER_BLOCK == 1
    
    def test_set_modulation_invalid(self):
        """Проверка обработки невалидной модуляции."""
        import modem_config
        result = modem_config.set_modulation("INVALID")
        assert result is False
    
    def test_fs_is_48000(self):
        """Проверка частоты дискретизации."""
        import modem_config
        assert modem_config.fs == 48000
    
    def test_nfft_is_512(self):
        """Проверка размера FFT."""
        import modem_config
        assert modem_config.Nfft == 512
    
    def test_ncp_is_128(self):
        """Проверка длины циклического префикса."""
        import modem_config
        assert modem_config.Ncp == 128
    
    def test_subcarrier_count(self):
        """Проверка количества поднесущих."""
        import modem_config
        assert modem_config.Nsub == 48
    
    def test_symbol_length(self):
        """Проверка длины OFDM символа."""
        import modem_config
        expected = modem_config.Nfft + modem_config.Ncp
        assert modem_config.SYMBOL_LEN == expected
    
    def test_rs_codec_exists(self):
        """Проверка наличия RS кодека."""
        import modem_config
        assert modem_config.rs is not None
    
    def test_make_subcarrier_phases_schroeder(self):
        """Проверка генерации фаз методом Schroeder."""
        import modem_config
        phases = modem_config.make_subcarrier_phases("schroeder", N=48)
        assert len(phases) == 48
        assert np.all(phases >= 0)
        assert np.all(phases <= 2 * np.pi)
    
    def test_make_subcarrier_phases_random(self):
        """Проверка генерации случайных фаз."""
        import modem_config
        phases = modem_config.make_subcarrier_phases("random", N=48, seed=42)
        assert len(phases) == 48
    
    def test_make_subcarrier_phases_zero(self):
        """Проверка генерации нулевых фаз."""
        import modem_config
        phases = modem_config.make_subcarrier_phases(None, N=48)
        assert len(phases) == 48
        assert np.all(phases == 0)


# =============================================================================
# Тесты кодирования/декодирования (encoder)
# =============================================================================

class TestEncoder:
    """Тесты модуля encoder."""
    
    def test_text_to_bits(self):
        """Проверка преобразования текста в биты."""
        from encoder import text_to_bits
        bits = text_to_bits(b"Hello")
        assert len(bits) == 40  # 5 байт * 8 бит
        assert bits.startswith("01001000")  # 'H' = 0x48
    
    def test_bits_to_bytes(self):
        """Проверка преобразования битов в байты."""
        from encoder import bits_to_bytes
        result = bits_to_bytes("0100100001100101")  # "He"
        assert result == b"He"
    
    def test_text_roundtrip(self):
        """Проверка кругового преобразования текст -> биты -> текст."""
        from encoder import text_to_bits, bits_to_bytes
        original = b"Test message 123"
        bits = text_to_bits(original)
        result = bits_to_bytes(bits)
        assert result == original
    
    def test_encrypt_decrypt_no_password(self):
        """Проверка шифрования без пароля (прямое кодирование)."""
        from encoder import encrypt_text, decrypt_text
        original = "Привет мир"
        encrypted = encrypt_text(original)
        decrypted = decrypt_text(encrypted)
        assert decrypted == original
    
    def test_encrypt_decrypt_with_password(self):
        """Проверка шифрования с паролем."""
        from encoder import encrypt_text, decrypt_text
        original = "Секретное сообщение"
        password = "мойпароль"
        encrypted = encrypt_text(original, password)
        decrypted = decrypt_text(encrypted, password)
        assert decrypted == original
    
    def test_encrypt_produces_different_output(self):
        """Проверка что шифрование с паролем меняет данные."""
        from encoder import encrypt_text
        original = "Тест"
        encrypted_no_pass = encrypt_text(original)
        encrypted_with_pass = encrypt_text(original, "password")
        assert encrypted_no_pass != encrypted_with_pass


# =============================================================================
# Тесты утилит сигналов (signal_utils)
# =============================================================================

class TestSignalUtils:
    """Тесты модуля signal_utils."""
    
    def test_fftconvolve_full(self):
        """Проверка свертки в режиме 'full'."""
        from signal_utils import fftconvolve
        a = np.array([1, 2, 3], dtype=np.float64)
        b = np.array([0, 1, 0.5], dtype=np.float64)
        result = fftconvolve(a, b, mode='full')
        expected = np.convolve(a, b, mode='full')
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fftconvolve_same(self):
        """Проверка свертки в режиме 'same'."""
        from signal_utils import fftconvolve
        a = np.array([1, 2, 3, 4], dtype=np.float64)
        b = np.array([1, 1], dtype=np.float64)
        result = fftconvolve(a, b, mode='same')
        expected = np.convolve(a, b, mode='same')
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fftconvolve_valid(self):
        """Проверка свертки в режиме 'valid'."""
        from signal_utils import fftconvolve
        a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        b = np.array([1, 1], dtype=np.float64)
        result = fftconvolve(a, b, mode='valid')
        expected = np.convolve(a, b, mode='valid')
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_medfilt_basic(self):
        """Проверка медианного фильтра."""
        from signal_utils import medfilt
        x = np.array([1, 2, 100, 3, 4], dtype=np.float64)  # 100 - выброс
        result = medfilt(x, kernel_size=3)
        # После фильтрации выброс должен быть сглажен
        assert result[2] < 100
    
    def test_medfilt_preserves_length(self):
        """Проверка сохранения длины сигнала после фильтрации."""
        from signal_utils import medfilt
        x = np.random.randn(100)
        result = medfilt(x, kernel_size=5)
        assert len(result) == len(x)
    
    def test_correlate_basic(self):
        """Проверка корреляции."""
        from signal_utils import correlate
        a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2], dtype=np.float64)
        result = correlate(a, v, mode='full')
        # Корреляция должна иметь максимум при совпадении
        assert len(result) > 0
    
    def test_find_peaks_basic(self):
        """Проверка поиска пиков."""
        from signal_utils import find_peaks
        x = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float64)
        peaks, _ = find_peaks(x)
        assert len(peaks) == 3
        assert 1 in peaks
        assert 3 in peaks
        assert 5 in peaks
    
    def test_find_peaks_with_height(self):
        """Проверка поиска пиков с фильтрацией по высоте."""
        from signal_utils import find_peaks
        x = np.array([0, 1, 0, 5, 0, 2, 0], dtype=np.float64)
        peaks, _ = find_peaks(x, height=3)
        assert len(peaks) == 1
        assert peaks[0] == 3
    
    def test_find_peaks_with_distance(self):
        """Проверка поиска пиков с фильтрацией по расстоянию."""
        from signal_utils import find_peaks
        x = np.array([0, 5, 0, 4, 0, 3, 0], dtype=np.float64)
        peaks, _ = find_peaks(x, distance=3)
        # Пики на позициях 1, 3, 5 с расстоянием 2 между ними
        # При distance=3 остаются пики с расстоянием >= 3
        # Пик 1 (pos=1) и пик 3 (pos=3): расстояние 2 < 3, убираем 3
        # Пик 1 (pos=1) и пик 5 (pos=5): расстояние 4 >= 3, оставляем 5
        assert len(peaks) == 2
        assert 1 in peaks
        assert 5 in peaks
    
    def test_firwin_lowpass(self):
        """Проверка проектирования ФНЧ."""
        from signal_utils import firwin
        h = firwin(31, 0.3, window='hamming')
        assert len(h) == 31
        # Симметричность для ФНЧ
        np.testing.assert_array_almost_equal(h, h[::-1])
    
    def test_firwin_highpass(self):
        """Проверка проектирования ФВЧ."""
        from signal_utils import firwin
        h = firwin(31, 0.3, pass_zero=False)
        assert len(h) == 31
    
    def test_butter_lowpass(self):
        """Проверка фильтра Баттерворта ФНЧ."""
        from signal_utils import butter
        b, a = butter(1, 0.3, btype='low')
        assert len(b) == 2
        assert len(a) == 2
        assert a[0] == 1.0
    
    def test_butter_highpass(self):
        """Проверка фильтра Баттерворта ФВЧ."""
        from signal_utils import butter
        b, a = butter(1, 0.3, btype='high')
        assert len(b) == 2
        assert len(a) == 2
    
    def test_freqz_basic(self):
        """Проверка частотной характеристики."""
        from signal_utils import freqz
        b = np.array([1.0, 1.0])
        w, h = freqz(b, worN=64)
        assert len(w) == 64
        assert len(h) == 64


# =============================================================================
# Тесты симулятора канала (channel_simulator)
# =============================================================================

class TestChannelSimulator:
    """Тесты модуля channel_simulator."""
    
    def test_add_awgn(self):
        """Проверка добавления белого гауссовского шума."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        noisy = sim.add_awgn(sig, snr_db=20)
        assert len(noisy) == len(sig)
        # Шум должен изменить сигнал
        assert not np.array_equal(sig, noisy)
    
    def test_add_awgn_infinite_snr(self):
        """Проверка что при бесконечном SNR сигнал не меняется."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.add_awgn(sig, snr_db=float('inf'))
        np.testing.assert_array_almost_equal(sig, result)
    
    def test_apply_multipath(self):
        """Проверка многолучевого распространения."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.apply_multipath(sig, delays_ms=[0, 1], gains=[0.7, 0.3])
        assert len(result) == len(sig)
    
    def test_apply_frequency_offset(self):
        """Проверка сдвига частоты."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.apply_frequency_offset(sig, offset_hz=10)
        assert len(result) == len(sig)
    
    def test_apply_frequency_offset_zero(self):
        """Проверка что нулевой сдвиг не меняет сигнал."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.apply_frequency_offset(sig, offset_hz=0)
        np.testing.assert_array_almost_equal(sig, result)
    
    def test_apply_phase_jitter(self):
        """Проверка фазового джиттера."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.apply_phase_jitter(sig, jitter_std_rad=0.1)
        assert len(result) == len(sig)
    
    def test_apply_hard_clipping(self):
        """Проверка жесткого ограничения."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.linspace(-2, 2, 1000)
        result = sim.apply_hard_clipping(sig, threshold=1.0)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
    
    def test_apply_soft_clipping_tanh(self):
        """Проверка мягкого ограничения (tanh)."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.linspace(-2, 2, 1000)
        result = sim.apply_soft_clipping(sig, threshold=1.0, clip_type='tanh')
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
    
    def test_apply_soft_clipping_cubic(self):
        """Проверка мягкого ограничения (кубическое)."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.linspace(-2, 2, 1000)
        result = sim.apply_soft_clipping(sig, threshold=1.0, clip_type='cubic')
        assert len(result) == len(sig)
    
    def test_apply_fading_rayleigh(self):
        """Проверка замирания Рэлея."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.apply_fading(sig, fader_type='rayleigh', block_size=256)
        assert len(result) == len(sig)
    
    def test_apply_fading_rician(self):
        """Проверка замирания Ричана."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.apply_fading(sig, fader_type='rician', block_size=256)
        assert len(result) == len(sig)
    
    def test_apply_distortion_even(self):
        """Проверка нелинейных искажений (четные)."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.apply_distortion(sig, drive=0.5, type='even')
        assert len(result) == len(sig)
    
    def test_apply_distortion_odd(self):
        """Проверка нелинейных искажений (нечетные)."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000)
        result = sim.apply_distortion(sig, drive=0.5, type='odd')
        assert len(result) == len(sig)
    
    def test_reset_channel_state(self):
        """Проверка сброса состояния канала."""
        from channel_simulator import ChannelSimulator
        sim = ChannelSimulator(fs=48000, seed=42)
        sim._current_freq_response = np.ones(100)
        sim.reset_channel_state()
        assert sim._current_freq_response is None
    
    def test_simulate_channel_function(self):
        """Проверка универсальной функции simulate_channel."""
        from channel_simulator import simulate_channel
        sig = np.sin(2 * np.pi * 1000 * np.arange(1000) / 48000).astype(np.float32)
        result = simulate_channel(
            sig,
            fs=48000,
            snr_db=20,
            freq_offset_hz=5.0,
            multipath_delays=[0.0, 2.0],
            multipath_gains=[0.7, 0.3],
            seed=42
        )
        assert len(result) == len(sig)


# =============================================================================
# Тесты базовой модуляции BPSK/QPSK
# =============================================================================

class TestModulation:
    """Тесты базовой модуляции BPSK/QPSK."""
    
    def test_bpsk_modulation(self):
        """Проверка BPSK модуляции: бит -> символ."""
        # BPSK: 0 -> -1, 1 -> +1
        bits = np.array([0, 1, 0, 1, 1, 0])
        symbols = 2 * bits - 1  # BPSK mapping
        expected = np.array([-1, 1, -1, 1, 1, -1])
        np.testing.assert_array_equal(symbols, expected)
    
    def test_bpsk_demodulation(self):
        """Проверка BPSK демодуляции: символ -> бит."""
        symbols = np.array([-1, 1, -1, 1, 1, -0.5])
        bits = (symbols > 0).astype(int)
        expected = np.array([0, 1, 0, 1, 1, 0])
        np.testing.assert_array_equal(bits, expected)
    
    def test_qpsk_modulation(self):
        """Проверка QPSK модуляции: 2 бита -> символ."""
        # QPSK: 00 -> (1+1j), 01 -> (1-1j), 11 -> (-1-1j), 10 -> (-1+1j)
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        # Группируем по 2 бита
        bit_pairs = bits.reshape(-1, 2)
        # QPSK mapping (Gray coding) - амплитуда sqrt(2) для не-нормированных символов
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): 1 - 1j,
            (1, 1): -1 - 1j,
            (1, 0): -1 + 1j
        }
        symbols = np.array([mapping[tuple(pair)] for pair in bit_pairs])
        assert len(symbols) == 4
        # Проверяем что все символы имеют одинаковую амплитуду sqrt(2)
        expected_amplitude = np.sqrt(2)
        np.testing.assert_array_almost_equal(np.abs(symbols), np.full(4, expected_amplitude))
    
    def test_qpsk_demodulation(self):
        """Проверка QPSK демодуляции: символ -> биты."""
        symbols = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])
        # Демодуляция по знакам реальной и мнимой частей
        bits = np.zeros(len(symbols) * 2, dtype=int)
        for i, s in enumerate(symbols):
            bits[2 * i] = 1 if np.real(s) < 0 else 0
            bits[2 * i + 1] = 1 if np.imag(s) < 0 else 0
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(bits, expected)
    
    def test_bpsk_qpsk_roundtrip(self):
        """Проверка кругового преобразования BPSK/QPSK."""
        # BPSK roundtrip
        original_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        bpsk_symbols = 2 * original_bits - 1
        recovered_bits = (bpsk_symbols > 0).astype(int)
        np.testing.assert_array_equal(original_bits, recovered_bits)
        
        # QPSK roundtrip
        qpsk_bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        bit_pairs = qpsk_bits.reshape(-1, 2)
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): 1 - 1j,
            (1, 1): -1 - 1j,
            (1, 0): -1 + 1j
        }
        symbols = np.array([mapping[tuple(pair)] for pair in bit_pairs])
        # Демодуляция
        recovered = np.zeros(8, dtype=int)
        for i, s in enumerate(symbols):
            recovered[2 * i] = 1 if np.real(s) < 0 else 0
            recovered[2 * i + 1] = 1 if np.imag(s) < 0 else 0
        np.testing.assert_array_equal(qpsk_bits, recovered)
    
    def test_modulation_with_noise(self):
        """Проверка устойчивости модуляции к шуму."""
        # BPSK с шумом
        original_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        symbols = 2 * original_bits - 1
        # Добавляем шум (маленький, чтобы не пересечь границу)
        noisy_symbols = symbols + np.random.randn(len(symbols)) * 0.1
        recovered_bits = (noisy_symbols > 0).astype(int)
        np.testing.assert_array_equal(original_bits, recovered_bits)


# =============================================================================
# Тесты облегченных утилит (lightweight_signal)
# =============================================================================

class TestLightweightSignal:
    """Тесты модуля lightweight_signal."""
    
    def test_fftconvolve(self):
        """Проверка свертки через FFT."""
        from lightweight_signal import fftconvolve
        a = np.array([1, 2, 3], dtype=np.float64)
        b = np.array([0, 1, 0.5], dtype=np.float64)
        result = fftconvolve(a, b)
        expected = np.convolve(a, b)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_find_peaks(self):
        """Проверка поиска пиков."""
        from lightweight_signal import find_peaks
        x = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float64)
        peaks = find_peaks(x)
        assert len(peaks) == 3
    
    def test_find_peaks_with_height(self):
        """Проверка поиска пиков с фильтрацией по высоте."""
        from lightweight_signal import find_peaks
        x = np.array([0, 1, 0, 5, 0, 2, 0], dtype=np.float64)
        peaks = find_peaks(x, height=3)
        assert len(peaks) == 1
    
    def test_correlate(self):
        """Проверка корреляции."""
        from lightweight_signal import correlate
        a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2], dtype=np.float64)
        result = correlate(a, v, mode='valid')
        assert len(result) == 4
    
    def test_medfilt(self):
        """Проверка медианного фильтра."""
        from lightweight_signal import medfilt
        x = np.array([1, 2, 100, 3, 4], dtype=np.float64)
        result = medfilt(x, kernel_size=3)
        assert len(result) == len(x)
        # Выброс должен быть сглажен
        assert result[2] < 100


# =============================================================================
# Тесты констант и параметров
# =============================================================================

class TestConstants:
    """Тесты констант и параметров системы."""
    
    def test_ofdm_parameters(self):
        """Проверка параметров OFDM."""
        import modem_config
        # Проверяем что параметры согласованы
        assert modem_config.Nfft > modem_config.Ncp
        assert modem_config.Nsub > 0
        assert modem_config.fs > 0
    
    def test_rs_parameters(self):
        """Проверка параметров Reed-Solomon."""
        import modem_config
        assert modem_config.RS_DATA_BYTES == 8
        assert modem_config.RS_PARITY_BYTES == 4
        assert modem_config.RS_CW_BYTES == modem_config.RS_DATA_BYTES + modem_config.RS_PARITY_BYTES
        assert modem_config.RS_CW_BITS == modem_config.RS_CW_BYTES * 8
    
    def test_frequency_range(self):
        """Проверка частотного диапазона."""
        import modem_config
        df = modem_config.df
        f_low = modem_config.k_low * df
        f_high = modem_config.k_high * df
        # Диапазон должен быть в слышимом спектре
        # Низкие частоты лучше проходят через динамик/микрофон
        assert f_low >= 180  # Минимум ~187.5 Гц (расширено вниз для лучшего прохождения)
        assert f_high <= modem_config.fs / 2  # Ниже частоты Найквиста


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
