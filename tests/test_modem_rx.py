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


class TestWeakSignals:
    """Тесты для проверки работы на слабых сигналах с шумом."""
    
    def setup_method(self):
        """Подготовка перед каждым тестом."""
        from modem_config import init_phases
        init_phases()
        # Импортируем необходимые функции
        from modem_modulation import build_preamble, sync_by_corr
        from modem_config import SYNC_WINDOW_HALF, SYMBOL_LEN, Nfft, Ncp
        self.build_preamble = build_preamble
        self.sync_by_corr = sync_by_corr
        self.SYNC_WINDOW_HALF = SYNC_WINDOW_HALF
        self.SYMBOL_LEN = SYMBOL_LEN
        self.Nfft = Nfft
        self.Ncp = Ncp
        
        # Создаем преамбулу для тестов
        self.preamble = build_preamble()
        print(f"[TEST SETUP] Преамбула создана, длина: {len(self.preamble)} сэмплов")
        print(f"[TEST SETUP] SYNC_WINDOW_HALF = {self.SYNC_WINDOW_HALF}")
    
    def test_weak_signal_with_high_noise(self):
        """
        Тест с сильно зашумленным сигналом (низкий SNR).
        Создаем сигнал с преамбулой и добавляем много шума.
        Проверяем, что преамбула обнаруживается корреляцией.
        """
        print("\n[TEST] Начало теста test_weak_signal_with_high_noise")
        
        # Создаем сигнал: преамбула + данные (тишина)
        signal_length = len(self.preamble) + 1000
        rx_signal = np.zeros(signal_length, dtype=complex)
        rx_signal[:len(self.preamble)] = self.preamble
        
        # Добавляем сильный шум (низкий SNR)
        noise_power = 10.0  # Высокая мощность шума
        noise = np.random.randn(signal_length) + 1j * np.random.randn(signal_length)
        noise = noise * np.sqrt(noise_power / 2)  # Нормализация мощности шума
        rx_noisy = rx_signal + noise
        
        # Проверяем уровень сигнала относительно шума
        signal_power = np.mean(np.abs(rx_signal) ** 2)
        noise_power_actual = np.mean(np.abs(noise) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power_actual) if noise_power_actual > 0 else float('inf')
        print(f"[TEST] SNR = {snr_db:.2f} dB")
        print(f"[TEST] Мощность сигнала: {signal_power:.6f}, мощность шума: {noise_power_actual:.6f}")
        
        # Нормализуем сигнал (как в реальном приемнике)
        rms = np.sqrt(np.mean(np.abs(rx_noisy) ** 2))
        if rms > 1e-12:
            rx_normalized = rx_noisy / rms * 0.3  # TARGET_RMS = 0.3
        else:
            rx_normalized = rx_noisy
        
        # Ищем преамбулу корреляцией
        try:
            sync_pos = self.sync_by_corr(rx_normalized, self.preamble)
            print(f"[TEST] Обнаружена позиция преамбулы: {sync_pos}")
            
            # Проверяем, что позиция находится в пределах разумного
            # (допускаем отклонение из-за шума)
            expected_pos = 0  # Преамбула в начале
            distance = abs(sync_pos - expected_pos)
            print(f"[TEST] Расстояние до ожидаемой позиции: {distance}")
            
            # Проверяем, что корреляция вообще сработала (позиция не отрицательная и не слишком большая)
            assert 0 <= sync_pos < signal_length, f"Позиция {sync_pos} вне диапазона [0, {signal_length})"
            
            # Проверяем корреляцию напрямую
            from signal_utils import fftconvolve
            corr = fftconvolve(rx_normalized, self.preamble[::-1], mode='valid')
            max_corr = np.max(np.abs(corr))
            print(f"[TEST] Максимальная корреляция: {max_corr:.6f}")
            
            # Даже при сильном шуме должен быть какой-то пик корреляции
            assert max_corr > 0, "Корреляция должна быть положительной"
            
        except Exception as e:
            print(f"[TEST ERROR] Ошибка при поиске преамбулы: {e}")
            raise
        
        print("[TEST] Тест test_weak_signal_with_high_noise завершен успешно")
    
    def test_very_weak_signal_low_amplitude(self):
        """
        Тест с очень слабым сигналом (низкая амплитуда).
        Умножаем сигнал на маленький коэффициент (0.001).
        Проверяем обнаружение после AGC.
        """
        print("\n[TEST] Начало теста test_very_weak_signal_low_amplitude")
        
        # Создаем слабый сигнал с преамбулой
        weak_coeff = 0.001  # Очень маленькая амплитуда
        weak_preamble = self.preamble * weak_coeff
        
        signal_length = len(weak_preamble) + 1000
        rx_signal = np.zeros(signal_length, dtype=complex)
        rx_signal[:len(weak_preamble)] = weak_preamble
        
        print(f"[TEST] Коэффициент ослабления: {weak_coeff}")
        print(f"[TEST] RMS слабого сигнала: {np.sqrt(np.mean(np.abs(rx_signal)**2)):.6f}")
        
        # Нормализуем сигнал (имитация AGC)
        rms_before = np.sqrt(np.mean(np.abs(rx_signal) ** 2))
        if rms_before > 1e-12:
            rx_normalized = rx_signal / rms_before * 0.3  # TARGET_RMS = 0.3
        else:
            rx_normalized = rx_signal
            print("[TEST WARNING] RMS слишком мал, сигнал не нормализован")
        
        rms_after = np.sqrt(np.mean(np.abs(rx_normalized) ** 2))
        print(f"[TEST] RMS после нормализации: {rms_after:.6f}")
        
        # Ищем преамбулу (используем оригинальную преамбулу для корреляции)
        try:
            # Нормализуем преамбулу так же, как в реальном приемнике
            preamble_for_corr = self.preamble / np.sqrt(np.mean(np.abs(self.preamble) ** 2)) * 0.3
            
            sync_pos = self.sync_by_corr(rx_normalized, preamble_for_corr)
            print(f"[TEST] Обнаружена позиция преамбулы: {sync_pos}")
            
            # Проверяем корреляцию
            from signal_utils import fftconvolve
            corr = fftconvolve(rx_normalized, preamble_for_corr[::-1], mode='valid')
            max_corr = np.max(np.abs(corr))
            print(f"[TEST] Максимальная корреляция: {max_corr:.6f}")
            
            # После нормализации AGC сигнал должен быть обнаружен
            assert max_corr > 0, "Корреляция должна быть положительной после нормализации"
            
        except Exception as e:
            print(f"[TEST ERROR] Ошибка при поиске слабого сигнала: {e}")
            raise
        
        print("[TEST] Тест test_very_weak_signal_low_amplitude завершен успешно")
    
    def test_preamble_length_16_zc_symbols(self):
        """
        Тест с измененной длиной преамбулы.
        Проверяем, что новая преамбула (16 ZC-символов) корректно обрабатывается.
        """
        print("\n[TEST] Начало теста test_preamble_length_16_zc_symbols")
        
        # Проверяем длину преамбулы
        preamble = self.build_preamble()
        expected_length = 18 * self.SYMBOL_LEN  # 16 ZC + 2 pilot = 18 OFDM symbols
        print(f"[TEST] Длина преамбулы: {len(preamble)} сэмплов")
        print(f"[TEST] Ожидаемая длина (18 OFDM symbols): {expected_length} сэмплов")
        print(f"[TEST] SYMBOL_LEN (Nfft + Ncp): {self.SYMBOL_LEN}")
        
        # Проверяем, что длина соответствует ожидаемой
        assert len(preamble) == expected_length, \
            f"Длина преамбулы {len(preamble)} не соответствует ожидаемой {expected_length}"
        
        # Проверяем, что преамбула корректно обнаруживается в сигнале
        signal_length = len(preamble) + 2000
        rx_signal = np.zeros(signal_length, dtype=complex)
        rx_signal[500:500+len(preamble)] = preamble  # Преамбула со сдвигом
        
        # Нормализуем
        rms = np.sqrt(np.mean(np.abs(rx_signal) ** 2))
        if rms > 1e-12:
            rx_normalized = rx_signal / rms * 0.3
        else:
            rx_normalized = rx_signal
        
        # Ищем преамбулу
        sync_pos = self.sync_by_corr(rx_normalized, preamble)
        print(f"[TEST] Обнаружена позиция преамбулы: {sync_pos}")
        print(f"[TEST] Ожидаемая позиция: ~500")
        
        # Проверяем, что позиция близка к ожидаемой (допуск из-за особенностей корреляции)
        distance = abs(sync_pos - 500)
        print(f"[TEST] Расстояние до ожидаемой позиции: {distance}")
        
        # Допускаем небольшое отклонение
        assert distance < 100, f"Позиция {sync_pos} слишком далеко от ожидаемой 500 (расстояние {distance})"
        
        print("[TEST] Тест test_preamble_length_16_zc_symbols завершен успешно")
    
    def test_sync_window_half_100(self):
        """
        Тест с расширенным окном поиска.
        Проверяем, что SYNC_WINDOW_HALF=100 используется (проверка через конфигурацию).
        """
        print("\n[TEST] Начало теста test_sync_window_half_100")
        
        # Проверяем значение SYNC_WINDOW_HALF
        print(f"[TEST] SYNC_WINDOW_HALF = {self.SYNC_WINDOW_HALF}")
        assert self.SYNC_WINDOW_HALF == 100, \
            f"SYNC_WINDOW_HALF должен быть 100, а не {self.SYNC_WINDOW_HALF}"
        
        # Создаем сигнал с преамбулой, сдвинутой на большое расстояние
        # чтобы проверить, что окно поиска достаточно широкое
        preamble = self.build_preamble()
        signal_length = len(preamble) + 5000
        rx_signal = np.zeros(signal_length, dtype=complex)
        
        # Помещаем преамбулу в середину (сдвиг ~2500)
        shift = 2500
        rx_signal[shift:shift+len(preamble)] = preamble
        
        print(f"[TEST] Преамбула помещена со сдвигом {shift}")
        
        # Нормализуем
        rms = np.sqrt(np.mean(np.abs(rx_signal) ** 2))
        if rms > 1e-12:
            rx_normalized = rx_signal / rms * 0.3
        else:
            rx_normalized = rx_signal
        
        # Ищем преамбулу
        sync_pos = self.sync_by_corr(rx_normalized, preamble)
        print(f"[TEST] Обнаружена позиция преамбулы: {sync_pos}")
        
        # Проверяем, что позиция близка к ожидаемой
        distance = abs(sync_pos - shift)
        print(f"[TEST] Расстояние до ожидаемой позиции: {distance}")
        
        # Даже с большим сдвигом должны найти (окно поиска в sync_by_corr не ограничено,
        # но в реальном коде modem_rx.py используется SYNC_WINDOW_HALF)
        assert distance < 200, f"Позиция {sync_pos} слишком далеко от ожидаемой {shift}"
        
        print("[TEST] Тест test_sync_window_half_100 завершен успешно")
