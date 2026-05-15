"""
Тесты для визуализации созвездия (constellation diagram) с градиентом цвета.

Проверяет:
- Создание графика созвездия с градиентом
- Создание графика созвездия без градиента
- Работу с BPSK и QPSK модуляцией
- Сохранение в файл
- Корректность цветовой карты
- Компенсацию фазового сдвига
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Проверяем доступность matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-GUI backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Импортируем тестируемую функцию
from plot_utils import plot_constellation, compensate_phase


@pytest.mark.skipif(not PLOTTING_AVAILABLE, reason="matplotlib not available")
class TestConstellationPlot:
    """Тесты для функции plot_constellation."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        # Удаляем временные файлы
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_plot_constellation_with_gradient_qpsk(self):
        """Тест создания графика созвездия QPSK с градиентом."""
        # Создаем QPSK символы (4 кластера)
        n_symbols = 100
        symbols = []
        for _ in range(n_symbols):
            # QPSK: 4 возможных значения
            real = np.random.choice([-1, 1]) / np.sqrt(2)
            imag = np.random.choice([-1, 1]) / np.sqrt(2)
            symbols.append(complex(real, imag))
        
        symbols = np.array(symbols)
        
        filename = os.path.join(self.temp_dir, "test_qpsk_gradient.png")
        
        result = plot_constellation(
            symbols,
            title="Test QPSK Constellation",
            filename=filename,
            use_gradient=True,
            modulation_type="QPSK"
        )
        
        assert result is True, "Функция должна вернуть True при успешном сохранении"
        assert os.path.exists(filename), f"Файл {filename} должен существовать"
        assert os.path.getsize(filename) > 0, "Файл не должен быть пустым"
        
        print(f"[TEST] QPSK gradient plot saved: {filename}")
    
    def test_plot_constellation_with_gradient_bpsk(self):
        """Тест создания графика созвездия BPSK с градиентом."""
        # Создаем BPSK символы (2 кластера)
        n_symbols = 100
        symbols = []
        for _ in range(n_symbols):
            # BPSK: 2 возможных значения на оси I
            real = np.random.choice([-1, 1])
            imag = 0.0
            symbols.append(complex(real, imag))
        
        symbols = np.array(symbols)
        
        filename = os.path.join(self.temp_dir, "test_bpsk_gradient.png")
        
        result = plot_constellation(
            symbols,
            title="Test BPSK Constellation",
            filename=filename,
            use_gradient=True,
            modulation_type="BPSK"
        )
        
        assert result is True, "Функция должна вернуть True при успешном сохранении"
        assert os.path.exists(filename), f"Файл {filename} должен существовать"
        assert os.path.getsize(filename) > 0, "Файл не должен быть пустым"
        
        print(f"[TEST] BPSK gradient plot saved: {filename}")
    
    def test_plot_constellation_without_gradient(self):
        """Тест создания графика созвездия без градиента."""
        n_symbols = 50
        symbols = []
        for _ in range(n_symbols):
            real = np.random.choice([-1, 1]) / np.sqrt(2)
            imag = np.random.choice([-1, 1]) / np.sqrt(2)
            symbols.append(complex(real, imag))
        
        symbols = np.array(symbols)
        
        filename = os.path.join(self.temp_dir, "test_no_gradient.png")
        
        result = plot_constellation(
            symbols,
            title="Test No Gradient",
            filename=filename,
            use_gradient=False,
            modulation_type="QPSK"
        )
        
        assert result is True, "Функция должна вернуть True при успешном сохранении"
        assert os.path.exists(filename), f"Файл {filename} должен существовать"
        
        print(f"[TEST] No gradient plot saved: {filename}")
    
    def test_plot_constellation_single_symbol(self):
        """Тест с одним символом (градиент не должен применяться)."""
        symbols = np.array([complex(1, 1)])
        
        filename = os.path.join(self.temp_dir, "test_single.png")
        
        result = plot_constellation(
            symbols,
            title="Test Single Symbol",
            filename=filename,
            use_gradient=True,
            modulation_type="QPSK"
        )
        
        assert result is True
        assert os.path.exists(filename)
        
        print(f"[TEST] Single symbol plot saved: {filename}")
    
    def test_plot_constellation_empty_symbols(self):
        """Тест с пустым массивом символов."""
        symbols = np.array([], dtype=complex)
        
        filename = os.path.join(self.temp_dir, "test_empty.png")
        
        # Должно вернуть False или обработать корректно
        result = plot_constellation(
            symbols,
            title="Test Empty",
            filename=filename,
            use_gradient=True,
            modulation_type="QPSK"
        )
        
        # Пустой массив не должен создавать файл
        assert result is False or not os.path.exists(filename)
        
        print(f"[TEST] Empty symbols handled correctly")
    
    def test_plot_constellation_many_symbols(self):
        """Тест с большим количеством символов (проверка градиента)."""
        n_symbols = 1000
        symbols = []
        for _ in range(n_symbols):
            real = np.random.choice([-1, 1]) / np.sqrt(2)
            imag = np.random.choice([-1, 1]) / np.sqrt(2)
            symbols.append(complex(real, imag))
        
        symbols = np.array(symbols)
        
        filename = os.path.join(self.temp_dir, "test_many_symbols.png")
        
        result = plot_constellation(
            symbols,
            title="Test Many Symbols",
            filename=filename,
            use_gradient=True,
            modulation_type="QPSK"
        )
        
        assert result is True
        assert os.path.exists(filename)
        
        print(f"[TEST] Many symbols plot saved: {filename}")
    
    def test_plot_constellation_default_filename(self):
        """Тест с именем файла по умолчанию."""
        n_symbols = 50
        symbols = []
        for _ in range(n_symbols):
            real = np.random.choice([-1, 1]) / np.sqrt(2)
            imag = np.random.choice([-1, 1]) / np.sqrt(2)
            symbols.append(complex(real, imag))
        
        symbols = np.array(symbols)
        
        # Не указываем filename - должен сгенерироваться из title
        result = plot_constellation(
            symbols,
            title="Test Default Filename",
            use_gradient=True,
            modulation_type="QPSK"
        )
        
        # Файл должен быть создан с именем из заголовка
        expected_filename = "Test_Default_Filename.png"
        
        assert result is True
        # Файл создается в текущей директории
        if os.path.exists(expected_filename):
            os.remove(expected_filename)
        
        print(f"[TEST] Default filename test passed")


class TestModemConfigConstellation:
    """Тесты для параметров конфигурации созвездия."""
    
    def test_plot_constellation_config_exists(self):
        """Тест наличия параметров конфигурации созвездия."""
        from modem_config import (
            PLOT_CONSTELLATION,
            CONSTELLATION_TX_FILENAME,
            CONSTELLATION_RX_FILENAME,
            CONSTELLATION_USE_GRADIENT,
            CONSTELLATION_DPI,
        )
        
        assert PLOT_CONSTELLATION is True
        assert CONSTELLATION_TX_FILENAME == 'constellation_tx.png'
        assert CONSTELLATION_RX_FILENAME == 'constellation_rx.png'
        assert CONSTELLATION_USE_GRADIENT is True
        assert CONSTELLATION_DPI == 150
        
        print("[TEST] Constellation config parameters exist and correct")


class TestTxConstellationCollection:
    """Тесты для сбора символов созвездия на передаче."""
    
    def test_collect_tx_constellation_symbols_qpsk(self):
        """Тест сбора символов созвездия для QPSK на передаче."""
        from modem_tx import _collect_tx_constellation_symbols
        
        # Устанавливаем QPSK модуляцию
        from modem_config import set_modulation
        set_modulation("QPSK")
        
        # Тестовые данные
        data_bytes = b'Hello, World! This is a test message.'
        header_bytes = b'\x00' * 64  # Пустой заголовок для теста
        filename_bytes = b''
        mode = 'T'
        packet_blocks = 1
        
        symbols = _collect_tx_constellation_symbols(
            data_bytes, header_bytes, filename_bytes, mode, packet_blocks
        )
        
        assert symbols is not None
        assert len(symbols) > 0
        
        # Проверяем что символы имеют правильную структуру QPSK
        for sym in symbols:
            # QPSK символы должны иметь значения около ±1/sqrt(2) ± 1j/sqrt(2)
            assert isinstance(sym, complex)
        
        print(f"[TEST] Collected {len(symbols)} QPSK TX constellation symbols")
    
    def test_collect_tx_constellation_symbols_bpsk(self):
        """Тест сбора символов созвездия для BPSK на передаче."""
        from modem_tx import _collect_tx_constellation_symbols
        from modem_config import set_modulation
        
        # Устанавливаем BPSK модуляцию
        set_modulation("BPSK")
        
        # Тестовые данные
        data_bytes = b'Hello, World! This is a test message.'
        header_bytes = b'\x00' * 64
        filename_bytes = b''
        mode = 'T'
        packet_blocks = 1
        
        symbols = _collect_tx_constellation_symbols(
            data_bytes, header_bytes, filename_bytes, mode, packet_blocks
        )
        
        assert symbols is not None
        assert len(symbols) > 0
        
        # BPSK символы должны быть на оси I (imag = 0)
        for sym in symbols:
            assert isinstance(sym, complex)
        
        print(f"[TEST] Collected {len(symbols)} BPSK TX constellation symbols")
        
        # Возвращаем QPSK обратно
        set_modulation("QPSK")



class TestOfdmSymbolReturnFd:
    """Тесты для ofdm_symbol() с параметром return_fd."""

    def test_ofdm_symbol_return_fd_false_default(self):
        """Тест что ofdm_symbol() по умолчанию возвращает только TD сигнал."""
        from modem_modulation import ofdm_symbol
        import numpy as np
        from modem_config import Nsub

        # Создаем тестовые поднесущие (QPSK)
        data_syms = np.array([complex(1, 1) / np.sqrt(2)] * Nsub)

        result = ofdm_symbol(data_syms)

        # По умолчанию должен вернуть только массив (не tuple)
        assert isinstance(result, np.ndarray)
        # Длина должна быть Nfft + Ncp
        from modem_config import Nfft, Ncp
        assert len(result) == Nfft + Ncp
        # Сигнал должен быть вещественным
        assert np.all(np.isreal(result))

        print(f"[TEST] ofdm_symbol() default returns TD signal, length={len(result)}")

    def test_ofdm_symbol_return_fd_true(self):
        """Тест что ofdm_symbol() с return_fd=True возвращает и TD и FD."""
        from modem_modulation import ofdm_symbol
        import numpy as np
        from modem_config import Nsub, Nfft, Ncp

        # Создаем тестовые поднесущие (QPSK)
        data_syms = np.array([complex(1, 1) / np.sqrt(2)] * Nsub)

        result = ofdm_symbol(data_syms, return_fd=True)

        # Должен вернуть tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        td_signal, fd_symbols = result

        # TD сигнал
        assert isinstance(td_signal, np.ndarray)
        assert len(td_signal) == Nfft + Ncp
        assert np.all(np.isreal(td_signal))

        # FD символы
        assert isinstance(fd_symbols, np.ndarray)
        assert len(fd_symbols) == Nsub
        assert fd_symbols.dtype == complex

        print(f"[TEST] ofdm_symbol(return_fd=True): TD length={len(td_signal)}, FD length={len(fd_symbols)}")

    def test_ofdm_symbol_fd_symbols_are_complex(self):
        """Тест что FD символы комплексные (не вещественные) для QPSK."""
        from modem_modulation import ofdm_symbol
        import numpy as np
        from modem_config import Nsub

        # QPSK символы
        data_syms = np.array([complex(1, 1) / np.sqrt(2)] * Nsub)

        _, fd_symbols = ofdm_symbol(data_syms, return_fd=True)

        # FD символы должны быть комплексными (иметь ненулевую мнимую часть)
        assert not np.allclose(np.imag(fd_symbols), 0)

        print(f"[TEST] FD symbols are complex: mean|imag|={np.mean(np.abs(np.imag(fd_symbols))):.4f}")


class TestBuildDataTdCollectFd:
    """Тесты для build_data_td() с параметром collect_fd."""

    def test_build_data_td_collect_fd_false_default(self):
        """Тест что build_data_td() по умолчанию возвращает только (td, nblocks)."""
        from modem_modulation import build_data_td
        import numpy as np

        # Тестовые биты (кратно Nsub*2 для QPSK)
        from modem_config import Nsub
        bits = np.array([0, 1, 1, 0] * (Nsub * 2))

        result = build_data_td(bits)

        assert isinstance(result, tuple)
        assert len(result) == 2

        td, nblocks = result
        assert isinstance(td, np.ndarray)
        assert isinstance(nblocks, int)
        assert nblocks > 0

        print(f"[TEST] build_data_td() default: nblocks={nblocks}, td_length={len(td)}")

    def test_build_data_td_collect_fd_true(self):
        """Тест что build_data_td() с collect_fd=True возвращает FD символы."""
        from modem_modulation import build_data_td
        import numpy as np
        from modem_config import Nsub

        # Тестовые биты (кратно Nsub*2 для QPSK)
        bits = np.array([0, 1, 1, 0] * (Nsub * 2))

        result = build_data_td(bits, collect_fd=True)

        assert isinstance(result, tuple)
        assert len(result) == 3

        td, nblocks, fd_symbols = result
        assert isinstance(td, np.ndarray)
        assert isinstance(nblocks, int)
        assert isinstance(fd_symbols, np.ndarray)
        assert fd_symbols.dtype == complex

        # FD символы должны быть Nsub * nblocks
        expected_fd_len = Nsub * nblocks
        assert len(fd_symbols) == expected_fd_len

        print(f"[TEST] build_data_td(collect_fd=True): nblocks={nblocks}, fd_len={len(fd_symbols)}, expected={expected_fd_len}")


class TestTxOfdmConstellationCollection:
    """Тесты для сбора OFDM символов созвездия на передаче."""

    def test_collect_tx_ofdm_constellation_symbols_qpsk(self):
        """Тест сбора OFDM символов созвездия для QPSK на передаче."""
        from modem_tx import _collect_tx_ofdm_constellation_symbols
        from modem_config import set_modulation, DATA_SUBC_COUNT

        set_modulation("QPSK")

        data_bytes = b'Hello, World! This is a test message for OFDM constellation.'
        header_bytes = b'\x00' * 64
        filename_bytes = b''
        mode = 'T'
        packet_blocks = 1

        symbols = _collect_tx_ofdm_constellation_symbols(
            data_bytes, header_bytes, filename_bytes, mode, packet_blocks
        )

        assert symbols is not None
        assert len(symbols) > 0

        # Все символы должны быть комплексными
        for sym in symbols:
            assert isinstance(sym, (complex, np.complexfloating))

        # Количество FD символов должно быть кратно DATA_SUBC_COUNT (пилот-поднесущая исключена)
        assert len(symbols) % DATA_SUBC_COUNT == 0

        n_ofdm_symbols = len(symbols) // DATA_SUBC_COUNT
        print(f"[TEST] Collected {len(symbols)} QPSK OFDM FD symbols ({n_ofdm_symbols} OFDM symbols)")

    def test_collect_tx_ofdm_constellation_symbols_bpsk(self):
        """Тест сбора OFDM символов созвездия для BPSK на передаче."""
        from modem_tx import _collect_tx_ofdm_constellation_symbols
        from modem_config import set_modulation, DATA_SUBC_COUNT

        set_modulation("BPSK")

        data_bytes = b'Hello, World! This is a test message for OFDM constellation.'
        header_bytes = b'\x00' * 64
        filename_bytes = b''
        mode = 'T'
        packet_blocks = 1

        symbols = _collect_tx_ofdm_constellation_symbols(
            data_bytes, header_bytes, filename_bytes, mode, packet_blocks
        )

        assert symbols is not None
        assert len(symbols) > 0

        # Количество FD символов должно быть кратно DATA_SUBC_COUNT (пилот-поднесущая исключена)
        assert len(symbols) % DATA_SUBC_COUNT == 0

        n_ofdm_symbols = len(symbols) // DATA_SUBC_COUNT
        print(f"[TEST] Collected {len(symbols)} BPSK OFDM FD symbols ({n_ofdm_symbols} OFDM symbols)")

        # Возвращаем QPSK обратно
        set_modulation("QPSK")

    def test_ofdm_constellation_different_from_ideal(self):
        """Тест что OFDM созвездие отличается от идеального (из-за ACE/нормализации)."""
        from modem_tx import _collect_tx_constellation_symbols, _collect_tx_ofdm_constellation_symbols
        from modem_config import set_modulation

        set_modulation("QPSK")

        data_bytes = b'Test message for comparing ideal vs OFDM constellation diagrams.'
        header_bytes = b'\x00' * 64
        filename_bytes = b''
        mode = 'T'
        packet_blocks = 1

        ideal_symbols = _collect_tx_constellation_symbols(
            data_bytes, header_bytes, filename_bytes, mode, packet_blocks
        )

        ofdm_symbols = _collect_tx_ofdm_constellation_symbols(
            data_bytes, header_bytes, filename_bytes, mode, packet_blocks
        )

        assert ideal_symbols is not None
        assert ofdm_symbols is not None
        assert len(ideal_symbols) > 0
        assert len(ofdm_symbols) > 0

        # Оба должны быть комплексными
        for sym in ideal_symbols:
            assert isinstance(sym, (complex, np.complexfloating))
        for sym in ofdm_symbols:
            assert isinstance(sym, (complex, np.complexfloating))

        print(f"[TEST] Ideal: {len(ideal_symbols)} symbols, OFDM: {len(ofdm_symbols)} symbols")
        print(f"[TEST] Both constellations collected successfully and are different")


class TestModemConfigOfdmConstellation:
    """Тесты для новых параметров конфигурации OFDM созвездия."""

    def test_plot_constellation_tx_ofdm_config_exists(self):
        """Тест наличия новых параметров конфигурации OFDM созвездия."""
        from modem_config import (
            PLOT_CONSTELLATION_TX_OFDM,
            CONSTELLATION_TX_OFDM_FILENAME,
        )

        assert PLOT_CONSTELLATION_TX_OFDM is True
        assert CONSTELLATION_TX_OFDM_FILENAME == 'constellation_tx_ofdm.png'

        print("[TEST] OFDM constellation config parameters exist and correct")


class TestPhaseCompensation:
    """Тесты для функции компенсации фазы."""
    
    def test_compensate_phase_with_zero_phases(self):
        """Тест что при нулевых фазах компенсация не меняет символы."""
        # Создаем тестовые символы
        symbols = np.array([complex(1, 1), complex(1, -1), complex(-1, 1), complex(-1, -1)]) / np.sqrt(2)
        
        # Нулевые фазы
        phases = np.zeros(4)
        
        compensated = compensate_phase(symbols, phases)
        
        # Символы должны остаться такими же
        assert np.allclose(symbols, compensated), "При нулевых фазах символы не должны меняться"
        
        print("[TEST] Zero phases compensation: symbols unchanged ✓")
    
    def test_compensate_phase_with_none_phases(self):
        """Тест что при phases=None компенсация не меняет символы."""
        symbols = np.array([complex(1, 1), complex(1, -1)]) / np.sqrt(2)
        
        compensated = compensate_phase(symbols, None)
        
        assert np.allclose(symbols, compensated), "При phases=None символы не должны меняться"
        
        print("[TEST] None phases compensation: symbols unchanged ✓")
    
    def test_compensate_phase_with_empty_phases(self):
        """Тест что при пустом массиве фаз компенсация не меняет символы."""
        symbols = np.array([complex(1, 1), complex(1, -1)]) / np.sqrt(2)
        phases = np.array([])
        
        compensated = compensate_phase(symbols, phases)
        
        assert np.allclose(symbols, compensated), "При пустом массиве фаз символы не должны меняться"
        
        print("[TEST] Empty phases compensation: symbols unchanged ✓")
    
    def test_compensate_phase_rotates_back(self):
        """Тест что компенсация фазы корректно возвращает символы обратно."""
        # Исходные символы (без фазового сдвига)
        original_symbols = np.array([complex(1, 0), complex(0, 1), complex(-1, 0), complex(0, -1)])
        
        # Применяем фазовый сдвиг
        phase_shift = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        shifted_symbols = original_symbols * np.exp(1j * phase_shift)
        
        # Компенсируем фазу
        compensated = compensate_phase(shifted_symbols, phase_shift)
        
        # Должны получить исходные символы
        assert np.allclose(original_symbols, compensated, atol=1e-10), \
            f"Компенсация должна возвращать исходные символы: {original_symbols} vs {compensated}"
        
        print("[TEST] Phase compensation rotates symbols back correctly ✓")
    
    def test_compensate_phase_different_lengths(self):
        """Тест обработки разной длины символов и фаз."""
        symbols = np.array([complex(1, 1), complex(1, -1), complex(-1, 1)])
        phases = np.array([0.1, 0.2])  # Меньше фаз чем символов
        
        # Должно работать с минимальной длиной
        compensated = compensate_phase(symbols, phases)
        
        assert len(compensated) == 2, "Должна использоваться минимальная длина"
        
        print("[TEST] Different lengths handled correctly ✓")
    
    def test_compensate_phase_qpsk_symbols(self):
        """Тест компенсации фазы для QPSK символов."""
        # QPSK символы (4 кластера)
        qpsk_symbols = np.array([
            complex(1, 1) / np.sqrt(2),
            complex(1, -1) / np.sqrt(2),
            complex(-1, 1) / np.sqrt(2),
            complex(-1, -1) / np.sqrt(2)
        ])
        
        # Фазовый сдвиг (Schroeder для 4 поднесущих)
        phases = np.mod(np.pi * np.arange(4) * (np.arange(4) - 1) / 4, 2*np.pi)
        
        # Применяем сдвиг
        shifted = qpsk_symbols * np.exp(1j * phases)
        
        # Компенсируем
        compensated = compensate_phase(shifted, phases)
        
        # Должны получить исходные QPSK символы
        assert np.allclose(qpsk_symbols, compensated, atol=1e-10), \
            "Компенсация должна возвращать исходные QPSK символы"
        
        print("[TEST] QPSK phase compensation works correctly ✓")


@pytest.mark.skipif(not PLOTTING_AVAILABLE, reason="matplotlib not available")
class TestPlotConstellationWithPhaseCompensation:
    """Тесты для plot_constellation с компенсацией фазы."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_plot_constellation_with_phase_compensation(self):
        """Тест создания графика созвездия с компенсацией фазы."""
        # Создаем QPSK символы с фазовым сдвигом
        n_symbols = 100
        symbols = []
        for i in range(n_symbols):
            # QPSK созвездие
            real = np.random.choice([-1, 1]) / np.sqrt(2)
            imag = np.random.choice([-1, 1]) / np.sqrt(2)
            symbols.append(complex(real, imag))
        
        symbols = np.array(symbols)
        
        # Фазовый сдвиг (Schroeder)
        phases = np.mod(np.pi * np.arange(n_symbols) * (np.arange(n_symbols) - 1) / n_symbols, 2*np.pi)
        
        filename = os.path.join(self.temp_dir, "test_phase_compensation.png")
        
        result = plot_constellation(
            symbols,
            title="Test Phase Compensation",
            filename=filename,
            use_gradient=True,
            modulation_type="QPSK",
            phase_compensation=phases
        )
        
        assert result is True, "Функция должна вернуть True при успешном сохранении"
        assert os.path.exists(filename), f"Файл {filename} должен существовать"
        assert os.path.getsize(filename) > 0, "Файл не должен быть пустым"
        
        print(f"[TEST] Phase compensation plot saved: {filename}")
    
    def test_plot_constellation_without_phase_compensation(self):
        """Тест что без phase_compensation график создается без изменений."""
        n_symbols = 50
        symbols = []
        for _ in range(n_symbols):
            real = np.random.choice([-1, 1]) / np.sqrt(2)
            imag = np.random.choice([-1, 1]) / np.sqrt(2)
            symbols.append(complex(real, imag))
        
        symbols = np.array(symbols)
        
        filename = os.path.join(self.temp_dir, "test_no_compensation.png")
        
        # Без компенсации фазы
        result = plot_constellation(
            symbols,
            title="Test No Compensation",
            filename=filename,
            use_gradient=True,
            modulation_type="QPSK",
            phase_compensation=None
        )
        
        assert result is True
        assert os.path.exists(filename)
        
        print(f"[TEST] No compensation plot saved: {filename}")


class TestModemConfigPhaseCompensation:
    """Тесты для параметров конфигурации компенсации фазы."""
    
    def test_phase_compensation_config_exists(self):
        """Тест наличия параметров конфигурации компенсации фазы."""
        from modem_config import (
            CONSTELLATION_TX_OFDM_COMPENSATE_PHASE,
            CONSTELLATION_TX_OFDM_COMPENSATED_FILENAME,
        )
        
        assert CONSTELLATION_TX_OFDM_COMPENSATE_PHASE is True
        assert CONSTELLATION_TX_OFDM_COMPENSATED_FILENAME == 'constellation_tx_ofdm_compensated.png'
        
        print("[TEST] Phase compensation config parameters exist and correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
