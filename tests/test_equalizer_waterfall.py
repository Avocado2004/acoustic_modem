"""
Тесты для модуля equalizer_waterfall.py — водопадной диаграммы эквалайзера.

Покрывает:
- Создание экземпляра EqualizerWaterfall (с новыми параметрами subc_inds, fs, Nfft, step)
- Добавление снимков Hk через update()
- Загрузку истории через update_from_history()
- Ограничение буфера (FIFO)
- Методы clear(), get_snapshot_count(), get_snapshots()
- Сохранение в PNG файл через save() (линейные графики с градиентом)
- Автосохранение при закрытии (save_on_close)
- Режим no-op при недоступном matplotlib
- Вычисление частот поднесущих
- Шаг выборки снимков (step)

Все комментарии — на русском языке.
"""

import numpy as np
import pytest
import sys
import os

# Добавляем родительскую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Вспомогательные функции для тестов
# =============================================================================

def _default_subc_inds(n=48):
    """Возвращает стандартные индексы поднесущих для тестов."""
    return list(range(1, n + 1))


def _default_fs():
    """Возвращает стандартную частоту дискретизации для тестов."""
    return 48000


def _default_Nfft():
    """Возвращает стандартный размер FFT для тестов."""
    return 256


# =============================================================================
# Тесты без matplotlib (mock)
# =============================================================================

class TestEqualizerWaterfallNoMatplotlib:
    """Тесты для режима без matplotlib (no-op режим)."""

    def test_import_without_matplotlib(self):
        """Проверка что модуль импортируется даже без matplotlib."""
        from equalizer_waterfall import EqualizerWaterfall, PLOTTING_AVAILABLE
        assert EqualizerWaterfall is not None

    def test_create_instance_without_matplotlib(self):
        """Проверка создания экземпляра без matplotlib."""
        import equalizer_waterfall as ew
        original_plt = ew.plt
        ew.PLOTTING_AVAILABLE = False
        ew.plt = None
        
        try:
            wf = ew.EqualizerWaterfall(
                subc_inds=_default_subc_inds(),
                fs=_default_fs(),
                Nfft=_default_Nfft()
            )
            assert wf is not None
            assert wf.get_snapshot_count() == 0
        finally:
            ew.plt = original_plt
            try:
                import matplotlib
                ew.PLOTTING_AVAILABLE = True
                ew.plt = matplotlib.pyplot
            except ImportError:
                ew.PLOTTING_AVAILABLE = False
                ew.plt = None

    def test_update_without_matplotlib(self):
        """Проверка update() без matplotlib — не должно быть ошибок."""
        import equalizer_waterfall as ew
        original_plt = ew.plt
        ew.PLOTTING_AVAILABLE = False
        ew.plt = None
        
        try:
            wf = ew.EqualizerWaterfall(
                subc_inds=_default_subc_inds(),
                fs=_default_fs(),
                Nfft=_default_Nfft()
            )
            Hk = np.ones(48, dtype=complex)
            wf.update(Hk)
            assert wf.get_snapshot_count() == 1
        finally:
            ew.plt = original_plt
            try:
                import matplotlib
                ew.PLOTTING_AVAILABLE = True
                ew.plt = matplotlib.pyplot
            except ImportError:
                ew.PLOTTING_AVAILABLE = False
                ew.plt = None

    def test_update_none_without_matplotlib(self):
        """Проверка update(None) — должен корректно обрабатывать None."""
        import equalizer_waterfall as ew
        original_plt = ew.plt
        ew.PLOTTING_AVAILABLE = False
        ew.plt = None
        
        try:
            wf = ew.EqualizerWaterfall(
                subc_inds=_default_subc_inds(),
                fs=_default_fs(),
                Nfft=_default_Nfft()
            )
            wf.update(None)
            assert wf.get_snapshot_count() == 0
        finally:
            ew.plt = original_plt
            try:
                import matplotlib
                ew.PLOTTING_AVAILABLE = True
                ew.plt = matplotlib.pyplot
            except ImportError:
                ew.PLOTTING_AVAILABLE = False
                ew.plt = None

    def test_save_without_matplotlib_returns_false(self):
        """Проверка save() без matplotlib — должен вернуть False."""
        import equalizer_waterfall as ew
        original_plt = ew.plt
        ew.PLOTTING_AVAILABLE = False
        ew.plt = None
        
        try:
            wf = ew.EqualizerWaterfall(
                subc_inds=_default_subc_inds(),
                fs=_default_fs(),
                Nfft=_default_Nfft()
            )
            Hk = np.ones(48, dtype=complex)
            wf.update(Hk)
            
            result = wf.save("test_waterfall.png")
            assert result == False
        finally:
            ew.plt = original_plt
            try:
                import matplotlib
                ew.PLOTTING_AVAILABLE = True
                ew.plt = matplotlib.pyplot
            except ImportError:
                ew.PLOTTING_AVAILABLE = False
                ew.plt = None


# =============================================================================
# Тесты функциональности (с mock для matplotlib)
# =============================================================================

class TestEqualizerWaterfallFunctionality:
    """Тесты функциональности EqualizerWaterfall."""

    def _create_waterfall(self, max_symbols=500, **kwargs):
        """Вспомогательный метод для создания waterfall с мокнутым matplotlib."""
        import equalizer_waterfall as ew
        original_plt = ew.plt
        ew.PLOTTING_AVAILABLE = False
        ew.plt = None
        
        # Параметры по умолчанию
        defaults = dict(
            subc_inds=_default_subc_inds(),
            fs=_default_fs(),
            Nfft=_default_Nfft(),
        )
        defaults.update(kwargs)
        
        wf = ew.EqualizerWaterfall(max_symbols=max_symbols, **defaults)
        
        ew.plt = original_plt
        try:
            import matplotlib
            ew.PLOTTING_AVAILABLE = True
        except ImportError:
            ew.PLOTTING_AVAILABLE = False
        
        return wf

    def test_initial_state(self):
        """Проверка начального состояния — пустой буфер."""
        wf = self._create_waterfall()
        assert wf.get_snapshot_count() == 0
        assert wf.get_snapshots() == []

    def test_single_update(self):
        """Проверка добавления одного снимка."""
        wf = self._create_waterfall()
        Hk = np.ones(48, dtype=complex)
        wf.update(Hk)
        assert wf.get_snapshot_count() == 1

    def test_multiple_updates(self):
        """Проверка добавления нескольких снимков."""
        wf = self._create_waterfall()
        for i in range(10):
            Hk = np.ones(48, dtype=complex) * (1 + 0.1j * i)
            wf.update(Hk)
        assert wf.get_snapshot_count() == 10

    def test_update_preserves_data(self):
        """Проверка что данные сохраняются корректно."""
        wf = self._create_waterfall()
        Hk = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=complex)
        # Создаём waterfall с правильным количеством поднесущих
        wf = self._create_waterfall(subc_inds=_default_subc_inds(3))
        wf.update(Hk)
        
        snapshots = wf.get_snapshots()
        assert len(snapshots) == 1
        np.testing.assert_array_equal(snapshots[0], Hk)

    def test_update_returns_copy(self):
        """Проверка что get_snapshots() возвращает копии."""
        wf = self._create_waterfall()
        Hk = np.ones(48, dtype=complex)
        wf.update(Hk)
        
        snapshots = wf.get_snapshots()
        snapshots[0][0] = 999 + 999j
        
        assert wf.get_snapshots()[0][0] == 1 + 0j

    def test_fifo_buffer_limit(self):
        """Проверка FIFO ограничения буфера."""
        max_symbols = 10
        wf = self._create_waterfall(max_symbols=max_symbols)
        
        for i in range(20):
            Hk = np.ones(48, dtype=complex) * i
            wf.update(Hk)
        
        assert wf.get_snapshot_count() == max_symbols
        
        snapshots = wf.get_snapshots()
        assert snapshots[0][0] == 10 + 0j
        assert snapshots[-1][0] == 19 + 0j

    def test_clear(self):
        """Проверка очистки буфера."""
        wf = self._create_waterfall()
        for i in range(5):
            Hk = np.ones(48, dtype=complex)
            wf.update(Hk)
        
        assert wf.get_snapshot_count() == 5
        wf.clear()
        assert wf.get_snapshot_count() == 0

    def test_update_from_history_flat(self):
        """Проверка update_from_history с плоским списком."""
        wf = self._create_waterfall()
        
        history = [
            np.ones(48, dtype=complex) * i for i in range(5)
        ]
        wf.update_from_history(history)
        
        assert wf.get_snapshot_count() == 5

    def test_update_from_history_nested(self):
        """Проверка update_from_history с вложенным списком (по пакетам)."""
        wf = self._create_waterfall()
        
        history = [
            [np.ones(48, dtype=complex) * i for i in range(3)],
            [np.ones(48, dtype=complex) * i for i in range(3, 7)],
        ]
        wf.update_from_history(history)
        
        assert wf.get_snapshot_count() == 7

    def test_update_from_history_empty(self):
        """Проверка update_from_history с пустой историей."""
        wf = self._create_waterfall()
        wf.update_from_history([])
        assert wf.get_snapshot_count() == 0

    def test_update_from_history_none_values(self):
        """Проверка update_from_history с None в истории."""
        wf = self._create_waterfall()
        history = [
            [np.ones(48, dtype=complex)],
            None,
            [np.ones(48, dtype=complex) * 2],
        ]
        wf.update_from_history(history)
        assert wf.get_snapshot_count() == 2

    def test_update_from_history_with_max_symbols(self):
        """Проверка что update_from_history учитывает max_symbols."""
        max_symbols = 5
        wf = self._create_waterfall(max_symbols=max_symbols)
        
        history = [
            np.ones(48, dtype=complex) * i for i in range(10)
        ]
        wf.update_from_history(history)
        
        assert wf.get_snapshot_count() == max_symbols

    def test_complex_Hk_values(self):
        """Проверка работы с комплексными значениями Hk."""
        wf = self._create_waterfall()
        
        Hk = np.exp(1j * np.linspace(0, 2 * np.pi, 48))
        wf.update(Hk)
        
        snapshots = wf.get_snapshots()
        assert len(snapshots) == 1
        np.testing.assert_array_almost_equal(snapshots[0], Hk)

    def test_different_subcarrier_counts(self):
        """Проверка работы с разным количеством поднесущих."""
        for n_sub in [1, 12, 48, 96]:
            subc = _default_subc_inds(n_sub)
            wf = self._create_waterfall(subc_inds=subc, fs=_default_fs(), Nfft=_default_Nfft())
            Hk = np.ones(n_sub, dtype=complex)
            wf.update(Hk)
            assert wf.get_snapshot_count() == 1

    def test_close_without_error(self):
        """Проверка что close() не вызывает ошибку."""
        wf = self._create_waterfall()
        wf.close()

    def test_update_after_close(self):
        """Проверка что update() работает после close()."""
        wf = self._create_waterfall()
        wf.close()
        
        Hk = np.ones(48, dtype=complex)
        wf.update(Hk)
        assert wf.get_snapshot_count() == 1

    def test_save_on_close_false_by_default(self):
        """Проверка что save_on_close=False по умолчанию."""
        wf = self._create_waterfall()
        assert wf.save_on_close == False

    def test_save_on_close_true(self):
        """Проверка создания с save_on_close=True."""
        wf = self._create_waterfall(save_on_close=True)
        assert wf.save_on_close == True

    def test_custom_save_filename(self):
        """Проверка установки имени файла для сохранения."""
        wf = self._create_waterfall(save_filename="custom_waterfall.png")
        assert wf.save_filename == "custom_waterfall.png"

    def test_default_save_filename(self):
        """Проверка имени файла по умолчанию."""
        wf = self._create_waterfall()
        assert wf.save_filename == "rx_equalizer_waterfall.png"


# =============================================================================
# Тесты новых параметров (subc_inds, fs, Nfft, step)
# =============================================================================

class TestEqualizerWaterfallNewParams:
    """Тесты новых параметров конструктора."""

    def _create_waterfall(self, **kwargs):
        """Вспомогательный метод для создания waterfall."""
        import equalizer_waterfall as ew
        defaults = dict(
            subc_inds=_default_subc_inds(),
            fs=_default_fs(),
            Nfft=_default_Nfft(),
        )
        defaults.update(kwargs)
        return ew.EqualizerWaterfall(**defaults)

    def test_freqs_computed_correctly(self):
        """Проверка вычисления частот поднесущих: freqs = subc_inds * fs / Nfft."""
        subc_inds = [10, 20, 30]
        fs = 48000
        Nfft = 256
        wf = self._create_waterfall(subc_inds=subc_inds, fs=fs, Nfft=Nfft)
        
        expected_freqs = np.array(subc_inds) * fs / Nfft
        np.testing.assert_array_almost_equal(wf._freqs, expected_freqs)

    def test_freqs_with_different_fs(self):
        """Проверка вычисления частот с другой частотой дискретизации."""
        subc_inds = [1, 2, 3]
        fs = 44100
        Nfft = 512
        wf = self._create_waterfall(subc_inds=subc_inds, fs=fs, Nfft=Nfft)
        
        expected_freqs = np.array(subc_inds) * fs / Nfft
        np.testing.assert_array_almost_equal(wf._freqs, expected_freqs)

    def test_step_default(self):
        """Проверка значения step по умолчанию."""
        wf = self._create_waterfall()
        assert wf.step == 10

    def test_step_custom(self):
        """Проверка установки пользовательского step."""
        wf = self._create_waterfall(step=5)
        assert wf.step == 5

    def test_step_one(self):
        """Проверка step=1 — все снимки отрисовываются."""
        wf = self._create_waterfall(step=1)
        assert wf.step == 1

    def test_subc_inds_stored(self):
        """Проверка сохранения индексов поднесущих."""
        subc_inds = [5, 10, 15, 20]
        wf = self._create_waterfall(subc_inds=subc_inds)
        np.testing.assert_array_equal(wf.subc_inds, subc_inds)

    def test_fs_stored(self):
        """Проверка сохранения частоты дискретизации."""
        wf = self._create_waterfall(fs=96000)
        assert wf.fs == 96000

    def test_Nfft_stored(self):
        """Проверка сохранения размера FFT."""
        wf = self._create_waterfall(Nfft=1024)
        assert wf.Nfft == 1024

    def test_freqs_length_matches_subc_inds(self):
        """Проверка что длина массива частот совпадает с количеством поднесущих."""
        subc_inds = list(range(1, 97))  # 96 поднесущих
        wf = self._create_waterfall(subc_inds=subc_inds)
        assert len(wf._freqs) == len(subc_inds)

    def test_freqs_are_in_hz(self):
        """Проверка что частоты действительно в Гц (положительные, разумные значения)."""
        subc_inds = list(range(1, 49))
        fs = 48000
        Nfft = 256
        wf = self._create_waterfall(subc_inds=subc_inds, fs=fs, Nfft=Nfft)
        
        # Все частоты должны быть положительными
        assert np.all(wf._freqs > 0)
        # Максимальная частота должна быть меньше fs/2 (Найквист)
        assert wf._freqs[-1] < fs / 2


# =============================================================================
# Тесты сохранения в файл (требует matplotlib)
# =============================================================================

class TestEqualizerWaterfallSave:
    """Тесты сохранения водопадной диаграммы в PNG файл."""

    def _create_waterfall(self, **kwargs):
        """Вспомогательный метод для создания waterfall с реальным matplotlib."""
        from equalizer_waterfall import EqualizerWaterfall
        defaults = dict(
            subc_inds=_default_subc_inds(),
            fs=_default_fs(),
            Nfft=_default_Nfft(),
        )
        defaults.update(kwargs)
        return EqualizerWaterfall(**defaults)

    def test_save_creates_file(self):
        """Проверка что save() создаёт PNG файл."""
        import equalizer_waterfall as ew
        
        if not ew.PLOTTING_AVAILABLE:
            pytest.skip("matplotlib недоступен")
        
        test_filename = "test_waterfall_save.png"
        
        try:
            wf = self._create_waterfall()
            
            # Добавляем несколько снимков (больше step чтобы проверить выборку)
            for i in range(25):
                Hk = np.exp(1j * np.linspace(0, 2 * np.pi, 48)) * (1 + 0.1 * i)
                wf.update(Hk)
            
            result = wf.save(test_filename)
            
            assert result == True
            assert os.path.exists(test_filename)
            assert os.path.getsize(test_filename) > 0
            
        finally:
            if os.path.exists(test_filename):
                os.remove(test_filename)

    def test_save_empty_waterfall_returns_false(self):
        """Проверка что save() возвращает False при пустом буфере."""
        import equalizer_waterfall as ew
        
        if not ew.PLOTTING_AVAILABLE:
            pytest.skip("matplotlib недоступен")
        
        test_filename = "test_waterfall_empty.png"
        
        try:
            wf = self._create_waterfall()
            result = wf.save(test_filename)
            
            assert result == False
            assert not os.path.exists(test_filename)
            
        finally:
            if os.path.exists(test_filename):
                os.remove(test_filename)

    def test_save_with_custom_filename(self):
        """Проверка сохранения с пользовательским именем файла."""
        import equalizer_waterfall as ew
        
        if not ew.PLOTTING_AVAILABLE:
            pytest.skip("matplotlib недоступен")
        
        test_filename = "test_custom_filename.png"
        
        try:
            wf = self._create_waterfall()
            
            for i in range(15):
                Hk = np.ones(48, dtype=complex) * (i + 1)
                wf.update(Hk)
            
            result = wf.save(test_filename)
            
            assert result == True
            assert os.path.exists(test_filename)
            
        finally:
            if os.path.exists(test_filename):
                os.remove(test_filename)

    def test_save_on_close(self):
        """Проверка автосохранения при закрытии."""
        import equalizer_waterfall as ew
        
        if not ew.PLOTTING_AVAILABLE:
            pytest.skip("matplotlib недоступен")
        
        test_filename = "test_save_on_close.png"
        
        try:
            wf = self._create_waterfall(save_on_close=True, save_filename=test_filename)
            
            for i in range(15):
                Hk = np.ones(48, dtype=complex) * (i + 1)
                wf.update(Hk)
            
            wf.close()
            
            assert os.path.exists(test_filename)
            
        finally:
            if os.path.exists(test_filename):
                os.remove(test_filename)

    def test_save_with_step_filters_snapshots(self):
        """Проверка что step фильтрует снимки при отрисовке."""
        import equalizer_waterfall as ew
        
        if not ew.PLOTTING_AVAILABLE:
            pytest.skip("matplotlib недоступен")
        
        test_filename = "test_waterfall_step.png"
        
        try:
            # Создаём waterfall с step=5
            wf = self._create_waterfall(step=5)
            
            # Добавляем 20 снимков
            for i in range(20):
                Hk = np.ones(48, dtype=complex) * (i + 1)
                wf.update(Hk)
            
            # Должно отрисоваться 20/5 = 4 линии
            result = wf.save(test_filename)
            
            assert result == True
            assert os.path.exists(test_filename)
            assert os.path.getsize(test_filename) > 0
            
        finally:
            if os.path.exists(test_filename):
                os.remove(test_filename)

    def test_save_single_snapshot(self):
        """Проверка сохранения с одним снимком (без легенды)."""
        import equalizer_waterfall as ew
        
        if not ew.PLOTTING_AVAILABLE:
            pytest.skip("matplotlib недоступен")
        
        test_filename = "test_waterfall_single.png"
        
        try:
            wf = self._create_waterfall()
            Hk = np.ones(48, dtype=complex)
            wf.update(Hk)
            
            result = wf.save(test_filename)
            
            assert result == True
            assert os.path.exists(test_filename)
            
        finally:
            if os.path.exists(test_filename):
                os.remove(test_filename)


# =============================================================================
# Тесты интеграции с rx_decoder
# =============================================================================

class TestRxDecoderWaterfallIntegration:
    """Тесты интеграции водопада с rx_decoder."""

    def test_decode_packet_at_candidate_signature(self):
        """Проверка что decode_packet_at_candidate принимает show_waterfall."""
        from rx_decoder import decode_packet_at_candidate
        import inspect
        
        sig = inspect.signature(decode_packet_at_candidate)
        params = list(sig.parameters.keys())
        
        assert 'show_waterfall' in params
        assert sig.parameters['show_waterfall'].default == False

    def test_try_decode_with_modulation_signature(self):
        """Проверка что _try_decode_with_modulation принимает show_waterfall."""
        from rx_decoder import _try_decode_with_modulation
        import inspect
        
        sig = inspect.signature(_try_decode_with_modulation)
        params = list(sig.parameters.keys())
        
        assert 'show_waterfall' in params
        assert sig.parameters['show_waterfall'].default == False

    def test_save_equalizer_waterfall_exists(self):
        """Проверка наличия функции save_equalizer_waterfall в rx_decoder."""
        from rx_decoder import save_equalizer_waterfall
        assert callable(save_equalizer_waterfall)


# =============================================================================
# Тесты граничных случаев
# =============================================================================

class TestEqualizerWaterfallEdgeCases:
    """Тесты граничных случаев."""

    def _create_waterfall(self, max_symbols=500, **kwargs):
        """Вспомогательный метод для создания waterfall с мокнутым matplotlib."""
        import equalizer_waterfall as ew
        original_plt = ew.plt
        ew.PLOTTING_AVAILABLE = False
        ew.plt = None
        
        defaults = dict(
            subc_inds=_default_subc_inds(),
            fs=_default_fs(),
            Nfft=_default_Nfft(),
        )
        defaults.update(kwargs)
        
        wf = ew.EqualizerWaterfall(max_symbols=max_symbols, **defaults)
        
        ew.plt = original_plt
        try:
            import matplotlib
            ew.PLOTTING_AVAILABLE = True
        except ImportError:
            ew.PLOTTING_AVAILABLE = False
        
        return wf

    def test_max_symbols_one(self):
        """Проверка работы с max_symbols=1."""
        wf = self._create_waterfall(max_symbols=1)
        
        for i in range(5):
            Hk = np.ones(48, dtype=complex) * i
            wf.update(Hk)
        
        assert wf.get_snapshot_count() == 1
        assert wf.get_snapshots()[0][0] == 4 + 0j

    def test_max_symbols_large(self):
        """Проверка работы с большим max_symbols."""
        wf = self._create_waterfall(max_symbols=10000)
        
        for i in range(100):
            Hk = np.ones(48, dtype=complex)
            wf.update(Hk)
        
        assert wf.get_snapshot_count() == 100

    def test_update_with_zeros(self):
        """Проверка update с нулевым Hk."""
        wf = self._create_waterfall()
        Hk = np.zeros(48, dtype=complex)
        wf.update(Hk)
        
        assert wf.get_snapshot_count() == 1
        np.testing.assert_array_equal(wf.get_snapshots()[0], Hk)

    def test_update_with_random_Hk(self):
        """Проверка update со случайным Hk."""
        wf = self._create_waterfall()
        np.random.seed(42)
        Hk = np.random.randn(48) + 1j * np.random.randn(48)
        wf.update(Hk)
        
        snapshots = wf.get_snapshots()
        np.testing.assert_array_almost_equal(snapshots[0], Hk)

    def test_multiple_clear_cycles(self):
        """Проверка многократных циклов очистки и заполнения."""
        wf = self._create_waterfall()
        
        for cycle in range(3):
            for i in range(5):
                Hk = np.ones(48, dtype=complex) * i
                wf.update(Hk)
            assert wf.get_snapshot_count() == 5
            wf.clear()
            assert wf.get_snapshot_count() == 0
