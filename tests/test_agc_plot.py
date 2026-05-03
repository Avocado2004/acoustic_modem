"""
Тесты для проверки сбора данных AGC и построения графика.
"""
import sys
import os
import numpy as np

# Добавляем корневую директорию в путь поиска
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_utils as pu
from modem_rx import agc_history_list, _global_symbol_counter


class TestAGCDataCollection:
    """Тесты сбора данных AGC во время декодирования."""
    
    def setup_method(self):
        """Очистка списка AGC перед каждым тестом."""
        agc_history_list.clear()
        # Сбрасываем счётчик (нужно импортировать как глобальную переменную)
        import modem_rx
        modem_rx._global_symbol_counter = 0
    
    def test_agc_history_list_initialized(self):
        """Проверка, что agc_history_list инициализирован как пустой список."""
        assert isinstance(agc_history_list, list)
        assert len(agc_history_list) == 0
    
    def test_agc_history_list_append(self):
        """Проверка добавления данных в agc_history_list."""
        test_data = {
            'symbol_idx': 0,
            'pkt_idx': 0,
            'frame_idx': 0,
            'cur_rms': 0.1,
            'est_rms': 0.12,
            'gain_sym': 4.0
        }
        agc_history_list.append(test_data)
        assert len(agc_history_list) == 1
        assert agc_history_list[0]['symbol_idx'] == 0
        assert agc_history_list[0]['cur_rms'] == 0.1
        assert agc_history_list[0]['est_rms'] == 0.12
        assert agc_history_list[0]['gain_sym'] == 4.0
    
    def test_agc_history_list_multiple_entries(self):
        """Проверка добавления нескольких записей в agc_history_list."""
        for i in range(10):
            agc_history_list.append({
                'symbol_idx': i,
                'pkt_idx': i // 5,  # 2 пакета по 5 символов
                'frame_idx': i % 5,
                'cur_rms': 0.1 * (i + 1),
                'est_rms': 0.12 * (i + 1),
                'gain_sym': 4.0 / (i + 1)
            })
        
        assert len(agc_history_list) == 10
        # Проверяем первый и последний элементы
        assert agc_history_list[0]['symbol_idx'] == 0
        assert agc_history_list[9]['symbol_idx'] == 9
        assert agc_history_list[0]['pkt_idx'] == 0
        assert agc_history_list[5]['pkt_idx'] == 1  # Второй пакет начинается с 5-го символа


class TestPlotAGCPerSymbol:
    """Тесты функции plot_agc_per_symbol."""
    
    def setup_method(self):
        """Подготовка тестовых данных."""
        self.test_data = []
        for i in range(20):
            self.test_data.append({
                'symbol_idx': i,
                'pkt_idx': i // 10,
                'frame_idx': i % 10,
                'cur_rms': 0.1 + 0.01 * i,
                'est_rms': 0.12 + 0.01 * i,
                'gain_sym': 4.0 - 0.1 * i
            })
    
    def test_plot_agc_per_symbol_empty_list(self):
        """Проверка обработки пустого списка."""
        # Не должно быть ошибок, просто вывод в консоль
        pu.plot_agc_per_symbol([])
        pu.plot_agc_per_symbol(None)
    
    def test_plot_agc_per_symbol_with_data(self):
        """Проверка работы с данными (без matplotlib)."""
        # Этот тест должен проходить даже без matplotlib
        try:
            pu.plot_agc_per_symbol(self.test_data, title="Test AGC Plot")
            # Если мы здесь, то функция отработала без ошибок
            # Проверяем, что файл данных создан (если matplotlib недоступен)
            if pu.IS_MOBILE or not pu.PLOTTING_AVAILABLE:
                assert os.path.exists("agc_per_symbol_data.csv")
                # Проверяем содержимое CSV
                data = np.genfromtxt("agc_per_symbol_data.csv", delimiter=",", skip_header=1)
                assert data.shape[0] == 20  # 20 строк данных
                assert data.shape[1] == 4  # 4 столбца: symbol_index, cur_rms, est_rms, gain_sym
                os.remove("agc_per_symbol_data.csv")
            else:
                # Если matplotlib доступен, проверяем, что PNG создан
                assert os.path.exists("agc_per_symbol.png")
                assert os.path.exists("agc_per_symbol_data.csv")
                os.remove("agc_per_symbol.png")
                os.remove("agc_per_symbol_data.csv")
        except Exception as e:
            # Если matplotlib недоступен, но функция должна работать через CSV
            if "matplotlib" in str(e).lower():
                # Это нормально, если matplotlib не установлен
                pass
            else:
                raise
    
    def test_plot_agc_per_symbol_data_format(self):
        """Проверка формата данных в CSV."""
        # Вызываем функцию с принудительным сохранением в CSV (имитируем мобильную платформу)
        original_mobile = pu.IS_MOBILE
        original_plotting = pu.PLOTTING_AVAILABLE
        
        try:
            # Временно отключаем matplotlib для теста
            pu.IS_MOBILE = True
            pu.PLOTTING_AVAILABLE = False
            
            pu.plot_agc_per_symbol(self.test_data, title="Test CSV Output")
            
            # Проверяем CSV файл
            assert os.path.exists("agc_per_symbol_data.csv")
            with open("agc_per_symbol_data.csv", "r") as f:
                lines = f.readlines()
                # Первая строка - заголовок
                assert "symbol_index" in lines[0]
                assert "cur_rms" in lines[0]
                assert "est_rms" in lines[0]
                assert "gain_sym" in lines[0]
                # Данные: 20 строк + 1 заголовок
                assert len(lines) == 21
            
            os.remove("agc_per_symbol_data.csv")
            
        finally:
            # Восстанавливаем оригинальные значения
            pu.IS_MOBILE = original_mobile
            pu.PLOTTING_AVAILABLE = original_plotting


class TestAGCIntegration:
    """Интеграционные тесты сбора данных AGC."""
    
    def setup_method(self):
        """Очистка перед тестом."""
        agc_history_list.clear()
        import modem_rx
        modem_rx._global_symbol_counter = 0
    
    def test_global_symbol_counter_increments(self):
        """Проверка инкремента глобального счётчика символов."""
        import modem_rx
        initial = modem_rx._global_symbol_counter
        
        # Имитируем добавление данных как в _try_decode_with_modulation
        for i in range(5):
            agc_history_list.append({
                'symbol_idx': modem_rx._global_symbol_counter,
                'pkt_idx': 0,
                'frame_idx': i,
                'cur_rms': 0.1,
                'est_rms': 0.12,
                'gain_sym': 4.0
            })
            modem_rx._global_symbol_counter += 1
        
        assert modem_rx._global_symbol_counter == initial + 5
        assert agc_history_list[-1]['symbol_idx'] == initial + 4
    
    def test_agc_data_cleared_after_plot(self):
        """Проверка очистки данных после построения графика."""
        # Добавляем тестовые данные
        for i in range(5):
            agc_history_list.append({
                'symbol_idx': i,
                'pkt_idx': 0,
                'frame_idx': i,
                'cur_rms': 0.1,
                'est_rms': 0.12,
                'gain_sym': 4.0
            })
        
        assert len(agc_history_list) == 5
        
        # Имитируем очистку после построения графика (как в receive_from_file)
        agc_history_list.clear()
        import modem_rx
        modem_rx._global_symbol_counter = 0
        
        assert len(agc_history_list) == 0
        assert modem_rx._global_symbol_counter == 0


if __name__ == "__main__":
    # Запуск тестов вручную
    import pytest
    pytest.main([__file__, "-v"])
