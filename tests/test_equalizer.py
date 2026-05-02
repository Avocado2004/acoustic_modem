"""
Тесты для проверки работы адаптивного эквалайзера (AdaptiveEqualizer).
Проверяют Decision-Directed логику и сглаживание.
"""

import numpy as np
import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modem_modulation import AdaptiveEqualizer

def test_equalizer_init():
    """Тест инициализации эквалайзера."""
    initial_Hk = np.ones(48, dtype=complex) + 0.1j
    eq = AdaptiveEqualizer(initial_Hk, alpha=0.1, modulation='QPSK')
    
    assert eq.Hk is not None, "Hk не инициализирован"
    assert eq.alpha == 0.1, "Неверное значение alpha"
    assert eq.modulation == 'QPSK', "Неверный тип модуляции"
    assert eq.symbol_count == 0, "Счетчик символов должен быть 0"
    print("[TEST] test_equalizer_init: OK")

def test_equalizer_qpsk_process():
    """Тест обработки QPSK символа."""
    # Идеальный канал: Hk = 1.0 + 0.0j
    initial_Hk = np.ones(48, dtype=complex)
    eq = AdaptiveEqualizer(initial_Hk, alpha=0.05, modulation='QPSK')
    
    # Создаем идеальный принятый символ (QPSK точки)
    # (1+1j)/sqrt(2)
    ideal_sym = np.full(48, (1+1j)/np.sqrt(2), dtype=complex)
    # Пропускаем через канал с коэффициентом 2.0
    rx_fd = ideal_sym * 2.0
    
    result = eq.process(rx_fd)
    
    # Проверяем, что результат близок к идеальному (2.0 / 1.0 = 2.0, но Hk обновится)
    assert result is not None, "Результат не должен быть None"
    assert len(result) == 48, "Длина результата должна быть 48"
    assert eq.symbol_count == 1, "Счетчик символов должен быть 1"
    print("[TEST] test_equalizer_qpsk_process: OK")

def test_equalizer_bpsk_process():
    """Тест обработки BPSK символа."""
    initial_Hk = np.ones(48, dtype=complex)
    eq = AdaptiveEqualizer(initial_Hk, alpha=0.05, modulation='BPSK')
    
    # BPSK: 1.0 или -1.0
    ideal_sym = np.ones(48, dtype=complex)
    rx_fd = ideal_sym * 0.5  # Канал ослабляет в 2 раза
    
    result = eq.process(rx_fd)
    
    assert result is not None, "Результат не должен быть None"
    assert eq.symbol_count == 1, "Счетчик символов должен быть 1"
    print("[TEST] test_equalizer_bpsk_process: OK")

def test_equalizer_decision_directed():
    """
    Тест Decision-Directed обновления.
    Проверяем, что Hk адаптируется к изменению канала.
    """
    np.random.seed(42)
    Nsub = 48
    
    # Начальный канал: Hk = 1.0
    initial_Hk = np.ones(Nsub, dtype=complex)
    eq = AdaptiveEqualizer(initial_Hk, alpha=0.1, modulation='QPSK')
    
    # Симулируем изменение канала: сначала 1.0, потом 2.0
    for i in range(20):
        if i < 10:
            channel = 1.0
        else:
            channel = 2.0
        
        # QPSK символ
        tx_sym = np.full(Nsub, (1+1j)/np.sqrt(2), dtype=complex)
        rx_fd = tx_sym * channel
        
        eq.process(rx_fd)
    
    # После 20 символов Hk должен приблизиться к 2.0
    avg_mag, _, _ = eq.get_debug_info()
    
    # Проверяем, что Hk изменился и приблизился к 2.0
    # (точное значение зависит от alpha и количества символов)
    assert 1.0 < avg_mag < 3.0, f"Hk должен адаптироваться к каналу 2.0, текущее значение: {avg_mag}"
    print(f"[TEST] test_equalizer_decision_directed: OK (avg_mag={avg_mag:.3f})")

def test_equalizer_smoothing():
    """
    Тест сглаживания (alpha).
    При маленьком alpha изменение Hk должно быть плавным.
    """
    Nsub = 48
    initial_Hk = np.ones(Nsub, dtype=complex)
    
    # Маленький alpha для сильного сглаживания
    eq = AdaptiveEqualizer(initial_Hk, alpha=0.01, modulation='QPSK')
    
    # Резко меняем канал на 10.0
    tx_sym = np.full(Nsub, (1+1j)/np.sqrt(2), dtype=complex)
    rx_fd = tx_sym * 10.0
    
    eq.process(rx_fd)
    
    avg_mag, _, _ = eq.get_debug_info()
    
    # При alpha=0.01 Hk должен измениться незначительно за один шаг
    assert avg_mag < 5.0, f"При alpha=0.01 Hk должен меняться плавно, текущее: {avg_mag}"
    print(f"[TEST] test_equalizer_smoothing: OK (avg_mag={avg_mag:.3f})")

def test_equalizer_get_debug_info():
    """Тест получения отладочной информации."""
    initial_Hk = np.array([1.0+0.1j, 2.0-0.2j, 1.5+0.3j], dtype=complex)
    eq = AdaptiveEqualizer(initial_Hk, alpha=0.05, modulation='QPSK')
    
    avg_mag, avg_phase, count = eq.get_debug_info()
    
    assert isinstance(avg_mag, (int, float)), "avg_mag должен быть числом"
    assert isinstance(avg_phase, (int, float)), "avg_phase должен быть числом"
    assert count == 0, "count должен быть 0 до обработки"
    print("[TEST] test_equalizer_get_debug_info: OK")

def test_equalizer_channel_compensation():
    """
    Тест компенсации искажений канала.
    Проверяем, что на выходе получаем символ, близкий к переданному.
    """
    Nsub = 48
    # Канал с разными коэффициентами на разных поднесущих
    Hk_real = np.linspace(0.5, 2.0, Nsub)
    initial_Hk = Hk_real + 0j
    
    eq = AdaptiveEqualizer(initial_Hk, alpha=0.0, modulation='QPSK')  # alpha=0, чтобы не менять Hk
    
    # Передаем идеальный символ
    tx_sym = np.full(Nsub, (1+1j)/np.sqrt(2), dtype=complex)
    rx_fd = tx_sym * Hk_real
    
    result = eq.process(rx_fd)
    
    # Ожидаемый результат: rx_fd / Hk = tx_sym
    expected = tx_sym
    error = np.mean(np.abs(result - expected))
    
    assert error < 1e-10, f"Ошибка компенсации канала слишком большая: {error}"
    print("[TEST] test_equalizer_channel_compensation: OK")

if __name__ == "__main__":
    test_equalizer_init()
    test_equalizer_qpsk_process()
    test_equalizer_bpsk_process()
    test_equalizer_decision_directed()
    test_equalizer_smoothing()
    test_equalizer_get_debug_info()
    test_equalizer_channel_compensation()
    print("\n[ALL TESTS PASSED]")
