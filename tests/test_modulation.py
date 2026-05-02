"""
Тесты для проверки модуляции BPSK и QPSK.
Проверяет функции маппинга и демаппинга битов в символы.
"""
import sys
import os
import numpy as np

# Добавляем корень проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modem_modulation as mm
import modem_config as cfg

def test_bpsk_map_demap():
    """Тест корректности BPSK модуляции и демодуляции."""
    print("\n[ТЕСТ] BPSK map/demap")
    # ИСПРАВЛЕНО: добавлены запятые между элементами массива
    bits = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    syms = mm.bpsk_map(bits)
    
    assert syms[0] == 1.0, "Ошибка маппинга BPSK: 0 должен быть 1.0"
    assert syms[1] == -1.0, "Ошибка маппинга BPSK: 1 должен быть -1.0"
    
    recovered_bits = mm.bpsk_demap(syms)
    # ИСПРАВЛЕНО: добавлена запятая между аргументами
    assert np.array_equal(bits, recovered_bits), "Ошибка демаппинга BPSK"
    print("[OK] BPSK map/demap работает корректно")

def test_qpsk_map_demap():
    """Тест корректности QPSK модуляции и демодуляции."""
    print("\n[ТЕСТ] QPSK map/demap")
    # ИСПРАВЛЕНО: добавлены запятые между элементами массива
    bits = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=int)
    syms = mm.qpsk_map(bits)
    
    assert len(syms) == 4, "Должно быть 4 символа для 8 бит"
    assert np.iscomplexobj(syms), "QPSK символы должны быть комплексными"
    
    recovered_bits = mm.qpsk_demap(syms)
    # ИСПРАВЛЕНО: добавлена запятая между аргументами
    assert np.array_equal(bits, recovered_bits[:len(bits)]), "Ошибка демаппинга QPSK"
    print("[OK] QPSK map/demap работает корректно")

def test_modulation_switch():
    """Тест переключения типа модуляции через modem_config."""
    print("\n[ТЕСТ] Переключение модуляции через modem_config")
    
    original_mod = cfg.MODULATION
    
    try:
        cfg.MODULATION = "BPSK"
        assert cfg.MODULATION == "BPSK"
        # ИСПРАВЛЕНО: добавлены запятые
        bits = np.array([0, 1, 0], dtype=int)
        syms = mm.build_data_td(bits)
        print(f"[OK] Режим BPSK активирован, символов: {len(syms)}")
        
        cfg.MODULATION = "QPSK"
        assert cfg.MODULATION == "QPSK"
        syms = mm.build_data_td(bits)
        print(f"[OK] Режим QPSK активирован, символов: {len(syms)}")
        
    finally:
        cfg.MODULATION = original_mod

def test_interleaving():
    """Тест работы интерливинга и деинтерливинга."""
    print("\n[ТЕСТ] Интерливинг битов")
    # ИСПРАВЛЕНО: добавлены запятые в аргументах randint
    bits = np.random.randint(0, 2, 96).astype(int)
    interleaved = mm.interleave_bits(bits)
    deinterleaved = mm.deinterleave_bits(interleaved)
    
    # ИСПРАВЛЕНО: добавлена запятая между аргументами
    assert np.array_equal(bits, deinterleaved), "Ошибка интерливинга/деинтерливинга"
    print("[OK] Интерливинг работает корректно")

if __name__ == "__main__":
    test_bpsk_map_demap()
    test_qpsk_map_demap()
    test_modulation_switch()
    test_interleaving()
    print("\n========== ВСЕ ТЕСТЫ МОДУЛЯЦИИ ПРОЙДЕНЫ ==========")