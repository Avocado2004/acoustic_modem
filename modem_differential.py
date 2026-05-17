"""
Модуль дифференциальной модуляции DQPSK/DBPSK.

DQPSK (Differential QPSK) - дифференциальная QPSK модуляция.
DBPSK (Differential BPSK) - дифференциальная BPSK модуляция.

Принцип работы:
- Пилот-поднесущая (индекс 24) - точка отсчёта фазы (всегда фаза 0)
- Данные кодируются как разность фаз между соседними поднесущими
- Порядок кодирования: пилот (24) → 23 → 25 → 22 → 26 → ... → 0 → 48
- Каждый новый OFDM символ начинается заново с пилота
"""

import numpy as np
import modem_config


def get_diff_order():
    """
    Получить порядок поднесущих для дифференциального кодирования.
    
    Порядок: пилот (24) → 23 → 25 → 22 → 26 → ... → 0 → 48
    
    Возвращает:
    - diff_order: список индексов поднесущих в порядке кодирования
    """
    pilot_idx = modem_config.PILOT_SUBC_INDEX
    diff_order = []
    diff_order.append(pilot_idx - 1)  # 23
    diff_order.append(pilot_idx + 1)  # 25
    for offset in range(2, pilot_idx + 1):
        left = pilot_idx - offset
        right = pilot_idx + offset
        if left >= 0:
            diff_order.append(left)
        if right < modem_config.Nsub:
            diff_order.append(right)
    return diff_order


def dqpsk_map(bits):
    """
    Дифференциальное кодирование битов в DQPSK символы.
    
    Таблица фазовых сдвигов (Gray coding):
    - биты '00' → Δφ = 0
    - биты '01' → Δφ = π/2
    - биты '11' → Δφ = π
    - биты '10' → Δφ = -π/2
    
    Возвращает:
    - syms: массив комплексных символов DQPSK
    """
    phase_shifts = {
        '00': 0,
        '01': np.pi / 2,
        '11': np.pi,
        '10': -np.pi / 2,
    }
    
    if len(bits) % 2:
        bits = np.append(bits, 0)
    
    syms = []
    current_phase = 0.0
    
    for i in range(0, len(bits), 2):
        bit_pair = f"{bits[i]}{bits[i+1]}"
        delta_phi = phase_shifts[bit_pair]
        current_phase += delta_phi
        syms.append(np.exp(1j * current_phase))
    
    return np.array(syms, dtype=complex)


def dqpsk_demap(syms, pilot_phase=0.0):
    """
    Дифференциальное декодирование DQPSK символов в биты.
    
    Параметры:
    - syms: массив комплексных символов DQPSK
    - pilot_phase: фаза пилот-символа
    
    Возвращает:
    - bits: массив декодированных битов
    """
    bits = []
    prev_phase = pilot_phase
    
    for s in syms:
        current_phase = np.angle(s)
        delta_phi = current_phase - prev_phase
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
        
        if delta_phi >= -np.pi / 4 and delta_phi < np.pi / 4:
            bits += [0, 0]
        elif delta_phi >= np.pi / 4 and delta_phi < 3 * np.pi / 4:
            bits += [0, 1]
        elif delta_phi >= 3 * np.pi / 4 or delta_phi < -3 * np.pi / 4:
            bits += [1, 1]
        else:
            bits += [1, 0]
        
        prev_phase = current_phase
    
    return np.array(bits, dtype=int)


def dbpsk_map(bits):
    """
    Дифференциальное кодирование битов в DBPSK символы.
    
    Таблица фазовых сдвигов:
    - бит '0' → Δφ = 0
    - бит '1' → Δφ = π
    
    Возвращает:
    - syms: массив комплексных символов DBPSK
    """
    syms = []
    current_phase = 0.0
    
    for b in bits:
        if b == 1:
            current_phase += np.pi
        syms.append(np.exp(1j * current_phase))
    
    return np.array(syms, dtype=complex)


def dbpsk_demap(syms, pilot_phase=0.0):
    """
    Дифференциальное декодирование DBPSK символов в биты.
    
    Параметры:
    - syms: массив комплексных символов DBPSK
    - pilot_phase: фаза пилот-символа
    
    Возвращает:
    - bits: массив декодированных битов
    """
    bits = []
    prev_phase = pilot_phase
    
    for s in syms:
        current_phase = np.angle(s)
        delta_phi = current_phase - prev_phase
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
        
        if abs(delta_phi) > np.pi / 2:
            bits.append(1)
        else:
            bits.append(0)
        
        prev_phase = current_phase
    
    return np.array(bits, dtype=int)
