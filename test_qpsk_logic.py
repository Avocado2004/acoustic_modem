#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import sys

# Импортируйте свои функции модуляции/демодуляции
# from your_module import modulate, demodulate

# Для демонстрации здесь заглушки, замените их на реальные:
def modulate(bit_string):
    # TODO: заменить на реальную функцию
    # должна принимать строку '0101…' и возвращать сигнал (список/массив)
    return bit_string

def demodulate(signal):
    # TODO: заменить на реальную функцию
    # должна принимать сигнал и возвращать строку битов '0101…'
    return signal

def bits_to_pairs(bits):
    """
    Разбивает строку bits в список 2-битных подстрок.
    Пример: '00011011' → ['00','01','10','11']
    """
    return [bits[i:i+2] for i in range(0, len(bits), 2)]

def test_pairing(bits, expected_pairs):
    pairs = bits_to_pairs(bits)
    assert pairs == expected_pairs, (
        f"Ошибка разбиения: для {bits} получили {pairs}, ожидали {expected_pairs}"
    )
    print(f"[OK] Разбиение для {bits}: {pairs}")

def test_round_trip(bits):
    tx = modulate(bits)
    rx = demodulate(tx)
    assert rx == bits, (
        f"Round-trip mismatch: после mod→demod получили {rx}, ожидали {bits}"
    )
    print(f"[OK] Round-trip для {bits}")

def main():
    # 1) Тест точного разбиения
    print("\n=== Проверка разбиения на пары бит ===")
    test_pairing("", [])
    test_pairing("00", ["00"])
    test_pairing("01", ["01"])
    test_pairing("00011011", ["00","01","10","11"])

    # 2) Мини-тест round-trip для жёстко заданных битов
    print("\n=== Мини-тест Round-Trip ===")
    test_round_trip("00011011")

    # 3) Серия случайных проверок
    print("\n=== Случайные тесты Round-Trip ===")
    for i in range(10):
        length = random.randint(1, 8) * 2  # длина кратна 2 битам
        bits = ''.join(random.choice("01") for _ in range(length))
        try:
            test_round_trip(bits)
        except AssertionError as e:
            print(f"\n[FAIL] Тест #{i+1} для {bits}\n", e)
            sys.exit(1)

    print("\nВсе тесты пройдены успешно 🎉")

if __name__ == "__main__":
    main()

