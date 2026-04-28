#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Словари маппинга для QPSK
bits_to_sym = {
    "00":  1+1j,
    "01": -1+1j,
    "11": -1-1j,
    "10":  1-1j,
}
sym_to_bits = {v: k for k, v in bits_to_sym.items()}

def modulate(bit_string):
    """
    1) Разбиваем бит-строку на 2-битные пары
    2) Маппим в QPSK-символы
    3) IFFT → временной сигнал
    """
    # 1) пары по 2 бита
    pairs = [bit_string[i:i+2] for i in range(0, len(bit_string), 2)]
    N = len(pairs)               # число поднесущих
    # 2) формируем массив комплексных символов
    symbols = np.array([bits_to_sym[p] for p in pairs], dtype=complex)
    # 3) IFFT
    tx_time = np.fft.ifft(symbols)
    return tx_time

def demodulate(rx_time):
    """
    1) FFT → возвращаем комплексные символы
    2) По знакам real/imag восстанавливаем биты
    """
    # 1) восстановление символов
    symbols = np.fft.fft(rx_time)
    # 2) демаппинг по знаку компонентов
    bits_out = ""
    for s in symbols:
        I = 1 if s.real >= 0 else -1
        Q = 1 if s.imag >= 0 else -1
        bits_out += sym_to_bits[I+1j*Q]
    return bits_out

if __name__ == "__main__":
    test_bits = "00011011"
    print("Исходные биты    :", test_bits)

    # Модуляция
    tx = modulate(test_bits)
    print("TX (time domain) :", np.round(tx, 3))

    # Канал без искажений
    rx = tx.copy()

    # Демодуляция
    rec_bits = demodulate(rx)
    print("Восстановленные :", rec_bits)

    if rec_bits == test_bits:
        print("OK! Round-trip без ошибок.")
    else:
        print("Ошибка восстановления бит.")

