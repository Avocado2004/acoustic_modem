#!/usr/bin/env python3
"""
Автоматический тест микрофонного режима (M).
Запускается без интерактивного ввода.
"""

import sys
import os
import random
import string
import time
import threading

# Добавляем текущую директорию в path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modem_config import init_phases, fs, MODULATION, Nfft, Ncp, Nsub, BITS_PER_SYMBOL
from modem_modulation import build_preamble
from modem_tx import transmit_text
from signal_processor import receive_text_from_microphone


def run_mic_test():
    """Автоматический тест режима Microphone."""
    print("\n[MIC] Запуск автоматического теста Microphone")
    
    # Устанавливаем QPSK модуляцию
    from modem_config import set_modulation
    set_modulation("QPSK")
    init_phases()
    
    # Тестовый текст (1 КБ)
    random.seed(42)
    test_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(1024))
    print(f"[MIC] Текст для передачи: '{test_text[:40]}...' (длина: {len(test_text)} символов)")
    
    # Контейнер для результата
    rx_result = {'text': None, 'done': False, 'error': None}
    
    def rx_thread_func():
        """Функция потока приёма."""
        try:
            print(f"[MIC-RX] Запуск приёма на микрофоне...")
            rx_result['text'] = receive_text_from_microphone()
            rx_result['done'] = True
            print(f"[MIC-RX] Приём завершён")
        except Exception as e:
            rx_result['error'] = str(e)
            rx_result['done'] = True
            print(f"[MIC-RX] Ошибка приёма: {e}")
            import traceback
            traceback.print_exc()
    
    # Запускаем приём в отдельном потоке
    print(f"\n[MIC] Запуск потока приёма...")
    t = threading.Thread(target=rx_thread_func, daemon=True)
    t.start()
    
    # Ждём 5 сек чтобы RX начал слушать и накопил достаточно данных
    print(f"[MIC] Ожидание 5 сек перед передачей...")
    time.sleep(5)
    
    # Запускаем передачу
    print(f"\n[MIC] === ПЕРЕДАЧА ===")
    try:
        transmit_text(test_text)
        print(f"[MIC] Передача завершена")
    except Exception as e:
        print(f"[MIC-ERR] Ошибка передачи: {e}")
        import traceback
        traceback.print_exc()
    
    # Ждём пока микрофон запишет весь сигнал
    # Сигнал длится ~2.5 сек, но нужно время на распространение + запас
    print(f"[MIC] Ожидание 10 сек пока микрофон запишет сигнал...")
    time.sleep(10)
    
    # Ждём завершения приёма (таймаут 120 сек)
    print(f"[MIC] Ожидание завершения приёма (таймаут 120 сек)...")
    t.join(timeout=120)
    
    if not rx_result['done']:
        print(f"[MIC-ERR] Приём не завершился за 60 секунд!")
        return False
    
    if rx_result['error']:
        print(f"[MIC-ERR] Ошибка в потоке приёма: {rx_result['error']}")
        return False
    
    received_text = rx_result['text']
    if received_text is None:
        print(f"[MIC-ERR] Принятый текст = None")
        return False
    
    # Выводим результат
    print(f"\n[MIC] === РЕЗУЛЬТАТ ===")
    print(f"[MIC] Оригинальный текст: '{test_text[:40]}...' ({len(test_text)} символов)")
    print(f"[MIC] Принятый текст:    '{received_text[:40]}...' ({len(received_text)} символов)")
    
    # Сравнение
    if test_text == received_text:
        print(f"\n[MIC] ✅ УСПЕХ: Текст совпадает полностью!")
        return True
    else:
        print(f"\n[MIC] ❌ ОШИБКА: Текст отличается")
        min_len = min(len(test_text), len(received_text))
        for i in range(min_len):
            if test_text[i] != received_text[i]:
                print(f"[MIC] Первое отличие в позиции {i}: ориг='{test_text[i]}', принято='{received_text[i]}'")
                break
        if len(test_text) != len(received_text):
            print(f"[MIC] Длина отличается: ориг={len(test_text)}, принято={len(received_text)}")
        
        # Поблочное сравнение
        BLOCK_SIZE = 64
        original_bytes = test_text.encode('utf-8')
        received_bytes = received_text.encode('utf-8')
        n_blocks = (len(original_bytes) + BLOCK_SIZE - 1) // BLOCK_SIZE
        ok_blocks = 0
        
        for i in range(n_blocks):
            start = i * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, len(original_bytes))
            orig_block = original_bytes[start:end]
            recv_block = received_bytes[start:end] if start < len(received_bytes) else b''
            
            if len(recv_block) < len(orig_block):
                recv_block = recv_block + b'\x00' * (len(orig_block) - len(recv_block))
            
            if orig_block == recv_block:
                ok_blocks += 1
        
        print(f"[MIC] Блоки: {ok_blocks}/{n_blocks} совпадают")
        return False


if __name__ == "__main__":
    success = run_mic_test()
    sys.exit(0 if success else 1)
