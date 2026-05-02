"""
Модуль CLI для OFDM Acoustic Modem.
Содержит функции для режимов передачи, приема и тестирования в петле.
"""

import sys
import os
import random
import string
import glob

# Импортируем функции из других модулей
from modem_config import init_phases
from modem_modulation import build_preamble
from modem_tx import transmit_text, transmit_file, _transmit_data
from modem_rx import receive_from_file, live_receive_and_process


def run_transmit():
    """Режим передачи: файл или текст."""
    mode = input("Режим передачи — [F]ile или [T]ext (по умолчанию F): ").strip().upper()
    mode = "T" if mode == "T" else "F"
    
    if mode == "F":
        file_path = input("Путь к файлу для передачи: ").strip()
        transmit_file(file_path)
    else:
        text = input("Введите текст для передачи: ").strip()
        transmit_text(text)


def run_receive():
    """Режим приема: из файла или с микрофона."""
    src = input("Откуда демодулировать? (file/mic): ").strip().lower()
    if src == "file":
        wav_path = input("Путь к WAV-файлу для приёма: ").strip()
        success = receive_from_file(wav_path)
        if not success:
            print("[MAIN] Ошибка при приеме из файла.")
            sys.exit(1)
    else:
        live_receive_and_process()
        sys.exit(0)


def run_loop():
    """Режим Loop: генерация -> передача -> прием -> проверка."""
    print("\n[LOOP] Запуск режима Loop (генерация -> передача -> прием -> проверка)")
    
    # 1. Генерируем рандомный текст
    random_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(50))
    print(f"[LOOP] Сгенерирован текст: {random_text[:20]}...")
    original_data = random_text.encode('utf-8')
    print(f"[LOOP] Размер данных: {len(original_data)} байт")
    
    # 2. Передаем текст
    print(f"\n[LOOP] Передача текста...")
    # Сохраняем текст во временный файл для проверки
    temp_original = "loop_original_text.bin"
    with open(temp_original, "wb") as f:
        f.write(original_data)
    print(f"[LOOP] Оригинальные данные сохранены в {temp_original}")
    
    # Передаем текст (используем transmit_text, который сохраняет в WAV)
    transmit_text(random_text)
    print(f"[LOOP] Передача завершена. Сигнал сохранен в ofdm_acoustic_tx_with_noise.wav")
    
    # 3. Принимаем из только что созданного WAV-файла
    print(f"\n[LOOP] Прием данных из WAV-файла...")
    received_file = "rx_text.txt"  # Теперь текст сохраняется в rx_text.txt
    # Удаляем старый файл, если есть
    if os.path.exists(received_file):
        os.remove(received_file)
    
    success = receive_from_file("ofdm_acoustic_tx_with_noise.wav")
    if not success:
        print("[LOOP-ERR] Ошибка при приеме из файла.")
        sys.exit(1)
    
    # 4. Сравниваем оригинальные данные с принятыми
    # Принятый текст сохраняется в rx_text.txt (см. modem_rx.py)
    if os.path.exists(received_file):
        with open(received_file, "r", encoding='utf-8') as f:
            received_text = f.read()
        print(f"\n[LOOP] Сравнение данных...")
        print(f"[LOOP] Оригинальный текст: {random_text[:30]}...")
        print(f"[LOOP] Принятый текст: {received_text[:30]}...")
        
        if random_text == received_text:
            print(f"[LOOP] ✅ УСПЕХ: Текст совпадает полностью!")
        else:
            print(f"[LOOP] ❌ ОШИБКА: Текст отличается")
            # Находим первое отличие
            min_len = min(len(random_text), len(received_text))
            for i in range(min_len):
                if random_text[i] != received_text[i]:
                    print(f"[LOOP] Первое отличие в позиции {i}: ориг='{random_text[i]}', принято='{received_text[i]}'")
                    break
            if len(random_text) != len(received_text):
                print(f"[LOOP] Длина отличается: ориг={len(random_text)}, принято={len(received_text)}")
    else:
        print(f"[LOOP-ERR] Файл {received_file} не найден. Прием не удался.")
        # Попробуем найти любой файл rx_*
        rx_files = glob.glob("rx_*")
        if rx_files:
            print(f"[LOOP] Найдены файлы: {rx_files}")
    
    # Опционально: удаляем временные файлы
    cleanup = input("\nУдалить временные файлы? (y/n, по умолчанию n): ").strip().lower()
    if cleanup == 'y':
        for f in [temp_original, received_file, "ofdm_acoustic_tx_with_noise.wav"]:
            if os.path.exists(f):
                os.remove(f)
                print(f"[LOOP] Удален {f}")


def main():
    """Основная функция CLI."""
    # Инициализация фаз и преамбулы
    init_phases()
    preamble_td = build_preamble()
    
    op = input("Режим работы — [T]ransmit, [R]eceive или [L]oop (по умолчанию T): ").strip().upper()
    op = "L" if op == "L" else ("R" if op == "R" else "T")
    
    if op == "T":
        run_transmit()
    elif op == "R":
        run_receive()
    else:  # Loop mode
        run_loop()


if __name__ == "__main__":
    main()