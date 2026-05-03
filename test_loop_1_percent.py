#!/usr/bin/env python3
"""
Скрипт для автоматического тестирования лупа с 1% искажений.
"""
import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Переопределяем input для автоматического ввода
original_input = __builtins__.input
input_calls = 0

def mock_input(prompt=""):
    global input_calls
    print(f"[MOCK-INPUT] Prompt: {prompt}")
    if "Режим работы" in prompt or "Режим работы" in prompt:
        return "L"  # Loop mode
    elif "процент искажений" in prompt.lower():
        return "1"  # 1% искажений
    elif "Модуляция" in prompt:
        return "Q"  # QPSK
    elif "Режим передачи" in prompt:
        return "T"  # Text
    elif "Введите текст" in prompt:
        return "Test text for loop 1% distortion"
    elif "Удалить временные файлы" in prompt:
        return "y"
    else:
        print(f"[MOCK-INPUT] Unhandled prompt: {prompt}")
        return ""

__builtins__.input = mock_input

# Импортируем и запускаем
try:
    from modem_cli import main
    main()
except SystemExit as e:
    print(f"[TEST] SystemExit with code {e.code}")
except Exception as e:
    print(f"[TEST-ERR] Exception: {e}")
    import traceback
    traceback.print_exc()
finally:
    __builtins__.input = original_input
