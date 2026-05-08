#!/usr/bin/env python3
"""
Автоматический Loop-тест для отладки декодирования пакетов.
Режим: Loop (идеальный канал), QPSK, текстовое сообщение.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Переопределяем input для автоматического ввода
original_input = __builtins__.input

def mock_input(prompt=""):
    print(f"[MOCK-INPUT] Prompt: {prompt}")
    if "Режим работы" in prompt or "режим работы" in prompt.lower():
        return "L"  # Loop mode
    elif "процент искажений" in prompt.lower() or "искажения" in prompt.lower():
        return "0"  # 0% искажений (идеальный канал)
    elif "Модуляция" in prompt or "модуляция" in prompt.lower():
        return "Q"  # QPSK
    elif "Режим передачи" in prompt or "режим передачи" in prompt.lower():
        return "T"  # Text
    elif "Введите текст" in prompt or "текст" in prompt.lower():
        return "Hello World! This is a test message for loop debugging."
    elif "Удалить временные файлы" in prompt or "удалить" in prompt.lower():
        return "n"  # Не удалять для анализа
    elif "packet" in prompt.lower() or "пакет" in prompt.lower():
        return "75"  # Количество блоков
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
