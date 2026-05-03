"""
Скрипт для автоматической проверки BPSK с 0% искажений.
Имитирует ввод для modem_cli.py --loop
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Имитируем ввод пользователя
simulated_input = [
    'L',      # Режим Loop
    '1',      # BPSK
    '0'       # 0% искажений
]

# Переопределяем input()
original_input = __builtins__.input
input_index = 0

def mock_input(prompt=''):
    global input_index
    if input_index < len(simulated_input):
        value = simulated_input[input_index]
        print(f"{prompt}{value}")
        input_index += 1
        return value
    return original_input(prompt)

__builtins__.input = mock_input

# Запускаем modem_cli
try:
    from modem_cli import main
    sys.argv = ['modem_cli.py', '--loop', '--debug']
    main()
except SystemExit:
    pass
except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()
finally:
    __builtins__.input = original_input
