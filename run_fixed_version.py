#!/usr/bin/env python3
"""
Скрипт для запуска исправленной версии кода.
Принудительно перезагружает все модули и проверяет их версию.
"""

import sys
import os

# Очистка кэша модулей
modules_to_remove = []
for key in list(sys.modules.keys()):
    if 'modem' in key or 'audio' in key:
        modules_to_remove.append(key)
        
for mod in modules_to_remove:
    if mod in sys.modules:
        del sys.modules[mod]

print("=== Очистка кэша завершена ===\n")

# Проверяем текущую директорию
print(f"Текущая директория: {os.getcwd()}")
print(f"Ожидаемый файл: {os.path.join(os.getcwd(), 'modem_rx.py')}\n")

# Проверяем наличие старой строки в файле
try:
    with open('modem_rx.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'Equalizer DISABLED' in content:
        print("❌ ОШИБКА: В файле modem_rx.py найдена старая строка!")
        print("   Пожалуйста, обновите файл")
        sys.exit(1)
    else:
        print("✓ В файле modem_rx.py НЕТ старой строки 'Equalizer DISABLED'")
        
    if 'eq_alpha' in content:
        print("✓ В файле modem_rx.py ЕСТЬ новая логика с eq_alpha")
    else:
        print("❌ ОШИБКА: Новая логика не найдена!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ ОШИБКА при чтении файла: {e}")
    sys.exit(1)

# Импортируем и проверяем
print("\n=== Импорт модулей ===")
try:
    import modem_rx
    print(f"✓ modem_rx загружен из: {modem_rx.__file__}")
    
    # Проверяем, что эквалайзер всегда включен
    import inspect
    source = inspect.getsource(modem_rx.decode_packet_at_candidate)
    if 'Equalizer DISABLED' in source:
        print("❌ ОШИБКА: Функция содержит старую логику!")
        sys.exit(1)
    else:
        print("✓ Функция decode_packet_at_candidate НЕ содержит 'Equalizer DISABLED'")
        
    if 'eq_alpha' in source:
        print("✓ Функция содержит новую логику с eq_alpha")
        
except Exception as e:
    print(f"❌ ОШИБКА при импорте: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Запуск режима Loop ===")
print("Если увидите 'Equalizer enabled for ALL packet lengths' - всё работает правильно!")
print("Если увидите 'Equalizer DISABLED' - значит запущена СТАРАЯ версия кода\n")

# Запускаем
try:
    from modem_cli import main
    # Переопределяем аргументы для режима loop
    sys.argv = ['modem_cli.py', '--loop', '--debug']
    main()
except SystemExit:
    pass
except Exception as e:
    print(f"\n❌ Ошибка при запуске: {e}")
    import traceback
    traceback.print_exc()
