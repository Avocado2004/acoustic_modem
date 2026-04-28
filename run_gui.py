#!/usr/bin/env python3
"""
Скрипт для запуска OFDM GUI на десктопных платформах.
Проверяет зависимости и выводит понятные сообщения об ошибках.
"""

import sys
import subprocess
import importlib

def check_dependency(package_name, import_name=None):
    """Проверяет, установлен ли пакет."""
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    print("Проверка зависимостей для OFDM GUI...")
    
    # Список необходимых пакетов
    dependencies = [
        ("kivy", "kivy"),
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("reedsolo", "reedsolo"),
        ("pycryptodome", "Crypto"),
    ]
    
    missing = []
    for package, import_name in dependencies:
        if not check_dependency(package, import_name):
            missing.append(package)
    
    if missing:
        print("ОШИБКА: Отсутствуют следующие зависимости:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nУстановите их с помощью:")
        print("  pip install -r requirements.txt")
        return 1
    
    print("Все зависимости найдены!")
    
    # Проверяем доступность аудиобэкенда
    try:
        from audio_backend import get_backend
        backend = get_backend()
        if backend is None:
            print("ПРЕДУПРЕЖДЕНИЕ: Аудиобэкенд не инициализирован. GUI будет работать, но воспроизведение/запись недоступны.")
        else:
            print(f"Аудиобэкенд успешно инициализирован: {type(backend).__name__}")
    except Exception as e:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось инициализировать аудиобэкенд: {e}")
        print("GUI будет работать, но воспроизведение/запись недоступны.")
    
    # Запускаем GUI
    print("\nЗапуск OFDM GUI...")
    try:
        from ofdm_gui_kivy import OfdmApp
        app = OfdmApp()
        app.run()
    except Exception as e:
        print(f"ОШИБКА при запуске GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())