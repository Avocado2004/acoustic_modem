#!/usr/bin/env python3
"""
Скрипт для запуска OFDM GUI на десктопных и мобильных платформах.
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

def is_mobile_platform():
    """Определяет, является ли платформа мобильной."""
    return (sys.platform in ['android', 'ios'] or 
            'ANDROID_ROOT' in os.environ or 
            'IPHONEOS_DEPLOYMENT_TARGET' in os.environ)

def main():
    import os
    
    print("Проверка зависимостей для OFDM GUI...")
    
    # Определяем платформу
    mobile = is_mobile_platform()
    
    if mobile:
        print(f"Обнаружена мобильная платформа: {sys.platform}")
        # На мобильных платформах проверяем только базовые зависимости
        dependencies = [
            ("kivy", "kivy"),
            ("numpy", "numpy"),
            ("reedsolo", "reedsolo"),
        ]
    else:
        print(f"Десктопная платформа: {sys.platform}")
        # На десктопе проверяем все зависимости
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
        if mobile:
            print("\nДля мобильных платформ установите:")
            print("  pip install kivy numpy reedsolo")
        else:
            print("\nУстановите их с помощью:")
            print("  pip install -r requirements.txt")
        return 1
    
    print("Все необходимые зависимости найдены!")
    
    # Проверяем доступность аудиобэкенда (только на десктопе)
    if not mobile:
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
        # Пытаемся использовать Kivy GUI (кроссплатформенный)
        from ofdm_gui_kivy import OfdmApp
        app = OfdmApp()
        app.run()
    except Exception as e:
        print(f"ОШИБКА при запуске Kivy GUI: {e}")
        # Пробуем Flet как запасной вариант
        try:
            print("Попытка запуска Flet GUI...")
            from ofdm_gui_flet import main as flet_main
            flet_main()
        except Exception as e2:
            print(f"ОШИБКА при запуске Flet GUI: {e2}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
