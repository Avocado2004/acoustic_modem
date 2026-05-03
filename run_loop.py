#!/usr/bin/env python3
"""
Скрипт для гарантированного запуска ОБНОВЛЕННОЙ версии.
Принудительно очищает кэш и проверяет версию перед запуском.
"""

import sys
import os

# Устанавливаем правильную директорию
project_dir = '/Users/mr_black/Acoustic_Modem'
os.chdir(project_dir)
sys.path.insert(0, project_dir)

print("="*70)
print("ЗАПУСК ОБНОВЛЕННОЙ ВЕРСИИ (эквалайзер ВСЕГДА включен)")
print("="*70)

# 1. ПРИНУДИТЕЛЬНОЕ удаление ВСЕГО кэша
print("\n[1] Удаление ВСЕХ .pyc и __pycache__...")
import shutil

deleted_cache = 0
deleted_pyc = 0

for root, dirs, files in os.walk('.'):
    # Удаляем __pycache__
    for d in dirs:
        if d == '__pycache__':
            cache_path = os.path.join(root, d)
            try:
                shutil.rmtree(cache_path)
                deleted_cache += 1
                print(f"    Удален: {cache_path}")
            except:
                pass
    
    # Удаляем .pyc
    for f in files:
        if f.endswith('.pyc'):
            try:
                os.remove(os.path.join(root, f))
                deleted_pyc += 1
            except:
                pass

print(f"    Удалено {deleted_cache} папок __pycache__ и {deleted_pyc} .pyc файлов")

# 2. Очистка sys.modules
print("\n[2] Очистка загруженных модулей...")
modules_to_remove = []
for key in list(sys.modules.keys()):
    if 'modem' in key or 'audio' in key or 'plot_utils' in key:
        modules_to_remove.append(key)

for mod in modules_to_remove:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"    Удален: {mod}")

# 3. Проверка файла modem_rx.py НА ДИСКЕ
print("\n[3] Проверка файла modem_rx.py на диске...")
modem_rx_path = os.path.join(project_dir, 'modem_rx.py')
try:
    with open(modem_rx_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'Equalizer DISABLED' in content:
        print("    ❌ ОШИБКА: На диске СТАРАЯ версия!")
        print("    Пожалуйста, обновите файл modem_rx.py")
        sys.exit(1)
    else:
        print("    ✓ На диске НОВАЯ версия (без 'Equalizer DISABLED')")
    
    if 'eq_alpha' in content:
        print("    ✓ Присутствует новая логика с eq_alpha")
    else:
        print("    ❌ ОШИБКА: Новая логика НЕ найдена!")
        sys.exit(1)
        
except Exception as e:
    print(f"    ❌ ОШИБКА: {e}")
    sys.exit(1)

# 4. Импорт и проверка
print("\n[4] Импорт обновленных модулей...")
try:
    import modem_rx
    print(f"    ✓ modem_rx загружен из:")
    print(f"      {modem_rx.__file__}")
    
    # Проверяем, что это правильный файл
    with open(modem_rx.__file__, 'r') as f:
        loaded_content = f.read()
    
    if 'Equalizer DISABLED' in loaded_content:
        print("    ❌ ОШИБКА: Загружена СТАРАЯ версия!")
        print("    Попробуйте перезапустить терминал полностью!")
        sys.exit(1)
    else:
        print("    ✓ Загружена ОБНОВЛЕННАЯ версия")
        
except Exception as e:
    print(f"    ❌ ОШИБКА при импорте: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Запуск
print("\n" + "="*70)
print("ВНИМАНИЕ: В логе НЕ должно быть 'Equalizer DISABLED'!")
print("Должно быть: 'Equalizer enabled for ALL packet lengths'")
print("="*70 + "\n")

try:
    from modem_cli import main
    sys.argv = ['modem_cli.py', '--loop', '--debug']
    main()
except SystemExit:
    pass
except Exception as e:
    print(f"\n❌ Ошибка при запуске: {e}")
    import traceback
    traceback.print_exc()
