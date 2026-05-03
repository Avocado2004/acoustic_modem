#!/usr/bin/env python3
"""
ТОЧКА ВХОДА - гарантированно обновленная версия.
Этот файл принудительно очищает ВЕСЬ кэш и запускает обновленную версию.
"""

import sys
import os

# 1. Устанавливаем правильную директорию
project_dir = '/Users/mr_black/Acoustic_Modem'
os.chdir(project_dir)
sys.path.insert(0, project_dir)

print("="*70)
print("ACOUSTIC MODEM - ТОЧКА ВХОДА (обновленная версия)")
print("="*70)
print(f"Рабочая директория: {os.getcwd()}")
print(f"Python путь: {sys.path[0]}\n")

# 2. ПРИНУДИТЕЛЬНАЯ очистка ВЕСЕГО кэша
print("[1] ПРИНУДИТЕЛЬНАЯ очистка кэша...")
import shutil

# Удаляем ВСЕ __pycache__ директории
count_cache = 0
for root, dirs, files in os.walk('.'):
    for d in dirs:
        if d == '__pycache__':
            cache_path = os.path.join(root, d)
            try:
                shutil.rmtree(cache_path)
                count_cache += 1
            except:
                pass

# Удаляем ВСЕ .pyc файлы
count_pyc = 0
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.pyc'):
            try:
                os.remove(os.path.join(root, f))
                count_pyc += 1
            except:
                pass

print(f"    Удалено {count_cache} папок __pycache__ и {count_pyc} .pyc файлов")

# 3. Очистка sys.modules
print("\n[2] Очистка загруженных модулей...")
modules_to_remove = []
for key in list(sys.modules.keys()):
    if 'modem' in key or 'audio' in key or 'plot_utils' in key:
        modules_to_remove.append(key)

for mod in modules_to_remove:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"    Удален: {mod}")

# 4. Проверка файла modem_rx.py на диске
print("\n[3] Проверка файла modem_rx.py на диске...")
modem_rx_path = os.path.join(project_dir, 'modem_rx.py')

try:
    with open(modem_rx_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    file_size = os.path.getsize(modem_rx_path)
    print(f"    Размер файла: {file_size} байт")
    
    if 'Equalizer DISABLED' in content:
        print("    ❌ ОШИБКА: На диске СТАРАЯ версия!")
        print("    Пожалуйста, обновите файл modem_rx.py")
        sys.exit(1)
    else:
        print("    ✓ На диске НОВАЯ версия (без 'Equalizer DISABLED')")
    
    if 'eq_alpha' in content:
        print("    ✓ Новая логика с eq_alpha присутствует")
    else:
        print("    ❌ ОШИБКА: Новая логика НЕ найдена!")
        sys.exit(1)
        
    if 'Equalizer enabled for ALL' in content:
        print("    ✓ Строка 'Equalizer enabled for ALL packet lengths' найдена")
        
    # Показываем строки с eq_alpha
    print("\n    Строки с eq_alpha:")
    for i, line in enumerate(content.split('\n'), 1):
        if 'eq_alpha' in line and 'print' not in line:
            print(f"      строка {i}: {line.strip()}")
            
except Exception as e:
    print(f"    ❌ ОШИБКА: {e}")
    sys.exit(1)

# 5. Импорт с принудительной перезагрузкой
print("\n[4] Импорт обновленных модулей...")

# Сначала импортируем modem_rx
import importlib
import modem_rx
importlib.reload(modem_rx)
print(f"    ✓ modem_rx перезагружен из:")
print(f"      {modem_rx.__file__}")

# Проверяем, что загружен правильный файл
with open(modem_rx.__file__, 'r') as f:
    loaded_content = f.read()

if 'Equalizer DISABLED' in loaded_content:
    print("    ❌ ОШИБКА: Загружена СТАРАЯ версия!")
    print("    Попробуйте перезапустить терминал полностью!")
    sys.exit(1)
else:
    print("    ✓ Загружена ОБНОВЛЕННАЯ версия")

# 6. Запуск
print("\n" + "="*70)
print("ЗАПУСК РЕЖИМА LOOP (обновленная версия)")
print("="*70)
print("ВНИМАНИЕ: В логе НЕ должно быть 'Equalizer DISABLED'!")
print("Должно быть: 'Equalizer enabled for ALL packet lengths'")
print("="*70 + "\n")

try:
    from modem_cli import main
    importlib.reload(sys.modules['modem_cli'])
    sys.argv = ['modem_cli.py', '--loop', '--debug']
    main()
except SystemExit:
    pass
except Exception as e:
    print(f"\n❌ Ошибка при запуске: {e}")
    import traceback
    traceback.print_exc()
