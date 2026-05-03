#!/usr/bin/env python3
"""
Скрипт для добавления кода отрисовки созвездия в live_receive_and_process.
"""

def add_constellation_plot_to_live_receive():
    """Добавляет код отрисовки созвездия в функцию live_receive_and_process."""
    
    # Читаем файл
    with open('modem_rx.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Код для добавления (с правильными отступами - 8 пробелов для блока finally)
    constellation_code = """        # Отрисовка созвездия с градиентом (синий->красный) по порядку символов
        if PLOTTING_AVAILABLE:
            try:
                from plot_utils import plot_constellation
                global rx_constellation_symbols
                if rx_constellation_symbols:
                    print(f"[DEBUG] Вызов plot_constellation с {len(rx_constellation_symbols)} символами")
                    plot_constellation(rx_constellation_symbols, "RX Constellation (gradient)", use_gradient=True)
                    rx_constellation_symbols.clear()  # Очищаем после отрисовки
            except Exception as e:
                print(f"[RX] Ошибка при построении созвездия: {e}")
"""
    
    # Находим место для вставки - после блока построения графика AGC в блоке finally
    # Ищем строку "print(f\"[RX] Ошибка при построении графика AGC: {e}\")" в блоке finally
    insert_idx = None
    for i, line in enumerate(lines):
        if 'print(f"[RX] Ошибка при построении графика AGC: {e}")' in line:
            # Проверяем, что это в блоке finally (ищем ближайший блок finally выше)
            for j in range(i, -1, -1):
                if 'finally:' in lines[j]:
                    insert_idx = i + 1  # Вставляем после этой строки
                    break
            if insert_idx:
                break
    
    if insert_idx is None:
        print("Ошибка: не найдено место для вставки кода")
        return False
    
    print(f"Вставка кода после строки {insert_idx}: {lines[insert_idx-1].rstrip()}")
    
    # Вставляем код
    lines.insert(insert_idx, constellation_code)
    
    # Записываем файл
    with open('modem_rx.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Код отрисовки созвездия добавлен в live_receive_and_process")
    return True

if __name__ == '__main__':
    if add_constellation_plot_to_live_receive():
        print("Готово!")
    else:
        print("Ошибка при добавлении кода")
