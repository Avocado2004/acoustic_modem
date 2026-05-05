"""
Модуль водопадной диаграммы (waterfall) адаптивного эквалайзера.

Сохраняет изменение амплитуды |Hk| и фазы angle(Hk) по поднесущим
в PNG файл по окончании приёма (без отображения в окне).

Кроссплатформенный: работает на macOS, Windows, Linux, Android, iOS.
Использует matplotlib с бэкендом 'Agg' для headless рендеринга.
Если matplotlib недоступен — переходит в "тихий" режим (no-op).

Все комментарии — на русском языке.
"""

import numpy as np

# Проверяем доступность matplotlib
try:
    import matplotlib
    # Используем headless бэкенд 'Agg' — работает на всех платформах
    # без отображения окон (для сохранения в файл)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
    print("[EQ-WF] matplotlib доступен (Agg бэкенд), сохранение в PNG включено")
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    print("[EQ-WF] matplotlib недоступен, визуализация отключена (no-op режим)")


class EqualizerWaterfall:
    """
    Класс для накопления и сохранения водопадной диаграммы эквалайзера.
    
    Создаёт изображение с 2 подграфиками:
    - Верхний: амплитуда |Hk| в dB (водопад)
    - Нижний: фаза angle(Hk) в радианах (водопад)
    
    Ось X — индексы поднесущих (0..Nsub-1).
    Ось Y — время (номер символа/снимка).
    Цвет — значение амплитуды/фазы.
    
    Сохранение в PNG через метод save() или автоматически при закрытии.
    """

    def __init__(self, max_symbols=500, title_prefix="Equalizer Waterfall", save_on_close=False, save_filename=None):
        """
        Инициализация водопадной диаграммы.
        
        :param max_symbols: Максимальное количество символов в истории (FIFO буфер).
        :param title_prefix: Префикс заголовка графика.
        :param save_on_close: Если True — автоматически сохраняет при закрытии.
        :param save_filename: Имя файла для автосохранения (по умолчанию 'rx_equalizer_waterfall.png').
        """
        self.max_symbols = max_symbols
        self.title_prefix = title_prefix
        self.save_on_close = save_on_close
        self.save_filename = save_filename or "rx_equalizer_waterfall.png"
        self._Hk_snapshots = []  # Список снимков Hk (комплексные массивы)
        self._fig = None
        self._ax_amp = None
        self._ax_phase = None
        
        print(f"[EQ-WF] EqualizerWaterfall создан: max_symbols={max_symbols}, "
              f"PLOTTING_AVAILABLE={PLOTTING_AVAILABLE}, save_on_close={save_on_close}")

    def update(self, Hk_snapshot):
        """
        Добавляет новый снимок Hk в буфер.
        
        :param Hk_snapshot: Комплексный массив Hk длиной Nsub.
        """
        if Hk_snapshot is None:
            print("[EQ-WF] update() получил None, пропуск")
            return
        
        # Добавляем снимок в буфер
        self._Hk_snapshots.append(np.array(Hk_snapshot, dtype=complex).copy())
        
        # Ограничиваем размер буфера (FIFO)
        if len(self._Hk_snapshots) > self.max_symbols:
            self._Hk_snapshots = self._Hk_snapshots[-self.max_symbols:]
        
        print(f"[EQ-WF] Добавлен снимок #{len(self._Hk_snapshots)}, "
              f"|Hk| mean={np.mean(np.abs(Hk_snapshot)):.4f}, "
              f"phase mean={np.mean(np.angle(Hk_snapshot)):.4f}")

    def update_from_history(self, equalizer_history_list):
        """
        Принимает полную историю эквалайзера и загружает все снимки.
        
        :param equalizer_history_list: Список историй по пакетам.
            Каждый элемент — список массивов Hk для одного пакета.
            Может быть также плоским списком массивов Hk.
        """
        if not equalizer_history_list:
            print("[EQ-WF] update_from_history() получил пустую историю")
            return
        
        # Определяем формат: вложенный список (по пакетам) или плоский
        self._Hk_snapshots = []
        
        first_item = equalizer_history_list[0]
        if isinstance(first_item, (list, tuple)):
            # Вложенный список — собираем все состояния из всех пакетов
            for pkt_history in equalizer_history_list:
                if pkt_history and len(pkt_history) > 0:
                    for hk in pkt_history:
                        self._Hk_snapshots.append(np.array(hk, dtype=complex).copy())
        elif isinstance(first_item, np.ndarray):
            # Плоский список массивов Hk
            for hk in equalizer_history_list:
                self._Hk_snapshots.append(np.array(hk, dtype=complex).copy())
        else:
            print(f"[EQ-WF] Неизвестный формат истории: {type(first_item)}")
            return
        
        # Ограничиваем размер буфера
        if len(self._Hk_snapshots) > self.max_symbols:
            self._Hk_snapshots = self._Hk_snapshots[-self.max_symbols:]
        
        print(f"[EQ-WF] Загружено {len(self._Hk_snapshots)} снимков из истории")

    def save(self, filename=None):
        """
        Сохраняет водопадную диаграмму в PNG файл.
        
        Создаёт изображение с 2 подграфиками:
        - Верхний: амплитуда |Hk| в dB
        - Нижний: фаза angle(Hk) в радианах
        
        :param filename: Имя файла для сохранения. Если None — используется self.save_filename.
        :return: True если сохранение успешно, False в случае ошибки.
        """
        if not PLOTTING_AVAILABLE:
            print("[EQ-WF] save() пропущен — matplotlib недоступен")
            return False
        
        if len(self._Hk_snapshots) == 0:
            print("[EQ-WF] save() пропущен — нет снимков для сохранения")
            return False
        
        filename = filename or self.save_filename
        
        try:
            # Формируем массив данных: [n_symbols, n_subcarriers]
            Hk_matrix = np.array(self._Hk_snapshots)
            n_symbols, n_subcarriers = Hk_matrix.shape
            
            print(f"[EQ-WF] Сохранение водопада: {n_symbols} символов x {n_subcarriers} поднесущих -> {filename}")
            
            # Амплитуда в dB: 20 * log10(|Hk|)
            # Защищаемся от log(0)
            amplitude = np.abs(Hk_matrix)
            amplitude_db = 20.0 * np.log10(np.maximum(amplitude, 1e-12))
            
            # Фаза в радианах
            phase = np.angle(Hk_matrix)
            
            # Создаём фигуру с 2 подграфиками
            fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Верхний график (амплитуда)
            im_amp = ax_amp.imshow(
                amplitude_db.T,           # Транспонируем: строки = поднесущие, столбцы = время
                aspect='auto',
                origin='lower',
                extent=[0, n_symbols, 0, n_subcarriers],
                cmap='viridis',
                interpolation='nearest'
            )
            ax_amp.set_title(f"{self.title_prefix} - Amplitude |Hk| (dB)")
            ax_amp.set_xlabel("Symbol Index (Time)")
            ax_amp.set_ylabel("Subcarrier Index")
            fig.colorbar(im_amp, ax=ax_amp, label="|Hk| (dB)")
            
            # Нижний график (фаза)
            im_phase = ax_phase.imshow(
                phase.T,
                aspect='auto',
                origin='lower',
                extent=[0, n_symbols, 0, n_subcarriers],
                cmap='twilight',
                interpolation='nearest'
            )
            ax_phase.set_title(f"{self.title_prefix} - Phase angle(Hk) (rad)")
            ax_phase.set_xlabel("Symbol Index (Time)")
            ax_phase.set_ylabel("Subcarrier Index")
            fig.colorbar(im_phase, ax=ax_phase, label="Phase (rad)")
            
            # Общая компоновка
            fig.tight_layout()
            
            # Сохраняем в файл
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"[EQ-WF] Водопадная диаграмма сохранена: {filename}")
            
            # Закрываем фигуру для освобождения памяти
            plt.close(fig)
            
            return True
            
        except Exception as e:
            print(f"[EQ-WF] Ошибка сохранения: {e}")
            import traceback
            traceback.print_exc()
            return False

    def close(self):
        """
        Закрывает водопадную диаграмму.
        Если save_on_close=True — автоматически сохраняет перед закрытием.
        """
        if self.save_on_close:
            self.save()
        
        # Очищаем ресурсы
        self._fig = None
        self._ax_amp = None
        self._ax_phase = None
        print("[EQ-WF] EqualizerWaterfall закрыт")

    def get_snapshot_count(self):
        """
        Возвращает количество накопленных снимков Hk.
        
        :return: Число снимков в буфере.
        """
        return len(self._Hk_snapshots)

    def get_snapshots(self):
        """
        Возвращает копию накопленных снимков Hk.
        
        :return: Список комплексных массивов Hk.
        """
        return [hk.copy() for hk in self._Hk_snapshots]

    def clear(self):
        """
        Очищает буфер снимков.
        """
        self._Hk_snapshots.clear()
        print("[EQ-WF] Буфер снимков очищен")

    def __del__(self):
        """Деструктор — сохраняет и закрывает при удалении объекта."""
        self.close()
