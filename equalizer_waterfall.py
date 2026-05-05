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
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    PLOTTING_AVAILABLE = True
    print("[EQ-WF] matplotlib доступен (Agg бэкенд), сохранение в PNG включено")
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    cm = None
    Normalize = None
    ColorbarBase = None
    print("[EQ-WF] matplotlib недоступен, визуализация отключена (no-op режим)")


class EqualizerWaterfall:
    """
    Класс для накопления и сохранения водопадной диаграммы эквалайзера.
    
    Создаёт изображение с 2 подграфиками:
    - Верхний: амплитуда |Hk| в dB (линейные графики по частоте)
    - Нижний: фаза angle(Hk) в радианах (линейные графики по частоте)
    
    Ось X — реальная частота в Гц (subc_inds * fs / Nfft).
    Каждая линия — один снимок эквалайзера (берётся каждый step-й снимок).
    Цвет линии — градиент от синего (старые) до красного (новые) через coolwarm.
    
    Сохранение в PNG через метод save() или автоматически при закрытии.
    """

    def __init__(self, subc_inds, fs, Nfft, step=10, max_symbols=500,
                 title_prefix="Equalizer Waterfall", save_on_close=False, save_filename=None):
        """
        Инициализация водопадной диаграммы.
        
        :param subc_inds: Индексы поднесущих (массив или список).
        :param fs: Частота дискретизации в Гц.
        :param Nfft: Размер FFT.
        :param step: Шаг выборки снимков для отрисовки (по умолчанию 10).
        :param max_symbols: Максимальное количество символов в истории (FIFO буфер).
        :param title_prefix: Префикс заголовка графика.
        :param save_on_close: Если True — автоматически сохраняет при закрытии.
        :param save_filename: Имя файла для автосохранения (по умолчанию 'rx_equalizer_waterfall.png').
        """
        self.subc_inds = np.array(subc_inds, dtype=int)
        self.fs = fs
        self.Nfft = Nfft
        self.step = step
        self.max_symbols = max_symbols
        self.title_prefix = title_prefix
        self.save_on_close = save_on_close
        self.save_filename = save_filename or "rx_equalizer_waterfall.png"
        self._Hk_snapshots = []  # Список снимков Hk (комплексные массивы)
        
        # Вычисляем реальные частоты поднесущих в Гц
        self._freqs = self.subc_inds * self.fs / self.Nfft
        
        print(f"[EQ-WF] EqualizerWaterfall создан: max_symbols={max_symbols}, "
              f"step={step}, n_subcarriers={len(subc_inds)}, "
              f"freq_range=[{self._freqs[0]:.1f}..{self._freqs[-1]:.1f}] Гц, "
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
        - Верхний: амплитуда |Hk| в dB (линейные графики по частоте)
        - Нижний: фаза angle(Hk) в радианах (линейные графики по частоте)
        
        Каждая линия — один снимок эквалайзера (берётся каждый step-й снимок).
        Цвет линии — градиент от синего (старые) до красного (новые) через coolwarm.
        
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
            # Выбираем каждый step-й снимок для отрисовки
            snapshots_to_plot = self._Hk_snapshots[::self.step]
            n_snapshots = len(snapshots_to_plot)
            
            print(f"[EQ-WF] Сохранение водопада: {len(self._Hk_snapshots)} снимков, "
                  f"отрисовывается {n_snapshots} (step={self.step}), "
                  f"{len(self.subc_inds)} поднесущих -> {filename}")
            
            # Цветовая карта coolwarm: синий -> белый -> красный
            cmap = cm.coolwarm
            
            # Нормализация для цветовой шкалы (по индексу снимка)
            norm = Normalize(vmin=0, vmax=max(n_snapshots - 1, 1))
            
            # Создаём фигуру с 3 подграфиками: амплитуда, фаза, colorbar
            fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(14, 9))
            
            # --- Верхний график: амплитуда |Hk| в dB ---
            for i, hk in enumerate(snapshots_to_plot):
                # Амплитуда в dB: 20 * log10(|Hk|), защита от log(0)
                amplitude = np.abs(hk)
                amplitude_db = 20.0 * np.log10(np.maximum(amplitude, 1e-12))
                
                # Цвет линии по градиенту
                color = cmap(norm(i))
                
                # Подпись только для первой и последней линии (чтобы не засорять)
                label = None
                if i == 0:
                    label = "First snapshot"
                elif i == n_snapshots - 1:
                    label = "Last snapshot"
                
                ax_amp.plot(self._freqs, amplitude_db, color=color, linewidth=0.8,
                           alpha=0.8, label=label)
            
            ax_amp.set_title(f"{self.title_prefix} - Amplitude |Hk| (dB)")
            ax_amp.set_xlabel("Frequency (Hz)")
            ax_amp.set_ylabel("|Hk| (dB)")
            ax_amp.grid(True, alpha=0.3)
            if n_snapshots > 1:
                ax_amp.legend(loc='upper right')
            
            # --- Нижний график: фаза angle(Hk) в радианах ---
            for i, hk in enumerate(snapshots_to_plot):
                # Фаза в радианах
                phase = np.angle(hk)
                
                # Цвет линии по градиенту (тот же что для амплитуды)
                color = cmap(norm(i))
                
                # Подпись только для первой и последней линии
                label = None
                if i == 0:
                    label = "First snapshot"
                elif i == n_snapshots - 1:
                    label = "Last snapshot"
                
                ax_phase.plot(self._freqs, phase, color=color, linewidth=0.8,
                             alpha=0.8, label=label)
            
            ax_phase.set_title(f"{self.title_prefix} - Phase angle(Hk) (rad)")
            ax_phase.set_xlabel("Frequency (Hz)")
            ax_phase.set_ylabel("Phase (rad)")
            ax_phase.grid(True, alpha=0.3)
            if n_snapshots > 1:
                ax_phase.legend(loc='upper right')
            
            # --- Colorbar для понимания какой цвет = какой момент времени ---
            # Создаём ось для colorbar справа от графиков
            fig.subplots_adjust(right=0.88, hspace=0.3)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
            cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('Snapshot index (time)')
            
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
