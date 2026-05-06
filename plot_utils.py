import sys
import os

# Определяем, является ли платформа мобильной
IS_MOBILE = False
if sys.platform in ['android', 'ios'] or 'ANDROID_ROOT' in os.environ or 'IPHONEOS_DEPLOYMENT_TARGET' in os.environ:
    IS_MOBILE = True

PLOTTING_AVAILABLE = False
plt = None

if not IS_MOBILE:
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-GUI backend
        import matplotlib.pyplot as plt
        PLOTTING_AVAILABLE = True
    except ImportError:
        print("[PLOT] matplotlib not available, plotting disabled")
        PLOTTING_AVAILABLE = False
else:
    print("[PLOT] Mobile platform detected, plotting disabled")


def compensate_phase(symbols, phases):
    """
    Компенсация фазового сдвига в символах.
    
    Вычитает из каждого символа соответствующую фазу поднесущей,
    умножая на exp(-j * phase). Это позволяет убрать начальный фазовый
    сдвиг (Schroeder, Habr) и увидеть только влияние ACE и других обработок.
    
    Параметры
    ----------
    symbols : array-like
        Массив комплексных символов (поднесущие)
    phases : array-like
        Массив фаз поднесущих (из modem_config.subc_phases)
    
    Возвращает
    -------
    numpy.ndarray
        Компенсированные символы
    
    Примеры использования
    ---------------------
    # Компенсация фазы
    compensated = compensate_phase(symbols, subc_phases)
    
    # Без компенсации (phases=None)
    original = compensate_phase(symbols, None)
    """
    import numpy as np
    
    symbols = np.asarray(symbols, dtype=complex)
    
    if phases is None or len(phases) == 0:
        print("[PHASE-COMP] No phases provided, returning original symbols")
        return symbols
    
    phases = np.asarray(phases)
    
    # Проверяем длины массивов
    n_symbols = len(symbols)
    n_phases = len(phases)
    
    if n_symbols != n_phases:
        print(f"[PHASE-COMP] Warning: symbols length ({n_symbols}) != phases length ({n_phases})")
        # Используем минимальную длину
        min_len = min(n_symbols, n_phases)
        symbols = symbols[:min_len]
        phases = phases[:min_len]
    
    # Компенсируем фазу: умножаем на exp(-j * phase)
    compensation = np.exp(-1j * phases)
    compensated = symbols * compensation
    
    print(f"[PHASE-COMP] Compensated {len(symbols)} symbols with phase compensation")
    
    return compensated


def plot_constellation(symb, title="Constellation", filename=None, use_gradient=False, modulation_type="", phase_compensation=None, max_points=5000, save_csv=True):
    """
    Отрисовка созвездия (constellation diagram) с опциональным градиентом цвета.
    
    Параметры
    ----------
    symb : array-like
        Список комплексных символов (I+Qj) для отрисовки
    title : str
        Заголовок графика
    filename : str, optional
        Имя файла для сохранения. Если None - генерируется из title
    use_gradient : bool
        Если True - используется градиент цвета от синего (начало) к красному (конец)
    modulation_type : str
        Тип модуляции ('BPSK' или 'QPSK') для информации в заголовке
    phase_compensation : array-like, optional
        Массив фаз для компенсации фазового сдвига. Если None - компенсация не применяется.
        Используется для визуализации созвездия без начального фазового сдвига (Schroeder/Habr).
    max_points : int, optional
        Максимальное количество точек для отрисовки (по умолчанию 5000).
        Если символов больше, они будут равномерно прорежены.
        Это ускоряет построение графика на больших данных.
    save_csv : bool, optional
        Если True (по умолчанию) — сохраняет данные созвездия в CSV файл.
        Отключите для ускорения работы при большом количестве символов.
    
    Возвращает
    -------
    bool
        True если график успешно сохранен, False в случае ошибки
    
    Примеры использования
    ---------------------
    # Простая отрисовка без градиента:
    plot_constellation(symbols, "RX Constellation")
    
    # С градиентом цвета:
    plot_constellation(symbols, "TX Constellation", use_gradient=True, modulation_type="QPSK")
    
    # С компенсацией фазы:
    plot_constellation(symbols, "TX OFDM (phase compensated)", phase_compensation=subc_phases)
    
    # С ограничением количества точек:
    plot_constellation(symbols, "RX Constellation", max_points=2000)
    
    # Без сохранения CSV (быстрее на больших данных):
    plot_constellation(symbols, "RX Constellation", save_csv=False)
    """
    if not PLOTTING_AVAILABLE or plt is None:
        # На мобильных платформах или при отсутствии matplotlib сохраняем данные в CSV
        try:
            import numpy as np
            data = np.column_stack((np.real(symb), np.imag(symb)))
            csv_filename = (filename if filename else title.replace(" ", "_")) + "_data.csv"
            np.savetxt(csv_filename, data, delimiter=",", header="I,Q", comments="")
            print(f"[PLOT] Constellation data saved to {csv_filename}")
        except Exception as e:
            print(f"[PLOT] Failed to save constellation data: {e}")
        return False
    
    try:
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        # Преобразуем в numpy массив если нужно
        symb = np.asarray(symb, dtype=complex)
        n_symbols = len(symb)
        
        # Применяем компенсацию фазы если переданы фазы
        if phase_compensation is not None:
            symb = compensate_phase(symb, phase_compensation)
            title = title + " [phase compensated]"
            print(f"[PLOT-CONSTELLATION] Phase compensation applied")
        
        # ОПТИМИЗАЦИЯ: прореживаем точки если их слишком много
        if n_symbols > max_points:
            print(f"[PLOT-CONSTELLATION] Downsampling from {n_symbols} to {max_points} points for faster rendering")
            # Равномерно выбираем индексы для прореживания
            indices_to_keep = np.linspace(0, n_symbols - 1, max_points, dtype=int)
            symb = symb[indices_to_keep]
            n_symbols = len(symb)
        
        print(f"[PLOT-CONSTELLATION] Plotting {n_symbols} symbols, gradient={use_gradient}, modulation={modulation_type}")
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if use_gradient and n_symbols > 1:
            # Создаем цветовую карту от синего к красному
            colors = [(0, 0, 1), (1, 0, 0)]  # синий -> красный
            cmap = LinearSegmentedColormap.from_list('blue_to_red', colors, N=256)
            
            # Индексы для градиента (от 0 до 1)
            indices = np.linspace(0, 1, n_symbols)
            
            # ОПТИМИЗАЦИЯ: рисуем все точки одним вызовом scatter с массивом цветов
            # Это намного быстрее, чем рисовать каждую точку отдельно в цикле
            colors_array = cmap(indices)
            ax.scatter(np.real(symb), np.imag(symb), c=colors_array, alpha=0.6, s=20, edgecolors='none')
            
            # Добавляем цветовую шкалу
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_symbols))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Symbol Index', shrink=0.8)
            cbar.set_ticks([0, n_symbols // 2, n_symbols])
            cbar.set_ticklabels(['Start', 'Middle', 'End'])
            
            print(f"[PLOT-CONSTELLATION] Gradient applied: blue (start) -> red (end)")
        else:
            # Простая отрисовка без градиента
            ax.plot(np.real(symb), np.imag(symb), 'o', markersize=2, alpha=0.6, color='blue')
            print(f"[PLOT-CONSTELLATION] Simple plot without gradient")
        
        # Настройка осей
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('In-phase (I)', fontsize=12)
        ax.set_ylabel('Quadrature (Q)', fontsize=12)
        
        # Формируем заголовок
        title_parts = [title]
        if modulation_type:
            title_parts.append(f"Modulation: {modulation_type}")
        title_parts.append(f"Symbols: {n_symbols}")
        full_title = " | ".join(title_parts)
        ax.set_title(full_title, fontsize=14)
        
        # Равные масштабы осей
        ax.set_aspect('equal')
        
        # Автоматическое масштабирование с отступами
        margin = 0.1
        x_range = np.max(np.real(symb)) - np.min(np.real(symb))
        y_range = np.max(np.imag(symb)) - np.min(np.imag(symb))
        x_margin = max(margin, x_range * 0.1)
        y_margin = max(margin, y_range * 0.1)
        ax.set_xlim(np.min(np.real(symb)) - x_margin, np.max(np.real(symb)) + x_margin)
        ax.set_ylim(np.min(np.imag(symb)) - y_margin, np.max(np.imag(symb)) + y_margin)
        
        plt.tight_layout()
        
        # Сохраняем график
        if filename is None:
            filename = title.replace(" ", "_") + ".png"
        
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"[PLOT-CONSTELLATION] Saved to {filename}")
            
            # Сохраняем данные в CSV если включено
            if save_csv:
                try:
                    csv_data = np.column_stack((np.real(symb), np.imag(symb)))
                    csv_filename = filename.replace(".png", "_data.csv")
                    np.savetxt(csv_filename, csv_data, delimiter=",", header="I,Q", comments="")
                    print(f"[PLOT-CONSTELLATION] Data saved to {csv_filename}")
                except Exception as csv_e:
                    print(f"[PLOT-CONSTELLATION] Failed to save CSV: {csv_e}")
            
            plt.close(fig)
            return True
        except Exception as e:
            print(f"[PLOT-CONSTELLATION] Failed to save: {e}")
            plt.close(fig)
            return False
            
    except Exception as e:
        print(f"[PLOT-CONSTELLATION] Error plotting constellation: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_constellation_csv(symb, filename="constellation_data.csv"):
    """
    Сохраняет данные созвездия в CSV файл.
    
    Параметры
    ----------
    symb : array-like
        Список комплексных символов (I+Qj)
    filename : str
        Имя файла для сохранения
        
    Возвращает
    -------
    bool
        True если данные успешно сохранены, False в случае ошибки
    """
    try:
        import numpy as np
        symb = np.asarray(symb, dtype=complex)
        data = np.column_stack((np.real(symb), np.imag(symb)))
        np.savetxt(filename, data, delimiter=",", header="I,Q", comments="")
        print(f"[PLOT-CONSTELLATION] Data saved to {filename}")
        return True
    except Exception as e:
        print(f"[PLOT-CONSTELLATION] Failed to save CSV: {e}")
        return False


def plot_signal(signal_data, fs, title="Signal", filename="signal.png"):
    """
    Plot or save signal data.
    On mobile platforms, saves data to CSV instead of plotting.
    """
    if not PLOTTING_AVAILABLE or plt is None:
        try:
            import numpy as np
            data = np.column_stack((np.arange(len(signal_data)) / fs, signal_data))
            fname = title.replace(" ", "_") + "_data.csv"
            np.savetxt(fname, data, delimiter=",", header="Time,Amplitude", comments="")
            print(f"[PLOT] Signal data saved to {fname}")
        except Exception as e:
            print(f"[PLOT] Failed to save signal data: {e}")
        return
     
    try:
        import numpy as np
        
        # Сохраняем данные в CSV параллельно с PNG
        t = np.arange(len(signal_data)) / fs
        data = np.column_stack((t, signal_data))
        csv_fname = title.replace(" ", "_") + "_data.csv"
        np.savetxt(csv_fname, data, delimiter=",", header="Time,Amplitude", comments="")
        print(f"[PLOT] Signal data saved to {csv_fname}")
        
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal_data, linewidth=0.5)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        try:
            fname = title.replace(" ", "_") + ".png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"[PLOT] Signal plot saved to {fname}")
        except Exception as e:
            print(f"[PLOT] Failed to save signal plot: {e}")
        plt.close()
    except Exception as e:
        print(f"[PLOT] Error plotting signal: {e}")


def plot_spectrum(freq, spectrum, title="Spectrum", filename="spectrum.png"):
    """
    Plot or save spectrum data.
    On mobile platforms, saves data to CSV instead of plotting.
    """
    if not PLOTTING_AVAILABLE or plt is None:
        try:
            import numpy as np
            data = np.column_stack((freq, spectrum))
            fname = title.replace(" ", "_") + "_data.csv"
            np.savetxt(fname, data, delimiter=",", header="Frequency,Power", comments="")
            print(f"[PLOT] Spectrum data saved to {fname}")
        except Exception as e:
            print(f"[PLOT] Failed to save spectrum data: {e}")
        return
     
    try:
        import numpy as np
        
        # Сохраняем данные в CSV параллельно с PNG
        data = np.column_stack((freq, spectrum))
        csv_fname = title.replace(" ", "_") + "_data.csv"
        np.savetxt(csv_fname, data, delimiter=",", header="Frequency,Power", comments="")
        print(f"[PLOT] Spectrum data saved to {csv_fname}")
        
        plt.figure(figsize=(10, 4))
        plt.plot(freq, spectrum, linewidth=0.5)
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.grid(True)
        try:
            fname = title.replace(" ", "_") + ".png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"[PLOT] Spectrum plot saved to {fname}")
        except Exception as e:
            print(f"[PLOT] Failed to save spectrum plot: {e}")
        plt.close()
    except Exception as e:
        print(f"[PLOT] Error plotting spectrum: {e}")


def plot_rx_equalizer(Hk, subc_inds, fs, Nfft, packet_idx=0):
    """
    Построить график эквалайзера для одного пакета.
    Сохраняет график амплитуды и фазы канала.
    На мобильных платформах сохраняет данные в CSV.
    """
    if not PLOTTING_AVAILABLE or plt is None:
        # Сохраняем данные в CSV
        try:
            import numpy as np
            freqs = subc_inds * fs / float(Nfft)
            amp = np.abs(Hk)
            phase = np.angle(Hk)
            data = np.column_stack((freqs, amp, phase))
            filename = f"rx_equalizer_pkt{packet_idx}_data.csv"
            np.savetxt(filename, data, delimiter=",", header="Frequency,Amplitude,Phase", comments="")
            print(f"[PLOT] Equalizer data saved to {filename}")
        except Exception as e:
            print(f"[PLOT] Failed to save equalizer data: {e}")
        return
    
    try:
        import numpy as np
        freqs = subc_inds * fs / float(Nfft)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Амплитуда
        ax1.plot(freqs, np.abs(Hk), 'o-', markersize=3)
        ax1.set_title(f"Equalizer Frequency Response - Packet {packet_idx}")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        
        # Фаза
        ax2.plot(freqs, np.angle(Hk), 'o-', markersize=3)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True)
        
        plt.tight_layout()
        
        try:
            filename = f"rx_equalizer_pkt{packet_idx}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"[PLOT] Equalizer plot saved to {filename}")
        except Exception as e:
            print(f"[PLOT] Failed to save equalizer plot: {e}")
        plt.close(fig)
    except Exception as e:
        print(f"[PLOT] Error plotting equalizer: {e}")


def plot_rx_equalizer_final(Hk_smooth_list, subc_inds, fs, Nfft):
    """
    Построить итоговый график эквалайзера на основе всех пакетов.
    Усредняет коэффициенты эквалайзера и сохраняет график.
    На мобильных платформах сохраняет данные в CSV.
    """
    if not Hk_smooth_list or len(Hk_smooth_list) == 0:
        print("[PLOT] No equalizer data to plot")
        return
    
    try:
        import numpy as np
        # Усредняем коэффициенты эквалайзера по всем пакетам
        Hk_array = np.array(Hk_smooth_list)
        Hk_mean = np.mean(Hk_array, axis=0)
        
        if not PLOTTING_AVAILABLE or plt is None:
            # Сохраняем данные в CSV
            freqs = subc_inds * fs / float(Nfft)
            amp = np.abs(Hk_mean)
            phase = np.angle(Hk_mean)
            data = np.column_stack((freqs, amp, phase))
            filename = "rx_equalizer_final_data.csv"
            np.savetxt(filename, data, delimiter=",", header="Frequency,Amplitude,Phase", comments="")
            print(f"[PLOT] Final equalizer data saved to {filename}")
            return
        
        freqs = subc_inds * fs / float(Nfft)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Амплитуда
        ax1.plot(freqs, np.abs(Hk_mean), 'o-', markersize=3, color='blue', label='Mean')
        ax1.set_title(f"Final Equalizer Frequency Response (averaged over {len(Hk_smooth_list)} packets)")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax1.legend()
        
        # Фаза
        ax2.plot(freqs, np.angle(Hk_mean), 'o-', markersize=3, color='blue', label='Mean')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        try:
            filename = "rx_equalizer_final.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"[PLOT] Final equalizer plot saved to {filename}")
        except Exception as e:
            print(f"[PLOT] Failed to save final equalizer plot: {e}")
        plt.close(fig)
        
        # Дополнительно сохраняем диаграмму созвездия эквалайзера
        try:
            plt.figure(figsize=(6, 6))
            plt.plot(np.real(Hk_mean), np.imag(Hk_mean), 'o', markersize=4)
            plt.axhline(0, color='grey', linewidth=0.5)
            plt.axvline(0, color='grey', linewidth=0.5)
            plt.title("Equalizer Complex Response (Final)")
            plt.xlabel("Real")
            plt.ylabel("Imaginary")
            plt.grid(True)
            plt.axis('equal')
            filename2 = "rx_equalizer_complex_final.png"
            plt.savefig(filename2, dpi=150, bbox_inches="tight")
            print(f"[PLOT] Equalizer complex response saved to {filename2}")
            plt.close()
        except Exception as e:
            print(f"[PLOT] Failed to save complex response: {e}")
            
        # Если есть история эквалайзера, строим график динамики
        try:
            from modem_rx import equalizer_history_list
            if equalizer_history_list and len(equalizer_history_list) > 0:
                print(f"[PLOT] Building equalizer dynamics plot from {len(equalizer_history_list)} packets")
                plot_equalizer_dynamics(equalizer_history_list, subc_inds, fs, Nfft, 
                                       title_prefix="Equalizer_Dynamics")
        except Exception as e:
            print(f"[PLOT] Error plotting equalizer dynamics: {e}")
            
    except Exception as e:
        print(f"[PLOT] Error plotting final equalizer: {e}")


def plot_equalizer_dynamics(equalizer_history_list, subc_inds, fs, Nfft, title_prefix="Equalizer Dynamics"):
    """
    Визуализация динамики работы адаптивного эквалайзера.
    Строит график "водопад" (waterfall) показывающий изменение амплитуды |Hk|
    по поднесущим во времени (по мере обработки символов).
    
    :param equalizer_history_list: Список историй эквалайзера по пакетам.
            Каждый элемент - список массивов Hk для одного пакета.
    :param subc_inds: Индексы поднесущих.
    :param fs: Частота дискретизации.
    :param Nfft: Размер FFT.
    :param title_prefix: Префикс для названия графика.
    """
    if not equalizer_history_list or len(equalizer_history_list) == 0:
        print("[PLOT] No equalizer dynamics data to plot")
        return
    
    try:
        import numpy as np
        
        # Собираем все состояния Hk в один массив
        all_states = []
        for pkt_history in equalizer_history_list:
            if pkt_history and len(pkt_history) > 0:
                all_states.extend(pkt_history)
        
        if len(all_states) == 0:
            print("[PLOT] No valid history states found")
            return
        
        # Преобразуем в массив: [n_symbols, n_subcarriers]
        Hk_dynamics = np.array(all_states)
        n_symbols, n_subcarriers = Hk_dynamics.shape
        
        freqs = subc_inds * fs / float(Nfft)
        
        # На мобильных платформах сохраняем данные в CSV
        if not PLOTTING_AVAILABLE or plt is None:
            # Сохраняем данные в CSV: Frequency, Symbol_0_Amplitude, Symbol_1_Amplitude, ...
            header = "Frequency," + ",".join([f"Symbol_{i}_Amplitude" for i in range(min(n_symbols, 100))])
            # Ограничиваем количество символов для CSV (чтобы файл не был слишком большим)
            max_symbols = min(n_symbols, 100)
            data = np.column_stack((freqs, np.abs(Hk_dynamics[:max_symbols, :]).T))
            filename = f"{title_prefix.replace(' ', '_')}_data.csv"
            np.savetxt(filename, data, delimiter=",", header=header, comments="")
            print(f"[PLOT] Equalizer dynamics data saved to {filename}")
            return
        
        # Строим графики динамики: амплитуда и фаза
        # Создаем фигуру с 4 подграфиками: 2 для амплитуды, 2 для фазы
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # --- АМПЛИТУДА ---
        # 1. Водопад: амплитуда |Hk| по поднесущим во времени
        amplitude_data = np.abs(Hk_dynamics)
        amp_min = np.min(amplitude_data)
        amp_max = np.max(amplitude_data)
        
        im1 = ax1.imshow(amplitude_data.T, aspect='auto', origin='lower',
                         extent=[0, n_symbols, freqs[0], freqs[-1]],
                         vmin=amp_min, vmax=amp_max, cmap='viridis')
        ax1.set_xlabel("Symbol Index")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_title(f"{title_prefix} - Amplitude Waterfall")
        plt.colorbar(im1, ax=ax1, label="|Hk|")
        
        # 2. Линии эволюции амплитуды
        n_lines = min(10, n_symbols)
        indices = np.linspace(0, n_symbols - 1, n_lines, dtype=int)
        
        for i, idx in enumerate(indices):
            alpha = 0.3 + 0.7 * (i / max(1, n_lines - 1))
            ax2.plot(freqs, np.abs(Hk_dynamics[idx, :]),
                    label=f"Sym {idx}", alpha=alpha, linewidth=1)
        
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude |Hk|")
        ax2.set_title(f"{title_prefix} - Amplitude Evolution")
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        # --- ФАЗА ---
        # 3. Водопад: фаза (angle) по поднесущим во времени
        phase_data = np.angle(Hk_dynamics)  # Фаза в радианах
        phase_min = np.min(phase_data)
        phase_max = np.max(phase_data)
        
        im2 = ax3.imshow(phase_data.T, aspect='auto', origin='lower',
                         extent=[0, n_symbols, freqs[0], freqs[-1]],
                         vmin=phase_min, vmax=phase_max, cmap='twilight')
        ax3.set_xlabel("Symbol Index")
        ax3.set_ylabel("Frequency (Hz)")
        ax3.set_title(f"{title_prefix} - Phase Waterfall")
        plt.colorbar(im2, ax=ax3, label="Phase (rad)")
        
        # 4. Линии эволюции фазы
        for i, idx in enumerate(indices):
            alpha = 0.3 + 0.7 * (i / max(1, n_lines - 1))
            ax4.plot(freqs, np.angle(Hk_dynamics[idx, :]),
                    label=f"Sym {idx}", alpha=alpha, linewidth=1)
        
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Phase (rad)")
        ax4.set_title(f"{title_prefix} - Phase Evolution")
        ax4.grid(True, alpha=0.3)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        
        try:
            filename = f"{title_prefix.replace(' ', '_')}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"[PLOT] Equalizer dynamics plot (amplitude + phase) saved to {filename}")
        except Exception as e:
            print(f"[PLOT] Failed to save dynamics plot: {e}")
        plt.close(fig)
        
        # Дополнительно: графики для выбранных поднесущих (амплитуда и фаза)
        try:
            fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 8))
            # Выбираем несколько поднесущих
            selected_indices = [0, n_subcarriers // 2, n_subcarriers - 1]
            
            # Амплитуда для выбранных поднесущих
            for idx in selected_indices:
                if idx < n_subcarriers:
                    ax5.plot(range(n_symbols), np.abs(Hk_dynamics[:, idx]),
                            label=f"Subc {subc_inds[idx]} ({freqs[idx]:.1f} Hz)")
            
            ax5.set_xlabel("Symbol Index")
            ax5.set_ylabel("Amplitude |Hk|")
            ax5.set_title(f"{title_prefix} - Amplitude for Selected Subcarriers")
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            # Фаза для выбранных поднесущих
            for idx in selected_indices:
                if idx < n_subcarriers:
                    ax6.plot(range(n_symbols), np.angle(Hk_dynamics[:, idx]),
                            label=f"Subc {subc_inds[idx]} ({freqs[idx]:.1f} Hz)")
            
            ax6.set_xlabel("Symbol Index")
            ax6.set_ylabel("Phase (rad)")
            ax6.set_title(f"{title_prefix} - Phase for Selected Subcarriers")
            ax6.grid(True, alpha=0.3)
            ax6.legend()
            
            plt.tight_layout()
            filename2 = f"{title_prefix.replace(' ', '_')}_selected.png"
            plt.savefig(filename2, dpi=150, bbox_inches="tight")
            print(f"[PLOT] Selected subcarriers dynamics (amplitude + phase) saved to {filename2}")
            plt.close(fig2)
        except Exception as e:
            print(f"[PLOT] Failed to save selected subcarriers plot: {e}")
            
    except Exception as e:
        print(f"[PLOT] Error plotting equalizer dynamics: {e}")


def plot_tx_diagrams(signal, fs, title_prefix="TX"):
    """
    Построить и сохранить диаграммы для переданного сигнала.
    Сохраняет графики временной области, спектра и гистограммы амплитуд.
    На мобильных платформах сохраняет данные в CSV вместо построения графиков.
    """
    if not PLOTTING_AVAILABLE or plt is None:
        # На мобильных платформах или при отсутствии matplotlib сохраняем данные
        try:
            import numpy as np
            # Сохраняем сигнал
            data = np.column_stack((np.arange(len(signal)) / fs, signal))
            fname = f"{title_prefix}_signal_data.csv"
            np.savetxt(fname, data, delimiter=",", header="Time,Amplitude", comments="")
            print(f"[PLOT] TX signal data saved to {fname}")
            
            # Сохраняем спектр
            if len(signal) > 0:
                freqs = np.fft.fftfreq(len(signal), 1.0/fs)
                spectrum = np.abs(np.fft.fft(signal))
                mask = freqs >= 0
                data_spec = np.column_stack((freqs[mask], spectrum[mask]))
                fname_spec = f"{title_prefix}_spectrum_data.csv"
                np.savetxt(fname_spec, data_spec, delimiter=",", header="Frequency,Power", comments="")
                print(f"[PLOT] TX spectrum data saved to {fname_spec}")
        except Exception as e:
            print(f"[PLOT] Failed to save TX data: {e}")
        return
     
    try:
        import numpy as np
        
        # Вычисляем общие данные для CSV
        t = np.arange(len(signal)) / fs
        freqs = np.fft.fftfreq(len(signal), 1.0/fs) if len(signal) > 0 else np.array([])
        spectrum = np.abs(np.fft.fft(signal)) if len(signal) > 0 else np.array([])
        mask = freqs >= 0 if len(freqs) > 0 else np.array([], dtype=bool)
        
        # Сохраняем данные сигнала в CSV
        data_signal = np.column_stack((t, signal))
        csv_fname1 = f"{title_prefix}_signal_data.csv"
        np.savetxt(csv_fname1, data_signal, delimiter=",", header="Time,Amplitude", comments="")
        print(f"[PLOT] TX signal data saved to {csv_fname1}")
        
        # 1. График сигнала во временной области
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal, linewidth=0.5)
        plt.title(f"{title_prefix} Signal (Time Domain)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        fname1 = f"{title_prefix}_signal.png"
        plt.savefig(fname1, dpi=150, bbox_inches="tight")
        print(f"[PLOT] TX signal plot saved to {fname1}")
        plt.close()
        
        # Сохраняем данные спектра в CSV
        if len(spectrum) > 0 and np.any(mask):
            data_spectrum = np.column_stack((freqs[mask], spectrum[mask]))
            csv_fname2 = f"{title_prefix}_spectrum_data.csv"
            np.savetxt(csv_fname2, data_spectrum, delimiter=",", header="Frequency,Power", comments="")
            print(f"[PLOT] TX spectrum data saved to {csv_fname2}")
        
        # 2. Спектр сигнала
        if len(signal) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(freqs[mask], spectrum[mask], linewidth=0.5)
            plt.title(f"{title_prefix} Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.grid(True, alpha=0.3)
            fname2 = f"{title_prefix}_spectrum.png"
            plt.savefig(fname2, dpi=150, bbox_inches="tight")
            print(f"[PLOT] TX spectrum plot saved to {fname2}")
            plt.close()
        
        # Сохраняем данные гистограммы в CSV
        counts, bin_edges = np.histogram(signal, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        data_hist = np.column_stack((bin_centers, counts))
        csv_fname3 = f"{title_prefix}_histogram_data.csv"
        np.savetxt(csv_fname3, data_hist, delimiter=",", header="Bin_Center,Count", comments="")
        print(f"[PLOT] TX histogram data saved to {csv_fname3}")
        
        # 3. Гистограмма амплитуд
        plt.figure(figsize=(8, 4))
        plt.hist(signal, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{title_prefix} Amplitude Distribution")
        plt.xlabel("Amplitude")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        fname3 = f"{title_prefix}_histogram.png"
        plt.savefig(fname3, dpi=150, bbox_inches="tight")
        print(f"[PLOT] TX histogram saved to {fname3}")
        plt.close()
        
    except Exception as e:
        print(f"[PLOT] Error plotting TX diagrams: {e}")


def plot_agc_per_symbol(agc_history_list, title="AGC per Symbol"):
    """
    Построить график значений AGC (Automatic Gain Control) для каждого символа.
    
    Параметры:
    - agc_history_list: список словарей с данными AGC по каждому символу.
      Каждый элемент содержит: symbol_idx, pkt_idx, frame_idx, cur_rms, est_rms, gain_sym
    - title: заголовок графика
    
    На настольных платформах строит график и сохраняет в PNG файл.
    На мобильных платформах или при недоступности matplotlib сохраняет данные в CSV.
    """
    if not agc_history_list or len(agc_history_list) == 0:
        print("[PLOT] No AGC data to plot")
        return
    
    try:
        import numpy as np
        
        # Извлекаем данные из списка
        symbol_indices = [item['symbol_idx'] for item in agc_history_list]
        cur_rms_values = [item['cur_rms'] for item in agc_history_list]
        est_rms_values = [item['est_rms'] for item in agc_history_list]
        gain_sym_values = [item['gain_sym'] for item in agc_history_list]
        
        # Проверяем, нужно ли использовать matplotlib
        if IS_MOBILE or not PLOTTING_AVAILABLE or plt is None:
            # Сохраняем данные в CSV (для мобильных платформ или если matplotlib недоступен)
            data = np.column_stack((symbol_indices, cur_rms_values, est_rms_values, gain_sym_values))
            filename = "agc_per_symbol_data.csv"
            np.savetxt(filename, data, delimiter=",", 
                         header="symbol_index,cur_rms,est_rms,gain_sym", comments="")
            print(f"[PLOT] AGC data saved to {filename}")
            return
        
        # Построение графика с использованием matplotlib
        plt.figure(figsize=(12, 6))
        
        # График cur_rms и est_rms (левая ось Y)
        ax1 = plt.gca()
        ax1.plot(symbol_indices, cur_rms_values, 'b-', linewidth=0.5, alpha=0.7, label='cur_rms (текущий RMS)')
        ax1.plot(symbol_indices, est_rms_values, 'g-', linewidth=0.5, alpha=0.7, label='est_rms (сглаженный RMS)')
        ax1.set_xlabel('Индекс символа')
        ax1.set_ylabel('RMS', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # График gain_sym (правая ось Y)
        ax2 = ax1.twinx()
        ax2.plot(symbol_indices, gain_sym_values, 'r-', linewidth=0.5, alpha=0.7, label='gain_sym (усиление)')
        ax2.set_ylabel('Усиление', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
        
        plt.title(title)
        plt.xlabel('Индекс символа')
        plt.grid(True, alpha=0.3)
        
        # Сохраняем график в файл
        filename = "agc_per_symbol.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"[PLOT] AGC per symbol plot saved to {filename}")
        plt.close()
        
        # Дополнительно сохраняем данные в CSV для анализа
        data = np.column_stack((symbol_indices, cur_rms_values, est_rms_values, gain_sym_values))
        csv_filename = "agc_per_symbol_data.csv"
        np.savetxt(csv_filename, data, delimiter=",", 
                     header="symbol_index,cur_rms,est_rms,gain_sym", comments="")
        print(f"[PLOT] AGC data also saved to {csv_filename}")
        
    except Exception as e:
        print(f"[PLOT] Error plotting AGC per symbol: {e}")
        import traceback
        traceback.print_exc()
