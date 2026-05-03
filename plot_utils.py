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

def plot_constellation(symb, title="Constellation"):
    """
    Plot or save constellation data.
    On mobile platforms, saves data to CSV instead of plotting.
    """
    if not PLOTTING_AVAILABLE or plt is None:
        # On mobile or when matplotlib is not available, save data to file
        try:
            import numpy as np
            data = np.column_stack((np.real(symb), np.imag(symb)))
            filename = title.replace(" ", "_") + "_data.csv"
            np.savetxt(filename, data, delimiter=",", header="I,Q", comments="")
            print(f"[PLOT] Constellation data saved to {filename}")
        except Exception as e:
            print(f"[PLOT] Failed to save constellation data: {e}")
        return

    try:
        import numpy as np
        plt.figure(figsize=(5, 5))
        plt.plot(np.real(symb), np.imag(symb), 'o', markersize=2, alpha=0.6)
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.title(title)
        plt.xlabel("In-phase")
        plt.ylabel("Quadrature")
        plt.grid(True)
        plt.axis('equal')
        # Save to file instead of showing
        try:
            filename = title.replace(" ", "_") + ".png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"[PLOT] Constellation saved to {filename}")
        except Exception as e:
            print(f"[PLOT] Failed to save constellation: {e}")
        plt.close()
    except Exception as e:
        print(f"[PLOT] Error plotting constellation: {e}")

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
        plt.figure(figsize=(10, 4))
        t = np.arange(len(signal_data)) / fs
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
        
        # 1. График сигнала во временной области
        plt.figure(figsize=(10, 4))
        t = np.arange(len(signal)) / fs
        plt.plot(t, signal, linewidth=0.5)
        plt.title(f"{title_prefix} Signal (Time Domain)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        fname1 = f"{title_prefix}_signal.png"
        plt.savefig(fname1, dpi=150, bbox_inches="tight")
        print(f"[PLOT] TX signal plot saved to {fname1}")
        plt.close()
        
        # 2. Спектр сигнала
        if len(signal) > 0:
            plt.figure(figsize=(10, 4))
            freqs = np.fft.fftfreq(len(signal), 1.0/fs)
            spectrum = np.abs(np.fft.fft(signal))
            mask = freqs >= 0
            plt.plot(freqs[mask], spectrum[mask], linewidth=0.5)
            plt.title(f"{title_prefix} Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.grid(True, alpha=0.3)
            fname2 = f"{title_prefix}_spectrum.png"
            plt.savefig(fname2, dpi=150, bbox_inches="tight")
            print(f"[PLOT] TX spectrum plot saved to {fname2}")
            plt.close()
        
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
