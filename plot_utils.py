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
            
    except Exception as e:
        print(f"[PLOT] Error plotting final equalizer: {e}")