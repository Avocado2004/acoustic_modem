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
