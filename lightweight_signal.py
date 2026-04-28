import numpy as np

# fftconvolve(in1, in2) - свертка через FFT
# Реализация через numpy.fft
# Примечание: scipy.signal.fftconvolve имеет параметр mode, который не реализован
def fftconvolve(in1, in2):
    n = len(in1) + len(in2) - 1
    fft_in1 = np.fft.fft(in1, n)
    fft_in2 = np.fft.fft(in2, n)
    product = fft_in1 * fft_in2
    return np.fft.ifft(product).real

# find_peaks(x, height=None, distance=None) - поиск локальных максимумов
# Реализация с фильтрацией по height и distance
def find_peaks(x, height=None, distance=None):
    x = np.array(x)
    if len(x) < 3:
        return np.array([])
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
    peaks = np.array(peaks)
    if height is not None:
        peaks = peaks[x[peaks] >= height]
    if distance is not None:
        filtered_peaks = []
        for p in peaks:
            if not filtered_peaks or p - filtered_peaks[-1] >= distance:
                filtered_peaks.append(p)
        peaks = np.array(filtered_peaks)
    return peaks

# correlate(a, v, mode='valid') - корреляция сигналов
# Реализация для mode='valid'
def correlate(a, v, mode='valid'):
    a = np.array(a)
    v = np.array(v)
    if mode == 'valid':
        output = np.zeros(len(a) - len(v) + 1)
        for i in range(output.shape[0]):
            output[i] = np.sum(a[i:i+len(v)] * v)
        return output
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

# medfilt(x, kernel_size=3) - медианная фильтрация
# Реализация с padding для краев
def medfilt(x, kernel_size=3):
    x = np.array(x)
    pad = kernel_size // 2
    padded = np.pad(x, (pad, pad), mode='reflect')
    filtered = np.zeros_like(x)
    for i in range(len(x)):
        window = padded[i:i+kernel_size]
        filtered[i] = np.median(window)
    return filtered