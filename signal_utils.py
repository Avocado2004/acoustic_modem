"""
Утилиты для обработки сигналов без зависимости от scipy.
Реализации основных функций scipy.signal через numpy и pure python.
"""

from typing import Tuple, Optional, List, Union, Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray


def fftconvolve(in1: ArrayLike, in2: ArrayLike, mode: str = 'full') -> NDArray[np.float64]:
    """
    Свертка через FFT (Fast Fourier Transform).
    
    Parameters
    ----------
    in1 : ArrayLike
        Первый входной массив
    in2 : ArrayLike
        Второй входной массив
    mode : str
        'full', 'same' или 'valid'
        
    Returns
    -------
    NDArray[np.float64]
        Результат свертки
    """
    in1 = np.asarray(in1, dtype=np.float64)
    in2 = np.asarray(in2, dtype=np.float64)
    
    n1 = len(in1)
    n2 = len(in2)
    n = n1 + n2 - 1
    
    # Ближайшая степень двойки для эффективности FFT
    nfft = 1
    while nfft < n:
        nfft *= 2
    
    # Прямое FFT
    f1 = np.fft.rfft(in1, n=nfft)
    f2 = np.fft.rfft(in2, n=nfft)
    
    # Умножение в частотной области
    result = np.fft.irfft(f1 * f2, n=nfft)
    
    # Обрезка до нужного размера
    result = result[:n]
    
    # Обработка режимов
    if mode == 'full':
        return result
    elif mode == 'same':
        # Центрирование результата
        start = (n - n1) // 2
        return result[start:start + n1]
    elif mode == 'valid':
        # Только полностью перекрывающаяся часть
        start = n2 - 1
        end = n1
        return result[start:end]
    else:
        raise ValueError(f"Неизвестный режим: {mode}")


def medfilt(x: ArrayLike, kernel_size: int = 3) -> NDArray[np.float64]:
    """
    Медианный фильтр.
    
    Parameters
    ----------
    x : ArrayLike
        Входной сигнал
    kernel_size : int
        Размер окна (должен быть нечетным)
        
    Returns
    -------
    NDArray[np.float64]
        Отфильтрованный сигнал
    """
    x = np.asarray(x, dtype=np.float64)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Делаем нечетным
    
    n = len(x)
    half = kernel_size // 2
    result = np.zeros_like(x)
    
    for i in range(n):
        # Границы окна
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = x[start:end]
        result[i] = np.median(window)
    
    return result


def correlate(in1: ArrayLike, in2: ArrayLike, mode: str = 'full') -> NDArray[np.float64]:
    """
    Взаимная корреляция через fftconvolve.
    
    Parameters
    ----------
    in1 : ArrayLike
        Первый входной массив
    in2 : ArrayLike
        Второй входной массив (будет развернут)
    mode : str
        'full', 'same' или 'valid'
        
    Returns
    -------
    NDArray[np.float64]
        Взаимная корреляция
    """
    in2 = np.asarray(in2, dtype=np.float64)
    # Разворачиваем второй массив для корреляции
    in2_reversed = in2[::-1].conj() if np.iscomplexobj(in2) else in2[::-1]
    return fftconvolve(in1, in2_reversed, mode=mode)


def find_peaks(x: ArrayLike, 
               height: Optional[float] = None,
               distance: Optional[int] = None,
               prominence: Optional[float] = None,
               width: Optional[int] = None) -> Tuple[NDArray[np.int64], dict]:
    """
    Поиск пиков в сигнале (упрощенная реализация).
    
    Parameters
    ----------
    x : ArrayLike
        Входной сигнал
    height : Optional[float]
        Минимальная высота пика
    distance : Optional[int]
        Минимальное расстояние между пиками
    prominence : Optional[float]
        Минимальная выраженность пика
    width : Optional[int]
        Минимальная ширина пика
        
    Returns
    -------
    Tuple[NDArray[np.int64], dict]
        Индексы пиков и словарь свойств
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    peaks = []
    
    # Находим локальные максимумы
    for i in range(1, n - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
    
    peaks = np.array(peaks, dtype=np.int64)
    
    # Фильтрация по высоте
    if height is not None:
        mask = x[peaks] >= height
        peaks = peaks[mask]
    
    # Фильтрация по расстоянию
    if distance is not None and len(peaks) > 0:
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered_peaks[-1] >= distance:
                filtered_peaks.append(p)
        peaks = np.array(filtered_peaks, dtype=np.int64)
    
    # Фильтрация по выраженности (prominence)
    if prominence is not None and len(peaks) > 0:
        prominences = []
        for p in peaks:
            # Простая оценка выраженности: высота пика минус среднее соседей
            left_min = np.min(x[max(0, p-10):p]) if p > 0 else x[p]
            right_min = np.min(x[p+1:min(n, p+11)]) if p < n-1 else x[p]
            prom = x[p] - max(left_min, right_min)
            prominences.append(prom)
        prominences = np.array(prominences)
        mask = prominences >= prominence
        peaks = peaks[mask]
    
    # Фильтрация по ширине (упрощенно)
    if width is not None and len(peaks) > 0:
        filtered_peaks = []
        for p in peaks:
            # Проверяем, что пик шире заданного значения
            w = 1
            while p - w >= 0 and p + w < n and x[p-w] < x[p] and x[p+w] < x[p]:
                w += 1
            if w >= width:
                filtered_peaks.append(p)
        peaks = np.array(filtered_peaks, dtype=np.int64)
    
    # Формируем словарь свойств
    properties = {
        'peak_heights': x[peaks] if len(peaks) > 0 else np.array([])
    }
    
    return peaks, properties


def firwin(numtaps: int, 
           cutoff: Union[float, ArrayLike], 
           window: str = 'hamming',
           pass_zero: bool = True) -> NDArray[np.float64]:
    """
    Проектирование КИХ (FIR) фильтра методом окон.
    
    Parameters
    ----------
    numtaps : int
        Количество отводов (должно быть нечетным)
    cutoff : Union[float, ArrayLike]
        Частота среза (нормализованная 0-1, где 1 = частота Найквиста)
    window : str
        Тип окна ('hamming', 'hann', 'blackman')
    pass_zero : bool
        True для фильтра нижних частот, False для фильтра верхних частот
        
    Returns
    -------
    NDArray[np.float64]
        Коэффициенты фильтра
    """
    if numtaps % 2 == 0:
        numtaps += 1  # Делаем нечетным для симметрии
    
    # Центральный индекс
    n = np.arange(numtaps)
    center = (numtaps - 1) / 2
    
    # Идеальная импульсная характеристика
    if pass_zero:
        # ФНЧ
        if np.isscalar(cutoff):
            cutoff = [cutoff]
        h = np.zeros(numtaps)
        for fc in np.atleast_1d(cutoff):
            h += 2 * fc * np.sinc(2 * fc * (n - center))
        h /= len(np.atleast_1d(cutoff))
    else:
        # ФВЧ
        fc = cutoff if np.isscalar(cutoff) else cutoff[0]
        h = -2 * fc * np.sinc(2 * fc * (n - center))
        h[(n - center) == 0] = 1 - 2 * fc
    
    # Применение окна
    if window == 'hamming':
        w = np.hamming(numtaps)
    elif window == 'hann':
        w = np.hanning(numtaps)
    elif window == 'blackman':
        w = np.blackman(numtaps)
    else:
        w = np.ones(numtaps)
    
    return h * w


def filtfilt(b: ArrayLike, 
             a: ArrayLike, 
             x: ArrayLike,
             padlen: Optional[int] = None) -> NDArray[np.float64]:
    """
    Фильтрация с нулевой фазой (упрощенная реализация).
    
    Parameters
    ----------
    b : ArrayLike
        Коэффициенты числителя
    a : ArrayLike
        Коэффициенты знаменателя
    x : ArrayLike
        Входной сигнал
    padlen : Optional[int]
        Длина дополнения (padding)
        
    Returns
    -------
    NDArray[np.float64]
        Отфильтрованный сигнал
    """
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Упрощенная реализация: используем прямую и обратную фильтрацию
    # Через свертку (предполагаем, что a[0] = 1 и фильтр только с b коэффициентами)
    
    if a[0] != 1:
        b = b / a[0]
        a = a / a[0]
    
    # Простая фильтрация через свертку (только для FIR)
    if len(a) == 1 or np.allclose(a[1:], 0):
        # FIR фильтр
        result = fftconvolve(x, b, mode='same')
        result = fftconvolve(result[::-1], b, mode='same')
        return result[::-1]
    else:
        # Для IIR используем упрощенный подход
        # Прямая фильтрация
        y = np.zeros_like(x)
        nb = len(b)
        na = len(a)
        
        for i in range(len(x)):
            y[i] = 0
            for j in range(nb):
                if i - j >= 0:
                    y[i] += b[j] * x[i - j]
            for j in range(1, na):
                if i - j >= 0:
                    y[i] -= a[j] * y[i - j]
            y[i] /= a[0]
        
        # Обратная фильтрация
        y_rev = y[::-1]
        y2 = np.zeros_like(y_rev)
        for i in range(len(y_rev)):
            y2[i] = 0
            for j in range(nb):
                if i - j >= 0:
                    y2[i] += b[j] * y_rev[i - j]
            for j in range(1, na):
                if i - j >= 0:
                    y2[i] -= a[j] * y2[i - j]
            y2[i] /= a[0]
        
        return y2[::-1]


def butter(N: int, 
          Wn: Union[float, ArrayLike],
          btype: str = 'low',
          analog: bool = False) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Фильтр Баттерворта (только 1-й порядок).
    
    Parameters
    ----------
    N : int
        Порядок фильтра (поддерживается только N=1)
    Wn : Union[float, ArrayLike]
        Частота среза (нормализованная 0-1)
    btype : str
        Тип фильтра: 'low' или 'high'
    analog : bool
        Аналоговый фильтр (не поддерживается, только цифровой)
        
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        Коэффициенты (b, a)
    """
    if N != 1:
        raise ValueError("Поддерживается только 1-й порядок (N=1)")
    
    if analog:
        raise ValueError("Аналоговые фильтры не поддерживаются")
    
    Wn = np.atleast_1d(Wn)[0]
    
    if btype == 'low':
        # ФНЧ 1-го порядка
        b = np.array([Wn, 0.0])
        a = np.array([1.0, Wn - 1.0])
    elif btype == 'high':
        # ФВЧ 1-го порядка
        b = np.array([1.0 - Wn, 1.0 - Wn])
        a = np.array([1.0, Wn - 1.0])
    else:
        raise ValueError(f"Неподдерживаемый тип фильтра: {btype}")
    
    # Нормализация так, чтобы a[0] = 1
    b = b / a[0]
    a = a / a[0]
    
    return b, a


def freqz(b: ArrayLike, 
          a: Optional[ArrayLike] = None,
          worN: int = 512,
          whole: bool = False) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Частотная характеристика фильтра.
    
    Parameters
    ----------
    b : ArrayLike
        Коэффициенты числителя
    a : Optional[ArrayLike]
        Коэффициенты знаменателя (по умолчанию [1])
    worN : int
        Количество точек частоты
    whole : bool
        Если True, от 0 до 2π, иначе от 0 до π
        
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.complex128]]
        Частоты (рад/семпл) и комплексная частотная характеристика
    """
    b = np.asarray(b, dtype=np.float64)
    if a is None:
        a = np.array([1.0])
    a = np.asarray(a, dtype=np.float64)
    
    # Частотная сетка
    if whole:
        w = np.linspace(0, 2 * np.pi, worN, endpoint=False)
    else:
        w = np.linspace(0, np.pi, worN)
    
    # Вычисление частотной характеристики
    # H(e^jw) = B(e^jw) / A(e^jw)
    # где B и A - это z-преобразования коэффициентов b и a
    
    # Используем DFT для вычисления
    jw = 1j * w[:, np.newaxis]
    z = np.exp(jw)
    
    # B(z) = sum(b[k] * z^(-k))
    B = np.dot(z ** -np.arange(len(b)), b)
    # A(z) = sum(a[k] * z^(-k))
    A = np.dot(z ** -np.arange(len(a)), a)
    
    h = B / A
    
    return w, h


def sosfiltfilt(sos: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """
    Фильтрация с секциями второго порядка (SOS) с нулевой фазой.
    
    Parameters
    ----------
    sos : ArrayLike
        Массив секций второго порядка shape (n_sections, 6)
        Каждая строка: [b0, b1, b2, a0, a1, a2]
    x : ArrayLike
        Входной сигнал
        
    Returns
    -------
    NDArray[np.float64]
        Отфильтрованный сигнал
    """
    sos = np.asarray(sos, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Прямая фильтрация через все секции
    y = x.copy()
    for section in sos:
        b = section[:3]
        a = section[3:]
        y = filtfilt(b, a, y)
    
    return y


# Для удобства импорта
__all__ = [
    'fftconvolve',
    'medfilt',
    'correlate',
    'find_peaks',
    'firwin',
    'filtfilt',
    'butter',
    'freqz',
    'sosfiltfilt'
]
