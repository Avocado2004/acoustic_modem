"""
Утилиты для работы с WAV файлами без зависимости от scipy.
Реализация чтения и записи WAV файлов через стандартный модуль wave.
"""

from typing import Tuple, Optional, Union
import numpy as np
import wave
import struct
from pathlib import Path


def read(filename: Union[str, Path]) -> Tuple[int, np.ndarray]:
    """
    Чтение WAV файла.
    
    Parameters
    ----------
    filename : Union[str, Path]
        Путь к WAV файлу
        
    Returns
    -------
    Tuple[int, np.ndarray]
        Кортеж (частота_дискретизации, данные)
        Данные возвращаются как numpy массив с типом, соответствующим глубине битности
    """
    with wave.open(str(filename), 'rb') as wav_file:
        # Параметры файла
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()  # байт на семпл
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Чтение всех фреймов
        raw_data = wav_file.readframes(n_frames)
        
        # Преобразование в numpy массив в зависимости от глубины
        if sampwidth == 1:
            # 8-bit unsigned
            data = np.frombuffer(raw_data, dtype=np.uint8)
            data = data.astype(np.float64) - 128  # Преобразование в signed
        elif sampwidth == 2:
            # 16-bit signed
            data = np.frombuffer(raw_data, dtype=np.int16)
        elif sampwidth == 3:
            # 24-bit signed (специфический случай)
            data = _read_24bit(raw_data, n_frames * n_channels)
        elif sampwidth == 4:
            # 32-bit signed или float
            # Пытаемся определить по заголовку, но обычно это int32 или float32
            data = np.frombuffer(raw_data, dtype=np.int32)
        else:
            raise ValueError(f"Неподдерживаемая глубина битности: {sampwidth} байт")
        
        # Если стерео, меняем форму массива
        if n_channels > 1:
            data = data.reshape(-1, n_channels)
        
        return framerate, data


def write(filename: Union[str, Path], 
          rate: int, 
          data: np.ndarray,
          sampwidth: Optional[int] = None) -> None:
    """
    Запись WAV файла.
    
    Parameters
    ----------
    filename : Union[str, Path]
        Путь к выходному WAV файлу
    rate : int
        Частота дискретизации
    data : np.ndarray
        Данные для записи
    sampwidth : Optional[int]
        Глубина в байтах (1, 2, 3, 4). Если None, определяется по типу данных
    """
    data = np.asarray(data)
    
    # Определяем параметры
    if data.ndim == 1:
        n_channels = 1
    else:
        n_channels = data.shape[1]
    
    # Определяем sampwidth, если не задан
    if sampwidth is None:
        if data.dtype == np.int16:
            sampwidth = 2
        elif data.dtype == np.int32 or data.dtype == np.float32:
            sampwidth = 4
        elif data.dtype == np.uint8 or data.dtype == np.int8:
            sampwidth = 1
        else:
            # По умолчанию 16-bit
            sampwidth = 2
            data = (data * 32767).astype(np.int16)
    
    # Подготовка данных
    if sampwidth == 1:
        # 8-bit unsigned
        if data.dtype != np.uint8:
            data = np.clip(data, -128, 127)
            data = (data + 128).astype(np.uint8)
    elif sampwidth == 2:
        # 16-bit signed
        if data.dtype != np.int16:
            data = np.clip(data, -32768, 32767).astype(np.int16)
    elif sampwidth == 3:
        # 24-bit
        data = _prepare_24bit(data)
    elif sampwidth == 4:
        # 32-bit
        if data.dtype != np.int32:
            data = np.clip(data, -2147483648, 2147483647).astype(np.int32)
    
    # Запись
    with wave.open(str(filename), 'wb') as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(rate)
        
        if sampwidth == 3:
            wav_file.writeframes(data)
        else:
            wav_file.writeframes(data.tobytes())


def _read_24bit(raw_data: bytes, n_samples: int) -> np.ndarray:
    """
    Чтение 24-bit WAV данных.
    
    Parameters
    ----------
    raw_data : bytes
        Сырые байты
    n_samples : int
        Количество семплов
        
    Returns
    -------
    np.ndarray
        32-bit signed массив
    """
    data = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(n_samples):
        # 24-bit little-endian
        start = i * 3
        # Добавляем нулевой байт для преобразования в 32-bit
        bytes_24 = raw_data[start:start+3] + b'\x00'
        value = struct.unpack('<i', bytes_24)[0]
        # Сдвигаем знак, если нужно
        if value & 0x800000:
            value -= 0x1000000
        data[i] = value
    
    return data


def _prepare_24bit(data: np.ndarray) -> bytes:
    """
    Подготовка данных для записи в 24-bit формате.
    
    Parameters
    ----------
    data : np.ndarray
        Входные данные (будут приведены к 24-bit)
        
    Returns
    -------
    bytes
        Байты для записи
    """
    data = np.clip(data, -8388608, 8388607).astype(np.int32)
    result = bytearray()
    
    for value in data.flat:
        # Преобразуем в 24-bit little-endian
        packed = struct.pack('<i', value)[:3]  # Берем только 3 младших байта
        result.extend(packed)
    
    return bytes(result)


# Создаем объект wavfile для совместимости со scipy.io.wavfile
class WavFile:
    """Объект для совместимости с scipy.io.wavfile"""
    
    @staticmethod
    def read(filename: Union[str, Path]) -> Tuple[int, np.ndarray]:
        """Чтение WAV файла"""
        return read(filename)
    
    @staticmethod
    def write(filename: Union[str, Path], rate: int, data: np.ndarray) -> None:
        """Запись WAV файла"""
        write(filename, rate, data)


# Экспортируем объект wavfile
wavfile = WavFile()

__all__ = ['read', 'write', 'wavfile', 'WavFile']
