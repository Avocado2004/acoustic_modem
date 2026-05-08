"""
Модуль абстракции источника данных для унифицированного приёма.

Содержит два интерфейса:
  - Legacy (DataSource)       : read_snapshot / wait_for_samples — для signal_processor.py legacy-режима
  - Chunk  (ChunkDataSource)  : read_chunk / has_more / get_all_data — для signal_processor.py chunk-режима

Реализации:
  - FileDataSource      : legacy-чтение WAV-файла порциями
  - MicrophoneDataSource : legacy-чтение с микрофона через ring buffer
  - FastFileDataSource   : chunk-чтение WAV-файла целиком из памяти
  - LoopDataSource       : обёртка FastFileDataSource с информацией о loop-режиме

Все комментарии на русском языке.
"""

import time
import threading
from collections import deque

import numpy as np
from scipy.io import wavfile


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Размер чанка чтения (в сэмплах) — используется FileDataSource и FastFileDataSource
CHUNK_SIZE = 2048


# ===========================================================================
# Legacy интерфейс
# ===========================================================================

class DataSource:
    """
    Базовый класс для источника данных (legacy-интерфейс).

    Используется signal_processor.py в режиме legacy (read_snapshot/wait_for_samples).
    Также используется напрямую в receive_from_file() и receive_from_microphone().
    """

    def read_snapshot(self) -> np.ndarray:
        """
        Возвращает текущий снимок всех накопленных данных.

        Returns
        -------
        np.ndarray
            Массив накопленных сэмплов (float64, нормализованы к [-1.0, 1.0]).
        """
        raise NotImplementedError

    def wait_for_samples(self, needed: int, timeout: float = 10.0) -> bool:
        """
        Ожидает пока накопится needed сэмплов или истечёт таймаут.

        Parameters
        ----------
        needed : int
            Сколько сэмплов нужно накопить.
        timeout : float
            Максимальное время ожидания в секундах.

        Returns
        -------
        bool
            True если достаточно данных накоплено, False если таймаут.
        """
        raise NotImplementedError

    def is_active(self) -> bool:
        """
        Возвращает True если источник активен (ещё принимает данные).

        Returns
        -------
        bool
        """
        raise NotImplementedError

    def stop(self):
        """Останавливает источник данных."""
        pass

    def get_samples_read(self) -> int:
        """
        Возвращает количество прочитанных (накопленных) сэмплов.

        Returns
        -------
        int
        """
        raise NotImplementedError


class FileDataSource(DataSource):
    """
    Источник данных из WAV-файла (legacy-интерфейс).

    Читает WAV-файл через scipy.io.wavfile.read, нормализует к float64 [-1.0, 1.0].
    При вызове read_snapshot() возвращает данные порциями по CHUNK_SIZE,
    имитируя потоковое чтение.
    """

    def __init__(self, wav_path: str):
        """
        Инициализация и чтение WAV-файла.

        Parameters
        ----------
        wav_path : str
            Путь к WAV-файлу.

        Raises
        ------
        FileNotFoundError
            Если файл не найден.
        ValueError
            Если файл пустой или не может быть прочитан.
        """
        self.wav_path = wav_path
        self.active = True
        self.current_pos = 0  # Текущая позиция «воспроизведения»

        print(f"[FileDataSource] Loading file: {wav_path}")

        try:
            sample_rate, raw_data = wavfile.read(wav_path)
        except FileNotFoundError:
            print(f"[FileDataSource] ERROR: File not found: {wav_path}")
            raise
        except Exception as e:
            print(f"[FileDataSource] ERROR: Failed to read WAV file: {e}")
            raise ValueError(f"Cannot read WAV file {wav_path}: {e}")

        self.sample_rate = sample_rate

        # Если стерео — берём первый канал
        if raw_data.ndim > 1:
            print(f"[FileDataSource] Multi-channel audio ({raw_data.shape[1]} ch), using first channel")
            raw_data = raw_data[:, 0]

        # Нормализация к float64 [-1.0, 1.0]
        if np.issubdtype(raw_data.dtype, np.integer):
            max_val = float(np.iinfo(raw_data.dtype).max)
        elif np.issubdtype(raw_data.dtype, np.floating):
            max_val = 1.0
        else:
            max_val = float(np.max(np.abs(raw_data)))
            if max_val == 0:
                max_val = 1.0

        self.data = raw_data.astype(np.float64) / max_val
        self.total_samples = len(self.data)

        if self.total_samples == 0:
            print(f"[FileDataSource] WARNING: File is empty (0 samples)")
        else:
            rms = float(np.sqrt(np.mean(self.data ** 2)))
            print(f"[FileDataSource] Loaded: {self.total_samples} samples, "
                  f"SR={self.sample_rate}, RMS={rms:.6f}, "
                  f"min={self.data.min():.6f}, max={self.data.max():.6f}")

    def read_snapshot(self) -> np.ndarray:
        """
        Возвращает данные до текущей позиции (имитация накопления).

        Каждый вызов продвигает позицию на CHUNK_SIZE сэмплов.

        Returns
        -------
        np.ndarray
            Массив накопленных данных (копия).
        """
        if not self.active or self.current_pos >= self.total_samples:
            return self.data[:self.current_pos].copy()

        # Продвигаем позицию на CHUNK_SIZE
        self.current_pos = min(self.current_pos + CHUNK_SIZE, self.total_samples)
        snapshot = self.data[:self.current_pos].copy()

        # Отладочный вывод: проверяем сигнал в позиции 10720
        if self.current_pos > 10720:
            sig_10720 = np.sqrt(np.mean(self.data[10720:11720]**2))
        else:
            sig_10720 = 0
        print(f"[FileDataSource] read_snapshot: pos={self.current_pos}/{self.total_samples} sig_10720_rms={sig_10720:.6f}")
        return snapshot

    def wait_for_samples(self, needed: int, timeout: float = 10.0) -> bool:
        """
        Для файла — продвигает current_pos до needed (но не уменьшает).

        Файл полностью загружен в память в конструкторе, поэтому
        позиция продвигается мгновенно. Позиция НЕ уменьшается —
        это критически важно для корректного декодирования пакетов
        после первого.

        Parameters
        ----------
        needed : int
            Сколько сэмплов нужно.
        timeout : float
            Таймаут (игнорируется для файла).

        Returns
        -------
        bool
            True если в файле достаточно данных.
        """
        if needed <= 0:
            print(f"[FileDataSource] wait_for_samples: needed={needed} <= 0, returning True")
            return True

        # Если уже достаточно данных — сразу True
        if self.current_pos >= needed:
            print(f"[FileDataSource] wait_for_samples: already have enough "
                  f"(pos={self.current_pos} >= needed={needed}), returning True")
            return True

        # Продвигаем current_pos до needed (НЕ уменьшаем!)
        old_pos = self.current_pos
        self.current_pos = min(needed, self.total_samples)

        result = self.current_pos >= needed
        print(f"[FileDataSource] wait_for_samples: needed={needed}, "
              f"pos {old_pos} -> {self.current_pos}, result={result}, "
              f"total={self.total_samples}")
        return result

    def is_active(self) -> bool:
        """
        Файл считается активным, пока не прочитан до конца.

        Returns
        -------
        bool
        """
        active = self.active and self.current_pos < self.total_samples
        print(f"[FileDataSource] is_active: {active} (current_pos={self.current_pos}, "
              f"total={self.total_samples}, active_flag={self.active})")
        return active

    def stop(self):
        """Останавливает «воспроизведение» файла."""
        self.active = False
        print(f"[FileDataSource] Stopped at pos={self.current_pos}/{self.total_samples}")

    def get_samples_read(self) -> int:
        """
        Возвращает количество прочитанных сэмплов.

        Returns
        -------
        int
        """
        return self.current_pos


class MicrophoneDataSource(DataSource):
    """
    Источник данных с микрофона (legacy-интерфейс).

    Использует audio_backend.py для захвата с микрофона.
    Ring buffer на deque для накопления данных.
    Callback заполняет ring buffer.
    """

    def __init__(self, fs: int, channels: int = 1, chunk: int = CHUNK_SIZE):
        """
        Инициализация аудио потока.

        Parameters
        ----------
        fs : int
            Частота дискретизации (Гц).
        channels : int
            Количество каналов (по умолчанию 1 — моно).
        chunk : int
            Размер чанка (блока) аудио.
        """
        self.fs = fs
        self.channels = channels
        self.chunk = chunk

        self._buffer_lock = threading.Lock()
        self._ring = deque()
        self._total_samples_in_buffer = 0

        self._audio = None
        self.active = False

        # Получаем аудио бэкенд
        from audio_backend import get_audio_backend
        self._audio_backend = get_audio_backend()

        print(f"[MicrophoneDataSource] Initialized: fs={fs}, channels={channels}, "
              f"chunk={chunk}, backend={type(self._audio_backend).__name__}")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Callback функция для аудио потока.

        Parameters
        ----------
        indata : np.ndarray
            Входные аудиоданные.
        frames : int
            Количество фреймов.
        time_info : object
            Информация о времени от бэкенда.
        status : object
            Статус потока (ошибки и т.д.).
        """
        if status:
            print(f"[MicrophoneDataSource] Audio status: {status}")

        # Берём первый канал, конвертируем в float64
        samples = indata[:, 0].astype(np.float64)

        with self._buffer_lock:
            self._ring.append(samples)
            self._total_samples_in_buffer += samples.size

    def start(self):
        """Запускает аудио поток."""
        if self._audio_backend is None:
            print("[MicrophoneDataSource] ERROR: No audio backend available!")
            raise RuntimeError("No audio backend available")

        self._audio = self._audio_backend
        self._audio.start_stream(
            self._audio_callback,
            samplerate=self.fs,
            channels=self.channels,
            blocksize=self.chunk
        )
        self.active = True
        print(f"[MicrophoneDataSource] Microphone stream started: "
              f"fs={self.fs}, channels={self.channels}, chunk={self.chunk}")

    def read_snapshot(self) -> np.ndarray:
        """
        Возвращает текущий снимок буфера.

        Returns
        -------
        np.ndarray
            Все накопленные данные (копия).
        """
        with self._buffer_lock:
            if not self._ring:
                return np.array([], dtype=np.float64)
            arr = np.concatenate(list(self._ring))

        print(f"[MicrophoneDataSource] read_snapshot: {len(arr)} samples")
        return arr.copy()

    def wait_for_samples(self, needed: int, timeout: float = 10.0) -> bool:
        """
        Ожидает накопления needed сэмплов в реальном времени.

        Parameters
        ----------
        needed : int
            Сколько сэмплов нужно.
        timeout : float
            Максимальное время ожидания в секундах.

        Returns
        -------
        bool
            True если данные накоплены, False если таймаут.
        """
        if needed <= 0:
            return True

        start = time.time()
        iteration = 0
        while True:
            with self._buffer_lock:
                cur = self._total_samples_in_buffer

            if cur >= needed:
                elapsed = time.time() - start
                print(f"[MicrophoneDataSource] wait_for_samples: got {cur}/{needed} "
                      f"samples in {elapsed:.3f}s")
                return True

            if (time.time() - start) > timeout:
                print(f"[MicrophoneDataSource] wait_for_samples TIMEOUT: "
                      f"got {cur}/{needed} in {timeout}s")
                return False

            iteration += 1
            if iteration % 100 == 0:
                print(f"[MicrophoneDataSource] wait_for_samples: waiting... "
                      f"{cur}/{needed}, elapsed={time.time() - start:.1f}s")

            time.sleep(0.01)

    def is_active(self) -> bool:
        """
        Проверяет, активен ли аудио поток.

        Returns
        -------
        bool
        """
        try:
            result = self.active and self._audio is not None and self._audio.is_active()
        except Exception as e:
            print(f"[MicrophoneDataSource] is_active check failed: {e}")
            result = False

        print(f"[MicrophoneDataSource] is_active: {result}")
        return result

    def stop(self):
        """Останавливает аудио поток."""
        try:
            if self._audio is not None:
                self._audio.stop()
                print("[MicrophoneDataSource] Audio stream stopped")
        except Exception as e:
            print(f"[MicrophoneDataSource] Error stopping audio: {e}")
        finally:
            self.active = False

    def get_samples_read(self) -> int:
        """
        Возвращает количество накопленных сэмплов.

        Returns
        -------
        int
        """
        with self._buffer_lock:
            return self._total_samples_in_buffer


# ===========================================================================
# Новый Chunk интерфейс
# ===========================================================================

class ChunkDataSource:
    """
    Базовый класс для источника данных с chunk-интерфейсом.

    Используется signal_processor.py в chunk-режиме (read_chunk/has_more).
    Отдаёт данные чанками по CHUNK_SIZE сэмплов.
    """

    def read_chunk(self) -> np.ndarray:
        """
        Читает следующий чанк данных.

        Returns
        -------
        np.ndarray
            Массив сэмплов. Пустой массив если данных больше нет.
        """
        raise NotImplementedError

    def has_more(self) -> bool:
        """
        Есть ли ещё данные для чтения.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    def get_all_data(self) -> np.ndarray:
        """
        Возвращает все накопленные данные.

        Returns
        -------
        np.ndarray
        """
        raise NotImplementedError

    def get_samples_read(self) -> int:
        """
        Сколько сэмплов прочитано.

        Returns
        -------
        int
        """
        raise NotImplementedError

    def stop(self):
        """Останавливает источник."""
        pass


class FastFileDataSource(ChunkDataSource):
    """
    Быстрый источник данных из WAV-файла (chunk-интерфейс).

    Читает WAV-файл целиком в память.
    Отдаёт данные чанками по CHUNK_SIZE (2048).
    Минимальная задержка между чанками (0.001 сек).
    """

    def __init__(self, wav_path: str, sleep_time: float = 0.001):
        """
        Инициализация и чтение WAV-файла.

        Parameters
        ----------
        wav_path : str
            Путь к WAV-файлу.
        sleep_time : float
            Задержка между чанками (секунды). По умолчанию 0.001.

        Raises
        ------
        FileNotFoundError
            Если файл не найден.
        ValueError
            Если файл пустой или не может быть прочитан.
        """
        self.wav_path = wav_path
        self.sleep_time = sleep_time
        self._current_pos = 0
        self._total_samples = 0
        self._data = None
        self._samples_read = 0

        print(f"[FastFileDataSource] Loading file: {wav_path}")

        try:
            sample_rate, raw_data = wavfile.read(wav_path)
        except FileNotFoundError:
            print(f"[FastFileDataSource] ERROR: File not found: {wav_path}")
            raise
        except Exception as e:
            print(f"[FastFileDataSource] ERROR: Failed to read WAV file: {e}")
            raise ValueError(f"Cannot read WAV file {wav_path}: {e}")

        self.sample_rate = sample_rate

        # Если стерео — берём первый канал
        if raw_data.ndim > 1:
            print(f"[FastFileDataSource] Multi-channel audio ({raw_data.shape[1]} ch), "
                  f"using first channel")
            raw_data = raw_data[:, 0]

        # Нормализация к float64 [-1.0, 1.0]
        if np.issubdtype(raw_data.dtype, np.integer):
            max_val = float(np.iinfo(raw_data.dtype).max)
        elif np.issubdtype(raw_data.dtype, np.floating):
            max_val = 1.0
        else:
            max_val = float(np.max(np.abs(raw_data)))
            if max_val == 0:
                max_val = 1.0

        self._data = raw_data.astype(np.float64) / max_val
        self._total_samples = len(self._data)

        if self._total_samples == 0:
            print(f"[FastFileDataSource] WARNING: File is empty (0 samples)")
        else:
            rms = float(np.sqrt(np.mean(self._data ** 2)))
            print(f"[FastFileDataSource] Loaded: {self._total_samples} samples, "
                  f"SR={self.sample_rate}, RMS={rms:.6f}, "
                  f"min={self._data.min():.6f}, max={self._data.max():.6f}")

    def read_chunk(self) -> np.ndarray:
        """
        Читает следующий чанк данных.

        Returns
        -------
        np.ndarray
            Чанк данных (до CHUNK_SIZE сэмплов). Пустой массив если файл дочитан.
        """
        if self._current_pos >= self._total_samples:
            print(f"[FastFileDataSource] read_chunk: EOF (pos={self._current_pos}, "
                  f"total={self._total_samples})")
            return np.array([], dtype=np.float64)

        end = min(self._current_pos + CHUNK_SIZE, self._total_samples)
        chunk = self._data[self._current_pos:end].copy()
        self._current_pos = end
        self._samples_read = self._current_pos

        # Минимальная задержка между чанками
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

        print(f"[FastFileDataSource] read_chunk: {len(chunk)} samples, "
              f"pos={self._current_pos}/{self._total_samples}")
        return chunk

    def has_more(self) -> bool:
        """
        Есть ли ещё данные для чтения.

        Returns
        -------
        bool
        """
        result = self._current_pos < self._total_samples
        print(f"[FastFileDataSource] has_more: {result} "
              f"(pos={self._current_pos}/{self._total_samples})")
        return result

    def get_all_data(self) -> np.ndarray:
        """
        Возвращает все данные файла.

        Returns
        -------
        np.ndarray
        """
        print(f"[FastFileDataSource] get_all_data: {self._total_samples} samples")
        return self._data.copy()

    def get_samples_read(self) -> int:
        """
        Сколько сэмплов прочитано.

        Returns
        -------
        int
        """
        return self._samples_read

    def stop(self):
        """Останавливает чтение (устанавливает позицию в конец)."""
        self._current_pos = self._total_samples
        print(f"[FastFileDataSource] Stopped at pos={self._current_pos}/{self._total_samples}")


class LoopDataSource(ChunkDataSource):
    """
    Источник данных для loop-режима (chunk-интерфейс).

    Оборачивает FastFileDataSource, добавляя информацию о loop-режиме.
    Хранит оригинальные данные для сравнения.
    """

    def __init__(self, wav_path: str, sleep_time: float = 0.001):
        """
        Инициализация loop-источника данных.

        Parameters
        ----------
        wav_path : str
            Путь к WAV-файлу.
        sleep_time : float
            Задержка между чанками (секунды).
        """
        self._source = FastFileDataSource(wav_path, sleep_time=sleep_time)
        self._original_data = self._source.get_all_data()
        self._total_samples = self._source._total_samples

        print(f"[LoopDataSource] Initialized from: {wav_path}, "
              f"{self._total_samples} samples")

    def read_chunk(self) -> np.ndarray:
        """
        Читает следующий чанк данных из внутреннего источника.

        Returns
        -------
        np.ndarray
            Чанк данных. Пустой массив если данных больше нет.
        """
        chunk = self._source.read_chunk()
        print(f"[LoopDataSource] read_chunk: {len(chunk)} samples")
        return chunk

    def has_more(self) -> bool:
        """
        Есть ли ещё данные для чтения.

        Returns
        -------
        bool
        """
        result = self._source.has_more()
        print(f"[LoopDataSource] has_more: {result}")
        return result

    def get_all_data(self) -> np.ndarray:
        """
        Возвращает все данные (оригинальные, без изменений).

        Returns
        -------
        np.ndarray
        """
        print(f"[LoopDataSource] get_all_data: {len(self._original_data)} samples")
        return self._original_data.copy()

    def get_samples_read(self) -> int:
        """
        Сколько сэмплов прочитано.

        Returns
        -------
        int
        """
        return self._source.get_samples_read()

    def stop(self):
        """Останавливает внутренний источник."""
        self._source.stop()
        print("[LoopDataSource] Stopped")

    @property
    def original_data(self) -> np.ndarray:
        """
        Оригинальные данные loop-файла (для сравнения).

        Returns
        -------
        np.ndarray
        """
        return self._original_data.copy()

    @property
    def total_samples(self) -> int:
        """
        Общее количество сэмплов.

        Returns
        -------
        int
        """
        return self._total_samples
