# Документация модуля channel_simulator.py

## Обзор

Модуль `channel_simulator.py` предназначен для симуляции различных помех и искажений в акустическом канале связи. Он позволяет тестировать модем (например, `test_modem_simple.py`) без изменения его кода, добавляя реалистичные искажения, возникающие при передаче звука через воздух.

## Установка

Модуль не требует специальной установки, только наличие библиотеки numpy:
```bash
pip install numpy
```

## Основной класс: `ChannelSimulator`

```python
class ChannelSimulator(fs: int = 48000, seed: Optional[int] = None)
```

**Параметры конструктора:**
- `fs` (int): Частота дискретизации сигнала (Гц). По умолчанию 48000.
- `seed` (Optional[int]): Зерно для генератора псевдослучайных чисел. Используется для воспроизводимости результатов.

---

## Методы класса

### 1. `add_awgn(sig, snr_db)`
Добавляет белый гауссовский шум (AWGN) к сигналу.

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `snr_db` (float): Отношение сигнал/шум в дБ. Если `None` или `-inf`, шум не добавляется.

**Возвращает:** Сигнал с шумом (np.ndarray).

---

### 2. `apply_multipath(sig, delays_ms, gains)`
Применяет модель многолучевого распространения (Multipath).

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `delays_ms` (list[float]): Список задержек лучей в миллисекундах.
- `gains` (list[float]): Список коэффициентов усиления для каждого луча (сумма должна быть ~1).

**Возвращает:** Сигнал с многолучевостью (np.ndarray).

**Пример:**
```python
# Два луча: прямой (0 мс) и отраженный (2 мс) с затуханием
sim.apply_multipath(signal, delays_ms=[0.0, 2.0], gains=[0.7, 0.3])
```

---

### 3. `apply_frequency_offset(sig, offset_hz)`
Вносит сдвиг частоты (Доплеровский сдвиг).

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `offset_hz` (float): Сдвиг частоты в Гц (может быть отрицательным).

**Возвращает:** Сигнал со сдвигом частоты (np.ndarray).

---

### 4. `apply_phase_jitter(sig, jitter_std_rad)`
Вносит случайный шум фазы (Phase Jitter).

**Параметры:**
- `sig` (np.ndarray): Входной сигнал (комплексный или вещественный).
- `jitter_std_rad` (float): СКО фазового шума в радианах.

**Возвращает:** Сигнал с искаженной фазой (np.ndarray).

---

### 5. `apply_sample_jitter(sig, jitter_ms)`
Симулирует джиттер семплов (нестабильность тактового генератора).

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `jitter_ms` (float): Максимальная амплитуда джиттера в миллисекундах.

**Возвращает:** Сигнал с джиттером (np.ndarray).

---

### 6. `apply_fading(sig, fader_type='rayleigh', block_size=512)`
Симулирует замирания (Fading).

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `fader_type` (str): Тип замираний: `'rayleigh'` или `'rician'`.
- `block_size` (int): Размер блока для изменения коэффициента усиления.

**Возвращает:** Сигнал с замираниями (np.ndarray).

---

### 7. `apply_hard_clipping(sig, threshold)`
Применяет жесткое ограничение (hard clipping) сигнала.

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `threshold` (float): Порог clipping (абсолютное значение).

**Возвращает:** Сигнал с clipping (np.ndarray).

---

### 8. `apply_soft_clipping(sig, threshold, clip_type='tanh')`
Применяет мягкое ограничение (soft clipping) сигнала.

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `threshold` (float): Порог clipping.
- `clip_type` (str): Тип clipping:
  - `'tanh'`: гиперболический тангенс (плавный)
  - `'cubic'`: кубическая функция
  - `'arctan'`: арктангенс

**Возвращает:** Сигнал с soft clipping (np.ndarray).

---

### 9. `apply_distortion(sig, drive=1.0, type='even')`
Применяет нелинейные искажения (distortion).

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `drive` (float): Коэффициент искажений (0-1).
- `type` (str): Тип искажений:
  - `'even'`: четные гармоники (мягкий звук)
  - `'odd'`: нечетные гармоники (жесткий звук)
  - `'full'`: все гармоники

**Возвращает:** Сигнал с искажениями (np.ndarray).

---

### 10. `apply_amplitude_compression(sig, threshold=0.5, ratio=4.0, attack_ms=1.0, release_ms=10.0)`
Симулирует компрессию амплитуды (audio compressor).

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `threshold` (float): Порог компрессии (0-1).
- `ratio` (float): Соотношение компрессии (например, 4.0 для 4:1).
- `attack_ms` (float): Время атаки в миллисекундах.
- `release_ms` (float): Время восстановления в миллисекундах.

**Возвращает:** Сигнал после компрессии (np.ndarray).

---

### 11. `apply_random_frequency_response(sig, f_low=300.0, f_high=5000.0, smoothness=15)`
Применяет случайную частотную характеристику канала в заданной полосе частот.

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `f_low` (float): Нижняя граница частоты (Гц).
- `f_high` (float): Верхняя граница частоты (Гц).
- `smoothness` (int): Степень сглаживания случайной АЧХ (больше = плавнее).

**Возвращает:** Сигнал с искаженной частотной характеристикой (np.ndarray).

---

### 12. `apply_distance_attenuation(sig, distance_m, ref_distance=1.0, air_absorption_db_per_m=0.01)`
Симулирует затухание сигнала при удалении динамика от микрофона. Учитывает геометрическое затухание (обратный квадрат) и поглощение воздухом. Низкие и высокие частоты затухают сильнее при больших расстояниях.

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `distance_m` (float): Расстояние между динамиком и микрофоном (метры).
- `ref_distance` (float): Опорное расстояние (метры), на котором затухание считается номинальным.
- `air_absorption_db_per_m` (float): Поглощение звука воздухом (дБ/м).

**Возвращает:** Ослабленный сигнал (np.ndarray).

**Пример:**
```python
# Затухание на расстоянии 5 метров
sim.apply_distance_attenuation(signal, distance_m=5.0)
```

---

### 13. `apply_impulse_noise(sig, impulse_level=1.0, min_spacing_ms=50.0, max_spacing_ms=200.0, impulse_duration_ms=1.0)`
Добавляет импульсные помехи (короткие импульсы белого шума). Имитирует резкие щелчки или кратковременные всплески шума в канале.

**Параметры:**
- `sig` (np.ndarray): Входной сигнал.
- `impulse_level` (float): Амплитуда импульсов шума (среднеквадратичное отклонение).
- `min_spacing_ms` (float): Минимальное расстояние между импульсами в миллисекундах.
- `max_spacing_ms` (float): Максимальное расстояние между импульсами в миллисекундах.
- `impulse_duration_ms` (float): Длительность каждого импульса в миллисекундах.

**Возвращает:** Сигнал с импульсными помехами (np.ndarray).

**Пример:**
```python
# Добавление редких импульсных помех
noisy_sig = sim.apply_impulse_noise(
    signal, 
    impulse_level=0.5, 
    min_spacing_ms=100.0, 
    max_spacing_ms=500.0
)
```

---

## Вспомогательная функция: `simulate_channel`

```python
def simulate_channel(sig: np.ndarray, 
                     fs: int = 48000, 
                     snr_db: Optional[float] = None,
                     freq_offset_hz: float = 0.0,
                     multipath_delays: Optional[list] = None,
                     multipath_gains: Optional[list] = None,
                     phase_jitter_rad: float = 0.0,
                     seed: Optional[int] = None) -> np.ndarray
```

Универсальная функция для применения набора искажений канала за один вызов.

**Параметры:**
- `sig` (np.ndarray): Исходный сигнал.
- `fs` (int): Частота дискретизации.
- `snr_db` (Optional[float]): SNR в дБ (None для отключения шума).
- `freq_offset_hz` (float): Сдвиг частоты.
- `multipath_delays` (Optional[list]): Задержки лучей [мс].
- `multipath_gains` (Optional[list]): Усиления лучей.
- `phase_jitter_rad` (float): Шум фазы.
- `seed` (Optional[int]): Зерно для ГПСЧ.

**Возвращает:** Искаженный сигнал (np.ndarray).

---

## Примеры использования

### Пример 1: Базовое использование класса

```python
import numpy as np
from channel_simulator import ChannelSimulator

# Создание симулятора
sim = ChannelSimulator(fs=48000, seed=42)

# Генерация тестового сигнала
fs = 48000
t = np.arange(0, 1.0, 1.0/fs)
test_sig = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

# Применение искажений
# 1. Сдвиг частоты (Доплер)
shifted = sim.apply_frequency_offset(test_sig, offset_hz=5.0)

# 2. Многолучевость
multipath = sim.apply_multipath(shifted, delays_ms=[0.0, 2.0], gains=[0.7, 0.3])

# 3. Шум
noisy = sim.add_awgn(multipath, snr_db=20)

# 4. Затухание с расстоянием
final = sim.apply_distance_attenuation(noisy, distance_m=3.0)
```

### Пример 2: Использование универсальной функции

```python
from channel_simulator import simulate_channel
import numpy as np

fs = 48000
t = np.arange(0, 1.0, 1.0/fs)
signal = np.sin(2 * np.pi * 1000 * t)

# Применение комплексных искажений
distorted = simulate_channel(
    signal,
    fs=fs,
    snr_db=20,               # SNR 20 дБ
    freq_offset_hz=5.0,        # Сдвиг частоты 5 Гц
    multipath_delays=[0.0, 2.0], # Два луча
    multipath_gains=[0.7, 0.3],
    phase_jitter_rad=0.1,      # Шум фазы
    seed=42
)
```

### Пример 3: Симуляция перегрузки и искажений

```python
sim = ChannelSimulator(fs=48000)

# Мягкое ограничение (soft clipping)
soft = sim.apply_soft_clipping(signal, threshold=0.8, clip_type='tanh')

# Жесткое ограничение (hard clipping)
hard = sim.apply_hard_clipping(soft, threshold=0.9)

# Нелинейные искажения (distortion)
distorted = sim.apply_distortion(hard, drive=0.5, type='odd')

# Компрессия
compressed = sim.apply_amplitude_compression(distorted, threshold=0.6, ratio=4.0)
```

### Пример 4: Частотная характеристика и затухание

```python
sim = ChannelSimulator(fs=48000)

# Случайная АЧХ в полосе 300-5000 Гц
freq_resp = sim.apply_random_frequency_response(
    signal, 
    f_low=300.0, 
    f_high=5000.0,
    smoothness=20
)

# Затухание на разных расстояниях
for dist in [1.0, 3.0, 5.0, 10.0]:
    attenuated = sim.apply_distance_attenuation(
        freq_resp, 
        distance_m=dist,
        ref_distance=1.0,
        air_absorption_db_per_m=0.02
    )
    print(f"Distance: {dist}m, RMS: {np.sqrt(np.mean(attenuated**2)):.4f}")
```

### Пример 5: Импульсные помехи

```python
sim = ChannelSimulator(fs=48000, seed=42)

# Генерация сигнала
t = np.arange(0, 2.0, 1.0/48000)
signal = np.sin(2 * np.pi * 1000 * t)

# Добавление импульсных помех
# Импульсы будут появляться каждые 50-200 мс, длительностью 1 мс
noisy_signal = sim.apply_impulse_noise(
    signal,
    impulse_level=1.0,      # Высокая амплитуда импульсов
    min_spacing_ms=50.0,
    max_spacing_ms=200.0,
    impulse_duration_ms=1.0
)
```

---

## Рекомендации по настройке

### Для реалистичного моделирования акустического канала:

1. **Шум (SNR):** Используйте значения 15-25 дБ для тихого помещения, 5-10 дБ для шумной среды.

2. **Многолучевость:** Для комнаты используйте 2-4 луча с задержками 1-10 мс. Сумма усилений должна быть около 1.0.

3. **Сдвиг частоты:** Для движущихся объектов используйте 5-20 Гц. Для стационарных - 0-2 Гц.

4. **Затухание с расстоянием:** 
   - 1-2 м: минимальные искажения
   - 3-5 м: заметное ослабление высоких и низких частот
   - 10+ м: сильное "фильтрующее" действие

5. **Clipping и Distortion:** Используйте для симуляции перегрузки усилителя или динамика. Порог threshold обычно 0.7-0.9 от максимума.

6. **Частотная характеристика:** smoothness=10-20 дает плавные изменения. Меньшие значения дают более "рваный" спектр.

7. **Импульсные помехи:** Используйте для симуляции щелчков или кратковременных помех. `impulse_level` обычно 0.5-1.5, интервалы `min_spacing_ms` и `max_spacing_ms` зависят от интенсивности помех.

---

## Тестирование

Для быстрой проверки работы модуля выполните:
```bash
python channel_simulator.py
```

Это запустит встроенный тест, который применит базовые искажения к тестовому сигналу и выведет статистику.

---

## Интеграция с test_modem_simple.py

Модуль можно использовать для тестирования модема следующим образом:

```python
# В test_modem_simple.py после генерации сигнала tx_out
from channel_simulator import ChannelSimulator

sim = ChannelSimulator(fs=48000, seed=123)

# Применяем искажения перед сохранением или воспроизведением
tx_distorted = sim.apply_distance_attenuation(tx_out, distance_m=5.0)
tx_distorted = sim.add_awgn(tx_distorted, snr_db=20)

# Добавляем импульсные помехи
tx_distorted = sim.apply_impulse_noise(tx_distorted, impulse_level=0.3)

# Теперь используйте tx_distorted вместо tx_out
```

Модуль не требует изменения кода модема, что позволяет легко тестировать устойчивость системы к различным помехам.

---

## Использование в режиме Loop (modem_cli.py)

Симулятор канала интегрирован в режим Loop (`modem_cli.py -> run_loop()`). При запуске режима Loop пользователь может указать процент искажений (0-100%), что позволяет гибко настраивать условия тестирования.

### Комбинированный режим масштабирования

Применяется **комбинированный режим**:
1. **Основные искажения** (всегда применяются, интенсивность масштабируется линейно):
   - **AWGN (шум)**: SNR от 30 дБ (0%) до 15 дБ (100%)
   - **Многолучевость**: задержки [0.0, 2.0, 5.0] мс, усиления масштабируются от прямого луча (0%) до полного набора (100%)
   - **Сдвиг частоты**: от 0 Гц (0%) до 5 Гц (100%)

2. **Дополнительные искажения** (включаются при достижении порога):
   - **Soft clipping** (tanh, threshold=0.8): включается при >50%
   - **Distortion** (type='odd', drive=0.5): включается при >80%
   - **Impulse noise** (impulse_level=0.5): включается при >70%

### Пример работы в режиме Loop

```bash
$ python modem_cli.py
Режим работы — [T]ransmit, [R]eceive или [L]oop (по умолчанию T): L

[LOOP] Запуск режима Loop (генерация -> передача -> прием -> проверка)
Введите процент искажений (0-100%, по умолчанию 100%): 75

[LOOP] Установлен процент искажений: 75.0%
[LOOP] Генерация текста: K5dF8s...
[LOOP] Передача завершена. Сигнал сохранен в ofdm_acoustic_tx_with_noise.wav

[LOOP] Применение искажений канала (75.0%)...
[CHANNEL] Применение AWGN: SNR = 16.2 дБ (процент искажений: 75%)
[CHANNEL] Применение многолучевости: задержки = [0.0, 2.0, 5.0] мс, усиления = [0.65, 0.3, 0.075]
[CHANNEL] Применение сдвига частоты: 3.75 Гц
[CHANNEL] Применение soft clipping: threshold = 0.8, type = tanh
[CHANNEL] Применение impulse noise: level = 0.5
[LOOP] Искаженный сигнал сохранен в ofdm_acoustic_tx_with_noise.wav

[LOOP] Прием данных из WAV-файла...
[LOOP] ✅ УСПЕХ: Текст совпадает полностью!
```

### Параметры масштабирования

Функция `scale_param(min_val, max_val, percent)` используется для линейного масштабирования параметров:
- При 0% используется `min_val`
- При 100% используется `max_val`
- Промежуточные значения рассчитываются по формуле: `min_val + (max_val - min_val) * (percent / 100.0)`

### Отладочный вывод

При применении каждого искажения в консоль выводится информация с префиксом `[CHANNEL]`, содержащая тип искажения и его текущие параметры. Это позволяет отслеживать, какие искажения применяются при заданном проценте.
