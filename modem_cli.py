"""
Модуль CLI для OFDM Acoustic Modem.
Содержит функции для режимов передачи, приема и тестирования в петле.
"""

import sys
import os
import random
import string
import glob
import numpy as np
import importlib

# Импортируем функции из других модулей
from modem_config import init_phases, fs, wavfile, MODULATION, Nfft, Ncp, Nsub, BITS_PER_SYMBOL
from modem_modulation import build_preamble
from modem_tx import transmit_text, transmit_file, _transmit_data
from signal_processor import receive_from_file
from rx_live import live_receive_and_process
from channel_simulator import ChannelSimulator


def scale_param(min_val, max_val, percent):
    """
    Масштабирует параметр от min_val до max_val в зависимости от процента искажений.
    
    Параметры:
    - min_val: значение при 0% искажений
    - max_val: значение при 100% искажений
    - percent: процент искажений (0-100)
    
    Возвращает:
    - промежуточное значение
    """
    return min_val + (max_val - min_val) * (percent / 100.0)


def print_distortion_params(percent, fs, data_size_bytes=1024):
    """
    Выводит в консоль все параметры искажений и конфигурации, которые будут применены.
    
    Параметры:
    - percent: процент искажений (0-100)
    - fs: частота дискретизации
    - data_size_bytes: размер передаваемых данных в байтах
    """
    from modem_config import MODULATION, Nfft, Ncp, Nsub, BITS_PER_SYMBOL
    
    print(f"\n[LOOP-PARAMS] ===== Параметры симуляции канала для {percent}% искажений =====")
    
    # Параметры конфигурации
    print(f"[LOOP-PARAMS] Параметры конфигурации:")
    print(f"[LOOP-PARAMS]   Частота дискретизации: {fs} Гц")
    print(f"[LOOP-PARAMS]   Тип модуляции: {MODULATION}")
    print(f"[LOOP-PARAMS]   Размер данных: {data_size_bytes} байт")
    print(f"[LOOP-PARAMS]   OFDM: Nfft={Nfft}, Ncp={Ncp}, Nsub={Nsub}")
    print(f"[LOOP-PARAMS]   Бит на символ: {BITS_PER_SYMBOL}")
    
    # Оценка длительности сигнала
    bits_per_ofdm = Nsub * BITS_PER_SYMBOL
    total_bits = data_size_bytes * 8
    num_ofdm_symbols = (total_bits + bits_per_ofdm - 1) // bits_per_ofdm  # округление вверх
    samples_per_ofdm = Nfft + Ncp
    total_samples = num_ofdm_symbols * samples_per_ofdm
    duration_sec = total_samples / fs
    
    print(f"[LOOP-PARAMS]   Оценка длительности сигнала:")
    print(f"[LOOP-PARAMS]     Всего бит данных: {total_bits}")
    print(f"[LOOP-PARAMS]     Бит на OFDM символ: {bits_per_ofdm}")
    print(f"[LOOP-PARAMS]     Количество OFDM символов: ~{num_ofdm_symbols}")
    print(f"[LOOP-PARAMS]     Длительность: ~{duration_sec:.3f} сек ({total_samples} samples)")
    
    # 1. AWGN: SNR от 30 дБ (0%) до 15 дБ (100%)
    snr_db = scale_param(30.0, 15.0, percent)
    print(f"[LOOP-PARAMS] 1. AWGN (Аддитивный белый гауссовский шум):")
    print(f"[LOOP-PARAMS]    SNR = {snr_db:.1f} дБ")
    
    # 2. Многолучевость
    print(f"[LOOP-PARAMS] 2. Многолучевость (Multipath):")
    if percent == 0:
        delays_ms = [0.0]
        gains = [1.0]
        print(f"[LOOP-PARAMS]    Задержки = {delays_ms} мс, усиления = {gains}")
    else:
        direct_gain = scale_param(1.0, 0.6, percent)
        multipath_gain1 = 0.3 if percent > 30 else 0.3 * (percent / 30.0)
        multipath_gain2 = 0.1 if percent > 60 else (0.1 * (percent / 60.0) if percent > 0 else 0.0)
        
        delays_ms = [0.0, 2.0, 5.0]
        gains = [direct_gain, multipath_gain1, multipath_gain2]
        print(f"[LOOP-PARAMS]    Задержки = {delays_ms} мс")
        print(f"[LOOP-PARAMS]    Усиления = {gains}")
        print(f"[LOOP-PARAMS]    Прямой луч: {delays_ms[0]} мс, усиление {gains[0]:.3f}")
        if percent > 30:
            print(f"[LOOP-PARAMS]    Луч 2: {delays_ms[1]} мс, усиление {gains[1]:.3f}")
        if percent > 60:
            print(f"[LOOP-PARAMS]    Луч 3: {delays_ms[2]} мс, усиление {gains[2]:.3f}")
    
    # 3. Сдвиг частоты
    freq_offset = scale_param(0.0, 5.0, percent)
    print(f"[LOOP-PARAMS] 3. Сдвиг частоты (Доплеровский сдвиг):")
    print(f"[LOOP-PARAMS]    Сдвиг = {freq_offset:.2f} Гц")
    
    # 4. Новые искажения: АЧХ и затухание
    print(f"[LOOP-PARAMS] 4. АЧХ и затухание с расстоянием:")
    if percent == 0:
        print(f"[LOOP-PARAMS]    ✓ АЧХ ровная (0% искажений)")
        print(f"[LOOP-PARAMS]    ✓ Расстояние 1 м (нет затухания)")
    else:
        distance = 1.0 + (5.0 - 1.0) * (percent / 100.0)
        print(f"[LOOP-PARAMS]    ✓ АЧХ случайная (до ±{12.0 * percent / 100.0:.1f} дБ)")
        print(f"[LOOP-PARAMS]    ✓ Расстояние {distance:.1f} м (затухание {1.0/distance:.2f}x)")
    
    # 5. Дополнительные искажения
    print(f"[LOOP-PARAMS] 5. Дополнительные искажения:")
    
    if percent > 50:
        threshold = 0.8
        print(f"[LOOP-PARAMS]    ✓ Soft clipping: threshold = {threshold}, type = tanh")
    else:
        print(f"[LOOP-PARAMS]    ✗ Soft clipping: НЕ применяется (нужно >50%)")
    
    if percent > 80:
        drive = 0.5
        print(f"[LOOP-PARAMS]    ✓ Distortion: type = odd, drive = {drive}")
    else:
        print(f"[LOOP-PARAMS]    ✗ Distortion: НЕ применяется (нужно >80%)")
    
    if percent > 70:
        impulse_level = 0.5
        print(f"[LOOP-PARAMS]    ✓ Impulse noise: level = {impulse_level}")
    else:
        print(f"[LOOP-PARAMS]    ✗ Impulse noise: НЕ применяется (нужно >70%)")
    
    print(f"[LOOP-PARAMS] ===========================================\n")


def apply_channel_distortions(sig, fs, percent, sim=None):
    """
    Применяет искажения канала в зависимости от процента искажений.
    
    Параметры:
    - sig: входной сигнал (может быть int16, float32, float64)
    - fs: частота дискретизации
    - percent: процент искажений (0-100)
    - sim: экземпляр ChannelSimulator (если None, создается новый)
    
    Возвращает:
    - искаженный сигнал (всегда float64 в диапазоне [-1.0, 1.0])
    """
    import numpy as np
    
    # При 0% искажений возвращаем нормализованный сигнал
    if percent == 0:
        print(f"[CHANNEL] Процент искажений 0% — сигнал не изменяется")
        # Нормализуем сигнал к диапазону [-1.0, 1.0] если он целочисленный
        if np.issubdtype(sig.dtype, np.integer):
            sig_float = sig.astype(np.float64) / np.iinfo(sig.dtype).max
        else:
            sig_float = sig.astype(np.float64)
        return sig_float
    
    # Преобразуем сигнал к float64 для избежания переполнения
    if np.issubdtype(sig.dtype, np.integer):
        # Целочисленный сигнал: нормализуем к [-1.0, 1.0]
        result = sig.astype(np.float64) / np.iinfo(sig.dtype).max
        print(f"[CHANNEL] Сигнал преобразован из {sig.dtype} в float64 (нормализация)")
    else:
        result = sig.astype(np.float64)
    
    print(f"[CHANNEL-DBG] Диапазон сигнала до искажений: [{np.min(result):.6f}, {np.max(result):.6f}]")
    print(f"[CHANNEL-DBG] RMS сигнала до искажений: {np.sqrt(np.mean(result**2)):.6f}")
    
    # Используем переданный экземпляр или создаем новый
    if sim is None:
        sim = ChannelSimulator(fs=fs)
    
    # Основные искажения (всегда применяются, интенсивность масштабируется)
    
    # 1. AWGN: SNR от 30 дБ (0%) до 15 дБ (100%)
    snr_db = scale_param(30.0, 15.0, percent)
    print(f"[CHANNEL] Применение AWGN: SNR = {snr_db:.1f} дБ (процент искажений: {percent}%)")
    result = sim.add_awgn(result, snr_db)
    
    # 2. Многолучевость: задержки [0.0, 2.0, 5.0] мс, усиления [0.6, 0.3, 0.1]
    # При 0% — только прямой луч (0.0 мс, 1.0), при 100% — полный набор
    if percent == 0:
        delays_ms = [0.0]
        gains = [1.0]
    else:
        # Масштабируем усиления от прямого луча к полному набору
        direct_gain = scale_param(1.0, 0.6, percent)
        # Остальные лучи появляются пропорционально проценту
        multipath_gain1 = 0.3 if percent > 30 else 0.3 * (percent / 30.0)
        multipath_gain2 = 0.1 if percent > 60 else (0.1 * (percent / 60.0) if percent > 0 else 0.0)
        
        delays_ms = [0.0, 2.0, 5.0]
        gains = [direct_gain, multipath_gain1, multipath_gain2]
    
    print(f"[CHANNEL] Применение многолучевости: задержки = {delays_ms} мс, усиления = {gains}")
    result = sim.apply_multipath(result, delays_ms, gains)
    
    # 3. Сдвиг частоты: от 0 Гц (0%) до 5 Гц (100%)
    freq_offset = scale_param(0.0, 5.0, percent)
    print(f"[CHANNEL] Применение сдвига частоты: {freq_offset:.2f} Гц")
    result = sim.apply_frequency_offset(result, freq_offset)
    
    # Новые искажения: сначала АЧХ (искажение формы спектра), потом затухание (уменьшение амплитуды)
    
    # 4. Случайная АЧХ: применяется в полосе частот (300-5000 Гц)
    print(f"[CHANNEL] Применение случайной АЧХ (percent={percent}%)")
    result = sim.apply_random_frequency_response(result, percent=percent)
    print(f"[CHANNEL-DBG] RMS после АЧХ: {np.sqrt(np.mean(result**2)):.6f}")
    
    # 5. Затухание с расстоянием: уменьшение уровня сигнала
    print(f"[CHANNEL] Применение затухания с расстоянием (percent={percent}%)")
    result = sim.apply_distance_attenuation(result, percent=percent)
    print(f"[CHANNEL-DBG] RMS после затухания: {np.sqrt(np.mean(result**2)):.6f}")
    
    # Дополнительные искажения (включаются при достижении порога)
    
    # 6. Soft clipping (tanh, threshold=0.8): включается при >50%
    if percent > 50:
        threshold = 0.8
        print(f"[CHANNEL] Применение soft clipping: threshold = {threshold}, type = tanh")
        result = sim.apply_soft_clipping(result, threshold, clip_type='tanh')
    
    # 7. Distortion (type='odd', drive=0.5): включается при >80%
    if percent > 80:
        drive = 0.5
        print(f"[CHANNEL] Применение distortion: type = odd, drive = {drive}")
        result = sim.apply_distortion(result, drive=drive, type='odd')
    
    # 8. Impulse noise (impulse_level=0.5): включается при >70%
    if percent > 70:
        impulse_level = 0.5
        print(f"[CHANNEL] Применение impulse noise: level = {impulse_level}")
        result = sim.apply_impulse_noise(result, impulse_level=impulse_level)
    
    print(f"[CHANNEL-DBG] Диапазон сигнала после всех искажений: [{np.min(result):.6f}, {np.max(result):.6f}]")
    print(f"[CHANNEL-DBG] RMS сигнала после всех искажений: {np.sqrt(np.mean(result**2)):.6f}")
    
    # Нормализуем сигнал обратно в диапазон [-1.0, 1.0] чтобы избежать переполнения при сохранении в int16
    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        print(f"[CHANNEL] Нормализация сигнала: деление на {max_val:.6f}")
        result = result / max_val
        print(f"[CHANNEL-DBG] После нормализации: [{np.min(result):.6f}, {np.max(result):.6f}]")
    
    return result


def run_transmit():
    """Режим передачи: файл или текст."""
    mode = input("Режим передачи — [F]ile или [T]ext (по умолчанию F): ").strip().upper()
    mode = "T" if mode == "T" else "F"
    
    # НОВОЕ: Выбор модуляции
    mod = input("Модуляция — [B]PSK или [Q]PSK (по умолчанию Q): ").strip().upper()
    modulation = "BPSK" if mod == "B" else "QPSK"
    
    # Устанавливаем глобальные настройки модуляции
    import modem_config
    modem_config.MODULATION = modulation
    if modulation == "BPSK":
        modem_config.BITS_PER_SYMBOL = 1
    else:
        modem_config.BITS_PER_SYMBOL = 2
    modem_config.BITS_PER_OFDM_SYMBOL = modem_config.Nsub * modem_config.BITS_PER_SYMBOL
    
    print(f"[TX] Используется модуляция: {modulation}")
    
    if mode == "F":
        file_path = input("Путь к файлу для передачи: ").strip()
        transmit_file(file_path)
    else:
        text = input("Введите текст для передачи: ").strip()
        transmit_text(text)


def run_receive():
    """Режим приема: из файла или с микрофона."""
    src = input("Откуда демодулировать? (file/mic): ").strip().lower()
    if src == "file":
        wav_path = input("Путь к WAV-файлу для приёма: ").strip()
        success = receive_from_file(wav_path)
        if not success:
            print("[MAIN] Ошибка при приеме из файла.")
            sys.exit(1)
    else:
        live_receive_and_process()
        sys.exit(0)


def run_loop():
    """Режим Loop: генерация -> передача -> прием -> проверка."""
    print("\n[LOOP] Запуск режима Loop (генерация -> передача -> прием -> проверка)")
    
    # Запрос типа модуляции
    print("\n[LOOP] Выберите тип модуляции:")
    print("  1. BPSK (1 бит на символ)")
    print("  2. QPSK (2 бита на символ, по умолчанию)")
    modulation_input = input("Ваш выбор (1/2, по умолчанию 2): ").strip()
    
    if modulation_input == '1':
        modulation_type = "BPSK"
    else:
        modulation_type = "QPSK"  # QPSK по умолчанию
    
    print(f"[LOOP] Выбрана модуляция: {modulation_type}")
    
    # Применяем выбранную модуляцию
    from modem_config import set_modulation
    success = set_modulation(modulation_type)
    if not success:
        print("[LOOP-ERR] Ошибка при установке модуляции. Используется QPSK по умолчанию.")
        set_modulation("QPSK")
    
    # Обновляем локальные ссылки на параметры конфигурации
    from modem_config import init_phases, fs, wavfile, MODULATION, Nfft, Ncp, Nsub, BITS_PER_SYMBOL
    
    # Переинициализируем фазы с новой модуляцией
    init_phases()
    
    print(f"[LOOP] Модуляция применена: {MODULATION}, BITS_PER_SYMBOL={BITS_PER_SYMBOL}")
    
    # Запрос процента искажений
    percent_input = input("Введите процент искажений (0-100%, по умолчанию 100%): ").strip()
    if not percent_input:
        percent = 100.0
    else:
        try:
            # Удаляем символ % если он есть
            cleaned_input = percent_input.replace('%', '').strip()
            percent = float(cleaned_input)
            if percent < 0:
                percent = 0.0
            elif percent > 100:
                percent = 100.0
        except ValueError:
            print("[LOOP] Неверный ввод, используется 100% по умолчанию")
            percent = 100.0
    
    print(f"[LOOP] Установлен процент искажений: {percent}%")
    
    # Выводим параметры искажений, которые будут применены
    # Сначала генерируем данные, чтобы знать их размер (1 КБ = 1024 байта)
    random_text = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(1024))
    original_data = random_text.encode('utf-8')
    print(f"[LOOP] Сгенерирован текст: {random_text[:20]}...")
    print(f"[LOOP] Размер данных: {len(original_data)} байт")
    
    # Выводим параметры искажений с учетом размера данных
    print_distortion_params(percent, fs, len(original_data))
    
    # 2. Передаем текст
    print(f"\n[LOOP] Передача текста...")
    # Сохраняем текст во временный файл для проверки
    temp_original = "loop_original_text.txt"
    with open(temp_original, "w", encoding='utf-8') as f:
        f.write(random_text)
    print(f"[LOOP] Оригинальные данные сохранены в {temp_original}")
    
    # Передаем текст (используем transmit_text, который сохраняет в WAV)
    transmit_text(random_text)
    print(f"[LOOP] Передача завершена. Сигнал сохранен в ofdm_acoustic_tx_with_noise.wav")
    
    # 3. Применяем искажения канала
    print(f"\n[LOOP] Применение искажений канала ({percent}%)...")
    wav_filename = "ofdm_acoustic_tx_with_noise.wav"
    
    try:
        # Загружаем WAV файл
        fs_wav, sig = wavfile.read(wav_filename)
        print(f"[LOOP] Загружен WAV файл: {wav_filename}, частота {fs_wav} Гц, {len(sig)} samples")
        
        # Проверяем, что частота дискретизации совпадает
        if fs_wav != fs:
            print(f"[LOOP-WARN] Частота дискретизации WAV ({fs_wav}) отличается от настроек ({fs})")
        
        # Создаем ОДИН экземпляр ChannelSimulator на весь сеанс,
        # чтобы АЧХ была постоянной
        sim = ChannelSimulator(fs=fs_wav)
        sim.reset_channel_state()  # Сбрасываем состояние перед новой симуляцией
        print(f"[LOOP] Создан ChannelSimulator (постоянная АЧХ на сеанс)")
        
        # Применяем искажения
        distorted_sig = apply_channel_distortions(sig, fs_wav, percent, sim=sim)
        
        # Сохраняем искаженный сигнал обратно в WAV файл
        # distorted_sig в float64 диапазоне [-1.0, 1.0], преобразуем в int16
        if distorted_sig.dtype != np.int16:
            distorted_int16 = (distorted_sig * 32767).astype(np.int16)
        else:
            distorted_int16 = distorted_sig
        
        wavfile.write(wav_filename, fs_wav, distorted_int16)
        print(f"[LOOP] Искаженный сигнал сохранен в {wav_filename}")
        
    except Exception as e:
        print(f"[LOOP-ERR] Ошибка при применении искажений: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Принимаем из искаженного WAV-файла
    print(f"\n[LOOP] Прием данных из WAV-файла...")
    received_file = "rx_text.txt"  # Теперь текст сохраняется в rx_text.txt
    # Удаляем старый файл, если есть
    if os.path.exists(received_file):
        os.remove(received_file)
    
    success = receive_from_file(wav_filename)
    if not success:
        print("[LOOP-ERR] Ошибка при приеме из файла.")
        sys.exit(1)
    
    # 5. Сравниваем оригинальные данные с принятыми
    # Принятый текст сохраняется в rx_text.txt (см. modem_rx.py)
    if os.path.exists(received_file):
        with open(received_file, "r", encoding='utf-8') as f:
            received_text = f.read()
        print(f"\n[LOOP] Сравнение данных...")
        print(f"[LOOP] Оригинальный текст: {random_text[:30]}...")
        print(f"[LOOP] Принятый текст: {received_text[:30]}...")
        
        if random_text == received_text:
            print(f"[LOOP] ✅ УСПЕХ: Текст совпадает полностью!")
        else:
            print(f"[LOOP] ❌ ОШИБКА: Текст отличается")
            # Находим первое отличие
            min_len = min(len(random_text), len(received_text))
            for i in range(min_len):
                if random_text[i] != received_text[i]:
                    print(f"[LOOP] Первое отличие в позиции {i}: ориг='{random_text[i]}', принято='{received_text[i]}'")
                    break
            if len(random_text) != len(received_text):
                print(f"[LOOP] Длина отличается: ориг={len(random_text)}, принято={len(received_text)}")
    else:
        print(f"[LOOP-ERR] Файл {received_file} не найден. Прием не удался.")
        # Попробуем найти любой файл rx_*
        rx_files = glob.glob("rx_*")
        if rx_files:
            print(f"[LOOP] Найдены файлы: {rx_files}")
    
    # Опционально: удаляем временные файлы
    cleanup = input("\nУдалить временные файлы? (y/n, по умолчанию n): ").strip().lower()
    if cleanup == 'y':
        for f in [temp_original, received_file, wav_filename]:
            if os.path.exists(f):
                os.remove(f)
                print(f"[LOOP] Удален {f}")


def main():
    """Основная функция CLI."""
    # Инициализация фаз и преамбулы
    init_phases()
    preamble_td = build_preamble()
    
    op = input("Режим работы — [T]ransmit, [R]eceive или [L]oop (по умолчанию T): ").strip().upper()
    op = "L" if op == "L" else ("R" if op == "R" else "T")
    
    if op == "T":
        run_transmit()
    elif op == "R":
        run_receive()
    else:  # Loop mode
        run_loop()


if __name__ == "__main__":
    main()
