"""
Модуль живого приёма OFDM Acoustic Modem.
Содержит функцию живого приёма с микрофона.

Использует signal_processor.process_signal_stream() для обработки сигнала,
что устраняет дублирование логики корреляции, синхронизации и декодирования.
"""

import modem_config
from modem_config import fs

from modem_modulation import build_preamble
from signal_processor import process_signal_stream

# Импортируем модуль состояния
import rx_state as _rx_st


def live_receive_and_process():
    """
    Живой приём с микрофона.

    Создаёт MicrophoneDataSource (legacy-интерфейс из data_source.py),
    запускает захват аудио и передаёт управление в signal_processor.process_signal_stream().

    Сохраняет ту же сигнатуру и поведение, что и оригинальная реализация,
    но без дублирования логики корреляции/синхронизации/декодирования.

    Returns
    -------
    bool
        True если приём успешен, False в случае ошибки.
    """
    from data_source import MicrophoneDataSource

    print("[LIVE] Initializing live receive...")

    # Инициализируем фазы и строим преамбулу
    from modem_config import init_phases
    init_phases()

    preamble_td_local = build_preamble()
    if preamble_td_local is None:
        print("[LIVE-ERR] Failed to build preamble")
        return False

    pre_len = len(preamble_td_local)
    print("[LIVE] Preamble built, length: {}".format(pre_len))

    # Обновляем глобальную переменную состояния
    _rx_st.preamble_td = preamble_td_local

    # Создаём источник данных с микрофона (legacy-интерфейс)
    CHUNK = 2048
    data_source = MicrophoneDataSource(fs, channels=1, chunk=CHUNK)

    print("[LIVE] Starting microphone stream...")
    try:
        data_source.start()
    except Exception as e:
        print("[LIVE-ERR] Failed to start microphone stream: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

    print("[LIVE] Microphone stream started, processing...")

    try:
        # Передаём управление универсальному процессору сигнала
        # process_signal_stream() поддерживает legacy-интерфейс (read_snapshot/wait_for_samples)
        result = process_signal_stream(data_source, preamble_td_local, config=modem_config)

        if result:
            print("[LIVE] Reception completed successfully")
        else:
            print("[LIVE-ERR] Reception failed")

        return result

    except KeyboardInterrupt:
        print("\n[LIVE] Stopped by user")
        return False
    except Exception as e:
        print("[LIVE-ERR] Exception: {}".format(e))
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Останавливаем источник данных
        try:
            data_source.stop()
            print("[LIVE] Audio stream stopped")
        except Exception:
            pass
