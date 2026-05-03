"""
Тесты для проверки отсутствия искажений при приеме.
Убеждаемся, что в режимах приема из файла и с микрофона
не применяются искажения из channel_simulator.
"""

import os
import pytest
import inspect


def get_source(filepath):
    """Читает исходный код файла."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


class TestReceiveFromFileNoDistortions:
    """Тесты для проверки отсутствия искажений при приеме из файла."""
    
    def test_receive_from_file_no_channel_simulator_import(self):
        """
        Проверка, что modem_rx.py не импортирует channel_simulator
        и не вызывает apply_channel_distortions.
        """
        rx_path = os.path.join(os.path.dirname(__file__), '..', 'modem_rx.py')
        rx_path = os.path.abspath(rx_path)
        source = get_source(rx_path)
        
        # Проверяем отсутствие импорта channel_simulator
        assert 'from channel_simulator import' not in source, \
            "modem_rx.py импортирует из channel_simulator"
        assert 'import channel_simulator' not in source, \
            "modem_rx.py импортирует channel_simulator"
        
        # Проверяем отсутствие вызова apply_channel_distortions
        assert 'apply_channel_distortions' not in source, \
            "modem_rx.py содержит вызов apply_channel_distortions"
    
    def test_receive_from_file_only_normalizes_signal(self):
        """
        Проверка, что в receive_from_file сигнал только нормализуется,
        но не искажается.
        """
        rx_path = os.path.join(os.path.dirname(__file__), '..', 'modem_rx.py')
        rx_path = os.path.abspath(rx_path)
        source = get_source(rx_path)
        
        # Находим функцию receive_from_file
        start_idx = source.find('def receive_from_file(')
        assert start_idx != -1, "Функция receive_from_file не найдена"
        
        # Находим конец функции (следующая функция или конец файла)
        next_func_idx = source.find('\ndef ', start_idx + 1)
        if next_func_idx == -1:
            func_source = source[start_idx:]
        else:
            func_source = source[start_idx:next_func_idx]
        
        # Проверяем, что есть нормализация
        assert 'astype(float)' in func_source or 'astype(np.float' in func_source, \
            "В receive_from_file нет нормализации сигнала"
        
        # Проверяем отсутствие искажений
        assert 'ChannelSimulator' not in func_source, \
            "В receive_from_file есть упоминание ChannelSimulator"
        assert 'apply_' not in func_source or 'apply_channel' not in func_source, \
            "В receive_from_file есть вызовы искажений"


class TestLiveReceiveNoDistortions:
    """Тесты для проверки отсутствия искажений при живом приеме с микрофона."""
    
    def test_live_receive_no_channel_simulator_import(self):
        """
        Проверка, что в live_receive_and_process не применяются искажения.
        """
        rx_path = os.path.join(os.path.dirname(__file__), '..', 'modem_rx.py')
        rx_path = os.path.abspath(rx_path)
        source = get_source(rx_path)
        
        # Находим функцию live_receive_and_process
        start_idx = source.find('def live_receive_and_process(')
        assert start_idx != -1, "Функция live_receive_and_process не найдена"
        
        # Находим конец функции
        next_func_idx = source.find('\ndef ', start_idx + 1)
        if next_func_idx == -1:
            func_source = source[start_idx:]
        else:
            func_source = source[start_idx:next_func_idx]
        
        # Проверяем отсутствие искажений
        assert 'ChannelSimulator' not in func_source, \
            "В live_receive_and_process есть упоминание ChannelSimulator"
        assert 'apply_channel_distortions' not in func_source, \
            "В live_receive_and_process есть вызов apply_channel_distortions"
        assert 'distortion' not in func_source.lower() or \
               'audio distortion' in func_source.lower(), \
            "В live_receive_and_process могут применяться искажения"
    
    def test_audio_callback_no_distortions(self):
        """
        Проверка, что audio_callback в live_receive_and_process
        не применяет искажения к данным.
        """
        rx_path = os.path.join(os.path.dirname(__file__), '..', 'modem_rx.py')
        rx_path = os.path.abspath(rx_path)
        source = get_source(rx_path)
        
        # Находим функцию live_receive_and_process
        start_idx = source.find('def live_receive_and_process(')
        assert start_idx != -1, "Функция live_receive_and_process не найдена"
        
        # Находим конец функции
        next_func_idx = source.find('\ndef ', start_idx + 1)
        if next_func_idx == -1:
            func_source = source[start_idx:]
        else:
            func_source = source[start_idx:next_func_idx]
        
        # Находим audio_callback
        callback_start = func_source.find('def audio_callback(')
        assert callback_start != -1, "audio_callback не найден в live_receive_and_process"
        
        # Находим конец callback (следующий def или конец функции)
        callback_end = func_source.find('\n    def ', callback_start + 1)
        if callback_end == -1:
            callback_source = func_source[callback_start:]
        else:
            callback_source = func_source[callback_start:callback_end]
        
        # Проверяем, что callback только сохраняет данные, но не искажает их
        assert 'append' in callback_source, \
            "audio_callback должен сохранять данные (append)"
        assert 'ChannelSimulator' not in callback_source, \
            "audio_callback содержит упоминание ChannelSimulator"
        assert 'apply_' not in callback_source, \
            "audio_callback содержит вызовы искажений"


class TestAudioBackendsNoDistortions:
    """Тесты для проверки отсутствия искажений в аудио-бэкендах."""
    
    def test_desktop_audio_no_distortions(self):
        """Проверка, что DesktopAudio не применяет искажения."""
        try:
            from audio_desktop import DesktopAudio
        except ImportError:
            pytest.skip("sounddevice not available")
        
        source = inspect.getsource(DesktopAudio)
        
        # Проверяем отсутствие упоминаний искажений
        assert 'apply_channel_distortions' not in source, \
            "DesktopAudio содержит вызов apply_channel_distortions"
        assert 'ChannelSimulator' not in source, \
            "DesktopAudio содержит упоминание ChannelSimulator"
        
        # Проверяем, что callback вызывается без искажений
        if 'callback' in source:
            # Находим место вызова callback
            callback_lines = [line for line in source.split('\n') if 'callback' in line and '(' in line]
            for line in callback_lines:
                assert 'apply_' not in line, \
                    f"Callback может применять искажения: {line.strip()}"
    
    def test_android_audio_no_distortions(self):
        """Проверка, что AndroidAudio не применяет искажения."""
        try:
            from audio_android import AndroidAudio
        except ImportError:
            pytest.skip("pyjnius not available")
        
        source = inspect.getsource(AndroidAudio)
        
        assert 'apply_channel_distortions' not in source, \
            "AndroidAudio содержит вызов apply_channel_distortions"
        assert 'ChannelSimulator' not in source, \
            "AndroidAudio содержит упоминание ChannelSimulator"
    
    def test_ios_audio_no_distortions(self):
        """Проверка, что IOSAudio не применяет искажения."""
        try:
            from audio_ios import IOSAudio
        except ImportError:
            pytest.skip("pyobjc not available")
        
        source = inspect.getsource(IOSAudio)
        
        assert 'apply_channel_distortions' not in source, \
            "IOSAudio содержит вызов apply_channel_distortions"
        assert 'ChannelSimulator' not in source, \
            "IOSAudio содержит упоминание ChannelSimulator"


class TestAudioBackendSelectorNoDistortions:
    """Тесты для проверки отсутствия искажений в audio_backend.py."""
    
    def test_audio_backend_no_channel_simulator_import(self):
        """Проверка, что audio_backend.py не импортирует channel_simulator."""
        backend_path = os.path.join(os.path.dirname(__file__), '..', 'audio_backend.py')
        backend_path = os.path.abspath(backend_path)
        source = get_source(backend_path)
        
        assert 'from channel_simulator import' not in source, \
            "audio_backend.py импортирует из channel_simulator"
        assert 'import channel_simulator' not in source, \
            "audio_backend.py импортирует channel_simulator"
        assert 'apply_channel_distortions' not in source, \
            "audio_backend.py содержит вызов apply_channel_distortions"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
