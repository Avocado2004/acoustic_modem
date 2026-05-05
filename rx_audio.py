"""
Модуль аудио бэкенда OFDM Acoustic Modem.
Содержит функцию получения аудио бэкенда для кроссплатформенной поддержки.
"""


def get_audio():
    """
    Возвращает аудио бэкенд для кроссплатформенной поддержки.
    Для обратной совместимости с тестами.
    """
    from audio_backend import get_audio_backend
    return get_audio_backend()
