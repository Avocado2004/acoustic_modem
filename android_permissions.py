"""
Модуль для обработки разрешений Android
"""
from jnius import autoclass, cast
from android.permissions import Permission, request_permissions, check_permission

def check_and_request_record_audio_permission():
    """
    Проверяет и запрашивает разрешение RECORD_AUDIO
    Возвращает True если разрешение предоставлено, False в противном случае
    """
    if check_permission(Permission.RECORD_AUDIO):
        return True
    
    def callback(permissions, results):
        if results[0]:  # RECORD_AUDIO
            print("Permission RECORD_AUDIO granted")
        else:
            print("Permission RECORD_AUDIO denied")
    
    request_permissions([Permission.RECORD_AUDIO], callback)
    # Примечание: request_permissions асинхронный, поэтому мы не можем вернуть результат сразу
    # В реальном приложении нужно использовать колбэк или проверять позже
    return check_permission(Permission.RECORD_AUDIO)

def check_and_request_modify_audio_settings_permission():
    """
    Проверяет и запрашивает разрешение MODIFY_AUDIO_SETTINGS
    """
    if check_permission(Permission.MODIFY_AUDIO_SETTINGS):
        return True
    
    def callback(permissions, results):
        if results[0]:  # MODIFY_AUDIO_SETTINGS
            print("Permission MODIFY_AUDIO_SETTINGS granted")
        else:
            print("Permission MODIFY_AUDIO_SETTINGS denied")
    
    request_permissions([Permission.MODIFY_AUDIO_SETTINGS], callback)
    return check_permission(Permission.MODIFY_AUDIO_SETTINGS)

def check_and_request_internet_permission():
    """
    Проверяет и запрашивает разрешение INTERNET
    """
    if check_permission(Permission.INTERNET):
        return True
    
    def callback(permissions, results):
        if results[0]:  # INTERNET
            print("Permission INTERNET granted")
        else:
            print("Permission INTERNET denied")
    
    request_permissions([Permission.INTERNET], callback)
    return check_permission(Permission.INTERNET)

def check_all_permissions():
    """
    Проверяет все необходимые разрешения
    """
    record_audio = check_and_request_record_audio_permission()
    modify_audio = check_and_request_modify_audio_settings_permission()
    internet = check_and_request_internet_permission()
    
    return record_audio and modify_audio and internet