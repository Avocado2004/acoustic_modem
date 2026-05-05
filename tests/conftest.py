"""
Фикстуры для тестов.
Обеспечивает изоляцию тестов, восстанавливая глобальное состояние.
"""

import pytest
import modem_config


@pytest.fixture(autouse=True)
def restore_modulation():
    """Фикстура для автоматического восстановления модуляции после каждого теста."""
    # Сохраняем текущую модуляцию
    original_modulation = modem_config.MODULATION
    yield
    # Восстанавливаем модуляцию
    if modem_config.MODULATION != original_modulation:
        modem_config.set_modulation(original_modulation)
