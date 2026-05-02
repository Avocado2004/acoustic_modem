#!/usr/bin/env python3
"""
Пример использования Mem0 для сохранения контекста проекта Acoustic_Modem.

Этот модуль демонстрирует:
1. Сохранение результатов тестирования в Mem0
2. Поиск сохраненной информации
3. Обновление контекста проекта

Требования:
- pip install mem0ai
- Установленная переменная окружения MEM0_API_KEY
"""

import os
import json
from datetime import datetime


def setup_mem0_client():
    """
    Настраивает клиент Mem0 с использованием API ключа из переменных окружения.
    
    Returns:
        Инициализированный клиент Mem0 или None в случае ошибки.
    """
    try:
        from mem0 import MemoryClient
        
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            print("[DEBUG] Переменная окружения MEM0_API_KEY не установлена")
            print("[DEBUG] Используем ключ по умолчанию для примера")
            api_key = "m0-w3qGCyAE9KquCWo9Ob3VyFQv2s1LXZ4fYzrsiLFO"
        
        print(f"[DEBUG] Инициализация Mem0 клиента с API ключом: {api_key[:10]}...")
        client = MemoryClient(api_key=api_key)
        print("[DEBUG] Mem0 клиент успешно инициализирован")
        return client
    except ImportError:
        print("[DEBUG] Ошибка: библиотека mem0ai не установлена")
        print("[DEBUG] Установите её командой: pip install mem0ai")
        return None
    except Exception as e:
        print(f"[DEBUG] Ошибка при инициализации Mem0 клиента: {e}")
        return None


def save_test_results(client, test_name, results):
    """
    Сохраняет результаты тестирования в Mem0.
    
    Args:
        client: Клиент Mem0
        test_name: Название теста
        results: Словарь с результатами теста
    
    Returns:
        ID сохраненной памяти или None в случае ошибки.
    """
    if client is None:
        print("[DEBUG] Клиент Mem0 не инициализирован")
        return None
    
    try:
        # Формируем сообщение для сохранения
        message = f"Результаты теста '{test_name}':\n"
        message += f"Время: {datetime.now().isoformat()}\n"
        message += f"Результаты: {json.dumps(results, indent=2, ensure_ascii=False)}\n"
        
        print(f"[DEBUG] Сохранение результатов теста '{test_name}' в Mem0...")
        print(f"[DEBUG] Сообщение для сохранения: {message[:200]}...")
        
        # Сохраняем в Mem0
        # Используем user_id для группировки памяти по проекту
        response = client.add(
            message,
            user_id="acoustic_modem_project",
            metadata={
                "project": "Acoustic_Modem",
                "test_name": test_name,
                "timestamp": datetime.now().isoformat(),
                "type": "test_results"
            }
        )
        
        print(f"[DEBUG] Результат сохранения: {response}")
        memory_id = response.get("id") if isinstance(response, dict) else None
        print(f"[DEBUG] ID сохраненной памяти: {memory_id}")
        return memory_id
    except Exception as e:
        print(f"[DEBUG] Ошибка при сохранении результатов теста: {e}")
        return None


def search_memories(client, query):
    """
    Выполняет поиск по сохраненным воспоминаниям.
    
    Args:
        client: Клиент Mem0
        query: Поисковый запрос
    
    Returns:
        Список найденных воспоминаний.
    """
    if client is None:
        print("[DEBUG] Клиент Mem0 не инициализирован")
        return []
    
    try:
        print(f"[DEBUG] Поиск в Mem0 по запросу: '{query}'...")
        
        results = client.search(
            query,
            user_id="acoustic_modem_project"
        )
        
        print(f"[DEBUG] Найдено результатов: {len(results)}")
        for i, result in enumerate(results[:5]):  # Показываем первые 5 результатов
            print(f"[DEBUG] Результат {i+1}: {str(result)[:200]}...")
        
        return results
    except Exception as e:
        print(f"[DEBUG] Ошибка при поиске в Mem0: {e}")
        return []


def get_all_memories(client):
    """
    Получает все сохраненные воспоминания для проекта.
    
    Args:
        client: Клиент Mem0
    
    Returns:
        Список всех воспоминаний.
    """
    if client is None:
        print("[DEBUG] Клиент Mem0 не инициализирован")
        return []
    
    try:
        print("[DEBUG] Получение всех воспоминаний для проекта acoustic_modem_project...")
        
        memories = client.get_all(user_id="acoustic_modem_project")
        
        print(f"[DEBUG] Всего воспоминаний: {len(memories)}")
        for i, memory in enumerate(memories[:10]):  # Показываем первые 10
            memory_text = memory.get("memory", "") if isinstance(memory, dict) else str(memory)
            print(f"[DEBUG] Воспоминание {i+1}: {memory_text[:150]}...")
        
        return memories
    except Exception as e:
        print(f"[DEBUG] Ошибка при получении воспоминаний: {e}")
        return []


def save_project_context(client):
    """
    Сохраняет общий контекст проекта Acoustic_Modem.
    
    Args:
        client: Клиент Mem0
    """
    if client is None:
        print("[DEBUG] Клиент Mem0 не инициализирован")
        return
    
    try:
        context_message = """
        Проект Acoustic_Modem - кроссплатформенное приложение для передачи данных через звук.
        
        Основные компоненты:
        - modem_tx.py: передатчик модема
        - modem_rx.py: приемник модема
        - modem_modulation.py: модуляция (BPSK, QPSK)
        - modem_packet.py: работа с пакетами
        - audio_backend.py: абстракция аудио бэкенда
        - platform_specific/: платформозависимый код
        
        Поддерживаемые платформы: macOS, Android, Windows, iOS.
        
        Текущий статус: настройка MCP серверов для улучшения разработки.
        """
        
        print("[DEBUG] Сохранение контекста проекта в Mem0...")
        response = client.add(
            context_message,
            user_id="acoustic_modem_project",
            metadata={
                "project": "Acoustic_Modem",
                "type": "project_context",
                "timestamp": datetime.now().isoformat()
            }
        )
        print(f"[DEBUG] Контекст проекта сохранен: {response}")
    except Exception as e:
        print(f"[DEBUG] Ошибка при сохранении контекста проекта: {e}")


def main():
    """
    Основная функция для демонстрации работы с Mem0.
    """
    print("=" * 60)
    print("Пример использования Mem0 для проекта Acoustic_Modem")
    print("=" * 60)
    
    # Инициализация клиента
    client = setup_mem0_client()
    
    if client is None:
        print("\n[DEBUG] Не удалось инициализировать клиент Mem0")
        print("[DEBUG] Убедитесь, что библиотека mem0ai установлена и API ключ корректен")
        return
    
    # Сохранение контекста проекта
    print("\n--- Сохранение контекста проекта ---")
    save_project_context(client)
    
    # Пример сохранения результатов теста
    print("\n--- Сохранение результатов теста ---")
    test_results = {
        "test_name": "test_modem_unit.py",
        "status": "passed",
        "tests_run": 15,
        "tests_passed": 15,
        "tests_failed": 0,
        "execution_time": "2.5s"
    }
    save_test_results(client, "unit_tests", test_results)
    
    # Еще один пример
    test_results_2 = {
        "test_name": "test_bpsk_qpsk_verification.py",
        "status": "passed",
        "modulation_types": ["BPSK", "QPSK"],
        "snr_range": [-10, 20],
        "ber_results": {"BPSK": 0.001, "QPSK": 0.002}
    }
    save_test_results(client, "modulation_tests", test_results_2)
    
    # Поиск по сохраненным данным
    print("\n--- Поиск по сохраненным данным ---")
    search_memories(client, "результаты теста")
    
    # Получение всех воспоминаний
    print("\n--- Все сохраненные воспоминания ---")
    get_all_memories(client)
    
    print("\n" + "=" * 60)
    print("Пример завершен")
    print("=" * 60)


if __name__ == "__main__":
    main()
