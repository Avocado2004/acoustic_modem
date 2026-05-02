#!/usr/bin/env python3
"""
OFDM Acoustic Modem — интерфейс командной строки.
Все функции вынесены в модули:
- modem_config: конфигурация
- modem_modulation: модуляция/демодуляция
- modem_packet: работа с заголовками
- modem_tx: передача
- modem_rx: прием
- modem_cli: CLI (интерфейс командной строки)
"""

# Импортируем CLI модуль
from modem_cli import main

if __name__ == "__main__":
    main()