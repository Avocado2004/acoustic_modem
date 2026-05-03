"""
Модуль работы с пакетами OFDM Acoustic Modem.
Содержит функции для создания и разбора заголовков (Transmission/Packet), CRC32 и позиционирования.
"""

import struct
import zlib
import numpy as np
import modem_config
from modem_config import (RS_DATA_BYTES, RS_HDR_CW_BYTES, RS_HDR_PARITY, DEFAULT_PACKET_BLOCKS,
                          Nsub, fs, Nfft, SYMBOL_LEN, GAP_SAMPLES_DEFAULT)
from modem_modulation import bytes_to_bits, bits_to_bytes, build_data_td 

# -----------------------
# Header: updated format (Transmission header contains packet_blocks at bytes 50..51)
# - изменение: устанавливаем верхние 4 бита первого байта в единицы при формировании Transmission header
# - добавлено: поле CRC32 в hdr[52:56] (big-endian)
# -----------------------
def build_header(mode: bytes, data_len: int, filename_bytes: bytes = b'', packet_no: int = 0, version: int = 0, packet_blocks: int = DEFAULT_PACKET_BLOCKS, crc32: int = 0) -> bytes:
    """
    Создание 64-байтного заголовка Transmission.
    Устанавливает верхние 4 бита hdr[0] в 1 для надежной маркировки.
    CRC32 (4 байта, big-endian) сохраняется в hdr[52:56].
    
    packet_blocks - количество ЛОГИЧЕСКИХ блоков (по 96 бит каждый).
    """
    if isinstance(mode, (bytes, bytearray)) and len(mode) > 0 and mode[:1] == b'F':
        tx_type = 0b11
    elif isinstance(mode, (bytes, bytearray)) and len(mode) > 0 and mode[:1] == b'T':
        tx_type = 0b10
    else:
        tx_type = 0b00

    # Устанавливаем верхние 4 бита в 1 (0xF0), биты 3-2 для модуляции, биты 1-0 для tx_type
    mod_bits = 0b00  # default QPSK
    if modem_config.MODULATION == "BPSK":
        mod_bits = 0b01
    flags = 0xF0 | ((mod_bits & 0x03) << 2) | (tx_type & 0x03)
    hdr = bytearray(64)
    hdr[0] = flags & 0xFF
    hdr[1] = int(version) & 0xFF
    hdr[2:10] = struct.pack('>Q', int(data_len) & 0xFFFFFFFFFFFFFFFF)

    if isinstance(mode, (bytes, bytearray)) and mode[:1] == b'T':
        fname_b = b''
        name_len = 0
    else:
        fname_b = filename_bytes if filename_bytes is not None else b''
        name_len = len(fname_b)
    hdr[10:14] = struct.pack('>I', int(name_len) & 0xFFFFFFFF)
    fname_field = (fname_b[:32]).ljust(32, b'\x00')
    hdr[14:14+32] = fname_field
    hdr[46:50] = struct.pack('>I', int(packet_no) & 0xFFFFFFFF)
    # packet_blocks сохраняется в 50..51 (2 байта) - это ЛОГИЧЕСКИЕ блоки
    hdr[50:52] = struct.pack('>H', int(packet_blocks) & 0xFFFF)
    # Сохраняем CRC32 в 52..55 (big-endian). Если crc32=0, сохраняет ноль.
    hdr[52:56] = struct.pack('>I', int(crc32) & 0xFFFFFFFF)
    # Оставшиеся байты (56..63) заполняем нулями
    hdr[56:64] = b'\x00' * 8
    return bytes(hdr)


def make_packet_header_bytes(packet_no: int, tx_type: int) -> bytes:
    """Создание короткого 4-байтного заголовка пакета. Верхние 4 бита остаются нулевыми."""
    flags = (0x00) | (tx_type & 0x03)
    hdr = bytearray(4)
    hdr[0] = flags & 0xFF
    hdr[1] = (packet_no >> 16) & 0xFF
    hdr[2] = (packet_no >> 8) & 0xFF
    hdr[3] = packet_no & 0xFF
    return bytes(hdr)


def parse_header(header64: bytes):
    """
    Разбор заголовка.
    Transmission заголовок, если верхняя тетрада header64[0] != 0 (т.е. любой из верхних 4 бит установлен).
    Также извлекает CRC32 из hdr[52:56], если присутствует.
    """
    if len(header64) < 1:
        raise ValueError("Empty header data")
    b0 = header64[0]
    is_transmission = bool(b0 & 0xF0)
    # Извлечение модуляции из битов 3-2 флагового байта
    mod_bits = (b0 >> 2) & 0x03
    modulation = "BPSK" if mod_bits == 0b01 else "QPSK"
    tx_type = b0 & 0x03
    if is_transmission:
        if len(header64) < 64:
            raise ValueError("Transmission header requires 64 bytes")
        version = int(header64[1])
        data_len = struct.unpack('>Q', header64[2:10])[0]
        name_len = struct.unpack('>I', header64[10:14])[0]
        name_bytes = header64[14:46]
        packet_no = struct.unpack('>I', header64[46:50])[0]
        packet_blocks = struct.unpack('>H', header64[50:52])[0]
        crc32_val = struct.unpack('>I', header64[52:56])[0]
        if tx_type == 0b10:
            filename = ''
            name_len = 0
            packet_no = 0
        else:
            if name_len > 0:
                flen = min(name_len, 32)
                filename = name_bytes[:flen].rstrip(b'\x00').decode('utf-8', errors='ignore')
            else:
                filename = ''
        mode = b'T' if tx_type == 0b10 else (b'F' if tx_type == 0b11 else b'\x00')
        return {
            'is_transmission': True,
            'version': version,
            'tx_type': tx_type,
            'mode': mode,
            'modulation': modulation,
            'data_len': data_len,
            'name_len': name_len,
            'filename': filename,
            'packet_no': packet_no,
            'packet_blocks': packet_blocks,  # ЛОГИЧЕСКИЕ блоки
            'crc32': crc32_val
        }
    else:
        # Для обычных заголовков (4 байта) возвращаем все поля с дефолтными значениями
        if len(header64) < 4:
            raise ValueError("Packet header requires 4 bytes")
        packet_no = (header64[1] << 16) | (header64[2] << 8) | header64[3]
        mode = b'\x00'
        return {
            'is_transmission': False,
            'tx_type': tx_type,
            'mode': mode,
            'data_len': 0,
            'filename': '',
            'packet_no': packet_no,
            'packet_blocks': 0,
            'crc32': 0,
            'header_size': 4
        }


def simulate_packet_positions(total_data_len_bytes, filename_bytes, mode_is_text, packet_blocks_local,
                              preamble_len, symbol_len, gap_samples):
    """
    Симуляция позиций пакетов для передачи.
    Возвращает список смещений (в отсчетах) для каждого пакета.
    
    packet_blocks_local - количество ЛОГИЧЕСКИХ блоков в пакете.
    """
    positions = []
    
    # Переводим логические блоки в физические OFDM символы
    physical_symbols_per_packet = packet_blocks_local * modem_config.OFDM_SYMBOLS_PER_BLOCK
    
    def max_payload_for_header(header_bytes):
        step = RS_DATA_BYTES
        if total_data_len_bytes <= step:
            td, n_physical_symbols = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * total_data_len_bytes))
            # Сравниваем физические символы
            if n_physical_symbols <= physical_symbols_per_packet:
                return total_data_len_bytes
            return 0
        
        # Максимальное количество данных: packet_blocks_local логических блоков * RS_DATA_BYTES * 8 бит
        # Но нужно учесть, что физических символов может быть больше
        hi_bound = min(total_data_len_bytes, physical_symbols_per_packet * RS_DATA_BYTES)
        lo = 0
        hi = step if step < hi_bound else hi_bound
        while True:
            td, n_physical_symbols = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * hi))
            if n_physical_symbols > physical_symbols_per_packet:
                break
            if hi >= hi_bound:
                break
            hi = min(hi * 2, hi_bound)
        best = 0
        lo = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            mid -= (mid % step)
            td, n_physical_symbols = bytes_to_ofdm_blocks_bytes(header_bytes + (b'\x00' * mid))
            if n_physical_symbols <= physical_symbols_per_packet:
                best = mid
                lo = mid + step
            else:
                hi = mid - step
        return best

    tx_type_bits = 0b10 if mode_is_text else 0b11
    # Первый пакет содержит три последовательных 64-байтных заголовка Transmission
    single_hdr = build_header(b'T' if mode_is_text else b'F', total_data_len_bytes, filename_bytes=filename_bytes, packet_no=0, version=0, packet_blocks=packet_blocks_local)
    hdr_first = single_hdr + single_hdr + single_hdr
    remaining = total_data_len_bytes
    offset = 0
    pkt_no = 0

    best_first = max_payload_for_header(hdr_first)
    payload = min(remaining, best_first)
    td, n_physical_symbols = bytes_to_ofdm_blocks_bytes(hdr_first + (b'\x00' * payload))
    # Используем физические символы для расчета сэмплов
    packet_samples = preamble_len + n_physical_symbols * symbol_len + gap_samples
    positions.append(offset)
    offset += packet_samples
    remaining -= payload
    pkt_no += 1

    while remaining > 0:
        hdr = make_packet_header_bytes(pkt_no, tx_type_bits)
        best = max_payload_for_header(hdr)
        payload = min(remaining, best)
        td, n_physical_symbols = bytes_to_ofdm_blocks_bytes(hdr + (b'\x00' * payload))
        packet_samples = preamble_len + n_physical_symbols * symbol_len + gap_samples
        positions.append(offset)
        offset += packet_samples
        remaining -= payload
        pkt_no += 1
        if pkt_no > 1000000:
            break

    return positions


def bytes_to_ofdm_blocks_bytes(bstream: bytes):
    """
    Разбивка потока байт на блоки OFDM с RS кодированием.
    Возвращает (time_domain_samples, n_physical_symbols).
    
    n_physical_symbols - количество ФИЗИЧЕСКИХ OFDM символов.
    """
    if len(bstream) == 0:
        return np.array([], dtype=float), 0
    blocks_local = [bstream[i:i+RS_DATA_BYTES] for i in range(0, len(bstream), RS_DATA_BYTES)]
    if len(blocks_local[-1]) < RS_DATA_BYTES:
        blocks_local[-1] += b'\x00' * (RS_DATA_BYTES - len(blocks_local[-1]))
    # Импортируем rs локально, чтобы избежать циклических импортов
    from modem_config import rs
    encoded = [rs.encode(b) for b in blocks_local]
    bits_blocks_local = [bytes_to_bits(b) for b in encoded]

    data_bits_local = np.concatenate(bits_blocks_local) if len(bits_blocks_local) > 0 else np.array([], dtype=int)
    td_local, n_physical_symbols = build_data_td(data_bits_local) if data_bits_local.size > 0 else (np.array([], dtype=float), 0)
    return td_local, n_physical_symbols
