# ofdm_gui_flet.py - Acoustic Modem GUI on Flet
import sys
import os
import threading
import importlib
import io
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import flet as ft

from audio_backend import play_audio, stop_audio, is_playing
from wav_utils import read as wav_read, write as wav_write

_modem = None
_import_err = None
try:
    _modem = importlib.import_module("test_modem_simple")
except Exception as e:
    import traceback
    _import_err = traceback.format_exc()
    _modem = None

def gattr(obj, name, default=None):
    return getattr(obj, name) if obj is not None and hasattr(obj, name) else default


class GuiLogger:
    def __init__(self, page, diag_field, recv_field):
        self.page = page
        self.diag_field = diag_field
        self.recv_field = recv_field
        self._buf = ''
        self.last_rx_saved_path = None

    def write(self, txt):
        if not txt:
            return
        self._buf += txt
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._append_line(line + "\n")

    def flush(self):
        if self._buf:
            self._append_line(self._buf)
            self._buf = ''
    
    def _append_line(self, line):
        self.diag_field.value = (self.diag_field.value or '') + line
        self.page.update()
        if '[RX TEXT]' in line:
            txt = line.split('[RX TEXT]')[1].strip()
            self._append_received(txt)
        if '[RX FILE]' in line and 'Saved' in line:
            try:
                after = line.split('Saved:', 1)[1].strip()
                path = after.split(' ')[0].strip()
                if os.path.exists(path):
                    self.last_rx_saved_path = path
            except Exception:
                pass
        if 'Saved:' in line and 'rx_' in line:
            try:
                after = line.split('Saved:', 1)[1].strip()
                path = after.split(' ')[0].strip()
                if os.path.exists(path):
                    self.last_rx_saved_path = path
            except Exception:
                pass

    def _append_received(self, txt):
        cur = self.recv_field.value or ''
        self.recv_field.value = (cur + "\n" + txt) if cur else txt
        self.page.update()


def fmt_log_line(msg):
    t = datetime.now().strftime('%H:%M:%S')
    return '[' + t + '] ' + msg + '\n'


def fallback_build_tx_audio_from_bytes(payload_bytes, mode_flag=b'F', filename_bytes=b'', settings=None):
    if _modem is None:
        raise RuntimeError('modem module required')
    fs = gattr(_modem, 'fs', 48000)
    SYMBOL_LEN = gattr(_modem, 'SYMBOL_LEN', None)
    if SYMBOL_LEN is None:
        Nfft = gattr(_modem, 'Nfft', None)
        Ncp = gattr(_modem, 'Ncp', None)
        if Nfft is None or Ncp is None:
            raise RuntimeError('modem lacks Nfft/Ncp')
        SYMBOL_LEN = Nfft + Ncp
    packet_blocks = int(gattr(_modem, 'DEFAULT_PACKET_BLOCKS', 75))
    version = 0
    tx_type_bits = 0b11 if mode_flag == b'F' else (0b10 if mode_flag == b'T' else 0b00)
    import zlib
    crc_val = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    build_header = gattr(_modem, 'build_header', None)
    if build_header is None:
        raise RuntimeError('modem does not expose build_header')
    transmission_header = build_header(
        b'F' if mode_flag == b'F' else b'T',
        len(payload_bytes),
        filename_bytes=filename_bytes,
        packet_no=0,
        version=version,
        packet_blocks=packet_blocks,
        crc32=crc_val
    )
    bytes_left = payload_bytes
    packet_no = 0
    samples_list = []
    build_preamble = gattr(_modem, 'build_preamble', None)
    if build_preamble is None:
        raise RuntimeError('modem does not expose build_preamble')
    preamble_td = build_preamble()
    bytes_to_blocks = gattr(_modem, 'bytes_to_ofdm_blocks_bytes', None)
    if bytes_to_blocks is None:
        raise RuntimeError('modem must expose bytes_to_ofdm_blocks_bytes')
    while True:
        is_first = (packet_no == 0)
        if is_first:
            header_bytes = transmission_header + transmission_header + transmission_header
        else:
            make_packet_header_bytes = gattr(_modem, 'make_packet_header_bytes', None)
            if make_packet_header_bytes is None:
                header_bytes = bytes([
                    0x00 | (tx_type_bits & 0x03),
                    (packet_no >> 16) & 0xFF,
                    (packet_no >> 8) & 0xFF,
                    packet_no & 0xFF
                ])
            else:
                header_bytes = make_packet_header_bytes(packet_no, tx_type_bits)
        allowed_blocks = packet_blocks
        lo, hi = 0, len(bytes_left)
        best_mid = 0
        best_td = None
        while lo <= hi:
            mid = (lo + hi) // 2
            test_bytes = header_bytes + bytes_left[:mid]
            try:
                td_local, nblocks_local = bytes_to_blocks(test_bytes)
            except Exception:
                nblocks_local = allowed_blocks + 1
            if nblocks_local <= allowed_blocks:
                best_mid = mid
                best_td = td_local
                lo = mid + 1
            else:
                hi = mid - 1
        if best_td is None:
            td_local, nblocks_local = bytes_to_blocks(header_bytes)
            best_td = td_local
        packet_samples = np.concatenate((preamble_td, best_td)) if best_td.size else preamble_td.copy()
        samples_list.append(packet_samples)
        bytes_left = bytes_left[best_mid:]
        packet_no += 1
        if len(bytes_left) == 0 or packet_no > 10000:
            break
    tx_concat = np.concatenate(samples_list) if samples_list else np.array([], dtype=float)
    preroll_len = int(0.1 * fs)
    postroll_len = int(0.1 * fs)
    tx_out = np.concatenate((
        np.zeros(preroll_len, dtype=tx_concat.dtype),
        tx_concat,
        np.zeros(postroll_len, dtype=tx_concat.dtype)
    ))
    info = {'samples': len(tx_out), 'fs': fs, 'packets': packet_no}
    return tx_out.astype(np.float64), info


def get_builder():
    if _modem is not None:
        b = gattr(_modem, 'build_tx_audio_from_bytes', None)
        if b is not None:
            return b
    return fallback_build_tx_audio_from_bytes


class OfdmApp:
    def __init__(self, page):
        self.page = page
        self.page.title = 'Acoustic Modem GUI (Flet)'
        self.page.window_width = 900
        self.page.window_height = 720
        self.page.padding = 10
        self.page.spacing = 6

        self.tx_audio = None
        self.tx_info = None
        self.selected_file = None
        self.loaded_wav_path = None
        self.last_rx_saved_path = None
        self.mode = 'Text'
        self.modulation = gattr(_modem, 'MODULATION', 'QPSK')
        self._receiving = False

        self.settings: Dict[str, Any] = {
            'Nfft': gattr(_modem, 'Nfft', 512),
            'Ncp': gattr(_modem, 'Ncp', 128),
            'REQUIRED_NSUB': gattr(_modem, 'REQUIRED_NSUB', 48),
            'RS_DATA_BYTES': gattr(_modem, 'RS_DATA_BYTES', 8),
            'RS_PARITY_BYTES': gattr(_modem, 'RS_PARITY_BYTES', 4),
            'PHASE_METHOD': gattr(_modem, 'PHASE_METHOD', 'schroeder'),
            'DEFAULT_PACKET_BLOCKS': gattr(_modem, 'DEFAULT_PACKET_BLOCKS', 75),
            'GAP_OFDM_SYMBOLS': gattr(_modem, 'GAP_OFDM_SYMBOLS', 0),
        }

        self._save_type = None
        self._save_content = None
        self._save_source_path = None
        self.selected_file_bytes = None
        self.selected_filename = None

        # Initialize FilePickers for different actions
        self.file_picker_pick = ft.FilePicker()
        self.file_picker_pick.on_result = self._on_file_picked
        self.page.overlay.append(self.file_picker_pick)

        self.file_picker_save = ft.FilePicker()
        self.file_picker_save.on_result = self._on_file_picked
        self.page.overlay.append(self.file_picker_save)
        
        self._file_action = None  # 'pick', 'save_wav', 'save_recv', 'load_wav'

        self._build_widgets()
        self._build_layout()
        self._apply_mode()

        self.logger = GuiLogger(self.page, self.diag_field, self.recv_field)
        sys.stdout = self.logger
        sys.stderr = self.logger

        if _import_err:
            self._append_diag(fmt_log_line('[IMPORT-ERR] ' + _import_err.splitlines()[0]))

    def _build_widgets(self):
        # Mode toggle buttons
        self.btn_mode_text = ft.Button(
            'Text',
            on_click=self._on_mode_text,
            bgcolor=ft.Colors.BLUE_200,
            color=ft.Colors.BLACK,
        )
        self.btn_mode_file = ft.Button(
            'File',
            on_click=self._on_mode_file,
            bgcolor=ft.Colors.GREY_300,
            color=ft.Colors.BLACK,
        )
        # Modulation toggle buttons
        self.btn_mod_bpsk = ft.Button(
            'BPSK',
            on_click=self._on_mod_bpsk,
            bgcolor=ft.Colors.GREY_300,
            color=ft.Colors.BLACK,
        )
        self.btn_mod_qpsk = ft.Button(
            'QPSK',
            on_click=self._on_mod_qpsk,
            bgcolor=ft.Colors.BLUE_200,
            color=ft.Colors.BLACK,
        )
        # Settings button
        self.btn_settings = ft.Button(
            'Settings',
            on_click=self._on_open_settings,
            bgcolor=ft.Colors.GREY_300,
            color=ft.Colors.BLACK,
        )
        # Select File button
        self.btn_select_file = ft.Button(
            'Select File',
            on_click=self._on_select_file,
            disabled=True,
            bgcolor=ft.Colors.GREY_300,
            color=ft.Colors.BLACK,
        )
        # Input field
        self.input_field = ft.TextField(
            label='Input',
            multiline=True,
            min_lines=3,
            max_lines=5,
            hint_text='Enter text here (or select a file in File Mode)',
            expand=True,
        )
        # Control buttons
        self.btn_generate = ft.Button(
            'Generate',
            on_click=self._on_generate,
            bgcolor=ft.Colors.GREEN_200,
            color=ft.Colors.BLACK,
        )
        self.btn_save_wav = ft.Button(
            'Save WAV',
            on_click=self._on_save_wav,
            bgcolor=ft.Colors.AMBER_200,
            color=ft.Colors.BLACK,
        )
        self.btn_load_wav = ft.Button(
            'Load WAV',
            on_click=self._on_load_wav,
            bgcolor=ft.Colors.AMBER_200,
            color=ft.Colors.BLACK,
        )
        self.btn_play_stop = ft.Button(
            'Play/Stop',
            on_click=self._on_play_stop,
            bgcolor=ft.Colors.LIGHT_BLUE_200,
            color=ft.Colors.BLACK,
        )
        # Receive button
        self.btn_receive = ft.Button(
            'Receive',
            on_click=self._on_receive,
            bgcolor=ft.Colors.ORANGE_200,
            color=ft.Colors.BLACK,
        )
        # Received field
        self.recv_field = ft.TextField(
            label='Received',
            multiline=True,
            min_lines=3,
            max_lines=5,
            read_only=False,
            expand=True,
        )
        # Save received / Copy buttons
        self.btn_save_recv = ft.Button(
            'Save received file',
            on_click=self._on_save_received,
            bgcolor=ft.Colors.GREY_300,
            color=ft.Colors.BLACK,
        )
        self.btn_copy_recv = ft.Button(
            'Copy',
            on_click=self._on_copy_received,
            bgcolor=ft.Colors.GREY_300,
            color=ft.Colors.BLACK,
        )
        # Diagnostics field
        self.diag_field = ft.TextField(
            label='Diagnostics',
            multiline=True,
            min_lines=4,
            max_lines=6,
            read_only=True,
            expand=True,
        )

    def _build_layout(self):
        # Top row: Mode toggle, Modulation toggle, Settings, Select File
        top_row = ft.Row(
            controls=[
                ft.Text('Mode:', size=14, weight=ft.FontWeight.BOLD),
                self.btn_mode_text,
                self.btn_mode_file,
                ft.VerticalDivider(width=1),
                ft.Text('Modulation:', size=14, weight=ft.FontWeight.BOLD),
                self.btn_mod_bpsk,
                self.btn_mod_qpsk,
                ft.VerticalDivider(width=1),
                self.btn_settings,
                self.btn_select_file,
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=8,
            wrap=True,
        )
        # Input area
        input_area = ft.Column(controls=[self.input_field], spacing=4, expand=True)
        # Control buttons row
        ctrl_row = ft.Row(
            controls=[self.btn_generate, self.btn_save_wav, self.btn_load_wav, self.btn_play_stop],
            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
            spacing=10,
            wrap=True,
        )
        # Diagnostics area
        diag_area = ft.Column(controls=[self.diag_field], spacing=4, expand=True)
        # Received area + buttons
        recv_area = ft.Column(
            controls=[
                self.recv_field,
                ft.Row(
                    controls=[self.btn_receive, self.btn_save_recv, self.btn_copy_recv],
                    alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                    spacing=10,
                    wrap=True,
                ),
            ],
            spacing=4,
            expand=True,
        )
        # Main layout
        main_col = ft.Column(
            controls=[
                top_row,
                ft.Divider(height=2, color=ft.Colors.GREY_400),
                ft.Text('Input Text / File:', size=14, weight=ft.FontWeight.BOLD),
                input_area,
                ft.Divider(height=2, color=ft.Colors.GREY_400),
                ctrl_row,
                ft.Divider(height=2, color=ft.Colors.GREY_400),
                ft.Text('Diagnostics:', size=14, weight=ft.FontWeight.BOLD),
                diag_area,
                ft.Divider(height=2, color=ft.Colors.GREY_400),
                ft.Text('Received:', size=14, weight=ft.FontWeight.BOLD),
                recv_area,
            ],
            spacing=6,
            expand=True,
            scroll=ft.ScrollMode.HIDDEN,
        )
        self.page.add(main_col)
        self.page.update()

    # Mode handlers
    def _on_mode_text(self, e):
        self.mode = 'Text'
        self._update_mode_buttons()
        self._apply_mode()

    def _on_mode_file(self, e):
        self.mode = 'File'
        self._update_mode_buttons()
        self._apply_mode()
        self._on_select_file(e)

    def _update_mode_buttons(self):
        if self.mode == 'Text':
            self.btn_mode_text.bgcolor = ft.Colors.BLUE_200
            self.btn_mode_file.bgcolor = ft.Colors.GREY_300
        else:
            self.btn_mode_text.bgcolor = ft.Colors.GREY_300
            self.btn_mode_file.bgcolor = ft.Colors.BLUE_200
        self.page.update()

    def _apply_mode(self):
        if self.mode == 'Text':
            self.input_field.read_only = False
            self.input_field.hint_text = 'Enter text here (or select a file in File Mode)'
            self.btn_select_file.disabled = True
        else:
            self.input_field.read_only = True
            self.input_field.hint_text = 'Selected file will appear here'
            self.btn_select_file.disabled = False
            if self.selected_file:
                self.input_field.value = self.selected_file
            else:
                self.input_field.value = '[File] No file selected'
        self._append_diag(fmt_log_line('Switched to ' + self.mode + ' mode'))
        self._append_diag(fmt_log_line('Select File button disabled=' + str(self.btn_select_file.disabled)))
        self.page.update()

    # Modulation handlers
    def _on_mod_bpsk(self, e):
        self.modulation = 'BPSK'
        self._update_mod_buttons()
        self._apply_modulation()

    def _on_mod_qpsk(self, e):
        self.modulation = 'QPSK'
        self._update_mod_buttons()
        self._apply_modulation()

    def _update_mod_buttons(self):
        if self.modulation == 'BPSK':
            self.btn_mod_bpsk.bgcolor = ft.Colors.BLUE_200
            self.btn_mod_qpsk.bgcolor = ft.Colors.GREY_300
        else:
            self.btn_mod_bpsk.bgcolor = ft.Colors.GREY_300
            self.btn_mod_qpsk.bgcolor = ft.Colors.BLUE_200
        self.page.update()

    def _apply_modulation(self):
        if _modem is not None:
            try:
                _modem.MODULATION = self.modulation
                if self.modulation == 'BPSK':
                    _modem.BITS_PER_SYMBOL = 1
                else:
                    _modem.BITS_PER_SYMBOL = 2
                _modem.BITS_PER_OFDM_SYMBOL = _modem.Nsub * _modem.BITS_PER_SYMBOL
                self._append_diag(fmt_log_line('Modulation set to ' + self.modulation))
            except Exception as ex:
                self._append_diag(fmt_log_line('Failed to set modulation: ' + str(ex)))
        else:
            self._append_diag(fmt_log_line('Modulation set to ' + self.modulation + ' (modem not loaded)'))

    def _on_file_picked(self, e):
        action = self._file_action
        self._file_action = None
        
        if action == 'pick' or action == 'load_wav':
            if not e.files:
                self._append_diag(fmt_log_line('File selection cancelled'))
                return
            file = e.files[0]
            if action == 'pick':
                self.selected_file_bytes = None
                self.selected_filename = getattr(file, 'name', None) or os.path.basename(file.path or '')
                self.selected_file = file.path
                if self.selected_file:
                    self.input_field.value = self.selected_file
                elif getattr(file, 'bytes', None) is not None:
                    self.selected_file_bytes = file.bytes
                    self.input_field.value = self.selected_filename or '[file selected]'
                else:
                    self.input_field.value = '[file selected]'
                self._append_diag(fmt_log_line('Selected file: ' + str(self.selected_filename or self.selected_file)))
                self.page.update()
            else:  # load_wav
                try:
                    # Сначала пробуем прочитать из bytes, если они есть (независимо от платформы)
                    if hasattr(file, 'bytes') and file.bytes is not None:
                        fs, data = wav_read(io.BytesIO(file.bytes))
                        self.loaded_wav_path = None
                    elif file.path:
                        fs, data = wav_read(file.path)
                        self.loaded_wav_path = file.path
                    else:
                        raise Exception("No file data available")
                    if data.dtype != np.float64:
                        if data.dtype == np.int16:
                            data = data.astype(np.float64) / 32767.0
                        elif data.dtype == np.int32:
                            data = data.astype(np.float64) / 2147483647.0
                        else:
                            data = data.astype(np.float64)
                    if _modem is not None:
                        _modem.rx = data
                        _modem.rx_fs = fs
                        self._append_diag(fmt_log_line('WAV loaded: ' + str(len(data)) + ' samples at ' + str(fs) + ' Hz'))
                    else:
                        self._append_diag(fmt_log_line('WAV loaded but modem not available'))
                except Exception as ex:
                    self._append_diag(fmt_log_line('Failed to load WAV: ' + str(ex)))
                self.page.update()
            
        elif action == 'save_wav':
            if not e.path:
                self._append_diag(fmt_log_line('Save cancelled'))
                return
            try:
                if self.tx_info:
                    fs = int(self.tx_info.get('fs', gattr(_modem, 'fs', 48000)))
                else:
                    fs = int(gattr(_modem, 'fs', 48000))
                max_abs = np.max(np.abs(self.tx_audio)) if self.tx_audio.size else 0.0
                if max_abs == 0:
                    data_int16 = (self.tx_audio * 0).astype(np.int16)
                else:
                    data_int16 = (self.tx_audio / max_abs * np.iinfo(np.int16).max).astype(np.int16)
                wav_write(e.path, fs, data_int16)
                self._append_diag(fmt_log_line('Saved WAV to ' + e.path))
            except Exception as ex:
                self._append_diag(fmt_log_line('Save WAV failed: ' + str(ex)))
                
        elif action == 'save_recv':
            if not e.path:
                self._append_diag(fmt_log_line('Save cancelled'))
                return
            mode = self.mode
            if mode == 'Text':
                content = self.recv_field.value
                try:
                    with open(e.path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self._append_diag(fmt_log_line('Saved received text to ' + e.path))
                except Exception as ex:
                    self._append_diag(fmt_log_line('Save failed: ' + str(ex)))
            else:
                if self.logger.last_rx_saved_path and os.path.exists(self.logger.last_rx_saved_path):
                    try:
                        shutil.copyfile(self.logger.last_rx_saved_path, e.path)
                        self._append_diag(fmt_log_line('Copied received file to ' + e.path))
                    except Exception as ex:
                        self._append_diag(fmt_log_line('Save failed: ' + str(ex)))
                else:
                    content = self.recv_field.value
                    try:
                        with open(e.path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        self._append_diag(fmt_log_line('Saved received text to ' + e.path))
                    except Exception as ex:
                        self._append_diag(fmt_log_line('Save failed: ' + str(ex)))

    def _on_select_file(self, e):
        self._append_diag(fmt_log_line('Opening file picker...'))
        self._file_action = 'pick'
        self.file_picker_pick.pick_files(
            'Select file',
            file_type=ft.FilePickerFileType.ANY,
            allow_multiple=False,
        )

    def _on_load_wav(self, e):
        self._append_diag(fmt_log_line('Opening WAV file picker...'))
        self._file_action = 'load_wav'
        self.file_picker_pick.pick_files(
            'Load WAV file with signal',
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=['wav'],
            allow_multiple=False,
            with_data=True,
        )

    def _on_save_wav(self, e):
        if self.tx_audio is None:
            self._append_diag(fmt_log_line('Save WAV: no audio (generate first)'))
            return
        self._file_action = 'save_wav'
        self.file_picker_save.save_file(
            'Save WAV as',
            file_name='ofdm_tx.wav',
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=['wav'],
        )

    def _on_save_received(self, e):
        self._file_action = 'save_recv'
        mode = self.mode
        if mode == 'Text':
            self.file_picker_save.save_file(
                'Save received text as',
                file_name='received.txt',
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=['txt'],
            )
        else:
            if self.logger.last_rx_saved_path and os.path.exists(self.logger.last_rx_saved_path):
                self.file_picker_save.save_file(
                    'Save received file as',
                    file_name=os.path.basename(self.logger.last_rx_saved_path),
                    file_type=ft.FilePickerFileType.ANY,
                )
            else:
                self.file_picker_save.save_file(
                    'Save received as',
                    file_name='received.txt',
                    file_type=ft.FilePickerFileType.CUSTOM,
                    allowed_extensions=['txt'],
                )

    def _on_generate(self, e):
        mode = self.mode
        try:
            if mode == 'Text':
                text = self.input_field.value
                if not text:
                    self._append_diag(fmt_log_line('Generate: Enter text first'))
                    return
                payload = text.encode('utf-8')
                filename_bytes = b''
            else:
                if not self.selected_file and self.selected_file_bytes is None:
                    self._append_diag(fmt_log_line('Generate: Select a file first'))
                    return
                if self.selected_file:
                    with open(self.selected_file, 'rb') as f:
                        payload = f.read()
                    filename = os.path.basename(self.selected_file)
                else:
                    payload = self.selected_file_bytes
                    filename = self.selected_filename or 'selected_file'
                filename_bytes = filename.encode('utf-8')
        except Exception as ex:
            self._append_diag(fmt_log_line('Generate error: ' + str(ex)))
            return

        def _build():
            try:
                builder = get_builder()
                tx, info = builder(
                    payload,
                    mode_flag=(b'T' if mode == 'Text' else b'F'),
                    filename_bytes=filename_bytes,
                    settings=self.settings,
                )
                self.tx_audio = tx
                self.tx_info = info
                self._append_diag(fmt_log_line(
                    'Generated audio: samples=' + str(info.get('samples', len(tx))) +
                    ' fs=' + str(info.get('fs', 'n/a')) +
                    ' packets=' + str(info.get('packets', 'n/a'))
                ))
            except Exception as ex:
                self._append_diag(fmt_log_line('Audio generation failed: ' + str(ex)))
        threading.Thread(target=_build, daemon=True).start()

    # Play/Stop
    def _on_play_stop(self, e):
        if self.tx_audio is None:
            self._append_diag(fmt_log_line('Play: no audio generated yet, generating now...'))
            self._on_generate(e)
            return
        if is_playing():
            stop_audio()
            self._append_diag(fmt_log_line('Playback stopped'))
        else:
            if self.tx_info:
                fs = int(self.tx_info.get('fs', gattr(_modem, 'fs', 48000)))
            else:
                fs = int(gattr(_modem, 'fs', 48000))
            self._append_diag(fmt_log_line('Playing audio...'))
            try:
                play_audio(self.tx_audio, fs)
            except Exception as ex:
                self._append_diag(fmt_log_line('Playback error: ' + str(ex)))

    # Receive
    def _on_receive(self, e):
        if self._receiving:
            self._receiving = False
            self.btn_receive.text = 'Receive'
            self.btn_receive.bgcolor = ft.Colors.ORANGE_200
            self._append_diag(fmt_log_line('Receive stopped'))
            self.page.update()
            return
        self.recv_field.value = ''
        self.page.update()
        if _modem is None:
            self._append_diag(fmt_log_line('Receive: Modem module not loaded'))
            return
        self._receiving = True
        self.btn_receive.text = 'Stop'
        self.btn_receive.bgcolor = ft.Colors.RED_200
        self.page.update()

        def _run_receive():
            try:
                self._append_diag(fmt_log_line('Preparing modem for live receive...'))
                for k, v in self.settings.items():
                    if hasattr(_modem, k):
                        setattr(_modem, k, v)
                if hasattr(_modem, 'MODULATION'):
                    _modem.MODULATION = self.modulation
                    if self.modulation == 'BPSK':
                        _modem.BITS_PER_SYMBOL = 1
                    else:
                        _modem.BITS_PER_SYMBOL = 2
                    _modem.BITS_PER_OFDM_SYMBOL = _modem.Nsub * _modem.BITS_PER_SYMBOL
                if self.loaded_wav_path:
                    self._append_diag(fmt_log_line('Receiving from file: ' + self.loaded_wav_path))
                else:
                    self._append_diag(fmt_log_line('Receiving from microphone...'))
                if not hasattr(_modem, 'preamble_td') or _modem.preamble_td is None:
                    if hasattr(_modem, 'build_preamble'):
                        _modem.preamble_td = _modem.build_preamble()
                        self._append_diag(fmt_log_line('preamble_td built'))
                if hasattr(_modem, 'init_phases'):
                    _modem.init_phases()
                    self._append_diag(fmt_log_line('init_phases() called'))
                _modem.rx = np.array([], dtype=float) if not self.loaded_wav_path else _modem.rx
                _modem.abs_corr = np.array([], dtype=float)
                _modem.post_sync = False
                _modem.rx_syms_list = []
                _modem.Hk_smooth_list = []
                self._append_diag(fmt_log_line('Calling live_receive_and_process()'))
                _modem.live_receive_and_process()
                self._append_diag(fmt_log_line('live_receive_and_process() returned'))
            except Exception as ex:
                self._append_diag(fmt_log_line('Live receive error: ' + str(ex)))
            finally:
                self._receiving = False
                self.btn_receive.text = 'Receive'
                self.btn_receive.bgcolor = ft.Colors.ORANGE_200
                self.page.update()
        threading.Thread(target=_run_receive, daemon=True).start()

    # Copy received
    def _on_copy_received(self, e):
        content = self.recv_field.value
        if not content:
            self._append_diag(fmt_log_line('Copy: nothing to copy'))
            return
        self.page.set_clipboard(content)
        self._append_diag(fmt_log_line('Copied received text to clipboard'))

    # Settings dialog
    def _on_open_settings(self, e):
        field_controls = []
        self._settings_fields = {}
        for key in self.settings:
            val = self.settings[key]
            tf = ft.TextField(label=key, value=str(val), width=300)
            self._settings_fields[key] = tf
            field_controls.append(ft.Row(controls=[ft.Text(key + ':', width=200), tf]))
        def _apply_settings(e):
            for key, tf in self._settings_fields.items():
                raw = tf.value
                try:
                    if raw.isdigit() or (raw.startswith('-') and raw[1:].isdigit()):
                        newv = int(raw)
                    else:
                        try:
                            newv = float(raw)
                        except ValueError:
                            low = raw.strip().lower()
                            if low in ('true', 'false'):
                                newv = low == 'true'
                            else:
                                newv = raw
                except Exception:
                    newv = raw
                self.settings[key] = newv
                if _modem is not None and hasattr(_modem, key):
                    try:
                        setattr(_modem, key, newv)
                    except Exception:
                        pass
            self._append_diag(fmt_log_line('Settings applied'))
            self.page.close_dialog()
        def _cancel_settings(e):
            self.page.close_dialog()
        dlg = ft.AlertDialog(
            title=ft.Text('Settings'),
            content=ft.Column(controls=field_controls, scroll=ft.ScrollMode.AUTO, height=400),
            actions=[
                ft.TextButton('Apply', on_click=_apply_settings),
                ft.TextButton('Cancel', on_click=_cancel_settings),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.dialog = dlg
        dlg.open = True
        self.page.update()

    def _append_diag(self, text):
        self.diag_field.value = (self.diag_field.value or '') + text
        self.page.update()


def main(page):
    page.theme_mode = ft.ThemeMode.LIGHT
    OfdmApp(page)


if __name__ == '__main__':
    ft.run(main)
