# ofdm_gui_kivy.py  — полный обновлённый файл
# Изменения в этой версии:
# - Поля ввода/диагностики/received масштабируемые, поддерживают copy/paste
# - Input Text в Text Mode гарантированно поддерживает выделение/копирование/вставку
# - Вся логика playback через временный WAV остается
# - Верхние и нижние панели: 3 кнопки в ряд равномерно заполняют ширину (size_hint_x=1/3)
# - Кнопка "Save received file" теперь открывает диалог, где поле "Filename" вынесено в верх, большего размера
# - В диалогах кнопки внизу имеют равную ширину и занимают всю ширину окна
# - Окно Settings: строки начинаются сверху, строки выше (высота 44), кнопки внизу поровну занимают ширину окна;
#   при закрытии Apply значения сразу записываются в self.settings и, при наличии модуля модема, присваиваются ему
# - При нажатии Receive поле received очищается перед началом приёма
# - Использует wav_utils для работы с WAV файлами (кроссплатформенность)

import os
import sys
import threading
import importlib
import json
import shutil
import tempfile
from datetime import datetime
from functools import partial

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.utils import platform as kivy_platform

# Platform detection
IS_ANDROID = kivy_platform == 'android'
IS_DESKTOP = kivy_platform in ('win', 'linux', 'macosx')

# Request Android permissions if needed
if IS_ANDROID:
    try:
        from android.permissions import request_permissions
        request_permissions(['READ_EXTERNAL_STORAGE', 
                           'WRITE_EXTERNAL_STORAGE',
                           'RECORD_AUDIO'])
    except Exception as e:
        print("Warning: Failed to request Android permissions: " + str(e))

# Import audio backend
try:
    from audio_backend import get_backend, play_audio as backend_play_audio, stop_audio as backend_stop_audio, is_playing as backend_is_playing
    _audio_backend = get_backend()
except Exception as e:
    print(f"Warning: Failed to import audio_backend: {e}")
    _audio_backend = None
    backend_play_audio = None
    backend_stop_audio = None
    backend_is_playing = None

import numpy as np
from wav_utils import write as save_wav

MODEM_MODULE_NAME = "test_modem_simple"

_modem = None
_import_err = None
try:
    _modem = importlib.import_module(MODEM_MODULE_NAME)
except Exception as e:
    import traceback
    _import_err = traceback.format_exc()
    _modem = None

def gattr(obj, name, default=None):
    return getattr(obj, name) if obj is not None and hasattr(obj, name) else default

class GuiLogger:
    def __init__(self, on_new_line):
        self._buf = ""
        self._lock = threading.Lock()
        self.on_new_line = on_new_line
    
    def write(self, txt):
        if not txt:
            return
        with self._lock:
            self._buf += txt
            while True:
                if "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    Clock.schedule_once(lambda dt, l=line + "\n": self.on_new_line(l),0)
                else:
                    break
    
    def flush(self):
        with self._lock:
            if self._buf:
                t = self._buf
                self._buf = ""
                Clock.schedule_once(lambda dt, l=t: self.on_new_line(l),0)

def fmt_log_line(msg):
    t = datetime.now().strftime("%H:%M:%S")
    return f"[{t}] {msg}\n"

def fallback_build_tx_audio_from_bytes(payload_bytes: bytes, mode_flag: bytes = b'F',
                                       filename_bytes: bytes = b'', settings=None):
    if _modem is None:
        raise RuntimeError("modem module required for fallback builder")
    
    fs = gattr(_modem, "fs", 48000)
    SYMBOL_LEN = gattr(_modem, "SYMBOL_LEN", None)
    if SYMBOL_LEN is None:
        Nfft = gattr(_modem, "Nfft", None)
        Ncp = gattr(_modem, "Ncp", None)
        if Nfft is None or Ncp is None:
            raise RuntimeError("modem lacks Nfft/Ncp, cannot compute SYMBOL_LEN in fallback builder")
        SYMBOL_LEN = Nfft + Ncp
    
    packet_blocks = int(gattr(_modem, "DEFAULT_PACKET_BLOCKS", 75))
    version = 0
    tx_type_bits = 0b11 if mode_flag == b'F' else (0b10 if mode_flag == b'T' else 0b00)
    
    import zlib
    crc_val = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    
    build_header = gattr(_modem, "build_header", None)
    if build_header is None:
        raise RuntimeError("modem does not expose build_header required for fallback builder")
    
    transmission_header = build_header(b'F' if mode_flag == b'F' else b'T',
                                       len(payload_bytes),
                                       filename_bytes=filename_bytes,
                                       packet_no=0,
                                       version=version,
                                       packet_blocks=packet_blocks,
                                       crc32=crc_val)
    
    bytes_left = payload_bytes
    packet_no = 0
    samples_list = []
    build_preamble = gattr(_modem, "build_preamble", None)
    if build_preamble is None:
        raise RuntimeError("modem does not expose build_preamble required for fallback builder")
    preamble_td = build_preamble()
    bytes_to_blocks = gattr(_modem, "bytes_to_ofdm_blocks_bytes", None)
    if bytes_to_blocks is None:
        raise RuntimeError("modem must expose bytes_to_ofdm_blocks_bytes for fallback builder")
    
    while True:
        is_first = (packet_no == 0)
        if is_first:
            header_bytes = transmission_header + transmission_header + transmission_header
        else:
            make_packet_header_bytes = gattr(_modem, "make_packet_header_bytes", None)
            if make_packet_header_bytes is None:
                header_bytes = bytes([0x00 | (tx_type_bits & 0x03),
                                      (packet_no >> 16) & 0xFF,
                                      (packet_no >> 8) & 0xFF,
                                      packet_no & 0xFF])
            else:
                header_bytes = make_packet_header_bytes(packet_no, tx_type_bits)
        
        allowed_blocks = packet_blocks
        
        lo = 0
        hi = len(bytes_left)
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
        if len(bytes_left) == 0:
            break
        if packet_no > 10000:
            break
    
    tx_concat = np.concatenate(samples_list) if len(samples_list) > 0 else np.array([], dtype=float)
    preroll_len = int(0.1 * fs)
    postroll_len = int(0.1 * fs)
    tx_out = np.concatenate((np.zeros(preroll_len, dtype=tx_concat.dtype), tx_concat, np.zeros(postroll_len, dtype=tx_concat.dtype)))
    info = {"samples": len(tx_out), "fs": fs, "packets": packet_no}
    return tx_out.astype(np.float64), info

def get_builder():
    if _modem is not None:
        b = gattr(_modem, "build_tx_audio_from_bytes", None)
        if b is not None:
            return b
    return fallback_build_tx_audio_from_bytes

# Audio playback using backend
def play_audio(arr, fs):
    if backend_play_audio is None:
        return "Playback not available (audio backend not loaded)"
    try:
        backend_play_audio(arr, fs)
    except Exception as e:
        return f"Playback error: {e}"
    return None

def stop_audio():
    if backend_stop_audio is not None:
        backend_stop_audio()

class OfdmRoot(BoxLayout):
    diag_text = ObjectProperty(None)
    received_text = ObjectProperty(None)
    input_text = ObjectProperty(None)
    mode_spinner = ObjectProperty(None)
    btn_select_file = ObjectProperty(None)
    
    last_rx_saved_path = None
    tx_audio = None
    tx_info = None
    selected_file = None
    settings = {}
    
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=6, padding=6, **kwargs)
        
        self.settings = {
            "Nfft":  gattr(_modem, "Nfft", 512),
            "Ncp":   gattr(_modem, "Ncp", 128),
            "REQUIRED_NSUB": gattr(_modem, "REQUIRED_NSUB", 48),
            "RS_DATA_BYTES": gattr(_modem, "RS_DATA_BYTES", 8),
            "RS_PARITY_BYTES": gattr(_modem, "RS_PARITY_BYTES", 4),
            "PHASE_METHOD": gattr(_modem, "PHASE_METHOD", "schroeder"),
            "DEFAULT_PACKET_BLOCKS": gattr(_modem, "DEFAULT_PACKET_BLOCKS", 75),
            "GAP_OFDM_SYMBOLS": gattr(_modem, "GAP_OFDM_SYMBOLS", 2)
        }
        
        self.logger = GuiLogger(self._on_new_log_line)
        sys.stdout = self.logger
        sys.stderr = self.logger
        
        self._build_top()
        self._build_middle()
        self._build_bottom()
    
    def _build_top(self):
        """Build top panel with mode selection and file selection"""
        top = BoxLayout(orientation='horizontal', size_hint_y=None, height='44dp', spacing=6)
        
        self.mode_spinner = Spinner(
            text='Text',
            values=('Text', 'File'),
            size_hint_x=1/3
        )
        self.mode_spinner.bind(text=self._on_mode_change)
        top.add_widget(self.mode_spinner)
        
        self.btn_select_file = Button(
            text='Select File',
            size_hint_x=1/3,
            disabled=True
        )
        self.btn_select_file.bind(on_release=self._select_file)
        top.add_widget(self.btn_select_file)
        
        btn_settings = Button(
            text='Settings',
            size_hint_x=1/3
        )
        btn_settings.bind(on_release=self._show_settings)
        top.add_widget(btn_settings)
        
        self.add_widget(top)
    
    def _build_middle(self):
        """Build middle panel with input, diagnostics and received text areas"""
        middle = BoxLayout(orientation='vertical', spacing=6)
        
        # Input text
        lbl_input = Label(text='Input Text:', size_hint_y=None, height='20dp', halign='left')
        lbl_input.bind(size=lbl_input.setter('text_size'))
        middle.add_widget(lbl_input)
        
        self.input_text = TextInput(
            multiline=True,
            size_hint_y=1,
            hint_text='Enter text to transmit...'
        )
        middle.add_widget(self.input_text)
        
        # Diagnostics
        lbl_diag = Label(text='Diagnostics:', size_hint_y=None, height='20dp', halign='left')
        lbl_diag.bind(size=lbl_diag.setter('text_size'))
        middle.add_widget(lbl_diag)
        
        self.diag_text = TextInput(
            multiline=True,
            size_hint_y=1,
            readonly=True,
            hint_text='Diagnostics output...'
        )
        middle.add_widget(self.diag_text)
        
        # Received text
        lbl_recv = Label(text='Received:', size_hint_y=None, height='20dp', halign='left')
        lbl_recv.bind(size=lbl_recv.setter('text_size'))
        middle.add_widget(lbl_recv)
        
        self.received_text = TextInput(
            multiline=True,
            size_hint_y=1,
            readonly=True,
            hint_text='Received text/files will appear here...'
        )
        middle.add_widget(self.received_text)
        
        self.add_widget(middle)
    
    def _build_bottom(self):
        """Build bottom panel with action buttons"""
        bottom = BoxLayout(orientation='horizontal', size_hint_y=None, height='44dp', spacing=6)
        
        btn_transmit = Button(
            text='Transmit',
            size_hint_x=1/3
        )
        btn_transmit.bind(on_release=self._transmit)
        bottom.add_widget(btn_transmit)
        
        btn_receive = Button(
            text='Receive',
            size_hint_x=1/3
        )
        btn_receive.bind(on_release=self._receive)
        bottom.add_widget(btn_receive)
        
        btn_save = Button(
            text='Save received file',
            size_hint_x=1/3
        )
        btn_save.bind(on_release=self._save_received)
        bottom.add_widget(btn_save)
        
        self.add_widget(bottom)
    
    def _on_mode_change(self, spinner, text):
        """Handle mode change between Text and File"""
        if text == 'File':
            self.btn_select_file.disabled = False
        else:
            self.btn_select_file.disabled = True
    
    def _select_file(self, *args):
        """Open file chooser dialog"""
        content = BoxLayout(orientation='vertical', spacing=6)
        
        filechooser = FileChooserListView()
        content.add_widget(filechooser)
        
        btn_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='44dp', spacing=6)
        
        btn_cancel = Button(text='Cancel', size_hint_x=1/2)
        btn_ok = Button(text='Select', size_hint_x=1/2)
        
        def on_cancel(*args):
            popup.dismiss()
        
        def on_select(*args):
            if filechooser.selection:
                self.selected_file = filechooser.selection[0]
                popup.dismiss()
        
        btn_cancel.bind(on_release=on_cancel)
        btn_ok.bind(on_release=on_select)
        
        btn_layout.add_widget(btn_cancel)
        btn_layout.add_widget(btn_ok)
        content.add_widget(btn_layout)
        
        popup = Popup(
            title='Select File',
            content=content,
            size_hint=(0.9, 0.9)
        )
        popup.open()
    
    def _show_settings(self, *args):
        """Show settings dialog"""
        content = BoxLayout(orientation='vertical', spacing=6)
        
        grid = GridLayout(cols=2, spacing=6, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))
        
        settings_widgets = {}
        for key, val in self.settings.items():
            lbl = Label(text=key + ':', size_hint_y=None, height='44dp', halign='right')
            lbl.bind(size=lbl.setter('text_size'))
            grid.add_widget(lbl)
            
            txt = TextInput(
                text=str(val),
                size_hint_y=None,
                height='44dp'
            )
            grid.add_widget(txt)
            settings_widgets[key] = txt
        
        scroll = ScrollView()
        scroll.add_widget(grid)
        content.add_widget(scroll)
        
        btn_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='44dp', spacing=6)
        
        btn_cancel = Button(text='Cancel', size_hint_x=1/2)
        btn_apply = Button(text='Apply', size_hint_x=1/2)
        
        def on_cancel(*args):
            popup.dismiss()
        
        def on_apply(*args):
            for key, txt in settings_widgets.items():
                try:
                    if isinstance(self.settings[key], int):
                        self.settings[key] = int(txt.text)
                    elif isinstance(self.settings[key], float):
                        self.settings[key] = float(txt.text)
                    else:
                        self.settings[key] = txt.text
                except Exception as e:
                    print(f"Error parsing setting {key}: {e}")
            
            # Apply to modem if available
            if _modem is not None:
                for key, val in self.settings.items():
                    if hasattr(_modem, key):
                        setattr(_modem, key, val)
            
            popup.dismiss()
        
        btn_cancel.bind(on_release=on_cancel)
        btn_apply.bind(on_release=on_apply)
        
        btn_layout.add_widget(btn_cancel)
        btn_layout.add_widget(btn_apply)
        content.add_widget(btn_layout)
        
        popup = Popup(
            title='Settings',
            content=content,
            size_hint=(0.9, 0.9)
        )
        popup.open()
    
    def _transmit(self, *args):
        """Handle transmit button press"""
        if _modem is None:
            print("ERROR: Modem module not loaded")
            return
        
        mode = self.mode_spinner.text
        print(f"Starting transmission in {mode} mode...")
        
        if mode == 'Text':
            text = self.input_text.text
            if not text:
                print("ERROR: No text to transmit")
                return
            payload = text.encode('utf-8')
            filename_bytes = b''
        else:
            if not self.selected_file:
                print("ERROR: No file selected")
                return
            try:
                with open(self.selected_file, 'rb') as f:
                    payload = f.read()
                filename_bytes = os.path.basename(self.selected_file).encode('utf-8')
            except Exception as e:
                print(f"ERROR reading file: {e}")
                return
        
        try:
            builder = get_builder()
            self.tx_audio, self.tx_info = builder(payload, b'F', filename_bytes, self.settings)
            print(f"TX audio prepared: {self.tx_info}")
            
            # Save to temp file and play
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_wav = f.name
                save_wav(temp_wav, int(gattr(_modem, "fs", 48000)), self.tx_audio)
            
            result = play_audio(self.tx_audio, int(gattr(_modem, "fs", 48000)))
            if result:
                print(f"Playback warning: {result}")
            else:
                print("Transmission started")
        
        except Exception as e:
            print(f"ERROR during transmission: {e}")
            import traceback
            traceback.print_exc()
    
    def _receive(self, *args):
        """Handle receive button press"""
        if _modem is None:
            print("ERROR: Modem module not loaded")
            return
        
        print("Starting reception...")
        self.received_text.text = ""  # Clear received text
        
        try:
            # This is a simplified version - actual reception would be more complex
            print("Reception not fully implemented in GUI - use test_modem_simple.py for now")
        except Exception as e:
            print(f"ERROR during reception: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_received(self, *args):
        """Save received file"""
        if not self.last_rx_saved_path or not os.path.exists(self.last_rx_saved_path):
            print("No received file to save")
            return
        
        content = BoxLayout(orientation='vertical', spacing=6)
        
        lbl = Label(text='Filename:', size_hint_y=None, height='20dp', halign='left')
        lbl.bind(size=lbl.setter('text_size'))
        content.add_widget(lbl)
        
        txt_filename = TextInput(
            text=os.path.basename(self.last_rx_saved_path),
            size_hint_y=None,
            height='44dp'
        )
        content.add_widget(txt_filename)
        
        btn_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='44dp', spacing=6)
        
        btn_cancel = Button(text='Cancel', size_hint_x=1/2)
        btn_save = Button(text='Save', size_hint_x=1/2)
        
        def on_cancel(*args):
            popup.dismiss()
        
        def on_save(*args):
            try:
                dest = os.path.join(os.getcwd(), txt_filename.text)
                shutil.copy2(self.last_rx_saved_path, dest)
                print(f"File saved to: {dest}")
                popup.dismiss()
            except Exception as e:
                print(f"ERROR saving file: {e}")
        
        btn_cancel.bind(on_release=on_cancel)
        btn_save.bind(on_release=on_save)
        
        btn_layout.add_widget(btn_cancel)
        btn_layout.add_widget(btn_save)
        content.add_widget(btn_layout)
        
        popup = Popup(
            title='Save Received File',
            content=content,
            size_hint=(0.9, 0.5)
        )
        popup.open()
    
    def _on_new_log_line(self, line):
        """Handle new log line"""
        self.diag_text.text += line
        # Scroll to bottom
        self.diag_text.cursor = (0, len(self.diag_text.text))

class OfdmApp(App):
    def build(self):
        return OfdmRoot()
    
    def on_start(self):
        print("OFDM GUI started")
        if _import_err:
            print("WARNING: Modem module import failed:")
            print(_import_err)
        else:
            print("Modem module loaded successfully")
        
        if _audio_backend is None:
            print("WARNING: Audio backend not available")
        else:
            print(f"Audio backend: {type(_audio_backend).__name__}")

if __name__ == '__main__':
    OfdmApp().run()
