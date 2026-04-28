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

try:
    import sounddevice as sd
except Exception:
    sd = None

import numpy as np

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
                    Clock.schedule_once(lambda dt, l=line + "\n": self.on_new_line(l), 0)
                else:
                    break

    def flush(self):
        with self._lock:
            if self._buf:
                t = self._buf
                self._buf = ""
                Clock.schedule_once(lambda dt, l=t: self.on_new_line(l), 0)

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

_play_lock = threading.Lock()
_is_playing = False

def play_via_tempfile(arr, fs):
    if sd is None:
        return "Playback not available (sounddevice not installed)"
    try:
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.integer):
            maxval = np.iinfo(a.dtype).max
            a = a.astype(np.float32) / float(maxval)
        else:
            a = a.astype(np.float32)
        peak = np.max(np.abs(a)) if a.size else 0.0
        if peak > 0 and peak < 0.98:
            a = a * (0.98 / peak)
    except Exception:
        a = arr

    try:
        max_abs = np.max(np.abs(a)) if a.size else 0.0
        if max_abs == 0:
            data_int16 = (a * 0).astype(np.int16)
        else:
            data_int16 = (a / max_abs * np.iinfo(np.int16).max).astype(np.int16)
    except Exception as e:
        return f"Conversion to int16 failed: {e}"

    tf = None
    try:
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tf_name = tf.name
        tf.close()
        try:
            from scipy.io import wavfile
            wavfile.write(tf_name, fs, data_int16)
        except Exception as e:
            try:
                os.unlink(tf_name)
            except Exception:
                pass
            return f"Failed to write temp WAV: {e}"

        try:
            from scipy.io import wavfile as wavread
            sr, wavd = wavread.read(tf_name)
            if wavd.dtype == np.int16:
                wav_arr = wavd.astype(np.float32) / np.iinfo(np.int16).max
            elif np.issubdtype(wavd.dtype, np.integer):
                wav_arr = wavd.astype(np.float32) / np.iinfo(wavd.dtype).max
            else:
                wav_arr = wavd.astype(np.float32)
            if wav_arr.ndim > 1:
                wav_arr = wav_arr[:, 0]
        except Exception:
            wav_arr = a
            sr = fs

        def _play_and_cleanup():
            global _is_playing
            try:
                _is_playing = True
                sd.stop()
                sd.play(wav_arr, sr)
                sd.wait()
            except Exception as e:
                try:
                    App.get_running_app().root.append_diag(fmt_log_line(f"Playback error: {e}"))
                except Exception:
                    pass
            finally:
                _is_playing = False
                try:
                    os.unlink(tf_name)
                except Exception:
                    pass

        threading.Thread(target=_play_and_cleanup, daemon=True).start()
        return None
    except Exception as e:
        if tf is not None:
            try:
                os.unlink(tf.name)
            except Exception:
                pass
        return f"Tempfile playback failed: {e}"

def play_audio(arr, fs):
    global _is_playing
    if sd is None:
        return "Playback not available (sounddevice not installed)"
    with _play_lock:
        if _is_playing:
            try:
                sd.stop()
            except Exception:
                pass
            _is_playing = False
            return None

        err = play_via_tempfile(arr, fs)
        if err is None:
            return None

        def _play_direct():
            global _is_playing
            try:
                _is_playing = True
                sd.stop()
                try:
                    a = np.asarray(arr)
                    if np.issubdtype(a.dtype, np.integer):
                        maxval = np.iinfo(a.dtype).max
                        a = a.astype(np.float32) / float(maxval)
                    else:
                        a = a.astype(np.float32)
                    peak = np.max(np.abs(a)) if a.size else 0.0
                    if peak > 0 and peak < 0.9:
                        a = a * (0.98 / peak)
                except Exception:
                    a = arr
                sd.play(a, fs)
                sd.wait()
            except Exception as e:
                App.get_running_app().root.append_diag(fmt_log_line(f"Playback error: {e}"))
            finally:
                _is_playing = False

        threading.Thread(target=_play_direct, daemon=True).start()
        return None

def stop_audio():
    global _is_playing
    try:
        sd.stop()
    except Exception:
        pass
    _is_playing = False

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
        self._build_input_area()
        self._build_controls()
        self._build_diag_recv()

        Clock.schedule_once(lambda dt: self._mode_changed(self.mode_spinner.text), 0)

        if _import_err:
            self.append_diag(fmt_log_line("[IMPORT-ERR] " + (_import_err.splitlines()[0] if _import_err else "import error")))

    def _build_top(self):
        top = BoxLayout(orientation='horizontal', size_hint_y=None, height=36, spacing=6)
        self.mode_spinner = Spinner(text='Text', values=('Text', 'File'), size_hint_x=1/3)
        self.mode_spinner.bind(text=lambda s, val: self._mode_changed(val))
        top.add_widget(self.mode_spinner)

        self.btn_select_file = Button(text='Select', size_hint_x=1/3)
        self.btn_select_file.bind(on_release=self.open_filechooser)
        top.add_widget(self.btn_select_file)

        btn_settings = Button(text='Settings', size_hint_x=1/3)
        btn_settings.bind(on_release=lambda b: self.open_settings())
        top.add_widget(btn_settings)

        self.add_widget(top)

    def _build_input_area(self):
        self.input_label = Label(text='Input', size_hint_y=None, height=24)
        self.add_widget(self.input_label)
        # Ensure copy/paste works: multiline True, write_tab True.
        # Keep input selectable when readonly. Kivy TextInput allows selection in readonly mode.
        self.input_text = TextInput(
            text='',
            multiline=True,
            write_tab=True,
            size_hint_y=0.36
        )
        self.input_text.hint_text = "Enter text here (or select a file in File Mode)"
        self.add_widget(self.input_text)

    def _build_controls(self):
        ctr = BoxLayout(orientation='horizontal', size_hint_y=None, height=44, spacing=6)
        btn_generate = Button(text='Generate', size_hint_x=1/3)
        btn_generate.bind(on_release=lambda b: self.on_generate_audio())
        ctr.add_widget(btn_generate)

        btn_save_audio = Button(text='Save WAV', size_hint_x=1/3)
        btn_save_audio.bind(on_release=lambda b: self.on_save_audio())
        ctr.add_widget(btn_save_audio)

        self.btn_transmit = Button(text='Play/Stop', size_hint_x=1/3)
        self.btn_transmit.bind(on_release=lambda b: self.on_transmit())
        ctr.add_widget(self.btn_transmit)

        self.add_widget(ctr)

    def _build_diag_recv(self):
        self.add_widget(Label(text='Diagnostics', size_hint_y=None, height=24))
        self.diag_text = TextInput(text='', readonly=True, multiline=True, write_tab=True, size_hint_y=0.28)
        self.add_widget(self.diag_text)

        self.add_widget(Label(text='Received', size_hint_y=None, height=24))
        self.received_text = TextInput(text='', readonly=False, multiline=True, write_tab=True, size_hint_y=0.36)
        self.add_widget(self.received_text)

        bottom = BoxLayout(orientation='horizontal', size_hint_y=None, height=48, spacing=6)
        btn_receive = Button(text='Receive', size_hint_x=1/3)
        btn_receive.bind(on_release=lambda b: self.on_receive())
        bottom.add_widget(btn_receive)

        btn_save_received = Button(text='Save received file', size_hint_x=1/3)
        btn_save_received.bind(on_release=lambda b: self.on_save_received())
        bottom.add_widget(btn_save_received)

        btn_copy_received = Button(text='Copy', size_hint_x=1/3)
        btn_copy_received.bind(on_release=lambda b: self.on_copy_received())
        bottom.add_widget(btn_copy_received)

        self.add_widget(bottom)

    def _on_new_log_line(self, line):
        self.append_diag(line)
        try:
            if "[RX TEXT]" in line:
                idx = line.find("[RX TEXT]")
                txt = line[idx + len("[RX TEXT]"):].strip()
                self.append_received(txt)
            if "[RX FILE]" in line and "Saved" in line:
                try:
                    after = line.split("Saved:", 1)[1].strip()
                    path = after.split(" ")[0].strip()
                    if os.path.exists(path):
                        self.last_rx_saved_path = path
                        self.append_diag(fmt_log_line(f"Detected saved received file: {path}"))
                except Exception:
                    pass
            if "Saved:" in line and "rx_" in line:
                try:
                    after = line.split("Saved:", 1)[1].strip()
                    path = after.split(" ")[0].strip()
                    if os.path.exists(path):
                        self.last_rx_saved_path = path
                        self.append_diag(fmt_log_line(f"Detected saved received file: {path}"))
                except Exception:
                    pass
        except Exception:
            pass

    def append_diag(self, text):
        def _append(dt):
            cur = self.diag_text.text
            self.diag_text.text = cur + text
            self.diag_text.cursor = (0, len(self.diag_text.text.splitlines()))
        Clock.schedule_once(_append, 0)

    def append_received(self, txt):
        def _append(dt):
            cur = self.received_text.text
            if cur:
                self.received_text.text = cur + "\n" + txt
            else:
                self.received_text.text = txt
        Clock.schedule_once(_append, 0)

    def _mode_changed(self, val):
        if val == "Text":
            # allow editing and copy/paste
            self.input_text.readonly = False
            self.input_label.text = "Input (text)"
            self.btn_select_file.disabled = True
        else:
            # make input readonly but still selectable (user can copy path)
            self.input_text.readonly = True
            self.input_label.text = "Selected file"
            if self.selected_file:
                self.input_text.text = self.selected_file
            else:
                self.input_text.text = "[File] No file selected"
            self.btn_select_file.disabled = False
        self.append_diag(fmt_log_line(f"Switched to {val}"))

    def open_filechooser(self, *a):
        content = BoxLayout(orientation='vertical')
        fc = FileChooserListView(path='.', size_hint=(1, 1))
        content.add_widget(fc)

        # filename row on top (allow typing name before selection if desired)
        name_row = BoxLayout(size_hint_y=None, height=40, spacing=6)
        name_row.add_widget(Label(text='Filename', size_hint_x=None, width=100))
        name_input = TextInput(text='', multiline=False, size_hint_x=1)
        name_row.add_widget(name_input)
        content.add_widget(name_row)

        btns = BoxLayout(size_hint_y=None, height=44, spacing=6)
        ok = Button(text='OK'); cancel = Button(text='Cancel')
        # make buttons equal width
        ok.size_hint_x = 0.5
        cancel.size_hint_x = 0.5
        btns.add_widget(ok); btns.add_widget(cancel)
        content.add_widget(btns)
        popup = Popup(title='Select file', content=content, size_hint=(0.9, 0.9))
        def _ok(b):
            selection = fc.selection
            if selection:
                self.selected_file = selection[0]
                self.input_text.text = self.selected_file
                self.input_label.text = "Selected file"
                self.append_diag(fmt_log_line(f"Selected file: {self.selected_file}"))
            else:
                # if user typed filename and chose folder, set that
                folder = fc.path
                typed = name_input.text.strip()
                if typed:
                    possible = os.path.join(folder, typed)
                    if os.path.exists(possible):
                        self.selected_file = possible
                        self.input_text.text = possible
                        self.input_label.text = "Selected file"
                        self.append_diag(fmt_log_line(f"Selected file: {self.selected_file}"))
            popup.dismiss()
        ok.bind(on_release=_ok)
        cancel.bind(on_release=lambda b: popup.dismiss())
        popup.open()

    def open_settings(self):
        # Use ScrollView + GridLayout so settings start at top and can scroll if many
        grid_root = BoxLayout(orientation='vertical')
        sv = ScrollView(size_hint=(1, 1))
        grid = GridLayout(cols=2, spacing=6, size_hint_y=None)
        row_h = 44
        keys = list(self.settings.keys())
        grid.bind(minimum_height=grid.setter('height'))
        for k in keys:
            lbl = Label(text=str(k), size_hint_y=None, height=row_h, halign='left', valign='middle')
            lbl.text_size = (lbl.width, None)
            grid.add_widget(lbl)
            val = self.settings.get(k)
            ti = TextInput(text=str(val), multiline=False, size_hint_y=None, height=row_h)
            grid.add_widget(ti)
        # store textinputs for apply
        # need to be able to access them later; attach to popup via closure
        sv.add_widget(grid)
        grid_root.add_widget(sv)

        # Buttons row: two equal buttons occupying full width
        btns = BoxLayout(size_hint_y=None, height=48, spacing=6)
        apply_btn = Button(text='Apply and Close'); cancel_btn = Button(text='Cancel')
        apply_btn.size_hint_x = 0.5
        cancel_btn.size_hint_x = 0.5
        btns.add_widget(apply_btn); btns.add_widget(cancel_btn)
        grid_root.add_widget(btns)

        popup = Popup(title='Settings (edit each parameter)', content=grid_root, size_hint=(0.9,0.9), auto_dismiss=False)

        # collect textinputs from grid children (they alternate label, input)
        text_inputs = {}
        children = list(grid.children)
        # grid.children is bottom-to-top; iterate keys to map
        inputs_list = [w for w in children if isinstance(w, TextInput)]
        inputs_list = inputs_list[::-1]  # reverse to natural top-to-bottom
        for i, k in enumerate(keys):
            if i < len(inputs_list):
                text_inputs[k] = inputs_list[i]
            else:
                text_inputs[k] = None

        def _apply(b):
            for k, ti in text_inputs.items():
                if ti is None:
                    continue
                raw = ti.text
                newv = raw
                try:
                    # try int
                    if raw.isdigit() or (raw.startswith('-') and raw[1:].isdigit()):
                        newv = int(raw)
                    else:
                        try:
                            fv = float(raw)
                            newv = fv
                        except Exception:
                            low = raw.strip().lower()
                            if low in ('true', 'false'):
                                newv = True if low == 'true' else False
                            else:
                                newv = raw
                except Exception:
                    newv = raw
                self.settings[k] = newv
                # immediate apply to modem globals if present
                if _modem is not None:
                    try:
                        if hasattr(_modem, k):
                            setattr(_modem, k, newv)
                    except Exception:
                        pass
            self.append_diag(fmt_log_line("Settings applied"))
            popup.dismiss()

        apply_btn.bind(on_release=_apply)
        cancel_btn.bind(on_release=lambda b: popup.dismiss())
        popup.open()

    def on_generate_audio(self):
        mode = self.mode_spinner.text
        try:
            if mode == "Text":
                text = self.input_text.text
                if not text:
                    self.append_diag(fmt_log_line("Generate: Enter text first"))
                    return
                payload = text.encode("utf-8")
                filename_bytes = b''
            else:
                if not self.selected_file:
                    self.append_diag(fmt_log_line("Generate: Select a file first"))
                    return
                with open(self.selected_file, "rb") as f:
                    payload = f.read()
                filename_bytes = os.path.basename(self.selected_file).encode("utf-8")
        except Exception as e:
            self.append_diag(fmt_log_line("Generate error: " + str(e)))
            return

        def _build():
            try:
                builder = get_builder()
                tx, info = builder(payload, mode_flag=(b'T' if mode == "Text" else b'F'),
                                    filename_bytes=filename_bytes, settings=self.settings)
                self.tx_audio = tx
                self.tx_info = info
                self.append_diag(fmt_log_line(f"Generated audio: samples={info.get('samples', len(tx))} fs={info.get('fs','n/a')} packets={info.get('packets','n/a')}"))
            except Exception as e:
                self.append_diag(fmt_log_line("Audio generation failed: " + str(e)))
        threading.Thread(target=_build, daemon=True).start()

    def on_save_audio(self):
        if self.tx_audio is None:
            self.append_diag(fmt_log_line("Save WAV: no audio (generate first)"))
            return
        content = BoxLayout(orientation='vertical')
        # Filename field on top
        top_row = BoxLayout(size_hint_y=None, height=48, spacing=6)
        top_row.add_widget(Label(text='Filename', size_hint_x=None, width=100))
        fn_input = TextInput(text='ofdm_tx.wav', multiline=False, size_hint_x=1)
        top_row.add_widget(fn_input)
        content.add_widget(top_row)

        fc = FileChooserListView(path='.', size_hint=(1, 1))
        content.add_widget(fc)

        btns = BoxLayout(size_hint_y=None, height=48, spacing=6)
        ok = Button(text='Save'); cancel = Button(text='Cancel')
        ok.size_hint_x = 0.5; cancel.size_hint_x = 0.5
        btns.add_widget(ok); btns.add_widget(cancel)
        content.add_widget(btns)
        popup = Popup(title='Save WAV', content=content, size_hint=(0.9,0.9))

        def _ok(b):
            folder = fc.path
            if fc.selection:
                # user selected file — use that
                path = fc.selection[0]
            else:
                name = fn_input.text.strip()
                if not name:
                    self.append_diag(fmt_log_line("Save WAV: filename empty"))
                    popup.dismiss()
                    return
                path = os.path.join(folder, name)
            try:
                fs = int(self.tx_info.get("fs", gattr(_modem, "fs", 48000))) if self.tx_info else int(gattr(_modem, "fs", 48000))
                from scipy.io import wavfile
                max_abs = np.max(np.abs(self.tx_audio)) if self.tx_audio.size else 0.0
                if max_abs == 0:
                    data_int16 = (self.tx_audio * 0).astype(np.int16)
                else:
                    data_int16 = (self.tx_audio / max_abs * np.iinfo(np.int16).max).astype(np.int16)
                wavfile.write(path, fs, data_int16)
                self.append_diag(fmt_log_line(f"Saved WAV to {path}"))
            except Exception as e:
                self.append_diag(fmt_log_line(f"Save WAV failed: {e}"))
            popup.dismiss()

        ok.bind(on_release=_ok)
        cancel.bind(on_release=lambda b: popup.dismiss())
        popup.open()

    def on_transmit(self):
        global _is_playing
        if self.tx_audio is None:
            self.on_generate_audio()
            def _wait_and_play():
                import time
                for _ in range(200):
                    if self.tx_audio is not None:
                        break
                    time.sleep(0.05)
                if self.tx_audio is None:
                    self.append_diag(fmt_log_line("Transmit aborted: generation timed out"))
                    return
                fs = int(self.tx_info.get("fs", gattr(_modem, "fs", 48000)))
                self.append_diag(fmt_log_line("Playing generated audio... (press Play/Stop to stop)"))
                err = play_audio(self.tx_audio, fs)
                if err:
                    self.append_diag(fmt_log_line(err))
            threading.Thread(target=_wait_and_play, daemon=True).start()
        else:
            if _is_playing:
                stop_audio()
                self.append_diag(fmt_log_line("Playback stopped"))
            else:
                fs = int(self.tx_info.get("fs", gattr(_modem, "fs", 48000)))
                self.append_diag(fmt_log_line("Playing audio... (press Play/Stop to stop)"))
                err = play_audio(self.tx_audio, fs)
                if err:
                    self.append_diag(fmt_log_line(err))

    def on_receive(self):
        # Clear received field for new incoming text immediately
        def _clear_received(dt):
            self.received_text.text = ""
        Clock.schedule_once(_clear_received, 0)

        if _modem is None:
            self.append_diag(fmt_log_line("Receive: Modem module not loaded"))
            return
        def run_live():
            try:
                self.append_diag(fmt_log_line("Preparing modem for live receive..."))
                try:
                    for k, v in self.settings.items():
                        if hasattr(_modem, k):
                            setattr(_modem, k, v)
                except Exception as e:
                    self.append_diag(fmt_log_line(f"Failed to apply settings into modem globals: {e}"))
                if not hasattr(_modem, "preamble_td") or _modem.preamble_td is None:
                    if hasattr(_modem, "build_preamble"):
                        try:
                            _modem.preamble_td = _modem.build_preamble()
                            self.append_diag(fmt_log_line("preamble_td built"))
                        except Exception as e:
                            self.append_diag(fmt_log_line(f"build_preamble() failed: {e}"))
                    else:
                        self.append_diag(fmt_log_line("Warning: build_preamble() not found in modem module"))
                if hasattr(_modem, "init_phases"):
                    try:
                        _modem.init_phases()
                        self.append_diag(fmt_log_line("init_phases() called"))
                    except Exception as e:
                        self.append_diag(fmt_log_line(f"init_phases() failed: {e}"))
                if not hasattr(_modem, "rx"):
                    _modem.rx = np.array([], dtype=float)
                if not hasattr(_modem, "abs_corr"):
                    _modem.abs_corr = np.array([], dtype=float)
                _modem.post_sync = False
                _modem.rx_syms_list = []
                _modem.Hk_smooth_list = []
                self.append_diag(fmt_log_line("Calling live_receive_and_process()"))
                _modem.live_receive_and_process()
                self.append_diag(fmt_log_line("live_receive_and_process() returned"))
            except Exception as e:
                self.append_diag(fmt_log_line("Live receive error: " + str(e)))
        threading.Thread(target=run_live, daemon=True).start()

    def on_save_received(self):
        mode = self.mode_spinner.text
        if mode == "Text":
            content = self.received_text.text
            if not content:
                self.append_diag(fmt_log_line("Save: Nothing received yet"))
                return
            root = BoxLayout(orientation='vertical')
            # Filename at top (large)
            top_row = BoxLayout(size_hint_y=None, height=48, spacing=6)
            top_row.add_widget(Label(text='Filename', size_hint_x=None, width=100))
            name_input = TextInput(text='received.txt', multiline=False, size_hint_x=1)
            top_row.add_widget(name_input)
            root.add_widget(top_row)

            fc = FileChooserListView(path='.', size_hint=(1,1))
            root.add_widget(fc)
            btns = BoxLayout(size_hint_y=None, height=48, spacing=6)
            ok = Button(text='Save'); cancel = Button(text='Cancel')
            ok.size_hint_x = 0.5; cancel.size_hint_x = 0.5
            btns.add_widget(ok); btns.add_widget(cancel)
            root.add_widget(btns)
            popup = Popup(title='Save received text as', content=root, size_hint=(0.9,0.9))
            def _ok(b):
                sel_dir = fc.path
                if fc.selection:
                    # if user selected a file, use folder or file selection
                    sel = fc.selection[0]
                    if os.path.isdir(sel):
                        folder = sel
                    else:
                        folder = os.path.dirname(sel)
                else:
                    folder = sel_dir
                filename = name_input.text.strip()
                if not filename:
                    self.append_diag(fmt_log_line("Save: filename empty"))
                    popup.dismiss()
                    return
                dest = os.path.join(folder, filename)
                try:
                    with open(dest, "w", encoding="utf-8") as f:
                        f.write(content)
                    self.append_diag(fmt_log_line(f"Saved received text to {dest}"))
                except Exception as e:
                    self.append_diag(fmt_log_line("Failed to save received text: " + str(e)))
                popup.dismiss()
            ok.bind(on_release=_ok)
            cancel.bind(on_release=lambda b: popup.dismiss())
            popup.open()
            return

        # File mode: if a received file path was recorded, allow choosing name and folder to copy to;
        if self.last_rx_saved_path and os.path.exists(self.last_rx_saved_path):
            root = BoxLayout(orientation='vertical')
            top_row = BoxLayout(size_hint_y=None, height=48, spacing=6)
            top_row.add_widget(Label(text='Filename', size_hint_x=None, width=100))
            default_name = os.path.basename(self.last_rx_saved_path)
            name_input = TextInput(text=default_name, multiline=False, size_hint_x=1)
            top_row.add_widget(name_input)
            root.add_widget(top_row)

            fc = FileChooserListView(path=os.path.dirname(self.last_rx_saved_path) or '.', size_hint=(1,1))
            root.add_widget(fc)
            btns = BoxLayout(size_hint_y=None, height=48, spacing=6)
            ok = Button(text='Save'); cancel = Button(text='Cancel')
            ok.size_hint_x = 0.5; cancel.size_hint_x = 0.5
            btns.add_widget(ok); btns.add_widget(cancel)
            root.add_widget(btns)
            popup = Popup(title='Save received file as', content=root, size_hint=(0.9,0.9))
            def _ok(b):
                dest_dir = fc.path if not fc.selection else (fc.selection[0] if os.path.isdir(fc.selection[0]) else os.path.dirname(fc.selection[0]))
                filename = name_input.text.strip()
                if not filename:
                    self.append_diag(fmt_log_line("Save received file: filename empty"))
                    popup.dismiss()
                    return
                dest = os.path.join(dest_dir, filename)
                try:
                    shutil.copyfile(self.last_rx_saved_path, dest)
                    self.append_diag(fmt_log_line(f"Copied received file to {dest}"))
                except Exception as e:
                    self.append_diag(fmt_log_line("Failed to copy received file: " + str(e)))
                popup.dismiss()
            ok.bind(on_release=_ok)
            cancel.bind(on_release=lambda b: popup.dismiss())
            popup.open()
            return

        # fallback to saving received_text content (shouldn't reach often)
        content = self.received_text.text
        if not content:
            self.append_diag(fmt_log_line("Save: Nothing received yet"))
            return
        root = BoxLayout(orientation='vertical')
        top_row = BoxLayout(size_hint_y=None, height=48, spacing=6)
        top_row.add_widget(Label(text='Filename', size_hint_x=None, width=100))
        name_input = TextInput(text='received.bin', multiline=False, size_hint_x=1)
        top_row.add_widget(name_input)
        root.add_widget(top_row)

        fc = FileChooserListView(path='.', size_hint=(1,1))
        root.add_widget(fc)
        btns = BoxLayout(size_hint_y=None, height=48, spacing=6)
        ok = Button(text='Save'); cancel = Button(text='Cancel')
        ok.size_hint_x = 0.5; cancel.size_hint_x = 0.5
        btns.add_widget(ok); btns.add_widget(cancel)
        root.add_widget(btns)
        popup = Popup(title='Save received file as', content=root, size_hint=(0.9,0.9))
        def _ok2(b):
            dest_dir = fc.path if not fc.selection else (fc.selection[0] if os.path.isdir(fc.selection[0]) else os.path.dirname(fc.selection[0]))
            filename = name_input.text.strip()
            if not filename:
                self.append_diag(fmt_log_line("Save received file: filename empty"))
                popup.dismiss()
                return
            dest = os.path.join(dest_dir, filename)
            try:
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(content)
                self.append_diag(fmt_log_line(f"Saved received text to {dest}"))
            except Exception as e:
                self.append_diag(fmt_log_line("Failed to save received text: " + str(e)))
            popup.dismiss()
        ok.bind(on_release=_ok2)
        cancel.bind(on_release=lambda b: popup.dismiss())
        popup.open()

    def on_copy_received(self):
        content = self.received_text.text
        if not content:
            self.append_diag(fmt_log_line("Copy: nothing to copy"))
            return
        from kivy.core.clipboard import Clipboard
        Clipboard.copy(content)
        self.append_diag(fmt_log_line("Copied received text to clipboard"))

class OfdmApp(App):
    def build(self):
        Window.size = (900, 1000)
        root = OfdmRoot()
        return root

if __name__ == "__main__":
    OfdmApp().run()

