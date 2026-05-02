#!/usr/bin/env python3
# ofdm_gui_pyqt_modified.py
# Vertical PyQt5 GUI for OFDM modem
# - Uses imported module (test_modem_simple) instead of subprocess
# - Allows direct modulation selection (QPSK/BPSK) via QComboBox
# - Redirects modem stdout/stderr into GUI
# - Fills "Received" text area when modem prints [RX TEXT]
# - File Mode: use "Save received file" to save the last received file
# - Text Mode: shows received text in Received area

import sys
import os
import threading
import importlib
import traceback
from datetime import datetime
from functools import partial

from PyQt5 import QtCore, QtWidgets
import numpy as np

# Try to import modem module
MODEM_MODULE_NAME = "test_modem_simple"

_modem = None
_import_err = None
try:
    _modem = importlib.import_module(MODEM_MODULE_NAME)
except Exception as e:
    _import_err = traceback.format_exc()
    _modem = None

def gattr(obj, name, default=None):
    return getattr(obj, name) if obj is not None and hasattr(obj, name) else default

# Logger signal
class Logger(QtCore.QObject):
    new = QtCore.pyqtSignal(str)

logger = Logger()

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    logger.new.emit(f"[{t}] {msg}\n")

# Redirect stdout/stderr into logger
class StdRedirector:
    def __init__(self):
        self._buf = ""
        self._lock = threading.Lock()
    
    def write(self, txt):
        if not txt:
            return
        with self._lock:
            self._buf += txt
            # extract full lines (keep line endings)
            while True:
                if "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    # re-add the newline so downstream sees the line similar to original
                    logger.new.emit(line + "\n")
                else:
                    break
    
    def flush(self):
        # emit any remaining partial content as a line (avoid losing tail)
        with self._lock:
            if self._buf:
                logger.new.emit(self._buf)
                self._buf = ""
        pass

# Settings dialog
class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.settings = settings.copy()
        
        layout = QtWidgets.QVBoxLayout()
        
        # Use ScrollView for many settings
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QFormLayout(scroll_widget)
        
        self.inputs = {}
        for key, value in sorted(self.settings.items()):
            label = QtWidgets.QLabel(str(key))
            input_field = QtWidgets.QLineEdit(str(value))
            scroll_layout.addRow(label, input_field)
            self.inputs[key] = input_field
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton("Apply")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
        apply_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
    
    def get_settings(self):
        new_settings = {}
        for key, input_field in self.inputs.items():
            value = input_field.text()
            # Try to preserve original type
            original = self.settings[key]
            if isinstance(original, bool):
                new_settings[key] = value.lower() in ('true', '1', 'yes')
            elif isinstance(original, int):
                try:
                    new_settings[key] = int(value)
                except ValueError:
                    new_settings[key] = original
            elif isinstance(original, float):
                try:
                    new_settings[key] = float(value)
                except ValueError:
                    new_settings[key] = original
            else:
                new_settings[key] = value
        return new_settings

# GUI class
class OfdmGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        self.selected_file = None
        self.rx_file = None
        self.received_data = ""
        self.tx_audio_data = None  # Store generated audio for saving
        self.is_running = False  # Protection against simultaneous operations
        self.modulation = 'QPSK'  # Default modulation
        
        # Default settings
        self.settings = {
            "Nfft":  gattr(_modem, "Nfft", 512),
            "Ncp":  gattr(_modem, "Ncp", 128),
            "REQUIRED_NSUB": gattr(_modem, "REQUIRED_NSUB", 48),
            "RS_DATA_BYTES": gattr(_modem, "RS_DATA_BYTES", 8),
            "RS_PARITY_BYTES": gattr(_modem, "RS_PARITY_BYTES", 4),
            "PHASE_METHOD": gattr(_modem, "PHASE_METHOD", "schroeder"),
            "DEFAULT_PACKET_BLOCKS": gattr(_modem, "DEFAULT_PACKET_BLOCKS", 75),
            "GAP_OFDM_SYMBOLS": gattr(_modem, "GAP_OFDM_SYMBOLS", 2),
            "fs": gattr(_modem, "fs", 48000),
            "PREROLL_SEC": 0.25,
            "POSTROLL_SEC": 0.50,
            "NOISE_LEVEL": 0.0001,
            "NOISE_SEED": 20231107,
        }
        
        # Redirect stdout/stderr
        sys.stdout = StdRedirector()
        sys.stderr = sys.stdout
        
        logger.new.connect(self.on_new_log)
        
        if _import_err:
            log(f"Modem import error: {_import_err.splitlines()[0]}")
    
    def init_ui(self):
        self.setWindowTitle("OFDM Acoustic Modem GUI")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Text", "File"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
        mode_layout.addWidget(self.mode_combo)
        
        self.select_file_btn = QtWidgets.QPushButton("Select File")
        self.select_file_btn.clicked.connect(self.select_file)
        self.select_file_btn.setEnabled(False)
        mode_layout.addWidget(self.select_file_btn)
        
        # Modulation selection
        mode_layout.addWidget(QtWidgets.QLabel("Modulation:"))
        self.modulation_combo = QtWidgets.QComboBox()
        self.modulation_combo.addItems(["QPSK", "BPSK"])
        self.modulation_combo.setCurrentText("QPSK")  # Default QPSK
        self.modulation_combo.currentTextChanged.connect(self.on_modulation_change)
        mode_layout.addWidget(self.modulation_combo)
        
        settings_btn = QtWidgets.QPushButton("Settings")
        settings_btn.clicked.connect(self.show_settings)
        mode_layout.addWidget(settings_btn)
        
        layout.addLayout(mode_layout)
        
        # Input text
        layout.addWidget(QtWidgets.QLabel("Input Text:"))
        self.input_text = QtWidgets.QTextEdit()
        layout.addWidget(self.input_text)
        
        # Diagnostics
        layout.addWidget(QtWidgets.QLabel("Diagnostics:"))
        self.diag_text = QtWidgets.QTextEdit()
        self.diag_text.setReadOnly(True)
        layout.addWidget(self.diag_text)
        
        # Received
        layout.addWidget(QtWidgets.QLabel("Received:"))
        self.received_text = QtWidgets.QTextEdit()
        self.received_text.setReadOnly(True)
        layout.addWidget(self.received_text)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.transmit_btn = QtWidgets.QPushButton("Generate & Play")
        self.transmit_btn.clicked.connect(self.transmit)
        btn_layout.addWidget(self.transmit_btn)
        
        self.receive_btn = QtWidgets.QPushButton("Receive")
        self.receive_btn.clicked.connect(self.receive)
        btn_layout.addWidget(self.receive_btn)
        
        save_btn = QtWidgets.QPushButton("Save Received File")
        save_btn.clicked.connect(self.save_received)
        btn_layout.addWidget(save_btn)
        
        # New buttons from Kivy version
        save_wav_btn = QtWidgets.QPushButton("Save WAV")
        save_wav_btn.clicked.connect(self.save_wav)
        btn_layout.addWidget(save_wav_btn)
        
        copy_btn = QtWidgets.QPushButton("Copy")
        copy_btn.clicked.connect(self.copy_received)
        btn_layout.addWidget(copy_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def on_modulation_change(self, text):
        """Handle modulation change"""
        self.modulation = text
        if _modem is not None:
            try:
                _modem.MODULATION = text
                if text == 'BPSK':
                    _modem.BITS_PER_SYMBOL = 1
                else:
                    _modem.BITS_PER_SYMBOL = 2
                _modem.BITS_PER_OFDM_SYMBOL = _modem.Nsub * _modem.BITS_PER_SYMBOL
                log(f"Modulation set to {text}")
            except Exception as e:
                log(f"Failed to set modulation: {e}")
        else:
            log(f"Modulation set to {text} (modem not loaded)")
    
    def on_mode_change(self, text):
        self.select_file_btn.setEnabled(text == "File")
    
    def select_file(self):
        # Improved file selection with filename input
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout()
        
        fc = QtWidgets.QFileDialog()
        fc.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        
        # Use a simpler approach: standard dialog with option to type name
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File")
        if fname:
            self.selected_file = fname
            self.input_text.setText(fname)
    
    def select_rx_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select WAV File for Receive", "", "WAV Files (*.wav)")
        if fname:
            self.rx_file = fname
            log(f"Receive file selected: {fname}")
    
    def show_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.settings = dialog.get_settings()
            # Apply settings to modem
            if _modem is not None:
                for k, v in self.settings.items():
                    if hasattr(_modem, k):
                        try:
                            setattr(_modem, k, v)
                        except Exception as e:
                            log(f"Failed to set {k}: {e}")
            log("Settings updated")
    
    def transmit(self):
        if self.is_running:
            QtWidgets.QMessageBox.warning(self, "Warning", "Operation already in progress")
            return
        
        mode = 'T'  # Transmit
        
        if self.mode_combo.currentText() == "Text":
            text = self.input_text.toPlainText()
            if not text:
                QtWidgets.QMessageBox.warning(self, "Warning", "No text to transmit")
                return
            submode = 'T'  # Text mode
            log(f"Starting text transmission with {self.modulation}...")
            # Apply modulation before transmit
            self.on_modulation_change(self.modulation_combo.currentText())
            # Generate audio
            self.generate_audio(text, is_text=True)
        else:
            if not self.selected_file:
                QtWidgets.QMessageBox.warning(self, "Warning", "No file selected")
                return
            submode = 'F'  # File mode
            log(f"Starting file transmission: {self.selected_file} with {self.modulation}")
            # Apply modulation before transmit
            self.on_modulation_change(self.modulation_combo.currentText())
            # Generate audio from file
            self.generate_audio(self.selected_file, is_text=False)
    
    def generate_audio(self, data, is_text=True):
        """Generate audio from text or file"""
        if _modem is None:
            log("Modem module not loaded")
            return
        
        self.is_running = True
        self.update_button_states()
        
        def _build():
            try:
                # Apply settings to modem
                for k, v in self.settings.items():
                    if hasattr(_modem, k):
                        try:
                            setattr(_modem, k, v)
                        except Exception:
                            pass
                
                # Apply modulation
                self.on_modulation_change(self.modulation_combo.currentText())
                
                if is_text:
                    payload = data.encode("utf-8")
                    filename_bytes = b''
                    mode_flag = b'T'
                else:
                    with open(data, "rb") as f:
                        payload = f.read()
                    filename_bytes = os.path.basename(data).encode("utf-8")
                    mode_flag = b'F'
                
                # Try to use modem's build_tx_audio_from_bytes if available
                if hasattr(_modem, 'build_tx_audio_from_bytes'):
                    tx, info = _modem.build_tx_audio_from_bytes(payload, mode_flag=mode_flag,
                                                          filename_bytes=filename_bytes)
                else:
                    # Fallback: use our own implementation
                    tx, info = self.fallback_build_tx_audio(payload, mode_flag, filename_bytes)
                
                self.tx_audio_data = tx
                log(f"Generated audio: samples={len(tx)} fs={info.get('fs', 'n/a')} packets={info.get('packets', 'n/a')}")
                
                # Play audio
                self.play_audio(tx, info.get('fs', 48000))
                
            except Exception as e:
                log(f"Audio generation failed: {e}")
            finally:
                self.is_running = False
                self.update_button_states()
        
        thread = threading.Thread(target=_build)
        thread.daemon = True
        thread.start()
    
    def fallback_build_tx_audio(self, payload_bytes, mode_flag=b'F', filename_bytes=b''):
        """Fallback audio builder if modem doesn't expose build_tx_audio_from_bytes"""
        if _modem is None:
            raise RuntimeError("modem module required for fallback builder")
        
        fs = gattr(_modem, "fs", 48000)
        Nfft = gattr(_modem, "Nfft", 512)
        Ncp = gattr(_modem, "Ncp", 128)
        SYMBOL_LEN = Nfft + Ncp
        
        packet_blocks = int(gattr(_modem, "DEFAULT_PACKET_BLOCKS", 75))
        version = 0
        tx_type_bits = 0b11 if mode_flag == b'F' else (0b10 if mode_flag == b'T' else 0b00)
        
        import zlib
        crc_val = zlib.crc32(payload_bytes) & 0xFFFFFFFF
        
        if hasattr(_modem, 'build_header'):
            transmission_header = _modem.build_header(b'F' if mode_flag == b'F' else b'T',
                                                   len(payload_bytes),
                                                   filename_bytes=filename_bytes,
                                                   packet_no=0,
                                                   version=version,
                                                   packet_blocks=packet_blocks,
                                                   crc32=crc_val)
        else:
            raise RuntimeError("modem does not expose build_header")
        
        bytes_left = payload_bytes
        packet_no = 0
        samples_list = []
        
        if hasattr(_modem, 'build_preamble'):
            preamble_td = _modem.build_preamble()
        else:
            raise RuntimeError("modem does not expose build_preamble")
        
        bytes_to_blocks = gattr(_modem, "bytes_to_ofdm_blocks_bytes", None)
        if bytes_to_blocks is None:
            raise RuntimeError("modem must expose bytes_to_ofdm_blocks_bytes")
        
        while True:
            is_first = (packet_no == 0)
            if is_first:
                header_bytes = transmission_header + transmission_header + transmission_header
            else:
                if hasattr(_modem, 'make_packet_header_bytes'):
                    header_bytes = _modem.make_packet_header_bytes(packet_no, tx_type_bits)
                else:
                    header_bytes = bytes([0x00 | (tx_type_bits & 0x03),
                                          (packet_no >> 16) & 0xFF,
                                          (packet_no >> 8) & 0xFF,
                                          packet_no & 0xFF])
            
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
    
    def play_audio(self, arr, fs):
        """Play audio using sounddevice or save to temp file"""
        try:
            import sounddevice as sd
            # Normalize
            max_val = np.max(np.abs(arr))
            if max_val > 0:
                arr = arr / max_val * 0.98
            sd.play(arr.astype(np.float32), fs)
            sd.wait()
        except ImportError:
            # Fallback: save to temp file and play with system player
            try:
                import tempfile
                import subprocess
                max_abs = np.max(np.abs(arr))
                if max_abs == 0:
                    data_int16 = (arr * 0).astype(np.int16)
                else:
                    data_int16 = (arr / max_abs * np.iinfo(np.int16).max).astype(np.int16)
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    from scipy.io import wavfile
                    wavfile.write(f.name, fs, data_int16)
                    temp_name = f.name
                
                # Play with system player (macOS)
                subprocess.Popen(["open", temp_name])
            except Exception as e:
                log(f"Playback failed: {e}")
        except Exception as e:
            log(f"Playback error: {e}")
    
    def receive(self):
        if self.is_running:
            QtWidgets.QMessageBox.warning(self, "Warning", "Operation already in progress")
            return
        
        log("Starting reception with {} modulation...".format(self.modulation))
        self.received_text.clear()
        
        # Apply modulation before receive
        self.on_modulation_change(self.modulation_combo.currentText())
        
        # Ask user for source
        items = ["mic", "file"]
        item, ok = QtWidgets.QInputDialog.getItem(self, "Select Source", "Receive from:", items, 0, False)
        if not ok:
            return
        
        self.is_running = True
        self.update_button_states()
        
        def run_receive():
            try:
                if _modem is None:
                    log("Modem module not loaded")
                    return
                
                # Apply settings to modem
                for k, v in self.settings.items():
                    if hasattr(_modem, k):
                        try:
                            setattr(_modem, k, v)
                        except Exception:
                            pass
                
                # Apply modulation
                self.on_modulation_change(self.modulation_combo.currentText())
                
                # Ensure preamble and phases are ready
                if not hasattr(_modem, "preamble_td") or _modem.preamble_td is None:
                    if hasattr(_modem, "build_preamble"):
                        try:
                            _modem.preamble_td = _modem.build_preamble()
                            log("preamble_td built")
                        except Exception as e:
                            log(f"build_preamble() failed: {e}")
                
                if hasattr(_modem, "init_phases"):
                    try:
                        _modem.init_phases()
                        log("init_phases() called")
                    except Exception as e:
                        log(f"init_phases() failed: {e}")
                
                if item == "file":
                    fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select WAV File", "", "WAV Files (*.wav)")
                    if not fname:
                        return
                    log(f"Receiving from file: {fname}")
                    # Try to use modem's file receive function
                    for func_name in ("receive_from_wav", "file_receive", "process_wav_file", "receive_wav"):
                        fn = gattr(_modem, func_name, None)
                        if fn is not None and callable(fn):
                            try:
                                log(f"Calling modem.{func_name}()")
                                fn(fname)
                                log(f"modem.{func_name}() returned")
                                break
                            except Exception as e:
                                log(f"modem.{func_name}() failed: {e}")
                else:
                    log("Receiving from microphone...")
                    if hasattr(_modem, "live_receive_and_process"):
                        _modem.live_receive_and_process()
                    else:
                        log("live_receive_and_process() not found in modem")
                
            except Exception as e:
                log(f"Receive error: {e}")
            finally:
                self.is_running = False
                self.update_button_states()
        
        thread = threading.Thread(target=run_receive)
        thread.daemon = True
        thread.start()
    
    def on_operation_finished(self):
        self.is_running = False
        self.update_button_states()
    
    def update_button_states(self):
        self.transmit_btn.setEnabled(not self.is_running)
        self.receive_btn.setEnabled(not self.is_running)
    
    def save_received(self):
        if not self.received_data:
            QtWidgets.QMessageBox.warning(self, "Warning", "No received data to save")
            return
        
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Received Data", 
                                                        "received_data.txt")
        if fname:
            try:
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(self.received_data)
                log(f"Data saved to: {fname}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
    
    def save_wav(self):
        """Save generated WAV file from transmission"""
        # Check if we have a generated WAV file
        wav_path = "ofdm_acoustic_tx_with_noise.wav"
        if os.path.exists(wav_path):
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save WAV File", 
                                                                    wav_path, "WAV Files (*.wav)")
            if fname:
                try:
                    import shutil
                    shutil.copy2(wav_path, fname)
                    log(f"WAV file saved to: {fname}")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save WAV file: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "No WAV file generated yet. Please transmit first.")
    
    def copy_received(self):
        """Copy received text to clipboard"""
        if self.received_text.toPlainText():
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(self.received_text.toPlainText())
            log("Received text copied to clipboard")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "No received text to copy")
    
    def on_modem_output(self, line):
        self.diag_text.append(line)
        # Scroll to bottom
        self.diag_text.verticalScrollBar().setValue(self.diag_text.verticalScrollBar().maximum())
        
        # Check for received text
        if '[RX TEXT]' in line:
            text = line.split('[RX TEXT]', 1)[-1].strip()
            self.received_text.append(text)
            self.received_data += text + "\n"
        elif '[RX FILE]' in line:
            self.received_text.append(f"File received: {line.strip()}")
        
        # Parse diagnostic information
        self.parse_diagnostics(line)
    
    def parse_diagnostics(self, line):
        """Parse and display diagnostic information from modem output"""
        # Parse configuration
        if '[CFG]' in line:
            self.diag_text.append(f"<b>Configuration:</b> {line.strip()}")
        # Parse phase information
        elif '[PHASE]' in line:
            self.diag_text.append(f"<b>Phase Method:</b> {line.strip()}")
        # Parse transmission info
        elif '[TX]' in line:
            self.diag_text.append(f"<b>Transmission:</b> {line.strip()}")
        # Parse reception info
        elif '[RX' in line:
            self.diag_text.append(f"<b>Reception:</b> {line.strip()}")
        # Parse debug info
        elif '[DBG]' in line:
            self.diag_text.append(f"<i>Debug:</i> {line.strip()}")
        # Parse error info
        elif '[ERR]' in line:
            self.diag_text.append(f"<font color='red'><b>Error:</b> {line.strip()}</font>")
    
    def on_modem_error(self, msg):
        self.diag_text.append(f"<font color='red'>ERROR: {msg}</font>")
    
    def on_new_log(self, line):
        self.diag_text.append(line)
        # Scroll to bottom
        self.diag_text.verticalScrollBar().setValue(self.diag_text.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = OfdmGui()
    gui.show()
    sys.exit(app.exec_())
