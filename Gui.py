#!/usr/bin/env python3
# ofdm_gui_pyqt_vertical_full.py
# Vertical PyQt5 GUI for OFDM modem (updated)
# - redirects modem stdout/stderr into GUI
# - fills "Received" text area when modem prints [RX TEXT]
# - File Mode: use "Save received file" to save the last received file (copies from modem output file if available)
# - Text Mode: shows "Copy received text" button instead of Save received file
#
# Place next to your modem module (default name below). Requires: PyQt5, numpy
# Optional: audio_backend for playback (replaces sounddevice)

import sys, os
print("[DBG] sys.executable:", sys.executable)
print("[DBG] cwd:", os.getcwd())
print("[DBG] sys.path:")
for p in sys.path:
    print("  ", p)

import os
import sys
import threading
import importlib
import json
import tempfile
import shutil
from datetime import datetime
from functools import partial

from PyQt5 import QtCore, QtWidgets
import numpy as np

# Use audio_backend instead of direct sounddevice import
try:
    from audio_backend import get_audio_backend, play_audio as backend_play_audio, stop_audio as backend_stop_audio
    _audio_backend = get_audio_backend()
    print("[GUI] Audio backend loaded successfully")
except Exception as e:
    print(f"[GUI] Warning: Failed to import audio_backend: {e}")
    _audio_backend = None
    backend_play_audio = None
    backend_stop_audio = None

# name of modem module file (without .py) — change if needed
MODEM_MODULE_NAME = "test_modem_simple"

# try import modem module
_modem = None
_import_err = None
try:
    _modem = importlib.import_module(MODEM_MODULE_NAME)
except Exception:
    import traceback
    _import_err = traceback.format_exc()
    _modem = None

# safe getattr helper
def gattr(obj, name, default=None):
    return getattr(obj, name) if obj is not None and hasattr(obj, name) else default

# Logger signal
class Logger(QtCore.QObject):
    new = QtCore.pyqtSignal(str)

logger = Logger()

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    logger.new.emit(f"[{t}] {msg}")

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

# Build TX audio using modem helpers (unchanged logic)
def build_tx_audio_from_bytes(payload_bytes: bytes, mode_flag: bytes = b'F', filename_bytes: bytes = b'', settings=None):
    if _modem is None:
        raise RuntimeError("Modem module not available")
    settings = settings or {}
    fs = int(gattr(_modem, "fs", 48000))
    Nfft = int(settings.get("Nfft", gattr(_modem, "Nfft", 512)))
    Ncp = int(settings.get("Ncp", gattr(_modem, "Ncp", 128)))
    DEFAULT_PACKET_BLOCKS = int(settings.get("DEFAULT_PACKET_BLOCKS", gattr(_modem, "DEFAULT_PACKET_BLOCKS", 75)))
    GAP_OFDM_SYMBOLS = int(settings.get("GAP_OFDM_SYMBOLS", gattr(_modem, "GAP_OFDM_SYMBOLS", 2)))
    PREROLL_SEC = float(settings.get("PREROLL_SEC", 0.25))
    POSTROLL_SEC = float(settings.get("POSTROLL_SEC", 0.50))
    NOISE_LEVEL = float(settings.get("NOISE_LEVEL", 0.0))
    NOISE_SEED = int(settings.get("NOISE_SEED", 20231107))

    build_header = gattr(_modem, "build_header", None)
    bytes_to_ofdm_blocks_bytes = gattr(_modem, "bytes_to_ofdm_blocks_bytes", None)
    build_preamble = gattr(_modem, "build_preamble", None)
    apply_softclip = gattr(_modem, "apply_softclip_to_target_crest", None)
    make_packet_header_bytes = gattr(_modem, "make_packet_header_bytes", None)

    if build_header is None or bytes_to_ofdm_blocks_bytes is None or build_preamble is None:
        raise RuntimeError("Required modem helpers missing (build_header/bytes_to_ofdm_blocks_bytes/build_preamble)")

    # CRC32
    try:
        crc_val = _modem.zlib.crc32(payload_bytes) & 0xFFFFFFFF
    except Exception:
        import zlib
        crc_val = zlib.crc32(payload_bytes) & 0xFFFFFFFF

    tx_header = build_header(b'F' if mode_flag == b'F' else b'T', len(payload_bytes),
                             filename_bytes=filename_bytes, packet_no=0, version=0,
                             packet_blocks=DEFAULT_PACKET_BLOCKS, crc32=crc_val)

    remaining = payload_bytes
    packet_no = 0
    samples_list = []
    SYMBOL_LEN = Nfft + Ncp

    while True:
        is_first = (packet_no == 0)
        if is_first:
            header_bytes = tx_header + tx_header + tx_header
        else:
            if make_packet_header_bytes is not None:
                header_bytes = make_packet_header_bytes(packet_no, 0x03 if mode_flag == b'F' else 0x02)
            else:
                header_bytes = bytes([
                    0x00 | (0x03 if mode_flag == b'F' else 0x02),
                    (packet_no >> 16) & 0xFF,
                    (packet_no >> 8) & 0xFF,
                    packet_no & 0xFF
                ])

        allowed_blocks = DEFAULT_PACKET_BLOCKS
        lo = 0
        hi = len(remaining)
        best_mid = 0
        best_td = None
        while lo <= hi:
            mid = (lo + hi) // 2
            test_bytes = header_bytes + remaining[:mid]
            try:
                td_local, nblocks_local = bytes_to_ofdm_blocks_bytes(test_bytes)
            except Exception:
                nblocks_local = allowed_blocks + 1
            if nblocks_local <= allowed_blocks:
                best_mid = mid
                best_td = td_local
                lo = mid + 1
            else:
                hi = mid - 1

        if best_td is None:
            td_local, nblocks_local = bytes_to_ofdm_blocks_bytes(header_bytes)
            best_td = td_local

        preamble_td = build_preamble()
        packet_samples = np.concatenate((preamble_td, best_td)) if best_td.size else preamble_td.copy()
        samples_list.append(packet_samples)

        remaining = remaining[best_mid:]
        packet_no += 1
        if len(remaining) == 0:
            break
        if packet_no > 10000:
            break

    tx_concat = np.concatenate(samples_list) if len(samples_list) > 0 else np.array([], dtype=float)
    preroll_len = int(PREROLL_SEC * fs)
    postroll_len = int(POSTROLL_SEC * fs)
    tx_out = np.concatenate((np.zeros(preroll_len, dtype=tx_concat.dtype), tx_concat, np.zeros(postroll_len, dtype=tx_concat.dtype)))
    info = {"samples": len(tx_out), "fs": fs, "packets": packet_no}
    return tx_out.astype(np.float64), info

# Playback using audio backend
def play_audio(arr, fs):
    if backend_play_audio is None:
        return "Playback not available (audio backend not loaded)"
    try:
        backend_play_audio(arr, fs)
        return None
    except Exception as e:
        return f"Playback error: {e}"

def stop_audio():
    if backend_stop_audio is not None:
        backend_stop_audio()

# GUI class
class OfdmGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        self.tx_audio = None
        self.tx_info = None
        self.selected_file = None
        self.last_rx_saved_path = None
        self.mode = 'Text'
        
        # Redirect stdout/stderr
        sys.stdout = StdRedirector()
        sys.stderr = sys.stdout
        
        logger.new.connect(self.on_new_log)
    
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
        
        transmit_btn = QtWidgets.QPushButton("Transmit")
        transmit_btn.clicked.connect(self.transmit)
        btn_layout.addWidget(transmit_btn)
        
        receive_btn = QtWidgets.QPushButton("Receive")
        receive_btn.clicked.connect(self.receive)
        btn_layout.addWidget(receive_btn)
        
        save_btn = QtWidgets.QPushButton("Save Received File")
        save_btn.clicked.connect(self.save_received)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def on_mode_change(self, text):
        self.mode = text
        self.select_file_btn.setEnabled(text == "File")
    
    def select_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File")
        if fname:
            self.selected_file = fname
            self.input_text.setText(fname)
    
    def show_settings(self):
        # Simplified settings dialog
        QtWidgets.QMessageBox.information(self, "Settings", "Settings dialog not fully implemented")
    
    def transmit(self):
        if _modem is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Modem module not loaded")
            return
        
        if self.mode == "Text":
            text = self.input_text.toPlainText()
            if not text:
                QtWidgets.QMessageBox.warning(self, "Warning", "No text to transmit")
                return
            payload = text.encode('utf-8')
            filename_bytes = b''
        else:
            if not self.selected_file:
                QtWidgets.QMessageBox.warning(self, "Warning", "No file selected")
                return
            try:
                with open(self.selected_file, 'rb') as f:
                    payload = f.read()
                filename_bytes = os.path.basename(self.selected_file).encode('utf-8')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read file: {e}")
                return
        
        try:
            self.tx_audio, self.tx_info = build_tx_audio_from_bytes(payload, b'F', filename_bytes)
            log(f"TX audio prepared: {self.tx_info}")
            
            result = play_audio(self.tx_audio, int(gattr(_modem, "fs", 48000)))
            if result:
                log(f"Playback warning: {result}")
            else:
                log("Transmission started")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Transmission failed: {e}")
    
    def receive(self):
        log("Starting reception...")
        self.received_text.clear()
        # Simplified - actual reception would be more complex
        log("Reception not fully implemented in GUI - use test_modem_simple.py for now")
    
    def save_received(self):
        if not self.last_rx_saved_path or not os.path.exists(self.last_rx_saved_path):
            QtWidgets.QMessageBox.warning(self, "Warning", "No received file to save")
            return
        
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Received File", 
                                                        os.path.basename(self.last_rx_saved_path))
        if fname:
            try:
                shutil.copy2(self.last_rx_saved_path, fname)
                log(f"File saved to: {fname}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
    
    def on_new_log(self, line):
        self.diag_text.append(line)
        # Scroll to bottom
        self.diag_text.verticalScrollBar().setValue(self.diag_text.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = OfdmGui()
    gui.show()
    sys.exit(app.exec_())
