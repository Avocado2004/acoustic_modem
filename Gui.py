#!/usr/bin/env python3
# ofdm_gui_pyqt_vertical_full.py
# Vertical PyQt5 GUI for OFDM modem (updated)
# - redirects modem stdout/stderr into GUI
# - fills "Received" text area when modem prints [RX TEXT]
# - File Mode: use "Save received file" to save the last received file (copies from modem output file if available)
# - Text Mode: shows "Copy received text" button instead of Save received file
#
# Place next to your modem module (default name below). Requires: PyQt5, numpy, scipy (for save wav), sounddevice optional.

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

# optional playback
try:
    import sounddevice as sd
except Exception:
    sd = None

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
    samples_packets = []
    rng = np.random.RandomState(NOISE_SEED)
    preamble_td = build_preamble()
    SYMBOL_LEN = Nfft + Ncp

    while True:
        is_first = (packet_no == 0)
        header_bytes = tx_header if is_first else (make_packet_header_bytes(packet_no, 0) if make_packet_header_bytes is not None else b'\x00'*4)
        lo, hi = 0, len(remaining)
        best_sz = 0
        best_td = None
        while lo <= hi:
            mid = (lo + hi) // 2
            test_bytes = header_bytes + remaining[:mid]
            td_local, nblocks_local = bytes_to_ofdm_blocks_bytes(test_bytes)
            if nblocks_local <= DEFAULT_PACKET_BLOCKS:
                best_sz = mid
                best_td = td_local
                lo = mid + 1
            else:
                hi = mid - 1
        if best_td is None:
            td_local, nblocks_local = bytes_to_ofdm_blocks_bytes(header_bytes)
            if nblocks_local > DEFAULT_PACKET_BLOCKS:
                td_local = td_local[:DEFAULT_PACKET_BLOCKS * SYMBOL_LEN]
            best_td = td_local
            best_sz = 0

        nblocks_now = len(best_td) // SYMBOL_LEN if SYMBOL_LEN > 0 else 0
        if nblocks_now < DEFAULT_PACKET_BLOCKS:
            needed_blocks = DEFAULT_PACKET_BLOCKS - nblocks_now
            filler_bytes_len = needed_blocks * gattr(_modem, "RS_DATA_BYTES", 8)
            td_filler, nblk_f = bytes_to_ofdm_blocks_bytes(b'\x00' * filler_bytes_len)
            if nblk_f >= needed_blocks and len(td_filler) >= needed_blocks * SYMBOL_LEN:
                best_td = np.concatenate((best_td, td_filler[:needed_blocks * SYMBOL_LEN]))
            else:
                best_td = np.concatenate((best_td, np.zeros(needed_blocks * SYMBOL_LEN, dtype=best_td.dtype)))
        elif nblocks_now > DEFAULT_PACKET_BLOCKS:
            best_td = best_td[:DEFAULT_PACKET_BLOCKS * SYMBOL_LEN]

        packet_samples = np.concatenate((preamble_td, best_td)) if len(best_td) > 0 else preamble_td.copy()

        gap_samples = GAP_OFDM_SYMBOLS * SYMBOL_LEN
        if NOISE_LEVEL and NOISE_LEVEL > 0.0:
            w = rng.normal(loc=0.0, scale=1.0, size=gap_samples).astype(np.float64)
            cur_rms = np.sqrt(np.mean(w**2)) if w.size > 0 else 1.0
            pkt_peak = np.max(np.abs(packet_samples)) if np.max(np.abs(packet_samples)) > 0 else 1.0
            noise_rms = NOISE_LEVEL * pkt_peak
            w = w * (noise_rms / cur_rms)
            gap_noise = w.astype(packet_samples.dtype)
        else:
            gap_noise = np.zeros(gap_samples, dtype=packet_samples.dtype)

        packet_samples = np.concatenate((packet_samples, gap_noise))
        samples_packets.append(packet_samples)

        remaining = remaining[best_sz:]
        packet_no += 1
        if len(remaining) == 0 or packet_no > 1000000:
            break

    tx_packets_concat = np.concatenate(samples_packets) if len(samples_packets) > 0 else np.array([], dtype=float)

    preroll_len = int(PREROLL_SEC * fs)
    postroll_len = int(POSTROLL_SEC * fs)
    preroll = np.zeros(preroll_len, dtype=tx_packets_concat.dtype)
    postroll = np.zeros(postroll_len, dtype=tx_packets_concat.dtype)
    tx_out = np.concatenate((preroll, tx_packets_concat, postroll))

    if apply_softclip is not None:
        try:
            tx_clipped, _ = apply_softclip(tx_out, target_db=gattr(_modem, "TARGET_CREST_DB", 6))
            tx_out = tx_clipped
        except Exception:
            pass

    max_abs = np.max(np.abs(tx_out)) if tx_out.size else 0.0
    tx_norm = tx_out.astype(np.float64) if max_abs == 0 else (tx_out.astype(np.float64) / float(max_abs))

    info = {"fs": fs, "samples": len(tx_norm), "crc": crc_val, "packets": packet_no, "preamble_len": len(preamble_td)}
    return tx_norm, info

# save wav
def save_wav(path, arr, fs):
    from scipy.io import wavfile
    max_abs = np.max(np.abs(arr)) if arr.size else 0.0
    if max_abs == 0:
        data_int16 = (arr * 0).astype(np.int16)
    else:
        data_int16 = (arr / max_abs * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(path, fs, data_int16)

# playback
def play_audio(arr, fs):
    if sd is None:
        log("Playback not available (sounddevice not installed)")
        return
    try:
        sd.stop()
    except Exception:
        pass
    def _play():
        try:
            sd.play(arr, fs)
            sd.wait()
        except Exception as e:
            log(f"Playback error: {e}")
    threading.Thread(target=_play, daemon=True).start()

# GUI
class OfdmGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OFDM Modem (vertical)")
        self.resize(640, 960)

        # default settings from module if available
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

        self.tx_audio = None
        self.tx_info = None
        self.selected_file = None

        # store last received file path reported by modem (if modem saved a file automatically)
        self._last_rx_saved_path = None

        self._build_ui()

        # wire logger
        logger.new.connect(self._append_log)
        logger.new.connect(self._process_log_line)

        # redirect global stdout/stderr to GUI logger so modem prints are captured
        sys.stdout = StdRedirector()
        sys.stderr = StdRedirector()

        if _import_err:
            log("[IMPORT-ERR] " + _import_err)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8,8,8,8)

        # top: mode + settings + select-file (visible only in File Mode)
        top_h = QtWidgets.QHBoxLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Text Mode", "File Mode"])
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        top_h.addWidget(self.mode_combo)

        self.btn_select_file = QtWidgets.QPushButton("Select file (File Mode)")
        self.btn_select_file.clicked.connect(self._select_file_dialog)
        self.btn_select_file.setVisible(False)
        top_h.addWidget(self.btn_select_file)

        btn_settings = QtWidgets.QPushButton("Settings")
        btn_settings.clicked.connect(self._open_settings)
        top_h.addWidget(btn_settings)

        layout.addLayout(top_h)

        # large input area
        self.input_label = QtWidgets.QLabel("Input")
        layout.addWidget(self.input_label)
        self.input_text = QtWidgets.QPlainTextEdit()
        self.input_text.setPlaceholderText("Enter text here (or select a file in File Mode)")
        self.input_text.setMinimumHeight(220)
        layout.addWidget(self.input_text)

        # controls row
        controls_h = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Generate audio")
        self.btn_generate.clicked.connect(self.on_generate_audio)
        controls_h.addWidget(self.btn_generate)

        self.btn_save_audio = QtWidgets.QPushButton("Save audio file")
        self.btn_save_audio.clicked.connect(self.on_save_audio)
        controls_h.addWidget(self.btn_save_audio)

        self.btn_transmit = QtWidgets.QPushButton("Transmit")
        self.btn_transmit.clicked.connect(self.on_transmit)
        controls_h.addWidget(self.btn_transmit)
        layout.addLayout(controls_h)

        # diagnostics (small)
        diag_label = QtWidgets.QLabel("Diagnostics")
        layout.addWidget(diag_label)
        self.diag_text = QtWidgets.QPlainTextEdit()
        self.diag_text.setReadOnly(True)
        self.diag_text.setMaximumHeight(140)
        layout.addWidget(self.diag_text)

        # received area (large)
        rec_label = QtWidgets.QLabel("Received")
        layout.addWidget(rec_label)
        self.received_text = QtWidgets.QPlainTextEdit()
        self.received_text.setMinimumHeight(220)
        layout.addWidget(self.received_text)

        # bottom controls: Receive + Save/Copy (mode dependent)
        bottom_h = QtWidgets.QHBoxLayout()
        self.btn_receive = QtWidgets.QPushButton("Receive")
        self.btn_receive.clicked.connect(self.on_receive)
        bottom_h.addWidget(self.btn_receive)

        # Save received file (File Mode only)
        self.btn_save_received = QtWidgets.QPushButton("Save received file")
        self.btn_save_received.clicked.connect(self.on_save_received)
        bottom_h.addWidget(self.btn_save_received)

        # Copy received text (Text Mode only)
        self.btn_copy_received = QtWidgets.QPushButton("Copy received text")
        self.btn_copy_received.clicked.connect(self.on_copy_received)
        bottom_h.addWidget(self.btn_copy_received)

        layout.addLayout(bottom_h)

        # initial mode update
        self._mode_changed(0)

    def _append_log(self, text):
        # append to diagnostics area (ensure in GUI thread)
        self.diag_text.appendPlainText(text.rstrip())

    def _process_log_line(self, line):
        # Capture [RX TEXT] and [RX FILE] lines from modem output and update GUI state
        try:
            if "[RX TEXT]" in line:
                # everything after tag is the received text line
                idx = line.find("[RX TEXT]")
                txt = line[idx + len("[RX TEXT]"):].strip()
                # append to received_text safely
                def _append():
                    cur = self.received_text.toPlainText()
                    if cur:
                        cur = cur + "\n" + txt
                    else:
                        cur = txt
                    self.received_text.setPlainText(cur)
                QtCore.QTimer.singleShot(0, _append)

            # detect when modem saved a file and printed its path; we look for patterns like:
            # "[RX FILE] Saved: <path>" or similar messages from modem
            if "[RX FILE]" in line and "Saved" in line:
                # attempt to extract path after colon
                try:
                    after = line.split("Saved:",1)[1].strip()
                    path = after.split(" ")[0].strip()
                    if os.path.exists(path):
                        self._last_rx_saved_path = path
                        log(f"Detected saved received file: {path}")
                except Exception:
                    pass

            # also try generic "Saved" messages
            if "Saved:" in line and "rx_" in line:
                try:
                    after = line.split("Saved:",1)[1].strip()
                    path = after.split(" ")[0].strip()
                    if os.path.exists(path):
                        self._last_rx_saved_path = path
                        log(f"Detected saved received file: {path}")
                except Exception:
                    pass

        except Exception:
            pass

    def _mode_changed(self, idx):
        mode = self.mode_combo.currentText()
        if mode == "Text Mode":
            self.input_text.setReadOnly(False)
            self.input_label.setText("Input (text)")
            self.btn_select_file.setVisible(False)
            self.btn_save_received.setVisible(False)
            self.btn_copy_received.setVisible(True)
        else:
            self.input_text.setReadOnly(True)
            self.input_label.setText("Selected file")
            if getattr(self, "selected_file", None):
                self.input_text.setPlainText(self.selected_file)
            else:
                self.input_text.setPlainText("[File Mode] No file selected")
            self.btn_select_file.setVisible(True)
            self.btn_save_received.setVisible(True)
            self.btn_copy_received.setVisible(False)
        log(f"Switched to {mode}")

    def _open_settings(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Settings")
        form = QtWidgets.QFormLayout(dlg)
        edits = {}
        for k, v in self.settings.items():
            le = QtWidgets.QLineEdit(str(v))
            form.addRow(k, le)
            edits[k] = le
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            for k, le in edits.items():
                val = le.text().strip()
                if k == "PHASE_METHOD":
                    self.settings[k] = val
                else:
                    try:
                        self.settings[k] = int(val)
                    except Exception:
                        self.settings[k] = val
            log("Settings applied: " + json.dumps(self.settings))
        else:
            log("Settings cancelled")

    def _select_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file to send")
        if path:
            self.selected_file = path
            self.input_text.setPlainText(path)
            self.input_label.setText("Selected file")
            log(f"Selected file: {path}")

    def on_generate_audio(self):
        try:
            mode = self.mode_combo.currentText()
            if mode == "Text Mode":
                text = self.input_text.toPlainText()
                if not text:
                    QtWidgets.QMessageBox.warning(self, "Generate audio", "Enter text first")
                    return
                payload = text.encode("utf-8")
                filename_bytes = b''
            else:
                if not getattr(self, "selected_file", None):
                    QtWidgets.QMessageBox.warning(self, "Generate audio", "Select a file first")
                    return
                with open(self.selected_file, "rb") as f:
                    payload = f.read()
                filename_bytes = os.path.basename(self.selected_file).encode("utf-8")

            def _build():
                try:
                    tx, info = build_tx_audio_from_bytes(payload,
                                                         mode_flag=b'F' if mode != "Text Mode" else b'T',
                                                         filename_bytes=filename_bytes,
                                                         settings=self.settings)
                    self.tx_audio = tx
                    self.tx_info = info
                    log(f"Generated audio: samples={info['samples']} fs={info['fs']} packets={info['packets']}")
                except Exception as e:
                    log(f"Audio generation failed: {e}")

            threading.Thread(target=_build, daemon=True).start()
        except Exception as e:
            log(f"Generate audio error: {e}")

    def on_save_audio(self):
        if getattr(self, "tx_audio", None) is None:
            QtWidgets.QMessageBox.information(self, "Save audio", "No generated audio yet. Press Generate audio first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save WAV file", filter="WAV files (*.wav)")
        if not path:
            return
        fs = int(self.tx_info.get("fs", gattr(_modem, "fs", 48000))) if self.tx_info else int(gattr(_modem, "fs", 48000))
        try:
            save_wav(path, self.tx_audio, fs)
            log(f"Saved WAV to {path}")
        except Exception as e:
            log(f"Save WAV failed: {e}")

    def on_transmit(self):
        if getattr(self, "tx_audio", None) is None:
            self.on_generate_audio()
            def _wait_and_play():
                import time
                for _ in range(200):
                    if getattr(self, "tx_audio", None) is not None:
                        break
                    time.sleep(0.05)
                if getattr(self, "tx_audio", None) is None:
                    log("Transmit aborted: generation timed out")
                    return
                fs = int(self.tx_info.get("fs", gattr(_modem, "fs", 48000)))
                log("Playing generated audio...")
                play_audio(self.tx_audio, fs)
            threading.Thread(target=_wait_and_play, daemon=True).start()
        else:
            fs = int(self.tx_info.get("fs", gattr(_modem, "fs", 48000)))
            log("Playing existing generated audio...")
            play_audio(self.tx_audio, fs)

    def on_receive(self):
        if _modem is None:
            QtWidgets.QMessageBox.warning(self, "Receive", "Modem module not loaded")
            return

        def run_live():
            try:
                log("Preparing modem globals for live receive...")
                # apply settings to module globals
                try:
                    for k, v in self.settings.items():
                        if hasattr(_modem, k):
                            setattr(_modem, k, v)
                except Exception as e:
                    log(f"Failed to apply settings into modem globals: {e}")

                if not hasattr(_modem, "preamble_td") or _modem.preamble_td is None:
                    if hasattr(_modem, "build_preamble"):
                        try:
                            _modem.preamble_td = _modem.build_preamble()
                            log("preamble_td built")
                        except Exception as e:
                            log(f"build_preamble() failed: {e}")
                    else:
                        log("Warning: build_preamble() not found in modem module")

                if hasattr(_modem, "init_phases"):
                    try:
                        _modem.init_phases()
                        log("init_phases() called")
                    except Exception as e:
                        log(f"init_phases() failed: {e}")

                # create common globals to avoid NameError inside module
                if not hasattr(_modem, "rx"):
                    _modem.rx = np.array([], dtype=float)
                if not hasattr(_modem, "abs_corr"):
                    _modem.abs_corr = np.array([], dtype=float)
                _modem.post_sync = False
                _modem.rx_syms_list = []
                _modem.Hk_smooth_list = []

                log("Calling live_receive_and_process()")
                _modem.live_receive_and_process()
                log("live_receive_and_process() returned")
            except Exception as e:
                log(f"Live receive error: {e}")

        threading.Thread(target=run_live, daemon=True).start()

    def on_save_received(self):
        # File Mode save behavior:
        # - if modem printed and saved a file (we detected its path), offer to copy that file to user-chosen path
        # - else, offer to save contents of received_text as text
        if getattr(self, "mode_combo", None) and self.mode_combo.currentText() == "Text Mode":
            QtWidgets.QMessageBox.information(self, "Save", "Save received file is not available in Text Mode.")
            return

        # File Mode
        if self._last_rx_saved_path and os.path.exists(self._last_rx_saved_path):
            dest, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save received file as", os.path.basename(self._last_rx_saved_path))
            if dest:
                try:
                    shutil.copyfile(self._last_rx_saved_path, dest)
                    log(f"Copied received file to {dest}")
                except Exception as e:
                    log(f"Failed to copy received file: {e}")
        else:
            # fallback: save received_text contents as text file
            content = self.received_text.toPlainText()
            if not content:
                QtWidgets.QMessageBox.information(self, "Save", "Nothing received yet")
                return
            dest, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save received text as", filter="Text files (*.txt);;All files (*)")
            if dest:
                try:
                    with open(dest, "w", encoding="utf-8") as f:
                        f.write(content)
                    log(f"Saved received text to {dest}")
                except Exception as e:
                    log(f"Failed to save received text: {e}")

    def on_copy_received(self):
        # copy received_text to clipboard (Text Mode)
        content = self.received_text.toPlainText()
        if not content:
            QtWidgets.QMessageBox.information(self, "Copy", "No received text to copy")
            return
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(content)
        log("Copied received text to clipboard")

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = OfdmGui()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

