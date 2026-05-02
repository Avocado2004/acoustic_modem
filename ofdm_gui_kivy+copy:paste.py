# ofdm_gui_kivy.py - rewritten to use test_modem_simple.py as subprocess
import os
import sys
import threading
import subprocess
from datetime import datetime
from functools import partial

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import ObjectProperty, StringProperty
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

def fmt_log_line(msg):
    t = datetime.now().strftime("%H:%M:%S")
    return f"[{t}] {msg}\n"

class OfdmRoot(BoxLayout):
    diag_text = ObjectProperty(None)
    received_text = ObjectProperty(None)
    input_text = ObjectProperty(None)
    mode_spinner = ObjectProperty(None)
    btn_select_file = ObjectProperty(None)
    
    selected_file = None
    rx_file = None
    received_data = ""
    last_rx_saved_path = None
    
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=6, padding=6, **kwargs)
        
        self._build_top()
        self._build_input_area()
        self._build_controls()
        self._build_diag_recv()
        
        # Redirect stdout/stderr
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        
        Clock.schedule_once(lambda dt: self._mode_changed(self.mode_spinner.text), 0)
    
    def _build_top(self):
        top = BoxLayout(orientation='horizontal', size_hint_y=None, height=36, spacing=6)
        self.mode_spinner = Spinner(text='Text', values=('Text', 'File'), size_hint_x=1/3)
        self.mode_spinner.bind(text=lambda s, val: self._mode_changed(val))
        top.add_widget(self.mode_spinner)
        
        self.btn_select_file = Button(text='Select File', size_hint_x=1/3)
        self.btn_select_file.bind(on_release=self.open_filechooser)
        top.add_widget(self.btn_select_file)
        
        btn_settings = Button(text='Settings', size_hint_x=1/3)
        btn_settings.bind(on_release=lambda b: self.show_settings())
        top.add_widget(btn_settings)
        
        self.add_widget(top)
    
    def _build_input_area(self):
        self.input_label = Label(text='Input', size_hint_y=None, height=24)
        self.add_widget(self.input_label)
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
        btn_transmit = Button(text='Transmit', size_hint_x=1/3)
        btn_transmit.bind(on_release=lambda b: self.on_transmit())
        ctr.add_widget(btn_transmit)
        
        btn_receive = Button(text='Receive', size_hint_x=1/3)
        btn_receive.bind(on_release=lambda b: self.on_receive())
        ctr.add_widget(btn_receive)
        
        self.add_widget(ctr)
    
    def _build_diag_recv(self):
        self.add_widget(Label(text='Diagnostics', size_hint_y=None, height=24))
        self.diag_text = TextInput(text='', readonly=True, multiline=True, write_tab=True, size_hint_y=0.28)
        self.add_widget(self.diag_text)
        
        self.add_widget(Label(text='Received', size_hint_y=None, height=24))
        self.received_text = TextInput(text='', readonly=False, multiline=True, write_tab=True, size_hint_y=0.36)
        self.add_widget(self.received_text)
        
        bottom = BoxLayout(orientation='horizontal', size_hint_y=None, height=48, spacing=6)
        btn_save_received = Button(text='Save Received', size_hint_x=1/3)
        btn_save_received.bind(on_release=lambda b: self.on_save_received())
        bottom.add_widget(btn_save_received)
        
        btn_copy_received = Button(text='Copy', size_hint_x=1/3)
        btn_copy_received.bind(on_release=lambda b: self.on_copy_received())
        bottom.add_widget(btn_copy_received)
        
        self.add_widget(bottom)
    
    def write(self, txt):
        """Redirect stdout/stderr to diag_text"""
        if not txt:
            return
        Clock.schedule_once(lambda dt, t=txt: self.append_diag(t), 0)
    
    def flush(self):
        pass
    
    def append_diag(self, text):
        cur = self.diag_text.text
        self.diag_text.text = cur + text
        # Scroll to bottom
        self.diag_text.cursor = (0, len(self.diag_text.text.splitlines()))
    
    def append_received(self, txt):
        cur = self.received_text.text
        if cur:
            self.received_text.text = cur + "\n" + txt
        else:
            self.received_text.text = txt
    
    def _mode_changed(self, val):
        if val == "Text":
            self.input_text.readonly = False
            self.input_label.text = "Input (text)"
            self.btn_select_file.disabled = True
        else:
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
        
        name_row = BoxLayout(size_hint_y=None, height=40, spacing=6)
        name_row.add_widget(Label(text='Filename', size_hint_x=None, width=100))
        name_input = TextInput(text='', multiline=False, size_hint_x=1)
        name_row.add_widget(name_input)
        content.add_widget(name_row)
        
        btns = BoxLayout(size_hint_y=None, height=44, spacing=6)
        ok = Button(text='OK')
        cancel = Button(text='Cancel')
        ok.size_hint_x = 0.5
        cancel.size_hint_x = 0.5
        btns.add_widget(ok)
        btns.add_widget(cancel)
        content.add_widget(btns)
        
        popup = Popup(title='Select file', content=content, size_hint=(0.9, 0.9))
        
        def _ok(b):
            selection = fc.selection
            if selection:
                self.selected_file = selection[0]
                self.input_text.text = self.selected_file
                self.append_diag(fmt_log_line(f"Selected file: {self.selected_file}"))
            else:
                folder = fc.path
                typed = name_input.text.strip()
                if typed:
                    possible = os.path.join(folder, typed)
                    if os.path.exists(possible):
                        self.selected_file = possible
                        self.input_text.text = possible
                        self.append_diag(fmt_log_line(f"Selected file: {self.selected_file}"))
            popup.dismiss()
        
        ok.bind(on_release=_ok)
        cancel.bind(on_release=lambda b: popup.dismiss())
        popup.open()
    
    def show_settings(self):
        popup = Popup(title='Settings', content=Label(text='Settings dialog not fully implemented'), size_hint=(0.6, 0.4))
        popup.open()
    
    def on_transmit(self):
        mode = 'T'  # Transmit
        
        if self.mode_spinner.text == "Text":
            text = self.input_text.text
            if not text:
                self.append_diag(fmt_log_line("Warning: No text to transmit"))
                return
            submode = 'T'  # Text mode
            self.append_diag(fmt_log_line(f"Starting text transmission..."))
            threading.Thread(target=self._run_modem, args=(mode, submode, text, None, None, None), daemon=True).start()
        else:
            if not self.selected_file:
                self.append_diag(fmt_log_line("Warning: No file selected"))
                return
            submode = 'F'  # File mode
            self.append_diag(fmt_log_line(f"Starting file transmission: {self.selected_file}"))
            threading.Thread(target=self._run_modem, args=(mode, submode, None, self.selected_file, None, None), daemon=True).start()
    
    def on_receive(self):
        self.received_text.text = ""  # Clear received text
        self.append_diag(fmt_log_line("Starting reception..."))
        
        # Ask user for source
        content = BoxLayout(orientation='vertical', spacing=6)
        content.add_widget(Label(text='Receive from:'))
        
        btn_mic = Button(text='Microphone')
        btn_file = Button(text='WAV File')
        content.add_widget(btn_mic)
        content.add_widget(btn_file)
        
        popup = Popup(title='Select Source', content=content, size_hint=(0.6, 0.4))
        
        def _mic(b):
            popup.dismiss()
            self.append_diag(fmt_log_line("Receiving from microphone..."))
            threading.Thread(target=self._run_modem, args=('R', None, None, None, 'mic', None), daemon=True).start()
        
        def _file(b):
            popup.dismiss()
            # Ask for WAV file
            content2 = BoxLayout(orientation='vertical')
            fc = FileChooserListView(path='.', filters=['*.wav'], size_hint=(1, 1))
            content2.add_widget(fc)
            
            btns = BoxLayout(size_hint_y=None, height=44, spacing=6)
            ok = Button(text='OK')
            cancel = Button(text='Cancel')
            ok.size_hint_x = 0.5
            cancel.size_hint_x = 0.5
            btns.add_widget(ok)
            btns.add_widget(cancel)
            content2.add_widget(btns)
            
            popup2 = Popup(title='Select WAV File', content=content2, size_hint=(0.9, 0.9))
            
            def _ok2(b):
                if fc.selection:
                    wav_path = fc.selection[0]
                    popup2.dismiss()
                    self.append_diag(fmt_log_line(f"Receiving from file: {wav_path}"))
                    threading.Thread(target=self._run_modem, args=('R', None, None, None, 'file', wav_path), daemon=True).start()
            
            ok.bind(on_release=_ok2)
            cancel.bind(on_release=lambda b: popup2.dismiss())
            popup2.open()
        
        btn_mic.bind(on_release=_mic)
        btn_file.bind(on_release=_file)
        popup.open()
    
    def _run_modem(self, mode, submode=None, text=None, filepath=None, rx_source=None, rx_filepath=None):
        """Run test_modem_simple.py as subprocess with automated input"""
        try:
            cmd = [sys.executable, "test_modem_simple.py"]
            
            # Prepare input sequence
            input_lines = []
            if mode == 'T':
                input_lines.append('T')  # Transmit mode
                if submode == 'F':
                    input_lines.append('F')  # File mode
                    input_lines.append(filepath)  # File path
                else:
                    input_lines.append('T')  # Text mode
                    input_lines.append(text)  # Text to transmit
            else:
                input_lines.append('R')  # Receive mode
                input_lines.append(rx_source)  # 'file' or 'mic'
                if rx_source == 'file' and rx_filepath:
                    input_lines.append(rx_filepath)  # WAV file path
            
            input_str = "\n".join(input_lines) + "\n"
            
            self.append_diag(fmt_log_line(f"Starting test_modem_simple.py with mode={mode}"))
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)) or '.',
                text=True,
                bufsize=1
            )
            
            # Send input
            if process.stdin:
                process.stdin.write(input_str)
                process.stdin.flush()
                process.stdin.close()
            
            # Read output
            received_text_lines = []
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        # Output to diag
                        Clock.schedule_once(lambda dt, l=line: self.append_diag(l), 0)
                        
                        # Check for received text
                        if '[RX TEXT]' in line:
                            text_part = line.split('[RX TEXT]', 1)[-1].strip()
                            received_text_lines.append(text_part)
                            Clock.schedule_once(lambda dt, t=text_part: self.append_received(t), 0)
                        elif '[RX FILE]' in line or 'Saved:' in line or 'Сохранён файл:' in line:
                            Clock.schedule_once(lambda dt, l=line: self.append_received(l.strip()), 0)
            
            process.wait()
            
            if received_text_lines:
                self.received_data = "\n".join(received_text_lines)
            
            self.append_diag(fmt_log_line(f"Process finished with return code {process.returncode}"))
            
        except Exception as e:
            self.append_diag(fmt_log_line(f"Error running modem: {e}"))
    
    def on_save_received(self):
        if not self.received_text.text:
            self.append_diag(fmt_log_line("Warning: No received data to save"))
            return
        
        content = BoxLayout(orientation='vertical', spacing=6)
        name_row = BoxLayout(size_hint_y=None, height=40, spacing=6)
        name_row.add_widget(Label(text='Filename:', size_hint_x=None, width=100))
        name_input = TextInput(text='received_data.txt', multiline=False, size_hint_x=1)
        name_row.add_widget(name_input)
        content.add_widget(name_row)
        
        btns = BoxLayout(size_hint_y=None, height=44, spacing=6)
        save_btn = Button(text='Save')
        cancel_btn = Button(text='Cancel')
        save_btn.size_hint_x = 0.5
        cancel_btn.size_hint_x = 0.5
        btns.add_widget(save_btn)
        btns.add_widget(cancel_btn)
        content.add_widget(btns)
        
        popup = Popup(title='Save Received Data', content=content, size_hint=(0.7, 0.4))
        
        def _save(b):
            fname = name_input.text.strip()
            if not fname:
                return
            try:
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(self.received_text.text)
                self.append_diag(fmt_log_line(f"Data saved to: {fname}"))
            except Exception as e:
                self.append_diag(fmt_log_line(f"Failed to save file: {e}"))
            popup.dismiss()
        
        save_btn.bind(on_release=_save)
        cancel_btn.bind(on_release=lambda b: popup.dismiss())
        popup.open()
    
    def on_copy_received(self):
        if self.received_text.text:
            # Kivy doesn't have built-in clipboard access like PyQt
            # This is a simplified version
            self.append_diag(fmt_log_line("Copy not fully implemented in Kivy version"))

class OfdmApp(App):
    def build(self):
        return OfdmRoot()

if __name__ == '__main__':
    OfdmApp().run()
