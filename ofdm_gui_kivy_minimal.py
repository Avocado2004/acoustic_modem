# Minimal OFDM GUI - for debugging BPSK crash
import os
import sys
import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.uix.button import Button

class OfdmRoot(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=6, padding=6, **kwargs)
        
        self.label = Label(text='Select modulation:', size_hint_y=0.2)
        self.add_widget(self.label)
        
        self.modulation_spinner = Spinner(
            text='QPSK',
            values=('QPSK', 'BPSK'),
            size_hint_y=0.2
        )
        # Correct binding: Kivy calls callback with (instance, value)
        self.modulation_spinner.bind(text=lambda instance, value: self._on_modulation_change(value))
        self.add_widget(self.modulation_spinner)
        
        # Add a button to test
        self.test_btn = Button(text='Test BPSK', size_hint_y=0.2)
        self.test_btn.bind(on_release=lambda x: self._test_bpsk())
        self.add_widget(self.test_btn)
        
        self.status_label = Label(text='Status: OK', size_hint_y=0.4)
        self.add_widget(self.status_label)
    
    def _on_modulation_change(self, text):
        """Handle modulation change - minimal version"""
        try:
            # Log to file
            with open('/tmp/bpsk_debug.log', 'a') as f:
                import time
                f.write(f"{time.strftime('%H:%M:%S')} - Called with: {text}\n")
            
            self.label.text = f'Selected: {text}'
            
            # Try to import and set modulation
            try:
                import test_modem_simple as modem
                modem.MODULATION = text
                if text == 'BPSK':
                    modem.BITS_PER_SYMBOL = 1
                else:
                    modem.BITS_PER_SYMBOL = 2
                
                with open('/tmp/bpsk_debug.log', 'a') as f:
                    f.write(f"{time.strftime('%H:%M:%S')} - Set BITS_PER_SYMBOL={modem.BITS_PER_SYMBOL}\n")
                
                self.status_label.text = f'OK: BITS_PER_SYMBOL={modem.BITS_PER_SYMBOL}'
            except Exception as e:
                with open('/tmp/bpsk_debug.log', 'a') as f:
                    import traceback
                    f.write(f"{time.strftime('%H:%M:%S')} - ERROR: {e}\n")
                    f.write(traceback.format_exc() + '\n')
                self.status_label.text = f'Error: {e}'
                
        except Exception as e:
            # Last resort - write to file
            with open('/tmp/bpsk_crash.log', 'w') as f:
                f.write(f"CRASH: {e}\n")
    
    def _test_bpsk(self):
        """Test BPSK directly"""
        self.modulation_spinner.text = 'BPSK'

class OfdmApp(App):
    def build(self):
        from kivy.core.window import Window
        Window.size = (900, 600)
        return OfdmRoot()

if __name__ == "__main__":
    OfdmApp().run()
