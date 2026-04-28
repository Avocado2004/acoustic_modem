import numpy as np
import lightweight_signal as signal
import sys
import importlib

# Platform detection
IS_ANDROID = sys.platform == 'android' or 'android' in sys.platform.lower()

# Try to import Android classes if on Android platform
if IS_ANDROID:
    try:
        from jnius import autoclass, cast
        from android import activity
        from android.permissions import Permission, request_permissions, check_permission
    except ImportError:
        IS_ANDROID = False

class AndroidAudioBackend:
    """Android audio backend using pyjnius"""
    
    def __init__(self):
        self.sample_rate = 48000
        self.channels = 1
        self.audio_format = 2  # ENCODING_PCM_16BIT = 2
        self.channel_config = 2  # CHANNEL_IN_MONO = 2
        self.stream_type = 3  # STREAM_MUSIC = 3
        
        self._audio_track = None
        self._audio_record = None
        self._is_playing = False
        self._is_recording = False
        
    def _init_audio_track(self, buffer_size):
        """Initialize AudioTrack for playback"""
        if not IS_ANDROID:
            return None
            
        AudioTrack = autoclass('android.media.AudioTrack')
        return AudioTrack(
            self.stream_type,  # streamType
            self.sample_rate,  # sampleRateInHz
            self.channel_config,  # channelConfig
            self.audio_format,  # audioFormat
            buffer_size,  # bufferSizeInBytes
            AudioTrack.MODE_STREAM  # mode
        )
        
    def _init_audio_record(self, buffer_size):
        """Initialize AudioRecord for recording"""
        if not IS_ANDROID:
            return None
            
        AudioRecord = autoclass('android.media.AudioRecord')
        return AudioRecord(
            AudioRecord.VOICE_RECOGNITION,  # audioSource
            self.sample_rate,  # sampleRateInHz
            self.channel_config,  # channelConfig
            self.audio_format,  # audioFormat
            buffer_size  # bufferSizeInBytes
        )
        
    def play(self, data: np.ndarray):
        """Play audio data through Android AudioTrack"""
        if not IS_ANDROID:
            print("Not on Android platform")
            return
            
        # Convert float32 to int16
        if data.dtype != np.int16:
            # Assume data is float32 in range [-1, 1]
            data_int16 = (data * 32767).astype(np.int16)
        else:
            data_int16 = data
            
        buffer_size = len(data_int16) * 2  # 2 bytes per int16
        
        if self._audio_track is None:
            self._audio_track = self._init_audio_track(buffer_size)
            if self._audio_track is None:
                return
            self._audio_track.play()
            self._is_playing = True
            
        # Convert to Java byte array
        # Create bytearray from int16 data
        byte_data = data_int16.tobytes()
        
        # Convert to Java byte array
        java_bytes = autoclass('java.lang.String')(byte_data.decode('latin-1')).getBytes('latin-1')
        
        # Write to AudioTrack
        self._audio_track.write(java_bytes, 0, len(java_bytes))
        
    def stop(self):
        """Stop audio playback"""
        if self._audio_track is not None:
            self._audio_track.stop()
            self._audio_track.release()
            self._audio_track = None
            self._is_playing = False
            
    def start_recording(self, duration_seconds: float = 5.0):
        """Start audio recording"""
        if not IS_ANDROID:
            print("Not on Android platform")
            return None
            
        buffer_size = int(self.sample_rate * duration_seconds * 2)  # 2 bytes per sample
        
        if self._audio_record is None:
            self._audio_record = self._init_audio_record(buffer_size)
            if self._audio_record is None:
                return None
            self._audio_record.startRecording()
            self._is_recording = True
            
        return buffer_size
        
    def stop_recording(self):
        """Stop recording and return recorded data"""
        if not IS_ANDROID or self._audio_record is None:
            return np.array([], dtype=np.int16)
            
        # Get recorded data
        buffer_size = self._audio_record.getRecordingState()
        
        if buffer_size > 0:
            # Read data from AudioRecord
            java_buffer = autoclass('java.nio.ByteBuffer').allocate(buffer_size)
            bytes_read = self._audio_record.read(java_buffer, buffer_size)
            
            if bytes_read > 0:
                # Convert to numpy array
                byte_data = bytes(java_buffer.array()[:bytes_read])
                data = np.frombuffer(byte_data, dtype=np.int16)
            else:
                data = np.array([], dtype=np.int16)
        else:
            data = np.array([], dtype=np.int16)
            
        self._audio_record.stop()
        self._audio_record.release()
        self._audio_record = None
        self._is_recording = False
        
        return data
        
    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        return self._is_playing
        
    def is_recording(self) -> bool:
        """Check if audio is currently recording"""
        return self._is_recording


class DesktopAudioBackend:
    """Desktop audio backend using sounddevice"""
    
    def __init__(self):
        self.sample_rate = 48000
        self._is_playing = False
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            self._sd = None
            
    def play(self, data: np.ndarray):
        """Play audio data through sounddevice"""
        if self._sd is None:
            print("sounddevice not available")
            return
            
        self._sd.stop()
        self._sd.play(data, self.sample_rate)
        self._is_playing = True
        
    def stop(self):
        """Stop audio playback"""
        if self._sd is not None:
            self._sd.stop()
        self._is_playing = False
            
    def start_recording(self, duration_seconds: float = 5.0):
        """Start audio recording - not implemented for desktop"""
        print("Desktop recording not implemented")
        return None
        
    def stop_recording(self):
        """Stop recording - not implemented for desktop"""
        print("Desktop recording not implemented")
        return np.array([], dtype=np.int16)
        
    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        return self._is_playing


# Global backend instance
_backend_instance = None
_is_playing = False

def get_backend():
    """Get appropriate audio backend for current platform"""
    global _backend_instance
    
    if _backend_instance is None:
        if IS_ANDROID:
            # Check and request permissions on Android
            try:
                from android.permissions import check_permission, Permission
                if not check_permission(Permission.RECORD_AUDIO):
                    print("Warning: RECORD_AUDIO permission not granted")
            except:
                pass
            _backend_instance = AndroidAudioBackend()
        else:
            _backend_instance = DesktopAudioBackend()
    
    return _backend_instance


def play_audio(arr, fs):
    """Play audio - interface compatible with ofdm_gui_kivy.py"""
    global _is_playing
    
    backend = get_backend()
    
    # Resample if needed
    if fs != backend.sample_rate:
        import scipy.signal as sig
        from scipy import interpolate
        duration = len(arr) / fs
        new_length = int(duration * backend.sample_rate)
        if new_length > 0:
            old_time = np.linspace(0, duration, len(arr))
            new_time = np.linspace(0, duration, new_length)
            if arr.ndim > 1:
                arr = arr[:, 0]
            f = interpolate.interp1d(old_time, arr, kind='linear', fill_value='extrapolate')
            arr = f(new_time)
    
    # Normalize
    if arr.dtype != np.int16:
        max_val = np.max(np.abs(arr))
        if max_val > 0:
            arr = arr / max_val * 0.98
    
    backend.play(arr)
    _is_playing = True
    

def stop_audio():
    """Stop audio playback"""
    global _is_playing
    
    backend = get_backend()
    backend.stop()
    _is_playing = False
    

def is_playing():
    """Check if audio is playing"""
    backend = get_backend()
    return backend.is_playing()
