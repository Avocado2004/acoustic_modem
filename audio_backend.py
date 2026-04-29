"""
Audio backend selector for cross-platform audio support.
Provides a unified interface for desktop (sounddevice), Android (pyjnius), and iOS (pyobjc).

Interface methods:
- start_stream(callback, samplerate, channels, blocksize): Start audio input stream
- play(data, samplerate): Play audio data
- stop(): Stop all audio streams

Usage:
    from audio_backend import get_audio_backend
    audio = get_audio_backend()
    audio.start_stream(callback, samplerate=48000, channels=1, blocksize=1024)
    audio.play(data, samplerate=48000)
    audio.stop()
"""

import sys
import numpy as np

# Platform detection
IS_ANDROID = sys.platform == 'android' or 'android' in sys.platform.lower()
IS_IOS = sys.platform == 'darwin' and hasattr(sys, 'implementation') and \
         getattr(sys.implementation, 'name', '') != 'cpython'  # Rough iOS detection
IS_DESKTOP = not IS_ANDROID and not IS_IOS

# Backend instances
_backend_instance = None


def get_audio_backend(backend_type=None):
    """
    Get appropriate audio backend for current platform.

    Args:
        backend_type: Force a specific backend ('desktop', 'android', 'ios')
                     If None, auto-detect based on platform.

    Returns:
        Audio backend instance with start_stream(), play(), stop() methods.
    """
    global _backend_instance

    if _backend_instance is not None and backend_type is None:
        return _backend_instance

    # Determine backend type
    if backend_type is None:
        if IS_ANDROID:
            backend_type = 'android'
        elif IS_IOS:
            backend_type = 'ios'
        else:
            backend_type = 'desktop'

    # Create backend instance
    if backend_type == 'android':
        try:
            from audio_android import AndroidAudio
            _backend_instance = AndroidAudio()
            print("[AudioBackend] Using Android audio backend")
        except ImportError as e:
            print(f"[AudioBackend] Failed to import Android audio: {e}")
            print("[AudioBackend] Falling back to desktop backend")
            from audio_desktop import DesktopAudio
            _backend_instance = DesktopAudio()
    elif backend_type == 'ios':
        try:
            from audio_ios import IOSAudio
            _backend_instance = IOSAudio()
            print("[AudioBackend] Using iOS audio backend")
        except ImportError as e:
            print(f"[AudioBackend] Failed to import iOS audio: {e}")
            print("[AudioBackend] Falling back to desktop backend")
            from audio_desktop import DesktopAudio
            _backend_instance = DesktopAudio()
    else:  # desktop
        try:
            from audio_desktop import DesktopAudio
            _backend_instance = DesktopAudio()
            print("[AudioBackend] Using desktop audio backend (sounddevice)")
        except ImportError as e:
            print(f"[AudioBackend] Failed to import desktop audio: {e}")
            _backend_instance = None

    return _backend_instance


class AudioBackendWrapper:
    """
    Wrapper class for backward compatibility with old audio_backend.py interface.
    Provides: play_audio(arr, fs), stop_audio(), is_playing()
    """

    def __init__(self):
        self._backend = get_audio_backend()
        self._playing = False

    def play_audio(self, arr, fs):
        """
        Play audio - interface compatible with ofdm_gui_kivy.py

        Args:
            arr: numpy array of audio samples
            fs: sample rate in Hz
        """
        if self._backend is None:
            print("[AudioBackendWrapper] No audio backend available")
            return

        # Resample if needed
        if fs != 48000:
            try:
                import scipy.signal as sig
                from scipy import interpolate
                duration = len(arr) / fs
                new_length = int(duration * 48000)
                if new_length > 0:
                    old_time = np.linspace(0, duration, len(arr))
                    new_time = np.linspace(0, duration, new_length)
                    if arr.ndim > 1:
                        arr = arr[:, 0]
                    f = interpolate.interp1d(old_time, arr, kind='linear', fill_value='extrapolate')
                    arr = f(new_time)
            except ImportError:
                print("[AudioBackendWrapper] scipy not available for resampling")

        # Normalize
        if arr.dtype != np.int16:
            max_val = np.max(np.abs(arr))
            if max_val > 0:
                arr = arr / max_val * 0.98

        self._backend.play(arr, samplerate=fs)
        self._playing = True

    def stop_audio(self):
        """Stop audio playback"""
        if self._backend is not None:
            self._backend.stop()
        self._playing = False

    def is_playing(self):
        """Check if audio is playing"""
        if self._backend is not None:
            return self._backend.is_playing()
        return self._playing


# Global wrapper instance for backward compatibility
_wrapper_instance = None


def get_backend():
    """
    Get backend wrapper for backward compatibility.
    Returns AudioBackendWrapper instance.
    """
    global _wrapper_instance
    if _wrapper_instance is None:
        _wrapper_instance = AudioBackendWrapper()
    return _wrapper_instance


def play_audio(arr, fs):
    """Play audio - interface compatible with ofdm_gui_kivy.py"""
    backend = get_backend()
    backend.play_audio(arr, fs)


def stop_audio():
    """Stop audio playback"""
    backend = get_backend()
    backend.stop_audio()


def is_playing():
    """Check if audio is playing"""
    backend = get_backend()
    return backend.is_playing()


# Convenience function to get the raw backend with start_stream support
def get_audio():
    """
    Get the raw audio backend with full interface (start_stream, play, stop).
    This is the preferred method for new code.

    Returns:
        Audio backend instance or None if not available.
    """
    return get_audio_backend()


# Test function
if __name__ == '__main__':
    print(f"Platform: {sys.platform}")
    print(f"IS_ANDROID: {IS_ANDROID}")
    print(f"IS_IOS: {IS_IOS}")
    print(f"IS_DESKTOP: {IS_DESKTOP}")

    audio = get_audio_backend()
    if audio is not None:
        print(f"Audio backend: {audio.__class__.__name__}")
        print("Testing play method...")
        test_data = np.random.randn(48000).astype(np.float32) * 0.1
        audio.play(test_data, samplerate=48000)
        import time
        time.sleep(1)
        audio.stop()
        print("Test complete.")
    else:
        print("No audio backend available!")
