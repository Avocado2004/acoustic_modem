"""
Desktop audio backend using sounddevice.
Supports macOS, Windows, and Linux.
"""

import numpy as np
import sounddevice as sd


class DesktopAudio:
    """
    Audio interface for desktop platforms using sounddevice.
    Provides start_stream(), play(), and stop() methods.
    """

    def __init__(self):
        self._stream = None
        self._play_stream = None
        self._callbacks = []

    def start_stream(self, callback, samplerate=48000, channels=1, blocksize=1024):
        """
        Start an audio input stream.

        Args:
            callback: function(indata, frames, time_info, status)
            samplerate: sample rate in Hz
            channels: number of channels (1=mono, 2=stereo)
            blocksize: number of frames per buffer
        """
        if self._stream is not None:
            self.stop()

        def wrapped_callback(indata, frames, time_info, status):
            if status:
                print(f"[DesktopAudio] Status: {status}")
            callback(indata, frames, time_info, status)

        self._stream = sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
            callback=wrapped_callback
        )
        self._stream.start()

    def play(self, data, samplerate=48000):
        """
        Play audio data.

        Args:
            data: numpy array of audio samples (float32 or float64 in [-1, 1])
            samplerate: sample rate in Hz
        """
        if self._play_stream is not None:
            self._play_stream.stop()
            self._play_stream.close()

        # Ensure data is 1D or 2D with channels last
        if data.ndim > 1:
            data = data[:, 0]

        self._play_stream = sd.OutputStream(
            samplerate=samplerate,
            channels=1,
            blocksize=1024
        )
        self._play_stream.start()

        # Write data to stream
        self._play_stream.write(data.astype(np.float32))

    def stop(self):
        """Stop all audio streams (input and output)."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"[DesktopAudio] Error stopping input stream: {e}")
            finally:
                self._stream = None

        if self._play_stream is not None:
            try:
                self._play_stream.stop()
                self._play_stream.close()
            except Exception as e:
                print(f"[DesktopAudio] Error stopping output stream: {e}")
            finally:
                self._play_stream = None

    def is_active(self):
        """Check if the input stream is currently active."""
        if self._stream is not None:
            return self._stream.active
        return False

    def is_playing(self):
        """Check if the output stream is currently playing."""
        if self._play_stream is not None:
            return self._play_stream.active
        return False
