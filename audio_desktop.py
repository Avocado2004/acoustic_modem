"""
Desktop audio backend using sounddevice.
Supports macOS, Windows, and Linux.
"""

import numpy as np
import sounddevice as sd
import time


class DesktopAudio:
    """
    Audio interface for desktop platforms using sounddevice.
    Provides start_stream(), play(), and stop() methods.
    """

    def __init__(self):
        self._stream = None
        self._play_stream = None
        self._callback = None

    def _wrapped_callback(self, indata, frames, time_info, status):
        """Internal callback wrapper that calls user callback."""
        if status:
            print(f"[DesktopAudio] Status: {status}")
        if self._callback is not None:
            self._callback(indata, frames, time_info, status)

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

        self._callback = callback

        self._stream = sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
            callback=self._wrapped_callback
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
            try:
                self._play_stream.stop()
                self._play_stream.close()
            except Exception as e:
                print(f"[DesktopAudio] Error stopping previous stream: {e}")
            finally:
                self._play_stream = None

        # Ensure data is 1D
        if data.ndim > 1:
            data = data[:, 0]

        # Print debug information
        print(f"[DesktopAudio] play() called with {len(data)} samples, samplerate={samplerate}")
        print(f"[DesktopAudio] Data stats: min={np.min(data):.6f}, max={np.max(data):.6f}, "
              f"RMS={np.sqrt(np.mean(data**2)):.6f}")

        # Check available audio devices
        try:
            devices = sd.query_devices()
            print(f"[DesktopAudio] Available devices:")
            for idx, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    print(f"  {idx}: {dev['name']} (outputs: {dev['max_output_channels']})")
        except Exception as e:
            print(f"[DesktopAudio] Could not query audio devices: {e}")

        try:
            # Create output stream
            self._play_stream = sd.OutputStream(
                samplerate=samplerate,
                channels=1,
                blocksize=1024
            )
            self._play_stream.start()
            print(f"[DesktopAudio] OutputStream started, writing {len(data)} samples...")

            # Write data to stream
            self._play_stream.write(data.astype(np.float32))
            print(f"[DesktopAudio] Data written to stream")

            # Wait for playback to complete
            # Calculate expected playback time
            playback_time = len(data) / samplerate
            print(f"[DesktopAudio] Waiting {playback_time:.3f} seconds for playback to complete...")

            # Use time-based waiting instead of relying on stream.active
            time.sleep(playback_time + 0.5)  # Add 0.5 second buffer

            print(f"[DesktopAudio] Playback completed")

        except Exception as e:
            print(f"[DesktopAudio] Error during playback: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            if self._play_stream is not None:
                try:
                    self._play_stream.stop()
                    self._play_stream.close()
                except Exception as e:
                    print(f"[DesktopAudio] Error closing stream: {e}")
                finally:
                    self._play_stream = None

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
