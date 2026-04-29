"""
Android audio backend using pyjnius.
Uses AudioRecord for recording and AudioTrack for playback.
"""

import numpy as np
import threading
import time

try:
    from jnius import autoclass, cast
    from android.permissions import Permission, check_permission, request_permissions
except ImportError:
    autoclass = None
    cast = None


class AndroidAudio:
    """
    Audio interface for Android using pyjnius.
    Provides start_stream(), play(), and stop() methods.
    """

    # Audio constants
    ENCODING_PCM_16BIT = 2
    CHANNEL_IN_MONO = 2
    CHANNEL_OUT_MONO = 4
    STREAM_MUSIC = 3
    STREAM_VOICE_RECOGNITION = 1
    MODE_STREAM = 1

    def __init__(self):
        self._audio_record = None
        self._audio_track = None
        self._record_thread = None
        self._play_thread = None
        self._recording = False
        self._playing = False
        self._callback = None
        self._samplerate = 48000
        self._channels = 1
        self._blocksize = 1024
        self._lock = threading.Lock()

        # Check permissions
        if autoclass is not None:
            try:
                if not check_permission(Permission.RECORD_AUDIO):
                    print("[AndroidAudio] Requesting RECORD_AUDIO permission...")
                    request_permissions([Permission.RECORD_AUDIO])
            except Exception as e:
                print(f"[AndroidAudio] Permission check failed: {e}")

    def _init_audio_record(self, samplerate, channels, blocksize):
        """Initialize AudioRecord for recording."""
        if autoclass is None:
            return None

        AudioRecord = autoclass('android.media.AudioRecord')

        # Calculate buffer size (minimum size * 2 for safety)
        min_buffer_size = AudioRecord.getMinBufferSize(
            samplerate,
            self.CHANNEL_IN_MONO if channels == 1 else 12,  # CHANNEL_IN_STEREO = 12
            self.ENCODING_PCM_16BIT
        )

        if min_buffer_size <= 0:
            print(f"[AndroidAudio] Invalid min buffer size: {min_buffer_size}")
            return None

        buffer_size = max(min_buffer_size, blocksize * 2 * 2)  # at least blocksize*2*2 bytes

        audio_source = self.STREAM_VOICE_RECOGNITION

        record = AudioRecord(
            audio_source,
            samplerate,
            self.CHANNEL_IN_MONO if channels == 1 else 12,
            self.ENCODING_PCM_16BIT,
            buffer_size
        )

        return record

    def _init_audio_track(self, samplerate, channels):
        """Initialize AudioTrack for playback."""
        if autoclass is None:
            return None

        AudioTrack = autoclass('android.media.AudioTrack')

        # Calculate buffer size
        min_buffer_size = AudioTrack.getMinBufferSize(
            samplerate,
            self.CHANNEL_OUT_MONO if channels == 1 else 12,  # CHANNEL_OUT_STEREO = 12
            self.ENCODING_PCM_16BIT
        )

        if min_buffer_size <= 0:
            print(f"[AndroidAudio] Invalid min buffer size: {min_buffer_size}")
            return None

        buffer_size = max(min_buffer_size, samplerate * channels * 2 // 10)  # 100ms buffer

        track = AudioTrack(
            self.STREAM_MUSIC,
            samplerate,
            self.CHANNEL_OUT_MONO if channels == 1 else 12,
            self.ENCODING_PCM_16BIT,
            buffer_size,
            self.MODE_STREAM
        )

        return track

    def _record_loop(self):
        """Thread function for recording audio and calling callback."""
        if self._audio_record is None:
            return

        try:
            self._audio_record.startRecording()
        except Exception as e:
            print(f"[AndroidAudio] Failed to start recording: {e}")
            return

        # Buffer for reading (int16)
        read_buffer_size = self._blocksize * 2  # 2 bytes per sample
        java_buffer = autoclass('java.nio.ByteBuffer').allocate(read_buffer_size)

        while self._recording:
            try:
                bytes_read = self._audio_record.read(java_buffer, read_buffer_size)
                if bytes_read > 0:
                    # Get data from buffer
                    byte_data = bytes(java_buffer.array()[:bytes_read])
                    # Convert to numpy int16
                    data_int16 = np.frombuffer(byte_data, dtype=np.int16)
                    # Convert to float64 in [-1, 1]
                    data_float = data_int16.astype(np.float64) / 32768.0

                    # Prepare indata-like array (samples x channels)
                    if self._channels == 1:
                        indata = data_float.reshape(-1, 1)
                    else:
                        indata = data_float.reshape(-1, self._channels)

                    # Call callback
                    if self._callback is not None:
                        # Simulate sounddevice callback signature
                        frames = len(data_float) // self._channels
                        # time_info and status are not available, use None
                        self._callback(indata, frames, None, None)

                    # Clear buffer for next read
                    java_buffer.clear()
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[AndroidAudio] Record error: {e}")
                break

        try:
            self._audio_record.stop()
        except Exception as e:
            print(f"[AndroidAudio] Error stopping record: {e}")

    def start_stream(self, callback, samplerate=48000, channels=1, blocksize=1024):
        """
        Start an audio input stream.

        Args:
            callback: function(indata, frames, time_info, status)
            samplerate: sample rate in Hz
            channels: number of channels (1=mono, 2=stereo)
            blocksize: number of frames per buffer
        """
        with self._lock:
            if self._recording:
                self.stop()

            self._callback = callback
            self._samplerate = samplerate
            self._channels = channels
            self._blocksize = blocksize

            self._audio_record = self._init_audio_record(samplerate, channels, blocksize)
            if self._audio_record is None:
                print("[AndroidAudio] Failed to initialize AudioRecord")
                return

            self._recording = True
            self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self._record_thread.start()

    def play(self, data, samplerate=48000):
        """
        Play audio data.

        Args:
            data: numpy array of audio samples (float32 or float64 in [-1, 1])
            samplerate: sample rate in Hz
        """
        with self._lock:
            if self._playing:
                self.stop()

            # Ensure data is 1D
            if data.ndim > 1:
                data = data[:, 0]

            # Convert float to int16
            data_int16 = (data * 32767).astype(np.int16)

            self._audio_track = self._init_audio_track(samplerate, 1)
            if self._audio_track is None:
                print("[AndroidAudio] Failed to initialize AudioTrack")
                return

            try:
                self._audio_track.play()
            except Exception as e:
                print(f"[AndroidAudio] Failed to start playback: {e}")
                return

            self._playing = True

            # Write data in a separate thread to avoid blocking
            def write_data():
                try:
                    java_bytes = data_int16.tobytes()
                    # Convert to Java byte array
                    String = autoclass('java.lang.String')
                    java_string = String(java_bytes.decode('latin-1'))
                    byte_array = java_string.getBytes('latin-1')

                    self._audio_track.write(byte_array, 0, len(byte_array))

                    # Wait for playback to complete
                    time.sleep(len(data) / samplerate)

                except Exception as e:
                    print(f"[AndroidAudio] Playback error: {e}")
                finally:
                    with self._lock:
                        self._playing = False
                        try:
                            self._audio_track.stop()
                            self._audio_track.release()
                        except Exception:
                            pass
                        self._audio_track = None

            self._play_thread = threading.Thread(target=write_data, daemon=True)
            self._play_thread.start()

    def stop(self):
        """Stop all audio streams (input and output)."""
        with self._lock:
            # Stop recording
            if self._recording:
                self._recording = False
                if self._record_thread is not None:
                    self._record_thread.join(timeout=2.0)
                    self._record_thread = None
                if self._audio_record is not None:
                    try:
                        self._audio_record.stop()
                        self._audio_record.release()
                    except Exception as e:
                        print(f"[AndroidAudio] Error releasing AudioRecord: {e}")
                    self._audio_record = None

            # Stop playback
            if self._playing:
                self._playing = False
                if self._play_thread is not None:
                    self._play_thread.join(timeout=2.0)
                    self._play_thread = None
                if self._audio_track is not None:
                    try:
                        self._audio_track.stop()
                        self._audio_track.release()
                    except Exception as e:
                        print(f"[AndroidAudio] Error releasing AudioTrack: {e}")
                    self._audio_track = None

    def is_active(self):
        """Check if the input stream is currently active."""
        return self._recording

    def is_playing(self):
        """Check if the output stream is currently playing."""
        return self._playing
