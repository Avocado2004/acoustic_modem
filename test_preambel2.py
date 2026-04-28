import sounddevice as sd
import scipy.io.wavfile as wav

fs = 48000
duration = 2
print("🎙️ Запись...")
audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float32')
sd.wait()
wav.write("mic_test.wav", fs, audio)
print("✅ Сохранено в mic_test.wav")

