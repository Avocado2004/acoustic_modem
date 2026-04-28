from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import wave

# 🔐 Шифрование текста
def encrypt_text(text, password=None):
    if not password:
        return text.encode('utf-8')
    key = hashlib.sha256(password.encode()).digest()
    cipher = AES.new(key, AES.MODE_CBC)
    encrypted = cipher.iv + cipher.encrypt(pad(text.encode('utf-8'), AES.block_size))
    return encrypted

# 🔓 Дешифрование
def decrypt_text(encrypted_bytes, password=None):
    if not password:
        return encrypted_bytes.decode('utf-8')
    key = hashlib.sha256(password.encode()).digest()
    iv = encrypted_bytes[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(encrypted_bytes[AES.block_size:]), AES.block_size)
    return decrypted.decode('utf-8')

# 📊 Байты → Биты
def text_to_bits(data_bytes):
    return ''.join(format(b, '08b') for b in data_bytes)

# 🔁 Биты → Байты
def bits_to_bytes(bits):
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_chunk = bits[i:i+8]
        if len(byte_chunk) == 8:
            byte_array.append(int(byte_chunk, 2))
    return bytes(byte_array)

# 🎵 Сохранение битов как WAV
def save_bits_to_wave(bits, filename='output.wav'):
    frames = bytes([int(bits[i:i+8], 2) for i in range(0, len(bits), 8) if len(bits[i:i+8]) == 8])
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(8000)
        wav_file.writeframes(frames)

# 🧪 Пример использования
if __name__ == "__main__":
    original_text = "Секретное сообщение"
    password = "мойпароль"

    # Шифрование и сохранение
    encrypted_bytes = encrypt_text(original_text, password)
    bits = text_to_bits(encrypted_bytes)
    save_bits_to_wave(bits, 'secret.wav')
    print("✅ WAV сохранён как 'secret.wav'")

    # Восстановление
    recovered_bytes = bits_to_bytes(bits)
    decrypted_text = decrypt_text(recovered_bytes, password)
    print("🔍 Расшифровано:", decrypted_text)

