import pyaudio
import wave

def record_audio(filename, duration, rate=16000, chunk=1024):
    # Thiết lập PyAudio
    audio = pyaudio.PyAudio()

    # Mở stream để ghi âm
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk,
                        input_device_index=1)

    print("Ghi âm...")

    # Ghi âm dữ liệu
    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Ghi âm xong")

    # Dừng stream và đóng
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Lưu dữ liệu vào file WAV
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Gọi hàm để ghi âm
record_audio('../Triton-ASR-Client/output.wav', duration=5)  # Ghi âm trong 5 giây và lưu vào output.wav
