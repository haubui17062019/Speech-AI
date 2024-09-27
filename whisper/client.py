import librosa
import numpy as np
import soundfile

import tritonclient
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype

import speech_recognition as sr
import pyaudio
import wave
import asyncio
import time
import types
import soundfile as sf
import audio_io


def record_audio(filename, duration=10, rate=16000, chunk=1024):
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


async def send_whisper(
        dps: list,
        triton_client: tritonclient.grpc.aio.InferenceServerClient,
        protocol_client: types.ModuleType,
        padding_duration=10,
        whisper_prompt: str = "<|startoftranscript|><|vi|><|transcribe|><|notimestamps|>",
        model_name: str = "whisper",
        name: str = "task-2",
        recognition=None

):
    task_id = int(name[5:])
    latency_data = []
    results = []
    while True:
        # for i, dp in enumerate(dps):
        i = 0
        with audio_io.MicrophoneStream(rate=16000, chunk=40000, device=1) as audio_chunk_iterator:
            waveform = np.frombuffer(audio_chunk_iterator.__next__(), dtype=np.int16)
            print('True')
            # waveform, sample_rate = soundfile.read(dp["audio_filepath"])
            sample_rate = 16000
            duration = int(len(waveform) / sample_rate)
            sf.write("mic.wav", waveform, sample_rate)

            # padding to nearset 10 seconds
            samples = np.zeros(
                (
                    1,
                    padding_duration * sample_rate * ((duration // padding_duration) + 1),
                ),
                dtype=np.float32,
            )

            samples[0, : len(waveform)] = waveform

            lengths = np.array([[len(waveform)]], dtype=np.int32)

            inputs = [
                protocol_client.InferInput(
                    "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
                ),
                protocol_client.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
            ]
            print(type(inputs[0]))
            inputs[0].set_data_from_numpy(samples)

            input_data_numpy = np.array([whisper_prompt], dtype=object)
            input_data_numpy = input_data_numpy.reshape((1, 1))
            inputs[1].set_data_from_numpy(input_data_numpy)

            outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
            sequence_id = 100000000 + i + task_id * 10
            start = time.time()
            response = await triton_client.infer(
                model_name, inputs, request_id=str(sequence_id), outputs=outputs
            )

            decoding_results = response.as_numpy("TRANSCRIPTS")[0]
            if type(decoding_results) == np.ndarray:
                decoding_results = b" ".join(decoding_results).decode("utf-8")
            else:
                # For wenet
                decoding_results = decoding_results.decode("utf-8")
            end = time.time() - start
            latency_data.append((end, duration))


            # results.append(
            #     (
            #         dp["id"],
            #         dp["text"].split(),
            #         decoding_results.split(),
            #     )
            # )

            print('[INFO]: ', decoding_results)


async def main():
    path_audio = "output.wav"
    dps_list = [

            {
                "audio_filepath": path_audio,
                "text": "foo",
                "id": 0
            }

    ]
    triton_client = grpcclient.InferenceServerClient(url="10.9.3.239:8001", verbose=False)
    protocol_client = grpcclient
    recognition = sr.Recognizer()

    tasks = []
    task = asyncio.create_task(send_whisper(dps=dps_list,
                                            triton_client=triton_client,
                                            protocol_client=protocol_client,
                                            recognition=recognition))
    tasks.append(task)
    print(tasks)

    ans_list = await asyncio.gather(*tasks)
    results = []
    total_duration = 0.0
    latency_data = []


if __name__ == "__main__":

    asyncio.run(main())
