import queue

import pyaudio


class MicrophoneStream:
    """Opens a recording stream as responses yielding the audio chunks."""

    def __init__(self, rate: int, chunk: int, device: int = None) -> None:
        self._rate = rate
        self._chunk = chunk
        self._device = device

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            input_device_index=self._device,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def close(self) -> None:
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the responses to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def __exit__(self, type, value, traceback):
        self.close()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def __next__(self) -> bytes:
        if self.closed:
            raise StopIteration
        chunk = self._buff.get()
        if chunk is None:
            raise StopIteration
        data = [chunk]

        while True:
            try:
                chunk = self._buff.get(block=False)
                if chunk is None:
                    assert not self.closed
                data.append(chunk)
            except queue.Empty:
                break

        return b''.join(data)

    def __iter__(self):
        return self


def list_input_devices() -> None:
    p = pyaudio.PyAudio()
    print("Input audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] < 1:
            continue
        print(f"{info['index']}: {info['name']}")
    p.terminate()


# list_input_devices()
#
# import numpy as np
#
# while True:
#     with MicrophoneStream(
#             rate=16000, chunk=1600, device=1
#     ) as audio_chunk_iterator:
#         # audio = np.fromb audio_chunk_iterator.__next__())
#         audio = np.frombuffer(audio_chunk_iterator.__next__(), dtype=np.int16)
#         print(audio.shape)
#         print('======')
