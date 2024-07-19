import os
import riva.client
import IPython.display as ipd
import numpy as np
import soundfile as sf

server = "localhost:50051"                # location of riva server
auth = riva.client.Auth(uri=server)
tts_service = riva.client.SpeechSynthesisService(auth)


text = "Is it recognize speech or wreck a nice beach?"
# language_code = lang                   # currently required to be "en-US"
# sample_rate_hz = sample_rate                    # the desired sample rate
# voice_name = voice      # subvoice to generate the audio output.
data_type = np.int16                      # For RIVA version < 1.10.0 please set this to np.float32

resp = tts_service.synthesize(text)
audio = resp.audio
meta = resp.meta
processed_text = meta.processed_text
predicted_durations = meta.predicted_durations

audio_samples = np.frombuffer(resp.audio, dtype=data_type)
print(processed_text)
# ipd.Audio(audio_samples, rate=sample_rate_hz)
sf.write("output.text", audio_samples, sample_rate_hz)