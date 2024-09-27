import tritonclient.grpc.aio
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import logging
import numpy as np
import librosa
import torch
import torchaudio
import time
import cv2


def main():
    MODEL_NAME = "wav2vec2"
    URL = "0.0.0.0:8819"
    client = grpcclient.InferenceServerClient(URL)
    input = np.random.randn(1, 10000).astype(np.float32)
    # wav_1, sr = librosa.load('/home/haubui/convert_model/torch_to_onnx/url.wav', sr=16000)
    # input = np.expand_dims(wav_1, 0)



    inputs = []
    inputs.append(grpcclient.InferInput("input", input.shape, "FP32"))
    inputs[0].set_data_from_numpy(input)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output"))

    results = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
        client_timeout=None)
    output = results.as_numpy('output')
    print(output.shape)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

