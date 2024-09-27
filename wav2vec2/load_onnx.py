import torch
import numpy as np
import onnxruntime

# run the model on the backend
model_onnx = onnxruntime.InferenceSession('/home/data3/haubui/wav2vec2/wav2vec.onnx', providers=['CPUExecutionProvider'])

# lấy tên của đầu vào đầu tiên của mô hình
input_name = model_onnx.get_inputs()[0].name
input_dim = model_onnx.get_inputs()[0].shape

image = np.random.randn(1, 4000).astype(np.float32)

ort_inputs = {input_name: image}
output = model_onnx.run(None, ort_inputs)
print(output)

# print(logits.shape)
# print(embeding.shape)
# import nemo.collections.asr as nemo_asr
# speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
# embs = speaker_model.get_embedding('audio_path')
