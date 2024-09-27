from nemo.collections.tts.models import FastPitchModel
import onnxruntime
import numpy as np
import soundfile as sf


spec_generator = FastPitchModel.restore_from("tts_en_fastpitch_align_ipa.nemo", map_location="cpu")
spec_generator.eval().cuda()

model_onnx = onnxruntime.InferenceSession("tts_riva.onnx", providers=["CUDAExecutionProvider"])

input_name = model_onnx.get_inputs()[0].name
input_dim = model_onnx.get_inputs()[0].shape
print('input_name: ', input_name)
print('input_dim: ', input_dim)
# print(model_onnx.get_inputs()[1])

tokens = spec_generator.parse("hello world").to('cpu').numpy()
# tokens = np.random.rand(1, 80).astype(np.int64)
print('tokens shape', tokens.shape)
ort_input = {input_name: tokens}
audio = model_onnx.run(None, ort_input)[0][0]
sf.write("output_1.wav", audio, 22050)
print(audio)




