from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel

import torch
import soundfile as sf
import numpy as np


class TtsRiva(torch.nn.Module):
    def __init__(self, spec_generator, vocoder):
        super().__init__()
        self.spec_generator = spec_generator
        self.vocoder = vocoder

    def forward(self, tokens):
        spectrogram = self.spec_generator.generate_spectrogram(tokens=tokens)
        audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        return audio.to('cpu').detach()

spec_generator = FastPitchModel.restore_from("tts_en_fastpitch_align_ipa.nemo", map_location="cpu")
spec_generator.eval()
vocoder = HifiGanModel.restore_from("tts_hifigan.nemo", map_location="cpu")
vocoder.eval()

tts_model = TtsRiva(spec_generator, vocoder)

tokens = spec_generator.parse("You can type your sentence here to get nemo to produce speech")
#
# audio = tts_model(tokens)
#
# print(audio)
# audio = np.ravel(audio)
#
# sf.write("output.wav", audio, 22050, format="wav")

# input_lengths = torch.tensor([tokens.shape[1]])
# tokens = tokens.to("cpu").numpy()
tokens = torch.rand((1, 64)).long()
# print('=======================================')
# torch.tensor([1, :])
# print('[INFO] type tokens of spec_generator: ', type(tokens), tokens.shape)
# print('[INFO] input_lengths: ', input_lengths)

torch.onnx.export(tts_model,
                  tokens,
                  f="tts_riva.onnx",
                  input_names=["inputs"],
                  output_names=["audio"],
                  opset_version=11,
                  export_params=True,
                  dynamic_axes={
                      "inputs": {1: "sentence_length"},
                      "audio": {1: "sentence_length"}
                  }
                  )



