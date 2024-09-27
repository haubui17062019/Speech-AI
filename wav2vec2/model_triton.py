import torch
import tritonclient.grpc as grpcclient
import numpy as np
from munch import munchify
import yaml
import math
import re
import librosa
from fairseq.data import Dictionary

from decode import W2lKenLMDecoder


def add_asr_eval_argument(args):
    args.kspmodel = None
    args.wfstlm = None
    args.task = "audio_finetuning"
    args.rnnt_decoding_type = "greedy"

    args.rnnt_len_penalty = -0.5
    args.lm_weight = 0.1
    # args.kenlm_model = "vi_lm_4grams.bin"
    args.kenlm_model = "./ckpt/lm_evn_wav2vec_2k.bin"
    args.lexicon = "./ckpt/lexicon.txt"
    args.unit_lm = False

    args.beam_threshold = 25.0
    args.beam_size_token = 100
    args.word_score = -1.0
    args.unk_weight = -math.inf
    args.sil_weight = 0.0
    args.dump_emissions = None
    args.dump_features = None
    args.load_emissions = None
    args.nbest = 1
    # args.path = "/home/damnguyen/Speech/fairseq/outputs/2021-11-15/20-50-52/checkpoints/checkpoint_best.pt"
    # args.path = "/home/damnguyen/Speech/fairseq/outputs/2021-11-23/23-48-47/checkpoints/checkpoint_best.pt"
    args.path = "/home/data3/haubui/wav2vec2/ckpt/wav2vec-fix-ng-h.onnx"
    args.criterion = "ctc"
    args.labels = "ltr"
    args.max_tokens = 4000000
    args.post_process = "letter"
    args.beam = 20
    args.data = ""
    args.gen_subset = "deploy"
    args.fp16 = False
    return args



class Wav2vec2Triton(object):
    def __init__(self, args, tgt_dict):
        self.model_name = "wav2vec2"
        self.url = "0.0.0.0:8819"
        self.client = grpcclient.InferenceServerClient(self.url)
        self.args = args
        self.tgt_dict = tgt_dict
        self.decoder = W2lKenLMDecoder(self.args, self.tgt_dict)

    def get_normalized_probs(self, f_out, log_probs=None):
        """
            Normalize prob with softmax and log_softmax
        """
        exp_f_out = np.exp(f_out)  # F x 1 x 105 (105 is number of token)
        sum_by_row = np.sum(exp_f_out, axis=-1, keepdims=True)  # F x 1
        normalized_prob = exp_f_out / sum_by_row
        if log_probs:
            normalized_prob = np.log(normalized_prob)
        return normalized_prob

    def forward_encode(self, wav_input):
        """
            Get encoder output
        :param wav_input: waveform load from audio
        :return: encoder: F x 1 x 105
        """

        print('[INFO] Call triton')
        f_outs = []
        assert type(wav_input).__name__ == "ndarray", "Input must be numpy array: {}".format(type(wav_input))
        if len(wav_input.shape) == 1:
            wav_input = np.expand_dims(wav_input, 0)

        inputs = [grpcclient.InferInput("input", wav_input.shape, "FP32")]
        inputs[0].set_data_from_numpy(wav_input)
        outputs = [grpcclient.InferRequestedOutput("output")]

        results = self.client.infer(model_name=self.model_name,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=None)
        f_out = results.as_numpy("output")
        f_outs.append(self.get_normalized_probs(f_out))

        return f_outs

    def process_predictions(self, hypos):
        # if "words" in hypos:
        # print('[INFO] hypos: ', hypos)
        hyp_kenlm = ""
        for hypo in hypos:
            print('[INFO] hypos: ', hypo)
            hypo = hypo[0]
            if len(hypo["words"]) == 1:
                hyp_kenlm = hypo["words"][0]
            else:
                hyp_kenlm = " ".join(hypo["words"])
            hyp_kenlm = re.sub('\s+', ' ', hyp_kenlm)
        return hyp_kenlm.strip()

    def forward_decoder(self, list_f_out, get_ctm: bool = False):
        """
            Forward Decoder
        """
        lst_setence = []
        for ix, emissions in enumerate(list_f_out):
            emissions = np.transpose(emissions, (1, 0, 2))
            emissions_torch = torch.from_numpy(emissions)
            hypos = self.decoder.decode(emissions_torch)
            sentence = self.process_predictions(hypos)
            lst_setence.append(sentence)
        return lst_setence

    def get_transcript(self, waveform):
        """

        :param waveform:
        :return:
        """
        print('[INFO] Start')

        lst_emissions = self.forward_encode(waveform)
        lst_sentence = self.forward_decoder(lst_emissions)
        concat_sentence = re.sub("\s+", " ", " ".join(lst_sentence)).lower()
        return concat_sentence



if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        args = munchify(yaml.safe_load(stream))
    args = add_asr_eval_argument(args)

    tgt_dict = Dictionary.load("./ckpt/target_dict.txt")
    audio_path = "/home1/data/haubui/Speech-AI/wav2vec2/check_ngram_5.m4a"
    waveform, sr = librosa.load(audio_path, sr=16000)
    print('[INFO] Load audio done')
    model = Wav2vec2Triton(args=args, tgt_dict=tgt_dict)

    transcript = model.get_transcript(waveform[:5000])
    print('[INFO] transcript: ', transcript)

