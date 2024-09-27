"""
    Implement Wrapped Wav2vec Model with ONNX & TensorRT Engine
"""
import sys
import re
import time
import math
import yaml

# import soundfile as sf
from decode import W2lKenLMDecoder

# from fairseq.data import Dictionary
from fairseq.data.data_utils import post_process
import numpy as np
import torch
# import librosa

from munch import munchify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM


def add_asr_eval_argument(args):
    args.kspmodel = None
    args.wfstlm = None
    args.task = "audio_finetuning"
    args.rnnt_decoding_type = "greedy"

    args.rnnt_len_penalty = -0.5
    args.lm_weight = 0.1
    # args.kenlm_model = "vi_lm_4grams.bin"
    args.kenlm_model = "/home/data3/haubui/wav2vec2/ckpt/lm_evn_wav2vec_2k.bin"
    args.lexicon = "/home/data3/haubui/wav2vec2/ckpt/lexicon.txt"
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


class Wav2Vec(object):
    def __init__(self,
                 args,
                 tgt_dict,
                 engine='ONNX',
                 huggingface_model_id="",
                 ):
        self.args = args
        self.engine = engine
        self.sample_rate = 16000
        self.truncate = 960000
        self.min_truncate = 8000
        self.tgt_dict = tgt_dict
        if self.engine == 'ONNX':
            import onnxruntime
            print('Create Wav2vec model with ONNX runtime')
            self.model = onnxruntime.InferenceSession(self.args.path, providers=["CUDAExecutionProvider"])
        elif self.engine == 'HUGGINGFACE':
            self.device = torch.device("cuda")
            self.processor = Wav2Vec2Processor.from_pretrained(huggingface_model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(huggingface_model_id)
            self.model.eval()
            self.model.to(self.device)
        else:
            raise NotImplementedError("Current support only ONNX & TRT runtime")

        self.decoder = W2lKenLMDecoder(self.args, self.tgt_dict)

    def forward_encoder(self, wav_inputs):
        """
            Get encoder output for multiple parts
        """
        net_outs = []
        for ix, wav_input in enumerate(wav_inputs):
            assert type(wav_input).__name__ == 'ndarray', 'Input must be numpy array: {}'.format(wav_input)
            assert len(wav_input.shape) == 2, "Input must be 2-dimensions: {}".format(wav_input.shape)
            if self.engine == 'ONNX':
                ort_inputs = {self.model.get_inputs()[0].name: wav_input}
                net_out = self.model.run(None, ort_inputs)[0]
            elif self.engine == 'TRT' or self.engine == 'TRITON':
                net_out = self.model.run(wav_input)[0]
                print(net_out.shape)
            elif self.engine == 'HUGGINGFACE':
                inputs = self.processor(wav_input[0], sampling_rate=16_000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    net_out = self.model(inputs.input_values.to(self.device),
                                         attention_mask=inputs.attention_mask.to(self.device)).logits
                net_out = net_out.permute((1, 0, 2)).detach().cpu().numpy()
                print(net_out.shape)
            else:
                raise NotImplementedError("Not support engine")
            net_outs.append(self.get_normalized_probs(net_out))
        return net_outs

    def get_normalized_probs(self, net_out, log_probs=False):
        """
            Normalize probabilites with softmax and log_softmax
        """
        exp_net_out = np.exp(net_out)  # F x 1 x 105 (105 is number of tokens)
        sum_by_row = np.sum(exp_net_out, axis=-1, keepdims=True)  # F x 1
        normalized_prob = exp_net_out / sum_by_row
        if log_probs:
            normalized_prob = np.log(normalized_prob)
        return normalized_prob  # F x 1 x 105

    def preprocess_audio(self, audio_path):
        """
            Read and preprocess audio
        """
        # assert self.truncate > 0, "Invalid truncation value: {}".format(self.truncate)
        lst_feats = []
        if self.engine == 'HUGGINGFACE':
            feats, curr_sample_rate = sf.read(audio_path, dtype="float32")
        else:
            feats, curr_sample_rate = librosa.load(audio_path, sr=16000)
        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert len(feats.shape) == 1, "Invalid feat shape: {}".format(feats.shape)

        for i in range(0, len(feats), self.truncate):
            lower = i
            higher = min(len(feats), i + self.truncate)
            if higher - lower > self.min_truncate:
                lst_feats.append(np.array([feats[lower:higher]]))

        return lst_feats

    def fix_ngh_gh(self, chars_viterbi):
        '''
            Fix special case: nge -> nghe
                              ngỉ -> nghỉ
        '''
        fixed_string = []
        lst_characters = chars_viterbi.strip().split()
        for ix, ch in enumerate(lst_characters):
            if ch == 'ng' or ch == 'g':
                if ix < len(lst_characters) - 1:
                    if lst_characters[ix + 1] in ['i', 'ỉ', 'ĩ', 'ì', 'ị', 'e', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ề', 'ể', 'ễ',
                                                  'ệ']:
                        if ch == 'ng':
                            fixed_string.append('ngh')  # ng -> ngh
                        else:
                            if lst_characters[ix + 2] != "|":
                                if lst_characters[ix + 1] != "i":
                                    fixed_string.append('gh')  # g -> gh
                                else:
                                    fixed_string.append(ch)
                            else:
                                if lst_characters[ix + 1] == "i":
                                    fixed_string.append('gh')
                                else:
                                    fixed_string.append(ch)
                    else:
                        fixed_string.append(ch)
                else:
                    fixed_string.append(ch)
            else:
                fixed_string.append(ch)
        fixed_string = " ".join(fixed_string)
        return fixed_string.strip()

    def process_ctm(self, timesteps, chars_viterbi, hyp_kenlm, index_of_truncate, show_analyst=False):
        '''
            Generate word-level timestamps
        '''
        lst_words_kenlm = re.split("\s+", hyp_kenlm)
        # lst_words_kenlm  =  hyp_kenlm.strip().split()
        hyp_viterbi = post_process(chars_viterbi, self.args.post_process)
        hyp_viterbi = re.sub('\s+', ' ', hyp_viterbi)
        lst_words_viterbi = re.split("\s+", hyp_viterbi)
        # lst_words_viterbi = hyp_viterbi.strip().split()

        # Ignore empty sequence
        if len(hyp_viterbi) == 0:
            return []
        chars_segment = chars_viterbi.strip().split()
        # if len(set(chars_segment)) == 1, "Empty "
        # assert len(timesteps) == len(chars_segment), "Not equal timesteps & number of characters: {} vs {}".format(timesteps, chars_segment)
        if len(timesteps) != len(chars_segment):
            if show_analyst:
                print("[WARNING] Not equal timesteps & number of characters: {} vs {}".format(len(timesteps),
                                                                                              len(chars_segment)))
            # print([(chars_segment[i], timesteps[i]) for i in range(min(len(timesteps), len(chars_segment)))])
        timesteps = timesteps[:len(chars_segment)]
        assert chars_segment[0] == '|', "Invalid chars segment: {}".format(chars_viterbi)
        # Assign word for each timestamps
        lst_timestamps = []
        ix = 0

        while ix < len(timesteps) - 1:
            assert chars_segment[ix] == '|'
            for iy in range(ix + 1, len(timesteps)):
                if chars_segment[iy] == '|':
                    break
            if iy - ix > 1:
                lst_timestamps.append((timesteps[ix + 1], timesteps[iy - 1]))
            ix = iy

        assert len(lst_timestamps) == len(lst_words_viterbi), "Length lm & viterbi not equal: {} vs {}".format(
            lst_timestamps, lst_words_viterbi)

        # Remove empty word with segment
        total_words = len(lst_timestamps)
        lst_timestamps = [lst_timestamps[ix] for ix in range(total_words) if len(lst_words_viterbi[ix]) > 0]
        lst_words_viterbi = [lst_words_viterbi[ix] for ix in range(total_words) if len(lst_words_viterbi[ix]) > 0]
        lst_words_kenlm = [x for x in lst_words_kenlm if len(x) > 0]
        # print('After clean: {}/{}'.format(len(lst_timestamps), total_words))
        assert len(lst_timestamps) == len(lst_words_viterbi), "Length lm & viterbi not equal: {} vs {}".format(
            lst_timestamps, lst_words_viterbi)

        result = []
        if len(lst_timestamps) == len(lst_words_kenlm):
            # Return result
            for ix in range(len(lst_timestamps)):
                lower, higher = lst_timestamps[ix]
                lower = round(index_of_truncate * 60 + (lower * 0.02), 2)
                higher = round(index_of_truncate * 60 + (higher * 0.02), 2)
                result.append((lower, higher, lst_words_kenlm[ix], 1.0))
            return result

        else:
            # Append EOS token
            lst_words_viterbi.append('EOS')
            lst_words_kenlm.append('EOS')
            lst_timestamps.append(None)

            # Heuristic merge
            if show_analyst:
                print("Length words & timestamps not equal: {} vs {}".format(len(lst_words_kenlm), len(lst_timestamps)))
            assert len(lst_timestamps) > len(lst_words_kenlm), "Invalid timestamps & kenLM: {} vs {}".format(
                len(lst_timestamps), len(lst_words_kenlm))

            counter_kenlm = 0
            counter_viterbi = 0
            while counter_kenlm < len(lst_words_kenlm):
                if lst_words_kenlm[counter_kenlm] == lst_words_viterbi[counter_viterbi]:
                    if lst_words_kenlm[counter_kenlm] != 'EOS' and lst_timestamps[counter_viterbi] is not None:
                        lower, higher = lst_timestamps[counter_viterbi]
                        lower = round(index_of_truncate * 60 + (lower * 0.02), 2)
                        higher = round(index_of_truncate * 60 + (higher * 0.02), 2)
                        result.append((lower, higher, lst_words_kenlm[counter_kenlm], 1.0))
                        # print('Similar: {}'.format(lst_words_kenlm[counter_kenlm]))
                        counter_kenlm += 1
                        counter_viterbi += 1
                    else:
                        print('[EOS] found, break')
                        break
                else:
                    # 2 different words
                    print('Searching for: {}'.format(lst_words_kenlm[counter_kenlm]))
                    if lst_words_kenlm[counter_kenlm] == 'EOS':
                        if lst_words_viterbi[counter_viterbi] != 'EOS':
                            assert len(lst_words_viterbi[counter_viterbi:]) <= 2, "Invalid token EOS: {}".format(
                                lst_words_viterbi[counter_viterbi:])
                            print("[WARNING] Invalid token EOS: {}".format(lst_words_viterbi[counter_viterbi:]))
                        break
                    it = 0
                    for iy in range(counter_kenlm + 1, len(lst_words_kenlm)):
                        # print('- Range {} - {} / {} - {}'.format(counter_kenlm + 1, len(lst_words_kenlm), counter_viterbi + it, counter_viterbi + 2 + it))
                        found = False
                        for iz in range(counter_viterbi + 1 + it, min(counter_viterbi + 5 + it,
                                                                      len(lst_words_viterbi))):  # 2 is maximum range for searching word
                            if lst_words_kenlm[iy] == lst_words_viterbi[iz]:
                                print('Found equal {}, {}'.format(iy, iz))
                                found = True
                                break
                        if found:
                            break
                        it += 1
                    word = " ".join(lst_words_kenlm[counter_kenlm: iy]).strip()
                    word_o = " ".join(lst_words_viterbi[counter_viterbi: iz]).strip()
                    if show_analyst:
                        print('Replace {} with {}'.format(word_o, word))

                    lower = round(index_of_truncate * 60 + (lst_timestamps[counter_viterbi][0] * 0.02), 2)
                    higher = round(index_of_truncate * 60 + (lst_timestamps[iz - 1][1] * 0.02), 2)
                    result.append((lower, higher, word, 1.0))
                    # if len(word) > 0 and len(word_o) > 0:
                    assert iy > counter_kenlm, "Infinity loop: '{}' '{}' {} {}".format(word, word_o, iy, counter_kenlm)
                    counter_kenlm = iy
                    counter_viterbi = iz

            assert counter_kenlm == len(lst_words_kenlm) - 1, "{} {} vs {} {}".format(counter_viterbi,
                                                                                      len(lst_words_viterbi),
                                                                                      counter_kenlm,
                                                                                      len(lst_words_kenlm))
            return result

    def process_predictions(self, hypos, index_of_truncate, get_ctm, show_analyst=False, is_vietnamese=True):
        '''
            Process prediction
            :params:     - hypos: hyposthesis from LM/Viterbi
                         - index_of_truncate: index of truncate part
        '''
        for hypo in hypos[: min(len(hypos), self.args.nbest)]:
            # print('hypo', hypo)

            scores = hypo["score"]
            chars_viterbi = self.tgt_dict.string(hypo["tokens"].int().cpu())
            if is_vietnamese:
                chars_viterbi = self.fix_ngh_gh(chars_viterbi)
            if show_analyst:
                print('[ORIG]', chars_viterbi)
            # hyp_kenlm = post_process(chars_viterbi, self.args.post_process)
            if "words" in hypo:  # use KenLM
                hyp_kenlm = " ".join(hypo["words"])
            else:  # use Viterbi
                hyp_kenlm = post_process(chars_viterbi, self.args.post_process)
            # print('[POST -]', hyp_kenlm)
            hyp_kenlm = re.sub('\s+', ' ', hyp_kenlm)
            if show_analyst:
                print('[POST]', hyp_kenlm)
            word_level = None
            score = hypo["score"]
            if get_ctm:
                timesteps = hypo.get("timesteps", None)
                if timesteps is not None:
                    word_level = self.process_ctm(timesteps, chars_viterbi, hyp_kenlm, index_of_truncate,
                                                  show_analyst=show_analyst)
        return hyp_kenlm.strip(), word_level, score

    def forward_decoder(self, lst_emissions, get_ctm, show_analyst=False, is_vietnamese=True):
        """
            Forward decoder
        """
        lst_sentence = []
        lst_word_level = None
        lst_score = []
        if get_ctm:
            lst_word_level = []
        for ix, emissions in enumerate(lst_emissions):
            emissions = np.transpose(emissions, (1, 0, 2))
            emissions_torch = torch.from_numpy(emissions)
            hypos = self.decoder.decode(emissions_torch)
            if type(hypos[0]).__name__ == 'dict':
                hypos = [hypos]
            sentence, word_level, score = self.process_predictions(hypos[0], index_of_truncate=ix, get_ctm=get_ctm,
                                                                   show_analyst=show_analyst,
                                                                   is_vietnamese=is_vietnamese)
            lst_sentence.append(sentence)
            lst_score.append(score)
            if get_ctm and word_level is not None:
                lst_word_level += word_level
        avg_score = sum(lst_score) / len(lst_score)
        return lst_sentence, lst_word_level, avg_score

    def get_transcript(self, audio_path, show_analyst=False, get_ctm=False, is_vietnamese=True):
        """
            Get audio transcript from audio path
        """
        t0 = time.time()
        feats = self.preprocess_audio(audio_path)
        t1 = time.time()
        lst_emissions = self.forward_encoder(feats)
        t2 = time.time()
        lst_sentence, lst_word_level, avg_score = self.forward_decoder(lst_emissions, get_ctm=get_ctm,
                                                                       show_analyst=show_analyst,
                                                                       is_vietnamese=is_vietnamese)
        t3 = time.time()
        concat_sentence = re.sub("\s+", " ", " ".join(lst_sentence)).lower()
        if show_analyst:
            print('=' * 50)
            print('- Truncate to {} part with length: {}'.format(len(feats), [part.shape[1] for part in feats]))
            print('- Time read audio:', t1 - t0)
            print('- Time infer:', t2 - t1)
            print('- Time CPU KENLM:', t3 - t2)
            print('- Total time: ', t3 - t0)
        return concat_sentence, lst_word_level, avg_score


if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        args = munchify(yaml.safe_load(stream))
    args = add_asr_eval_argument(args)

    args.kenlm_model = "/home/data3/haubui/wav2vec2/ckpt/vi_lm_4grams.bin"

    from fairseq.data import Dictionary

    tgt_dict = Dictionary.load("/home/data3/haubui/wav2vec2/ckpt/target_dict.txt")
    model = Wav2Vec(args=args, tgt_dict=tgt_dict)

    import librosa
    import soundfile as sf
    import glob

    lst_audio = glob.glob('/home/data3/haubui/wav2vec2/data_test/*.m4a')

    for path in lst_audio:
        # path = "/home/data3/haubui/wav2vec2/data_test/check_ngram_6.m4a"
        print('=====')
        print(path)
        wav, sr = librosa.load(path)
        sf.write('test.wav', wav, sr)

        transcript, word_level, score = model.get_transcript(path, show_analyst=False, get_ctm=True)

        print(transcript)
