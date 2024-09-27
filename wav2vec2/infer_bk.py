import os
import math
import sys
import torch
import torch.nn.functional as F
import numpy as np
import itertools as it
import torch.nn as nn
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.tasks.audio_pretraining import AudioPretrainingTask
from fairseq.data import Dictionary
from fairseq.models import BaseFairseqModel
import soundfile as sf
from wav2letter.decoder import CriterionType
# from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
import contextlib
import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.models import FairseqEncoder
from examples.wav2vec2.tasks.audio_pretraining import Wav2vec2PretrainingTask


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, args, tgt_dict=None):
        self.apply_mask = args.apply_mask

        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }

        if getattr(args, "w2v_args", None) is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.w2v_path, arg_overrides
            )
            w2v_args = state["args"]
        else:
            state = None
            w2v_args = args.w2v_args

        assert args.normalize == w2v_args.normalize, 'Fine-tuning works best when data normalization is the same'

        w2v_args.data = args.data
        task = Wav2vec2PretrainingTask.setup_task(w2v_args)
        model = task.build_model(w2v_args)

        if state is not None and not args.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()
        super().__init__(task.source_dictionary)

        d = w2v_args.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(args, 'decoder_embed_dim', d) != d:
            self.proj = Linear(d, args.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def base_architecture(args):
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)


class W2lDecoder(object):
    def __init__(self, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = 1

        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.asg_transitions = None

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)

        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))


# class W2lViterbiDecoder(W2lDecoder):
#     def __init__(self, tgt_dict):
#         super().__init__(tgt_dict)
#
#     def decode(self, emissions):
#         B, T, N = emissions.size()
#         hypos = list()
#
#         if self.asg_transitions is None:
#             transitions = torch.FloatTensor(N, N).zero_()
#         else:
#             transitions = torch.FloatTensor(self.asg_transitions).view(N, N)
#
#         viterbi_path = torch.IntTensor(B, T)
#         workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
#         CpuViterbiPath.compute(
#             B,
#             T,
#             N,
#             get_data_ptr_as_bytes(emissions),
#             get_data_ptr_as_bytes(transitions),
#             get_data_ptr_as_bytes(viterbi_path),
#             get_data_ptr_as_bytes(workspace),
#         )
#         return [
#             [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}] for b in range(B)
#         ]


class Wav2VecCtc(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_common_args(parser)

    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, target_dict):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args, target_dict)
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


def get_feature(filepath):
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    wav, sample_rate = sf.read(filepath)
    feats = torch.from_numpy(wav).float()
    feats = postprocess(feats, sample_rate)
    return feats


def load_target_dict(manifest_path='./manifest'):
    dict_path = os.path.join(manifest_path, "dict.ltr.txt")
    target_dict = Dictionary.load(dict_path)
    return target_dict


def load_model(model_path, target_dict):
    # state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
    # args = state["args"]
    w2v = torch.load(model_path)

    # from examples.wav2vec2.models.wav2vec2_asr import Wav2Vec2Model
    model = Wav2VecCtc.build_model(w2v["args"], target_dict)
    model.load_state_dict(w2v["model"], strict=True)

    return [model]


def main():
    sample, input = dict(), dict()
    WAV_PATH = 'test.wav'
    W2V_PATH = '/home/data3/haubui/wav2vec2/ckpt/checkpoint_best.pt'

    manifest_path = "MANIFEST_PATH"
    feature = get_feature(WAV_PATH)

    use_cuda = torch.cuda.is_available()

    target_dict = load_target_dict(manifest_path)
    model = load_model(W2V_PATH, target_dict)
    model[0].eval()

    generator = W2lViterbiDecoder(target_dict)
    input["source"] = feature.unsqueeze(0)

    padding_mask = torch.BoolTensor(input["source"].size(1)).fill_(False).unsqueeze(0)

    input["padding_mask"] = padding_mask
    sample["net_input"] = input

    with torch.no_grad():
        hypo = generator.generate(model, sample, prefix_tokens=None)

    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())
    print(post_process(hyp_pieces, 'letter'))


if __name__ == '__main__':
    main()