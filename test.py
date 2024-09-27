import librosa
import soundfile as sf
from omegaconf import OmegaConf
import numpy as np
# from nemo.collections.asr.models import ClusteringDiarizer
from cluster_custom_test import ClusteringDiarizer
import torch
import shutil
import os

cfg = OmegaConf.load('./config/diar_infer_meeting_test.yaml')
diarizer_model = ClusteringDiarizer(cfg=cfg).to(cfg.device)


def get_segment_clus():
    with open('./result_test/pred_rttms/input_wav_test.rttm', "r") as file:
        lines = file.readlines()
    diar_segments = dict()
    i = 0
    while i <= len(lines) - 1:
        # Xử lý từng dòng trong tệp RTTM tại đây
        a, start, c = lines[i].split("   ")
        duration, _, _, speaker, _, _ = c.split(" ")
        _, speaker_id = speaker.split("_")
        end = float(start) + float(duration)
        if len(lines) - 1 == 0:
            if speaker_id not in diar_segments:
                diar_segments[speaker_id] = []
            diar_segments[speaker_id].append([float(start), end])
        count = 0
        for j in range(i+1, len(lines)):
            print(lines[j])
            print(j,  len(lines) - 1)
            a_, start_, c_ = lines[j].split("   ")
            duration_, _, _, speaker_, _, _ = c_.split(" ")
            _, speaker_id_check = speaker_.split("_")
            if speaker_id != speaker_id_check:
                if speaker_id not in diar_segments:
                    diar_segments[speaker_id] = []
                diar_segments[speaker_id].append([float(start), end])
                if j == len(lines) - 1:
                    if speaker_id_check not in diar_segments:
                        diar_segments[speaker_id_check] = []
                    diar_segments[speaker_id_check].append([float(start_), float(start_) + float(duration_)])
                break
            count = count + 1
            end = float(start_) + float(duration_)
            if j == len(lines) - 1:
                if speaker_id not in diar_segments:
                    diar_segments[speaker_id] = []
                diar_segments[speaker_id].append([float(start), end])
        i = i + count + 1
    return diar_segments


def get_silence(duration):
    with open('./result_test/vad_outputs/input_wav.txt', 'r') as f:
        lines = f.readlines()
    s_time, e_time = 0, 0
    silence_chunks = []
    for line in lines:
        e_time, dur, _ = line.split(' ')
        silence_chunks.append(
            [np.round_(float(s_time), 2), np.round_(float(e_time), 2)])
        s_time = float(e_time) + float(dur)

    if np.round(s_time, 2) != np.round(duration, 2):
        silence_chunks.append([np.round_(s_time, 2), np.round_(duration, 2)])
    return silence_chunks


def process_diarizate(filename: list):
    diari_result = dict()
    # wav, sr = librosa.load(filename, sr=16000)
    # duration = librosa.get_duration(y=wav, sr=sr)
    # sf.write('./manifest/input_wav_test.wav', wav, sr)
    diarizer_model.diarize()

    dia_segment = get_segment_clus()
    # silence_chunks = get_silence(duration)

    # diari_result[filename] = {
    #     'diari_segment': dia_segment,
    #     'silence_chunk': silence_chunks
    # }

    # format api NLP
    diari_result = dia_segment

    # shutil.rmtree('./result')
    # os.remove('./manifest/input_wav_test.wav')
    torch.cuda.empty_cache()
    return diari_result


print(process_diarizate("/home/ubuntu/audio-thunghiem/test_file_2h.mp3"))
