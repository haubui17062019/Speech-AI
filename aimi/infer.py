from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import glob
import os
from jiwer import cer, wer
import statistics
from mutagen.wave import WAVE


# define function to read in sound file
def map_to_array(list_audio):
    batch = {
        "speech": []
    }
    for audio in list_audio:
        speech, _ = sf.read(audio)
        batch["speech"].append(speech)
    return batch


def infer(path_folder_audio, path_test_txt):
    print('[INFO] Start')
    with open(path_test_txt, "r") as fin:
        list_batch_6 = []
        list_batch_4 = []
        for line in fin:
            fileID, reference = line[:-1].split("\t")
            audio_path = f"{path_folder_audio}/{fileID}.wav"
            duration = WAVE(audio_path).info.length
            if duration <= 4:
                list_batch_6.append(audio_path)
            else:
                list_batch_4.append(audio_path)

            if len(list_batch_6) == 6:
                ds = map_to_array(list_batch_6)
                list_batch_6 = []
                # tokenize
                input_values = processor(ds["speech"], return_tensors="pt",
                                         padding="longest").input_values
                # retrieve logits
                logits = model(input_values.to(device)).logits
                # take argmax and decode
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)

                print('==================')
                print('[INFO] transcript batch 6: ', transcription)

            if len(list_batch_4) == 4:
                ds = map_to_array(list_batch_4)
                list_batch_4 = []

                # tokenize
                input_values = processor(ds["speech"], return_tensors="pt", padding="longest").input_values
                # retrieve logits
                logits = model(input_values.to(device)).logits
                # take argmax and decode
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)

                print('==================')
                print('[INFO] transcript batch 4: ', transcription)


if __name__ == "__main__":
    path_folder_audio = "./common_voice/wav"
    path_test_txt = "./common_voice/test.txt"
    # load model and tokenizer
    device = 'cuda'
    processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model.to(device)
    infer(path_folder_audio, path_test_txt)







