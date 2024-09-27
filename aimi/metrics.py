from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import glob
import os
from jiwer import cer, wer
import statistics


# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def infer(path_folder_audio, path_test_txt):
    print('[INFO] Start')
    list_wer, list_cer = [], []
    with open(path_test_txt, "r") as fin:
        for line in fin:
            fileID, reference = line[:-1].split("\t")
            reference = reference.lower()
            audio_path = f"{path_folder_audio}/{fileID}.wav"
            ds = map_to_array({
                "file": audio_path
            })
            # tokenize
            input_values = processor(ds["speech"], return_tensors="pt", padding="longest").input_values  # Batch size 1
            # retrieve logits
            logits = model(input_values.to(device)).logits
            # take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            print('==================')
            print('[INFO] reference: ', reference)
            print('[INFO] transcript: ', transcription)

            list_cer.append(cer(reference, transcription))
            list_wer.append(wer(reference, transcription))
    print('==========================')
    print('[INFO] CER: ', statistics.mean(list_cer))
    print('[INFO] WER: ', statistics.mean(list_wer))
    with open("./common_voice/result.txt", "w+") as f:
        f.write(f'CER: {statistics.mean(list_cer)}\n')
        f.write(f'WER: {statistics.mean(list_wer)}\n')


if __name__ == "__main__":
    path_folder_audio = "./common_voice/wav"
    path_test_txt = "./common_voice/test.txt"
    # load model and tokenizer

    device = 'cuda'
    processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model.to(device)
    infer(path_folder_audio, path_test_txt)







