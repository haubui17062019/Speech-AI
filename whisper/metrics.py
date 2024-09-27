import soundfile as sf
import torch
import json
from jiwer import cer, wer
import statistics
import whisper
from transformers import pipeline



def infer():
    print('[INFO] Start')
    list_wer, list_cer = [], []
    with open('/home2/haubui/Speech-AI/ASR/prepare_data/manifest_test.json', 'r') as fin:
        for line in fin:
            meta = json.loads(line)
            audio_filepath = meta['audio_filepath']
            reference = meta["text"]

            transcription_predict = transcriber.transcribe(audio_filepath)['text'].lower()
            transcription = ""
            for i in transcription_predict:
                if i not in ["'", '"', ';', ':', ',', '.', '!', '?']:
                    transcription += i

            print('==================')
            print('[INFO] reference: ', reference)
            print('[INFO] transcript: ', transcription)

            list_cer.append(cer(reference, transcription))
            list_wer.append(wer(reference, transcription))
    print('==========================')
    print('[INFO] CER: ', statistics.mean(list_cer))
    print('[INFO] WER: ', statistics.mean(list_wer))
    with open("result.txt", "w+") as f:
        f.write(f'CER: {statistics.mean(list_cer)}\n')
        f.write(f'WER: {statistics.mean(list_wer)}\n')


if __name__ == "__main__":
    device = 'cuda'
    # transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large", device='cuda')
    transcriber = whisper.load_model("medium", device="cuda")

    infer()







