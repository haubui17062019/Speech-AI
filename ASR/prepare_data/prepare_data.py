import glob
import json
import librosa
from sklearn.utils import shuffle


def vlsp_build_manifest(folder_data):
    """ Build vlsp2020 data asr """
    
    lst_file_audio = glob.glob(f"{folder_data}/*.wav")
    lst_file_audio = shuffle(lst_file_audio)

    total_file_audio = len(lst_file_audio)

    for ix, file_audio in enumerate(lst_file_audio):
        
        file_transcript = file_audio.split(".wav")[0] + ".txt"
        
        with open(file_transcript, 'r') as f:
            transcript = f.read()
        
        duration = librosa.core.get_duration(filename=file_audio)

        metadata = {"audio_filepath": file_audio, "duration": duration, "text": transcript}

        if ix < int(total_file_audio * 0.2):
            with open("./prapare_data/test_manifest.json", 'a') as f:
                json.dump(metadata, f)
                f.write("\n")
        else:
            with open("./prapare_data/train_manifest.json", 'a') as fout:
                json.dump(metadata, fout)
                fout.write("\n")


if __name__ == "__main__":
    folder_data = "/home1/data/haubui/Speech-AI/ASR/prapare_data/vlsp2020_train_set_02"
    vlsp_build_manifest(folder_data)






