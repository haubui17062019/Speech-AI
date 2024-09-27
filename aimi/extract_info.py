from mutagen.mp3 import MP3
from mutagen.wave import WAVE
import glob
import os
import librosa
import soundfile as sf


def extract_info_audios(path_folder, path_save_txt):
    list_path_audio = glob.glob(f"{path_folder}/*")

    with open(path_save_txt, "w+") as f:
        for path_audio in list_path_audio:
            fileID = os.path.basename(path_audio).split(".")[0]
            if path_audio.endswith("mp3"):
                sample_rate = MP3(path_audio).info.sample_rate
                duration = MP3(path_audio).info.length
            elif path_audio.endswith("wav"):
                sample_rate = WAVE(path_audio).info.sample_rate
                duration = WAVE(path_audio).info.length
            else:
                raise 'Do not format!'
            f.write(f"{fileID}\t{sample_rate}\t{duration}\n")


def convert_to_wav(path_folder, path_folder_save):
    list_path_audio = glob.glob(f"{path_folder}/*")
    os.makedirs(path_folder_save, exist_ok=True)

    for path_audio in list_path_audio:
        fileID = os.path.basename(path_audio).split(".")[0]
        audio, sr = librosa.load(path_audio, sr=16000)

        path_file_save = f"{path_folder_save}/{fileID}.wav"
        sf.write(path_file_save, audio, sr)


if __name__ == "__main__":
    path_folder = "./common_voice/mp3"
    path_save_txt = "./common_voice/ex2_1.txt"
    path_folder_save = "./common_voice/wav"
    extract_info_audios(path_folder, path_save_txt)
    convert_to_wav(path_folder, path_folder_save)