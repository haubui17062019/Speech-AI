import soundfile as sf
import librosa
import json
import os


def resample_audio(input_file_path, target_sampling_rate = 16000):
    """Resample a single audio file.

    Args:
        input_file_path (str): Path to the input audio file.
        target_sampling_rate (int): Sampling rate for output audio file.

    Returns:
        No explicit returns
    """

    if not input_file_path.endswith(".wav"):
        raise NotImplementedError("Loading only implemented for wav files.")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Cannot file input at {input_file_path}")
    audio, sampling_rate = librosa.load(input_file_path, sr=target_sampling_rate)
    duration = librosa.get_duration(y=audio, sr=sampling_rate)
    if duration == 0:
        print(f'0 duration audio at {input_file_path}')
        return None
    sf.write(input_file_path, audio, samplerate=target_sampling_rate, format="wav")
    return input_file_path, duration


def build_manifest(path_txt, path_manifest):
    with open(path_txt, "r") as fin:
        with open(path_manifest, "w") as fout:
            for ix, line in enumerate(fin):
                path_audio, text = line[:-1].split("\t")
                path_audio = os.path.join(path_folder_data, path_audio)
                for char in ['"', '.', '!', '?', '-', '*', '&', '#', '@', '(', ')', '<', '>', '[', ']', '{', '}',
                             '+', "'", ',', ':', '|', '/']:
                    text = text.replace(char, "")
                text = text.lower()

                if ix % 10 == 0:
                    print(text)

                path_audio, duration = resample_audio(path_audio, 16000)
                metadata = {"audio_filepath": path_audio, "text": text, "duration": duration}
                json.dump(metadata, fout)
                fout.write("\n")


if __name__ == "__main__":
    path_txt = "/home1/data/haubui/Speech-AI/TTS-Riva/dataset/meta_data.tsv"
    path_manifest = "/home1/data/haubui/Speech-AI/TTS-Riva/dataset/manifest.json"
    path_folder_data = "/home1/data/haubui/Speech-AI/TTS-Riva/dataset"

    build_manifest(path_txt, path_manifest)