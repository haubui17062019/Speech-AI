import librosa
import json


def build_manifest(path_txt, manifest_path):
    with open(path_txt, "r") as fin:
        with open(manifest_path, "w") as fout:
            for ix, line in enumerate(fin):
                try:
                    # if ix > 7500 * 9:
                    #     exit(1)
                    line = line[:-1]
                    path_audio, transcript = line.split(",")
                    path_audio = path_audio.replace("audio", "train")
                    duration = librosa.core.get_duration(filename=path_audio)

                    # write the metadata to manifest
                    metadata = {"audio_filepath": path_audio, "duration": duration, "text": transcript}
                    json.dump(metadata, fout)
                    fout.write("\n")
                except Exception as ve:
                    print('[ERROR] ', str(ve))


if __name__ == "__main__":
    # path_txt_test = "/home2/tuyentran/bud500/test.csv"
    path_txt_train = "/home2/tuyentran/bud500/train.csv"
    #
    # manifest_test = "./prepare_data/manifest_test.json"
    manifest_train = "./prepare_data/manifest_train_full.json"
    #
    # # build manifest file
    # build_manifest(path_txt_test, manifest_test)
    build_manifest(path_txt_train, manifest_train)

    # path_txt_val = "/home2/tuyentran/bud500/valid.csv"
    # manifest_val = "./prepare_data/manifest_val.json"
    # build_manifest(path_txt_val, manifest_val)





