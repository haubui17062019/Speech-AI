with open("/home1/data/haubui/Speech-AI/TTS-Riva/dataset/manifest.json", "r") as f:
    list_line = f.readlines()

for ix, line in enumerate(list_line):
    if ix < int(len(list_line) * 0.2):
        with open("/home1/data/haubui/Speech-AI/TTS-Riva/dataset/manifest_val_1.json", "a") as f_val:
            f_val.write(line)
    else:
        with open("/home1/data/haubui/Speech-AI/TTS-Riva/dataset/manifest_train_1.json", "a") as f_train:
            f_train.write(line)

