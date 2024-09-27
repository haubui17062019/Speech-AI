import json


with open("manifest_train_full.json", "r") as f:
    list_line = f.readlines()

max_duration = 0
for line in list_line:
    meta = json.loads(line[:-1])
    if meta["duration"] > max_duration:
        max_duration = meta["duration"]

print(max_duration)