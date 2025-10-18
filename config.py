import json
import torch
import os

# load config.json
with open("config.json", "r") as f:
    config = json.load(f)

# set device automatically if set to "auto"
if config["system"]["device"] == "auto":
    config["system"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

for key, path in config["data"].items():
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def get_config():
    return config
