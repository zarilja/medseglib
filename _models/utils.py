import hashlib
import os
from urllib.request import urlopen, Request

import torch

CHUNK_SIZE = 8192


def download_weights_to_model(model: torch.nn.Module, url: str, hash_: str):
    sha256 = hashlib.sha256()
    req = Request(url)
    u = urlopen(req)


    if os.path.exists(f"weights/{hash_}.pt"):
        model.load_state_dict(torch.load(f"weights/{hash_}.pt", map_location=torch.device("cpu")))
        return

    os.makedirs("weights", exist_ok=True)
    with open(f"weights/{hash_}.pt", "wb") as f:
        while True:
            buffer = u.read(CHUNK_SIZE)
            if len(buffer) == 0:
                break
            f.write(buffer)
            sha256.update(buffer)
        digest = sha256.hexdigest()
        if digest != hash_:
            raise RuntimeError(f'invalid hash value (expected "{hash_}", got "{digest}")')

    model.load_state_dict(torch.load(f"weights/{hash_}.pt", map_location=torch.device("cpu")))
