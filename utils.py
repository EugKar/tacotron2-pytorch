import numpy as np
from scipy.io.wavfile import read
import torch
import librosa


def get_mask_from_lengths(lengths, max_len=None, invert=False):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, dtype=torch.long, device=lengths.device)
    if invert:
        mask = ids >= lengths.unsqueeze(1)
    else:
        mask = ids < lengths.unsqueeze(1)
    return mask


def load_wav_to_torch(full_path, sr=None):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = librosa.load(full_path, sr)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_device(x, device=None, dtype=None):
    x = x.contiguous()
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    return x.to(device=device, dtype=dtype, non_blocking=True, copy=True)
