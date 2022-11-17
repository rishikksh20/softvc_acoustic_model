import torch
import torchaudio
import numpy as np
import time
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from acoustic.model import AcousticModel

hifigan = torch.hub.load("/home/rishikesh/Dev/soft-vc/hifigan", "hifigan_hubert_soft", source="local").cpu()

acoustic = AcousticModel(use_gst=True).cpu()
checkpoint = torch.load("model-41000.pt", map_location=torch.device('cpu'))
consume_prefix_in_state_dict_if_present(checkpoint["acoustic-model"], "module.")
acoustic.load_state_dict(checkpoint["acoustic-model"])
acoustic.eval()

hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cpu()

def synthesis(audio):
    wav, sr = torchaudio.load(audio)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.unsqueeze(0).cpu()
    with torch.inference_mode():
        units = hubert.units(wav)
        start_time = time.time()
        mel = acoustic.generate(units)
        mel = mel.transpose(1, 2)
        # new_wav, sr = hifigan.generate(mel)
        new_wav = hifigan(mel)
        new_wav = new_wav.squeeze(0).cpu()
    torchaudio.save("out.wav", new_wav, 16000)

synthesis("in.wav")