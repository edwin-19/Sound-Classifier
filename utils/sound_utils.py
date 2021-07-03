import torchaudio
import torch
import numpy as np

from typing import Tuple

class SoundUtil(object):
    def __init__(self):
        pass
    
    def load_sound(self, sound_path) -> Tuple[torch.tensor, ]:
        return torchaudio.load(sound_path)
    
    def convert_sound(self, wav_path):
        pass