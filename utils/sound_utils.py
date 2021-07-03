import torchaudio
import torch
import numpy as np
import random

from typing import Tuple

from torchaudio.transforms import Resample

class SoundUtil(object):
    def load_sound(self, sound_path) -> Tuple[torch.Tensor, int]:
        return torchaudio.load(sound_path)
    
    def rechannel(self, aud, new_channel):
        sig, sr = aud
        
        if sig.shape[0] == new_channel:
            return aud
        
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])
            
        return resig, sr
    
    def resample(self, aud, new_sr=44100) -> Tuple[torch.Tensor, int]:
        sig, sr = aud
        
        if sr == new_sr:
            return aud
        
        num_channels = sig.shape[0]
        resig = torchaudio.transforms(Resample, new_sr)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, new_sr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
            
        return resig, new_sr
    
    def pad_truc(self, aud, max_ms=8000) -> Tuple[torch.Tensor, int]:
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        
        if sig_len > max_len:
            # Truncate
            sig = sig[:, :max_len] 
        elif sig_len < max_len:
            # Pad
            pad_begin_len = np.random.randint(0, max_len - sig_len, (1,))
            pad_end_len = max_len - sig_len - pad_begin_len
            
            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            
            sig = torch.cat((pad_begin, sig, pad_end), 1)
            
        return sig, sr
    
    def convert_melspectogram(self, aud, n_mels=64, n_fft=1024, hop_len=None, top_db=80) -> torch.Tensor:
        sig, sr = aud
        
        spec = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
        )(sig)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(sig)
        
        return spec
    
    # Augment Part
    def time_shif(self, aud, shift_limit) -> Tuple[torch.tensor, int]:
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        
        return sig.roll(shift_amt), sr
    
    def spectro_augment(self, spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
            
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
            
        return aug_spec
    
    def convert_sound(
        self, wav_path, sr=44100,
        channel=2, shift_pct=0.4,
        duration=8000, n_mel=64, n_fft=1024, hop_len=None,
        is_augment=False
    ) -> torch.Tensor:
        aud = self.load_sound(wav_path)
        
        reaud = self.resample(aud, sr)
        rechan = self.rechannel(reaud, channel)
        dur_aud = self.pad_truc(rechan, duration)
        
        if is_augment:
            dur_aud = self.time_shif(dur_aud, shift_pct)
        
        mel_spec = self.convert_melspectogram(dur_aud, n_mels=n_mel, n_fft=n_fft, hop_len=hop_len)
        
        if is_augment:
            mel_spec = self.spectro_augment(
                mel_spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
            )
        
        return mel_spec