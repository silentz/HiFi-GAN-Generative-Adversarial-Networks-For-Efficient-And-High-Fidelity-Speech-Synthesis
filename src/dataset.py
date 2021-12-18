import random
import torch
import torchaudio
from torch.utils.data import Dataset

import os
from typing import List


# from: https://github.com/markovka17/dla/blob/2021/hw3_tts/aligner.ipynb
class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root, download=True)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveforn_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


class CutLJSpeechDataset(LJSpeechDataset):

    def __init__(self, max_length: float, sample_rate: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.sample_rate = sample_rate

    def __getitem__(self, idx: int):
        waveform, wave_len, transcript, tokens, tokens_len = super().__getitem__(idx)

        origin_length = wave_len[0].item()
        result_length = int(self.sample_rate * self.max_length)
        result_length = min(origin_length, result_length)
        max_idx = origin_length - result_length

        start_idx = random.randint(0, max_idx)
        finish_idx = start_idx + result_length

        res_wave = waveform[:, start_idx:finish_idx]
        res_len = torch.tensor([result_length], device='cpu', dtype=torch.long)

        return res_wave, res_len, transcript, tokens, tokens_len


class TestDataset(Dataset):

    def __init__(self, root: str, files: List[str]):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, self.files[idx])
        wav, _ = torchaudio.load(path)
        wav_len = torch.tensor([len(wav[0])])
        return wav, wav_len, '', torch.tensor([[]]), torch.tensor([])


class OverfitDataset(Dataset):

    def __init__(self, dataset: Dataset,
                       num_samples: int,
                       length: int):
        self.dataset = dataset
        self.num_samples = num_samples
        self.length = length

    def __getitem__(self, idx: int):
        return self.dataset[idx % self.num_samples]

    def __len__(self):
        return self.length


class PartialDataset(Dataset):

    def __init__(self, dataset: Dataset,
                       start_idx: int,
                       finish_idx: int):
        # interval has form: [start_idx; finish_idx)
        self.dataset = dataset
        self.start_idx = start_idx
        self.finish_idx = finish_idx

    def __getitem__(self, idx: int):
        return self.dataset[idx + self.start_idx]

    def __len__(self):
        return self.finish_idx - self.start_idx
