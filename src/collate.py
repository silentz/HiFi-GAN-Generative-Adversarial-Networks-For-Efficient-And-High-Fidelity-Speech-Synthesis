# from: https://github.com/markovka17/dla/blob/2021/hw3_tts/aligner.ipynb

import torch
from torch.nn.utils.rnn import pad_sequence

import copy
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor

    def to(self, device: torch.device) -> 'Batch':
        return Batch(
                waveform=self.waveform.to(device),
                waveform_length=self.waveform_length.to(device),
                transcript=copy.deepcopy(self.transcript),
                tokens=self.tokens.to(device),
                token_lengths=self.token_lengths.to(device),
            )


class LJSpeechCollator:

    def __call__(self, instances: List[Any]) -> Batch:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )
        transcript = list(transcript)

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(waveform, waveform_length, transcript, tokens, token_lengths)
