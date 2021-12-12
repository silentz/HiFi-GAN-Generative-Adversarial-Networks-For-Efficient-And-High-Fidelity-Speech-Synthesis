import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResBlock(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: int,
                       dilations: Tuple[int, int],
                       negative_slope: float = 0.1):
        super().__init__()

        pad = lambda dil: (kernel_size * dil - dil) // 2
        self._negative_slope = negative_slope

        self._conv_1 = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilations[0],
                padding=pad(dilations[0]),
            )

        self._conv_2 = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilations[1],
                padding=pad(dilations[1]),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stage_1 = F.leaky_relu(input, negative_slope=self._negative_slope)
        stage_1 = self._conv_1(stage_1)
        stage_1 = input + stage_1

        stage_2 = F.leaky_relu(stage_1, negative_slope=self._negative_slope)
        stage_2 = self._conv_2(stage_2)
        stage_2 = stage_1 + stage_2

        return stage_2


class MRFBlock(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_sizes: List[int],
                       dilations: List[Tuple[int, int]]):
        super().__init__()

        assert len(kernel_sizes) == len(dilations)
        self._n_blocks = len(kernel_sizes)

        self._layers = nn.ModuleList([
                ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=k_size,
                        dilations=dil,
                    )
                for (k_size, dil) in zip(kernel_sizes, dilations)
            ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = [layer(input) for layer in self._layers]
        result = sum(outputs) / self._n_blocks
        return result


class Generator(nn.Module):

    def __init__(self, n_mels: int = 80,
                       initial_channels: int = 256,
                       upsample_rates: List[int] = [8, 8, 4],
                       upsample_kernel_sizes: List[int] = [16, 16, 8],
                       mrf_kernel_sizes: List[int] = [3, 5, 7],
                       mrf_dilations: List[Tuple[int, int]] = [(1, 2), (2, 6), (3, 12)],
                       negative_slope: float = 0.1):
        super().__init__()
        assert len(upsample_rates) == len(upsample_kernel_sizes)
        n_layers = len(upsample_rates)

        self._prenet = nn.Sequential(
                nn.Conv1d(
                        in_channels=n_mels,
                        out_channels=initial_channels,
                        kernel_size=7,
                        padding='same',
                    ),
            )

        self._layers = nn.ModuleList([
                nn.Sequential(
                    nn.LeakyReLU(
                            negative_slope=negative_slope,
                        ),
                    nn.ConvTranspose1d(
                            in_channels=initial_channels // (2 ** idx),
                            out_channels=initial_channels // (2 ** (idx + 1)),
                            kernel_size=upsample_kernel_sizes[idx],
                            stride=upsample_rates[idx],
                            padding=(upsample_kernel_sizes[idx] - upsample_rates[idx]) // 2,
                        ),
                    MRFBlock(
                            in_channels=initial_channels // (2 ** (idx + 1)),
                            out_channels=initial_channels // (2 ** (idx + 1)),
                            kernel_sizes=mrf_kernel_sizes,
                            dilations=mrf_dilations,
                        ),
                ) for idx in range(n_layers)
            ])

        self._postnet = nn.Sequential(
                nn.LeakyReLU(
                        negative_slope=negative_slope,
                    ),
                nn.Conv1d(
                        in_channels=initial_channels // (2 ** n_layers),
                        out_channels=1,
                        kernel_size=7,
                        padding='same',
                    ),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        X = self._prenet(input)
        for layer in self._layers:
            X = layer(X)
        X = self._postnet(X)
        X = torch.tanh(X)
        return X


class PeriodDiscriminator(nn.Module):

    def __init__(self, period: int,
                       kernel_size: int = 5,
                       stride: int = 3,
                       negative_slope: float = 0.1):
        super().__init__()
        self._period = period

        self._layers = nn.ModuleList([
                nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=32,
                                  kernel_size=(kernel_size, 1), stride=(stride, 1), padding=2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                    ),
                nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=128,
                                  kernel_size=(kernel_size, 1), stride=(stride, 1), padding=2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                    ),
                nn.Sequential(
                        nn.Conv2d(in_channels=128, out_channels=512,
                                  kernel_size=(kernel_size, 1), stride=(stride, 1), padding=2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                    ),
                nn.Sequential(
                        nn.Conv2d(in_channels=512, out_channels=1024,
                                  kernel_size=(kernel_size, 1), stride=(stride, 1), padding=2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                    ),
                nn.Sequential(
                        nn.Conv2d(in_channels=1024, out_channels=1024,
                                  kernel_size=(kernel_size, 1), stride=(stride, 1), padding=2),
                        nn.LeakyReLU(negative_slope=negative_slope),
                    ),
            ])

        self._postnet = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1,
                          kernel_size=(3, 1), stride=1, padding=1),
            )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # pad input
        batch_size, channels, length = input.shape
        n_padded = (self._period - (length % self._period)) % self._period
        X = F.pad(input, (0, n_padded), mode='reflect')
        length = length + n_padded

        # reshape
        X = X.view(batch_size, channels, length // self._period, self._period)

        # forward
        inter_out = []

        for layer in self._layers:
            X = layer(X)
            inter_out.append(X)

        X = self._postnet(X)
        inter_out.append(X)

        X = torch.flatten(X, 1, -1)
        return X, inter_out

