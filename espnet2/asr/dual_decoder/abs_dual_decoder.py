from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from espnet.nets.scorer_interface import ScorerInterface


class AbsDualDecoder(torch.nn.Module, ScorerInterface, ABC):
    @abstractmethod
    def forward(
        self,
        hs_pad: torch.Tensor,  # encoder output
        hlens: torch.Tensor,
        ys_in_pad_asr: torch.Tensor,
        ys_in_lens_asr: torch.Tensor,
        ys_in_pad_mt: torch.Tensor,
        ys_in_lens_mt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
