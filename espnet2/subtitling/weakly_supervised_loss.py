#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Weakly Supervised Loss.
Inspired by: Word Order Does Not Matter For Speech Recognition
        but subword vocabulary (no <unk>) and no prior probability on <blank> token
"""


import torch
from torch import nn


class WeaklySupervisedLoss(nn.Module):
    """Weakly Supervised loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(WeaklySupervisedLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)

        with torch.no_grad():
            ignore = target == self.padding_idx
            target = target.masked_fill(ignore, 0)  # [B, T]
            target_lens = (~ignore).sum(dim=1)  # [B]
            total = len(target.view(-1)) - ignore.sum().item()

            true_dist = x.clone()  # [B, T, V]
            true_dist.fill_(self.smoothing / (self.size - 1))
            true_dist.scatter_(2, target.unsqueeze(2), self.confidence)  # [B, T, V]
            true_dist = torch.div(true_dist.sum(dim=1), target_lens.unsqueeze(1))

        logprobs = torch.log_softmax(x, dim=2)
        masked_logprobs = logprobs.masked_fill(ignore.unsqueeze(2), -float('inf'))
        collapsed_probs = torch.logsumexp(masked_logprobs, dim=1) - target_lens.unsqueeze(1)
        kl = self.criterion(collapsed_probs, true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.sum() / denom
