#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

import torch
from torch import nn
from torch.nn.functional import conv1d, pad


class ConvLabelSmoothingLoss(nn.Module):
    """Label-smoothing loss extended with a convolution kernel applied to
    specific labels to smooth the target distribution.

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
        apply_weights=True,
        kernel=None,
        start_idx=None,
        end_idx=None,
        weight_keys=None,
        weight_values=None,
    ):
        """Construct an LabelSmoothingLoss object."""
        super(ConvLabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length
        self.apply_weights = apply_weights
        self.kernel = kernel
        if self.kernel is not None:
            _npad = kernel.size(-1) // 2
            self.padding = (_npad, _npad)
            self.start_idx = start_idx
            self.end_idx = end_idx
        self.weight_keys = weight_keys
        self.weight_values = weight_values

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
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            _smooth_value = self.smoothing / (self.size - 1)
            true_dist.fill_(_smooth_value)
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # timestamp smoothing
            if self.kernel is not None:
                _part = true_dist[:, self.start_idx : self.end_idx + 1]
                _part_padded = pad(_part.unsqueeze(1), self.padding)
                _kernel = self.kernel.to(x.device)
                _part_smoothed = conv1d(_part_padded, _kernel).squeeze(1)
                _norm_factors = (
                    _part.sum(dim=1) / _part_smoothed.sum(dim=1)
                ).unsqueeze(1)
                _final_part = _part_smoothed * _norm_factors
                true_dist[:, self.start_idx : self.end_idx + 1] = _final_part
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        # apply weight based on token type
        if self.apply_weights:
            _weights = torch.zeros((kl.size(0),), dtype=torch.float32, device=kl.device)
            _keys = torch.tensor(self.weight_keys, dtype=torch.int64, device=kl.device)
            _values = torch.tensor(
                self.weight_values, dtype=torch.float32, device=kl.device
            )
            _weights = torch.where(
                target.unsqueeze(1) == _keys, _values, torch.tensor(0.0)
            ).sum(dim=1)
            _weights = torch.where(_weights == 0, torch.tensor(1.0), _weights)
            # _weights.fill_(1.0)
            # _weights.scatter_(0, _keys, _values)
            kl = kl * _weights.unsqueeze(1)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
