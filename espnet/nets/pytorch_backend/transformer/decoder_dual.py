#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Copyright 2019 Hang Le (hangtp.le@gmail.com)
# https://github.com/formiel/speech-translation/blob/8ee06ad97e083918331e208ddd6096a5b8678fc6/espnet/nets/pytorch_backend/transformer/decoder_dual.py

"""Dual Decoder definition."""

import logging

from typing import Any
from typing import List
from typing import Tuple

import torch
from torch import nn

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer_dual import DualDecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "output_norm.", prefix + "after_norm.", state_dict)


class DualDecoder(BatchScorerInterface, torch.nn.Module):
    """Transfomer Dual decoder module.

    Args:
        vocab_size_src (int): Output dimension of ASR.
        vocab_size_tgt (int): Output dimension of ST/subs.
        self_attention_layer_type (str): Self-attention layer type.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        conv_wshare (int): The number of kernel of convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]): Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in self_attention_layer_type
            == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): Whether to use bias in convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        self_attention_dropout_rate (float): Dropout rate in self-attention.
        src_attention_dropout_rate (float): Dropout rate in source-attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        use_output_layer (bool): Whether to use output layer.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        vocab_size_tgt: int,
        vocab_size_src: int,
        selfattention_layer_type: str = "selfattn",
        attention_dim: int = 256,
        attention_heads: int = 4,
        conv_wshare: int = 4,
        conv_kernel_length: int = 11,
        conv_usebias: bool = False,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        cross_operator=None,
        cross_weight_learnable: bool = False,
        cross_weight: float = 0.0,
        cross_self: bool = False,
        cross_src: bool = False,
        cross_to_asr: bool = True,
        cross_to_st: bool = True,
    ):
        """Construct a Decoder object."""
        torch.nn.Module.__init__(self)
        self._register_load_state_dict_pre_hook(_pre_hook)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size_tgt, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.embed_asr = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size_src, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size_tgt, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.embed_asr = torch.nn.Sequential(
                torch.nn.Linear(vocab_size_src, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.embed_asr = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("decoder self-attention layer type = self-attention")
            decoder_selfattn_layer = MultiHeadedAttention
            decoder_selfattn_layer_args = [
                (
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                )
            ] * num_blocks
        elif selfattention_layer_type == "lightconv":
            logging.info("decoder self-attention layer type = lightweight convolution")
            decoder_selfattn_layer = LightweightConvolution
            decoder_selfattn_layer_args = [
                (
                    conv_wshare,
                    attention_dim,
                    self_attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),
                    True,
                    conv_usebias,
                )
                for lnum in range(num_blocks)
            ]
        elif selfattention_layer_type == "lightconv2d":
            logging.info(
                "decoder self-attention layer "
                "type = lightweight convolution 2-dimensional"
            )
            decoder_selfattn_layer = LightweightConvolution2D
            decoder_selfattn_layer_args = [
                (
                    conv_wshare,
                    attention_dim,
                    self_attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),
                    True,
                    conv_usebias,
                )
                for lnum in range(num_blocks)
            ]
        elif selfattention_layer_type == "dynamicconv":
            logging.info("decoder self-attention layer type = dynamic convolution")
            decoder_selfattn_layer = DynamicConvolution
            decoder_selfattn_layer_args = [
                (
                    conv_wshare,
                    attention_dim,
                    self_attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),
                    True,
                    conv_usebias,
                )
                for lnum in range(num_blocks)
            ]
        elif selfattention_layer_type == "dynamicconv2d":
            logging.info(
                "decoder self-attention layer type = dynamic convolution 2-dimensional"
            )
            decoder_selfattn_layer = DynamicConvolution2D
            decoder_selfattn_layer_args = [
                (
                    conv_wshare,
                    attention_dim,
                    self_attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),
                    True,
                    conv_usebias,
                )
                for lnum in range(num_blocks)
            ]

        self.dual_decoders = repeat(
            num_blocks,
            lambda lnum: DualDecoderLayer(
                attention_dim, attention_dim,
                decoder_selfattn_layer(*decoder_selfattn_layer_args[lnum]),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                decoder_selfattn_layer(*decoder_selfattn_layer_args[lnum]),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate=dropout_rate,
                cross_self_attn=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate) if (
                            cross_self and cross_to_st) else None,
                cross_self_attn_asr=MultiHeadedAttention(attention_heads, attention_dim,
                                                         self_attention_dropout_rate) if (
                            cross_self and cross_to_asr) else None,
                cross_src_attn=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate) if (
                            cross_src and cross_to_st) else None,
                cross_src_attn_asr=MultiHeadedAttention(attention_heads, attention_dim,
                                                        self_attention_dropout_rate) if (
                            cross_src and cross_to_asr) else None,
                normalize_before=normalize_before,
                concat_after=concat_after,
                cross_operator=cross_operator,
                cross_weight_learnable=cross_weight_learnable,
                cross_weight=cross_weight,
                cross_to_asr=cross_to_asr,
                cross_to_st=cross_to_st,
            ),
        )
        self.selfattention_layer_type = selfattention_layer_type
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
            self.after_norm_asr = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size_tgt)
            self.output_layer_asr = torch.nn.Linear(attention_dim, vocab_size_src)
        else:
            self.output_layer = None
            self.output_layer_asr = None

    def forward(self, tgt, tgt_mask, tgt_asr, tgt_mask_asr,
                memory, memory_mask, cross_mask, cross_mask_asr,
                cross_self=False, cross_src=False,
                cross_self_from="before-self", cross_src_from="before-src"):
        """Forward decoder.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out) if
                input_layer == "embed". In the other case, input tensor
                (#batch, maxlen_out, odim).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).

        Returns:
            torch.Tensor: Decoded token score before softmax (#batch, maxlen_out, odim)
                   if use_output_layer is True. In the other case,final block outputs
                   (#batch, maxlen_out, attention_dim).
            torch.Tensor: Score mask before softmax (#batch, maxlen_out).

        """
        x = self.embed(tgt)
        x_asr = self.embed_asr(tgt_asr)

        x, tgt_mask, x_asr, tgt_mask_asr, memory, memory_mask, _, _, _, _, _, _ = self.dual_decoders(
            x, tgt_mask, x_asr, tgt_mask_asr, memory, memory_mask, cross_mask, cross_mask_asr,
            cross_self, cross_src, cross_self_from, cross_src_from)

        if self.normalize_before:
            x = self.after_norm(x)
            x_asr = self.after_norm_asr(x_asr)
        if self.output_layer is not None:
            x = self.output_layer(x)
            x_asr = self.output_layer_asr(x_asr)
        return x, tgt_mask, x_asr, tgt_mask_asr

    def forward_one_step(self, tgt, tgt_mask, tgt_asr, tgt_mask_asr, memory,
                         cross_mask=None, cross_mask_asr=None, cross_self=False, cross_src=False,
                         cross_self_from="before-self", cross_src_from="before-src",
                         cache=None, cache_asr=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (batch, maxlen_out, odim).
            List[torch.Tensor]: List of cache tensors of each decoder layer.

        """
        x = self.embed(tgt)
        x_asr = self.embed_asr(tgt_asr)
        if cache is None:
            cache = [None] * len(self.dual_decoders)
        if cache_asr is None:
            cache_asr = [None] * len(self.dual_decoders)
        new_cache = []
        new_cache_asr = []
        for c, c_asr, dual_decoder in zip(cache, cache_asr, self.dual_decoders):
            x, tgt_mask, x_asr, tgt_mask_asr, memory, _, _, _, _, _, _, _ = dual_decoder(
                x, tgt_mask, x_asr, tgt_mask_asr, memory, None,
                cross_mask, cross_mask_asr, cross_self, cross_src,
                cross_self_from, cross_src_from,
                cache=c, cache_asr=c_asr)
            new_cache.append(x)
            new_cache_asr.append(x_asr)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
            y_asr = self.after_norm_asr(x_asr[:, -1])
        else:
            y = x[:, -1]
            y_asr = x_asr[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
            y_asr = torch.log_softmax(self.output_layer_asr(y_asr), dim=-1)

        return y, new_cache, y_asr, new_cache_asr

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if self.selfattention_layer_type != "selfattn":
            # TODO(karita): implement cache
            logging.warning(
                f"{self.selfattention_layer_type} does not support cached decoding."
            )
            state = None
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.dual_decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
