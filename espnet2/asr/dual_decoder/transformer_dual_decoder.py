# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import torch
from typeguard import check_argument_types

import logging

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer_dual import DualDecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import create_cross_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorer_interface import BatchDualScorerInterface
from espnet2.asr.dual_decoder.abs_dual_decoder import AbsDualDecoder
from espnet2.asr.decoder.transformer_decoder import BaseTransformerDecoder

class DualTransformerDecoder(AbsDualDecoder, BatchDualScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size_src: output dim ASR
        vocab_size_tgt: output dim SUBS
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size_src: int,
        vocab_size_tgt: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = True,
        cross_operator: str = None,
        cross_weight_learnable: bool = False,
        cross_weight: float = 0.0,
        cross_self: bool = False,
        cross_src: bool = False,
        cross_to_asr: bool = True,
        cross_to_st: bool = True,
        wait_k_asr: int = 0,
        wait_k_st: int = 0,
        cross_src_from: str = "embedding",
        cross_self_from: str = "embedding",
        ignore_id: int = -1,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

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
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
            self.after_norm_asr = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size_tgt)
            self.output_layer_asr = torch.nn.Linear(attention_dim, vocab_size_src)
        else:
            self.output_layer = None
            self.output_layer_asr = None

        self.dual_decoders = repeat(
            num_blocks,
            lambda lnum: DualDecoderLayer(
                attention_dim, attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
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

        self.cross_operator = cross_operator
        self.cross_weight_learnable = cross_weight_learnable
        self.cross_weight = cross_weight
        self.cross_self = cross_self
        self.cross_src = cross_src
        self.cross_to_asr = cross_to_asr
        self.cross_to_st = cross_to_st
        self.wait_k_asr = wait_k_asr
        self.wait_k_st = wait_k_st
        self.cross_src_from = cross_src_from
        self.cross_self_from = cross_self_from
        self.ignore_id = ignore_id

        # Check parameters
        if self.cross_operator == 'sum' and self.cross_weight <= 0:
            assert (not self.cross_to_asr) and (not self.cross_to_st)
        if self.cross_to_asr or self.cross_to_st:
            assert self.cross_self or self.cross_src
        assert bool(self.cross_operator) == (self.cross_to_asr or self.cross_to_st)
        if self.cross_src_from != "embedding" or self.cross_self_from != "embedding":
            assert self.normalize_before
        if self.wait_k_asr > 0:
            assert self.wait_k_st == 0
        elif self.wait_k_st > 0:
            assert self.wait_k_asr == 0
        else:
            assert self.wait_k_asr == 0
            assert self.wait_k_st == 0

        logging.info("*** Parallel attention parameters ***")
        if self.cross_to_asr:
            logging.info("| Cross to ASR")
        if self.cross_to_st:
            logging.info("| Cross to ST")
        if self.cross_self:
            logging.info("| Cross at Self")
        if self.cross_src:
            logging.info("| Cross at Source")
        if self.cross_to_asr or self.cross_to_st:
            logging.info(f'| Cross operator: {self.cross_operator}')
        logging.info(f'| Cross sum weight: {self.cross_weight}')
        if self.cross_src:
            logging.info(f'| Cross source from: {self.cross_src_from}')
        if self.cross_self:
            logging.info(f'| Cross self from: {self.cross_self_from}')
        logging.info(f'| wait_k_asr = {self.wait_k_asr}')
        logging.info(f'| wait_k_st = {self.wait_k_st}')

        if (self.cross_src_from != "embedding" and self.cross_src) and (not self.normalize_before):
            logging.warning(f'WARNING: Resort to using self.cross_src_from == embedding for cross at source attention.')
        if (self.cross_self_from != "embedding" and self.cross_self) and (not self.normalize_before):
            logging.warning(f'WARNING: Resort to using self.cross_self_from == embedding for cross at self attention.')

    def forward(
        self,
        hs_pad: torch.Tensor,  # encoder output
        hlens: torch.Tensor,
        ys_in_pad_asr: torch.Tensor,
        ys_in_lens_asr: torch.Tensor,
        ys_in_pad_mt: torch.Tensor,
        ys_in_lens_mt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        tgt_mt = ys_in_pad_mt
        # tgt_mask: (B, 1, L)
        tgt_mask_mt = (~make_pad_mask(ys_in_lens_mt)[:, None, :]).to(tgt_mt.device)
        # m: (1, L, L)
        m_mt = subsequent_mask(tgt_mask_mt.size(-1),
                               device=tgt_mask_mt.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask_mt = tgt_mask_mt & m_mt

        tgt_asr = ys_in_pad_asr
        # tgt_mask: (B, 1, L)
        tgt_mask_asr = (~make_pad_mask(ys_in_lens_asr)[:, None, :]).to(tgt_asr.device)
        # m: (1, L, L)
        m_asr = subsequent_mask(tgt_mask_asr.size(-1),
                                device=tgt_mask_asr.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask_asr = tgt_mask_asr & m_asr

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        if self.wait_k_asr > 0:
            cross_mask_mt = create_cross_mask(ys_in_pad_mt, ys_in_pad_asr,
                                           self.ignore_id, wait_k_cross=self.wait_k_asr)
            cross_mask_asr = create_cross_mask(ys_in_pad_asr, ys_in_pad_mt,
                                               self.ignore_id, wait_k_cross=-self.wait_k_asr)
        elif self.wait_k_st > 0:
            cross_mask_mt = create_cross_mask(ys_in_pad_mt, ys_in_pad_asr,
                                           self.ignore_id, wait_k_cross=-self.wait_k_st)
            cross_mask_asr = create_cross_mask(ys_in_pad_asr, ys_in_pad_mt,
                                               self.ignore_id, wait_k_cross=self.wait_k_st)
        else:
            cross_mask_mt = create_cross_mask(ys_in_pad_mt, ys_in_pad_asr,
                                           self.ignore_id, wait_k_cross=0)
            cross_mask_asr = create_cross_mask(ys_in_pad_asr, ys_in_pad_mt,
                                               self.ignore_id, wait_k_cross=0)

        x_mt = self.embed(tgt_mt)
        x_asr = self.embed_asr(tgt_asr)

        x_mt, tgt_mask_mt, x_asr, tgt_mask_asr, memory, memory_mask, _, _, _, _, _, _ = self.dual_decoders(
            x_mt, tgt_mask_mt, x_asr, tgt_mask_asr, memory, memory_mask,
            cross_mask_mt, cross_mask_asr)

        if self.normalize_before:
            x_mt = self.after_norm(x_mt)
            x_asr = self.after_norm_asr(x_asr)
        if self.output_layer is not None:
            x_mt = self.output_layer(x_mt)
            x_asr = self.output_layer_asr(x_asr)

        #olens = tgt_mask.sum(1)
        #return x, olens

        return x_mt, tgt_mask_mt, x_asr, tgt_mask_asr


    def forward_one_step(
        self,
        tgt_asr: torch.Tensor,
        tgt_mt: torch.Tensor,
        tgt_mask_asr: torch.Tensor,
        tgt_mask_mt: torch.Tensor,
        memory: torch.Tensor,
        cross_mask_asr: torch.Tensor,
        cross_mask_mt: torch.Tensor,
        cache_asr: List[torch.Tensor] = None,
        cache_mt: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x_asr = self.embed_asr(tgt_asr)
        x_mt = self.embed(tgt_mt)

        if cache_asr is None:
            cache_asr = [None for i in range(len(self.dual_decoders))]
        if cache_mt is None:
            cache_mt = [None for i in range(len(self.dual_decoders))]

        new_cache_asr = []
        new_cache_mt = []
        for c_mt, c_asr, decoder in zip(cache_mt, cache_asr, self.dual_decoders):
            x_mt, tgt_mask_mt, x_asr, tgt_mask_asr, memory, _, \
            _, _, _, _, _, _ = decoder(x_mt, tgt_mask_mt, x_asr, tgt_mask_asr,
                                       memory, None, cross_mask_mt, cross_mask_asr,
                                       cache=c_mt, cache_asr=c_asr)
            new_cache_asr.append(x_asr)
            new_cache_mt.append(x_mt)

        if self.normalize_before:
            y_mt = self.after_norm(x_mt[:, -1])
            y_asr = self.after_norm_asr(x_asr[:, -1])

        if self.output_layer is not None:
            y_mt = torch.log_softmax(self.output_layer(y_mt), dim=-1)
            y_asr = torch.log_softmax(self.output_layer_asr(y_asr), dim=-1)

        return y_mt, new_cache_mt, y_asr, new_cache_asr

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache_asr=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, ys_asr: torch.Tensor,
            states: List[Any], states_asr: List[Any],
            xs: torch.Tensor, pos: int = True,
            use_cache: bool = True,  # False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Any], List[Any]]:
        """Score new token batch.

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
            batch_state_mt = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state_mt = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # merge states
        n_batch_asr = len(ys_asr)
        if states_asr[0] is None:
            batch_state_asr = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state_asr = [
                torch.stack([states_asr[b][i] for b in range(n_batch_asr)])
                for i in range(n_layers)
            ]

        # batch decoding
        if self.wait_k_asr > 0:
            if pos < self.wait_k_asr:
                ys_mask_mt = subsequent_mask(1, device=xs.device).unsqueeze(0)
            else:
                ys_mask_mt = subsequent_mask(pos - self.wait_k_asr + 1, device=xs.device).unsqueeze(0)
        else:
            ys_mask_mt = subsequent_mask(pos + 1, device=xs.device).unsqueeze(0)

        if self.wait_k_st > 0:
            if pos < self.wait_k_st:
                ys_mask_asr = subsequent_mask(1).unsqueeze(0)
            else:
                ys_mask_asr = subsequent_mask(pos - self.wait_k_st + 1).unsqueeze(0)
        else:
            ys_mask_asr = subsequent_mask(pos + 1).unsqueeze(0)

        cross_mask_mt = create_cross_mask(ys, ys_asr, self.ignore_id, wait_k_cross=self.wait_k_asr)
        cross_mask_asr = create_cross_mask(ys_asr, ys, self.ignore_id, wait_k_cross=self.wait_k_st)

        if not use_cache:
            batch_state_asr = None
            batch_state_mt = None

        logp_mt, states_mt, logp_asr, states_asr = self.forward_one_step(ys_asr, ys, ys_mask_asr, ys_mask_mt,
                                                                         xs, cross_mask_asr, cross_mask_mt,
                                                                         cache_asr=batch_state_asr,
                                                                         cache_mt=batch_state_mt)

        # transpose state of [layer, batch] into [batch, layer]
        state_list_asr = [[states_asr[i][b] for i in range(n_layers)] for b in range(n_batch_asr)]
        state_list_mt = [[states_mt[i][b] for i in range(n_layers)] for b in range(n_batch)]

        return logp_mt, logp_asr, state_list_mt, state_list_asr


class TransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
