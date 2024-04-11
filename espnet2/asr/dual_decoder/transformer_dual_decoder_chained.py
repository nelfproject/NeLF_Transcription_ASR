# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from typeguard import check_argument_types

import logging

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, mask_based_pad
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer_dual import DualDecoderLayer
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
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
from espnet2.asr.dual_decoder.abs_chained_decoder import AbsChainedDecoder
from espnet2.asr.decoder.transformer_decoder import BaseTransformerDecoder

class ChainedDualTransformerDecoder(AbsChainedDecoder, BatchScorerInterface):
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
        num_blocks_asr: int = 6,
        num_blocks_mt: int = 3,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = True,
        chain_mode: str = "cascade",
        cross_from: str = "decoder",
        separate_embedding: bool = False,
        merge_mode: str = "sum",
        ignore_id: int = -1,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        assert (chain_mode in ['cascade', 'triangle']), \
            'Only supporting chain mode = cascade / triangle'
        if chain_mode == 'triangle':
            assert (cross_from in ['decoder', 'encoder', 'embedding']), \
                'Triangle chain mode only supports cross_from = encoder / decoder / embedding'
            if not separate_embedding:
                assert (vocab_size_src == vocab_size_tgt), \
                    'Can not use the same embedding because vocab_size_src != vocab_size_tgt. ' \
                    'Please set separate_embedding = True'
            assert (merge_mode in ["sum", "concat", "learned_sum"]), \
                'Triangle chain mode only supports merge_mode = sum / concat / learned_sum'

        self.embed = None  # default
        if input_layer == "embed":
            self.embed_asr = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size_src, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            if not separate_embedding and chain_mode == "triangle":
                self.embed = torch.nn.Sequential(
                    torch.nn.Embedding(vocab_size_tgt, attention_dim),
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
        elif input_layer == "linear":
            self.embed_asr = torch.nn.Sequential(
                torch.nn.Linear(vocab_size_src, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            if not separate_embedding and chain_mode == "triangle":
                self.embed = torch.nn.Sequential(
                    torch.nn.Linear(vocab_size_tgt, attention_dim),
                    torch.nn.LayerNorm(attention_dim),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.ReLU(),
                    pos_enc_class(attention_dim, positional_dropout_rate),
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

        self.decoder_asr = repeat(
            num_blocks_asr,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
            ),
        )
        self.decoder_mt = repeat(
            num_blocks_mt,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
            )
        )

        self.ignore_id = ignore_id
        self.chain_mode = chain_mode
        self.eos = vocab_size_src - 1

        logging.info("*** Parallel attention parameters ***")
        logging.info('Using chain mode = %s' % chain_mode)
        if chain_mode == "triangle":
            logging.info("with input from %s" % cross_from)
            logging.info('with separate_embedding = %s' % str(separate_embedding))
            logging.info('and merge mode = %s' % merge_mode)


    def forward(
        self,
        hs_pad: torch.Tensor,  # encoder output
        hlens: torch.Tensor,
        ys_in_pad_asr: torch.Tensor,
        ys_in_lens_asr: torch.Tensor,
        ys_in_pad_mt: torch.Tensor,
        ys_in_lens_mt: torch.Tensor,
        mask_asr: torch.Tensor,
        mask_mt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        tgt_asr = ys_in_pad_asr
        # tgt_mask: (B, 1, L)
        tgt_mask_asr = (~make_pad_mask(ys_in_lens_asr)[:, None, :]).to(tgt_asr.device)
        # m: (1, L, L)
        m_asr = subsequent_mask(tgt_mask_asr.size(-1),
                                device=tgt_mask_asr.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask_asr = tgt_mask_asr & m_asr

        tgt_mt = ys_in_pad_mt
        # tgt_mask: (B, 1, L)
        tgt_mask_mt = (~make_pad_mask(ys_in_lens_mt)[:, None, :]).to(tgt_mt.device)
        # m: (1, L, L)
        m_mt = subsequent_mask(tgt_mask_mt.size(-1),
                               device=tgt_mask_mt.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask_mt = tgt_mask_mt & m_mt

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        #x_asr = self.embed_asr(tgt_asr)
        #x_mt = self.embed(tgt_mt)

        tgt = mask_based_pad(mask_asr, tgt_asr, tgt_mt, pad_val=self.eos)
        tgt_mask = mask_based_pad(mask_asr, tgt_mask_asr, tgt_mask_mt, pad_val=False)

        x = self.embed_asr(tgt)

        # 1: ASR decoder
        x_asr, tgt_mask_asr, memory, memory_mask = self.decoder_asr(
            x, tgt_mask, memory, memory_mask
        )

        #x_mt = self.embed(tgt_mt)
        x_mt, tgt_mask_mt, memory, memory_mask = self.decoder_mt(
            x_asr, tgt_mask_asr, memory, memory_mask
        )

        if self.normalize_before:
            x_mt = self.after_norm(x_mt)
            x_asr = self.after_norm_asr(x_asr)
        if self.output_layer is not None:
            x_mt = self.output_layer(x_mt)
            x_asr = self.output_layer_asr(x_asr)

        #return x_mt, tgt_mask_mt_pad, x_asr, tgt_mask_asr_pad
        return x_mt, tgt_mask_mt, x_asr, tgt_mask_asr

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
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
        n_layers = len(self.decoders)
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
