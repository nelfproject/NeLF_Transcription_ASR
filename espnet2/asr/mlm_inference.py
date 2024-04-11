import logging
from contextlib import contextmanager
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import numpy
import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class MLMInference(torch.nn.Module):
    """Mask-CTC-based non-autoregressive inference"""

    def __init__(
        self,
        decoder: MLMDecoder,
        multi_decoder: bool,
        mask_token: int,
        token_list: List,
        n_iterations: int,
        threshold_probability: float,
        init_masked: bool = True,
    ):
        """Initialize Mask-CTC inference"""
        super().__init__()
        self.mlm = decoder
        self.multi_decoder = multi_decoder
        self.mask_token = mask_token
        self.n_iterations = n_iterations
        self.threshold_probability = threshold_probability
        self.converter = TokenIDConverter(token_list=token_list)
        self.init_masked = init_masked

    def ids2text(self, ids: List[int]):
        text = "".join(self.converter.ids2tokens(ids))
        return text.replace("<mask>", "_").replace("<space>", " ")

    def forward(self, enc_out: torch.Tensor, sentence_len: torch.Tensor) -> List[Hypothesis]:
        """Perform Mask-CTC inference"""
        if self.multi_decoder:
            enc_out_multi = enc_out[1].unsqueeze(0)
            enc_out = enc_out[0].unsqueeze(0)
        else:
            enc_out = enc_out.unsqueeze(0)

        sentence_len = int(sentence_len[0])

        if self.init_masked:
            mask_num = sentence_len
            y_in = (
                    torch.zeros(1, sentence_len, dtype=torch.long).to(enc_out.device)
                    + self.mask_token
            )
        else:
            raise NotImplementedError
        logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

        p_thres = self.threshold_probability
        mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)

        # iterative decoding
        if not mask_num == 0:
            K = self.n_iterations
            num_iter = K if mask_num >= K and K > 0 else mask_num

            for t in range(num_iter - 1):
                if self.multi_decoder:
                    pred, _ = self.mlm(enc_out, [enc_out.size(1)], enc_out_multi, [enc_out_multi.size(1)], y_in, [y_in.size(1)])
                else:
                    pred, _ = self.mlm(enc_out, [enc_out.size(1)], y_in, [y_in.size(1)])
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred_score, pred_id = pred[0][mask_idx].max(dim=-1)
                cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                y_in[0][mask_idx[cand]] = pred_id[cand]
                mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)

                logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

            # predict leftover masks (|masks| < mask_num // num_iter)
            if self.multi_decoder:
                pred, _ = self.mlm(enc_out, [enc_out.size(1)], enc_out_multi, [enc_out_multi.size(1)], y_in,
                                   [y_in.size(1)])
            else:
                pred, _ = self.mlm(enc_out, [enc_out.size(1)], y_in, [y_in.size(1)])
            if isinstance(pred, tuple):
                pred = pred[0]
            y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)

            logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor(
            [self.mask_token] + y_in.tolist()[0] + [self.mask_token], device=y_in.device
        )

        return Hypothesis(yseq=yseq)


class MaskCTCInference(torch.nn.Module):
    """Mask-CTC-based non-autoregressive inference"""

    def __init__(
        self,
        decoder: MLMDecoder,
        ctc: CTC,
        mask_token: int,
        token_list: List,
        n_iterations: int,
        threshold_probability: float,
    ):
        """Initialize Mask-CTC inference"""
        super().__init__()
        self.ctc = ctc
        self.mlm = decoder
        self.mask_token = mask_token
        self.n_iterations = n_iterations
        self.threshold_probability = threshold_probability
        self.converter = TokenIDConverter(token_list=token_list)

    def ids2text(self, ids: List[int]):
        text = "".join(self.converter.ids2tokens(ids))
        return text.replace("<mask>", "_").replace("<space>", " ")

    def forward(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Perform Mask-CTC inference"""
        # greedy ctc outputs
        enc_out = enc_out.unsqueeze(0)
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(enc_out)).max(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        logging.info("ctc:{}".format(self.ids2text(y_hat[y_idx].tolist())))

        # calculate token-level ctc probabilities by taking
        # the maximum probability of consecutive frames with
        # the same ctc symbols
        probs_hat = []
        cnt = 0
        for i, y in enumerate(y_hat.tolist()):
            probs_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if probs_hat[i] < ctc_probs[0][cnt]:
                    probs_hat[i] = ctc_probs[0][cnt].item()
                cnt += 1
        probs_hat = torch.from_numpy(numpy.array(probs_hat))

        # mask ctc outputs based on ctc probabilities
        p_thres = self.threshold_probability
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
        mask_num = len(mask_idx)

        y_in = (
            torch.zeros(1, len(y_idx), dtype=torch.long).to(enc_out.device)
            + self.mask_token
        )
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx]

        logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

        # iterative decoding
        if not mask_num == 0:
            K = self.n_iterations
            num_iter = K if mask_num >= K and K > 0 else mask_num

            for t in range(num_iter - 1):
                pred, _ = self.mlm(enc_out, [enc_out.size(1)], y_in, [y_in.size(1)])
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred_score, pred_id = pred[0][mask_idx].max(dim=-1)
                cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                y_in[0][mask_idx[cand]] = pred_id[cand]
                mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)

                logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

            # predict leftover masks (|masks| < mask_num // num_iter)
            pred, _ = self.mlm(enc_out, [enc_out.size(1)], y_in, [y_in.size(1)])
            if isinstance(pred, tuple):
                pred = pred[0]
            y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)

            logging.info("msk:{}".format(self.ids2text(y_in[0].tolist())))

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor(
            [self.mask_token] + y_in.tolist()[0] + [self.mask_token], device=y_in.device
        )

        return Hypothesis(yseq=yseq)
