"""Parallel beam search module."""

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search_dual import BeamSearch
from espnet.nets.beam_search_dual import Hypothesis


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: torch.Tensor = torch.tensor([])  # (batch, maxlen)
    yseq_asr: torch.Tensor = torch.tensor([])

    score: torch.Tensor = torch.tensor([])  # (batch,)
    length: torch.Tensor = torch.tensor([])  # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch,)

    score_asr: torch.Tensor = torch.tensor([])  # (batch,)
    length_asr: torch.Tensor = torch.tensor([])  # (batch,)
    scores_asr: Dict[str, torch.Tensor] = dict()  # values: (batch,)

    states: Dict[str, Dict] = dict()
    states_asr: Dict[str, Dict] = dict()

    combined_score: torch.Tensor = torch.tensor([])

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        if len(hyps) == 0:
            return BatchHypothesis()
        return BatchHypothesis(
            yseq=pad_sequence(
                [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
            ),
            yseq_asr=pad_sequence(
                [h.yseq_asr for h in hyps], batch_first=True, padding_value=self.eos
            ),
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            length_asr=torch.tensor([len(h.yseq_asr) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            score_asr=torch.tensor([h.score_asr for h in hyps]),
            scores={k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers},
            scores_asr={k: torch.tensor([h.scores_asr[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
            states_asr={k: [h.states_asr[k] for h in hyps] for k in self.scorers},
            combined_score=torch.tensor([h.combined_score for h in hyps]),
        )

    def _batch_select(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            yseq_asr=hyps.yseq_asr[ids],
            score=hyps.score[ids],
            score_asr=hyps.score_asr[ids],
            combined_score=hyps.combined_score[ids],
            length=hyps.length[ids],
            length_asr=hyps.length_asr[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            scores_asr={k: v[ids] for k, v in hyps.scores_asr.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
            states_asr={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states_asr.items()
            },
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            yseq_asr=hyps.yseq_asr[i, : hyps.length[i]],
            score=hyps.score[i],
            score_asr=hyps.score_asr[i],
            combined_score=hyps.combined_score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            scores_asr={k: v[i] for k, v in hyps.scores_asr.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
            states_asr={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states_asr.items()
            },
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                yseq_asr=batch_hyps.yseq_asr[i][: batch_hyps.length_asr[i]],
                score=batch_hyps.score[i],
                combined_score=batch_hyps.combined_score[i],
                score_asr=batch_hyps.score_asr[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                scores_asr={k: batch_hyps.scores_asr[k][i] for k in self.scorers},
                states={
                    k: v.select_state(batch_hyps.states[k], i)
                    for k, v in self.scorers.items()
                },
                states_asr={
                    k: v.select_state(batch_hyps.states_asr[k], i)
                    for k, v in self.scorers.items()
                },
            )
            for i in range(len(batch_hyps.length))
        ]

    def batch_beam(
        self, weighted_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # ids: torch.Tensor  [torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids (torch.Tensor): The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`

        """
        top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
        # Because of the flatten above, `top_ids` is organized as:
        # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
        # where V is `self.n_vocab` and K is `self.beam_size`
        prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab
        return prev_hyp_ids, new_token_ids  #, prev_hyp_ids, new_token_ids

    def batch_beam_sum_eospred(
        self, weighted_scores_asr: torch.Tensor, weighted_scores_mt: torch.Tensor,
            running_hyps: BatchHypothesis, pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk token ids as tuple of token pairs
        """
        n_batch = running_hyps.yseq.shape[0]
        num_tokens = weighted_scores_mt.shape[-1]
        is_eos_mt = (
                running_hyps.yseq[torch.arange(n_batch), -1]
                == self.eos
        )  # shape [beam]
        is_eos_asr = (
                running_hyps.yseq_asr[torch.arange(n_batch), -1]
                == self.eos
        )  # shape [beam]

        if pos == 0:  # hyp always starts with SOS
            is_eos_mt = ~is_eos_mt
            is_eos_asr = ~is_eos_asr

        eos_token_mask = ~(torch.arange(num_tokens).unsqueeze(0).expand(n_batch, -1) == self.eos)

        mt_mask = is_eos_mt.view(-1, 1).expand(-1, num_tokens) & eos_token_mask # beam x num_tokens
        asr_mask = is_eos_asr.view(-1, 1).expand(-1, num_tokens) & eos_token_mask

        weighted_scores_mt = weighted_scores_mt.masked_fill(mt_mask, -np.inf)
        weighted_scores_asr = weighted_scores_asr.masked_fill(asr_mask, -np.inf)

        # asr_mask = is_eos_asr.view(-1, 1).expand(-1, weighted_scores_asr.shape[-1])
        # mt_mask = is_eos_mt.view(-1, 1).expand(-1, weighted_scores_mt.shape[-1])
        #
        # # weighted scores: shape [beam, n_vocab]
        # xk, ixk = weighted_scores_asr.masked_fill(asr_mask, -np.inf).view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]
        # yk, iyk = weighted_scores_mt.masked_fill(mt_mask, -np.inf).view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]



        # weighted scores: shape [beam, n_vocab]
        xk, ixk = weighted_scores_asr.view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]
        yk, iyk = weighted_scores_mt.view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]

        # [beam x beam] matrix with probs of all combinations summed
        S = self.asr_dec_weight * torch.mm(torch.t(xk), torch.ones_like(xk)) \
            + self.mt_dec_weight * torch.mm(torch.t(torch.ones_like(yk)), yk)

        # all combinations as tuples [beam^2, 2]
        s2v = torch.LongTensor([[i, j] for i in ixk.squeeze(0) for j in iyk.squeeze(0)])

        # best combinations [beam]
        local_best_scores, id2k = S.flatten().topk(self.beam_size)
        I = s2v[id2k]
        local_best_ids_asr = I[:, 0]
        local_best_ids_mt = I[:, 1]

        asr_prev_hyp_ids = local_best_ids_asr // self.n_vocab
        asr_new_token_ids = local_best_ids_asr % self.n_vocab

        mt_prev_hyp_ids = local_best_ids_mt // self.n_vocab
        mt_new_token_ids = local_best_ids_mt % self.n_vocab

        if pos > 0:
            asr_new_token_ids = asr_new_token_ids.masked_fill(is_eos_asr[asr_prev_hyp_ids], self.eos)
            mt_new_token_ids = mt_new_token_ids.masked_fill(is_eos_mt[asr_prev_hyp_ids], self.eos)

        return asr_prev_hyp_ids, asr_new_token_ids, mt_prev_hyp_ids, mt_new_token_ids, local_best_scores

    def batch_beam_joint(
        self, weighted_scores_asr: torch.Tensor, weighted_scores_mt: torch.Tensor,
            running_hyps: BatchHypothesis, pos: int)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk token ids as tuple of token pairs
        """
        n_batch = running_hyps.yseq.shape[0]

        is_eos_mt = (
                running_hyps.yseq[torch.arange(n_batch), -1]
                == self.eos
        )  # shape [beam]
        is_eos_asr = (
                running_hyps.yseq_asr[torch.arange(n_batch), -1]
                == self.eos
        )  # shape [beam]

        if pos <= self.wait_k_asr:
            is_eos_mt = ~is_eos_mt

        if pos <= self.wait_k_st:
            is_eos_asr = ~is_eos_asr

        if pos < self.wait_k_asr:  # wait with ST
            asr_dec_weight = 1.0
            mt_dec_weight = 0.0
        elif pos < self.wait_k_st:  # wait with ASR
            asr_dec_weight = 0.0
            mt_dec_weight = 1.0
        else:  # normal
            asr_dec_weight = self.asr_dec_weight
            mt_dec_weight = self.mt_dec_weight

        # 1. Not reached EOS yet
        not_eos = ~(is_eos_mt | is_eos_asr)
        if not_eos.any():
            not_eos_hyp_ids = torch.nonzero(not_eos, as_tuple=False).view(-1)

            if self.dec_method == 2 or self.dec_method == 4:
                if self.dec_method == 4:
                    # Combined weighted score of allowed combinations with beamsize-best ASR
                    # weighted scores: shape [beam, n_vocab]
                    xk, ixk = weighted_scores_asr[not_eos_hyp_ids].view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]

                    asr_prev_hyp_ids = not_eos_hyp_ids[ixk // self.n_vocab]

                    # [beam_size x beam_size]
                    mt_top_scores_of_hyp, mt_idx = weighted_scores_mt[asr_prev_hyp_ids].topk(self.beam_size)
                    mt_top_scores_of_hyp = mt_top_scores_of_hyp.squeeze(0)
                    mt_idx = mt_idx.squeeze(0)

                    # [beam x beam] matrix with probs of all combinations summed
                    S = asr_dec_weight * torch.mm(torch.t(xk), torch.ones_like(xk)) \
                        + mt_dec_weight * mt_top_scores_of_hyp

                    # all combinations as tuples [beam^2, 2]
                    s2v = torch.LongTensor([[i, j] for k, i in enumerate(ixk.squeeze(0)) for j in mt_idx[k, :]])

                elif self.dec_method == 2:
                    # Combined weighted score of best combinations
                    # weighted scores: shape [beam, n_vocab]
                    xk, ixk = weighted_scores_asr[not_eos_hyp_ids].view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]

                    yk, iyk = weighted_scores_mt[not_eos_hyp_ids].view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]

                    # [beam x beam] matrix with probs of all combinations summed
                    S = asr_dec_weight * torch.mm(torch.t(xk), torch.ones_like(xk)) \
                        + mt_dec_weight * torch.mm(torch.t(torch.ones_like(yk)), yk)

                    # all combinations as tuples [beam^2, 2]
                    s2v = torch.LongTensor([[i, j] for i in ixk.squeeze(0) for j in iyk.squeeze(0)])

                else:
                    raise NotImplementedError

                # best combinations [beam]
                local_best_scores, id2k = S.flatten().topk(self.beam_size)
                I = s2v[id2k]
                local_best_ids_asr = I[:, 0]
                local_best_ids_mt = I[:, 1]

                asr_prev_hyp_ids = not_eos_hyp_ids[local_best_ids_asr // self.n_vocab]
                asr_new_token_ids = local_best_ids_asr % self.n_vocab

                mt_prev_hyp_ids = asr_prev_hyp_ids
                mt_new_token_ids = local_best_ids_mt % self.n_vocab

            elif self.dec_method == 5:
                n_hyps = not_eos_hyp_ids.shape[0]
                xk, ixk = weighted_scores_asr[not_eos_hyp_ids].topk(self.beam_size, dim=1)
                yk, iyk = weighted_scores_mt[not_eos_hyp_ids].topk(self.beam_size, dim=1)

                combs = asr_dec_weight * torch.bmm(xk.view(n_hyps, self.beam_size, 1),
                                                    torch.ones(n_hyps, 1, self.beam_size)) \
                    + mt_dec_weight * torch.bmm(torch.ones(n_hyps, self.beam_size, 1),
                                                yk.view(n_hyps, 1, self.beam_size))

                best_combs, best_combs_idx = combs.view(n_hyps, -1).topk(self.beam_size, dim=1)
                mt_idx = best_combs_idx // self.beam_size
                asr_idx = best_combs_idx % self.beam_size

                beam_scores, beam_idx = best_combs.flatten().topk(self.beam_size)
                hyp_id = beam_idx // self.beam_size
                comb_id = beam_idx % self.beam_size

                prev_hyp_ids = not_eos_hyp_ids[hyp_id]

                asr_ids_after_topk = asr_idx[hyp_id, comb_id]
                asr_new_token_ids = ixk[hyp_id, asr_ids_after_topk]

                mt_ids_after_topk = mt_idx[hyp_id, comb_id]
                mt_new_token_ids = iyk[hyp_id, mt_ids_after_topk]

                asr_prev_hyp_ids = prev_hyp_ids
                mt_prev_hyp_ids = prev_hyp_ids
                local_best_scores = beam_scores

        else:
            local_best_scores = torch.Tensor([])
            asr_prev_hyp_ids, asr_new_token_ids = torch.LongTensor([]), torch.LongTensor([])
            mt_prev_hyp_ids, mt_new_token_ids = torch.LongTensor([]), torch.LongTensor([])

        # 2. MT hyp is already EOS
        if is_eos_mt.any():
            eos_mt_hyp_ids = torch.nonzero(is_eos_mt, as_tuple=False).view(-1)
            # only use asr score for these hyps
            eos_mt_local_best_scores, idx = weighted_scores_asr[eos_mt_hyp_ids].flatten().topk(self.beam_size)

            # new token ids for asr hyp
            eos_mt_prev_hyp_ids = eos_mt_hyp_ids[idx // self.n_vocab]
            eos_mt_new_token_ids = idx % self.n_vocab

            # eos_mt_eos_scores = weighted_scores_mt[eos_mt_prev_hyp_ids, self.eos]
            # eos_mt_local_best_scores = self.asr_dec_weight * eos_mt_local_best_scores \
            #                            + self.mt_dec_weight * eos_mt_eos_scores

            local_best_scores = torch.cat((local_best_scores, eos_mt_local_best_scores))
            mt_prev_hyp_ids = torch.cat((mt_prev_hyp_ids, eos_mt_prev_hyp_ids))
            asr_prev_hyp_ids = torch.cat((asr_prev_hyp_ids, eos_mt_prev_hyp_ids))

            # -1 for mt, will be converted to eos
            mt_new_token_ids = torch.cat((mt_new_token_ids, torch.full(eos_mt_new_token_ids.shape, -1,
                                                                         dtype=eos_mt_new_token_ids.dtype)))
            asr_new_token_ids = torch.cat((asr_new_token_ids, eos_mt_new_token_ids))

        # 3. ASR hyp is already at EOS
        if is_eos_asr.any():
            eos_asr_hyp_ids = torch.nonzero(is_eos_asr, as_tuple=False).view(-1)
            # only use mt scores
            eos_asr_local_best_scores, idx = weighted_scores_mt[eos_asr_hyp_ids].flatten().topk(self.beam_size)

            # new token ids for mt hyp
            eos_asr_prev_hyp_ids = eos_asr_hyp_ids[idx // self.n_vocab]
            eos_asr_new_token_ids = idx // self.n_vocab

            # eos_asr_eos_scores = weighted_scores_asr[eos_asr_hyp_ids, self.eos]
            # eos_asr_local_best_scores = self.asr_dec_weight * eos_asr_eos_scores \
            #                             + self.mt_dec_weight * eos_asr_local_best_scores

            local_best_scores = torch.cat((local_best_scores, eos_asr_local_best_scores))
            mt_prev_hyp_ids = torch.cat((mt_prev_hyp_ids, eos_asr_prev_hyp_ids))
            asr_prev_hyp_ids = torch.cat((asr_prev_hyp_ids, eos_asr_prev_hyp_ids))

            mt_new_token_ids = torch.cat((mt_new_token_ids, eos_asr_new_token_ids))
            asr_new_token_ids = torch.cat((asr_new_token_ids, torch.full(eos_asr_new_token_ids.shape, -1,
                                                                         dtype=eos_asr_new_token_ids.dtype)))

        if local_best_scores.shape[0] > self.beam_size:
            best_scores, best_idx = local_best_scores.topk(self.beam_size)
            mt_prev_hyp_ids, asr_prev_hyp_ids = mt_prev_hyp_ids[best_idx], asr_prev_hyp_ids[best_idx]
            mt_new_token_ids, asr_new_token_ids = mt_new_token_ids[best_idx], asr_new_token_ids[best_idx]
            local_best_scores = best_scores

        return asr_prev_hyp_ids, asr_new_token_ids, mt_prev_hyp_ids, mt_new_token_ids, local_best_scores

    def batch_beam_sum(
        self, weighted_scores_asr: torch.Tensor, weighted_scores_mt: torch.Tensor,
            running_hyps: BatchHypothesis, pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk token ids as tuple of token pairs
        """
        n_batch = running_hyps.yseq.shape[0]
        num_tokens = weighted_scores_mt.shape[-1]

        if self.mask_eos_sentences:
            is_eos_mt = (
                    running_hyps.yseq[torch.arange(n_batch), -1]
                    == self.eos
            )  # shape [beam]
            is_eos_asr = (
                    running_hyps.yseq_asr[torch.arange(n_batch), -1]
                    == self.eos
            )  # shape [beam]

            if pos == 0:  # hyp always starts with SOS
                is_eos_mt = ~is_eos_mt
                is_eos_asr = ~is_eos_asr

            if is_eos_asr.any() or is_eos_mt.any():
                a = 1
                pass

            not_eos_mask = ~(torch.arange(num_tokens).unsqueeze(0).expand(n_batch, -1) == self.eos)

            mt_mask = is_eos_mt.view(-1, 1).expand(-1, num_tokens) & not_eos_mask  # beam x num_tokens
            asr_mask = is_eos_asr.view(-1, 1).expand(-1, num_tokens) & not_eos_mask

            weighted_scores_mt = weighted_scores_mt.masked_fill(mt_mask, -np.inf)
            weighted_scores_asr = weighted_scores_asr.masked_fill(asr_mask, -np.inf)

        if self.dec_method == 1:
            # beamsize-best ASR hyps with for every hyp the 1-best MT hyp
            # plus the beamsize-best MT hyps with for every hyp the 1-best ASR hyp
            # (so generates 2 x beamsize hyps)

            # ASR first
            xk, ixk = weighted_scores_asr.view(-1).topk(self.beam_size)
            prev_hyp_ids_asr = ixk // self.n_vocab
            asr_new_token_ids_asr = ixk % self.n_vocab

            mt_best_score_per_hyp, mt_best_id_per_hyp = weighted_scores_mt.max(dim=1)
            mt_new_token_ids_asr = mt_best_id_per_hyp[prev_hyp_ids_asr]

            # MT second
            yk, iyk = weighted_scores_mt.view(-1).topk(self.beam_size)
            prev_hyp_ids_mt = iyk // self.n_vocab
            mt_new_token_ids_mt = iyk % self.n_vocab

            asr_best_score_per_hyp, asr_best_id_per_hyp = weighted_scores_asr.max(dim=1)
            asr_new_token_ids_mt = asr_best_id_per_hyp[prev_hyp_ids_mt]

            asr_scores = torch.cat([xk, asr_best_score_per_hyp[prev_hyp_ids_mt]])
            mt_scores = torch.cat([mt_best_score_per_hyp[prev_hyp_ids_asr], yk])
            local_best_scores = self.asr_dec_weight * asr_scores + self.mt_dec_weight * mt_scores

            asr_prev_hyp_ids = torch.cat([prev_hyp_ids_asr, prev_hyp_ids_mt])
            mt_prev_hyp_ids = torch.cat([prev_hyp_ids_asr, prev_hyp_ids_mt])
            mt_new_token_ids = torch.cat([mt_new_token_ids_asr, mt_new_token_ids_mt])
            asr_new_token_ids = torch.cat([asr_new_token_ids_asr, asr_new_token_ids_mt])

        elif self.dec_method == 2:
            # Use the beamsize best combinations of ASR and MT hyps, based on the combined weighted score
            # Should use cache=None because not all combinations are valid
            # (because next state of ASR is based on both the ASR + the MT hyp up to that point)

            # weighted scores: shape [beam, n_vocab]
            xk, ixk = weighted_scores_asr.view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]
            yk, iyk = weighted_scores_mt.view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]

            # [beam x beam] matrix with probs of all combinations summed
            S = self.asr_dec_weight * torch.mm(torch.t(xk), torch.ones_like(xk)) \
                + self.mt_dec_weight * torch.mm(torch.t(torch.ones_like(yk)), yk)

            # all combinations as tuples [beam^2, 2]
            s2v = torch.LongTensor([[i, j] for i in ixk.squeeze(0) for j in iyk.squeeze(0)])

            # best combinations [beam]
            local_best_scores, id2k = S.flatten().topk(self.beam_size)
            I = s2v[id2k]
            local_best_ids_asr = I[:, 0]
            local_best_ids_mt = I[:, 1]

            asr_prev_hyp_ids = local_best_ids_asr // self.n_vocab
            asr_new_token_ids = local_best_ids_asr % self.n_vocab

            mt_prev_hyp_ids = local_best_ids_mt // self.n_vocab
            mt_new_token_ids = local_best_ids_mt % self.n_vocab


        elif self.dec_method == 3:
            # beamsize-best ASR hyps with for every hyp the 1-best MT hyp

            xk, ixk = weighted_scores_asr.view(-1).topk(self.beam_size)
            asr_prev_hyp_ids = ixk // self.n_vocab
            asr_new_token_ids = ixk % self.n_vocab

            mt_prev_hyp_ids = asr_prev_hyp_ids
            mt_best_score_per_hyp, mt_best_id_per_hyp = weighted_scores_mt.max(dim=1)
            mt_new_token_ids = mt_best_id_per_hyp[asr_prev_hyp_ids]

            local_best_scores = self.asr_dec_weight * xk \
                                + self.mt_dec_weight * mt_best_score_per_hyp[asr_prev_hyp_ids]

        elif self.dec_method == 4:
            # Combined weighted score of allowed combinations with beamsize-best ASR

            # weighted scores: shape [beam, n_vocab]
            xk, ixk = weighted_scores_asr.view(1, -1).topk(self.beam_size)  # [ 1, beam_size ]

            asr_prev_hyp_ids = ixk // self.n_vocab

            # [beam_size x beam_size]
            mt_top_scores_of_hyp, mt_idx = weighted_scores_mt[asr_prev_hyp_ids].topk(self.beam_size)
            mt_top_scores_of_hyp = mt_top_scores_of_hyp.squeeze(0)
            mt_idx = mt_idx.squeeze(0)

            # [beam x beam] matrix with probs of all combinations summed
            S = self.asr_dec_weight * torch.mm(torch.t(xk), torch.ones_like(xk)) \
                + self.mt_dec_weight * mt_top_scores_of_hyp

            # all combinations as tuples [beam^2, 2]
            s2v = torch.LongTensor([[i, j] for k, i in enumerate(ixk.squeeze(0)) for j in mt_idx[k, :]])

            # best combinations [beam]
            local_best_scores, id2k = S.flatten().topk(self.beam_size)
            I = s2v[id2k]
            local_best_ids_asr = I[:, 0]
            local_best_ids_mt = I[:, 1]

            asr_prev_hyp_ids = local_best_ids_asr // self.n_vocab
            asr_new_token_ids = local_best_ids_asr % self.n_vocab

            mt_prev_hyp_ids = asr_prev_hyp_ids
            mt_new_token_ids = local_best_ids_mt

        else:
            raise ValueError

        return asr_prev_hyp_ids, asr_new_token_ids, mt_prev_hyp_ids, mt_new_token_ids, local_best_scores

    def init_hyp(self, x: torch.Tensor) -> BatchHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        init_states_asr = dict()
        init_scores_asr = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0
            init_states_asr[k] = d.batch_init_state(x)
            init_scores_asr[k] = 0.0
        return self.batchfy(
            [
                Hypothesis(
                    score=0.0,
                    score_asr=0.0,
                    combined_score=0.0,
                    scores=init_scores,
                    scores_asr=init_scores_asr,
                    states=init_states,
                    states_asr=init_states_asr,
                    yseq=torch.tensor([self.sos], device=x.device),
                    yseq_asr=torch.tensor([self.sos], device=x.device),
                )
            ]
        )

    def score_full(
        self, hyp: BatchHypothesis, x: torch.Tensor, pos: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        scores_asr = dict()
        states = dict()
        states_asr = dict()
        for k, d in self.full_scorers.items():
            scores[k], scores_asr[k], states[k], states_asr[k] = d.batch_score(hyp.yseq, hyp.yseq_asr,
                                                                hyp.states[k], hyp.states_asr[k], x,
                                                                pos=pos, use_cache=self.use_cache)
        return scores, scores_asr, states, states_asr

    def score_partial(
        self, hyp: BatchHypothesis, ids: torch.Tensor, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (torch.Tensor): 2D tensor of new partial tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.batch_score_partial(
                hyp.yseq_asr, ids, hyp.states_asr[k], x
            )
        return scores, states

    def merge_states(self, states: Any, part_states: Any = None, part_idx: int = None) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        if part_states is not None:
            for k, v in part_states.items():
                new_states[k] = v
        return new_states

    def search(self, running_hyps: BatchHypothesis, x: torch.Tensor, pos: int) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)
            int: position in beam/utt

        Returns:
            BatchHypothesis: Best sorted hypotheses

        """
        n_batch = len(running_hyps)
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores_asr = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )
        weighted_scores_mt = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )

        scores, scores_asr, states, states_asr = self.score_full(running_hyps, x.expand(n_batch, *x.shape), pos)
        for k in self.full_scorers:
            weighted_scores_asr += self.weights[k] * scores_asr[k]
            weighted_scores_mt += self.weights[k] * scores[k]

        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores_asr
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]

        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)  # only scores best asr beams and produces CTC output: hyp.yseq_asr
        for k in self.part_scorers:
            weighted_scores_asr += self.weights[k] * part_scores[k]

        # add previous hyp scores
        weighted_scores_asr += running_hyps.score_asr.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        weighted_scores_mt += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        # TODO(karita): do not use list. use batch instead
        # see also https://github.com/espnet/espnet/pull/1402#discussion_r354561029
        # update hyps
        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)

        # batch_beam ignores part_ids, just does top-k
        # for (
        #     prev_hyp_id,
        #     new_token_id,
        #     mt_prev_hyp_id,
        #     mt_new_token_id
        # ) in zip(*self.batch_beam(weighted_scores_asr), *self.batch_beam(weighted_scores_mt)):

        for (
            prev_hyp_id,
            new_token_id,
            mt_prev_hyp_id,
            mt_new_token_id,
            local_best_score
        ) in zip(*self.batch_beam_joint(weighted_scores_asr, weighted_scores_mt, running_hyps, pos)):
            prev_hyp = prev_hyps[prev_hyp_id]
            mt_prev_hyp = prev_hyps[mt_prev_hyp_id]

            if mt_new_token_id == -1 or pos < self.wait_k_asr:
                if pos < self.wait_k_asr:
                    # wait with MT
                    next_token_mt = None
                else:
                    # MT is eos
                    next_token_mt = self.eos
                best_hyps.append(
                    Hypothesis(
                        score=mt_prev_hyp.score,
                        score_asr=weighted_scores_asr[prev_hyp_id, new_token_id],
                        combined_score=local_best_score,
                        yseq=self.append_token(mt_prev_hyp.yseq, next_token_mt),
                        yseq_asr=self.append_token(prev_hyp.yseq_asr, new_token_id),
                        scores_asr=self.merge_scores(
                            prev_hyp.scores_asr,
                            {k: v[prev_hyp_id] for k, v in scores_asr.items()},
                            new_token_id,
                            {k: v[prev_hyp_id] for k, v in part_scores.items()},
                            new_token_id,
                        ),
                        scores=mt_prev_hyp.scores,
                        states_asr=self.merge_states(
                            {
                                k: self.full_scorers[k].select_state(v, prev_hyp_id)
                                for k, v in states_asr.items()
                            },
                            {
                                k: self.part_scorers[k].select_state(
                                    v, prev_hyp_id, new_token_id
                                )
                                for k, v in part_states.items()
                            },
                            new_token_id,
                        ),
                        states=self.merge_states(
                            {
                                k: self.full_scorers[k].select_state(v, mt_prev_hyp_id)
                                for k, v in states.items()
                            },
                            {
                                k: self.part_scorers[k].select_state(
                                    v, prev_hyp_id, new_token_id
                                )
                                for k, v in part_states.items()
                            },
                            new_token_id,
                        ),
                    )
                )
            elif new_token_id == -1 or pos < self.wait_k_st:
                if pos < self.wait_k_st:
                    # wait with ASR
                    next_token_asr = None
                else:
                    # ASR is eos
                    next_token_asr = self.eos

                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores_mt[mt_prev_hyp_id, mt_new_token_id],
                        score_asr=prev_hyp.score_asr,
                        combined_score=local_best_score,
                        yseq=self.append_token(mt_prev_hyp.yseq, mt_new_token_id),
                        yseq_asr=self.append_token(prev_hyp.yseq_asr, next_token_asr),
                        scores_asr=prev_hyp.scores_asr,
                        scores=self.merge_scores(
                            mt_prev_hyp.scores,
                            {k: v[mt_prev_hyp_id] for k, v in scores.items()},
                            mt_new_token_id,
                            {k: v[mt_prev_hyp_id] for k, v in part_scores.items()},
                            new_token_id,
                        ),
                        states_asr=self.merge_states(
                            {
                                k: self.full_scorers[k].select_state(v, prev_hyp_id)
                                for k, v in states_asr.items()
                            },
                            {
                                k: self.part_scorers[k].select_state(
                                    v, prev_hyp_id, new_token_id
                                )
                                for k, v in part_states.items()
                            },
                            new_token_id,
                        ),
                        states=self.merge_states(
                            {
                                k: self.full_scorers[k].select_state(v, mt_prev_hyp_id)
                                for k, v in states.items()
                            },
                            {
                                k: self.part_scorers[k].select_state(
                                    v, prev_hyp_id, new_token_id
                                )
                                for k, v in part_states.items()
                            },
                            new_token_id,
                        ),
                    )
                )
            else:
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores_mt[mt_prev_hyp_id, mt_new_token_id],
                        score_asr=weighted_scores_asr[prev_hyp_id, new_token_id],
                        combined_score=local_best_score,
                        yseq=self.append_token(mt_prev_hyp.yseq, mt_new_token_id),
                        yseq_asr=self.append_token(prev_hyp.yseq_asr, new_token_id),
                        scores_asr=self.merge_scores(
                            prev_hyp.scores_asr,
                            {k: v[prev_hyp_id] for k, v in scores_asr.items()},
                            new_token_id,
                            {k: v[prev_hyp_id] for k, v in part_scores.items()},
                            new_token_id,
                        ),
                        scores=self.merge_scores(
                            mt_prev_hyp.scores,
                            {k: v[mt_prev_hyp_id] for k, v in scores.items()},
                            mt_new_token_id,
                            {k: v[mt_prev_hyp_id] for k, v in part_scores.items()},
                            new_token_id,
                        ),
                        states_asr=self.merge_states(
                            {
                                k: self.full_scorers[k].select_state(v, prev_hyp_id)
                                for k, v in states_asr.items()
                            },
                            {
                                k: self.part_scorers[k].select_state(
                                    v, prev_hyp_id, new_token_id
                                )
                                for k, v in part_states.items()
                            },
                            new_token_id,
                        ),
                        states=self.merge_states(
                            {
                                k: self.full_scorers[k].select_state(v, mt_prev_hyp_id)
                                for k, v in states.items()
                            },
                            {
                                k: self.part_scorers[k].select_state(
                                    v, prev_hyp_id, new_token_id
                                )
                                for k, v in part_states.items()
                            },
                            new_token_id,
                        ),
                    )
                )
        return self.batchfy(best_hyps)

    def post_process(
        self,
        i: int,
        maxlen: int,
        maxlenratio: float,
        running_hyps: BatchHypothesis,
        ended_hyps: List[Hypothesis],
    ) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (BatchHypothesis): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        n_batch = running_hyps.yseq.shape[0]
        logging.debug(f"the number of running hypothes: {n_batch}")
        if self.token_list is not None:
            logging.debug(
                "best hypo: "
                + "".join(
                    [
                        self.token_list[x]
                        for x in running_hyps.yseq[0, 1 : running_hyps.length[0]]
                    ]
                )
            )
            logging.debug(
                "best hypo asr: "
                + "".join(
                    [
                        self.token_list[x]
                        for x in running_hyps.yseq_asr[0, 1: running_hyps.length_asr[0]]
                    ]
                )
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            yseq_eos = torch.cat(
                (
                    running_hyps.yseq,
                    torch.full(
                        (n_batch, 1),
                        self.eos,
                        device=running_hyps.yseq.device,
                        dtype=torch.int64,
                    ),
                ),
                1,
            )
            running_hyps.yseq.resize_as_(yseq_eos)
            running_hyps.yseq[:] = yseq_eos
            running_hyps.length[:] = yseq_eos.shape[1]

            logging.info("adding <eos> in the last position in the loop for asr")
            yseq_asr_eos = torch.cat(
                (
                    running_hyps.yseq_asr,
                    torch.full(
                        (n_batch, 1),
                        self.eos,
                        device=running_hyps.yseq_asr.device,
                        dtype=torch.int64,
                    ),
                ),
                1,
            )
            running_hyps.yseq_asr.resize_as_(yseq_asr_eos)
            running_hyps.yseq_asr[:] = yseq_asr_eos
            running_hyps.length_asr[:] = yseq_asr_eos.shape[1]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        is_eos_mt = (
            running_hyps.yseq[torch.arange(n_batch), running_hyps.length - 1]
            == self.eos
        )
        is_eos_asr = (
                running_hyps.yseq_asr[torch.arange(n_batch), running_hyps.length_asr - 1]
                == self.eos
        )
        is_eos = is_eos_mt & is_eos_asr
        for b in torch.nonzero(is_eos, as_tuple=False).view(-1):
            hyp = self._select(running_hyps, b)
            ended_hyps.append(hyp)
        remained_ids = torch.nonzero(is_eos == 0, as_tuple=False).view(-1)
        return self._batch_select(running_hyps, remained_ids)
