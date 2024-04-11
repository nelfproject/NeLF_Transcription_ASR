#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0, '../../..')
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import List
from typing import Dict

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args
from espnet2.asr.mlm_inference import MLMInference, MaskCTCInference
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.s2t_subs import S2TTaskSubtitle
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none

# Alias for typing
ListOfHypothesis = List[
    Tuple[
        Optional[str],
        List[str],
        List[int],
        Optional[str],
        Hypothesis,
    ]
]


class ScoreFilter(BatchScorerInterface, torch.nn.Module):
    """Filter scores based on pre-defined rules.

    See comments in the score method.

    """

    def __init__(
        self,
        notimestamps: int,
        utttimestamps: int,
        wordtimestamps: int,
        first_time: int,
        last_time: int,
        first_spk: int,
        last_spk: int,
        spk_change: int,
        sos: int,
        eos: int,
        vocab_size: int,
    ):
        super().__init__()

        self.notimestamps = notimestamps
        self.utttimestamps = utttimestamps
        self.wordtimestamps = wordtimestamps
        self.first_time = first_time
        self.last_time = last_time
        self.first_spk = first_spk
        self.last_spk = last_spk
        self.spk_change = spk_change
        self.sos = sos
        self.eos = eos
        self.vocab_size = vocab_size

        # dummy param used to obtain the current dtype and device
        self.param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """

        score = torch.zeros(
            self.vocab_size, dtype=self.param.dtype, device=self.param.device
        )

        ## 1. NO TIMINGS
        if self.notimestamps in y:
            # Suppress timestamp tokens if we don't predict time
            score[self.first_time: self.last_time + 1] = -np.inf
            score[self.first_spk: self.last_spk + 1] = -np.inf

        ## 2. UTTERANCE LEVEL TIMINGS
        elif self.utttimestamps in y:
            # Suppress word-level speaker tags (but not speaker change)
            score[self.first_spk: self.last_spk + 1] = -np.inf

            # First token of sentence
            if y[-4] == self.sos:
                # The first token must be a timestamp if we predict time
                score[: self.first_time] = -np.inf
                score[self.last_time + 1 :] = -np.inf
            else:
                # Get all previously decoded timestamps
                prev_times = y[torch.logical_and(y >= self.first_time, y <= self.last_time)]

                # If there are an odd number of timestamps, the sentence is incomplete
                if len(prev_times) % 2 == 1:
                    score[self.eos] = -np.inf

                    # a starting timestamp of an utterance has to be followed by a word (or speaker change)
                    if y[-1] >= self.first_time and y[-1] <= self.last_time:
                        score[self.first_time: self.last_time + 1] = -np.inf
                    else:
                        # Can be another word-token or a closing timestamp, given that timestamps are monotonic
                        score[self.first_time : prev_times[-1] + 1] = -np.inf

                # If there are an even number of timestamps (all are paired), the next token should be <eos> or again an opening timestamp
                else:
                    if y[-1] >= self.first_time and y[-1] <= self.last_time:
                        score[: y[-1]] = -np.inf
                        score[self.last_time + 1 :] = -np.inf
                        score[self.eos] = 0.0
                    else:
                        # this is an illegal hyp
                        score[:] = -np.inf

        ## 3. WORD-LEVEL TIMINGS
        elif self.wordtimestamps in y:
            # No speaker change tokens, only speaker-specific tags
            score[self.spk_change] = -np.inf

            # First tokens of sentence
            if y[-4] == self.sos:
                # The first token must be a speaker id
                score[: self.first_spk] = -np.inf
                score[self.last_spk + 1 :] = -np.inf

            elif y[-5] == self.sos:
                # Second token is a timestamp
                score[: self.first_time] = -np.inf
                score[self.last_time + 1:] = -np.inf

            else:
                # Get all previously decoded timestamps
                prev_times = y[torch.logical_and(y >= self.first_time, y <= self.last_time)]

                # Spk always followed by timing (which can be same as previous)
                if y[-1] >= self.first_spk and y[-1] <= self.last_spk:
                    score[: prev_times[-1]] = -np.inf
                    score[self.last_time + 1:] = -np.inf

                # If there are an odd number of timestamps, the sentence is incomplete
                elif len(prev_times) % 2 == 1:
                    score[self.eos] = -np.inf

                    # a starting timestamp has to be followed by a word token
                    if y[-1] >= self.first_time and y[-1] <= self.last_time:
                        score[self.first_time: self.last_time + 1] = -np.inf
                        score[self.first_spk: self.last_spk + 1] = -np.inf
                    else:
                        # word token has to be followed by another word token or a closing timestamp, but in that case monotonically increasing
                        score[self.first_time: prev_times[-1]] = -np.inf
                        score[self.first_spk: self.last_spk + 1] = -np.inf

                # If there are an even number of timestamps (all are paired), the next token should be <eos> or again a speaker tag
                else:
                    if y[-1] >= self.first_time and y[-1] <= self.last_time:
                        # the next token should be a speaker or eos
                        score[: self.first_spk] = -np.inf
                        score[self.last_spk + 1:] = -np.inf
                        score[self.eos] = 0.0

                    # if y[-1] is spk, already caught this case in first if-clause
                    else:
                        # this is an illegal hyp
                        score[:] = -np.inf

        return score, None

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

        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates


class Speech2TextAndSubs:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        subtitle_ctc_weight: float = 0.0,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        maskctc_n_iterations: int = 10,
        maskctc_threshold_probability: float = 0.99,
        maskctc_init_masked: bool = True,
        category_sym_asr: str = "<lang_nl>",
        category_sym_subs: str = "<lang_nl>",
        task_sym_asr: str = "<task_asr>",
        task_sym_subs: str = "<task_subs>",
        time_sym_asr: Optional[str] = "<notimestamps>",
        time_sym_subs: Optional[str] = "<notimestamps>",
        return_verbatim_only: bool = False,
    ):

        assert check_argument_types()

        # 1. Build ASR+SUBS model
        scorers = {}
        asr_model, asr_train_args = S2TTaskSubtitle.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        # HACK
        asr_model.joint_decoder = True

        # 2. ASR decoding
        decoder = asr_model.decoder
        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list  #src_token_list

        if not hasattr(asr_model, 'task_labels'):
            self.task_labels = False
        else:
            self.task_labels = asr_model.task_labels

        if self.task_labels:
            asr_task_token = token_list.index("<verbatim>")
            subs_task_token = token_list.index("<subtitle>")
        else:
            asr_task_token, subs_task_token = None, None

        if asr_task_token is not None and subs_task_token is not None:
            logging.info("Decoding with task labels <verbatim> at idx %i and <subtitle> at idx %i." % (asr_task_token, subs_task_token))
        else:
            logging.info("Decoding without task labels.")

        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
            scorefilter=ScoreFilter(
                notimestamps=token_list.index(
                    asr_train_args.preprocessor_conf["notime_symbol"]
                ),
                utttimestamps=token_list.index(
                    asr_train_args.preprocessor_conf["notime_symbol"]
                ),
                wordtimestamps=token_list.index(
                    asr_train_args.preprocessor_conf["word_time_symbol"]
                ),
                first_time=token_list.index(
                    asr_train_args.preprocessor_conf["utt_time_symbol"]
                ),
                last_time=token_list.index(
                    asr_train_args.preprocessor_conf["last_time_symbol"]
                ),
                first_spk=token_list.index('<spk-0>') if '<spk-0>' in token_list else None,
                last_spk=token_list.index('<spk-4>') if '<spk-4>' in token_list else None,
                spk_change=token_list.index('<spk>') if '<spk>' in token_list else None,
                sos=asr_model.sos,
                eos=asr_model.eos,
                vocab_size=len(token_list),
            ),
        )

        # Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        self.decoder_mlm = asr_model.decoder_mlm if hasattr(asr_model, 'decoder_mlm') else False
        self.subtitle_decoder_mlm = asr_model.subtitle_decoder_mlm if hasattr(asr_model, 'subtitle_decoder_mlm') else False

        if self.decoder_mlm:
            raise NotImplementedError
            #vocab_size = asr_model.vocab_size - 1
            #token_list = asr_model.token_list[:-1]
        else:
            vocab_size = asr_model.vocab_size
            token_list = asr_model.token_list

        if self.subtitle_decoder_mlm:
            raise NotImplementedError
            #vocab_size = asr_model.vocab_size - 1
            #token_list = asr_model.token_list[:-1]
        else:
            vocab_size = asr_model.vocab_size
            token_list = asr_model.token_list

        if self.decoder_mlm:
            maskctc_inference = MaskCTCInference(
                decoder=asr_model.decoder,
                ctc=asr_model.ctc,
                mask_token=asr_model.src_mask_token,
                token_list=asr_model.src_token_list,
                n_iterations=maskctc_n_iterations,
                threshold_probability=maskctc_threshold_probability,
            )
            maskctc_inference.to(device=device, dtype=getattr(torch, dtype)).eval()
            beam_search = None
        else:
            maskctc_inference = None
            # Build BeamSearch object
            weights = dict(
                decoder=1.0 - ctc_weight,
                ctc=ctc_weight,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
                scorefilter=1.0,
            )

            beam_search = BeamSearch(
                beam_size=beam_size,
                weights=weights,
                scorers=scorers,
                sos=asr_model.sos,
                eos=asr_model.eos,
                vocab_size=vocab_size,
                token_list=token_list,
                pre_beam_score_key=None if ctc_weight == 1.0 else "full",
                task_token=asr_task_token,
            )

            # TODO(karita): make all scorers batchfied
            if batch_size == 1:
                non_batch = [
                    k
                    for k, v in beam_search.full_scorers.items()
                    if not isinstance(v, BatchScorerInterface)
                ]
                if len(non_batch) == 0:
                    if streaming:
                        beam_search.__class__ = BatchBeamSearchOnlineSim
                        beam_search.set_streaming_config(asr_train_config)
                        logging.info("BatchBeamSearchOnlineSim implementation is selected.")
                    else:
                        beam_search.__class__ = BatchBeamSearch
                        logging.info("BatchBeamSearch implementation is selected.")
                else:
                    logging.warning(
                        f"As non-batch scorers {non_batch} are found, "
                        f"fall back to non-batch implementation."
                    )
            logging.info
            beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            for scorer in scorers.values():
                if isinstance(scorer, torch.nn.Module):
                    scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 2: Subtitles decoder
        if asr_model.joint_decoder:
            subtitle_decoder = asr_model.decoder
        else:
            subtitle_decoder = asr_model.subtitle_decoder
        subs_token_list = asr_model.token_list

        if self.subtitle_decoder_mlm:
            subs_mlm_inference = MLMInference(
                decoder=subtitle_decoder,
                multi_decoder=asr_model.condition_subtitle_decoder,
                mask_token=asr_model.mask_token,
                token_list=subs_token_list,
                n_iterations=maskctc_n_iterations,
                threshold_probability=maskctc_threshold_probability,
                init_masked=maskctc_init_masked,
            )
            subs_mlm_inference.to(device=device, dtype=getattr(torch, dtype)).eval()
            subs_beam_search = None
        else:
            if hasattr(asr_model, 'subtitle_ctc') and asr_model.subtitle_ctc is not None and subtitle_ctc_weight > 0:
                subtitle_ctc = CTCPrefixScorer(ctc=asr_model.subtitle_ctc,
                                               eos=asr_model.eos)
            else:
                subtitle_ctc = None

            subs_mlm_inference = None
            subs_scorers = {}

            subs_scorers.update(
                decoder=subtitle_decoder,
                ctc=subtitle_ctc,
                length_bonus=LengthBonus(len(subs_token_list)),
                scorefilter=ScoreFilter(
                    notimestamps=subs_token_list.index(
                        asr_train_args.preprocessor_conf["notime_symbol"]
                    ),
                    utttimestamps=token_list.index(
                        asr_train_args.preprocessor_conf["notime_symbol"]
                    ),
                    wordtimestamps=token_list.index(
                        asr_train_args.preprocessor_conf["word_time_symbol"]
                    ),
                    first_time=token_list.index(
                        asr_train_args.preprocessor_conf["utt_time_symbol"]
                    ),
                    last_time=token_list.index(
                        asr_train_args.preprocessor_conf["last_time_symbol"]
                    ),
                    first_spk=token_list.index('<spk-0>') if '<spk-0>' in token_list else None,
                    last_spk=token_list.index('<spk-4>') if '<spk-4>' in token_list else None,
                    spk_change=token_list.index('<spk>') if '<spk>' in token_list else None,
                    sos=asr_model.sos,
                    eos=asr_model.eos,
                    vocab_size=len(subs_token_list),
                ),
            )

            # Build Language model
            if lm_train_config is not None:
                lm, lm_train_args = LMTask.build_model_from_file(
                    lm_train_config, lm_file, device
                )
                subs_scorers["lm"] = lm.lm

            # Build ngram model
            if ngram_file is not None:
                if ngram_scorer == "full":
                    from espnet.nets.scorers.ngram import NgramFullScorer

                    ngram = NgramFullScorer(ngram_file, token_list)
                else:
                    from espnet.nets.scorers.ngram import NgramPartScorer

                    ngram = NgramPartScorer(ngram_file, token_list)
            else:
                ngram = None
            subs_scorers["ngram"] = ngram

            # Build BeamSearch object
            subs_weights = dict(
                decoder=1.0 - subtitle_ctc_weight,
                ctc=subtitle_ctc_weight,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
                scorefilter=1.0,
            )

            subs_beam_search = BeamSearch(
                beam_size=beam_size,
                weights=subs_weights,
                scorers=subs_scorers,
                sos=asr_model.sos,
                eos=asr_model.eos,
                vocab_size=len(subs_token_list),
                token_list=subs_token_list,
                pre_beam_score_key=None if subtitle_ctc_weight == 1.0 else "full",
                task_token=subs_task_token,
            )

            # TODO(karita): make all scorers batchfied
            if batch_size == 1:
                non_batch = [
                    k
                    for k, v in subs_beam_search.full_scorers.items()
                    if not isinstance(v, BatchScorerInterface)
                ]
                if len(non_batch) == 0:
                    if streaming:
                        subs_beam_search.__class__ = BatchBeamSearchOnlineSim
                        subs_beam_search.set_streaming_config(asr_train_config)
                        logging.info("BatchBeamSearchOnlineSim implementation is selected.")
                    else:
                        subs_beam_search.__class__ = BatchBeamSearch
                        logging.info("BatchBeamSearch implementation is selected.")
                else:
                    logging.warning(
                        f"As non-batch scorers {non_batch} are found, "
                        f"fall back to non-batch implementation."
                    )
            subs_beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            for scorer in subs_scorers.values():
                if isinstance(scorer, torch.nn.Module):
                    scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            logging.info(f"Beam_search: {subs_beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=asr_model.token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.subs_beam_search = subs_beam_search
        self.maskctc_inference = maskctc_inference
        self.subs_mlm_inference = subs_mlm_inference
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.return_verbatim_only = return_verbatim_only

        self.condition_asr_decoder = self.asr_model.condition_asr_decoder if hasattr(self.asr_model, 'condition_asr_decoder') else False

        self.category_id_asr = converter.token2id[category_sym_asr]
        self.category_id_subs = converter.token2id[category_sym_subs]
        self.task_id_asr = converter.token2id[task_sym_asr]
        self.task_id_subs = converter.token2id[task_sym_subs]
        self.time_id_asr = converter.token2id[time_sym_asr]
        self.time_id_subs = converter.token2id[time_sym_subs]

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        text_prev: Optional[Union[torch.Tensor, np.ndarray, str]] = None,
    ) -> Union[
        ListOfHypothesis,
        Tuple[
            ListOfHypothesis,
            Optional[Dict[int, List[str]]],
        ],
    ]:
        """Inference for a short utterance

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Prepare hyp_primer
        if text_prev is not None:
            if isinstance(text_prev, str):
                text_prev = self.converter.tokens2ids(
                    self.tokenizer.text2tokens(text_prev)
                )
            else:
                text_prev = text_prev.tolist()

            # Check if text_prev is valid
            if self.asr_model.na in text_prev:
                text_prev = None

        if text_prev is not None:
            hyp_primer = (
                    [self.asr_model.sop]
                    + text_prev
                    + [self.asr_model.sos, self.category_id_asr, self.task_id_asr, self.time_id_asr]
            )
        else:
            hyp_primer = [self.asr_model.sos, self.category_id_asr, self.task_id_asr, self.time_id_asr]
        self.beam_search.set_hyp_primer(hyp_primer)

        if text_prev is not None:
            hyp_primer_subs = (
                    [self.asr_model.sop]
                    + text_prev
                    + [self.asr_model.sos, self.category_id_subs, self.task_id_subs, self.time_id_subs]
            )
        else:
            hyp_primer_subs = [self.asr_model.sos, self.category_id_subs, self.task_id_subs, self.time_id_subs]
        self.subs_beam_search.set_hyp_primer(hyp_primer_subs)

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        logging.info("speech length: " + str(speech.size(1)))

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_len = self.asr_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        # if self.condition_asr_decoder:
        #     sub_enc, sub_enc_len, _ = self.asr_model.subtitle_encoder(
        #         enc, enc_len)
        #     asr_dec_input = (enc[0], sub_enc[0])
        # else:
        #     asr_dec_input = enc[0]
        asr_dec_input = enc[0]

        # c. Passed the encoder result and the beam search
        logging.info("ASR DECODER")
        if self.decoder_mlm:
            nbest_hyps = [self.maskctc_inference(asr_dec_input)]
        else:
            nbest_hyps = self.beam_search(
                x=asr_dec_input, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )
            nbest_hyps = nbest_hyps[: self.nbest]

        results_asr = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[:last_pos]
            else:
                token_int = hyp.yseq[:last_pos].tolist()
            token_int = token_int[token_int.index(self.asr_model.sos) + 1:]

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            token_int_notime = [x for x in token_int if x not in self.asr_model.time_ids and x not in self.asr_model.spk_order_ids]

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)
            token_notime = self.converter.ids2tokens(token_int_notime)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
                text_notime = self.tokenizer.tokens2text(token_notime)
            else:
                text = None
            results_asr.append((text, token, token_int, hyp, text_notime, token_notime, token_int_notime))

        if self.return_verbatim_only:
            results_subs = [None] * len(results_asr)
            return list(zip(results_asr, results_subs))

        # # HERE COMES THE ASR DECODER (hyp.yseq) / SUBTITLE ENCODER / DECODER
        # best_asr_hyp = nbest_hyps[0].yseq.unsqueeze(0)  # <SOS> ... <EOS>
        # asr_hyp_len = torch.tensor([best_asr_hyp.size(1)])  # includes <EOS>
        #
        # # TODO: Jakob, can we not use nbest_hyps[0].states['decoder'] here? And then final layer states?
        # if self.asr_model.use_asr_feats == "decoder":
        #     asr_decoder_out, asr_decoder_out_lens = self.asr_model.decoder(
        #         enc, enc_len, best_asr_hyp, asr_hyp_len
        #     )
        #
        #     if isinstance(asr_decoder_out, tuple):
        #         asr_decoder_hidden = asr_decoder_out[1]
        #         asr_decoder_out = asr_decoder_out[0]
        #
        # if self.asr_model.subtitle_encoder is not None:
        #     if self.asr_model.condition_asr_decoder:
        #         # already computed them
        #         sub_encoder_out, sub_encoder_out_lens = sub_enc, sub_enc_len
        #     else:
        #         if self.asr_model.use_asr_feats == "encoder":
        #             sub_encoder_out, sub_encoder_out_lens, _ = self.asr_model.subtitle_encoder(
        #                 enc, enc_len)
        #         elif self.asr_model.use_asr_feats == "decoder":
        #             sub_encoder_out, sub_encoder_out_lens, _ = self.asr_model.subtitle_encoder(
        #                 asr_decoder_hidden, asr_hyp_len)
        #         else:
        #             raise ValueError('subtitle_encoder param use_asr_feats should '
        #                              'be one of encoder/decoder.')
        # else:
        #     if self.asr_model.joint_decoder:
        #         sub_encoder_out, sub_encoder_out_lens = enc, enc_len
        #     elif self.asr_model.use_asr_feats == "encoder":
        #         sub_encoder_out, sub_encoder_out_lens = enc, enc_len
        #     elif self.asr_model.use_asr_feats == "decoder":
        #         sub_encoder_out, sub_encoder_out_lens = asr_decoder_hidden, asr_hyp_len
        #     else:
        #         raise ValueError('subtitle_encoder param use_asr_feats should '
        #                          'be one of encoder/decoder.')
        #
        # if self.asr_model.condition_subtitle_decoder:
        #     subs_enc = (sub_encoder_out[0], enc[0])
        # else:
        #     subs_enc = sub_encoder_out[0]

        logging.info("SUBTITLE DECODER")
        subs_enc = asr_dec_input

        if self.subtitle_decoder_mlm:
            # TODO JAKOB --> hier lengte = asr utt gekozen
            subs_nbest_hyps = [self.subs_mlm_inference(subs_enc, asr_hyp_len)]
        else:
            subs_nbest_hyps = self.subs_beam_search(
                x=subs_enc, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )
            subs_nbest_hyps = subs_nbest_hyps[: self.nbest]

        results_subs = []
        for hyp in subs_nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[:last_pos]
            else:
                token_int = hyp.yseq[:last_pos].tolist()
            token_int = token_int[token_int.index(self.asr_model.sos) + 1:]

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            token_int_notime = [x for x in token_int if
                                x not in self.asr_model.time_ids and x not in self.asr_model.spk_order_ids]

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)
            token_notime = self.converter.ids2tokens(token_int_notime)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
                text_notime = self.tokenizer.tokens2text(token_notime)
            else:
                text = None
            results_subs.append((text, token, token_int, hyp, text_notime, token_notime, token_int_notime))

        #assert check_return_type(results_asr)
        #assert check_return_type(results_subs)

        return list(zip(results_asr, results_subs))


    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Text instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2TextAndSubs(**kwargs)


def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    subtitle_ctc_weight: float,
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    st_train_config: Optional[str],
    st_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    src_token_type: Optional[str],
    bpemodel: Optional[str],
    src_bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    streaming: bool,
    maskctc_n_iterations: int,
    maskctc_threshold_probability: float,
    maskctc_init_masked: bool,
    return_verbatim_only: bool = False,
    category_sym_asr: str = "<lang_nl>",
    category_sym_subs: str = "<lang_nl>",
    task_sym_asr: str = "<task_asr>",
    task_sym_subs: str = "<task_asr>",
    time_sym_asr: Optional[str] = "<notimestamps>",
    time_sym_subs: Optional[str] = "<notimestamps>",
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=st_train_config,
        asr_model_file=st_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        subtitle_ctc_weight=subtitle_ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
        maskctc_n_iterations=maskctc_n_iterations,
        maskctc_threshold_probability=maskctc_threshold_probability,
        maskctc_init_masked=maskctc_init_masked,
        category_sym_asr=category_sym_asr,
        category_sym_subs=category_sym_subs,
        task_sym_asr=task_sym_asr,
        task_sym_subs=task_sym_subs,
        time_sym_asr=time_sym_asr,
        time_sym_subs=time_sym_subs,
        return_verbatim_only=return_verbatim_only,
    )

    speech2textandsubs = Speech2TextAndSubs.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = S2TTaskSubtitle.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=S2TTaskSubtitle.build_preprocess_fn(speech2textandsubs.asr_train_args, False),
        collate_fn=S2TTaskSubtitle.build_collate_fn(speech2textandsubs.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2textandsubs(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp]] * nbest

            # Only supporting batch_size==1
            key = keys[0]
            logging.info("KEY: %s" % str(key))
            for n, (asr_results, subs_results) in zip(range(1, nbest + 1), results):
                text_asr, token_asr, token_int_asr, hyp_asr, text_asr_notime, token_asr_notime, token_int_asr_notime = asr_results

                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token_asr)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int_asr))
                ibest_writer["score"][key] = str(hyp_asr.score)

                if text_asr is not None:
                    ibest_writer["text"][key] = text_asr

                if text_asr_notime is not None:
                    ibest_writer["text_notime"][key] = text_asr_notime

                logging.info("ASR: %s" % str(text_asr))

                if return_verbatim_only:
                    logging.info("")
                    continue

                text_subs, token_subs, token_int_subs, hyp_subs, text_subs_notime, token_subs_notime, token_int_subs_notime = subs_results

                # Write the result to each file
                ibest_writer["token_subs"][key] = " ".join(token_subs)
                ibest_writer["token_int_subs"][key] = " ".join(map(str, token_int_subs))
                ibest_writer["score_subs"][key] = str(hyp_subs.score)

                if text_subs is not None:
                    ibest_writer["subs"][key] = text_subs

                if text_subs_notime is not None:
                    ibest_writer["subs_notime"][key] = text_subs_notime

                logging.info("SUB: %s" % str(text_subs))
                logging.info("")

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--st_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--st_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--ngram_file",
        type=str,
        help="N-gram parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths."
        "If maxlenratio<0.0, its absolute value is interpreted"
        "as a constant max output length",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )

    group.add_argument(
        "--subtitle_ctc_weight",
        type=float,
        default=0.0,
        help="Subtitle CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)
    group.add_argument("--maskctc_n_iterations", type=int, default=10,
                       help="Number of maskctc decoding iterations")
    group.add_argument("--maskctc_threshold_probability", type=float, default=0.99,
                       help="Treshold probability in maskctc decoding")
    group.add_argument("--maskctc_init_masked", type=bool, default=True,
                       help="Initialize mlm decoding with all mask tokens")

    group.add_argument("--category_sym_asr", type=str, default="<lang-nl>",
                       help="Language to decode")
    group.add_argument("--task_sym_asr", type=str, default="<task_asr>",
                       help="Task to do: transcription or subtitling")
    group.add_argument("--time_sym_asr", type=str, default="<utttimestamps>",
                       help="Generate word-level, utterance-level or no timestamps")
    group.add_argument("--category_sym_subs", type=str, default="<lang-nl>",
                       help="Language to decode")
    group.add_argument("--task_sym_subs", type=str, default="<task_subs>",
                       help="Task to do: transcription or subtitling")
    group.add_argument("--time_sym_subs", type=str, default="<utttimestamps>",
                       help="Generate word-level, utterance-level or no timestamps")
    group.add_argument("--return_verbatim_only", type=bool, default=False, help="Only compute verbatim output")

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--src_token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--src_bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
             "If not given, refers from the training args",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    print(kwargs)
    inference(**kwargs)


if __name__ == "__main__":
    main()
