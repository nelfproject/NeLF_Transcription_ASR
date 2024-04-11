#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import List

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
from espnet2.tasks.subtitle_chained import SubtitleTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


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
    ):

        assert check_argument_types()

        # 1. Build ASR+SUBS model
        scorers = {}
        asr_model, asr_train_args = SubtitleTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        # 2. ASR decoding
        decoder = asr_model.decoder
        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list  #src_token_list

        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
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

        if asr_model.subtitle_decoder_mlm:
            vocab_size = asr_model.vocab_size - 1
            token_list = asr_model.token_list[:-1]
        else:
            vocab_size = asr_model.vocab_size
            token_list = asr_model.token_list

        self.decoder_mlm = asr_model.decoder_mlm
        self.subtitle_decoder_mlm = asr_model.subtitle_decoder_mlm

        if self.subtitle_decoder_mlm:
            vocab_size = asr_model.vocab_size - 1
            token_list = asr_model.token_list[:-1]
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
            beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            for scorer in scorers.values():
                if isinstance(scorer, torch.nn.Module):
                    scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 2: Subtitles decode
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

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_len = self.asr_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        # c. Passed the encoder result and the beam search
        logging.info("ASR DECODER")
        if self.decoder_mlm:
            nbest_hyps = [self.maskctc_inference(enc[0])]
        else:
            nbest_hyps = self.beam_search(
                x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )
            nbest_hyps = nbest_hyps[: self.nbest]

        results_asr = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results_asr.append((text, token, token_int, hyp))

        # HERE COMES THE ASR DECODER (hyp.yseq) / SUBTITLE ENCODER / DECODER
        best_asr_hyp = nbest_hyps[0].yseq.unsqueeze(0)  # <SOS> ... <EOS>
        asr_hyp_len = torch.tensor([best_asr_hyp.size(1)])  # includes <EOS>

        # TODO: Jakob, can we not use nbest_hyps[0].states['decoder'] here? And then final layer states?
        asr_decoder_out, asr_decoder_out_lens = self.asr_model.decoder(
            enc, enc_len, best_asr_hyp, asr_hyp_len
        )

        if isinstance(asr_decoder_out, tuple):
            asr_decoder_hidden = asr_decoder_out[1]
            asr_decoder_out = asr_decoder_out[0]

        if self.asr_model.subtitle_encoder is not None:
            if self.asr_model.use_asr_feats == "encoder":
                sub_encoder_out, sub_encoder_out_lens, _ = self.asr_model.subtitle_encoder(
                    enc, enc_len)
            elif self.asr_model.use_asr_feats == "decoder":
                sub_encoder_out, sub_encoder_out_lens, _ = self.asr_model.subtitle_encoder(
                    asr_decoder_hidden, asr_hyp_len)
            else:
                raise ValueError('subtitle_encoder param use_asr_feats should '
                                 'be one of encoder/decoder.')
        else:
            if self.asr_model.use_asr_feats == "encoder":
                sub_encoder_out, sub_encoder_out_lens = enc, enc_len
            elif self.asr_model.use_asr_feats == "decoder":
                sub_encoder_out, sub_encoder_out_lens = asr_decoder_hidden, asr_hyp_len
            else:
                raise ValueError('subtitle_encoder param use_asr_feats should '
                                 'be one of encoder/decoder.')

        if self.asr_model.condition_subtitle_decoder:
            subs_enc = (sub_encoder_out[0], enc[0])
        else:
            subs_enc = sub_encoder_out[0]

        logging.info("SUBTITLE DECODER")
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
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results_subs.append((text, token, token_int, hyp))

        assert check_return_type(results_asr)
        assert check_return_type(results_subs)

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
    )

    speech2textandsubs = Speech2TextAndSubs.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = SubtitleTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=SubtitleTask.build_preprocess_fn(speech2textandsubs.asr_train_args, False),
        collate_fn=SubtitleTask.build_collate_fn(speech2textandsubs.asr_train_args, False),
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
            for n, (asr_results, subs_results) in zip(range(1, nbest + 1), results):
                text_asr, token_asr, token_int_asr, hyp_asr = asr_results
                text_subs, token_subs, token_int_subs, hyp_subs = subs_results

                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token_asr)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int_asr))
                ibest_writer["score"][key] = str(hyp_asr.score)

                if text_asr is not None:
                    ibest_writer["text"][key] = text_asr

                # Write the result to each file
                ibest_writer["token_subs"][key] = " ".join(token_subs)
                ibest_writer["token_int_subs"][key] = " ".join(map(str, token_int_subs))
                ibest_writer["score_subs"][key] = str(hyp_subs.score)

                if text_subs is not None:
                    ibest_writer["subs"][key] = text_subs

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
