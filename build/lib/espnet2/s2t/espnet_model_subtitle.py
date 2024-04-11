from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy, pad_list
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.multi_mlm_decoder import MultiMLMDecoder
from espnet2.asr.decoder.abs_decoder import AbsDecoder, AbsMultiDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.text.token_id_converter import TokenIDConverter

from espnet2.subtitling.weakly_supervised_loss import WeaklySupervisedLoss

from espnet2.s2t.label_smoothing_loss_conv import ConvLabelSmoothingLoss

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

def add_sos_eos_index(ys_pad, sos, eos, sos_index, ignore_id,
                      mask_index=False):
    """Add <sos> at a specific sos_index and <eos> labels at the end.
    (for example, useful if first label is a task label like <transcribe>)
    If mask_index is True, then mask all out labels until sos_index with ignore_id.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int sos: index of <sos>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    _sos = ys_pad.new([sos])
    _eos = ys_pad.new([eos])
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    if sos_index <= 0:
        ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    else:
        ys_in = [torch.cat([y[:sos_index], _sos, y[sos_index:]], dim=0) for y in ys]
        ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
        if mask_index:
            for idx in range(len(ys_out)):
                ys_out[idx][:sos_index] = ignore_id
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

class ESPnetS2TSubtitleModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Union[AbsDecoder, AbsMultiDecoder],
        subtitle_encoder: Optional[AbsEncoder],
        subtitle_decoder: Optional[Union[AbsDecoder, AbsMultiDecoder]],
        ctc: CTC,
        subtitle_ctc: Union[CTC, None],
        interctc_weight: float = 0.0,
        src_vocab_size: int = 0,
        src_token_list: Union[Tuple[str, ...], List[str]] = [],
        asr_weight: float = 0.0,
        subs_weight: float = 0.0,
        ctc_weight: float = 0.0,
        subtitle_ctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight_asr: float = 0.0,
        lsm_weight_mt: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        report_bleu: bool = False,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_mask: str = "<mask>",
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        sym_sop: str = "<sop>",  # start of prev
        sym_na: str = "<na>",  # not available
        extract_feats_in_collect_stats: bool = True,
        condition_subtitle_decoder: bool = False,
        condition_asr_decoder: bool = False,
        use_asr_feats: str = "encoder",  # decoder
        asr_criterion: str = "sequential",
        subs_criterion: str = "sequential",
        joint_decoder: bool = False,
        task_labels: bool = False,
    ):
        assert check_argument_types()
        assert 0.0 <= asr_weight <= 1.0, "asr_weight should be [0.0, 1.0]"
        assert 0.0 <= subs_weight <= 1.0, "subs_weight should be [0.0, 1.0]"
        assert 0.0 <= ctc_weight <= 1.0, "ctc_weight should be [0.0, 1.0]"
        assert 0.0 <= interctc_weight <= 1.0, "interctc_weight should be [0.0, 1.0]"
        assert 0.0 <= subtitle_ctc_weight <= 1.0, "subtitle_ctc_weight should be [0.0, 1.0]"

        super().__init__()

        self.blank_id = token_list.index(sym_blank)
        self.sos = token_list.index(sym_sos)
        self.eos = token_list.index(sym_eos)
        self.sop = token_list.index(sym_sop)
        self.na = token_list.index(sym_na)

        self.task_labels = task_labels
        self.vocab_size = vocab_size
        self.src_vocab_size = src_vocab_size
        self.ignore_id = ignore_id
        self.asr_weight = asr_weight
        self.subs_weight = subs_weight
        self.token_list = token_list.copy()
        self.interctc_weight = interctc_weight
        self.ctc_weight = ctc_weight
        self.subtitle_ctc_weight = subtitle_ctc_weight

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        self.decoder = decoder
        self.subtitle_encoder = subtitle_encoder
        self.subtitle_decoder = subtitle_decoder
        self.ctc = ctc
        self.subtitle_ctc = subtitle_ctc
        self.joint_decoder = joint_decoder


        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                src_vocab_size, self.encoder.output_size()
            )

        if isinstance(self.decoder, MLMDecoder):
            self.decoder_mlm = True
            # Add <mask> and override inherited fields
            src_token_list.append(sym_mask)
            src_vocab_size += 1
            self.src_vocab_size = src_vocab_size
            self.src_mask_token = src_vocab_size - 1
            self.src_token_list = src_token_list.copy()
        else:
            self.decoder_mlm = False

        if isinstance(self.subtitle_decoder, MLMDecoder) or isinstance(self.subtitle_decoder, MultiMLMDecoder):
            self.subtitle_decoder_mlm = True
            # Add <mask> and override inherited fields
            token_list.append(sym_mask)
            vocab_size += 1
            self.vocab_size = vocab_size
            self.mask_token = vocab_size - 1
            self.token_list = token_list.copy()
        else:
            self.subtitle_decoder_mlm = False

        self.condition_subtitle_decoder = condition_subtitle_decoder
        self.condition_asr_decoder = condition_asr_decoder
        self.use_asr_feats = use_asr_feats

        assert self.use_asr_feats in ["encoder", "decoder", "encoder-decoder"]
        if self.condition_asr_decoder:
            assert self.use_asr_feats == "encoder"

        if subs_criterion == 'sequential':
            self.criterion_subs = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight_mt,
                normalize_length=length_normalized_loss,
            )
        elif subs_criterion == 'collapsed':
            self.criterion_subs = WeaklySupervisedLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight_mt,
                normalize_length=length_normalized_loss,
            )
        else:
            raise NotImplementedError('Unknown asr_criterion %s, '
                                      'should be sequential (=normal) or '
                                      'collapsed (=weakly supervised)' % asr_criterion)

        if asr_criterion == 'sequential':
            self.criterion_asr = LabelSmoothingLoss(
                size=src_vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight_asr,
                normalize_length=length_normalized_loss,
            )
        elif asr_criterion == 'collapsed':
            self.criterion_asr = WeaklySupervisedLoss(
                size=src_vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight_asr,
                normalize_length=length_normalized_loss,
            )
        else:
            raise NotImplementedError('Unknown asr_criterion %s, '
                                      'should be sequential (=normal) or '
                                      'collapsed (=weakly supervised)' % asr_criterion)

        # MT error calculator
        if report_bleu:
            self.subtitle_error_calculator = MTErrorCalculator(
                token_list, sym_space, sym_blank, report_bleu
            )
        else:
            self.subtitle_error_calculator = None

        # ASR error calculator
        if report_cer or report_wer:
            assert (
                src_token_list is not None
            ), "Missing src_token_list, cannot add asr module to st model"
            self.asr_error_calculator = ASRErrorCalculator(
                src_token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.asr_error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__
        if self.is_encoder_whisper:
            assert (
                    self.frontend is None
            ), "frontend should be None when using full Whisper model"

        logging.info(self)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        text_ctc: torch.Tensor,
        text_ctc_lengths: torch.Tensor
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            text_prev: (Batch, Length)
            text_prev_lengths: (Batch,)
            text_ctc: (Batch, Length)
            text_ctc_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == text_prev.shape[0]
            == text_prev_lengths.shape[0]
            == text_ctc.shape[0]
            == text_ctc_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            text_prev.shape,
            text_prev_lengths.shape,
            text_ctc.shape,
            text_ctc_lengths.shape,
        )
        batch_size = speech.shape[0]

        # -1 is used as padding index in collate fn
        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]
        if src_text is not None:
            src_text = src_text[:, : src_text_lengths.max()]

        # Define stats to report
        loss_ctc, cer_ctc, loss_interctc = 0.0, None, 0.0
        loss_asr_att, acc_asr_att, cer_asr_att, wer_asr_att = 0.0, None, None, None
        loss_subs_att, acc_subs_att, bleu_subs = 0.0, None, None
        global_acc = None
        stats = dict()

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # 2. CTC branch
        if self.ctc_weight > 0.0:
            mask = text_ctc.sum(-1) > 0
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text_ctc, text_ctc_lengths, mask
            )

            # Collect CTC branch stats
            if loss_ctc is not None:
                stats["loss_ctc"] = loss_ctc.detach() if type(loss_ctc) is not float else loss_ctc
            else:
                stats["loss_ctc"] = None
            stats["cer_ctc"] = cer_ctc

        # 2a. Intermediate CTC (optional)
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            mask = text_ctc.sum(-1) > 0
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text_ctc, text_ctc_lengths, mask
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                if loss_ic is not None:
                    stats["loss_interctc_layer{}".format(layer_idx)] = (
                        loss_ic.detach() if type(loss_ic) is not float else loss_ic
                    )
                else:
                    stats["loss_interctc_layer{}".format(layer_idx)] = (None)
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (1 - self.interctc_weight) * loss_ctc + self.interctc_weight * loss_interctc

        # 3. Decoders
        mask_asr = text_ctc.sum(-1) > 0
        mask_subs = text_ctc.sum(-1) <= 0

        loss_asr_att, acc_asr_att, cer_asr_att, \
        wer_asr_att, loss_subs_att, acc_subs_att, _, \
        loss_subtitle_ctc, cer_subtitle_ctc = \
            self._calc_att_loss(encoder_out, encoder_out_lens,
                                text, text_lengths,
                                text_prev, text_prev_lengths,
                                mask_asr, mask_subs)

        # 4. Loss computation
        #loss = self.ctc_weight * loss_ctc + self.asr_weight * loss_asr_att + self.subs_weight * loss_subs_att + self.subtitle_ctc_weight * loss_subtitle_ctc
        loss = self.asr_weight * (self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_asr_att) + self.subs_weight * (self.subtitle_ctc_weight * loss_subtitle_ctc + (1 - self.subtitle_ctc_weight) * loss_subs_att)

        if acc_asr_att is not None and acc_subs_att is not None:
            global_acc = self.asr_weight * acc_asr_att + self.subs_weight * acc_subs_att

        stats = dict(
            loss=loss.detach() if type(loss) is not float else loss,
            loss_asr=loss_asr_att.detach() if type(loss_asr_att) is not float else loss_asr_att,
            loss_subs=loss_subs_att.detach() if type(loss_subs_att) is not float else loss_subs_att,
            loss_ctc=loss_ctc.detach() if type(loss_ctc) is not float else loss_ctc,
            loss_interctc=loss_interctc.detach() if type(loss_interctc) is not float else loss_interctc,
            acc_asr=acc_asr_att,
            acc_subs=acc_subs_att,
            acc=global_acc,
            cer_ctc=cer_ctc,
            cer=cer_asr_att,
            wer=wer_asr_att,
            bleu=bleu_subs,
            loss_subtitle_ctc=loss_subtitle_ctc.detach() if type(loss_subtitle_ctc) is not float else loss_subtitle_ctc,
            cer_subtitle_ctc=cer_subtitle_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size),
                                               loss.device if type(loss) is not float else speech.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: Optional[torch.Tensor],
        src_text_lengths: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by st_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
            ys_prev_pad: torch.Tensor,
            ys_prev_pad_lens: torch.Tensor,
            mask_asr: torch.Tensor,
            mask_subs: torch.Tensor,
    ):
        # encoder_out: shape [BS, T, D]  e.g. [32, 47, 256]

        # 0. Prepare input and output with sos, eos, sop
        ys = [y[y != self.ignore_id] for y in ys_pad]
        ys_prev = [y[y != self.ignore_id] for y in ys_prev_pad]

        _sos = ys_pad.new([self.sos])
        _eos = ys_pad.new([self.eos])
        _sop = ys_pad.new([self.sop])
        ys_in = []
        ys_in_lens = []
        ys_out = []

        #########TODO from here ###########
        # https://github.com/espnet/espnet/blob/master/espnet2/s2t/espnet_model.py
        # /esat/audioslave/btamm/espnet/espnet2/s2t/espnet_model.py

        # 0. ST ENCODER FORWARD (if ASR DECODER conditions on this output)
        if self.condition_asr_decoder:
            sub_encoder_out, sub_encoder_out_lens, _ = self.subtitle_encoder(
                encoder_out, encoder_out_lens)

        # 1. ASR DECODER FORWARD
        if self.decoder_mlm:
            ys_in_pad_asr, ys_out_pad_asr = mask_uniform(
                ys_pad_asr, self.src_mask_token, self.eos, self.ignore_id
            )
            ys_in_lens_asr = ys_pad_lens_asr
        else:
            ys_in_pad_asr, ys_out_pad_asr = add_sos_eos(ys_pad_asr, self.sos, self.eos, self.ignore_id)
            ys_in_lens_asr = ys_pad_lens_asr + 1

        if (self.use_asr_feats == "encoder") and (self.asr_weight == 0.0):
            asr_decoder_out, asr_decoder_out_lens = None, None
        else:
            if self.condition_asr_decoder:
                asr_decoder_out, asr_decoder_out_lens = self.decoder(
                    encoder_out, encoder_out_lens, sub_encoder_out, sub_encoder_out_lens, ys_in_pad_asr, ys_in_lens_asr
                )
            else:
                asr_decoder_out, asr_decoder_out_lens = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad_asr, ys_in_lens_asr
                )

        if isinstance(asr_decoder_out, tuple):
            asr_decoder_hidden = asr_decoder_out[1]
            asr_decoder_out = asr_decoder_out[0]

            # asr_decoder_out: shape [BS, L, V] e.g. [32, 14, 5000]
        # hidden: [BS, L, D] e.g. [32, 4, 256]

        # 2. ASR DECODER LOSS
        if (mask_asr.sum() == 0) or (self.asr_weight == 0.0):
            loss_asr_att, acc_asr_att, cer_asr_att, wer_asr_att = 0.0, None, None, None

        else:
            asr_decoder_out_m = asr_decoder_out[mask_asr]
            ys_out_pad_asr_m = ys_out_pad_asr[mask_asr]
            ys_pad_asr_m = ys_pad_asr[mask_asr]

            loss_asr_att = self.criterion_asr(asr_decoder_out_m, ys_out_pad_asr_m)
            acc_asr_att = th_accuracy(
                asr_decoder_out_m.view(-1, self.src_vocab_size),
                ys_out_pad_asr_m,
                ignore_label=self.ignore_id,
            )

            # Compute cer/wer using attention-decoder
            if self.training or self.asr_error_calculator is None:
                cer_asr_att, wer_asr_att = None, None
            else:
                ys_hat_asr = asr_decoder_out_m.argmax(dim=-1)
                cer_asr_att, wer_asr_att = self.asr_error_calculator(ys_hat_asr.cpu(), ys_pad_asr_m.cpu())

        # 3. ST ENCODER FORWARD
        if self.subtitle_encoder is not None and not self.condition_asr_decoder:
            if self.use_asr_feats == "encoder":
                sub_encoder_out, sub_encoder_out_lens, _ = self.subtitle_encoder(
                    encoder_out, encoder_out_lens)
            elif self.use_asr_feats == "decoder":
                sub_encoder_out, sub_encoder_out_lens, _ = self.subtitle_encoder(
                    asr_decoder_hidden, ys_in_lens_asr)
            elif self.use_asr_feats == "encoder-decoder":
                sub_enc_input = torch.cat((asr_decoder_hidden, encoder_out), dim=1)
                sub_enc_input_len = ys_in_lens_asr + encoder_out_lens
                sub_encoder_out, sub_encoder_out_lens, _ = self.subtitle_encoder(sub_enc_input, sub_enc_input_len)
            else:
                raise ValueError('subtitle_encoder param use_asr_feats should '
                                 'be one of encoder/decoder/encoder-decoder.')
        else:
            if self.use_asr_feats == "encoder":
                sub_encoder_out, sub_encoder_out_lens = encoder_out, encoder_out_lens
            elif self.use_asr_feats == "decoder":
                sub_encoder_out, sub_encoder_out_lens = asr_decoder_hidden, ys_in_lens_asr
            elif self.use_asr_feats == "encoder-decoder":
                sub_encoder_out = torch.cat((asr_decoder_hidden, encoder_out), dim=1)
                sub_encoder_out_lens = ys_in_lens_asr + encoder_out_lens
            else:
                raise ValueError('subtitle_encoder param use_asr_feats should '
                                 'be one of encoder/decoder.')

        # 3.1 (Optional) CTC branch
        if self.subtitle_ctc_weight > 0.0:
            loss_subtitle_ctc, cer_subtitle_ctc = self._calc_subtitle_ctc_loss(
                sub_encoder_out, sub_encoder_out_lens, ys_pad_subs, ys_pad_lens_subs, mask_subs
            )
        else:
            loss_subtitle_ctc, cer_subtitle_ctc = 0.0, None

        # 4. ST DECODER FORWARD
        if self.subtitle_decoder_mlm:
            ys_in_pad_subs, ys_out_pad_subs = mask_uniform(
                ys_pad_subs, self.mask_token, self.eos, self.ignore_id
            )
            ys_in_lens_subs = ys_pad_lens_subs
        else:
            ys_in_pad_subs, ys_out_pad_subs = add_sos_eos(ys_pad_subs, self.sos, self.eos, self.ignore_id)
            ys_in_lens_subs = ys_pad_lens_subs + 1

        if self.condition_subtitle_decoder:
            subs_decoder_out, _ = self.subtitle_decoder(
                sub_encoder_out, sub_encoder_out_lens, encoder_out, encoder_out_lens, ys_in_pad_subs, ys_in_lens_subs
            )
        else:
            subs_decoder_out, _ = self.subtitle_decoder(
                sub_encoder_out, sub_encoder_out_lens, ys_in_pad_subs, ys_in_lens_subs
            )

        # 5. ST DECODER LOSS
        if mask_subs.sum() == 0:
            loss_subs_att, acc_subs_att, bleu_subs_att = 0.0, None, None
        else:
            subs_decoder_out_m = subs_decoder_out[mask_subs]
            ys_out_pad_subs_m = ys_out_pad_subs[mask_subs]
            ys_pad_subs_m = ys_pad_subs[mask_subs]

            # 2. Compute attention loss
            loss_subs_att = self.criterion_subs(subs_decoder_out_m, ys_out_pad_subs_m)
            acc_subs_att = th_accuracy(
                subs_decoder_out_m.view(-1, self.vocab_size),
                ys_out_pad_subs_m,
                ignore_label=self.ignore_id,
            )

            # Compute cer/wer using attention-decoder
            if self.training or self.subtitle_error_calculator is None:
                bleu_subs_att = None
            else:
                ys_hat_subs = subs_decoder_out_m.argmax(dim=-1)
                bleu_subs_att = self.subtitle_error_calculator(ys_hat_subs.cpu(), ys_pad_subs_m.cpu())

        return loss_asr_att, acc_asr_att, cer_asr_att, wer_asr_att, loss_subs_att, acc_subs_att, bleu_subs_att, loss_subtitle_ctc, cer_subtitle_ctc

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Mask
        if mask.sum() == 0:
            return 0.0, None

        encoder_out_m = encoder_out[mask]
        encoder_out_lens_m = encoder_out_lens[mask]
        ys_pad_m = ys_pad[mask]
        ys_pad_lens_m = ys_pad_lens[mask]

        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out_m, encoder_out_lens_m, ys_pad_m, ys_pad_lens_m)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.asr_error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out_m).data
            cer_ctc = self.asr_error_calculator(ys_hat.cpu(), ys_pad_m.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_subtitle_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Mask
        if mask.sum() == 0:
            return 0.0, None

        encoder_out_m = encoder_out[mask]
        encoder_out_lens_m = encoder_out_lens[mask]
        ys_pad_m = ys_pad[mask]
        ys_pad_lens_m = ys_pad_lens[mask]

        # Calc CTC loss
        loss_ctc = self.subtitle_ctc(encoder_out_m, encoder_out_lens_m, ys_pad_m, ys_pad_lens_m)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.asr_error_calculator is not None:
            ys_hat = self.subtitle_ctc.argmax(encoder_out_m).data
            cer_ctc = self.asr_error_calculator(ys_hat.cpu(), ys_pad_m.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
