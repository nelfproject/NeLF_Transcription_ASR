#!/usr/bin/env python3
import os
from espnet2.tasks.subtitle_chained import SubtitleTask

if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].replace("CUDA", "")


def get_parser():
    parser = SubtitleTask.get_parser()
    return parser


def main(cmd=None):
    r"""Subtitling+asr training.
    """
    SubtitleTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
