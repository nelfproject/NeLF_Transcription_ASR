#!/usr/bin/env python3
from espnet2.tasks.s2t_subs import S2TTaskSubtitle


def get_parser():
    parser = S2TTaskSubtitle.get_parser()
    return parser


def main(cmd=None):
    r"""S2T training.

    Example:

        % python s2t_train.py s2t --print_config --optim adadelta \
                > conf/train_s2t.yaml
        % python s2t_train.py --config conf/train_s2t.yaml
    """
    S2TTaskSubtitle.main(cmd=cmd)


if __name__ == "__main__":
    main()