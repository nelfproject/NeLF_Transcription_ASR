#!/usr/bin/env python3
import os
from espnet2.tasks.st import STTask

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].replace("CUDA", "")

def get_parser():
    parser = STTask.get_parser()
    return parser


def main(cmd=None):
    r"""ST training.
    Example:
        % python st_train.py st --print_config --optim adadelta \
                > conf/train_st.yaml
        % python st_train.py --config conf/train_st.yaml
    """
    STTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
