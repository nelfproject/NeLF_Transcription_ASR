from typing import Iterator
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.fileio.read_text import read_2column_text
from espnet2.samplers.abs_sampler import AbsSampler


class CustomFoldedBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        fold_lengths: Sequence[int],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        utt2category_file: str = None,
        batch_asr_ratio: float = None,
    ):
        assert check_argument_types()
        assert batch_size > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_size = batch_size
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        self.custom_batch = True
        if batch_asr_ratio is not None:
            self.batch_weights = [batch_asr_ratio, 1 - batch_asr_ratio]
        else:
            self.batch_weights = [0.5, 0.5]
        self.categories = ['verbatim', 'subtitle']  # 'transcribed_subtitle'
        assert len(self.categories) == len(self.batch_weights)
        assert sum(self.batch_weights) == 1.0

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")

        category2utt = {k: [] for k in self.categories}
        if utt2category_file is not None:
            utt2category = read_2column_text(utt2category_file)
            if set(utt2category) != set(first_utt2shape):
                raise RuntimeError(
                    "keys are mismatched between "
                    f"{utt2category_file} != {shape_files[0]}"
                )
            for k in keys:
                category2utt[utt2category[k]].append(k)
        else:
            raise NotImplementedError

        hack_batch_cat = False
        if batch_asr_ratio == 1.0:
            self.categories = ['verbatim']
            self.batch_weights = [1.0]
            hack_batch_cat = True
        elif batch_asr_ratio == 0.0:
            self.categories = ['subtitle']
            self.batch_weights = [1.0]
            hack_batch_cat = True

        self.batch_list = []

        if self.custom_batch:
            batch_size_per_cat = [int(batch_size * bw) for bw in self.batch_weights]
            batch_sizes = [[] for _ in self.categories]

            for idx, cat in enumerate(self.categories):
                start = 0
                while True:
                    k = category2utt[cat][start]
                    factor = max(int(d[k][0] / m) for d, m in zip(utt2shapes, fold_lengths))
                    bs = max(min_batch_size, int(batch_size_per_cat[idx] / (1 + factor)))
                    if self.drop_last and start + bs > len(category2utt[cat]):
                        # This if-block avoids 0-batches
                        if len(self.batch_list) > 0:
                            break

                    bs = min(len(category2utt[cat]) - start, bs)
                    batch_sizes[idx].append(bs)
                    start += bs
                    if start >= len(category2utt[cat]):
                        break

                if len(batch_sizes[idx]) == 0:
                    # Maybe we can't reach here
                    raise RuntimeError("0 batches for category %s" % cat)

                # If the last batch-size is smaller than minimum batch_size,
                # the samples are redistributed to the other mini-batches
                if len(batch_sizes[idx]) > 1 and batch_sizes[idx][-1] < min_batch_size:
                    for i in range(batch_sizes[idx].pop(-1)):
                        batch_sizes[idx][-(i % len(batch_sizes)) - 2] += 1

                if not self.drop_last:
                    # Bug check
                    assert sum(batch_sizes[idx]) == len(
                        category2utt[cat]
                    ), f"{sum(batch_sizes)} != {len(category2utt[cat])}"

            if not hack_batch_cat:
                assert sum([sum(batch_sizes[i]) for i in range(len(self.categories))]) == len(set(first_utt2shape))

            # Set mini-batch
            cur_batch_list = []
            starts_per_cat = [0 for _ in self.categories]
            offset_per_cat = [0 for _ in self.categories]

            total_num_batches = max([len(x) for x in batch_sizes])

            for idx in range(total_num_batches):
                minibatch_keys = []
                for j, cat in enumerate(self.categories):
                    if (idx - offset_per_cat[j]) == len(batch_sizes[j]):  # used up all batches for this category, start over at 0
                        offset_per_cat[j] = idx
                        starts_per_cat[j] = 0

                    start = starts_per_cat[j]
                    bs = batch_sizes[j][idx - offset_per_cat[j]]
                    minibatch = category2utt[cat][start: start + bs]

                    if sort_in_batch == "descending":
                        minibatch.reverse()

                    minibatch_keys.extend(minibatch)
                    starts_per_cat[j] += bs

                cur_batch_list.append(tuple(minibatch_keys))

            self.batch_list.extend(cur_batch_list)

        else:
            for d, v in category2utt.items():
                category_keys = v
                # Decide batch-sizes
                start = 0
                batch_sizes = []
                while True:
                    k = category_keys[start]
                    factor = max(int(d[k][0] / m) for d, m in zip(utt2shapes, fold_lengths))
                    bs = max(min_batch_size, int(batch_size / (1 + factor)))
                    if self.drop_last and start + bs > len(category_keys):
                        # This if-block avoids 0-batches
                        if len(self.batch_list) > 0:
                            break

                    bs = min(len(category_keys) - start, bs)
                    batch_sizes.append(bs)
                    start += bs
                    if start >= len(category_keys):
                        break

                if len(batch_sizes) == 0:
                    # Maybe we can't reach here
                    raise RuntimeError("0 batches")

                # If the last batch-size is smaller than minimum batch_size,
                # the samples are redistributed to the other mini-batches
                if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
                    for i in range(batch_sizes.pop(-1)):
                        batch_sizes[-(i % len(batch_sizes)) - 2] += 1

                if not self.drop_last:
                    # Bug check
                    assert sum(batch_sizes) == len(
                        category_keys
                    ), f"{sum(batch_sizes)} != {len(category_keys)}"

                # Set mini-batch
                cur_batch_list = []
                start = 0
                for bs in batch_sizes:
                    assert len(category_keys) >= start + bs, "Bug"
                    minibatch_keys = category_keys[start : start + bs]
                    start += bs
                    if sort_in_batch == "descending":
                        minibatch_keys.reverse()
                    elif sort_in_batch == "ascending":
                        # Key are already sorted in ascending
                        pass
                    else:
                        raise ValueError(
                            "sort_in_batch must be ascending or "
                            f"descending: {sort_in_batch}"
                        )
                    cur_batch_list.append(tuple(minibatch_keys))

                if sort_batch == "ascending":
                    pass
                elif sort_batch == "descending":
                    cur_batch_list.reverse()
                else:
                    raise ValueError(
                        f"sort_batch must be ascending or descending: {sort_batch}"
                    )
                self.batch_list.extend(cur_batch_list)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_files={self.shape_files}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch},  "
            f"batch_weights={' '.join(['%.3f'.format(x) for x in self.batch_weights])}, "
            f"categories={' '.join(self.categories)})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
