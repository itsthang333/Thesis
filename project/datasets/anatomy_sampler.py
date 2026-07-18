from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class AnatomyMatchedBatchSampler(Sampler[list[int]]):
    """Guarantee tumor/normal hard-negative pairs for all three regions."""

    def __init__(self, samples: Sequence[dict[str, object]], batch_size: int, seed: int = 42) -> None:
        if batch_size < 6:
            raise ValueError("AnatomyMatchedBatchSampler requires batch_size >= 6")
        self.samples = samples
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            region = int(sample["anatomy_region"])
            tumor_status = int(int(sample["tumor_type"]) != 0)
            self.groups[(region, tumor_status)].append(index)
        expected = {(region, status) for region in range(3) for status in range(2)}
        missing = sorted(expected - set(self.groups))
        if missing:
            raise ValueError(f"Missing anatomy/status cells required for matched sampling: {missing}")
        self.keys = sorted(expected)
        self.num_batches = math.ceil(len(samples) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        all_indices = list(range(len(self.samples)))
        for _ in range(self.num_batches):
            batch = [rng.choice(self.groups[key]) for key in self.keys]
            while len(batch) < self.batch_size:
                batch.append(rng.choice(all_indices))
            rng.shuffle(batch)
            yield batch
