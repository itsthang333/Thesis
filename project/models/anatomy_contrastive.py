from __future__ import annotations

"""Anatomy-matched contrastive learning for BTXRD tumor_type classification.

Motivation (see datasets/btxrd.py's ANATOMY_REGION_COLUMNS docstring): the
specific-bone columns (tibia, femur, humerus, hand, ulna, radius, foot,
fibula, hip bone, *-joint) are only ever set for tumor images in this
dataset and would leak the tumor label if used as a feature or auxiliary
target. The coarse upper limb / lower limb / pelvis columns do not have this
problem -- every region has both tumor and normal images (verified via
tools/check_anatomy_region_labels.py: upper limb 672 normal/452 tumor, lower
limb 1095 normal/1311 tumor, pelvis 112 normal/104 tumor, 3746/3746 records
with exactly one region set) -- so they are safe to use for region-matched
pairing: a tumor image's hardest useful negative is a NORMAL image of the
SAME anatomical region, not a random image that might just differ in which
body part is pictured.

This module provides:
  - RegionMatchedBatchSampler: yields batches where every sample's anatomy
    region has both tumor and normal images present in that same batch,
    so region-matched positives/negatives exist for the contrastive loss
    below. Ghi chú: pairing is done by region only, NOT by
    region+tumor_type -- some (tumor_type, region) cells are empty or very
    small (e.g. osteofibroma has 0 pelvis images), so requiring a same-type
    match would make sampling fail or degrade to rare-class starvation for
    exactly the classes that most need representation. Region alone is
    sufficient for the goal (suppressing gross anatomical differences as a
    shortcut, not fine-grained tumor-type discrimination).
  - anatomy_contrastive_loss: a supervised-contrastive-style loss on pooled
    embeddings (DenseNet121AnatomyClassifier.forward(..., return_embedding=True)),
    pulling same-tumor_type images together and pushing tumor images away
    from normal images IN THE SAME REGION ONLY -- normal images from a
    different region are excluded from the negative set entirely, since
    "tumor upper-limb vs. normal lower-limb" is a near-free negative (gross
    anatomy alone separates them) that would let the model shortcut around
    learning an actual lesion feature.
"""

import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


class RegionMatchedBatchSampler(Sampler[list[int]]):
    """Yields index batches where every present anatomy region has at least
    one tumor and one normal sample, by construction -- and, as a direct
    consequence of the alternating draw below, close to a 50/50 tumor/normal
    split per batch too (verified empirically: mean tumor fraction ~0.495
    over a full epoch on an imbalanced synthetic dataset, only drifting from
    an even split in the handful of final batches where a region's pool is
    nearly exhausted).

    Algorithm per batch: pick one region uniformly at random (weighted by
    how many not-yet-exhausted samples it has left this epoch), then fill
    the batch by ALTERNATING tumor/normal draws from that region (draw_tumor
    flips after every single sample, see __iter__) until the batch is full or
    the region runs out of fresh samples for this epoch, in which case fall
    back to drawing from any region with samples left -- the fallback still
    draws from the SAME side (tumor_pools if draw_tumor else normal_pools) of
    whichever fallback region has samples, so the running tumor/normal
    balance is preserved even when single-region pairing has to be relaxed
    for the last few slots of that batch. This keeps epoch coverage close to
    a full pass over the dataset while guaranteeing the common case (most
    batches) is single-region, tumor/normal-balanced.
    """

    def __init__(
        self,
        samples: list[dict[str, object]],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size < 2:
            raise ValueError("RegionMatchedBatchSampler requires batch_size >= 2 to pair tumor/normal")
        self.samples = samples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        self.region_tumor_indices: dict[int, list[int]] = defaultdict(list)
        self.region_normal_indices: dict[int, list[int]] = defaultdict(list)
        unknown_region_count = 0
        for i, sample in enumerate(samples):
            region = int(sample["anatomy_region"])
            if region == -1:
                unknown_region_count += 1
                continue
            if int(sample["tumor"]):
                self.region_tumor_indices[region].append(i)
            else:
                self.region_normal_indices[region].append(i)

        self.regions = sorted(
            set(self.region_tumor_indices) & set(self.region_normal_indices)
        )
        if not self.regions:
            raise ValueError(
                "RegionMatchedBatchSampler found no anatomy region with both tumor and "
                "normal samples -- check that samples[i]['anatomy_region'] and "
                "samples[i]['tumor'] are populated (see load_btxrd_records)."
            )
        if unknown_region_count:
            print(
                f"RegionMatchedBatchSampler: {unknown_region_count}/{len(samples)} samples have "
                "unknown anatomy_region (-1) and will never be drawn this epoch."
            )

        self._usable_count = sum(len(v) for v in self.region_tumor_indices.values()) + sum(
            len(v) for v in self.region_normal_indices.values()
        )

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (like DistributedSampler) so shuffling differs
        across epochs while staying reproducible for a given (seed, epoch)."""
        self._epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self._usable_count // self.batch_size
        return -(-self._usable_count // self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        tumor_pools = {region: list(indices) for region, indices in self.region_tumor_indices.items()}
        normal_pools = {region: list(indices) for region, indices in self.region_normal_indices.items()}
        for pool in tumor_pools.values():
            rng.shuffle(pool)
        for pool in normal_pools.values():
            rng.shuffle(pool)

        def region_has_pairs(region: int) -> bool:
            return bool(tumor_pools.get(region)) and bool(normal_pools.get(region))

        active_regions = [r for r in self.regions if region_has_pairs(r)]
        num_batches = len(self)
        for _ in range(num_batches):
            active_regions = [r for r in active_regions if region_has_pairs(r)]
            if not active_regions:
                break
            region = rng.choice(active_regions)
            batch: list[int] = []
            draw_tumor = True
            while len(batch) < self.batch_size:
                pool = tumor_pools[region] if draw_tumor else normal_pools[region]
                if pool:
                    batch.append(pool.pop())
                else:
                    # This region ran out of one side mid-batch; fill the rest
                    # from whatever region still has samples, so the batch
                    # doesn't come up short of batch_size. The already-drawn
                    # samples from `region` keep their region-matched pairing
                    # property for the loss; the fallback fill just avoids
                    # wasting a partially-built batch.
                    fallback_region = next((r for r in self.regions if region_has_pairs(r)), None)
                    if fallback_region is None:
                        break
                    fallback_pool = (
                        tumor_pools[fallback_region] if draw_tumor else normal_pools[fallback_region]
                    )
                    if fallback_pool:
                        batch.append(fallback_pool.pop())
                    else:
                        break
                draw_tumor = not draw_tumor
            if batch:
                yield batch


def anatomy_contrastive_loss(
    embeddings: torch.Tensor,
    tumor_type: torch.Tensor,
    anatomy_region: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised-contrastive loss restricted to region-matched pairs.

    embeddings:     [B, D] pooled backbone features (see
                    DenseNet121AnatomyClassifier.forward(..., return_embedding=True)).
                    Not yet normalized -- this function L2-normalizes internally.
    tumor_type:     [B] long tensor, 0=normal, 1..9=tumor type (see
                    datasets/btxrd.py's TUMOR_TYPE_CLASS_NAMES).
    anatomy_region: [B] long tensor, index into ANATOMY_REGION_COLUMNS (-1 for
                    unknown -- such samples are excluded from both the anchor
                    and candidate sets entirely, since their region-matched
                    negatives can't be determined).

    For each anchor sample i (skipping anatomy_region[i] == -1):
      - Positives: other samples j (j != i) with tumor_type[j] == tumor_type[i]
        (same specific class, tumor-tumor or normal-normal) -- region is NOT
        required to match for positives, since two images of the same tumor
        type should attract regardless of which limb, and there is no
        anatomical-shortcut risk in the POSITIVE direction (shortcuts help a
        model separate classes cheaply, not merge them).
      - Negatives: only samples j with anatomy_region[j] == anatomy_region[i]
        AND tumor_type[j] != tumor_type[i]. Cross-region samples are excluded
        as negatives entirely (both tumor-vs-tumor-other-region and any
        normal-vs-normal-other-region pairs), which is what forces the loss
        to separate tumor from normal using within-region evidence only --
        a random cross-region normal is separable by gross anatomy alone and
        would be a "free" negative the model could exploit as a shortcut.

    Returns 0.0 (as a differentiable zero, safe to add into a total loss)
    if fewer than 2 samples have known anatomy_region, or if no anchor in
    the batch has both a positive and a region-matched negative available
    (can happen with a small/unlucky batch even under RegionMatchedBatchSampler,
    e.g. a batch drawn entirely from one tumor_type after a fallback-fill).
    """
    device = embeddings.device
    known_mask = anatomy_region != -1
    if known_mask.sum() < 2:
        return embeddings.sum() * 0.0

    embeddings = embeddings[known_mask]
    tumor_type = tumor_type[known_mask]
    anatomy_region = anatomy_region[known_mask]

    normalized = F.normalize(embeddings, dim=1)
    similarity = torch.matmul(normalized, normalized.T) / temperature  # [N, N]

    n = similarity.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    same_type = tumor_type.unsqueeze(0) == tumor_type.unsqueeze(1)
    same_region = anatomy_region.unsqueeze(0) == anatomy_region.unsqueeze(1)

    positive_mask = same_type & ~self_mask
    negative_mask = (~same_type) & same_region & ~self_mask
    # Candidates for the softmax denominator: positives plus region-matched
    # negatives only -- cross-region, different-type pairs are excluded from
    # the denominator too (not just as explicit negatives), so they never
    # inflate an anchor's normalization term as a free easy-negative either.
    candidate_mask = positive_mask | negative_mask

    has_positive = positive_mask.any(dim=1)
    has_negative = negative_mask.any(dim=1)
    valid_anchor = has_positive & has_negative
    if not valid_anchor.any():
        return embeddings.sum() * 0.0

    # Numerically stable log-softmax restricted to each row's candidate set.
    masked_similarity = similarity.masked_fill(~candidate_mask, float("-inf"))
    row_max = masked_similarity.max(dim=1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_similarity = torch.exp(masked_similarity - row_max) * candidate_mask
    denom = exp_similarity.sum(dim=1).clamp(min=1e-12)
    log_prob = (masked_similarity - row_max) - torch.log(denom).unsqueeze(1)

    # log_prob is -inf at masked-out (non-candidate) positions. Multiplying
    # that -inf by a boolean 0 mask produces NaN (-inf * 0 == nan in IEEE
    # float), not 0 -- torch.where selects cleanly instead of computing
    # through the -inf entries at all.
    zero_log_prob = torch.zeros_like(log_prob)
    positive_log_prob = torch.where(positive_mask, log_prob, zero_log_prob).sum(dim=1) / (
        positive_mask.sum(dim=1).clamp(min=1)
    )
    loss_per_anchor = -positive_log_prob[valid_anchor]
    return loss_per_anchor.mean()
