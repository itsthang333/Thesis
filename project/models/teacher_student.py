from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from models.layercam import LayerCAM
from models.torch_morphology import cam_to_soft_attention_target


class EMATeacher:
    """A frozen copy of a student model, updated via exponential moving
    average of the student's weights (not gradient descent) after an initial
    warmup period during which the teacher stays exactly at its starting
    checkpoint.

    Rationale for the warmup (per this project's design discussion): at
    epoch 0, if teacher == student (both freshly loaded from the same
    checkpoint), the attention-consistency signal is ~0 and the teacher has
    nothing meaningful to teach yet. Freezing the teacher for a few epochs
    while the student trains on CE alone lets the student's CAM start to
    diverge from the teacher's before EMA updates begin -- otherwise EMA
    would just be averaging two copies of the same (diffuse) starting point.
    """

    def __init__(self, student: nn.Module, decay: float = 0.999) -> None:
        self.model = copy.deepcopy(student)
        self.model.eval()
        # NOTE: params intentionally keep requires_grad=True here, even
        # though the teacher is never trained by an optimizer (EMA.update()
        # is the only thing that ever changes these weights). LayerCAM's
        # _compute_cam() needs score.backward() to produce a real gradient
        # w.r.t. the hooked activations (A * relu(G) in its CAM formula) --
        # with requires_grad=False, backward() has nothing to differentiate
        # through and raises. This is safe: no optimizer ever holds a
        # reference to teacher.model.parameters(), so nothing outside
        # EMATeacher.update() can ever modify these weights via gradient
        # descent, regardless of requires_grad being True.
        self.decay = decay
        self.layercam = LayerCAM(self.model, device=next(self.model.parameters()).device)

    def update(self, student: nn.Module) -> None:
        """EMA-update the teacher's weights from the student's current weights."""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.model.parameters(), student.parameters()):
                teacher_param.mul_(self.decay).add_(student_param.detach(), alpha=1 - self.decay)
            for teacher_buffer, student_buffer in zip(self.model.buffers(), student.buffers()):
                teacher_buffer.copy_(student_buffer)

    def compute_soft_attention_target(
        self,
        images: torch.Tensor,
        target_class: torch.Tensor,
        percentile: float = 96.0,
        output_size: tuple[int, int] | None = None,
        blur_kernel_size: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher CAM (LayerCAM, ground-truth-class-conditioned) -> soft
        percentile-centered gate -> Gaussian blur -> soft target in [0, 1].
        No gradient ever flows through this (teacher is frozen, LayerCAM's
        own backward() call only touches the teacher's own graph, and the
        gating/blur step is wrapped in torch.no_grad() internally). See
        cam_to_soft_attention_target's docstring for why this uses a soft
        gate instead of a hard threshold + morphological opening/closing:
        at DenseNet121's native 12x12 feature-map resolution, a hard
        percentile-96 threshold keeps only ~6 of 144 pixels (almost never
        forming a solid neighborhood), and morphological erosion then wipes
        out every single one -- confirmed empirically to degenerate the
        target to all-zero on every tested batch, including real DenseNet121.

        Runs ONE batched forward+backward pass through the teacher (using
        LayerCAM's per-sample class_index support -- each sample gets its
        own ground-truth class via one indexed sum + one backward(), not one
        backward() per sample). An earlier version of this method looped
        per-sample (batch_size separate full DenseNet121 forward+backward
        passes every training step), which would have made training 5-10x
        slower once the teacher activates -- caught in review before ever
        being run on real data/GPU.

        Forced to run outside any fp16 autocast: LayerCAM's score.backward()
        has no GradScaler involved (unlike the student's own training step),
        so under an enclosing autocast(enabled=True) context, small
        gradients here can silently underflow to zero in fp16, producing a
        degenerate all-zero CAM -- a different failure mode than the
        earlier-fixed activation-overflow bug (this is gradient underflow in
        the backward pass, not activation overflow in the forward pass).
        """
        with torch.cuda.amp.autocast(enabled=False):
            out = self.layercam.cam_for_class(images.float(), class_index=target_class)
            teacher_cam = out.cam  # [B, H, W], upsampled to input image size by LayerCAM

            # Teacher's own confidence in the ground-truth class, from the
            # SAME forward pass that produced the CAM (no extra compute).
            # Used by the caller to weight the attention loss per-sample:
            # a teacher that's unsure about the class it's localizing for
            # is also unlikely to have localized it correctly (classic
            # confidence-weighting pattern from FixMatch/Mean Teacher/
            # SoftTeacher/Unbiased Teacher/DenseTeacher-style pseudo-label
            # methods, guarding against confirmation bias -- an EMA teacher
            # naively teaching a diffuse CAM as if it were ground truth).
            batch_indices = torch.arange(out.logits.shape[0], device=out.logits.device)
            teacher_conf = F.softmax(out.logits, dim=1)[batch_indices, target_class]

            if output_size is not None:
                # Downsample the CAM to the comparison resolution BEFORE
                # morphology, not after -- morphology's kernels are sized in
                # pixels (e.g. 5px), and applying them at input resolution
                # (e.g. 384x384) then downsampling ~32x to the student's
                # native feature-map resolution (e.g. 12x12) would wash the
                # kernel's effect out to near-nothing (a 5px kernel at 384
                # corresponds to well under 1 pixel at 12x12). Running
                # morphology at the actual comparison resolution instead
                # makes the kernel size meaningful relative to that grid.
                teacher_cam = F.interpolate(
                    teacher_cam.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
                ).squeeze(1)

            soft_target, valid_mask = cam_to_soft_attention_target(
                teacher_cam, percentile=percentile, blur_kernel_size=blur_kernel_size,
            )
            return soft_target, valid_mask, teacher_conf.detach()


def student_cam_for_attention_loss(
    model: nn.Module, feature_map: torch.Tensor, target_class: torch.Tensor
) -> torch.Tensor:
    """The student's own CAM for the attention-consistency loss, using the
    same classic-CAM formula as models/puzzle_cam.py (GAP-weight einsum) --
    NOT LayerCAM, since LayerCAM's backward()-per-sample loop isn't suitable
    inside the student's own training graph (it would call model.zero_grad()
    and .backward() on the very model whose gradients the optimizer needs
    intact for the classification loss). This differs from the teacher's
    target generation (LayerCAM) by construction, but that's fine: the
    student only needs a differentiable proxy for its own attention to match
    the teacher's already-computed (frozen, LayerCAM-based) target -- the
    student doesn't need to replicate LayerCAM's exact formula, only to learn
    to produce a similarly-shaped, similarly-localized CAM.

    Returns: [B, H, W], min-max normalized per-sample to [0, 1].
    """
    from models.puzzle_cam import classic_cam

    cam = classic_cam(feature_map, model.classifier, target_class)
    flat = cam.view(cam.shape[0], -1)
    mn = flat.min(dim=1).values.view(-1, 1, 1)
    mx = flat.max(dim=1).values.view(-1, 1, 1)
    cam_range = mx - mn
    normalized = (cam - mn) / (cam_range + 1e-8)
    return torch.where(cam_range > 1e-4, normalized, torch.zeros_like(normalized))


def attention_distillation_loss(
    teacher: EMATeacher,
    student_model: nn.Module,
    student_features: torch.Tensor,
    images: torch.Tensor,
    target_class: torch.Tensor,
    percentile: float = 96.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_attention: BCE between the student's own CAM and the teacher's
    refined soft attention target (Teacher LayerCAM -> percentile ->
    morphology -> blur, see EMATeacher.compute_soft_attention_target),
    weighted by the teacher's own confidence in the ground-truth class.

    BCE (not L1/MSE) is used since the teacher's soft target, after the
    percentile+morphology+blur pipeline, is a smoothly-varying [0,1] map
    that behaves like a soft probability of "this pixel is part of the
    lesion" -- BCE is the natural loss for matching a probability-like
    target, and its steeper gradient near 0/1 helps pull the student's CAM
    toward the teacher's sharp, denoised region rather than just averaging
    toward it (as L1 would).

    Returns: (weighted_loss, mean_teacher_conf, valid_fraction) -- the latter
    two values are for logging only (train_classifier.py tracks them to check
    whether teacher confidence correlates with actual CAM quality, per this
    project's own methodology discussion: a teacher can be confident about
    *classification* while still localizing the lesion incorrectly, so this
    correlation needs to be checked empirically, not assumed).
    """
    # NOT wrapped in torch.no_grad(): LayerCAM's _compute_cam() needs an
    # active graph to run its own internal score.backward() against the
    # teacher's hooked activations (see EMATeacher's requires_grad note).
    # The result is .detach()'d below before being used as a target, so no
    # gradient from this computation ever reaches the student's optimizer --
    # this only prevents LayerCAM's own internal backward() from crashing.
    #
    # output_size is passed through so the teacher downsamples its CAM to
    # the student's native feature-map resolution (e.g. 12x12 for a 384x384
    # input under stride 32) BEFORE running percentile+morphology+blur, not
    # after -- morphology kernels are sized in pixels, and running them at
    # full input resolution then downsampling ~32x would wash the kernel's
    # effect out to near-nothing at the comparison resolution (a 5px kernel
    # at 384 is well under 1px at 12x12). morph/blur kernel_size=3 (default)
    # is a meaningful fraction of a 12x12 grid, unlike the 384-resolution
    # default of 5.
    # Forced outside fp16 autocast entirely: on real GPU training (unlike
    # the CPU-only synthetic tests this was first verified with, where
    # GradScaler(enabled=False) never actually exercises autocast's runtime
    # checks), PyTorch hard-blocks F.binary_cross_entropy inside an
    # autocast(enabled=True) region -- it raises
    # "binary_cross_entropy and BCELoss are unsafe to autocast" rather than
    # silently miscomputing, because BCE's log() near 0/1 can produce -inf
    # in fp16. student_features (computed by the caller under the outer
    # autocast) is cast back to fp32 here for the same reason classic_cam's
    # einsum needed it in puzzle_cam.py.
    with torch.cuda.amp.autocast(enabled=False):
        student_spatial_size = student_features.shape[-2:]
        soft_target, valid_mask, teacher_conf = teacher.compute_soft_attention_target(
            images, target_class, percentile=percentile, output_size=student_spatial_size,
        )
        soft_target = soft_target.detach()
        soft_target = soft_target.squeeze(1)  # [B, H, W]

        student_cam = student_cam_for_attention_loss(
            student_model, student_features.float(), target_class
        )

        # Degenerate samples (teacher's CAM too flat to threshold
        # meaningfully -- confirmed empirically on a random-init classifier:
        # this can be the WHOLE batch right after warmup, before the teacher
        # has learned any real localization signal) must be excluded from
        # the loss entirely, not trained against an all-zero "no lesion
        # anywhere" target -- that target is wrong (not "no signal"), and
        # BCE against it was found to produce ~200x the gradient magnitude
        # of a normal CrossEntropy step, since it penalizes every positive
        # activation in the student's CAM as if it were a false positive
        # across the whole spatial grid.
        if not valid_mask.any():
            # Must stay part of student_cam's graph (not a bare constant) --
            # train_classifier.py does loss.backward() on the combined total
            # loss unconditionally, so this needs a valid (if zero-valued)
            # grad_fn rather than crashing with "does not require grad".
            return student_cam.sum() * 0.0, teacher_conf.mean(), valid_mask.float().mean()

        per_sample_bce = F.binary_cross_entropy(
            student_cam.clamp(1e-6, 1 - 1e-6), soft_target, reduction="none"
        ).mean(dim=(1, 2))

        # Confidence-weighted (squared) attention loss: guards against
        # confirmation bias, where an EMA teacher that isn't actually sure
        # about the ground-truth class teaches the student a wrong/diffuse
        # CAM as if it were reliable ground truth. Squaring (rather than a
        # hard confidence threshold, which would discard most of a small
        # dataset like BTXRD early in training when the teacher is still
        # weak) steeply discounts low-confidence samples (0.5 -> 0.25,
        # 0.3 -> 0.09) while barely touching high-confidence ones
        # (0.99 -> 0.98) -- same pattern used by FixMatch/SoftTeacher/
        # Unbiased Teacher/DenseTeacher-style pseudo-label weighting.
        #
        # IMPORTANT: averaging by weight.sum() (a weighted MEAN) would
        # silently defeat this -- found in review before ever training on
        # it: if every sample in a batch shares roughly the same low
        # confidence (e.g. all 0.1, weight~0.01), the weight cancels between
        # numerator and denominator and the result equals the UNWEIGHTED
        # mean, exactly the outcome this weighting is meant to prevent
        # (teaching from an entirely-unsure-teacher batch as if it were
        # normal). Averaging by the fixed number of valid samples instead
        # (a weighted SUM over a constant denominator) means a
        # uniformly-low-confidence batch genuinely produces a small loss.
        weight = teacher_conf.pow(2)
        weighted_bce = per_sample_bce * weight

    num_valid = valid_mask.sum().clamp(min=1)
    loss = weighted_bce[valid_mask].sum() / num_valid
    return loss, teacher_conf[valid_mask].mean(), valid_mask.float().mean()
