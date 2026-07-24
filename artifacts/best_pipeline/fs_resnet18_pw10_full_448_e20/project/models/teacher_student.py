from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from models.layercam import LayerCAM
from models.torch_morphology import cam_to_soft_attention_target


class EMATeacher:
    def __init__(self, student: nn.Module, decay: float = 0.999) -> None:
        self.model = copy.deepcopy(student)
        self.model.eval()
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
        with torch.cuda.amp.autocast(enabled=False):
            out = self.layercam.cam_for_class(images.float(), class_index=target_class)
            teacher_cam = out.cam  # [B, H, W], upsampled to input image size by LayerCAM

            batch_indices = torch.arange(out.logits.shape[0], device=out.logits.device)
            teacher_conf = F.softmax(out.logits, dim=1)[batch_indices, target_class]

            if output_size is not None:
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

        if not valid_mask.any():
            return student_cam.sum() * 0.0, teacher_conf.mean(), valid_mask.float().mean()

        per_sample_bce = F.binary_cross_entropy(
            student_cam.clamp(1e-6, 1 - 1e-6), soft_target, reduction="none"
        ).mean(dim=(1, 2))

        weight = teacher_conf.pow(2)
        weighted_bce = per_sample_bce * weight

    num_valid = valid_mask.sum().clamp(min=1)
    loss = weighted_bce[valid_mask].sum() / num_valid
    return loss, teacher_conf[valid_mask].mean(), valid_mask.float().mean()
