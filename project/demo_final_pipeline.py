from __future__ import annotations

"""One-image, annotation-free demonstration of the finalized Rich Gallery G1 pipeline.

This module is intentionally an inference orchestrator. It reuses the production
candidate generator and the frozen G1/RAD-DINO primitives, but limits the cohort
to one manifest-selected image so that every stage can be shown interactively.
It never opens BTXRD polygon annotations and it does not compute Dice or IoU.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import argparse

import numpy as np
import torch
from PIL import Image

from .final_selector import select_candidate
from .evaluation.frozen_test_guard import verify_frozen_test_config
from .frozen_io import sha256_file, verify_model_snapshot
from .generate_biomedclip_saliency import load_biomedclip
from .merge_frozen_candidate_galleries import merge_payloads
from .models.biomedclip_saliency import FrozenBiomedClipSaliency, resize_map
from .models.nominal_patch_memory import make_seeded_random_projection
from .models.rad_dino_mask_bag_mil import (
    MaskBagMILConfig,
    RadDinoMaskBagMIL,
    smooth_mil_pool,
)
from .pseudo.candidate_diagnostics import (
    save_candidate_diagnostics,
    validate_candidate_diagnostics_manifest,
    write_candidate_diagnostics_manifest,
)
from .run_rad_dino_mask_bag_mil_probe import (
    SELECTED_HIDDEN_LAYERS,
    ProjectedMultiLayerEncoder,
    build_descriptor_cache,
    seed_everything,
)
from .run_rich_gallery_candidate_supply import build_generation_command


CHECKPOINT_HASHES = {
    "classifier_320": "ca630ddf816c1b6a55fab9b99fe824877bba9a83905ce71fd20cf9c2b1640621",
    "classifier_448": "b40dc5ec0f601ea7392fd0e8ed0be5f1e7cd66ad07d654392db516a0766d451e",
    "sam_vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "biomedclip": "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be",
    "rad_dino_config": "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede",
    "rad_dino_preprocessor": "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56",
    "rad_dino_weight": "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae",
    "g1": "634e1200330e87692fab4a2e35ba70806790937d7b19ed8b0a3c4968471bfe8c",
}


@dataclass(frozen=True)
class DemoConfig:
    repository_root: Path
    checkpoint_root: Path
    dataset_root: Path
    split_manifest: Path
    classifier_320_split_manifest: Path
    split: str
    image_id: str
    work_dir: Path
    frozen_config: Path | None = None
    device: str = "cuda"

    @property
    def classifier_320(self) -> Path:
        return self.checkpoint_root / "classifiers" / "classifier_320_binary.pt"

    @property
    def classifier_448(self) -> Path:
        return self.checkpoint_root / "classifiers" / "classifier_448_binary.pt"

    @property
    def sam_checkpoint(self) -> Path:
        return self.checkpoint_root / "sam_vit_b" / "sam_vit_b_01ec64.pth"

    @property
    def biomedclip_dir(self) -> Path:
        return self.checkpoint_root / "biomedclip"

    @property
    def rad_dino_dir(self) -> Path:
        return self.checkpoint_root / "rad_dino"

    @property
    def g1_checkpoint(self) -> Path:
        return self.checkpoint_root / "g1_selector" / "rad_dino_mask_bag_mil.pt"

    @classmethod
    def from_json(cls, path: Path) -> "DemoConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        for name in (
            "repository_root", "checkpoint_root", "dataset_root",
            "split_manifest", "classifier_320_split_manifest", "work_dir",
        ):
            values[name] = Path(values[name])
        if values.get("frozen_config"):
            values["frozen_config"] = Path(values["frozen_config"])
        else:
            values["frozen_config"] = None
        return cls(**values)

    def write_json(self, path: Path) -> Path:
        payload = asdict(self)
        for name, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[name] = str(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


def _canonical_row(config: DemoConfig) -> dict[str, str]:
    with config.split_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        matches = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("image_id") == config.image_id
            and row.get("split") == config.split
            and row.get("eligible") == "1"
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one eligible {config.split} row for {config.image_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_demo(config: DemoConfig) -> dict[str, object]:
    if config.split not in {"val", "test"}:
        raise ValueError("The demo split must be val or test")
    if config.split == "test" and config.frozen_config is None:
        raise ValueError("A locked final-test protocol is required for a test demo")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("The demo device must be 'cpu' or 'cuda'")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not config.classifier_320_split_manifest.is_file():
        raise FileNotFoundError(
            "Missing split manifest bound to the 320 px classifier: "
            f"{config.classifier_320_split_manifest}"
        )
    files = {
        "classifier_320": config.classifier_320,
        "classifier_448": config.classifier_448,
        "sam_vit_b": config.sam_checkpoint,
        "biomedclip": config.biomedclip_dir / "open_clip_pytorch_model.bin",
        "rad_dino_config": config.rad_dino_dir / "config.json",
        "rad_dino_preprocessor": config.rad_dino_dir / "preprocessor_config.json",
        "rad_dino_weight": config.rad_dino_dir / "model.safetensors",
        "g1": config.g1_checkpoint,
    }
    verified: dict[str, object] = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing demo asset: {path}")
        actual = sha256_file(path)
        if actual != CHECKPOINT_HASHES[name]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {name}")
        verified[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    row = _canonical_row(config)
    if config.split == "test":
        verify_frozen_test_config(
            config.frozen_config,
            split="test",
            split_manifest=config.split_manifest,
            requested_artifacts={
                "classifier_checkpoint": config.classifier_320,
                "sam_checkpoint": config.sam_checkpoint,
                "g1_checkpoint": config.g1_checkpoint,
                "biomedclip_weight": config.biomedclip_dir / "open_clip_pytorch_model.bin",
                "rad_dino_weight": config.rad_dino_dir / "model.safetensors",
            },
        )
    image_path = config.dataset_root / "images" / config.image_id
    if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
        raise ValueError("Demo image is missing or differs from the canonical manifest")
    return {
        "image_id": config.image_id,
        "split": config.split,
        "tumor_image_label": int(row["tumor"]),
        "image_path": str(image_path),
        "checkpoints": verified,
        "spatial_ground_truth_read": False,
    }


def _fresh(path: Path, *, work_root: Path) -> None:
    root = work_root.expanduser().resolve()
    target = path.expanduser().resolve()
    if target == root or root not in target.parents:
        raise ValueError("A demo stage may only replace a child of work_dir")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)


def generate_biomedclip_demo(config: DemoConfig) -> dict[str, object]:
    """Generate the frozen full-view plus Top-3 tiled BiomedCLIP saliency map."""

    validate_demo(config)
    row = _canonical_row(config)
    output = config.work_dir / "01_biomedclip"
    _fresh(output, work_root=config.work_dir)
    import open_clip

    model, preprocess, tokenizer, weight_path = load_biomedclip(
        open_clip,
        model_dir=config.biomedclip_dir,
    )
    saliency_model = FrozenBiomedClipSaliency(
        model,
        preprocess,
        tokenizer,
        device=torch.device(config.device),
        crop_fraction=0.5,
        positions_per_axis=3,
        top_k_tiles=3,
    )
    image_path = config.dataset_root / "images" / config.image_id
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if int(row["tumor"]) == 0:
        saliency = np.zeros((320, 320), dtype=np.float32)
    else:
        result = saliency_model(image)
        saliency = np.clip(resize_map(result.saliency, 320, 320), 0.0, 1.0)
    map_path = output / "maps" / f"{Path(config.image_id).stem}.npy"
    map_path.parent.mkdir(parents=True)
    np.save(map_path, saliency.astype(np.float16), allow_pickle=False)
    manifest_path = output / "saliency_manifest.csv"
    fields = (
        "image_id", "tumor_image_label", "source_image_sha256", "map_path",
        "map_sha256", "map_height", "map_width", "map_min", "map_max",
        "map_mean", "map_dynamic_range",
    )
    manifest_row = {
        "image_id": config.image_id,
        "tumor_image_label": int(row["tumor"]),
        "source_image_sha256": row["image_sha256"],
        "map_path": f"maps/{map_path.name}",
        "map_sha256": sha256_file(map_path),
        "map_height": 320,
        "map_width": 320,
        "map_min": float(saliency.min()),
        "map_max": float(saliency.max()),
        "map_mean": float(saliency.mean()),
        "map_dynamic_range": float(saliency.max() - saliency.min()),
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(manifest_row)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=config.repository_root, text=True
    ).strip()
    metadata = {
        "stage": "prediction-first BiomedCLIP saliency generation",
        "supervision": "images and binary image-level labels only",
        "source_commit": source_commit,
        "source_files": {"demo_final_pipeline.py": sha256_file(Path(__file__))},
        "split": config.split,
        "split_manifest_sha256": sha256_file(config.split_manifest),
        "manifest_sha256": sha256_file(manifest_path),
        "model": {"weight_sha256": sha256_file(weight_path)},
        "prompts": {"mode": "frozen_binary_contrast"},
        "view_contract": {
            "full_view": "black pad to square",
            "crop_fraction_of_short_side": 0.5,
            "positions_per_axis": 3,
            "top_k_tiles_by_contrast_score": 3,
            "fusion": "pixelwise maximum of full view and selected tiles",
            "output_size": 320,
        },
        "validation_gt_read": False,
        "test_images_read": 1 if config.split == "test" else 0,
        "test_evaluated": False,
    }
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output),
        "map_path": str(map_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_sha256": sha256_file(metadata_path),
        "source_commit": source_commit,
        "model_weight_sha256": CHECKPOINT_HASHES["biomedclip"],
    }


def generate_candidate_demo(config: DemoConfig, mode: str) -> dict[str, object]:
    if mode not in {"anchor", "addition"}:
        raise ValueError("mode must be anchor or addition")
    validate_demo(config)
    output = config.work_dir / ("02_anchor" if mode == "anchor" else "03_addition")
    _fresh(output, work_root=config.work_dir)
    image_list = config.work_dir / "demo_image.txt"
    image_list.parent.mkdir(parents=True, exist_ok=True)
    image_list.write_text(config.image_id + "\n", encoding="utf-8")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=config.repository_root, text=True
    ).strip()
    saliency_root = config.work_dir / "01_biomedclip"
    metadata = json.loads((saliency_root / "run_metadata.json").read_text(encoding="utf-8"))
    command = build_generation_command(
        mode=mode,
        source_root=config.repository_root,
        data_root=config.dataset_root,
        split_manifest=config.split_manifest,
        classifier_split_manifest=(
            config.classifier_320_split_manifest if mode == "anchor" else config.split_manifest
        ),
        split=config.split,
        classifier=(config.classifier_320 if mode == "anchor" else config.classifier_448),
        sam=config.sam_checkpoint,
        output_dir=output,
        external_root=(saliency_root if mode == "anchor" else None),
        external_manifest_sha256=(metadata["manifest_sha256"] if mode == "anchor" else None),
        external_metadata_sha256=(sha256_file(saliency_root / "run_metadata.json") if mode == "anchor" else None),
        external_source_commit=(source_commit if mode == "anchor" else None),
        external_weight_sha256=(CHECKPOINT_HASHES["biomedclip"] if mode == "anchor" else None),
        frozen_config=config.frozen_config,
    )
    # The production wrapper defaults both proposal models to CUDA.  A local
    # thesis demo must also be able to run deterministically on a CPU-only
    # workstation (or on a GPU whose VRAM is too small for SAM ViT-B).
    for option in ("--classifier-device", "--sam-device"):
        option_index = command.index(option)
        command[option_index + 1] = config.device
    command.extend(["--image-list", str(image_list), "--num-workers", "0"])
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    subprocess.run(command, cwd=config.repository_root, env=env, check=True)
    summary = json.loads((output / "candidate_diagnostics_summary.json").read_text(encoding="utf-8"))
    return {"output_dir": str(output), **summary}


def merge_candidate_demo(config: DemoConfig) -> dict[str, object]:
    output = config.work_dir / "04_merged_gallery"
    _fresh(output, work_root=config.work_dir)
    anchor_root = config.work_dir / "02_anchor"
    addition_root = config.work_dir / "03_addition"
    anchor_rows, anchor_summary = validate_candidate_diagnostics_manifest(
        anchor_root, expected_image_names=[config.image_id], split=config.split
    )
    addition_rows, _ = validate_candidate_diagnostics_manifest(
        addition_root, expected_image_names=[config.image_id], split=config.split
    )
    stem = Path(config.image_id).stem
    with np.load(anchor_root / anchor_rows[stem]["diagnostic_path"], allow_pickle=False) as source:
        anchor = {name: source[name].copy() for name in source.files}
    with np.load(addition_root / addition_rows[stem]["diagnostic_path"], allow_pickle=False) as source:
        addition = {name: source[name].copy() for name in source.files}
    merged, stats = merge_payloads(anchor, addition, addition_namespace="classifier448")
    payload_path = output / "candidate_diagnostics" / f"{stem}.npz"
    result = save_candidate_diagnostics(
        payload_path,
        sam_masks=merged["sam_masks"],
        refined_mask=anchor["refined_mask"],
        final_mask=anchor["final_mask"],
        bone_support=(anchor["bone_support"] if int(anchor["bone_support_present"][0]) else None),
        prompt_map=merged["prompt_map"],
        positive_points=anchor["positive_points"],
        negative_points=anchor["negative_points"],
        boxes=anchor["boxes"],
        sam_scores=merged["sam_scores"],
        selection_scores=merged["selection_scores"],
        classifier_causal_scores=merged["classifier_causal_scores"],
        component_ids=merged["component_ids"],
        prompt_modes=merged["prompt_modes"],
        proposal_source_ids=merged["proposal_source_ids"],
    )
    summary = write_candidate_diagnostics_manifest(
        output,
        [{"image_name": config.image_id, **result, **stats}],
        expected_image_names=[config.image_id],
        split=config.split,
        image_size=int(anchor_summary["image_size"]),
        pseudo_manifest_sha256=str(anchor_summary["pseudo_manifest_sha256"]),
        selection_method="geometry_v3_unconditional_gallery_union",
        support_clip_kernel=int(anchor_summary["support_clip_kernel"]),
        cam_percentile=float(anchor_summary["cam_percentile"]),
        cohort="all",
    )
    return {"output_dir": str(output), **summary, **stats}


def score_and_select_demo(config: DemoConfig) -> dict[str, object]:
    """Run RAD-DINO, G1, and the frozen equal percentile-rank selector."""

    validate_demo(config)
    output = config.work_dir / "05_final"
    _fresh(output, work_root=config.work_dir)
    row = _canonical_row(config)
    candidate_root = config.work_dir / "04_merged_gallery"
    summary = json.loads((candidate_root / "candidate_diagnostics_summary.json").read_text(encoding="utf-8"))
    candidate_rows, _ = validate_candidate_diagnostics_manifest(
        candidate_root,
        expected_image_names=[config.image_id],
        split=config.split,
        expected_manifest_sha256=summary["manifest_sha256"],
        expected_pseudo_manifest_sha256=summary["pseudo_manifest_sha256"],
    )
    verify_model_snapshot(
        config.rad_dino_dir,
        expected_config_sha256=CHECKPOINT_HASHES["rad_dino_config"],
        expected_preprocessor_sha256=CHECKPOINT_HASHES["rad_dino_preprocessor"],
        expected_weight_sha256=CHECKPOINT_HASHES["rad_dino_weight"],
    )
    seed_everything(42)
    from transformers import AutoModel

    device = torch.device(config.device)
    projection = make_seeded_random_projection(input_dim=768, output_dim=128, seed=42)
    backbone = AutoModel.from_pretrained(config.rad_dino_dir, local_files_only=True)
    backbone.requires_grad_(False).eval()
    encoder = ProjectedMultiLayerEncoder(backbone, torch.from_numpy(projection)).to(device).eval()
    mil_config = MaskBagMILConfig(token_dim=128, token_layers=len(SELECTED_HIDDEN_LAYERS))
    args = SimpleNamespace(
        dataset_root=config.dataset_root,
        input_size=448,
        maximum_candidates=243,
        encoder_batch_size=1,
    )
    cache = build_descriptor_cache(
        [row], candidate_rows, candidate_root, encoder, mil_config, args, device, split=config.split
    )
    checkpoint = torch.load(config.g1_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = MaskBagMILConfig(**checkpoint["config"])
    if asdict(checkpoint_config) != asdict(mil_config):
        raise ValueError("G1 descriptor configuration mismatch")
    model = RadDinoMaskBagMIL(checkpoint_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).to(device).eval()
    record = cache[0]
    kept = np.asarray(record["kept_indices"], dtype=np.int32)
    valid = torch.ones((1, len(kept)), dtype=torch.bool, device=device)
    with torch.inference_mode():
        original, _ = model.score_descriptors(
            torch.from_numpy(np.asarray(record["descriptors"], dtype=np.float32))[None].to(device), valid
        )
        mirrored, _ = model.score_descriptors(
            torch.from_numpy(np.asarray(record["flipped_descriptors"], dtype=np.float32))[None].to(device), valid
        )
        logits = 0.5 * (original + mirrored)
        bag_logit = smooth_mil_pool(logits, valid, temperature=model.config.bag_temperature)[0]
    payload_path = candidate_root / candidate_rows[Path(config.image_id).stem]["diagnostic_path"]
    with np.load(payload_path, allow_pickle=False) as payload:
        all_masks = payload["sam_masks"].astype(np.uint8)
        upstream = payload["selection_scores"].astype(np.float32)[kept]
        sources = payload["proposal_source_ids"].astype("U96")[kept]
        prompt_map = payload["prompt_map"].astype(np.float32)
    g1_logits = logits[0].cpu().numpy().astype(np.float32)
    selected_local, fused = select_candidate(g1_logits, upstream)
    selected_global = int(kept[selected_local])
    selected_mask = all_masks[selected_global]
    Image.fromarray(selected_mask * 255).save(output / "selected_mask_320.png")
    image_path = config.dataset_root / "images" / config.image_id
    with Image.open(image_path) as image:
        native = Image.fromarray(selected_mask * 255).resize(image.size, Image.Resampling.NEAREST)
    native.save(output / "selected_mask_native.png")
    np.save(output / "prompt_map.npy", prompt_map.astype(np.float32), allow_pickle=False)
    score_path = output / "candidate_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("candidate_index", "source", "g1_logit", "upstream_score", "fused_rank_score", "selected"),
        )
        writer.writeheader()
        for local, global_index in enumerate(kept):
            writer.writerow({
                "candidate_index": int(global_index),
                "source": str(sources[local]),
                "g1_logit": float(g1_logits[local]),
                "upstream_score": float(upstream[local]),
                "fused_rank_score": float(fused[local]),
                "selected": int(local == selected_local),
            })
    result = {
        "image_id": config.image_id,
        "tumor_image_label": int(row["tumor"]),
        "candidate_count": int(len(all_masks)),
        "g1_valid_candidate_count": int(len(kept)),
        "selected_candidate_index": selected_global,
        "selected_source": str(sources[selected_local]),
        "bag_probability": float(torch.sigmoid(bag_logit).item()),
        "selected_mask_320": str(output / "selected_mask_320.png"),
        "selected_mask_native": str(output / "selected_mask_native.png"),
        "candidate_scores": str(score_path),
        "spatial_ground_truth_read": False,
    }
    (output / "demo_receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def show_demo(config: DemoConfig):
    """Return a matplotlib figure containing only prediction-time artifacts."""

    import matplotlib.pyplot as plt

    image_path = config.dataset_root / "images" / config.image_id
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    saliency = np.load(
        config.work_dir / "01_biomedclip" / "maps" / f"{Path(config.image_id).stem}.npy",
        allow_pickle=False,
    )
    prompt = np.load(config.work_dir / "05_final" / "prompt_map.npy", allow_pickle=False)
    mask = np.asarray(Image.open(config.work_dir / "05_final" / "selected_mask_native.png")) > 0
    overlay = image.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.asarray([255, 45, 45])).astype(np.uint8)
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = ((image, "Input X-ray"), (saliency, "BiomedCLIP saliency"), (prompt, "Fused prompt evidence"), (overlay, "Final selected mask"))
    for axis, (values, title) in zip(axes, panels, strict=True):
        axis.imshow(values, cmap="magma" if values.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    return figure


__all__ = [
    "CHECKPOINT_HASHES",
    "DemoConfig",
    "generate_biomedclip_demo",
    "generate_candidate_demo",
    "merge_candidate_demo",
    "score_and_select_demo",
    "show_demo",
    "validate_demo",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one stage of the BTXRD demo")
    parser.add_argument(
        "stage",
        choices=("verify", "biomedclip", "anchor", "addition", "merge", "score"),
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = DemoConfig.from_json(args.config)
    if args.stage == "verify":
        result = validate_demo(config)
    elif args.stage == "biomedclip":
        result = generate_biomedclip_demo(config)
    elif args.stage in {"anchor", "addition"}:
        result = generate_candidate_demo(config, args.stage)
    elif args.stage == "merge":
        result = merge_candidate_demo(config)
    else:
        result = score_and_select_demo(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
