from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from btxrd_wsss.types import CandidateMask


def _atomic_npz(path: Path, **arrays: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return path


def source_map_path(output_dir: str | Path, image_id: str) -> Path:
    return Path(output_dir) / "source_maps" / f"{image_id}.npz"


def save_source_maps(
    output_dir: str | Path,
    image_id: str,
    maps: dict[str, np.ndarray],
    confidences: dict[str, float],
) -> Path:
    return _atomic_npz(
        source_map_path(output_dir, image_id),
        **maps,
        confidences=np.asarray(
            [confidences[key] for key in ("hrnet_full", "hrnet_tile", "biomedclip")], np.float32
        ),
    )


def load_source_maps(
    output_dir: str | Path, image_id: str
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with np.load(source_map_path(output_dir, image_id)) as payload:
        maps = {
            key: payload[key].astype(np.float32)
            for key in ("hrnet_full", "hrnet_tile", "biomedclip")
        }
        values = payload["confidences"].tolist()
    return maps, dict(
        zip(("hrnet_full", "hrnet_tile", "biomedclip"), map(float, values), strict=True)
    )


def gallery_path(output_dir: str | Path, image_id: str) -> Path:
    return Path(output_dir) / "galleries" / f"{image_id}.npz"


def save_gallery(output_dir: str | Path, image_id: str, candidates: list[CandidateMask]) -> Path:
    if not candidates:
        return _atomic_npz(
            gallery_path(output_dir, image_id),
            shape=np.asarray([0, 0]),
            masks=np.empty((0, 0), np.uint8),
            components=np.empty((0, 0), np.uint8),
            metadata=np.asarray("[]"),
        )
    shape = candidates[0].mask.shape
    masks = np.packbits(
        np.stack([item.mask for item in candidates]).reshape(len(candidates), -1), axis=1
    )
    components = np.packbits(
        np.stack([item.metadata["source_component"] for item in candidates]).reshape(
            len(candidates), -1
        ),
        axis=1,
    )
    metadata = []
    for item in candidates:
        extra = {key: value for key, value in item.metadata.items() if key != "source_component"}
        for key, value in list(extra.items()):
            if isinstance(value, np.generic):
                extra[key] = value.item()
            elif isinstance(value, tuple):
                extra[key] = list(value)
        metadata.append(
            {
                "candidate_id": item.candidate_id,
                "proposal_id": item.proposal_id,
                "proposal_source": item.proposal_source,
                "sam_backend": item.sam_backend,
                "prompt_type": item.prompt_type,
                "predicted_iou": item.predicted_iou,
                "stability": item.stability,
                "roi_scale": item.roi_scale,
                "metadata": extra,
            }
        )
    return _atomic_npz(
        gallery_path(output_dir, image_id),
        shape=np.asarray(shape, np.int32),
        masks=masks,
        components=components,
        metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
    )


def load_gallery(output_dir: str | Path, image_id: str) -> list[CandidateMask]:
    with np.load(gallery_path(output_dir, image_id)) as payload:
        shape = tuple(int(value) for value in payload["shape"])
        metadata = json.loads(str(payload["metadata"]))
        if not metadata:
            return []
        count, pixels = len(metadata), int(np.prod(shape))
        masks = (
            np.unpackbits(payload["masks"], axis=1)[:, :pixels].reshape(count, *shape).astype(bool)
        )
        components = (
            np.unpackbits(payload["components"], axis=1)[:, :pixels]
            .reshape(count, *shape)
            .astype(bool)
        )
    return [
        CandidateMask(
            candidate_id=row["candidate_id"],
            mask=masks[index],
            proposal_id=row["proposal_id"],
            proposal_source=row["proposal_source"],
            sam_backend=row["sam_backend"],
            prompt_type=row["prompt_type"],
            predicted_iou=float(row["predicted_iou"]),
            stability=float(row["stability"]),
            roi_scale=float(row["roi_scale"]),
            metadata={**row["metadata"], "source_component": components[index]},
        )
        for index, row in enumerate(metadata)
    ]


def descriptor_path(output_dir: str | Path, image_id: str) -> Path:
    return Path(output_dir) / "descriptors" / f"{image_id}.npz"


def save_descriptors(
    output_dir: str | Path, image_id: str, values: np.ndarray, candidate_ids: tuple[str, ...]
) -> Path:
    return _atomic_npz(
        descriptor_path(output_dir, image_id),
        values=np.asarray(values, np.float32),
        candidate_ids=np.asarray(candidate_ids),
    )


def load_descriptors(output_dir: str | Path, image_id: str) -> tuple[np.ndarray, tuple[str, ...]]:
    with np.load(descriptor_path(output_dir, image_id)) as payload:
        return payload["values"].astype(np.float32), tuple(
            map(str, payload["candidate_ids"].tolist())
        )
