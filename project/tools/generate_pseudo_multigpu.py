from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pseudo-mask generation on one or two independent GPUs and merge shards."
    )
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "generator_args",
        nargs=argparse.REMAINDER,
        help="Arguments for generate_pseudo_masks.py, placed after --.",
    )
    return parser.parse_args()


def merge_shards(output_dir: Path, shard_dirs: list[Path], num_gpus: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for shard_dir in shard_dirs:
        for directory_name in ("masks", "overlays", "candidate_cache"):
            source = shard_dir / directory_name
            if source.exists():
                shutil.copytree(source, output_dir / directory_name, dirs_exist_ok=True)

    skipped: set[str] = set()
    for shard_dir in shard_dirs:
        path = shard_dir / "skipped_low_confidence.txt"
        if path.exists():
            skipped.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if skipped:
        (output_dir / "skipped_low_confidence.txt").write_text(
            "\n".join(sorted(skipped)) + "\n", encoding="utf-8"
        )

    metadata_path = shard_dirs[0] / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update(
        {
            "parallel_generation": "independent_gpu_shards",
            "num_gpus": num_gpus,
            "shard_directories": [str(path) for path in shard_dirs],
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    generator_args = list(args.generator_args)
    if generator_args and generator_args[0] == "--":
        generator_args = generator_args[1:]
    forbidden = {
        "--output-dir", "--num-shards", "--shard-index",
        "--classifier-device", "--sam-device",
    }
    supplied = {token.split("=", 1)[0] for token in generator_args if token.startswith("--")}
    overlap = sorted(forbidden & supplied)
    if overlap:
        raise ValueError(
            "The multi-GPU wrapper owns these arguments; remove them after --: "
            + ", ".join(overlap)
        )
    available = torch.cuda.device_count()
    if available < args.num_gpus:
        raise RuntimeError(
            f"Requested {args.num_gpus} GPU(s), but PyTorch sees only {available}."
        )

    output_dir = args.output_dir.resolve()
    shard_root = output_dir / "_gpu_shards"
    shard_dirs = [shard_root / f"gpu_{index}" for index in range(args.num_gpus)]
    processes: list[tuple[int, subprocess.Popen]] = []
    for index, shard_dir in enumerate(shard_dirs):
        shard_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT / "generate_pseudo_masks.py"),
            *generator_args,
            "--output-dir", str(shard_dir),
            "--num-shards", str(args.num_gpus),
            "--shard-index", str(index),
            "--classifier-device", "cuda",
            "--sam-device", "cuda",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(index)
        print(f"[launcher] GPU {index}: {' '.join(command)}", flush=True)
        processes.append(
            (index, subprocess.Popen(command, cwd=ROOT, env=environment))
        )

    failures: list[tuple[int, int]] = []
    for index, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((index, return_code))
    if failures:
        raise RuntimeError(f"Pseudo-mask shard failures: {failures}")

    merge_shards(output_dir, shard_dirs, args.num_gpus)
    mask_count = len(list((output_dir / "masks").glob("*.png")))
    cache_count = len(list((output_dir / "candidate_cache").glob("*.npz")))
    print(
        f"[launcher] merged {mask_count} masks and {cache_count} candidate caches "
        f"into {output_dir}"
    )


if __name__ == "__main__":
    main()
