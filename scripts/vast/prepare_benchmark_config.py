from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-source")
    args = parser.parse_args()

    source, target = Path(args.source), Path(args.target)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["experiment"]["output_dir"] = args.output_dir
    payload["hrnet"]["epochs"] = 1
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    if args.checkpoint_source:
        source_checkpoints = Path(args.checkpoint_source)
        target_checkpoints = Path(args.output_dir) / "checkpoints"
        target_checkpoints.mkdir(parents=True, exist_ok=True)
        for name in ("hrnet_best.pt", "hrnet_last.pt", "hrnet_history.json"):
            candidate = source_checkpoints / name
            if candidate.exists():
                shutil.copy2(candidate, target_checkpoints / name)


if __name__ == "__main__":
    main()
