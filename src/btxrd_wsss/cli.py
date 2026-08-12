from __future__ import annotations

import argparse
import json
from pathlib import Path

from btxrd_wsss.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btxrd-wsss")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--require-assets", action="store_true")
    commands.add_parser("show-config")
    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--output")
    commands.add_parser("train-hrnet")
    commands.add_parser("smoke-models")
    for name in ("source-maps", "sam-gallery", "rad-dino", "select", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--splits", default="train,val,test")
        if name in {"source-maps", "sam-gallery", "rad-dino"}:
            command.add_argument("--limit", type=int)
    commands.add_parser("train-g1")
    return parser


def _splits(value: str) -> set[str]:
    result = {item.strip() for item in value.split(",") if item.strip()}
    unknown = result - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    return result


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    result: dict[str, object] = {"command": args.command, "experiment": config.experiment.name}
    if args.command == "preflight":
        from btxrd_wsss.preflight import run_preflight

        result = run_preflight(config, require_assets=args.require_assets)
    elif args.command == "build-manifest":
        from btxrd_wsss.data.build_manifest import build_manifest

        path = build_manifest(
            data_root=config.data.root,
            output_path=args.output or config.data.manifest,
            tumor_columns=config.data.tumor_columns,
            seed=config.experiment.seed,
        )
        result["manifest"] = str(path.resolve())
    elif args.command == "train-hrnet":
        from btxrd_wsss.stages.hrnet import train_hrnet

        result["checkpoint"] = str(train_hrnet(config))
    elif args.command == "smoke-models":
        from btxrd_wsss.stages.smoke import smoke_models

        result["models"] = smoke_models(config)
    elif args.command == "source-maps":
        from btxrd_wsss.stages.supply import generate_source_maps

        generate_source_maps(config, splits=_splits(args.splits), limit=args.limit)
    elif args.command == "sam-gallery":
        from btxrd_wsss.stages.supply import generate_sam_galleries

        generate_sam_galleries(config, splits=_splits(args.splits), limit=args.limit)
    elif args.command == "rad-dino":
        from btxrd_wsss.stages.supply import generate_rad_dino_descriptors

        generate_rad_dino_descriptors(config, splits=_splits(args.splits), limit=args.limit)
    elif args.command == "train-g1":
        from btxrd_wsss.stages.g1 import train_g1

        result["checkpoint"] = str(train_g1(config))
    elif args.command == "select":
        from btxrd_wsss.stages.g1 import run_final_selection

        run_final_selection(config, splits=_splits(args.splits))
    elif args.command == "evaluate":
        from btxrd_wsss.stages.evaluate import evaluate_spatial_stages

        evaluate_spatial_stages(config, splits=_splits(args.splits))
    else:
        result.update(
            {
                "config": str(Path(args.config).resolve()),
                "stages": ["hrnet", "source_maps", "adaptive_sam", "rad_dino", "g1", "selection"],
                "sam_raw_cap": config.sam.maximum_raw_candidates,
                "g1_gallery_cap": config.sam.maximum_selected_candidates,
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
