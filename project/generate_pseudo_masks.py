from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.fracatlas import FracAtlasClassificationDataset
from models.classifier import DenseNet121AnatomyClassifier
from models.gradcam import GradCAM
from pseudo.cam_to_mask import aggregate_cam_heatmaps, cam_to_pseudo_mask, normalize_min_max
from pseudo.visualization import save_mask, save_overlay, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo masks from Grad-CAM")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "FracAtlas")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "classifier" / "best.pt")
    parser.add_argument("--target-columns", type=str, default="hand,leg,hip,shoulder")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pseudo_masks")
    parser.add_argument("--percentile", type=float, default=80.0)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--fusion", type=str, default="weighted_mean", choices=("weighted_mean", "max"))
    parser.add_argument("--use-clahe", action="store_true")
    return parser.parse_args()


def load_classifier(checkpoint_path: Path, num_classes: int, device: torch.device) -> DenseNet121AnatomyClassifier:
    model = DenseNet121AnatomyClassifier(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path or (args.data_root / "dataset.csv")
    image_root = args.image_root or (args.data_root / "images")
    target_columns = [column.strip() for column in args.target_columns.split(",") if column.strip()]

    dataset = FracAtlasClassificationDataset(
        csv_path=csv_path,
        image_roots=image_root,
        target_columns=target_columns,
        image_size=args.image_size,
        augment=False,
        use_clahe=args.use_clahe,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier(args.checkpoint, num_classes=len(target_columns), device=device)
    gradcam = GradCAM(model, model.features.denseblock4, device=device)

    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for images, _, image_names in tqdm(loader, desc="pseudo-masks"):
        images = images.to(device)

        for index, image_name in enumerate(image_names):
            image_tensor = images[index : index + 1]
            with torch.no_grad():
                logits = model(image_tensor)
                class_weights = torch.sigmoid(logits)[0].detach().cpu().numpy()

            per_class_cams: list[np.ndarray] = []
            class_names = target_columns
            for class_index, class_name in enumerate(class_names):
                cam_output = gradcam.cam_for_class(image_tensor, class_index=class_index)
                per_class_cams.append(cam_output.cam[0].detach().cpu().numpy())
                save_overlay(
                    tensor_to_pil(image_tensor[0].detach().cpu()),
                    normalize_min_max(per_class_cams[-1]),
                    overlay_dir / f"{Path(image_name).stem}_{class_name}.png",
                )

            aggregated_cam = aggregate_cam_heatmaps(per_class_cams, weights=class_weights, mode=args.fusion)
            pseudo_mask = cam_to_pseudo_mask(
                aggregated_cam,
                percentile=args.percentile,
                min_area=args.min_area,
                kernel_size=args.kernel_size,
            )

            image_pil = tensor_to_pil(image_tensor[0].detach().cpu())
            mask_path = mask_dir / f"{Path(image_name).stem}.png"
            overlay_path = overlay_dir / f"{Path(image_name).stem}_aggregated.png"
            save_mask(pseudo_mask, mask_path)
            save_overlay(image_pil, aggregated_cam, overlay_path)

    gradcam.close()


if __name__ == "__main__":
    main()
