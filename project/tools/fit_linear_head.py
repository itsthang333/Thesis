from __future__ import annotations

"""Fit a class-balanced image-level linear head on a frozen DenseNet.

This is an image-label-only classifier experiment.  Polygon annotations are
never loaded, so the resulting checkpoint remains valid for WSSS generation.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.factory import build_classification_dataset
from models.classifier import DenseNet121AnatomyClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-checkpoint", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--C-values", type=str, default="0.01,0.1,1.0,10.0")
    p.add_argument("--target-columns", type=str, default="tumor_type",
                   choices=["tumor_type", "tumor"])
    p.add_argument("--one-vs-rest", action="store_true",
                   help="For tumor_type, fit ten balanced binary heads instead of a multinomial head. "
                        "The resulting logits are image-label-only and can be used for contrastive CAM.")
    return p.parse_args()


def extract(model, dataset, batch_size: int, num_workers: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc=f"features-{dataset.split}"):
            images = images.to(device)
            fmap = model.forward_features(images)
            pooled = model.avgpool(fmap).flatten(1)
            features.append(pooled.cpu().numpy().astype(np.float32))
            labels.append(targets.view(-1).cpu().numpy().astype(np.int64))
    return np.concatenate(features), np.concatenate(labels)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu")
    source_num_classes = int(state.get("num_classes", 10))
    model = DenseNet121AnatomyClassifier(
        num_classes=source_num_classes, pretrained=False,
        anatomy_num_classes=int(state.get("anatomy_num_classes", 0)),
    )
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()

    target_columns = [args.target_columns]
    train_ds = build_classification_dataset(
        "btxrd", args.dataset_root, "train", target_columns, args.image_size,
        normalization=state.get("normalization", "imagenet"),
    )
    val_ds = build_classification_dataset(
        "btxrd", args.dataset_root, "val", target_columns, args.image_size,
        normalization=state.get("normalization", "imagenet"),
    )
    X_train, y_train = extract(model, train_ds, args.batch_size, args.num_workers, device)
    X_val, y_val = extract(model, val_ds, args.batch_size, args.num_workers, device)

    best = None
    for raw_c in args.C_values.split(","):
        C = float(raw_c.strip())
        base_head = LogisticRegression(
                C=C,
                max_iter=500,
                solver="lbfgs",
                multi_class="multinomial" if args.target_columns == "tumor_type" else "auto",
                class_weight="balanced",
                random_state=42,
            )
        if args.one_vs_rest and args.target_columns == "tumor_type":
            head = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    max_iter=500,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
                n_jobs=1,
            )
            y_fit = np.eye(source_num_classes, dtype=np.float32)[y_train]
        else:
            head = base_head
            y_fit = y_train
        head.fit(X_train, y_fit)
        pred = head.predict(X_val)
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)
        score = float(f1_score(y_val, pred, average="macro", zero_division=0))
        print(f"C={C:g} val_macro_f1={score:.6f}")
        if best is None or score > best[0]:
            best = (score, C, head)

    assert best is not None
    score, C, head = best
    new_state = dict(state)
    model_state = dict(state["model_state_dict"])
    if isinstance(head, OneVsRestClassifier):
        coef = np.stack([est.coef_[0] for est in head.estimators_], axis=0)
        intercept = np.asarray([est.intercept_[0] for est in head.estimators_], dtype=np.float32)
    else:
        coef = head.coef_.astype(np.float32)
        intercept = head.intercept_.astype(np.float32)
    model_state["classifier.weight"] = torch.from_numpy(coef.astype(np.float32))
    model_state["classifier.bias"] = torch.from_numpy(intercept.astype(np.float32))
    new_state["model_state_dict"] = model_state
    new_state["best_metric"] = score
    new_state["linear_head_C"] = C
    new_state["linear_head_class_weight"] = "balanced"
    new_state["linear_head_one_vs_rest"] = bool(args.one_vs_rest and args.target_columns == "tumor_type")
    new_state["linear_head_backbone_checkpoint"] = str(args.checkpoint.resolve())
    new_state["num_classes"] = int(coef.shape[0])
    new_state["target_columns"] = target_columns
    new_state["task"] = "multi-label" if args.target_columns == "tumor" else "single-label"
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state, args.output_checkpoint)
    print(f"Saved balanced linear-head checkpoint to {args.output_checkpoint} (val_macro_f1={score:.6f}, C={C:g})")


if __name__ == "__main__":
    main()
