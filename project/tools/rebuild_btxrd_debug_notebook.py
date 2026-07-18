from __future__ import annotations

import json
from pathlib import Path


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells: list[dict] = []

cells.append(markdown("""# BTXRD WSSS Debug Notebook — canonical `btxrd_best`

This notebook is intentionally split into small, inspectable stages. It calls the current repository scripts rather
than reimplementing the production pipeline in hidden notebook code.

```text
tumor_type (10 image-level classes)
 -> DenseNet121 CE, 320 px
 -> LayerCAM denseblock2/3/4, weights 0.2/0.3/0.5
 -> class-vs-normal contrast
 -> CAM percentiles 85/90/95
 -> up to 3 CAM components
 -> box + positive/negative points
 -> SAM ViT-B, 512 px
 -> box_point + point + box prompt ensemble
 -> coverage_mass_sam, component_topk=1
 -> tumor morphology and pseudo-mask
```

`predicted` and `ground_truth` CAM protocols are always written to separate directories. Polygon GT is loaded only in
explicit diagnostic/evaluation cells and is never passed to classifier, CAM, prompts, SAM, candidate selection, or
post-processing.
"""))

cells.append(markdown("""## Execution order and switches

1. Runtime/repository/dataset audit
2. Split and leakage checks
3. Ground-truth visualization (diagnostic only)
4. Classifier training and CAM snapshots
5. Single-image CAM/morphology/prompt/SAM debug trace
6. Separate full validation runs for `predicted` and `ground_truth`
7. Metrics, oracle decomposition, qualitative panels, and artifact manifest

Expensive stages are controlled in Cell 1. No test-set tuning is performed.
"""))

cells.append(code(r'''
# Cell 1 — paths and run switches
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys

NOTEBOOK_ROOT = Path.cwd()
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
DEFAULT_WORKING = KAGGLE_WORKING if KAGGLE_WORKING.exists() else NOTEBOOK_ROOT / "notebook_runs"
REPO_URL = os.environ.get("BTXRD_REPO_URL", "https://github.com/itsthang333/Thesis.git")
GIT_BRANCH = os.environ.get("BTXRD_GIT_BRANCH", "TN_exp")
DATASET_OVERRIDE = os.environ.get("BTXRD_ROOT", "")

if (NOTEBOOK_ROOT / "project").exists():
    PROJECT_PARENT = NOTEBOOK_ROOT
else:
    PROJECT_PARENT = KAGGLE_WORKING / "Thesis"
PROJECT_DIR = PROJECT_PARENT / "project"

OUTPUT_ROOT = Path(os.environ.get("BTXRD_OUTPUT", str(DEFAULT_WORKING / "btxrd_debug")))
CLASSIFIER_OUTPUT = OUTPUT_ROOT / "classifier_btxrd_best"
PREDICTED_OUTPUT = OUTPUT_ROOT / "pseudo_predicted"
GROUND_TRUTH_OUTPUT = OUTPUT_ROOT / "pseudo_ground_truth"
DEBUG_OUTPUT = OUTPUT_ROOT / "single_image_debug"
EVAL_OUTPUT = OUTPUT_ROOT / "evaluations"
DEFAULT_SAM = NOTEBOOK_ROOT / "sam_vit_b_01ec64.pth" if (NOTEBOOK_ROOT / "sam_vit_b_01ec64.pth").exists() else OUTPUT_ROOT / "sam_vit_b_01ec64.pth"
SAM_CHECKPOINT = Path(os.environ.get("SAM_CHECKPOINT", str(DEFAULT_SAM)))

IMAGE_SIZE = 320
SAM_IMAGE_SIZE = 512
NUM_WORKERS = int(os.environ.get("BTXRD_NUM_WORKERS", "2"))
DEBUG_IMAGE_NAME = os.environ.get("BTXRD_DEBUG_IMAGE", "")

INSTALL_DEPENDENCIES = True
RUN_TRAIN_CLASSIFIER = True
RUN_SINGLE_IMAGE_DEBUG = True
RUN_FULL_PREDICTED = True
RUN_FULL_GROUND_TRUTH = True
RUN_SUPERVISED_ORACLE_BASELINE = False
RUN_TEST_REPORT = False

for path in [OUTPUT_ROOT, CLASSIFIER_OUTPUT, PREDICTED_OUTPUT, GROUND_TRUTH_OUTPUT, DEBUG_OUTPUT, EVAL_OUTPUT]:
    path.mkdir(parents=True, exist_ok=True)
print("PROJECT_DIR:", PROJECT_DIR)
print("GIT_BRANCH:", GIT_BRANCH)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
'''))

cells.append(markdown("""## 1. Repository and environment setup

The notebook checks out a specific branch and uses the source files directly. This prevents notebook-only logic from
drifting away from `train_classifier.py` and `generate_pseudo_masks.py`.
"""))

cells.append(code(r'''
# Cell 2 — checkout, imports, and streaming subprocess helper
if not PROJECT_DIR.exists():
    PROJECT_PARENT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--branch", GIT_BRANCH, REPO_URL, str(PROJECT_PARENT)], check=True)
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

def run_streaming(cmd, cwd=PROJECT_DIR, check=True):
    cmd = [str(x) for x in cmd]
    if cmd and Path(cmd[0]).name.lower().startswith("python"):
        cmd = [cmd[0], "-u", *cmd[1:]]
    print("$", " ".join(shlex.quote(x) for x in cmd))
    process = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    result = process.wait()
    if check and result:
        raise subprocess.CalledProcessError(result, cmd)
    return result

if INSTALL_DEPENDENCIES:
    run_streaming([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT_DIR / "requirements.txt")])
    run_streaming([sys.executable, "-m", "pip", "install", "-q", "pandas", "openpyxl", "opencv-python"])
print("cwd:", Path.cwd())
'''))

cells.append(code(r'''
# Cell 3 — hardware/runtime audit
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(index, torch.cuda.get_device_name(index), round(props.total_memory / 2**30, 2), "GiB")
'''))

cells.append(markdown("""## 2. Dataset resolution, split audit, and leakage guard

BTXRD has no predefined split. The loader derives a deterministic 80/10/10 split stratified by all ten `tumor_type`
classes with seed 42. The polygon segmentation dataset is instantiated below only for visualization/evaluation.
"""))

cells.append(code(r'''
# Cell 4 — locate BTXRD and inspect records
from datasets.btxrd import (
    TUMOR_TYPE_CLASS_NAMES, BTXRDSegmentationDataset, load_btxrd_records,
    resolve_btxrd_root, split_btxrd_records,
)
from datasets.factory import build_classification_dataset, build_segmentation_dataset

def find_btxrd_root(base: Path):
    candidates = [base, *base.glob("*"), *base.glob("*/*")] if base and base.exists() else []
    for candidate in candidates:
        try:
            return resolve_btxrd_root(candidate)
        except FileNotFoundError:
            pass
    return None

search_root = Path(DATASET_OVERRIDE) if DATASET_OVERRIDE else (KAGGLE_INPUT if KAGGLE_INPUT.exists() else NOTEBOOK_ROOT)
BTXRD_ROOT = find_btxrd_root(search_root)
if BTXRD_ROOT is None:
    raise FileNotFoundError("BTXRD not found. Set BTXRD_ROOT or attach images/, Annotations/, dataset.csv/xlsx.")
records = load_btxrd_records(BTXRD_ROOT)
print("BTXRD_ROOT:", BTXRD_ROOT)
print("records:", len(records))
'''))

cells.append(code(r'''
# Cell 5 — split/class distribution and deterministic image list
def split_frame(name):
    rows = split_btxrd_records(records, split=name, seed=42)
    frame = pd.DataFrame(rows)
    frame["tumor_type_name"] = frame["tumor_type"].map(dict(enumerate(TUMOR_TYPE_CLASS_NAMES)))
    return frame

split_frames = {name: split_frame(name) for name in ["train", "val", "test"]}
parts = []
for name, frame in split_frames.items():
    counts = frame["tumor_type_name"].value_counts().reindex(TUMOR_TYPE_CLASS_NAMES, fill_value=0)
    parts.append(pd.DataFrame({"split": name, "class": counts.index, "count": counts.values}))
    print(f"{name}: n={len(frame)} tumor={int((frame.tumor_type > 0).sum())} normal={int((frame.tumor_type == 0).sum())}")
split_summary = pd.concat(parts, ignore_index=True)
display(split_summary.pivot(index="class", columns="split", values="count"))
split_summary.pivot(index="class", columns="split", values="count").plot.bar(figsize=(14, 4), grid=True)
plt.title("BTXRD 80/10/10 stratification by tumor_type"); plt.tight_layout(); plt.show()

if not DEBUG_IMAGE_NAME:
    DEBUG_IMAGE_NAME = str(split_frames["val"].query("tumor_type > 0").iloc[0]["image_id"])
DEBUG_IMAGE_LIST = OUTPUT_ROOT / "debug_image_list.txt"
DEBUG_IMAGE_LIST.write_text(DEBUG_IMAGE_NAME + "\n", encoding="utf-8")
print("DEBUG_IMAGE_NAME:", DEBUG_IMAGE_NAME)
'''))

cells.append(code(r'''
# Cell 6 — leakage guard: classification labels vs polygon masks
classification_ds = build_classification_dataset(
    "btxrd", root=BTXRD_ROOT, split="val", target_columns=["tumor_type"], image_size=IMAGE_SIZE,
)
segmentation_ds = build_segmentation_dataset("btxrd", root=BTXRD_ROOT, split="val", image_size=IMAGE_SIZE, augment=False)
classification_names = {str(x["image_id"]) for x in classification_ds.samples}
segmentation_names = {str(x["image_id"]) for x in segmentation_ds.samples}
print("classification records:", len(classification_names))
print("segmentation records:", len(segmentation_names))
print("same IDs:", classification_names == segmentation_names)
assert classification_ds.target_columns == ["tumor_type"]
print("Generation will use image-level target only:", classification_ds.target_columns)
print("Segmentation masks remain diagnostic/evaluation-only objects.")
'''))

cells.append(markdown("""## 3. Ground-truth visualization (diagnostic only)

This cell rasterizes LabelMe polygons so annotation errors can be seen. It must not be used as an input to any
generation cell. The final plot title explicitly marks the polygon as diagnostic.
"""))

cells.append(code(r'''
# Cell 7 — original image, polygon GT, and overlay
def show_gt_case(want_tumor: bool):
    frame = split_frames["val"]
    row = frame[frame.tumor_type.gt(0) if want_tumor else frame.tumor_type.eq(0)].iloc[0]
    image_name = str(row.image_id)
    image = Image.open(BTXRD_ROOT / "images" / image_name).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    index = next(i for i, x in enumerate(segmentation_ds.samples) if str(x["image_id"]) == image_name)
    _, mask_tensor, _ = segmentation_ds[index]
    mask = mask_tensor[0].numpy() > 0.5
    image_np = np.asarray(image)
    overlay = image_np.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 30, 30])).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(image_np); axes[0].set_title(f"{image_name}\noriginal")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("polygon GT")
    axes[2].imshow(overlay); axes[2].set_title("diagnostic overlay")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()

show_gt_case(True)
show_gt_case(False)
'''))

cells.append(markdown("""## 4. SAM checkpoint and canonical classifier training

The next cells train the paired image-level classifier using `--pipeline-profile btxrd_best`. The profile fixes
`tumor_type`, 320 px, batch 4, 25 epochs, seed 42, inverse-frequency CE, and disables PuzzleCAM/teacher-attention
losses for the selected CE/320 model.
"""))

cells.append(code(r'''
# Cell 8 — SAM checkpoint
SAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
if not SAM_CHECKPOINT.exists():
    import urllib.request
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", str(SAM_CHECKPOINT))
print("SAM:", SAM_CHECKPOINT, "GiB:", round(SAM_CHECKPOINT.stat().st_size / 2**30, 3))

# Cell 9 — train classifier
CLASSIFIER_CHECKPOINT = CLASSIFIER_OUTPUT / "best_classifier.pt"
classifier_cmd = [
    sys.executable, "train_classifier.py", "--dataset", "btxrd", "--pipeline-profile", "btxrd_best",
    "--ram-root", str(BTXRD_ROOT), "--num-workers", str(NUM_WORKERS),
    "--save-cam-epochs", "1,5,10,15,20,25", "--cam-preview-count", "4", "--output-dir", str(CLASSIFIER_OUTPUT),
]
if RUN_TRAIN_CLASSIFIER:
    run_streaming(classifier_cmd)
else:
    print("RUN_TRAIN_CLASSIFIER=False")
if not CLASSIFIER_CHECKPOINT.exists():
    raise FileNotFoundError(CLASSIFIER_CHECKPOINT)
'''))

cells.append(code(r'''
# Cell 10 — training curves and checkpoint audit
training_log = CLASSIFIER_OUTPUT / "training_log.csv"
if training_log.exists():
    train_df = pd.read_csv(training_log)
    display(train_df)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    train_df[["train_loss", "val_loss"]].plot(ax=axes[0], marker="o", title="CE loss")
    train_df[["train_acc", "val_acc"]].plot(ax=axes[1], marker="o", title="accuracy")
    train_df[["train_f1", "val_f1"]].plot(ax=axes[2], marker="o", title="macro-F1")
    for ax in axes: ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

state = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu")
expected = {"task": "single-label", "target_columns": ["tumor_type"], "num_classes": 10, "normalization": "imagenet"}
for key, value in expected.items():
    print(key, state.get(key), "expected", value)
    assert state.get(key) == value
print("checkpoint:", CLASSIFIER_CHECKPOINT)
'''))

cells.append(code(r'''
# Cell 11 — CAM snapshots across epochs
from collections import defaultdict
import re
cam_dir = CLASSIFIER_OUTPUT / "cam_preview"
by_sample = defaultdict(list)
for path in sorted(cam_dir.glob("cam_epoch*.png")) if cam_dir.exists() else []:
    match = re.match(r"cam_epoch(\d+)_(.+)\.png", path.name)
    if match: by_sample[match.group(2)].append((int(match.group(1)), path))
if by_sample:
    rows = sorted(by_sample.items()); ncols = max(len(x) for _, x in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.2*ncols, 3.2*len(rows)), squeeze=False)
    for r, (stem, entries) in enumerate(rows):
        for c, (epoch, path) in enumerate(sorted(entries)):
            axes[r, c].imshow(Image.open(path)); axes[r, c].set_title(f"{stem}\nepoch {epoch}"); axes[r, c].axis("off")
        for c in range(len(entries), ncols): axes[r, c].axis("off")
    plt.tight_layout(); plt.show()
else:
    print("No CAM snapshots found.")
'''))

cells.append(markdown("""## 5. Single-image pre-SAM trace

This trace stops before SAM and exposes the exact class choice, class-vs-normal CAM, percentile supports, connected
components, box padding, and positive/negative points. The `predicted` trace is end-to-end behavior; the optional
`ground_truth` trace is a localization diagnostic and is never passed to pseudo-mask generation unless explicitly run
as the separate oracle protocol below.
"""))

cells.append(code(r'''
# Cell 12 - LayerCAM, threshold, components, boxes, and points for one image
from models.layercam import LayerCAM
from pseudo.generate_layercam import generate_fused_cam
from pseudo.tumor_morphology import build_class_conditioned_components
from generate_pseudo_masks import load_classifier

classifier, checkpoint_meta = load_classifier(
    CLASSIFIER_CHECKPOINT, fallback_num_classes=10, device=torch.device("cpu"), expected_target_columns=["tumor_type"],
    expected_task="single-label", expected_num_classes=10,
)
classifier.eval()
trace_ds = build_classification_dataset(
    "btxrd", root=BTXRD_ROOT, split="val", target_columns=["tumor_type"], image_size=IMAGE_SIZE,
)
trace_index = next(i for i, x in enumerate(trace_ds.samples) if str(x["image_id"]) == DEBUG_IMAGE_NAME)
trace_tensor, trace_target, _ = trace_ds[trace_index]
trace_rgb = np.asarray(Image.open(BTXRD_ROOT / "images" / DEBUG_IMAGE_NAME).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))

def _norm_map(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def trace_pre_sam(image_name=DEBUG_IMAGE_NAME, protocol="predicted"):
    idx = next(i for i, x in enumerate(trace_ds.samples) if str(x["image_id"]) == str(image_name))
    image_tensor, target, _ = trace_ds[idx]
    batch = image_tensor.unsqueeze(0)
    with torch.no_grad():
        logits = classifier(batch)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    selected = int(probs.argmax()) if protocol == "predicted" else int(target)
    if selected == 0:
        selected = int(np.argsort(probs[1:])[-1] + 1)
    weights = np.zeros(10, dtype=np.float32); weights[selected] = 1.0
    cam_engine = LayerCAM(
        classifier, device=torch.device("cpu"), layer_weights=[0.2, 0.3, 0.5], gradient_mode="positive",
    )
    fused_cam, _, _ = generate_fused_cam(cam_engine, batch, class_weights=weights, confidence_threshold=0.5)
    contrast_output = cam_engine.cam_for_class_contrast(batch, selected, reference_index=0)
    contrast = _norm_map(contrast_output.cam.detach().cpu().numpy()[0])
    components = {}
    for percentile in [85, 90, 95]:
        likelihood, support, component_list = build_class_conditioned_components(
            trace_rgb, [contrast], [1.0], cam_percentile=percentile,
            min_component_area=100, max_components=3, points_per_component=5,
            bbox_padding_ratio=0.02, negative_points_per_component=4,
        )
        components[percentile] = {"likelihood": likelihood, "support_mask": support, "components": component_list}
    return {"image": trace_rgb, "target": int(target), "selected": selected, "probs": probs,
            "cam": fused_cam, "contrast": contrast, "components": components}

trace_predicted = trace_pre_sam(protocol="predicted")
trace_ground_truth = trace_pre_sam(protocol="ground_truth")
print("image:", DEBUG_IMAGE_NAME, "image-level target:", trace_predicted["target"],
      "predicted class:", trace_predicted["selected"], "GT class:", trace_ground_truth["selected"])

def plot_pre_sam_trace(trace, title):
    fig, axes = plt.subplots(1, 5, figsize=(21, 4))
    axes[0].imshow(trace["image"]); axes[0].set_title(f"Original\n{title}")
    axes[1].imshow(trace["image"]); axes[1].imshow(trace["cam"], cmap="magma", alpha=.48); axes[1].set_title("Fused LayerCAM")
    axes[2].imshow(trace["image"]); axes[2].imshow(trace["contrast"], cmap="jet", alpha=.48); axes[2].set_title("Class-vs-normal CAM")
    support = trace["components"][85]["support_mask"]
    axes[3].imshow(trace["image"]); axes[3].imshow(support, cmap="Greens", alpha=.45); axes[3].set_title("p85 support")
    axes[4].imshow(trace["image"])
    for comp in trace["components"][85]["components"]:
        x0, y0, x1, y1 = comp["bbox"]
        axes[4].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color="yellow", linewidth=2))
        pos = np.asarray(comp["positive_points"]); neg = np.asarray(comp["negative_points"])
        if len(pos): axes[4].scatter(pos[:, 0], pos[:, 1], c="red", s=25, label="positive")
        if len(neg): axes[4].scatter(neg[:, 0], neg[:, 1], c="cyan", s=20, label="negative")
    axes[4].set_title("Box + points")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()

plot_pre_sam_trace(trace_predicted, "predicted class")
plot_pre_sam_trace(trace_ground_truth, "ground-truth class (diagnostic)")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, percentile in zip(axes, [85, 90, 95]):
    ax.imshow(trace_predicted["image"])
    ax.imshow(trace_predicted["components"][percentile]["support_mask"], cmap="Greens", alpha=.45)
    ax.set_title(f"predicted support p{percentile}\\ncomponents={len(trace_predicted['components'][percentile]['components'])}")
    ax.axis("off")
plt.tight_layout(); plt.show()
'''))

cells.append(markdown("""## 6. Actual SAM candidate trace (one image)

The next cell invokes the production generator with `--debug`, so the displayed candidates are not a notebook
reimplementation. It records every SAM mask and score before selection, plus morphology intermediates.
"""))

cells.append(code(r'''
# Cell 13 - run the production generator on one deterministic validation image
DEBUG_OUTPUT.mkdir(parents=True, exist_ok=True)
debug_cmd = [
    sys.executable, "generate_pseudo_masks.py", "--dataset", "btxrd", "--pipeline-profile", "btxrd_best",
    "--ram-root", str(BTXRD_ROOT), "--split", "val", "--classifier-checkpoint", str(CLASSIFIER_CHECKPOINT),
    "--sam-checkpoint", str(SAM_CHECKPOINT), "--image-list", str(DEBUG_IMAGE_LIST), "--max-images", "1",
    "--debug", "--evaluate-prompt-quality", "--save-visuals-limit", "1", "--output-dir", str(DEBUG_OUTPUT),
]
if RUN_SINGLE_IMAGE_DEBUG:
    run_streaming(debug_cmd)
else:
    print("RUN_SINGLE_IMAGE_DEBUG=False; command prepared only.")
'''))

cells.append(code(r'''
# Cell 14 - display actual CAM/morphology/SAM artifacts and candidate table
debug_case_dir = DEBUG_OUTPUT / "debug" / Path(DEBUG_IMAGE_NAME).stem
def display_artifact(path, title=None, cmap=None):
    if not path.exists():
        print("missing:", path); return
    image = Image.open(path)
    plt.figure(figsize=(4, 4)); plt.imshow(image, cmap=cmap); plt.title(title or path.name); plt.axis("off"); plt.show()

for filename in ["simple_tumor_likelihood.png", "simple_tumor_support.png", "tumor_likelihood.png",
                 "tumor_seeds.png", "tumor_support.png"]:
    display_artifact(debug_case_dir / filename)
display_artifact(DEBUG_OUTPUT / "masks" / f"{Path(DEBUG_IMAGE_NAME).stem}.png", "final pseudo-mask", cmap="gray")
score_path = debug_case_dir / "scores.json"
if score_path.exists():
    score_records = json.loads(score_path.read_text(encoding="utf-8"))
    candidate_table = pd.DataFrame.from_dict(score_records, orient="index").rename_axis("candidate").reset_index()
    display(candidate_table.head(12))
    overlay_paths = sorted(debug_case_dir.glob("overlay_mask_*.png"))[:12]
    if overlay_paths:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12)); axes = axes.ravel()
        for ax, path in zip(axes, overlay_paths):
            ax.imshow(Image.open(path)); ax.set_title(path.name); ax.axis("off")
        for ax in axes[len(overlay_paths):]: ax.axis("off")
        plt.tight_layout(); plt.show()
else:
    print("No scores.json yet; run Cell 13 first.")
'''))

cells.append(markdown("""## 7. End-to-end predicted protocol

`CAM_TARGET_CLASS=predicted` is the deployable protocol. Its classifier prediction, not the image-level annotation,
chooses the CAM class. Results are saved under a dedicated directory and are never merged with the localization
protocol.
"""))

cells.append(code(r'''
# Cell 15 - full validation pseudo-masks: predicted protocol
PREDICTED_CMD = [
    sys.executable, "generate_pseudo_masks.py", "--dataset", "btxrd", "--pipeline-profile", "btxrd_best",
    "--ram-root", str(BTXRD_ROOT), "--split", "val", "--classifier-checkpoint", str(CLASSIFIER_CHECKPOINT),
    "--sam-checkpoint", str(SAM_CHECKPOINT), "--process-all", "--evaluate-prompt-quality",
    "--save-visuals-limit", "10", "--output-dir", str(PREDICTED_OUTPUT),
]
if RUN_FULL_PREDICTED:
    run_streaming(PREDICTED_CMD)
else:
    print("RUN_FULL_PREDICTED=False; command prepared only.")
'''))

cells.append(code(r'''
# Cell 16 - evaluate predicted protocol against polygons (evaluation only)
PREDICTED_EVAL = EVAL_OUTPUT / "predicted.csv"
predicted_eval_cmd = [
    sys.executable, "evaluate_ramh1200_masks.py", "--dataset", "btxrd", "--ram-root", str(BTXRD_ROOT),
    "--split", "val", "--image-size", str(IMAGE_SIZE), "--pred-mask-root", str(PREDICTED_OUTPUT / "masks"),
    "--output-csv", str(PREDICTED_EVAL),
]
if RUN_FULL_PREDICTED:
    run_streaming(predicted_eval_cmd)
else:
    print("Evaluation command prepared:", predicted_eval_cmd)
'''))

cells.append(markdown("""## 8. Localization/oracle protocol (kept separate)

`CAM_TARGET_CLASS=ground_truth` supplies the known image-level class only to test localization quality. It is not an
end-to-end result and must not be compared as if it were deployment inference. Polygon masks remain evaluation-only.
"""))

cells.append(code(r'''
# Cell 17 - full validation pseudo-masks: ground-truth-class localization protocol
GROUND_TRUTH_CMD = [
    sys.executable, "generate_pseudo_masks.py", "--dataset", "btxrd", "--pipeline-profile", "btxrd_best",
    "--ram-root", str(BTXRD_ROOT), "--split", "val", "--classifier-checkpoint", str(CLASSIFIER_CHECKPOINT),
    "--sam-checkpoint", str(SAM_CHECKPOINT), "--process-all", "--cam-target-class", "ground_truth",
    "--evaluate-prompt-quality", "--save-visuals-limit", "10", "--output-dir", str(GROUND_TRUTH_OUTPUT),
]
if RUN_FULL_GROUND_TRUTH:
    run_streaming(GROUND_TRUTH_CMD)
else:
    print("RUN_FULL_GROUND_TRUTH=False; command prepared only.")
'''))

cells.append(code(r'''
# Cell 18 - evaluate ground-truth-class protocol separately
GROUND_TRUTH_EVAL = EVAL_OUTPUT / "ground_truth.csv"
ground_truth_eval_cmd = [
    sys.executable, "evaluate_ramh1200_masks.py", "--dataset", "btxrd", "--ram-root", str(BTXRD_ROOT),
    "--split", "val", "--image-size", str(IMAGE_SIZE), "--pred-mask-root", str(GROUND_TRUTH_OUTPUT / "masks"),
    "--output-csv", str(GROUND_TRUTH_EVAL),
]
if RUN_FULL_GROUND_TRUTH:
    run_streaming(ground_truth_eval_cmd)
else:
    print("Evaluation command prepared:", ground_truth_eval_cmd)
'''))

cells.append(markdown("""## 9. Metrics and oracle diagnostics

The evaluation CSVs are summarized side by side. The prompt-quality file decomposes errors into CAM/support loss,
SAM candidate loss, selection loss, and post-processing delta. This is the main debugging table; macro-F1 is not a
localization metric and is intentionally not substituted for Dice.
"""))

cells.append(code(r'''
# Cell 19 - keep predicted and ground_truth metrics separate, then decompose failures
def read_eval_summary(path):
    if not path.exists(): return {}
    rows = pd.read_csv(path, header=None)
    result = {}
    for row in rows.itertuples(index=False, name=None):
        if len(row) >= 5:
            result[str(row[0])] = row[4] if row[4] == row[4] else (row[5] if len(row) > 5 else np.nan)
    return result

protocol_summary = pd.DataFrame([
    {"protocol": "predicted", **read_eval_summary(PREDICTED_EVAL)},
    {"protocol": "ground_truth", **read_eval_summary(GROUND_TRUTH_EVAL)},
])
display(protocol_summary.T)

def load_quality(output_dir, protocol):
    path = output_dir / "prompt_quality.csv"
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.insert(0, "protocol", protocol)
    return frame

quality = pd.concat([
    load_quality(PREDICTED_OUTPUT, "predicted"),
    load_quality(GROUND_TRUTH_OUTPUT, "ground_truth"),
], ignore_index=True)
if not quality.empty:
    diagnostic_columns = ["protocol", "tumor_type", "foreground_iou", "foreground_recall", "foreground_precision",
                          "point_hit_rate", "box_recall", "box_precision", "oracle_best_single_dice",
                          "oracle_best_single_dice_clipped", "selected_dice", "support_loss_dice",
                          "selection_loss_dice", "final_dice", "postprocess_delta_dice"]
    display(quality[diagnostic_columns].describe(include="all").T)
    display(quality.groupby(["protocol", "tumor_type"])[["selected_dice", "final_dice", "support_loss_dice",
                                                           "selection_loss_dice"]].mean())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    quality.boxplot(column="foreground_iou", by="protocol", ax=axes[0]); axes[0].set_title("CAM/prompt foreground IoU")
    quality.boxplot(column="selection_loss_dice", by="protocol", ax=axes[1]); axes[1].set_title("Selection loss")
    quality.boxplot(column="postprocess_delta_dice", by="protocol", ax=axes[2]); axes[2].set_title("Post-process delta")
    for ax in axes: ax.set_xlabel(""); ax.grid(alpha=.25)
    plt.suptitle(""); plt.tight_layout(); plt.show()
else:
    print("No prompt_quality.csv found yet; run Cells 15 and 17 with --evaluate-prompt-quality.")
'''))

cells.append(code(r'''
# Cell 20 - qualitative predicted-vs-localization panel (polygon is diagnostic only)
def find_visual(output_dir, image_name):
    candidates = [output_dir / "overlays" / f"{Path(image_name).stem}_fused_layercam.png",
                  output_dir / "masks" / f"{Path(image_name).stem}.png",
                  output_dir / "debug" / Path(image_name).stem / "overlay_mask_0.png"]
    return next((p for p in candidates if p.exists()), None)

pred_visual = find_visual(PREDICTED_OUTPUT, DEBUG_IMAGE_NAME)
gt_visual = find_visual(GROUND_TRUTH_OUTPUT, DEBUG_IMAGE_NAME)
gt_image = Image.open(BTXRD_ROOT / "images" / DEBUG_IMAGE_NAME).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
axes[0].imshow(gt_image); axes[0].set_title("Original")
axes[1].imshow(gt_image); axes[1].imshow(trace_predicted["contrast"], cmap="jet", alpha=.45); axes[1].set_title("Predicted-class CAM")
if pred_visual: axes[2].imshow(Image.open(pred_visual)); axes[2].set_title("Predicted protocol output")
else: axes[2].text(.5, .5, "predicted visual missing", ha="center"); axes[2].set_title("Predicted protocol")
if gt_visual: axes[3].imshow(Image.open(gt_visual)); axes[3].set_title("GT-class diagnostic output")
else: axes[3].text(.5, .5, "ground_truth visual missing", ha="center"); axes[3].set_title("GT-class protocol")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
'''))

cells.append(markdown("""## 10. Optional supervised oracle baseline

This is not weak supervision and must not be reported as the pseudo-mask result. It answers a separate question:
whether the image resolution/model/training setup can segment BTXRD when polygons are available for training. Keep it
disabled for the WSSS experiment.
"""))

cells.append(code(r'''
# Cell 21 - optional polygon-supervised oracle baseline
SUPERVISED_ORACLE_OUTPUT = OUTPUT_ROOT / "supervised_unet_oracle"
supervised_cmd = [
    sys.executable, "train_segmentation.py", "--dataset", "btxrd", "--ram-root", str(BTXRD_ROOT),
    "--train-split", "train", "--val-split", "val", "--image-size", str(IMAGE_SIZE),
    "--num-workers", str(NUM_WORKERS), "--output-dir", str(SUPERVISED_ORACLE_OUTPUT),
]
if RUN_SUPERVISED_ORACLE_BASELINE:
    run_streaming(supervised_cmd)
else:
    print("RUN_SUPERVISED_ORACLE_BASELINE=False; no polygon labels enter WSSS generation.")
'''))

cells.append(code(r'''
# Cell 22 - reproducibility manifest and final audit
manifest = {
    "git_branch_requested": GIT_BRANCH,
    "dataset_root": str(BTXRD_ROOT),
    "classifier_checkpoint": str(CLASSIFIER_CHECKPOINT),
    "sam_checkpoint": str(SAM_CHECKPOINT),
    "profile": "btxrd_best",
    "image_size": IMAGE_SIZE,
    "sam_image_size": SAM_IMAGE_SIZE,
    "predicted_output": str(PREDICTED_OUTPUT),
    "ground_truth_output": str(GROUND_TRUTH_OUTPUT),
    "predicted_eval": str(PREDICTED_EVAL),
    "ground_truth_eval": str(GROUND_TRUTH_EVAL),
    "polygon_usage": "evaluation and oracle diagnostics only",
    "test_tuning": False,
}
(OUTPUT_ROOT / "notebook_artifacts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
display(pd.Series(manifest, name="value").to_frame())
print("Notebook debug run is complete when every requested artifact exists under", OUTPUT_ROOT)
'''))

cells.append(markdown("""## Interpretation checklist

- Report `predicted` as end-to-end inference and `ground_truth` only as localization/oracle diagnostics.
- Use conditional tumor Dice, end-to-end tumor Dice, normal empty-mask specificity, skipped count, and prompt-quality
  decomposition together; do not call macro-F1 a CAM metric.
- A low `foreground_recall`/high `support_loss_dice` indicates CAM or morphology failure. A good support but low
  `oracle_best_single_dice` indicates prompt/SAM candidate failure. A good oracle candidate but high
  `selection_loss_dice` indicates selection failure. A large `postprocess_delta_dice` indicates morphology failure.
- Polygon files are read only by evaluation/diagnostic cells and by the explicitly disabled supervised oracle baseline;
  no polygon-derived value is passed into CAM, prompts, candidate ranking, or WSSS post-processing.
- Keep the validation set for debugging and reserve the test split for one final locked report.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("btxrd_kaggle.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {Path('btxrd_kaggle.ipynb').resolve()} with {len(cells)} cells")
