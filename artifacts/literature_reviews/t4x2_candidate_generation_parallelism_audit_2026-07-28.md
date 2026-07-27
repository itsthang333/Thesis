# T4x2 candidate-generation parallelism audit

Date: 2026-07-28  
Status: source/performance design audit only; no benchmark or new Kaggle job

## Current execution is not fully parallel

The version-6 wrapper invokes candidate generation with:

- classifier/LayerCAM on `cuda:0`;
- SAM on `cuda:1`;
- batch size 1;
- one `generate_pseudo_masks.py` process.

Inside the image loop, classifier/CAM computation finishes before prompt
construction and SAM inference for that image. The next image is not sent
through the classifier concurrently with the current SAM call. Assigning two
models to different T4s avoids memory contention and proves both devices are
usable, but it does not by itself establish concurrent throughput.

The later RAD-DINO cache stage does use `nn.DataParallel` with encoder batch
size 4, so each T4 receives a batch shard. PyTorch documents that DataParallel
splits inputs along the batch dimension and that batch size should exceed the
GPU count:
https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

No GPU-utilization trace is available from the compact v6 evidence yet.
Therefore this note does not fabricate a measured speedup or utilization.

## Static memory-feasibility lower bound

The dual-replica design is plausible on two T4s, but static model size is not
the same as peak runtime memory:

- NVIDIA specifies 16 GB GDDR6 per T4:
  https://www.nvidia.com/en-us/data-center/tesla-t4/
- The official TorchVision DenseNet-121 entry reports 7,978,856 parameters
  and a 30.8 MB pretrained weight file:
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html
- Meta's official SAM repository identifies the employed checkpoint as the
  ViT-B variant and links its immutable public checkpoint:
  https://github.com/facebookresearch/segment-anything
- A direct HTTP HEAD request to that official checkpoint URL on 2026-07-28
  reported `Content-Length: 375042383` bytes (357.67 MiB):
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

One classifier checkpoint plus one SAM ViT-B checkpoint is therefore well
below 1 GiB of serialized weights per replica. This is only a lower bound:
loaded tensors, CAM hooks, 1024-square activations, CUDA kernels/workspaces,
allocator fragmentation and image buffers can dominate peak memory. It does
not prove that a full classifier+SAM worker fits safely, much less establish
throughput. The 32-image benchmark must record
`torch.cuda.max_memory_allocated()` and `max_memory_reserved()` separately on
both GPUs. Promotion requires a predeclared 14 GiB peak-reserved ceiling per
T4, leaving at least 2 GiB for runtime variance; an out-of-memory retry with
scientific settings changed is forbidden.

## Fastest plausible T4x2 design for independent images

Candidate generation is embarrassingly parallel by image. The preferred
conditional design is two **independent spawned processes**, one per T4:

1. deterministically shard the sorted image IDs by stable index parity;
2. load a complete classifier plus SAM replica on each T4;
3. run the unchanged per-image pipeline independently on each shard;
4. write disjoint candidate/pseudo artifacts;
5. merge rows in the exact original sorted order;
6. verify every payload hash and require the four merged manifests to equal
   version 6 exactly.

This is not DDP training: the workers have no shared optimizer or gradients.
One process per GPU avoids Python-GIL serialization between independent GPU
pipelines. PyTorch's CUDA multiprocessing guidance requires `spawn` or
`forkserver`, not `fork`, and warns about CPU oversubscription:
https://docs.pytorch.org/docs/stable/notes/multiprocessing.html

Each worker must therefore:

- start via `spawn` before any parent CUDA initialization;
- set its own visible/indexed device;
- limit CPU/OpenCV/Torch threads to at most half the available CPU budget;
- use one data-loader worker initially, then benchmark two only if CPU is not
  saturated;
- never pass CUDA tensors through queues;
- write only its own shard paths.

## Required GT-blind benchmark

Do not make parallelism a scientific variable. Before full regeneration, use
a frozen 32-image clean-train subset selected by sorted image ID, containing
both image labels but no masks:

- Arm S: current single-process split-device execution;
- Arm P: two spawned one-GPU replicas over even/odd shards.

For both arms record:

- wall-clock time after model load and separately including model load;
- peak allocated/reserved memory on each T4;
- per-stage timings for classifier/CAM, CPU prompt construction and SAM;
- every physical candidate/pseudo payload hash;
- merged manifest bytes and hashes.

Promotion requires:

1. every per-image payload and merged manifest is byte-identical;
2. no fallback/count/provenance difference;
3. both T4s pass a real convolution and show successful nontrivial inference;
4. peak memory remains below a predeclared safe limit;
5. steady-state throughput improves by at least 30%.

If byte identity fails, or speedup is below 30%, retain the audited sequential
generator. Scientific correctness takes priority over a modest runtime gain.
The benchmark uses no validation GT and cannot select a model or metric.

## Merge implementation requirements

A future sharded generator must not concatenate arbitrary files. It must:

- preserve exact image-name sort order used by the original writer;
- reject duplicate/missing/unexpected image IDs;
- verify the full split counts `2981/371`;
- rebuild pseudo and candidate summaries with the unchanged schema;
- bind each candidate summary to the merged pseudo-manifest SHA-256;
- run `validate_candidate_diagnostics_manifest` on every merged physical NPZ;
- compare merged train/validation candidate and pseudo hashes with terminal v6;
- delete shard directories only after final compact/audit hashes exist.

## Decision

The correction-v3 wrapper remains unfinalized until v6 terminal evidence
arrives. If correction is authorized and candidate payloads must be
regenerated, first run the bounded GT-blind concurrency benchmark. Use the
dual-replica path only on exact equivalence plus material speedup. The
RAD-DINO encoder stage retains its existing two-T4 DataParallel execution;
selector training on cached descriptors is too small to benefit materially
from DDP.
