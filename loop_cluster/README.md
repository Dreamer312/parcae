# loop_cluster

Phase 0 of the "clustering in looped LMs" project. Nothing in `parcae_lm/` is changed
on this branch; these are additions only.

| File | What it is |
| --- | --- |
| `diagnose_loop_clusters.py` | Inference-only diagnostic on a Parcae checkpoint: how the token states cluster across loops, and how much loss is lost when prefix tokens are merged into cluster means. |
| `k8s/parcae-baseline-job.yaml` | Kubernetes Job for one pretraining run on a single 8-GPU pod, for a cluster without internet. |
| `docker/Dockerfile.offline` | Image with every training dependency preinstalled on a PyTorch 2.5 base. Not yet built. |

## Offline setup

The cluster has no internet, so three things are fetched elsewhere and copied over:

```bash
# 1. image: build where PyPI is reachable, push to the internal registry
docker build -f loop_cluster/docker/Dockerfile.offline --build-arg BASE_IMAGE=<pytorch 2.5 image> -t <registry>/parcae-offline:torch2.5 .
# 2. tokenizer
huggingface-cli download SandyResearch/parcae-tokenizer --local-dir /resource/tokenizers/parcae-tokenizer
# 3. released checkpoint, for the diagnostic and for comparing validation loss
huggingface-cli download SandyResearch/parcae-140m --local-dir /resource/models/parcae-140m
```

Then set `tokenizer_path` in the launch config to `/resource/tokenizers/parcae-tokenizer`,
and pass local directories to the diagnostic:
`--model /resource/models/parcae-140m --tokenizer /resource/tokenizers/parcae-tokenizer`.

No flash-attn install is needed; see the header of `docker/Dockerfile.offline`.

**Status: the diagnostic script has only been syntax-checked. It has never been run.**
Expect to fix small things on the first run.

## Data

Parcae trains on `SandyResearch/fineweb-edu-shuffled`, not `HuggingFaceFW/fineweb_edu_100BT`.
The configs read `/resource/data/fineweb-edu-shuffled/data/sample-100BT/train` and
`.../data/val/train`.

```bash
huggingface-cli download SandyResearch/fineweb-edu-shuffled --repo-type dataset \
  --include "data/sample-100BT/*" "data/val/*" \
  --local-dir /resource/data/fineweb-edu-shuffled
```

The diagnostic only needs `data/val` (7.3 GB).

## Diagnostic

```bash
pip install -q -e .
python loop_cluster/diagnose_loop_clusters.py \
  --model SandyResearch/parcae-140m \
  --val_dir /resource/data/fineweb-edu-shuffled/data/val/train \
  --out diag-140m.json
```

One GPU is enough. Repeat with `SandyResearch/parcae-370m`, `parcae-770m` and
`parcae-1.3b` to see how the numbers move with model size.

Background, decisions so far and next steps: see `HANDOFF.md`.

How to read the output:

- **Part A**: `eff_rank` falling and `mean_cos` rising over loops means the states are
  collapsing toward fewer directions. `ev_kmeans` well above `ev_contiguous` and
  `ev_random` means there is real cluster structure that fixed chunking does not capture.
- **Part A2** (`adjacent`): share of neighbouring tokens in the same k-means group.
  Contiguous chunks of ratio `r` give `1 - 1/r`. Close to that: clusters are spans, and
  chunk pooling already covers them. Far below: clusters are scattered across the sequence.
- **Part B**: loss increase on suffix tokens after merging the prefix. The go signal is
  k-means merging at 4x costing clearly less than contiguous and random merging.

## Baseline training

Fill in the four `CHANGE-ME` values in `k8s/parcae-baseline-job.yaml` (namespace, image,
PVC, GPU resource name), then `kubectl apply -f` it. It mounts the PVC at `/resource` and
expects the repo at `/resource/parcae`, the data under `/resource/data` and the tokenizer
under `/resource/tokenizers`; nothing is downloaded at start-up. Logging is W&B offline.
