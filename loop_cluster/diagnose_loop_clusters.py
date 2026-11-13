#!/usr/bin/env python
"""Phase 0 diagnostic: do looped LMs cluster their token states, and how much
does merging tokens into cluster centroids cost?

Inference only. Works on Parcae checkpoints (github.com/SandyResearch/parcae).
Needs the repo installed first: `pip install -e .` from the repo root.

Part A - geometry of the recurrent state after every loop (prefix tokens only):
  eff_rank    participation ratio of the centered token states
  mean_cos    mean pairwise cosine similarity between token states
  rel_change  ||x_s - x_{s-1}|| / ||x_s||, how close the loop is to a fixed point
  ev_*        variance explained when tokens are replaced by group means, for
              k-means groups, contiguous chunks and random groups
  nmi         agreement of the k-means partition between consecutive loops,
              next to the same score against a shuffled partition (the floor)
  adjacent    share of neighbouring tokens in the same k-means group: high means
              the natural groups are contiguous spans, low means they are
              scattered across the sequence

Part B - merge and keep looping:
  After loop `s`, every PREFIX token state is replaced by the mean of its group
  and the remaining loops run as usual. Loss is measured on SUFFIX tokens only,
  so no future information can leak into a prediction. "persist" re-applies the
  same grouping after every later loop, which is closer to really looping on K
  concept states.

  Note for reading the numbers: Parcae re-injects the token-level prelude output
  at every loop, so merged states get their token identity back for free.

Example:
  python loop_cluster/diagnose_loop_clusters.py --model SandyResearch/parcae-140m \
      --val_dir /resource/data/fineweb-edu-shuffled/data/val/train --out diag-140m.json
"""
import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_tokens(val_dir, tok, n_tokens, max_files):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(val_dir, "**", "*.parquet"), recursive=True))[:max_files]
    if not files:
        raise SystemExit(f"No parquet files under {val_dir}")
    bos = getattr(tok, "bos_id", None)
    ids = []
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not text:
                    continue
                ids.extend(tok.encode(text, prepend=bos, return_tensors=False))
                if len(ids) >= n_tokens:
                    return ids[:n_tokens]
    raise SystemExit(f"Only {len(ids)} tokens in {len(files)} files, need {n_tokens}. Raise --max_files.")


# ----------------------------------------------------------------------------
# Grouping helpers. x: [B, P, D] float32, a: [B, P] long group ids in [0, K)
# ----------------------------------------------------------------------------
def group_mean(x, a, K):
    """Replace every token by the mean of its group."""
    B, P, D = x.shape
    xf = x.float().reshape(B * P, D)
    idx = (a + torch.arange(B, device=x.device)[:, None] * K).reshape(-1)
    sums = torch.zeros(B * K, D, device=x.device).index_add_(0, idx, xf)
    cnt = torch.zeros(B * K, device=x.device).index_add_(0, idx, torch.ones(B * P, device=x.device))
    cent = sums / cnt.clamp_min(1)[:, None]
    return cent[idx].reshape(B, P, D).to(x.dtype)


def explained_variance(x, a, K):
    x = x.float()
    resid = (x - group_mean(x, a, K)).pow(2).sum((1, 2))
    total = (x - x.mean(1, keepdim=True)).pow(2).sum((1, 2))
    return (1 - resid / total.clamp_min(1e-12)).mean().item()


def kmeans_one(x, K, iters, init):
    """x: [P, D] float32. init: [K] token indices used as the first centroids."""
    c = x[init].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=K).to(x.dtype)
        keep = cnt > 0
        c[keep] = sums[keep] / cnt[keep, None]
    return torch.cdist(x, c).argmin(1)


def assign_kmeans(x, K, iters, seed):
    """Same initial token indices for a given seed, so partitions are comparable across loops."""
    B, P, _ = x.shape
    gen = torch.Generator().manual_seed(seed)
    out = []
    for b in range(B):
        init = torch.randperm(P, generator=gen)[:K].to(x.device)
        out.append(kmeans_one(x[b].float(), K, iters, init))
    return torch.stack(out)


def assign_contiguous(x, K, ratio):
    B, P, _ = x.shape
    return (torch.arange(P, device=x.device) // ratio).clamp_max(K - 1).expand(B, P)


def assign_random(x, K, seed):
    B, P, _ = x.shape
    gen = torch.Generator().manual_seed(seed + 7919)
    return torch.stack([torch.randperm(P, generator=gen) % K for _ in range(B)]).to(x.device)


def nmi(a, b, K):
    n = a.numel()
    joint = torch.bincount(a * K + b, minlength=K * K).view(K, K).double() / n
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * (joint[nz].log() - (pa[:, None] * pb[None, :])[nz].log())).sum()
    ha = -(pa[pa > 0] * pa[pa > 0].log()).sum()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum()
    return (2 * mi / (ha + hb)).item() if (ha + hb) > 0 else 1.0


def geometry(x):
    x = x.float()
    s = torch.linalg.svdvals(x - x.mean(1, keepdim=True))
    lam = s.pow(2)
    eff_rank = lam.sum(1).pow(2) / lam.pow(2).sum(1).clamp_min(1e-12)
    u = F.normalize(x, dim=-1)
    P = x.shape[1]
    mean_cos = (u.sum(1).pow(2).sum(-1) - P) / (P * (P - 1))
    return eff_rank.mean().item(), mean_cos.mean().item()


# ----------------------------------------------------------------------------
# Hook into the loop. Parcae calls self.update_recurrent_state once per loop.
# ----------------------------------------------------------------------------
class LoopTap:
    def __init__(self, model):
        if not hasattr(model, "update_recurrent_state"):
            raise SystemExit("This model has no update_recurrent_state; the script supports Parcae models only.")
        self.orig = model.update_recurrent_state
        self.reset()
        model.update_recurrent_state = self

    def reset(self, record=False, edit=None):
        self.record, self.edit, self.states = record, edit, []

    def __call__(self, x, input_embeds, freqs_cis, mask, step, total_steps, **kw):
        out = self.orig(x, input_embeds, freqs_cis, mask, step, total_steps, **kw)
        done = int(step) + 1  # loops finished so far
        if self.record:
            self.states.append(out.detach().float())
        if self.edit is not None:
            out = self.edit(done, out)
        return out


def make_merge_edit(assign_fn, K, merge_step, persist, P):
    cache = {}

    def edit(done, x):
        if done < merge_step or (done > merge_step and not persist):
            return x
        if "a" not in cache:
            cache["a"] = assign_fn(x[:, :P])
        return torch.cat([group_mean(x[:, :P], cache["a"], K), x[:, P:]], dim=1)

    return edit


def suffix_loss(model, ids, R, P, seed, amp):
    """Sum of next-token losses at positions P..T-2 (targets are suffix tokens only)."""
    torch.manual_seed(seed)  # the recurrent state starts from noise; keep it identical across conditions
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out = model(ids, return_logits=True, num_steps_pair=torch.tensor([R, 0]))
    logits = out["logits"][:, P:-1].float()
    tgt = ids[:, P + 1 :]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
    return loss.item(), tgt.numel()


class Mean:
    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, key, value, weight=1.0):
        self.s[key] = self.s.get(key, 0.0) + value * weight
        self.n[key] = self.n.get(key, 0.0) + weight

    def get(self, key):
        return self.s[key] / self.n[key]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="SandyResearch/parcae-140m")
    ap.add_argument("--tokenizer", default="SandyResearch/parcae-tokenizer")
    ap.add_argument("--val_dir", required=True, help="directory with parquet files that have a 'text' column")
    ap.add_argument("--max_files", type=int, default=4)
    ap.add_argument("--n_seqs", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--prefix_len", type=int, default=1536, help="tokens that may be merged; loss is on the rest")
    ap.add_argument("--recurrence", type=int, default=0, help="loops at inference; 0 = the model's mean_recurrence")
    ap.add_argument("--merge_steps", default="2,4,6", help="merge after this many loops")
    ap.add_argument("--ratios", default="2,4,8,16", help="compression ratios: tokens per group")
    ap.add_argument("--stability_ratio", type=int, default=4, help="ratio used for the partition-stability score")
    ap.add_argument("--kmeans_iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="loop_cluster_diag.json")
    args = ap.parse_args()

    import parcae_lm
    from parcae_lm.tokenizer import Tokenizer

    device = torch.device(args.device)
    amp = device.type == "cuda"
    model = parcae_lm.from_pretrained(args.model, device=device)
    tok = Tokenizer.from_pretrained(args.tokenizer)
    tap = LoopTap(model)

    R = args.recurrence or model.config.mean_recurrence
    T, P = args.seq_len, args.prefix_len
    assert 0 < P < T - 1 and T <= model.config.block_size
    ratios = [int(r) for r in args.ratios.split(",")]
    merge_steps = [int(s) for s in args.merge_steps.split(",") if int(s) <= R]

    ids_all = torch.tensor(load_tokens(args.val_dir, tok, args.n_seqs * T, args.max_files), dtype=torch.long)
    ids_all = ids_all.view(args.n_seqs, T)

    geo, merge = Mean(), Mean()
    with torch.no_grad():
        for bi in range(0, args.n_seqs, args.batch_size):
            ids = ids_all[bi : bi + args.batch_size].to(device)
            seed = args.seed + bi

            # Baseline pass: record the state after every loop.
            tap.reset(record=True)
            loss, n = suffix_loss(model, ids, R, P, seed, amp)
            merge.add("baseline", loss / n, n)
            states = [s[:, :P] for s in tap.states]
            assert len(states) == R, f"expected {R} loops, saw {len(states)}"

            prev_x, prev_a = None, None
            Ks = args.stability_ratio and P // args.stability_ratio
            for step, x in enumerate(states, start=1):
                er, mc = geometry(x)
                geo.add(("eff_rank", step), er)
                geo.add(("mean_cos", step), mc)
                if prev_x is not None:
                    rel = (x - prev_x).norm(dim=(1, 2)) / x.norm(dim=(1, 2)).clamp_min(1e-12)
                    geo.add(("rel_change", step), rel.mean().item())
                for r in ratios:
                    K = P // r
                    a_km = assign_kmeans(x, K, args.kmeans_iters, seed)
                    geo.add(("ev_kmeans", step, r), explained_variance(x, a_km, K))
                    geo.add(("adjacent", step, r), (a_km[:, 1:] == a_km[:, :-1]).float().mean().item())
                    geo.add(("ev_contiguous", step, r), explained_variance(x, assign_contiguous(x, K, r), K))
                    geo.add(("ev_random", step, r), explained_variance(x, assign_random(x, K, seed), K))
                a = assign_kmeans(x, Ks, args.kmeans_iters, seed)
                if prev_a is not None:
                    for b in range(a.shape[0]):
                        geo.add(("nmi", step), nmi(prev_a[b], a[b], Ks))
                        shuffled = a[b][torch.randperm(P, device=a.device)]
                        geo.add(("nmi_floor", step), nmi(prev_a[b], shuffled, Ks))
                prev_x, prev_a = x, a
            del states

            # Merge passes.
            for s in merge_steps:
                for r in ratios:
                    K = P // r
                    methods = {
                        "kmeans": lambda xp, K=K: assign_kmeans(xp, K, args.kmeans_iters, seed),
                        "contiguous": lambda xp, K=K, r=r: assign_contiguous(xp, K, r),
                        "random": lambda xp, K=K: assign_random(xp, K, seed),
                    }
                    for name, fn in methods.items():
                        for persist in (False, True):
                            tap.reset(edit=make_merge_edit(fn, K, s, persist, P))
                            loss, n = suffix_loss(model, ids, R, P, seed, amp)
                            merge.add((name, s, r, persist), loss / n, n)
            tap.reset()
            print(f"[{min(bi + args.batch_size, args.n_seqs)}/{args.n_seqs}] sequences done", flush=True)

    # ---- report -------------------------------------------------------------
    base = merge.get("baseline")
    print(f"\nmodel={args.model}  loops={R}  seq={T}  prefix={P}  suffix-loss baseline={base:.4f} nats")

    print("\nPart A: state geometry per loop (prefix tokens)")
    head = "loop  eff_rank  mean_cos  rel_change  nmi(floor)   " + "  ".join(f"EV@{r}x km/ctg/rnd" for r in ratios)
    print(head)
    for step in range(1, R + 1):
        rel = f"{geo.get(('rel_change', step)):.3f}" if step > 1 else "  -  "
        st = f"{geo.get(('nmi', step)):.2f}({geo.get(('nmi_floor', step)):.2f})" if step > 1 else "    -     "
        evs = "  ".join(
            f"{geo.get(('ev_kmeans', step, r)):.2f}/{geo.get(('ev_contiguous', step, r)):.2f}/{geo.get(('ev_random', step, r)):.2f}"
            for r in ratios
        )
        print(f"{step:>4}  {geo.get(('eff_rank', step)):>8.1f}  {geo.get(('mean_cos', step)):>8.3f}  {rel:>10}  {st:>10}   {evs}")

    print("\nPart A2: share of neighbouring tokens that fall in the same k-means group")
    print("  (contiguous chunks would give " + ", ".join(f"{1 - 1 / r:.2f}@{r}x" for r in ratios) + "; position-blind groups give about 0)")
    for step in range(1, R + 1):
        print(f"{step:>4}  " + "  ".join(f"{geo.get(('adjacent', step, r)):.2f}@{r}x" for r in ratios))

    print("\nPart B: suffix-loss increase (nats) after merging prefix tokens; kmeans / contiguous / random")
    for persist in (False, True):
        print(f"  {'merge after every later loop (persist)' if persist else 'merge once'}")
        print("    after loop  " + "  ".join(f"{str(r) + 'x':>22}" for r in ratios))
        for s in merge_steps:
            cells = []
            for r in ratios:
                d = [merge.get((m, s, r, persist)) - base for m in ("kmeans", "contiguous", "random")]
                cells.append(f"{d[0]:+.3f}/{d[1]:+.3f}/{d[2]:+.3f}".rjust(22))
            print(f"    {s:>10}  " + "  ".join(cells))

    result = {
        "args": vars(args),
        "loops": R,
        "baseline_suffix_loss": base,
        "geometry": {"|".join(map(str, k)): geo.get(k) for k in geo.s},
        "merge_loss": {"|".join(map(str, k)): merge.get(k) for k in merge.s if k != "baseline"},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
