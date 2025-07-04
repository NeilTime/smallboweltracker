#!/usr/bin/env python3
"""
rl_data_prep.py ─ Collect CNN inputs & outputs for the RL tracker
===============================================================
This utility script pulls together exactly the inputs that the
future reinforcement‑learning (actor/critic) model will need.

Workflow (per sample)
---------------------
1.  **Patch extraction** – identical to the training procedure: pick a
    random centre‑line point and call `get_sample()` so we get the
    standard `(image patch, GT direction targets, GT seg patch)` triple.
2.  **CNN inference** – run the already‑trained
    *AbdominalTrackerPlusSegmentationModel* (or any compatible V‑Net
    variant).  We keep the network in `eval()` mode and never back‑prop
    here.
3.  **Packaging** – bundle the following tensors/arrays in a Python
    `dict` so they can be fed straight into an RL replay buffer:

    • `image_patch   ∶ [C, p, p, p]` – raw MRI patch  (float32)
    • `seg_probs     ∶ [C, p, p, p]` – CNN soft‑max probabilities  (float32)
    • `dir_logits    ∶ [S, nVerts]` – shell‑wise logits (pre‑softmax) (float32)
    • `center_mm     ∶ (3,)`         – patch centre in *mm* (float32)
    • `path_pred_mm  ∶ [T, 3]`       – stochastic tracker median trace (mm)
    • `path_gt_mm    ∶ [T, 3]`       – full annotated centre‑line (mm)

Usage
-----
```bash
python rl_data_prep.py \
    --checkpoint_dir checkpoints/AbdTrackSeg/ \
    --n_samples 128 \
    --save_dir   rl_buffer/train
```
All samples are written as *Torch* `.pt` files so they stay on GPU friendly.

The script is intentionally free of any RL‑specific logic – it is purely
about *data collection*.  You can import `collect_single_sample()` in a
trainer later on and choose whether you want on‑the‑fly sampling or a
pre‑generated replay buffer.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np

# Project‑internal imports ----------------------------------------------------
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from guided_tracking_evaluator import compute_stochastic_trace


def _build_test_options(checkpoint_dir: str) -> TestOptions:
    """Return a *TestOptions* instance configured for inference only."""
    opt = TestOptions().parse()  # grabs CLI flags / defaults

    # ─ inference‑safe overrides ────────────────────────────────────────────
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1  # disable visdom

    opt.phase = "test"
    opt.isTrain = False
    return opt


@torch.no_grad()
def collect_single_sample(model, dataset) -> Dict[str, Any]:
    """Grab one random patch, run it through the CNN, return a payload dict."""

    # 1️⃣  Random index – guarantees centre‑line sampling distribution
    idx = random.randrange(len(dataset))
    batch = dataset[idx]

    # 2️⃣  Add batch dim so the CNN is happy
    for key in ("A", "B", "C"):
        if key in batch:
            batch[key] = batch[key].unsqueeze(0)

    model.set_input(batch)
    model.forward()

    # 3️⃣  Gather outputs ----------------------------------------------------
    seg_probs = (
        torch.stack(model.fake_seg, dim=1)  # list → [B,C,p,p,p]
        .squeeze(0)
        .cpu()
    )
    dir_logits = model.fake_B.squeeze(0).cpu()  # [S, nVerts]

    # 4️⃣  Centre‑line paths -------------------------------------------------
    vol_id = int(batch["vol_id"])
    vol     = dataset.volumes[vol_id]

    # ground truth centre‑line (first key is fine – every sample belongs to *one* intestine)
    cl_key       = next(iter(vol.sint_segs_dense_vox.keys()))
    gt_path_vox  = vol.sint_segs_dense_vox[cl_key]
    path_gt_mm   = gt_path_vox * vol.spacing  # vox → mm  (simple diag affine)

    # predicted centre‑line from stochastic tracker (median trace)
    path_pred_mm, *_ = compute_stochastic_trace(
        model, vol, key=cl_key, gtdist_thres=False
    )

    return dict(
        image_patch=batch["A"].squeeze(0).cpu(),
        seg_probs=seg_probs,
        dir_logits=dir_logits,
        center_mm=batch["center_mm"].cpu(),
        vol_id=vol_id,
        path_pred_mm=np.asarray(path_pred_mm, dtype=np.float32),
        path_gt_mm=np.asarray(path_gt_mm, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect CNN inference results to feed an RL tracker.", add_help=True
    )
    parser.add_argument(
        "--pretrained_cnn_checkpoint",
        required=True,
        help="Folder that contains <experiment_name>/latest_net_G_A.pth",
    )
    parser.add_argument("--n_samples", type=int, default=1, help="how many samples to grab")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="If set, write each sample as *.pt into this directory",
    )
    # TODO: Switch to this one when everything is working
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    # ▓ initialise CNN & dataset ▓───────────────────────────────────────────
    opt = _build_test_options(args.pretrained_cnn_checkpoint)
    dataset = create_dataset(opt).dataset  # unwrap DataLoader → louisDataset

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ▓ sampling loop ▓─────────────────────────────────────────────────────
    outdir = Path(args.save_dir) if args.save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        sample = collect_single_sample(model, dataset)
        if outdir:
            fname = outdir / f"sample_{i:05d}.pt"
            torch.save(sample, fname)
            print(f"✔ saved → {fname}")
        else:
            print("sample keys:", list(sample.keys()))


if __name__ == "__main__":
    main()
