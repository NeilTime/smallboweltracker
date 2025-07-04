#!/usr/bin/env python3
"""run_rl_tracker.py – Helper launcher for RL‑enhanced bowel tracker
===================================================================

Mimics the original *run_louis.py* UX but targets the upcoming
reinforcement‑learning (actor‑critic) tracker that is trained *on top of*
our frozen V‑Net segmenter/heading‑predictor.

Usage
-----
$ python run_rl_tracker.py             # interactive menu
$ python run_rl_tracker.py train debug # non‑interactive quick‑start

Notes
-----
*   **Training** calls *train_rl.py* – expected to live next to this
    script. Adjust paths as needed.
*   **Inference** calls *rl_tracker_evaluator.py* – analogous to
    *guided_tracking_evaluator.py* but RL‑aware.
*   TensorBoard is offered instead of Visdom (port 6006).
*   Defaults mirror the hyper‑parameters in the thesis (replace if you
    settle on different ones).
"""

from __future__ import annotations

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATAROOT = r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility"
CHECKPOINTS_DIR = THIS_DIR / "rl_checkpoints"

###############################################################################
# Helpers
###############################################################################

def start_tensorboard(logdir: str | os.PathLike, port: int = 6006) -> None:
    """Spawn TensorBoard (non‑blocking) and optionally open browser."""
    print(f"Starting TensorBoard on port {port} (logdir={logdir})…")
    tb_log = open("tensorboard.log", "w")
    subprocess.Popen([
        sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--port", str(port)
    ], stdout=tb_log, stderr=subprocess.STDOUT)

    if input("Open TensorBoard in browser? (y/n): ").strip().lower() == "y":
        time.sleep(5)  # Give TB a moment to spin up
        webbrowser.open(f"http://localhost:{port}")

###############################################################################
# Training
###############################################################################

def run_training(debug: bool = False) -> None:
    print("Launching RL training…")

    cmd: list[str] = [sys.executable, "reinforcement_tracker.py"]

    # ---------------------------------------------------------------------
    # DEBUG / quick iteration settings
    # ---------------------------------------------------------------------
    if debug:
        cmd += [
            "--name", "testrun",
            "--dataroot", DEFAULT_DATAROOT,
            "--dataset_mode", "louis",
            "--patch_size", "32",
            "--pretrained_cnn_checkpoint", str(THIS_DIR / "checkpoints" / "testrun" / "latest_net_G.pth"),
            "--masterpath", "MOT3D_multi_tslice_MII##a.hdf5",
            "--model", "abdominal_tracker_plus_segmentation",
            "--checkpoints_dir", "checkpoints",
            "--seg_path", "masked_segmentations/MII##a_seg_t2_5c.nii",
            "--output_nc", "5",
            # PPO‑ish hyper‑params (tiny):
            # "--algo", "ppo",
            # "--num_updates", "10",
            # "--env_steps", "1024",
            # "--num_envs", "4",
            # "--lr", "3e-4",
            # "--gamma", "0.99",
            # "--gae_lambda", "0.95",
            # "--clip_param", "0.2",
            # "--entropy_coef", "0.01",
            # "--value_loss_coef", "0.5",
            # "--max_grad_norm", "0.5",
            # "--save_interval", "1",
            # "--log_interval", "1",
            "--checkpoints_dir", "checkpoints",
            "--save_dir", str(CHECKPOINTS_DIR / "debug"),
            "--trainvols", "5,7,10,13,14,15,16,18,19,23,25,26",
            "--validationvol", "22",
        ]

    # ---------------------------------------------------------------------
    # Full experiment (interactive)
    # ---------------------------------------------------------------------
    else:
        runname = input("Enter run name (e.g., rl_run1): ").strip()
        cnn_chk = input("Path to pretrained CNN checkpoint (leave blank for default): ").strip()
        if not cnn_chk:
            cnn_chk = THIS_DIR / "checkpoints" / runname / "latest_net_G.pth"

        cmd += [
            "--name", runname,
            "--dataroot", DEFAULT_DATAROOT,
            "--dataset_mode", "louis",
            "--patch_size", "32",
            "--pretrained_cnn_checkpoint", str(cnn_chk),
            "--algo", "ppo",
            "--num_updates", "40000",
            "--env_steps", "8192",
            "--num_envs", "8",
            "--lr", "3e-4",
            "--gamma", "0.99",
            "--gae_lambda", "0.95",
            "--clip_param", "0.1",
            "--entropy_coef", "0.001",
            "--value_loss_coef", "0.5",
            "--max_grad_norm", "0.5",
            "--save_interval", "50",
            "--log_interval", "10",
            "--checkpoints_dir", str(CHECKPOINTS_DIR),
        ]

    # Spawn TensorBoard?
    if input("Start TensorBoard? (y/n): ").strip().lower() == "y":
        start_tensorboard(CHECKPOINTS_DIR)

    print("Command to run:")
    print(" ".join(map(str, cmd)))

    result = subprocess.run(cmd)
    print("Training script finished with code:", result.returncode)

###############################################################################
# Inference
###############################################################################

def run_inference() -> None:
    print("Launching RL tracker inference…")

    runname = input("Enter run name (e.g., rl_run1): ").strip()
    vnum = input("Enter volume number (e.g., 14): ").strip()
    timepoint = input("Enter timepoint (e.g., 3): ").strip()

    cmd: list[str] = [
        sys.executable, "rl_tracker_evaluator.py",
        "--checkpoints_dir", str(CHECKPOINTS_DIR),
        "--dataroot", DEFAULT_DATAROOT,
        "--dataset_mode", "louis",
        "--name", runname,
        "--volume_number", vnum,
        "--tslice_start", timepoint,
        "--patch_size", "32",
        # --- Evaluation knobs ------------------------------------------------
        "--n_traces", "64",
        "--step_size", "2.5",
        "--start_randomness_mm", "2.5",
        "--confidence_thres", "0.015",
        "--results_dir", "results_rl",
    ]

    print("Command to run:")
    print(" ".join(map(str, cmd)))
    result = subprocess.run(cmd)
    print("Inference script finished with code:", result.returncode)

###############################################################################
# Entry point
###############################################################################

def main() -> None:
    # Non‑interactive shortcut: e.g. python run_rl_tracker.py train debug
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
        if mode == "train":
            run_training(debug=(len(sys.argv) >= 3 and sys.argv[2] == "debug"))
            return
        elif mode == "infer":
            run_inference()
            return

    print("Choose mode: (1) Train or (2) Inference")
    choice = input("Enter 1 for training, 2 for inference: ").strip()

    if choice == "1":
        debug_mode = input("Enable debug mode? (y/n): ").strip().lower() == "y"
        run_training(debug=debug_mode)
    elif choice == "2":
        run_inference()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
