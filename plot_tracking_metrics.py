#!/usr/bin/env python3
"""
plot_tracking_metrics_with_stats.py

Generate side‑by‑side box‑plots of tracking metrics and trace‑length (steps)
from one **or many** ``metrics_per_key.csv`` files produced by
``guided_tracking_evaluator.py`` and print/save the mean ± standard deviation
so you can copy‑paste them straight into your paper/tables.

NEW IN THIS VERSION (2025‑07‑03)
--------------------------------
* **Folder mode** – pass ``--runfolder`` that points to a directory containing
  *multiple* run‑version folders. The script walks each run‑version,
  recursively locates the *first* ``metrics_per_key.csv`` and concatenates
  them into a single DataFrame.
* Accepts *absolute* or *relative* ``--runfolder``. If relative, it is resolved
  below ``<root_dir>/results<runname>/``.
* Gracefully skips run‑versions that miss the CSV and warns the user.
* Adds a ``RunVersion`` column so you can facet/filter later if desired.
* Keeps the original single‑run behaviour when ``--runfolder`` is omitted.
* Fixes the path construction bug ("results" + runname → "results<runname>").

Usage examples
--------------
# 1) Legacy (single run‑version)
python plot_tracking_metrics_with_stats.py \
       --runname official_run2 --runversion 9_2025_06_26_21_56

# 2) *All* run‑versions under a given folder
python plot_tracking_metrics_with_stats.py \
       --runname official_run2 --runfolder good_runs

# 3) Absolute folder anywhere on disk
python plot_tracking_metrics_with_stats.py \
       --runfolder C:/experiments/sweep42

Output (per call)
-----------------
* ``plots/<runlabel>/metrics_<runlabel>.png`` – the plot
* ``plots/<runlabel>/summary_stats_<runlabel>.csv`` – tidy table with μ & σ

The *runlabel* is the folder name when using ``--runfolder`` or the
run‑version name when analysing a single run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ────────────────────────────────────────────────────────────────────────────
#  CLI ─ option parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot tracking metrics + steps box‑plots **and** print summary "
            "stats. Works on a *single* run‑version *or* a folder with many."
        )
    )

    p.add_argument(
        "--root_dir",
        default=r"C:\\Users\\P096347\\MSc Thesis\\abdominal_tracker_original",
        help="Project root containing results<runname>/ … (default: %(default)s)",
    )
    p.add_argument(
        "--runname",
        default="test_bidirectional",
        help="Name of the run, i.e. the X in results_X (default: %(default)s)",
    )

    p.add_argument(
        "--runfolder",
        default="",
        help=(
            "Optional: folder that holds *many* run‑versions. May be an absolute "
            "path or relative to <root_dir>/results<runname>/. If given, all "
            "CSV files found recursively below are aggregated."
        ),
    )

    p.add_argument(
        "--runversion",
        default="latest",
        help="Single run‑version to analyse (ignored when --runfolder is used).",
    )
    p.add_argument(
        "--csv",
        default="metrics_per_key.csv",
        help="Name of the evaluator CSV (default: %(default)s)",
    )
    p.add_argument(
        "--outdir",
        default=r"C:\\Users\\P096347\\MSc Thesis\\abdominal_tracker_original\\plots",
        help="Directory where PNG & stats CSV are written (default: %(default)s)",
    )
    p.add_argument(
        "--fontsize", type=int, default=13, help="Base font size for plots.",
    )

    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
#  Matplotlib / Seaborn globals
# ────────────────────────────────────────────────────────────────────────────

def apply_global_font_settings(base_fs: int) -> None:
    """Set matplotlib/Seaborn global font sizes so everything scales together."""

    plt.rcParams.update(
        {
            "font.size": base_fs,
            "axes.titlesize": base_fs + 2,
            "axes.labelsize": base_fs + 1,
            "legend.fontsize": base_fs - 1,
            "legend.title_fontsize": base_fs,
            "xtick.labelsize": base_fs - 1,
            "ytick.labelsize": base_fs - 1,
        }
    )


# ────────────────────────────────────────────────────────────────────────────
#  Data loading helpers
# ────────────────────────────────────────────────────────────────────────────

def _canonical_results_folder(root_dir: Path, runname: str) -> Path:
    """Return <root_dir>/results<runname> (creates no files)."""
    return root_dir / f"results{runname}"


def collect_csvs(args: argparse.Namespace) -> Tuple[pd.DataFrame, str]:
    """Return concatenated DF and a *runlabel* describing the data on disk."""

    base_results = _canonical_results_folder(Path(args.root_dir), args.runname)

    # ─── Folder mode – aggregate many run‑versions ────────────────────────
    if args.runfolder:
        runfolder = Path(args.runfolder)
        if not runfolder.is_absolute():
            runfolder = base_results / runfolder

        if not runfolder.is_dir():
            raise FileNotFoundError(f"❌  Folder {runfolder} does not exist")

        dfs: list[pd.DataFrame] = []
        for runversion in sorted(p for p in runfolder.iterdir() if p.is_dir()):
            try:
                csv_path = next(runversion.rglob(args.csv))
            except StopIteration:
                print(f"⚠️  {runversion.name}: no {args.csv} – skipped")
                continue

            print(f"↳  Reading {csv_path.relative_to(base_results.parent)}")
            df = pd.read_csv(csv_path)
            df["RunVersion"] = runversion.name  # keep provenance
            dfs.append(df)

        if not dfs:
            raise RuntimeError("❌  No CSVs were found – nothing to plot!")

        return pd.concat(dfs, ignore_index=True), runfolder.name

    # ─── Single run‑version (legacy) ───────────────────────────────────────
    if args.runversion == "latest":
        runs = [d for d in base_results.iterdir() if d.is_dir() and "_" in d.name]
        if not runs:
            raise RuntimeError(f"❌  No run‑versions found under {base_results}")
        latest_run = max(runs, key=lambda d: int(d.name.split("_", 1)[0]))
        args.runversion = latest_run.name

    csv_path = base_results / args.runversion / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"❌  {csv_path} not found")

    print(f"Reading single run {csv_path.relative_to(base_results.parent)}")
    df = pd.read_csv(csv_path)
    df["RunVersion"] = args.runversion

    return df, args.runversion


# ────────────────────────────────────────────────────────────────────────────
#  Statistics helper
# ────────────────────────────────────────────────────────────────────────────

def summarise_stats(metrics_long: pd.DataFrame, steps_df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy DataFrame with mean & std for each Metric / Dir (+ steps)."""

    by_metric_dir = (
        metrics_long.groupby(["Metric", "Dir"], sort=False)["Score"].agg(["mean", "std"]).reset_index()
    )

    overall = (
        metrics_long.groupby("Metric", sort=False)["Score"].agg(["mean", "std"]).reset_index().assign(Dir="All")
    )

    steps_stats = (
        steps_df.groupby("Dir", sort=False)["Steps"].agg(["mean", "std"]).reset_index().rename(columns={"Dir": "Metric"})
    )
    steps_stats["Metric"] = steps_stats["Metric"].str.capitalize() + " Steps"
    steps_stats.insert(1, "Dir", "Single‑dir")

    return pd.concat([by_metric_dir, overall, steps_stats], ignore_index=True, sort=False)


# ────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    apply_global_font_settings(args.fontsize)

    df, runlabel = collect_csvs(args)

    # ─── choose which metrics go where ────────────────────────────────────
    plot_metrics = ["Precision", "Recall", "F1", "Dice"]
    stats_metrics = plot_metrics + ["MSD"]  # MSD only in stats

    hue_order_metrics = ["Forward", "Reverse", "Reverse vs Forward"]
    hue_order_steps = ["Forward", "Reverse"]

    metrics_long = df.melt(
        id_vars=["Key", "Dir"],
        value_vars=stats_metrics,
        var_name="Metric",
        value_name="Score",
    )

    stats_df = summarise_stats(metrics_long, df[df["Dir"].isin(hue_order_steps)])

    # Optional: keep console / CSV ordered
    all_order = stats_metrics + ["Forward Steps", "Reverse Steps"]
    stats_df["Metric"] = pd.Categorical(stats_df["Metric"], categories=all_order, ordered=True)
    stats_df = stats_df.sort_values(["Metric", "Dir"])

    print("\n📊  Mean ± SD of tracking metrics and steps (μ ± σ):")
    for _, row in stats_df.iterrows():
        metric_dir = f" ({row.Dir})" if row.Dir not in {"All", "Single‑dir"} else ""
        print(f"  • {row.Metric}{metric_dir}: {row['mean']:.3f} ± {row['std']:.3f}")

    # ─── output paths ─────────────────────────────────────────────────────
    outdir = Path(args.outdir) / runlabel / args.runname
    outdir.mkdir(parents=True, exist_ok=True)

    stats_csv_path = outdir / f"summary_stats_{runlabel}.csv"
    plot_path = outdir / f"metrics_{runlabel}.png"

    stats_df.to_csv(stats_csv_path, index=False)

    # ─── figure ----------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [4, 1]},
        constrained_layout=True,
    )

    sns.boxplot(
        ax=axes[0],
        data=metrics_long,
        x="Metric",
        y="Score",
        order=plot_metrics,
        hue="Dir",
        hue_order=hue_order_metrics,
        width=0.55,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 7},
    )
    sns.stripplot(
        ax=axes[0],
        data=metrics_long,
        x="Metric",
        y="Score",
        order=plot_metrics,
        hue="Dir",
        hue_order=hue_order_metrics,
        dodge=True,
        jitter=True,
        linewidth=0,
        alpha=0.35,
        size=3,
        legend=False,
    )

    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Tracking Metrics – Forward, Reverse, Reverse vs Forward", pad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), title="Direction", loc="lower right", frameon=True)

    # Steps pane -----------------------------------------------------------
    steps_df = df[df["Dir"].isin(hue_order_steps)]

    sns.boxplot(
        ax=axes[1],
        data=steps_df,
        x="Dir",
        y="Steps",
        order=hue_order_steps,
        width=0.55,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 7},
    )

    axes[1].set_xlabel("Direction")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Trace Length", pad=10)
    if not steps_df["Steps"].empty:
        axes[1].set_ylim(0, steps_df["Steps"].max() * 1.05)

    if axes[1].legend_ is not None:
        axes[1].legend_.remove()

    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    print(f"\n✅  Saved plot to   {plot_path.resolve()}")
    print(f"✅  Saved stats to  {stats_csv_path.resolve()}")


if __name__ == "__main__":
    main()
