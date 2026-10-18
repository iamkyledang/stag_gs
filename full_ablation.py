#
# stag_gs anchor_stride ablation driver, NeRF-DS only, mask ENABLED (ignore_mask=false).
#
# Runs train -> render (test split) -> metrics for every (scene, anchor_stride) combination, where
# scenes = every folder under datasets/nerf_ds and anchor_stride values come from
# configs_ablation.json's "ablation.values" list (CLI --values overrides).
#
# WHY A SEPARATE configs_ablation.json: arguments/__init__.py hardcodes _CONFIGS_PATH to
# "<repo_root>/configs.json" -- train.py/render.py/metrics.py can only ever read that one file, they
# have no --config_path flag. So this script copies configs_ablation.json onto configs.json once at
# startup (after backing up whatever was already there), and restores the backup when the whole sweep
# finishes (or is interrupted) -- see _install_ablation_config/_restore_config. The actual
# anchor_stride value under test is NOT read from configs.json at all: it is passed explicitly as
# "--anchor_stride <value>" on train.py's command line, which always wins over the configs.json
# default (CLI flags override config-file values, see ParamGroup.__init__).
#
# ---------------------------------------------------------------------------------------------
# TIME BUDGET DERIVATION (target: 20-24h total on a single H100, 20GB VRAM) -- see also the repo
# memory note "configs_nerf_ds_without_mask.json retuned 2026-10-17 after real full_eval.py run"
# for the empirical numbers this is extrapolated from:
#   - Measured fact: anchor_stride=4, knn_k=48, coarse_levels=3, iterations=40000 took 4h23m for one
#     nerf_ds scene on this exact 20GB GPU (as_novel_view), and OOM'd on two heavier scenes before
#     finishing (N grew to 230-290k Gaussians while anchors were still refining towards stride 4).
#   - configs_ablation.json shrinks the budget heavily versus that measured run: iterations 40000->8000
#     (5x fewer), densify_until_iter 16000->6000 (so N stops growing much earlier and never reaches the
#     ranges that OOM'd), densify_grad_threshold 0.0006->0.001 (slower Gaussian growth), knn_k 48->32,
#     coarse_levels 3->2. This is a deliberately reduced-fidelity proxy budget for comparing the
#     RELATIVE effect of anchor_stride across scenes, not a publication-quality per-scene result.
#   - Rough per-run cost model (minutes), calibrated off the 4h23m/40000-iter/stride4/knn48 data point
#     scaled down by iteration count and by anchor density (M ~ 1/anchor_stride, and the dense
#     per-anchor tensors/fits are the main cost that scales with M on top of a roughly fixed
#     render+backward cost that does not depend on anchor_stride):
#       stride  4 (most anchors)  ~65 min/scene
#       stride  8 (baseline)      ~50 min/scene
#       stride 16                 ~43 min/scene
#       stride 32 (fewest anchors)~38 min/scene
#     Average ~49 min/scene-run x 7 nerf_ds scenes x 4 stride values = 28 runs ~ 22.9h total.
#   - These are ESTIMATES, not a benchmark of this exact reduced config. The script times every run
#     live and prints a running actual-vs-estimated comparison so you can recalibrate (e.g. lower
#     --values count, or edit configs_ablation.json's iterations) if real throughput differs.
#   - Recommended before committing to the full unattended sweep: smoke-test the single
#     heaviest combination first, e.g.
#       python full_ablation.py --scenes bell_novel_view --values 4
#     (bell_novel_view has the most training frames, 881, so anchor_stride=4 is both the slowest and
#     the most memory-hungry cell in the whole sweep -- if that one is safe/fast enough, the rest will
#     be too).
# ---------------------------------------------------------------------------------------------
#
# Usage:
#   python full_ablation.py                                   # full 7-scene x 4-stride sweep
#   python full_ablation.py --dry_run                         # just print the plan + time estimate
#   python full_ablation.py --values 4 8                      # only these anchor_stride values
#   python full_ablation.py --scenes bell_novel_view as_novel_view
#   python full_ablation.py --output_path ./output_ablation --no_restore
#

import os
import json
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser

NERF_DS_ROOT = "datasets/nerf_ds"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_JSON = os.path.join(REPO_ROOT, "configs.json")
CONFIGS_BACKUP = os.path.join(REPO_ROOT, "configs.json.ablation_backup")

# rough per-scene minute estimates used only for the printed plan/ETA, see the derivation above
_ESTIMATED_MINUTES = {4: 65, 8: 50, 16: 43, 32: 38}
_DEFAULT_ESTIMATE_MIN = 50


def discover_scenes(root):
    return sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))


def load_ablation_config(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def install_ablation_config(ablation_cfg, configs_path, backup_path):
    """Copy configs.json -> backup (only if no backup exists yet, so a crashed/interrupted sweep
    doesn't clobber the real original on a second invocation), then write the ablation hyperparameters
    (everything except the 'ablation'/'_comment' bookkeeping keys) onto configs.json."""
    if os.path.isfile(configs_path) and not os.path.isfile(backup_path):
        shutil.copy2(configs_path, backup_path)
        print(f"Backed up existing {configs_path} -> {backup_path}")
    elif os.path.isfile(backup_path):
        print(f"Backup {backup_path} already exists, leaving it as-is (assuming a prior interrupted ablation run)")

    to_write = {k: v for k, v in ablation_cfg.items() if k not in ("ablation", "_comment")}
    with open(configs_path, "w", encoding="utf-8") as f:
        json.dump(to_write, f, indent=2)
    print(f"Installed ablation hyperparameters onto {configs_path}")


def restore_config(configs_path, backup_path):
    if os.path.isfile(backup_path):
        shutil.move(backup_path, configs_path)
        print(f"Restored original {configs_path} from backup")
    else:
        print(f"No backup found at {backup_path}, leaving {configs_path} as the ablation config")


def run_pipeline(source, model, iteration, extra_train_args, env):
    """train -> render -> metrics for one (scene, anchor_stride) cell; skips any stage whose output
    already exists, so an interrupted multi-hour sweep can be resumed by just re-running the script."""
    final_ply = os.path.join(model, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if os.path.isfile(final_ply):
        print(f"  [skip train]  {final_ply} already exists")
    else:
        cmd = [sys.executable, "train.py", "-s", source, "-m", model] + extra_train_args
        print("  $", " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            print(f"  !! train.py failed (exit {result.returncode}) for {model}, skipping render/metrics for this cell")
            return False

    render_done = os.path.join(model, "test", f"ours_{iteration}", "comparison.mp4")
    if os.path.isfile(render_done):
        print(f"  [skip render] {render_done} already exists")
    else:
        cmd = [sys.executable, "render.py", "-s", source, "-m", model, "--iteration", str(iteration),
               "--mode", "render", "--skip_train"]
        print("  $", " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            print(f"  !! render.py failed (exit {result.returncode}) for {model}, skipping metrics for this cell")
            return False

    results_json = os.path.join(model, "results.json")
    if os.path.isfile(results_json):
        print(f"  [skip metrics] {results_json} already exists")
    else:
        cmd = [sys.executable, "metrics.py", "-m", model]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return True


def summarize(output_path, values, scenes, iteration):
    """Read every cell's results.json and write a per-stride mean PSNR/SSIM/LPIPS table."""
    summary = {}
    for value in values:
        metrics_by_metric = {"SSIM": [], "PSNR": [], "LPIPS": []}
        for scene in scenes:
            model = os.path.join(output_path, f"stride_{value}", scene)
            results_json = os.path.join(model, "results.json")
            if not os.path.isfile(results_json):
                continue
            with open(results_json, encoding="utf-8") as f:
                data = json.load(f)
            method_key = f"ours_{iteration}"
            scene_results = data.get(model, data.get(model.replace("\\", "/"), {}))
            method_results = scene_results.get(method_key)
            if not method_results:
                continue
            for metric_name in metrics_by_metric:
                if metric_name in method_results:
                    metrics_by_metric[metric_name].append(method_results[metric_name])
        summary[str(value)] = {
            metric_name: (sum(vals) / len(vals) if vals else None)
            for metric_name, vals in metrics_by_metric.items()
        }
        summary[str(value)]["num_scenes"] = len(metrics_by_metric["PSNR"])

    summary_path = os.path.join(output_path, "ablation_summary.json")
    os.makedirs(output_path, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Ablation summary (mean over scenes with available results) ===")
    print(f"{'anchor_stride':>14} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | scenes")
    for value in values:
        row = summary[str(value)]
        def fmt(x):
            return f"{x:8.4f}" if x is not None else f"{'--':>8}"
        print(f"{value:>14} | {fmt(row['PSNR'])} | {fmt(row['SSIM'])} | {fmt(row['LPIPS'])} | {row['num_scenes']}")
    print(f"Written to {summary_path}")


def main():
    parser = ArgumentParser(description="anchor_stride ablation over all NeRF-DS scenes (mask enabled)")
    parser.add_argument("--configs", default=os.path.join(REPO_ROOT, "configs_ablation.json"))
    parser.add_argument("--output_path", default="./output_ablation")
    parser.add_argument("--values", nargs="+", type=int, default=None,
                         help="anchor_stride values to sweep (default: configs_ablation.json's ablation.values)")
    parser.add_argument("--scenes", nargs="+", default=None,
                         help="subset of nerf_ds scene folder names (default: every scene under datasets/nerf_ds)")
    parser.add_argument("--iteration", type=int, default=None,
                         help="checkpoint iteration to render/evaluate (default: configs_ablation.json's final save_iterations entry)")
    parser.add_argument("--dry_run", action="store_true", help="print the plan and time estimate, run nothing")
    parser.add_argument("--no_restore", action="store_true",
                         help="leave configs.json set to the ablation config after finishing instead of restoring the backup")
    args = parser.parse_args()

    ablation_cfg = load_ablation_config(args.configs)
    values = args.values or ablation_cfg["ablation"]["values"]
    scenes = args.scenes or discover_scenes(os.path.join(REPO_ROOT, NERF_DS_ROOT))
    iteration = args.iteration or ablation_cfg["optimization"]["iterations"]

    total_minutes = sum(_ESTIMATED_MINUTES.get(v, _DEFAULT_ESTIMATE_MIN) for v in values) * len(scenes)
    print(f"Plan: {len(scenes)} scenes x {len(values)} anchor_stride values = {len(scenes) * len(values)} runs")
    print(f"Scenes: {scenes}")
    print(f"anchor_stride values: {values}")
    print(f"Estimated total time: ~{total_minutes / 60:.1f}h (see script docstring for the derivation; "
          f"actual timing is tracked live below once runs start)")

    if args.dry_run:
        return

    install_ablation_config(ablation_cfg, CONFIGS_JSON, CONFIGS_BACKUP)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sweep_start = time.time()
    completed_runs = 0
    total_runs = len(scenes) * len(values)
    try:
        for value in values:
            for scene in scenes:
                source = os.path.join(NERF_DS_ROOT, scene)
                model = os.path.join(args.output_path, f"stride_{value}", scene)
                print(f"\n--- anchor_stride={value}  scene={scene}  ({completed_runs + 1}/{total_runs}) ---")
                run_start = time.time()
                run_pipeline(source, model, iteration, ["--anchor_stride", str(value)], env)
                run_elapsed = time.time() - run_start
                completed_runs += 1
                elapsed = time.time() - sweep_start
                avg_per_run = elapsed / completed_runs
                remaining = avg_per_run * (total_runs - completed_runs)
                print(f"  run took {run_elapsed / 60:.1f} min | elapsed {elapsed / 3600:.2f}h | "
                      f"avg {avg_per_run / 60:.1f} min/run | projected remaining {remaining / 3600:.2f}h")
    finally:
        summarize(args.output_path, values, scenes, iteration)
        if args.no_restore:
            print(f"--no_restore set: leaving {CONFIGS_JSON} as the ablation config "
                  f"(backup of your original is at {CONFIGS_BACKUP})")
        else:
            restore_config(CONFIGS_JSON, CONFIGS_BACKUP)


if __name__ == "__main__":
    main()
