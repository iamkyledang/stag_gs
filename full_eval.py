#
# stag_gs benchmark driver (python version of full_eval.sh):
# train -> render (test split) -> metrics, run on every scene under datasets/nerf_ds.
# HyperNeRF (interp) is wired up but disabled for now, matching full_eval.sh.
#
# All training hyperparameters (iterations, test_iterations, save_iterations, etc.) come from
# configs.json -- train.py/arguments.py already load it as the default for every flag we don't
# pass explicitly, so train_scenes*/train_scenes_all below must NOT override --iterations,
# --test_iterations or --save_iterations on the command line, or they'd silently win over configs.json.
#
# Usage:
#   python full_eval.py [--output_path ./output] [--iteration <iter>]
#   (--iteration defaults to configs.json's final save_iterations entry; only pass it to render/
#   evaluate a different checkpoint than the one configs.json trains to.)

import os
from argparse import ArgumentParser

from arguments import CONFIGS

_train_cfg = CONFIGS.get("train", {})
_opt_cfg = CONFIGS.get("optimization", {})
# the checkpoint training actually ends on, per configs.json (not a hardcoded guess)
_DEFAULT_ITERATION = _opt_cfg.get("iterations") or max(_train_cfg.get("save_iterations", [20000]))

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="./output")
parser.add_argument("--iteration", default=_DEFAULT_ITERATION, type=int)
args = parser.parse_args()

NERF_DS_ROOT = "datasets/nerf_ds"
HYPERNERF_ROOT = "datasets/hypernerf_interp"

# scene name -> path relative to the dataset root (the folder that holds dataset.json)
HYPERNERF_SCENES = {
    "aleks-teapot": "interp_aleks-teapot/aleks-teapot",
    "chickchicken": "interp_chickchicken/chickchicken",
    "cut-lemon": "interp_cut-lemon/cut-lemon1",
    "hand": "interp_hand/hand1-dense-v2",
    "slice-banana": "interp_slice-banana/slice-banana",
    "torchocolate": "interp_torchocolate/torchocolate",
}


def discover_scenes(root):
    """Every immediate subfolder of root (each one a scene dir containing dataset.json)."""
    return sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))


def run_pipeline(source, model):
    """train -> render -> metrics for a single scene, in that order, so a crash/early stop leaves
    complete, evaluated scenes behind instead of all-trained-but-nothing-evaluated."""
    # model_path is created by train.py's prepare_output_and_logger() before a single iteration
    # runs, so checking for the dir alone would wrongly "skip" a crashed/partial run; the final
    # point_cloud.ply (scene/__init__.py's Scene.save) only exists once training actually finished.
    final_ply = os.path.join(model, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if os.path.isfile(final_ply):
        print(f"Skipping training for {model}: {final_ply} already exists")
    else:
        # eval/load2gpu_on_the_fly/quiet/iterations/test_iterations/save_iterations intentionally
        # omitted: configs.json's "model"/"optimization"/"train" sections supply them. Passing any of
        # these store_true flags here would force them True on the CLI and silently override configs.json.
        os.system(f"python train.py -s {source} -m {model}")
    os.system(f"python render.py -s {source} -m {model} --iteration {args.iteration} --mode render --skip_train")
    os.system(f"python metrics.py -m \"{model}\"")


def full_eval_all(root, tag):
    for scene in discover_scenes(root):
        source = os.path.join(root, scene)
        model = os.path.join(args.output_path, tag, scene)
        run_pipeline(source, model)


def full_eval_scenes(root, tag, scenes):
    for scene, rel_path in scenes.items():
        source = os.path.join(root, rel_path)
        model = os.path.join(args.output_path, tag, scene)
        run_pipeline(source, model)


if __name__ == "__main__":
    # ---------------- NeRF-DS (all scenes under NERF_DS_ROOT) ----------------
    full_eval_all(NERF_DS_ROOT, "nerf_ds")

    # ---------------- HyperNeRF (interp) ----------------
    # disabled for now
    # full_eval_scenes(HYPERNERF_ROOT, "hypernerf", HYPERNERF_SCENES)
