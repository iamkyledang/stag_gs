#
# stag_gs benchmark driver (python version of full_eval.sh):
# train -> render (test split) -> metrics, run on every scene under datasets/nerf_ds.
# HyperNeRF (interp) is wired up but disabled for now, matching full_eval.sh.
#
# Usage:
#   python full_eval.py [--output_path ./output] [--iteration 20000]

import os
from argparse import ArgumentParser

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="./output")
parser.add_argument("--iteration", default=20000, type=int)
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


def train_scenes_all(root, tag):
    for scene in discover_scenes(root):
        source = os.path.join(root, scene)
        model = os.path.join(args.output_path, tag, scene)
        if os.path.isdir(model):
            print(f"Skipping training for {scene}: {model} already exists")
            continue
        os.system(
            f"python train.py -s {source} -m {model} --eval --load2gpu_on_the_fly --quiet "
            f"--iterations {args.iteration} --test_iterations {args.iteration} --save_iterations {args.iteration}"
        )


def render_scenes_all(root, tag):
    for scene in discover_scenes(root):
        source = os.path.join(root, scene)
        model = os.path.join(args.output_path, tag, scene)
        os.system(
            f"python render.py -s {source} -m {model} --iteration {args.iteration} --mode render --skip_train --quiet"
        )


def metrics_for_all(root, tag):
    models = " ".join(f"\"{os.path.join(args.output_path, tag, scene)}\"" for scene in discover_scenes(root))
    os.system(f"python metrics.py -m {models}")


def train_scenes(root, tag, scenes):
    for scene, rel_path in scenes.items():
        source = os.path.join(root, rel_path)
        model = os.path.join(args.output_path, tag, scene)
        if os.path.isdir(model):
            print(f"Skipping training for {scene}: {model} already exists")
            continue
        os.system(
            f"python train.py -s {source} -m {model} --eval --load2gpu_on_the_fly --quiet "
            f"--iterations {args.iteration} --test_iterations {args.iteration} --save_iterations {args.iteration}"
        )


def render_scenes(root, tag, scenes):
    for scene, rel_path in scenes.items():
        source = os.path.join(root, rel_path)
        model = os.path.join(args.output_path, tag, scene)
        os.system(
            f"python render.py -s {source} -m {model} --iteration {args.iteration} --mode render --skip_train --quiet"
        )


def metrics_for(tag, scenes):
    models = " ".join(f"\"{os.path.join(args.output_path, tag, scene)}\"" for scene in scenes)
    os.system(f"python metrics.py -m {models}")


if __name__ == "__main__":
    # ---------------- NeRF-DS (all scenes under NERF_DS_ROOT) ----------------
    train_scenes_all(NERF_DS_ROOT, "nerf_ds")
    render_scenes_all(NERF_DS_ROOT, "nerf_ds")
    metrics_for_all(NERF_DS_ROOT, "nerf_ds")

    # ---------------- HyperNeRF (interp) ----------------
    # disabled for now
    # train_scenes(HYPERNERF_ROOT, "hypernerf", HYPERNERF_SCENES)
    # render_scenes(HYPERNERF_ROOT, "hypernerf", HYPERNERF_SCENES)
    # metrics_for("hypernerf", HYPERNERF_SCENES)
