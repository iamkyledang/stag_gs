# stag_gs: Sparse-Anchor Dynamic 3D Gaussians

Monocular dynamic scene reconstruction with **canonical 3D Gaussian Splatting + sparse per-anchor motion deltas**,
reconstructed on demand at query time (no deformation MLP). The full method description is in [framework.tex](framework.tex).

This is a fork of [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians): the canonical
3DGS model, data loaders, rasterizer and metrics are kept as-is, and the deformation MLP is replaced entirely by
[scene/sparse_anchor_motion.py](scene/sparse_anchor_motion.py). The GUI viewer, which depended on the MLP, was removed.

## How it works

Every dynamic Gaussian stores translation/rotation/scale deltas at sparse temporal anchors placed every `A` training
frames. A query at time `t` locates the bracketing anchors, fits a local motion rate from a window of `W` anchor
intervals on each side (weighted least squares over the `K` spatial neighbours), transports the bracketing states to
`t`, and blends the two results. The deltas are fed to the standard rasterizer on top of the canonical Gaussians.
Storage is sparsified: a control subset is always stored, everything else only where it deviates from its neighbours'
prediction.

Training is from scratch: a static warm-up, coarse-to-fine anchors, then a one-shot step that freezes clearly static
Gaussians. Regularisers: local rigidity, temporal smoothness, scale-delta L2.

## Repo layout

| path | role |
| --- | --- |
| `train.py` | from-scratch training (canonical 3DGS + sparse-anchor motion) |
| `render.py` | render test/train split, or time/view interpolation videos |
| `metrics.py` | PSNR / SSIM / LPIPS over rendered outputs |
| `full_eval.sh` | train -> render -> metrics over NeRF-DS and HyperNeRF (interp) |
| `scene/sparse_anchor_motion.py` | anchors, KNN, local fit, transport, dynamic-set selection, sparse encode/decode |
| `scene/gaussian_model.py` | canonical 3DGS, with hooks so motion parameters follow clone/split/prune |
| `arguments/__init__.py` | CLI params, defaulted from [configs.json](configs.json) |

## Setup

```shell
git clone <this repo> --recursive
cd stag_gs

conda create -n stag_gs python=3.7
conda activate stag_gs

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Datasets

- [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/) and [HyperNeRF](https://hypernerf.github.io/) (interp split),
  both in Nerfies `dataset.json` format.
- [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) (`transforms_*.json` format).

Point `-s` at the scene folder that contains `dataset.json` (or `transforms_train.json` for D-NeRF):

```
├── nerf_ds/as_novel_view/
├── hypernerf_interp/interp_aleks-teapot/aleks-teapot/
├── dnerf/hook/
└── stag_gs/          <- this repository
```

## Train

```shell
# NeRF-DS / HyperNeRF (20k iterations by default)
python train.py -s ../nerf_ds/as_novel_view -m output/as --eval --load2gpu_on_the_fly

# D-NeRF
python train.py -s ../dnerf/hook -m output/hook --eval --white_background
```

Most useful sparse-anchor motion flags (full list and defaults in `SparseAnchorParams`, [arguments/__init__.py](arguments/__init__.py)):

| flag | meaning | default |
| --- | --- | --- |
| `--anchor_stride` | final anchor spacing, in training frames | 8 |
| `--temporal_window` | anchor intervals per side used for the local motion fit (`1` = LERP/SLERP) | 2 |
| `--knn_k` | spatial neighbourhood size for the local fit | 48 |
| `--warm_up` | static canonical iterations before motion parameters exist | 3000 |
| `--anchor_refine_iters` | iterations at which the anchor stride is halved | `5000 7000 9000` |
| `--dyn_select_iter` | iteration at which static Gaussians are frozen | 11000 |
| `--epsilon` | storage sparsification threshold for dynamic deltas | 0.05 |
| `--lambda_rigid`, `--lambda_temporal`, `--lambda_scale` | regulariser weights | 1.0 / 0.05 / 1.0 |

All defaults come from [configs.json](configs.json); CLI flags override it.

## Render & evaluate

```shell
python render.py -m output/as --mode render
python metrics.py -m output/as

# query-time ablations on an already-trained model (no retraining)
python render.py -m output/as --mode render --sparse_anchor_W 1 --sparse_anchor_K 8 --sparse_anchor_epsilon 0.1
```

`--mode` is one of `render` (test images), `time` (time interpolation), `view` (view synthesis), `all` (time + view),
`original` (time + view along the captured trajectory, real-world data).

Whole benchmark (train -> render -> metrics over NeRF-DS and HyperNeRF):

```shell
./full_eval.sh [output_path] [iteration]
```

## Output layout

Same layout as the Deformable-3D-Gaussians baseline, with the deform-MLP checkpoint replaced by `motion/`:

```
output/<exp>/
├── cfg_args, cameras.json, input.ply         # written once at train start
├── point_cloud/iteration_<N>/point_cloud.ply # canonical Gaussians (train.py / render.py load from here)
├── motion/iteration_<N>/motion.npz           # sparse motion deltas + anchor times + dynamic index (replaces deform/*.pth)
├── train|test/ours_<N>/{renders,gt,depth}    # render.py --mode render
└── train|test/ours_<N>/interpolate_*_<N>/    # render.py --mode time|view|pose|original|all
```

`metrics.py` reads `test/ours_<N>/{renders,gt}` directly, so the same `train.py -> render.py -> metrics.py` sequence
used for the baseline works unchanged here.

## Acknowledgments

This repository builds on [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) and
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/); datasets from
[D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html), [HyperNeRF](https://hypernerf.github.io/) and
[NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/). Code is released under the original Inria license (see `LICENSE`).

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
@Article{kerbl3Dgaussians,
    author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal={ACM Transactions on Graphics},
    number={4},
    volume={42},
    month={July},
    year={2023},
    url={https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
