#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import json
import sys
import os

# configs.json (stag_gs/configs.json) is the single source of truth for hyperparameter defaults;
# CLI flags still override whatever is loaded here.
_CONFIGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs.json")
try:
    with open(_CONFIGS_PATH) as _f:
        CONFIGS = json.load(_f)
except FileNotFoundError:
    CONFIGS = {}


def eval_config():
    return CONFIGS.get("eval", {})


def train_config():
    return CONFIGS.get("train", {})


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False, config_section=None):
        overrides = CONFIGS.get(config_section, {}) if config_section else {}
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            value = overrides.get(key, value)
            t = type(value)
            elem_t = type(value[0]) if t == list and len(value) > 0 else int
            value = value if not fill_none else None
            if t == list:
                group.add_argument("--" + key, default=value, nargs="+", type=elem_t)
            elif shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        super().__init__(parser, "Loading Parameters", sentinel, config_section="model")

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters", config_section="pipeline")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 20_000
        self.warm_up = 3_000                # static (canonical-only) iterations before motion parameters are created
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0007
        super().__init__(parser, "Optimization Parameters", config_section="optimization")


class SparseAnchorParams(ParamGroup):
    """Sparse-anchor dynamic Gaussians (framework.tex): A / W / K and the training schedule."""

    def __init__(self, parser, sentinel=False):
        # representation (A, W, K)
        self.anchor_stride = 8            # A: anchors every A training frames (after coarse-to-fine)
        self.temporal_window = 2          # W: anchor intervals used on each side of the query
        self.knn_k = 48                   # K: spatial neighbourhood size (same-object graph, degree-3 cubic MLS)
        self.transport_order = 2          # 1 = velocity only (W=1 -> pure LERP), 2 = velocity + acceleration
        self.consensus_sigma = 1.0        # softness of the fit-residual confidence in lambda_i
        self.no_fit_confidence = False    # lambda_i = time weight only
        self.wls_eps = 1e-2               # relative Tikhonov term epsilon_J in the cubic MLS fit
        # from-scratch training schedule
        self.coarse_levels = 3            # start at stride A*2^levels
        self.anchor_refine_iters = [5000, 7000, 9000]
        self.dyn_select_iter = 11000      # one-shot background freeze (segmentation-driven dynamic/static split)
        self.dyn_w_min = 10.0             # min accumulated mask-vote visibility to trust get_dyn_prob at freeze
        self.knn_update_interval = 1000
        # mask-vote splatting (dynamic/static segmentation prior, see change_log)
        self.mask_vote_decay = 0.999        # EMA factor per iteration on the per-Gaussian vote sums (~1000-iter memory)
        # storage sparsification (object-bounding-radius denominator, see change_log)
        self.epsilon = 0.05               # normalised change score above which a dynamic Gaussian's delta is stored
        self.score_kappa_rot = 0.2        # radians that count as one unit of change
        self.score_kappa_scale = 0.2      # log-scale units that count as one unit of change
        # regularisation
        self.lambda_rigid = 1.0
        self.lambda_temporal = 0.05
        self.lambda_scale = 1.0
        # learning rates (exponential decay to 10% of init, rescoped to start at warm_up/each refine -- see
        # SparseAnchorMotionModel._rebuild_schedule)
        self.motion_lr_init = 0.0008
        self.motion_lr_final = 0.00008
        self.motion_rot_lr = 0.001
        self.motion_scale_lr = 0.001
        super().__init__(parser, "Sparse Anchor Motion Parameters", sentinel, config_section="sparse_anchor")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
        elif k not in merged_dict:
            # arg didn't exist yet when cfg_args was saved during training; fall back to its default
            merged_dict[k] = v
    return Namespace(**merged_dict)
