#
# LazyGS: lazy-evaluated dynamic 3D Gaussians (see framework.tex), trained from scratch through the
# query-time operator. Derived from train.py of Deformable-3D-Gaussians / 3D Gaussian Splatting
# (Inria / GRAPHDECO license, see LICENSE.md).
#

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import LazyParams, ModelParams, OptimizationParams, PipelineParams, train_config
from gaussian_renderer import render
from scene import GaussianModel, LazyMotionModel, Scene
from utils.general_utils import get_linear_noise_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def count_train_frames(cameras):
    return len(set(round(float(cam.fid.reshape(-1)[0]), 6) for cam in cameras))


def motion_deltas(motion, fid, t_noise=0.0):
    """(d_xyz, d_rotation, d_scaling, info) for the renderer; zeros before the motion phase starts."""
    if not motion.active:
        return 0.0, 0.0, 0.0, None
    return motion.deltas_at(float(fid.reshape(-1)[0]) + t_noise)


def training(dataset, opt, pipe, lazy, testing_iterations, saving_iterations, checkpoint_iterations, start_checkpoint):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    T = count_train_frames(scene.getTrainCameras())
    motion = LazyMotionModel(lazy, T, scene.cameras_extent)

    resumed_iteration = 0
    if start_checkpoint:
        # exact bit-for-bit resume (dense params + Adam moments), correct at any iteration -- before
        # warm_up (motion_state is None), between warm_up and dyn_select_iter (is_dyn still all-True),
        # or after freeze. Unlike the sparsified point_cloud/motion.npz pair, nothing is reconstructed.
        model_params, motion_state, resumed_iteration = torch.load(start_checkpoint)
        gaussians.restore(model_params, opt)
        if motion_state is not None:
            motion.restore(gaussians, motion_state, opt)
    else:
        gaussians.training_setup(opt)

    gaussians.on_densify = motion.on_densify
    gaussians.on_prune = motion.on_prune
    print("[Lazy] {} training frames, A={} (start stride {}), W={}, K={}".format(
        T, lazy.anchor_stride, lazy.anchor_stride * 2 ** lazy.coarse_levels, lazy.temporal_window, lazy.knn_k))

    # AST time-noise (real-data trick from Deformable-3D-Gaussians): jitters the query time by up to
    # ~half a frame early in training, annealed to ~0 by the end, so the motion model doesn't overfit to
    # the exact per-frame timestamps of a hand-held/jittery capture.
    time_interval = 1.0 / max(T, 1)
    ast_smooth = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=opt.iterations)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_reg = {'rigid': 0.0, 'temporal': 0.0, 'scale': 0.0}
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress", initial=resumed_iteration)
    for iteration in range(resumed_iteration + 1, opt.iterations + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ---- lazy-motion schedule: warm-up -> coarse-to-fine anchors -> dynamic-set selection
        if iteration == opt.warm_up and not motion.active:
            motion.setup(gaussians, opt)
        if motion.active:
            if iteration in lazy.anchor_refine_iters:
                motion.refine_anchors(iteration)
            if iteration == lazy.dyn_select_iter:
                motion.freeze_background(lazy.dyn_w_min)
            if iteration % lazy.knn_update_interval == 0:
                motion.knn_dirty = True

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        ast_noise = float(torch.randn((), device='cuda').item()) * time_interval * ast_smooth(iteration) \
            if motion.active else 0.0
        d_xyz, d_rotation, d_scaling, info = motion_deltas(motion, fid, ast_noise)

        # Render (+ visibility-weighted mask-vote splatting in the same pass when this frame has a
        # dynamic-object mask; masks are missing for HyperNeRF until segmented, see generate_masks_sam2.py)
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling,
                               pix_label=viewpoint_cam.mask)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if info is not None:
            reg = motion.regularization(info)
            loss = loss + lazy.lambda_rigid * reg['rigid'] + lazy.lambda_temporal * reg['temporal'] \
                + lazy.lambda_scale * reg['scale']
            for k in ema_reg:
                ema_reg[k] = 0.4 * reg[k].item() + 0.6 * ema_reg[k]
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                post = {"Loss": f"{ema_loss_for_log:.{5}f}", "N": gaussians.get_xyz.shape[0]}
                if motion.active:
                    post.update({"dyn": int(motion.is_dyn.sum()), "M": motion.num_anchors,
                                 "rigid": f"{ema_reg['rigid']:.{4}f}", "temp": f"{ema_reg['temporal']:.{4}f}"})
                progress_bar.set_postfix(post)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Mask-vote splatting: EMA-merge this frame's per-Gaussian votes (sized for the pre-densification
            # point set, so this must run before densify_and_prune below)
            if viewpoint_cam.mask is not None:
                gaussians.accumulate_dyn_votes(render_pkg_re["dyn_votes_fg"], render_pkg_re["dyn_votes_w"],
                                               lazy.mask_vote_decay)

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), motion,
                                       dataset.load2gpu_on_the_fly)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if motion.active:
                    motion.save(dataset.model_path, iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                motion_state = motion.capture() if motion.active else None
                torch.save((gaussians.capture(), motion_state, iteration),
                           os.path.join(dataset.model_path, "chkpnt" + str(iteration) + ".pth"))

            # Densification (motion parameters follow through the GaussianModel on_densify / on_prune hooks)
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                if motion.optimizer is not None:
                    motion.optimizer.step()
                    motion.optimizer.zero_grad(set_to_none=True)
                    motion.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, motion, load2gpu_on_the_fly):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # accumulate scalar sums instead of stacking every frame's image on GPU (OOM risk on
                # large/frequent test sets); only the first 5 views are kept (as tensors) for TB logging.
                l1_sum = 0.0
                psnr_sum = 0.0
                n_views = len(config['cameras'])
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    d_xyz, d_rotation, d_scaling, _ = motion_deltas(motion, viewpoint.fid)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_sum += l1_loss(image, gt_image).item()
                    psnr_sum += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    del image, gt_image

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')

                l1_test = l1_sum / n_views
                psnr_test = psnr_sum / n_views
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = torch.tensor(psnr_test)
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                torch.cuda.empty_cache()

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            if motion.active:
                tb_writer.add_scalar('lazy/dynamic_points', int(motion.is_dyn.sum()), iteration)
                tb_writer.add_scalar('lazy/num_anchors', motion.num_anchors, iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    tc = train_config()  # defaults for the flags below live in configs.json's "train" section

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    zp = LazyParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=tc.get("detect_anomaly", False))
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=tc.get("test_iterations", [5000, 6000, 7_000] + list(range(10000, 40001, 1000))))
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=tc.get("save_iterations", [7_000, 10_000, 20_000, 30_000, 40000]))
    parser.add_argument("--quiet", action="store_true", default=tc.get("quiet", False))
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int,
                        default=tc.get("checkpoint_iterations", []),
                        help="Save a resumable (gaussians+motion+optimizer state) .pth at these iterations")
    parser.add_argument("--start_checkpoint", type=str, default=tc.get("start_checkpoint", None),
                        help="Resume training from a .pth saved via --checkpoint_iterations")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), zp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
