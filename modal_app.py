"""Modal.com app to train stag_gs on a cloud GPU.

Usage (from the stag_gs directory, with the `modal` CLI installed/authenticated):
    modal run modal_app.py --scene as_novel_view --iterations 40000

Datasets must already be uploaded to the `stag-gs-data` Volume under nerf_ds/<scene>, e.g.:
    modal volume put stag-gs-data ../nerf_ds nerf_ds
"""

import pathlib

import modal

STAG_GS_DIR = pathlib.Path(__file__).parent

app = modal.App("stag-gs")

data_volume = modal.Volume.from_name("stag-gs-data", create_if_missing=True)
output_volume = modal.Volume.from_name("stag-gs-output", create_if_missing=True)

# Modal's add_python only supports >=3.10, so the image driver python is 3.11 (used only for the
# modal client itself); python3.7 is built from source to actually run stag_gs, matching its
# torch==1.13.1+cu116 / Python 3.7 requirement. (deadsnakes no longer ships python3.7-dev/
# python3.7-distutils for focal since 3.7 is EOL, so the apt route is no longer reliable.)
# CUDA 11.6 nvcc can't target Ada (sm_89, e.g. L40S) directly; "8.0+PTX" builds the two custom
# extensions with PTX for compute_80 so the driver JIT-compiles to whatever SM is actually present.
image = (
    modal.Image.from_registry("nvidia/cuda:11.6.2-devel-ubuntu20.04", add_python="3.11")
    # avoids an interactive tzdata prompt when apt later pulls it in as a dependency
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install(
        "git", "build-essential", "curl", "wget", "tzdata",
        "libssl-dev", "libffi-dev", "libbz2-dev", "libreadline-dev",
        "libsqlite3-dev", "zlib1g-dev", "liblzma-dev", "libncurses5-dev",
        "libgdbm-dev", "uuid-dev", "tk-dev",
    )
    .run_commands(
        "wget -q https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz -O /tmp/Python-3.7.17.tgz",
        "tar -xzf /tmp/Python-3.7.17.tgz -C /tmp",
        # no --enable-optimizations: PGO re-runs the full test suite (slow/flaky in containers)
        # and isn't worth it here since the GPU, not the interpreter, is the bottleneck.
        "cd /tmp/Python-3.7.17 && ./configure --with-ensurepip=install"
        " && make -j$(nproc) && make altinstall",
    )
    .run_commands(
        "python3.7 -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 "
        "--extra-index-url https://download.pytorch.org/whl/cu116",
        "python3.7 -m pip install plyfile==0.8.1 tqdm imageio==2.27.0 opencv-python "
        "imageio-ffmpeg scipy dearpygui lpips",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.0+PTX"})
    .add_local_dir(
        str(STAG_GS_DIR),
        remote_path="/root/stag_gs",
        copy=True,
        ignore=[
            "**/output_*/**", "**/_smoketest_model/**", "**/.git/**",
            "**/__pycache__/**", "**/change_log/**",
        ],
    )
    .run_commands(
        "python3.7 -m pip install --no-build-isolation /root/stag_gs/submodules/simple-knn",
        "python3.7 -m pip install --no-build-isolation /root/stag_gs/submodules/depth-diff-gaussian-rasterization",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": data_volume, "/output": output_volume},
    timeout=60 * 60 * 12,
)
def train(scene: str = "as_novel_view", iterations: int = 40000, tag: str = ""):
    import subprocess

    source_path = f"/data/nerf_ds/{scene}"
    model_path = f"/output/{scene}_{tag}" if tag else f"/output/{scene}"
    cmd = [
        "python3.7", "train.py",
        "-s", source_path,
        "-m", model_path,
        "--eval", "--load2gpu_on_the_fly",
        "--iterations", str(iterations),
    ]
    subprocess.run(cmd, check=True, cwd="/root/stag_gs")
    output_volume.commit()


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": data_volume, "/output": output_volume},
    timeout=60 * 60,
)
def render(scene: str = "as_novel_view", iteration: int = -1, tag: str = ""):
    import subprocess

    # -s is omitted: get_combined_args reads source_path back from cfg_args saved in model_path
    model_path = f"/output/{scene}_{tag}" if tag else f"/output/{scene}"
    cmd = [
        "python3.7", "render.py",
        "-m", model_path,
        "--iteration", str(iteration),
    ]
    subprocess.run(cmd, check=True, cwd="/root/stag_gs")
    output_volume.commit()


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": data_volume, "/output": output_volume},
    timeout=60 * 60,
)
def metrics(scene: str = "as_novel_view", tag: str = ""):
    import subprocess

    model_path = f"/output/{scene}_{tag}" if tag else f"/output/{scene}"
    cmd = ["python3.7", "metrics.py", "-m", model_path]
    subprocess.run(cmd, check=True, cwd="/root/stag_gs")
    output_volume.commit()


@app.local_entrypoint()
def main(
    scene: str = "as_novel_view",
    iterations: int = 40000,
    action: str = "train",
    iteration: int = -1,
    tag: str = "",
):
    # spawn (not .remote()) so the job survives client disconnects/Ctrl+C without needing --detach;
    # check progress with `modal app logs stag-gs` instead of waiting on this call.
    if action == "train":
        call = train.spawn(scene=scene, iterations=iterations, tag=tag)
    elif action == "render":
        call = render.spawn(scene=scene, iteration=iteration, tag=tag)
    elif action == "metrics":
        call = metrics.spawn(scene=scene, tag=tag)
    else:
        raise ValueError(f"Unknown action: {action}")
    print(f"Spawned {action} as call id: {call.object_id}")
    print("Tail logs with: modal app logs stag-gs")
