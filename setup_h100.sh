#!/usr/bin/env bash
# STAG-GS / NVIDIA H100 (SM 9.0) - installation, validation and activation.
# Persistent files live under /network-volume/stag_gs (including Miniforge).
#
# Recommended: source /network-volume/stag_gs/setup_h100.sh
#   -> verify existing env; install missing pieces if necessary; ACTIVATE in this shell.
# Fast login: source /network-volume/stag_gs/setup_h100.sh --activate-only
#   -> verify without installing, then activate.
# Install without activating parent shell: bash /network-volume/stag_gs/setup_h100.sh
# Options: --check, --activate-only, --install, --rebuild, --keep-cache, --repair-conda
#
# Never deletes an existing Conda environment, datasets, outputs or checkpoints.
# Does not git pull, change drivers, or clone datasets.

STAG_REPO=/network-volume/stag_gs
STAG_SCRIPT_PATH="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
STAG_CONDA_ROOT="$STAG_REPO/.conda"
STAG_ENV_PREFIX="$STAG_CONDA_ROOT/envs/stag_h100"
STAG_REBUILD=0
STAG_KEEP_CACHE=0
STAG_REPAIR_CONDA=0
STAG_CHECK_ONLY=0
STAG_ACTIVATE_ONLY=0
STAG_FORCE_INSTALL=0
STAG_SOURCED=0
[[ "${BASH_SOURCE[0]}" != "$0" ]] && STAG_SOURCED=1

_stag_usage() {
    cat <<'EOF'
Usage:
  source /network-volume/stag_gs/setup_h100.sh                 # Check, repair if needed, activate
  source /network-volume/stag_gs/setup_h100.sh --activate-only # Check and activate; never installs
  bash   /network-volume/stag_gs/setup_h100.sh                 # Setup/check, cannot activate parent shell
  bash   /network-volume/stag_gs/setup_h100.sh --rebuild       # Recompile CUDA extensions
  bash   /network-volume/stag_gs/setup_h100.sh --check         # Read-only environment check
Options:
  --install        Run the setup stages even if the environment passes the initial check
  --rebuild        Force rebuild of both CUDA extensions
  --repair-conda   Explicitly repair an incomplete Miniforge base using installer -u
  --keep-cache     Do not clean package caches after installing
EOF
}

for _stag_arg in "$@"; do
    case "$_stag_arg" in
        --check) STAG_CHECK_ONLY=1 ;;
        --activate-only) STAG_ACTIVATE_ONLY=1 ;;
        --install) STAG_FORCE_INSTALL=1 ;;
        --rebuild) STAG_REBUILD=1; STAG_FORCE_INSTALL=1 ;;
        --keep-cache) STAG_KEEP_CACHE=1 ;;
        --repair-conda) STAG_REPAIR_CONDA=1; STAG_FORCE_INSTALL=1 ;;
        -h|--help)
            _stag_usage
            if (( STAG_SOURCED )); then return 0; else exit 0; fi ;;
        *)
            printf 'Unknown option: %s\n' "$_stag_arg" >&2
            _stag_usage >&2
            if (( STAG_SOURCED )); then return 2; else exit 2; fi ;;
    esac
done
if (( STAG_CHECK_ONLY && (STAG_ACTIVATE_ONLY || STAG_FORCE_INSTALL) )); then
    echo 'ERROR: --check cannot be combined with --activate-only, --install or --rebuild.' >&2
    if (( STAG_SOURCED )); then return 2; else exit 2; fi
fi
if (( STAG_ACTIVATE_ONLY && STAG_FORCE_INSTALL )); then
    echo 'ERROR: --activate-only cannot be combined with installation/rebuild options.' >&2
    if (( STAG_SOURCED )); then return 2; else exit 2; fi
fi

# The work runs in a SUBSHELL. It can safely use set -e, error traps, and log
# redirection without changing your interactive shell's settings or stdout.
_stag_worker() (
    set -Eeuo pipefail

    say() { printf '\n==> %s\n' "$*"; }
    die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

    conda_usable() {
        [[ -x "$STAG_CONDA_ROOT/bin/conda" && -f "$STAG_CONDA_ROOT/etc/profile.d/conda.sh" ]] &&
            "$STAG_CONDA_ROOT/bin/conda" --version >/dev/null 2>&1
    }

    # Import-check deliberately imports torch BEFORE the two .so extensions;
    # this also helps the dynamic linker find libc10.so on some installations.
    env_usable() {
        [[ -f "$STAG_REPO/train.py" ]] || return 1
        conda_usable || return 1
        [[ -x "$STAG_ENV_PREFIX/bin/python" && -x "$STAG_ENV_PREFIX/bin/nvcc" ]] || return 1
        [[ -x "$STAG_ENV_PREFIX/bin/tmux" && -x "$STAG_ENV_PREFIX/bin/zip" && -x "$STAG_ENV_PREFIX/bin/unzip" ]] || return 1
        "$STAG_ENV_PREFIX/bin/nvcc" --version 2>/dev/null | grep -q 'release 12\.1' || return 1
        LD_LIBRARY_PATH="$STAG_ENV_PREFIX/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$STAG_ENV_PREFIX/bin/python" - <<'PY' >/dev/null 2>&1
import sys, inspect
assert sys.version_info[:2] == (3, 10)
import torch
assert torch.__version__.split('+')[0] == '2.1.2'
assert torch.version.cuda == '12.1'
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (9, 0)
import torchvision
import numpy, scipy, cv2, plyfile, imageio, imageio_ffmpeg, tqdm, lpips, tensorboard
assert torchvision.__version__.split('+')[0] == '0.16.2'
assert numpy.__version__ == '1.26.4'
assert scipy.__version__ == '1.11.4'
assert cv2.__version__ == '4.10.0'
assert imageio.__version__ == '2.27.0'
from importlib.metadata import version
assert version('plyfile') == '0.8.1'
import simple_knn._C
import diff_gaussian_rasterization
from diff_gaussian_rasterization import GaussianRasterizer
assert 'pix_label' in inspect.signature(GaussianRasterizer.forward).parameters
PY
    }

    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Requires Linux x86_64.'
    [[ -d "$STAG_REPO" && -f "$STAG_REPO/train.py" ]] || die "Expected STAG-GS repo at $STAG_REPO"

    # --check/--activate-only perform only lightweight checks and never install.
    if (( STAG_CHECK_ONLY || STAG_ACTIVATE_ONLY )); then
        if env_usable; then
            echo "Environment OK: $STAG_ENV_PREFIX"
            exit 0
        fi
        echo 'Environment incomplete, unavailable, or wrong GPU architecture.' >&2
        echo "Run: bash $STAG_SCRIPT_PATH" >&2
        exit 1
    fi

    if (( ! STAG_FORCE_INSTALL )) && env_usable; then
        echo "Environment already OK: $STAG_ENV_PREFIX; nothing to install."
        exit 0
    fi

    mkdir -p "$STAG_REPO" "$STAG_REPO/.tmp" "$STAG_REPO/.cache/pip" "$STAG_CONDA_ROOT/pkgs"
    export TMPDIR="$STAG_REPO/.tmp"
    export PIP_CACHE_DIR="$STAG_REPO/.cache/pip"
    export CONDA_PKGS_DIRS="$STAG_CONDA_ROOT/pkgs"

    say '[1/7] Install or repair Miniforge'
    if (( STAG_REPAIR_CONDA )) || ! conda_usable; then
        MINIFORGE_SH="$TMPDIR/miniforge.sh"
        curl -fsSL -o "$MINIFORGE_SH" \
            https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
        bash "$MINIFORGE_SH" -b -u -p "$STAG_CONDA_ROOT"
        rm -f "$MINIFORGE_SH"
    else
        echo "Miniforge already usable at $STAG_CONDA_ROOT; skipping."
    fi
    source "$STAG_CONDA_ROOT/etc/profile.d/conda.sh"

    say '[2/7] Create the stag_h100 Conda environment (python 3.10)'
    if [[ ! -x "$STAG_ENV_PREFIX/bin/python" ]]; then
        conda create -y -p "$STAG_ENV_PREFIX" python=3.10
    else
        echo "Conda environment already exists at $STAG_ENV_PREFIX; skipping."
    fi
    conda activate "$STAG_ENV_PREFIX" || die "Failed to activate $STAG_ENV_PREFIX"

    say '[3/7] Install the CUDA 12.1 toolkit (nvcc) into the environment'
    if "$STAG_ENV_PREFIX/bin/nvcc" --version 2>/dev/null | grep -q 'release 12\.1'; then
        echo 'CUDA 12.1 toolkit already installed; skipping.'
    else
        conda install -y -c "nvidia/label/cuda-12.1.0" -c nvidia cuda-toolkit=12.1
    fi

    say '[4/7] Install pinned PyTorch 2.1.2 (cu121) and Python dependencies'
    python -m pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install --no-cache-dir \
        numpy==1.26.4 scipy==1.11.4 opencv-python==4.10.0.84 plyfile==0.8.1 \
        imageio==2.27.0 imageio-ffmpeg tqdm lpips dearpygui tensorboard

    say '[5/7] Install tmux, zip and unzip into the environment'
    if [[ -x "$STAG_ENV_PREFIX/bin/tmux" && -x "$STAG_ENV_PREFIX/bin/zip" && -x "$STAG_ENV_PREFIX/bin/unzip" ]]; then
        echo 'tmux, zip and unzip already present; skipping.'
    else
        conda install -y -c conda-forge tmux zip unzip
    fi

    say '[6/7] Configure a GCC/G++ toolchain compatible with CUDA 12.1'
    unset CC CXX || true
    if [[ -x /usr/bin/gcc-12 && -x /usr/bin/g++-12 ]]; then
        export CC=/usr/bin/gcc-12
        export CXX=/usr/bin/g++-12
    elif command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
        GCC_MAJOR="$(gcc -dumpversion | cut -d. -f1)"
        if [[ "$GCC_MAJOR" =~ ^[0-9]+$ ]] && (( GCC_MAJOR >= 7 && GCC_MAJOR <= 12 )); then
            export CC="$(command -v gcc)"
            export CXX="$(command -v g++)"
        fi
    fi
    if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
        conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 cmake ninja
        command -v x86_64-conda-linux-gnu-g++ >/dev/null 2>&1 \
            || die 'Conda GNU toolchain did not install correctly.'
        export CC="$(command -v x86_64-conda-linux-gnu-gcc)"
        export CXX="$(command -v x86_64-conda-linux-gnu-g++)"
    fi
    printf 'CC=%s\nCXX=%s\n' "$CC" "$CXX"

    say '[7/7] Build the patched CUDA extensions (simple_knn, diff_gaussian_rasterization)'
    cd "$STAG_REPO"
    export TORCH_CUDA_ARCH_LIST='9.0'
    export MAX_JOBS="${MAX_JOBS:-4}"
    # torch.utils.cpp_extension imports pkg_resources, which setuptools>=81 dropped.
    python -m pip install --no-cache-dir 'setuptools<81' wheel
    if (( STAG_REBUILD )) || ! python -c 'import torch; import simple_knn._C' >/dev/null 2>&1; then
        python -m pip install --no-cache-dir --no-build-isolation --no-deps \
            --force-reinstall -v "$STAG_REPO/submodules/simple-knn"
    else
        echo 'simple_knn already imports correctly; skipping rebuild.'
    fi
    if (( STAG_REBUILD )) || ! python - <<'PY' >/dev/null 2>&1
import torch, inspect
from diff_gaussian_rasterization import GaussianRasterizer
assert 'pix_label' in inspect.signature(GaussianRasterizer.forward).parameters
PY
    then
        python -m pip install --no-cache-dir --no-build-isolation --no-deps \
            --force-reinstall -v "$STAG_REPO/submodules/depth-diff-gaussian-rasterization"
    else
        echo 'Gaussian rasterizer already exposes pix_label; skipping rebuild.'
    fi

    if ! env_usable; then
        echo 'Post-install verification failed. Re-running the Python import/version checks verbosely:' >&2
        LD_LIBRARY_PATH="$STAG_ENV_PREFIX/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$STAG_ENV_PREFIX/bin/python" - <<'PY' || true
import sys, inspect
print('python:', sys.version_info[:2])
import torch
print('torch:', torch.__version__, '| cuda:', torch.version.cuda,
      '| available:', torch.cuda.is_available(),
      '| capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
import torchvision
import numpy, scipy, cv2, plyfile, imageio, imageio_ffmpeg, tqdm, lpips, tensorboard
print('torchvision:', torchvision.__version__, '| numpy:', numpy.__version__,
      '| scipy:', scipy.__version__, '| cv2:', cv2.__version__, '| imageio:', imageio.__version__)
from importlib.metadata import version
print('plyfile:', version('plyfile'))
import simple_knn._C
print('simple_knn._C: OK')
import diff_gaussian_rasterization
from diff_gaussian_rasterization import GaussianRasterizer
print('pix_label present:', 'pix_label' in inspect.signature(GaussianRasterizer.forward).parameters)
PY
        die 'Post-install verification failed; see output above.'
    fi

    if (( ! STAG_KEEP_CACHE )); then
        say 'Cleaning Conda and pip caches'
        conda clean --all -y
        python -m pip cache purge || true
    fi

    say 'Installation complete'
)

_stag_worker
_stag_rc=$?

if (( _stag_rc != 0 )); then
    if (( STAG_SOURCED )); then return "$_stag_rc"; else exit "$_stag_rc"; fi
fi

if (( STAG_CHECK_ONLY )); then
    if (( STAG_SOURCED )); then return 0; else exit 0; fi
fi

if (( STAG_SOURCED )); then
    printf '\n==> %s\n' 'Activating stag_h100 environment in this shell'
    source "$STAG_CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$STAG_ENV_PREFIX" || return 1
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    export TORCH_CUDA_ARCH_LIST='9.0'
    export MAX_JOBS="${MAX_JOBS:-4}"
    STAG_TORCH_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
    if [[ -d "$STAG_TORCH_LIB" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$STAG_TORCH_LIB:"*) ;;
            *) export LD_LIBRARY_PATH="$STAG_TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
    cd "$STAG_REPO" || return 1
    printf 'STAG-GS ready | env: %s | repo: %s\n' "$CONDA_PREFIX" "$PWD"
else
    echo "Setup complete. To activate in your shell, run: source $STAG_SCRIPT_PATH --activate-only"
fi
