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
