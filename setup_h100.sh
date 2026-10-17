#!/usr/bin/env bash
# All-in-one H100 setup for STAG-GS. Safe to re-run; skips steps already completed.
# Usage: bash setup_h100.sh [--rebuild] [--clean-cache]
set -euo pipefail

die() { echo "ERROR: $1" >&2; exit 1; }
say() { printf '\n=== %s ===\n' "$1"; }

REBUILD=0
CLEAN_CACHE=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        --clean-cache) CLEAN_CACHE=1 ;;
        *) die "Unknown option: $arg" ;;
    esac
done

export REPO="${REPO:-/network-volume/stag_gs}"
mkdir -p "$REPO"
cd "$REPO" || die "Cannot cd into $REPO"

CONDA_ROOT="$REPO/.conda"
ENV_PREFIX="$CONDA_ROOT/envs/stag_h100"
ACTIVATE_FILE="$REPO/.activate_h100_local.sh"
LOGFILE="$REPO/setup_h100.log"
exec > >(tee -a "$LOGFILE") 2>&1
printf '\nSetup run started: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

export TMPDIR="$REPO/.tmp"
export XDG_CACHE_HOME="$REPO/.cache"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export CONDA_PKGS_DIRS="$CONDA_ROOT/pkgs"
export TORCH_EXTENSIONS_DIR="$REPO/.cache/torch_extensions"
export HF_HOME="$REPO/.cache/huggingface"
export TORCH_HOME="$REPO/.cache/torch"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TORCH_EXTENSIONS_DIR"

say '[1/8] Install Miniconda into the persistent network volume'
if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    MINICONDA_SH="$TMPDIR/miniconda.sh"
    curl -fsSL -o "$MINICONDA_SH" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$MINICONDA_SH" -b -p "$CONDA_ROOT"
    rm -f "$MINICONDA_SH"
else
    echo "Miniconda already installed at $CONDA_ROOT; skipping."
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

say '[2/8] Create (or reuse) the stag_h100 Conda environment'
if [[ ! -d "$ENV_PREFIX" ]]; then
    conda create -y -p "$ENV_PREFIX" python=3.10
else
    echo "Conda environment already exists at $ENV_PREFIX; skipping."
fi
conda activate "$ENV_PREFIX" || die "Failed to activate $ENV_PREFIX"

say '[3/8] Install PyTorch (CUDA 12.1) and Python dependencies'
if ! python -c 'import torch' >/dev/null 2>&1; then
    python -m pip install --no-cache-dir torch torchvision \
        --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
else
    echo 'PyTorch already installed; skipping.'
fi
grep -v '^submodules/' requirements.txt > "$TMPDIR/requirements.no_submodules.txt"
python -m pip install --no-cache-dir -r "$TMPDIR/requirements.no_submodules.txt"

say '[4/8] Verify CUDA is visible and the GPU is an H100 (sm_90)'
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is not visible to PyTorch; check GPU drivers/runtime.'
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name, '| SM:', (p.major, p.minor), '| VRAM:', round(p.total_memory / 1024**3, 2), 'GiB')
assert (p.major, p.minor) == (9, 0), 'This script targets NVIDIA H100 sm_90; adapt build flags for other GPUs.'
PY

say '[5/8] Configure C++ compiler for CUDA 12.1'
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
"$CXX" --version | head -1

say '[6/8] Install persistent tmux, zip and unzip via Conda'
if [[ -x "$ENV_PREFIX/bin/tmux" && -x "$ENV_PREFIX/bin/zip" && -x "$ENV_PREFIX/bin/unzip" ]]; then
    echo 'tmux, zip and unzip already present.'
else
    conda install -y -c conda-forge tmux zip unzip
fi
"$ENV_PREFIX/bin/tmux" -V

say '[7/8] Build CUDA extensions (non-editable installs)'
cd "$REPO"
if (( REBUILD )) || ! python -c 'import torch; import simple_knn._C' >/dev/null 2>&1; then
    echo 'Installing simple_knn from local STAG-GS source.'
    python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        --force-reinstall -v "$REPO/submodules/simple-knn"
else
    echo 'simple_knn imports correctly; skipping rebuild.'
fi
if (( REBUILD )) || ! python - <<'PY' >/dev/null 2>&1
import torch, inspect
from diff_gaussian_rasterization import GaussianRasterizer
assert 'pix_label' in inspect.signature(GaussianRasterizer.forward).parameters
PY
then
    echo 'Installing patched Gaussian rasterizer from local STAG-GS source.'
    python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        --force-reinstall -v "$REPO/submodules/depth-diff-gaussian-rasterization"
else
    echo 'Rasterizer imports correctly and exposes pix_label; skipping rebuild.'
fi

say '[8/8] Write activation helper and verify installation'
cat > "$ACTIVATE_FILE" <<'ACTIVATE_SH'
#!/usr/bin/env bash
# Usage: source /network-volume/stag_gs/.activate_h100_local.sh
# Do not use "bash" or "./"; the script must affect your current shell.
export REPO=/network-volume/stag_gs
if [[ ! -f "$REPO/.conda/etc/profile.d/conda.sh" ]]; then
    echo "Conda not found at $REPO/.conda" >&2
    return 1
fi
export TMPDIR="$REPO/.tmp"
export XDG_CACHE_HOME="$REPO/.cache"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export CONDA_PKGS_DIRS="$REPO/.conda/pkgs"
export TORCH_EXTENSIONS_DIR="$REPO/.cache/torch_extensions"
export HF_HOME="$REPO/.cache/huggingface"
export TORCH_HOME="$REPO/.cache/torch"
source "$REPO/.conda/etc/profile.d/conda.sh" || return 1
conda activate "$REPO/.conda/envs/stag_h100" || return 1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='9.0'
export MAX_JOBS=4
TORCH_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
if [[ -d "$TORCH_LIB" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$TORCH_LIB:"*) ;;
        *) export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$REPO" || return 1
printf 'STAG-GS ready | env: %s | repo: %s\n' "$CONDA_PREFIX" "$PWD"
ACTIVATE_SH
chmod 644 "$ACTIVATE_FILE"

python - <<'PY'
import inspect
import torch  # Load torch first, making libc10.so visible for custom extensions.
import numpy, scipy, cv2
import simple_knn._C
import diff_gaussian_rasterization
from diff_gaussian_rasterization import GaussianRasterizer
assert 'pix_label' in inspect.signature(GaussianRasterizer.forward).parameters
print('simple_knn: OK')
print('Gaussian rasterizer and pix_label: OK')
print('torch:', torch.__version__, '| numpy:', numpy.__version__, '| OpenCV:', cv2.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('Import/API checks passed; still perform a training smoke test.')
PY
python -m pip check

if (( CLEAN_CACHE )); then
    say 'Clean Conda and pip caches to save space'
    conda clean --all -y
    python -m pip cache purge || true
fi

say 'Setup completed successfully'
echo "Activate in a NEW shell: source $ACTIVATE_FILE"
echo "Start a tmux session: source $ACTIVATE_FILE && tmux new -s stag_train"
echo "Rebuild patched CUDA extensions: bash $REPO/setup_h100.sh --rebuild"
echo "Log saved to: $LOGFILE"
du -sh "$CONDA_ROOT" "$ENV_PREFIX" "$REPO/.cache" 2>/dev/null || true
df -h "$REPO"
