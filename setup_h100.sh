# Use system GCC 12 when available; otherwise install a GCC 12 toolchain into Conda.
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
    echo 'System GCC 12 not available; installing Conda GCC/G++ 12.'
    conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 cmake ninja
    export CC="$(command -v x86_64-conda-linux-gnu-gcc)"
    export CXX="$(command -v x86_64-conda-linux-gnu-g++)"
fi
printf 'CC=%s\nCXX=%s\n' "$CC" "$CXX"
"$CXX" --version | head -1

say '6/8 Install tmux, zip, and unzip inside the persistent Conda environment'
if [[ -x "$CONDA_PREFIX/bin/tmux" && -x "$CONDA_PREFIX/bin/zip" && -x "$CONDA_PREFIX/bin/unzip" ]]; then
    echo 'tmux, zip, unzip already present in Conda environment.'
else
    conda install -y -c conda-forge tmux zip unzip
fi
"$CONDA_PREFIX/bin/tmux" -V

say '7/8 Install or validate native STAG-GS CUDA extensions'
cd "$REPO"
if (( REBUILD )) || ! python -c 'import torch; import simple_knn._C' >/dev/null 2>&1; then
    echo 'Building simple_knn (non-editable install).'
    python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        --force-reinstall -v "$REPO/submodules/simple-knn"
else
    echo 'simple_knn already imports successfully; skipping rebuild.'
fi
if (( REBUILD )) || ! python - <<'PY' >/dev/null 2>&1
import torch
import inspect
import diff_gaussian_rasterization
from diff_gaussian_rasterization import GaussianRasterizer
assert 'pix_label' in inspect.signature(GaussianRasterizer.forward).parameters
PY
then
    echo 'Building patched diff_gaussian_rasterization (non-editable install).'
    python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        --force-reinstall -v "$REPO/submodules/depth-diff-gaussian-rasterization"
else
    echo 'Gaussian rasterizer imports and exposes pix_label; skipping rebuild.'
fi

say '8/8 Create local activation helper and verify installation'
cat > "$ACTIVATE_FILE" <<'ACTIVATE_SH'
#!/usr/bin/env bash
# Usage: source /network-volume/stag_gs/.activate_h100_local.sh
# Must be sourced, not executed in a child shell.
export REPO=/network-volume/stag_gs
if [[ ! -f "$REPO/.conda/etc/profile.d/conda.sh" ]]; then
    echo 'Conda installation not found. Run bash /network-volume/stag_gs/setup.sh first.' >&2
    return 1
fi
export TMPDIR="$REPO/.tmp"
export XDG_CACHE_HOME="$REPO/.cache"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export CONDA_PKGS_DIRS="$REPO/.conda/pkgs"
export TORCH_EXTENSIONS_DIR="$REPO/.cache/torch_extensions"
export HF_HOME="$REPO/.cache/huggingface"
export TORCH_HOME="$REPO/.cache/torch"
source "$REPO/.conda/etc/profile.d/conda.sh"
conda activate "$REPO/.conda/envs/stag_h100" || return 1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='9.0'
export MAX_JOBS=4
TORCH_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
if [[ -d "$TORCH_LIB" ]]; then
    export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$REPO" || return 1
printf 'STAG-GS ready | env: %s | repo: %s\n' "$CONDA_PREFIX" "$PWD"
ACTIVATE_SH
chmod 644 "$ACTIVATE_FILE"

python - <<'PY'
import inspect
import numpy
import torch   # Import torch first; makes libc10.so available.
import cv2
import scipy
import simple_knn._C
import diff_gaussian_rasterization
from diff_gaussian_rasterization import GaussianRasterizer
assert 'pix_label' in inspect.signature(GaussianRasterizer.forward).parameters
print('simple_knn: OK')
print('Gaussian rasterizer: OK (pix_label present)')
print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('NumPy:', numpy.__version__, '| OpenCV:', cv2.__version__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('Import/API checks passed; run a short training smoke test before a full 45k run.')
PY
python -m pip check

if (( CLEAN_CACHE )); then
    say 'Cleaning Conda and pip caches to save network-volume space'
    conda clean --all -y
    python -m pip cache purge || true
fi

say 'Installation complete'
echo "Activation (from a new SSH shell): source $ACTIVATE_FILE"
echo "Start a tmux session: source $ACTIVATE_FILE && tmux new -s stag_train"
echo "Rebuild extensions after CUDA source changes: bash $REPO/setup.sh --rebuild"
echo "Setup log: $LOGFILE"
du -sh "$CONDA_ROOT" "$ENV_PREFIX" "$REPO/.cache" 2>/dev/null || true
df -h "$REPO"
