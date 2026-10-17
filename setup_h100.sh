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
