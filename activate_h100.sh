cd /network-volume/stag_gs

cat > activate_stag.sh <<'EOF'
#!/bin/bash

# STAG-GS project directory
export REPO=/network-volume/stag_gs

# Activate Conda environment
source "$REPO/.conda/etc/profile.d/conda.sh"
conda activate "$REPO/.conda/envs/stag_h100"

# CUDA configuration for NVIDIA H100 (SM 9.0)
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4

# Cache and temporary directories
export TMPDIR="$REPO/.tmp"
export XDG_CACHE_HOME="$REPO/.cache"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export CONDA_PKGS_DIRS="$REPO/.conda/pkgs"
export TORCH_EXTENSIONS_DIR="$REPO/.cache/torch_extensions"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# PyTorch shared libraries (e.g. libc10.so)
TORCH_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
if [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Enter project directory
cd "$REPO"

echo "===================================="
echo "STAG-GS environment activated!"
echo "Conda: $CONDA_PREFIX"
echo "CUDA: $CUDA_HOME"
echo "GPU architecture: $TORCH_CUDA_ARCH_LIST"
echo "===================================="
EOF