cat > "$REPO/activate_stag.sh" <<'EOF'
#!/bin/bash

export REPO=/network-volume/stag_gs

export TMPDIR="$REPO/.tmp"
export XDG_CACHE_HOME="$REPO/.cache"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export CONDA_PKGS_DIRS="$REPO/.conda/pkgs"
export TORCH_EXTENSIONS_DIR="$REPO/.cache/torch_extensions"
export HF_HOME="$REPO/.cache/huggingface"

source "$REPO/.conda/etc/profile.d/conda.sh"
conda activate "$REPO/.conda/envs/stag_h100"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4

cd "$REPO"
EOF