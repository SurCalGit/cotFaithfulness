#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing system dependencies"
apt-get update -qq
apt-get install -y libxml2-dev libxslt-dev python3-dev

# ── Step 1: PyTorch (must come before requirements.txt) ───────────────────────
echo "==> Installing PyTorch 2.1.0 with CUDA 12.1"
pip install --force-reinstall torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Step 2: Install everything from requirements.txt ──────────────────────────
echo "==> Installing dependencies from requirements.txt"
pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r "${REPO_ROOT}/sparse_coding/requirements.txt" \
    --ignore-installed torch torchvision \
    || true   # torchaudio version conflict warning is harmless

# ── Step 3: Force-pin critical packages LAST so nothing above can override them
# Each group is pinned together in one command so pip can't upgrade one as a
# side effect of resolving another.
echo "==> Force-pinning critical packages (last step — nothing runs after this)"

# numpy + pyarrow + datasets + fsspec must be pinned together: pyarrow is
# compiled against numpy 1.x and will crash if numpy 2.x is present;
# fsspec 2023.6+ changed glob_translate in a way that breaks datasets==2.12.0.
pip install --force-reinstall \
    numpy==1.24.3 \
    pyarrow==12.0.0 \
    fsspec==2023.5.0 \
    datasets==2.12.0

# HuggingFace stack: these versions must be mutually compatible.
pip install --force-reinstall \
    transformers==4.30.1 \
    huggingface-hub==0.15.1 \
    tokenizers==0.13.3

# wandb stack: pydantic v2 breaks wandb; protobuf 4.x required by wandb 0.15.3.
pip install --force-reinstall \
    wandb==0.15.3 \
    pydantic==1.10.8 \
    protobuf==4.23.2 \
    typing_extensions==4.6.2

# TransformerLens pinned commit — must come after torch so it picks up 2.1.0.
pip install --force-reinstall \
    git+https://github.com/neelnanda-io/TransformerLens@ae32fa54ad40cb2c3f3a60f1837d0b4899c8daae

# PyTorch re-pinned last in case TransformerLens pulled in a different version.
pip install --force-reinstall torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Step 4: Replace neuron_explainer with repo version ────────────────────────
echo "==> Replacing neuron_explainer in site-packages with repo version"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
INSTALLED_PKG="${SITE_PACKAGES}/neuron_explainer"
REPO_PKG="${REPO_ROOT}/automated-interpretability/neuron-explainer/neuron_explainer"

if [ ! -d "${REPO_PKG}" ]; then
    echo "ERROR: Repo neuron_explainer not found at ${REPO_PKG}"
    exit 1
fi

if [ ! -d "${INSTALLED_PKG}" ]; then
    echo "ERROR: Installed neuron_explainer not found at ${INSTALLED_PKG}"
    echo "       neuron-explainer may not have installed correctly."
    exit 1
fi

echo "    Removing ${INSTALLED_PKG}"
rm -rf "${INSTALLED_PKG}"
echo "    Copying ${REPO_PKG} -> ${INSTALLED_PKG}"
cp -r "${REPO_PKG}" "${INSTALLED_PKG}"

echo ""
echo "Setup complete. neuron_explainer replaced with repo version."
