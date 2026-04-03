#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing system dependencies"
apt-get update -qq
apt-get install -y libxml2-dev libxslt-dev python3-dev

# ── Pin order-sensitive packages before requirements.txt ──────────────────────
# These must be installed first so pip's resolver sees them before pulling
# newer incompatible versions as transitive dependencies.

echo "==> Installing PyTorch 2.0.1 with CUDA 11.8"
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo "==> Pinning pyarrow before datasets (newer pyarrow breaks datasets==2.12.0)"
pip install pyarrow==12.0.0

echo "==> Installing numpy pin (1.25+ breaks several downstream packages)"
pip install numpy==1.24.3

# ── Install everything else from requirements.txt ─────────────────────────────
# Skips: cmake (commented out), nvidia-* (provided by CUDA image),
#        torch/torchvision (already installed above).

echo "==> Installing remaining dependencies from requirements.txt"
pip install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r "${REPO_ROOT}/sparse_coding/requirements.txt" \
    --ignore-installed torch torchvision \
    || true   # non-zero exit from torchaudio conflict warning is harmless

# ── Replace neuron_explainer with repo version ────────────────────────────────
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
