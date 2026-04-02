#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing system dependencies"
sudo apt-get update -qq
sudo apt-get install -y libxml2-dev libxslt-dev python3-dev

echo "==> Installing PyTorch 2.0.1 with CUDA 11.8"
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo "==> Installing core ML/data packages"
pip install \
    numpy==1.24.3 \
    scipy==1.10.1 \
    scikit-learn==1.2.2

echo "==> Installing Transformers stack"
pip install \
    transformers==4.30.1 \
    tokenizers==0.13.3 \
    huggingface-hub==0.15.1 \
    safetensors==0.3.1

echo "==> Installing TransformerLens (pinned commit)"
pip install git+https://github.com/neelnanda-io/TransformerLens@ae32fa54ad40cb2c3f3a60f1837d0b4899c8daae

echo "==> Installing SAE / interpretability packages"
pip install \
    einops==0.6.1 \
    fancy-einsum==0.0.3 \
    jaxtyping==0.2.19

echo "==> Installing dataset and tokenization packages"
pip install \
    datasets==2.12.0 \
    tiktoken==0.4.0

echo "==> Installing plotting and analysis packages"
pip install \
    matplotlib==3.7.1 \
    seaborn \
    pandas==2.0.2 \
    plotly==5.14.1

echo "==> Installing lxml"
pip install lxml==4.9.2

echo "==> Installing neuron-explainer from upstream"
pip install git+https://github.com/openai/automated-interpretability.git#subdirectory=neuron-explainer

echo "==> Installing misc packages"
pip install tqdm wandb PyYAML

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
