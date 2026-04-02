#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Configurable parameters ────────────────────────────────────────────────────
MODEL="EleutherAI/pythia-2.8b"
LAYER=2
LAYER_LOC_COLLECT="residual"   # used by pipeline.py (activation collection)
LAYER_LOC_PATCH="resid"        # used by patching scripts (TransformerLens hook name)
RANK=6                         # rank 6 ≈ l1_alpha=0.0027 (paper default)
DICT_RATIOS="4 8"
N_CHUNKS=16
N_EPOCHS=5
BATCH_SIZE=1024
MAX_SAMPLES=2000
TOPK=20
DEVICE="cuda:0"

# Derived paths
SAE_DIR="sae"
RESULTS_DIR="results"

# ── Step 1: Collect activations + train SAEs ───────────────────────────────────
echo "========================================================"
echo "STEP 1: Collect activations and train SAEs (Pythia-2.8B)"
echo "========================================================"

python pipeline.py \
    --model        "$MODEL" \
    --layer        "$LAYER" \
    --layer_loc    "$LAYER_LOC_COLLECT" \
    --dict_ratios  $DICT_RATIOS \
    --n_chunks     "$N_CHUNKS" \
    --n_epochs     "$N_EPOCHS" \
    --batch_size   "$BATCH_SIZE" \
    --device       "$DEVICE"

# ── Step 2 & 3: Patching analysis for each dict_ratio ─────────────────────────
for RATIO in $DICT_RATIOS; do
    COT_DICT="${SAE_DIR}/cot_sae_l${LAYER}_r${RATIO}/learned_dicts.pt"
    NOCOT_DICT="${SAE_DIR}/nocot_sae_l${LAYER}_r${RATIO}/learned_dicts.pt"
    COT_ACTS="${SAE_DIR}/cot_acts_l${LAYER}_${LAYER_LOC_COLLECT}"
    NOCOT_ACTS="${SAE_DIR}/nocot_acts_l${LAYER}_${LAYER_LOC_COLLECT}"

    echo ""
    echo "========================================================"
    echo "STEP 2: Activation patching  (dict_ratio=${RATIO})"
    echo "========================================================"

    python activation_patching.py \
        --model          "$MODEL" \
        --layer          "$LAYER" \
        --layer_loc      "$LAYER_LOC_PATCH" \
        --rank           "$RANK" \
        --dict_cot       "$COT_DICT" \
        --dict_nocot     "$NOCOT_DICT" \
        --acts_cot_dir   "$COT_ACTS" \
        --acts_nocot_dir "$NOCOT_ACTS" \
        --topk           "$TOPK" \
        --max_samples    "$MAX_SAMPLES" \
        --out            "${RESULTS_DIR}/patch_hist_2.8B_r${RATIO}.png"

    echo ""
    echo "========================================================"
    echo "STEP 3: Patch curve sweep  (dict_ratio=${RATIO})"
    echo "========================================================"

    python patch_curve.py \
        --model          "$MODEL" \
        --layer          "$LAYER" \
        --layer_loc      "$LAYER_LOC_PATCH" \
        --rank           "$RANK" \
        --dict_cot       "$COT_DICT" \
        --dict_nocot     "$NOCOT_DICT" \
        --acts_cot_dir   "$COT_ACTS" \
        --acts_nocot_dir "$NOCOT_ACTS" \
        --max_samples    "$MAX_SAMPLES" \
        --out            "${RESULTS_DIR}/patch_curve_2.8B_r${RATIO}.png" \
        --save_stats     "${RESULTS_DIR}/ttest_2.8B_r${RATIO}.txt"
done

echo ""
echo "========================================================"
echo "All done. Results saved to ${RESULTS_DIR}/"
echo "========================================================"
