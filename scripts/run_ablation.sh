#!/bin/bash
# FontDiffuser Ablation Experiment Runner
# Edit the variables below, then run: bash run_ablation.sh

# ============================================================
# Path configuration (edit these)
# ============================================================
CKPT_DIR="/root/workspace/ckpt/ckpt"
CONTENT_IMAGE="/root/autodl-tmp/data/test/ContentImage/源.jpg"
STYLE_IMAGE="/root/autodl-tmp/data/test/TargetImage/HYChangYiTiF/HYChangYiTiF+阿.jpg"
GT_IMAGE="/root/autodl-tmp/data/test/TargetImage/HYChangYiTiF/HYChangYiTiF+源.jpg"  # Target font reference (ground truth)
OUTPUT_BASE="/root/autodl-tmp/output/ablation_layers/test1/xi/final_set_01"
DEVICE="cuda:0"
NUM_RUNS=8
SEED=42

# ============================================================
# Ablation scale arrays
# content_scales: length 3  [Down1, Down2, Mid]
# style_scales:   length 5  [Down1, Down2, Mid, Up1, Up2]
# 0 = disable injection, 1 = default strength, 0.x / 2.0 = scaled
# ============================================================

# Experiment (1): Content ablation (enable content only at one position, disable all style)
# "Only inject at position 0", "only at 1", "only at 2"
CONTENT_CONFIGS=("1,1,1" "0,1,1" "1,0,1" "1,1,0")
STYLE_DISABLED="1,1,1,1,1"
CONTENT_ABLATION_NAMES=("noraml" "pos0_down1" "pos1_down2" "pos2_mid")

# Experiment (2): Style ablation (enable style only at one position, disable all content)
STYLE_CONFIGS=("1,1,1,1,1" "0,1,1,1,1" "1,0,1,1,1" "1,1,0,1,1" "1,1,1,0,1" "1,1,1,1,0")
CONTENT_DISABLED="1,1,1"
STYLE_ABLATION_NAMES=("normal" "pos0_down1" "pos1_down2" "pos2_mid" "pos3_up1" "pos4_up2")

# Sampling / model args (optional overrides)
NUM_INFERENCE_STEPS=20
ORDER=2
GUIDANCE_SCALE=7.5

# Project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

run_one() {
    local out_dir="$1"
    local content_scales="$2"
    local style_scales="$3"
    python scripts/ablation_inference.py \
        --ckpt_dir="$CKPT_DIR" \
        --content_image_path="$CONTENT_IMAGE" \
        --style_image_path="$STYLE_IMAGE" \
        --gt_image_path="$GT_IMAGE" \
        --output_dir="$out_dir" \
        --content_scales="$content_scales" \
        --style_scales="$style_scales" \
        --num_runs="$NUM_RUNS" \
        --device="$DEVICE" \
        --seed="$SEED" \
        --num_inference_steps="$NUM_INFERENCE_STEPS" \
        --order="$ORDER" \
        --guidance_scale="$GUIDANCE_SCALE"
}

echo "=== Content ablation (style disabled) ==="
for i in "${!CONTENT_CONFIGS[@]}"; do
    out="$OUTPUT_BASE/content_ablation/${CONTENT_ABLATION_NAMES[$i]}"
    echo "Running: content=${CONTENT_CONFIGS[$i]}, style=$STYLE_DISABLED -> $out"
    run_one "$out" "${CONTENT_CONFIGS[$i]}" "$STYLE_DISABLED"
done

echo "=== Style ablation (content disabled) ==="
for i in "${!STYLE_CONFIGS[@]}"; do
    out="$OUTPUT_BASE/style_ablation/${STYLE_ABLATION_NAMES[$i]}"
    echo "Running: content=$CONTENT_DISABLED, style=${STYLE_CONFIGS[$i]} -> $out"
    run_one "$out" "$CONTENT_DISABLED" "${STYLE_CONFIGS[$i]}"
done

echo "=== Ablation complete. Results in $OUTPUT_BASE ==="
