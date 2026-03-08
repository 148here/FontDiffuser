"""
Single evaluation script for FontDiffuser.
Content: one image path
Style: one image path, OR style_char + style_folder for single-char mode
Samples without GT are skipped (for save_compare).
"""
import os
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from sample import load_fontdiffuer_pipeline
from configs.fontdiffuser import get_parser

# ============ Config (edit these) ============
CONTENT_PATH = "data_examples/sampling/example_content.jpg"
STYLE_PATH = "data_examples/sampling/example_style.jpg"
# Single-char style mode: set STYLE_CHAR and STYLE_FOLDER, leave STYLE_PATH as None
STYLE_CHAR = None   # e.g. "曦"
STYLE_FOLDER = None # e.g. "data_examples/train/TargetImage/style0"
# When using STYLE_CHAR, STYLE_FOLDER must point to the style subdir: TargetImage/{style}/
# and we look for {style}+{char}.png/.jpg inside it

CKPT_DIR = "ckpt"
OUTPUT_DIR = "outputs/eval_single"
SAVE_RESULT = True
SAVE_COMPARE = False
GT_PATH = None  # Optional: if provided and SAVE_COMPARE, use this as GT; otherwise skip compare
SEED = 123
DEVICE = "cuda:0"
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 20
RESOLUTION = 96
# ============================================


def _find_file(style_dir, style_name, content_name, extensions=(".png", ".jpg")):
    for ext in extensions:
        path = os.path.join(style_dir, f"{style_name}+{content_name}{ext}")
        if os.path.isfile(path):
            return path
    return None


def resolve_style_path():
    """Resolve style image path from STYLE_PATH or STYLE_CHAR + STYLE_FOLDER."""
    if STYLE_PATH is not None and os.path.isfile(STYLE_PATH):
        return STYLE_PATH
    if STYLE_CHAR and STYLE_FOLDER and os.path.isdir(STYLE_FOLDER):
        style_name = os.path.basename(STYLE_FOLDER.rstrip(os.sep))
        path = _find_file(STYLE_FOLDER, style_name, STYLE_CHAR)
        if path:
            return path
    return None


def concat_three_images(result_pil, gt_pil, style_pil, resolution):
    w = resolution
    new = Image.new("RGB", (w * 3, w))
    result_pil = result_pil.resize((w, w), Image.BILINEAR)
    gt_pil = gt_pil.resize((w, w), Image.BILINEAR)
    style_pil = style_pil.resize((w, w), Image.BILINEAR)
    new.paste(result_pil, (0, 0))
    new.paste(gt_pil, (w, 0))
    new.paste(style_pil, (w * 2, 0))
    return new


def run():
    from accelerate.utils import set_seed
    import torchvision.transforms as transforms

    set_seed(SEED)

    style_path = resolve_style_path()
    if style_path is None:
        print("Error: could not resolve style image. Set STYLE_PATH or STYLE_CHAR+STYLE_FOLDER.")
        return

    if not os.path.isfile(CONTENT_PATH):
        print(f"Error: content path not found: {CONTENT_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build args
    parser = get_parser()
    args = parser.parse_args([])
    args.ckpt_dir = "ckpt"  # default, override in config if needed
    args.device = DEVICE
    args.guidance_scale = GUIDANCE_SCALE
    args.num_inference_steps = NUM_INFERENCE_STEPS
    args.resolution = RESOLUTION
    args.seed = SEED
    content_size = args.content_image_size
    style_size = args.style_image_size
    args.style_image_size = (style_size, style_size)
    args.content_image_size = (content_size, content_size)

    args.ckpt_dir = CKPT_DIR

    print("Loading pipeline...")
    pipe = load_fontdiffuer_pipeline(args=args)

    content_tf = transforms.Compose([
        transforms.Resize((content_size, content_size),
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    style_tf = transforms.Compose([
        transforms.Resize((style_size, style_size),
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    content_img = Image.open(CONTENT_PATH).convert("RGB")
    style_img = Image.open(style_path).convert("RGB")
    content_tensor = content_tf(content_img)[None, :].to(args.device)
    style_tensor = style_tf(style_img)[None, :].to(args.device)

    with torch.no_grad():
        images = pipe.generate(
            content_images=content_tensor,
            style_images=style_tensor,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn,
        )

    result_pil = images[0]
    content_name = Path(CONTENT_PATH).stem
    style_name = Path(style_path).stem.replace("+", "_")
    out_name = f"{content_name}_{style_name}"

    if SAVE_RESULT:
        single_path = os.path.join(OUTPUT_DIR, f"{out_name}.png")
        result_pil.save(single_path)
        print(f"Saved: {single_path}")

    if SAVE_COMPARE:
        gt_path = GT_PATH
        if gt_path is None or not os.path.isfile(gt_path):
            print("SAVE_COMPARE=True but no valid GT_PATH. Skipping compare output.")
        else:
            out_subdir = os.path.join(OUTPUT_DIR, out_name)
            os.makedirs(out_subdir, exist_ok=True)
            result_pil.save(os.path.join(out_subdir, "result.png"))
            style_img.save(os.path.join(out_subdir, "style_input.png"))
            gt_pil = Image.open(gt_path).convert("RGB")
            gt_pil.save(os.path.join(out_subdir, "gt.png"))
            concat = concat_three_images(result_pil, gt_pil, style_img, RESOLUTION)
            concat.save(os.path.join(out_subdir, "concat.png"))
            print(f"Saved compare: {out_subdir}")

    print("Done.")


if __name__ == "__main__":
    run()
