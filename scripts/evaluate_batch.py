"""
Batch evaluation script for FontDiffuser.
Content: all images from ContentImage/
Style: all styles from TargetImage/ (random pick per style, or fixed char if STYLE_CHAR set)
Samples without GT are skipped.
"""
import os
import sys
import random
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms as transforms
from PIL import Image
from accelerate.utils import set_seed

from sample import load_fontdiffuer_pipeline
from configs.fontdiffuser import get_parser

# ============ Config (edit these) ============
DATA_ROOT = "data_examples/train"
CKPT_DIR = "ckpt"
OUTPUT_DIR = "outputs/eval_batch"
INFERENCE_BATCH_SIZE = 4
SAVE_RESULT = True
SAVE_COMPARE = True
STYLE_CHAR = None  # e.g. "曦" for single-char style mode; None for random pick
SEED = 123
DEVICE = "cuda:0"
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 20
RESOLUTION = 96
# ============================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _find_file(style_dir, style_name, content_name, extensions=(".png", ".jpg")):
    """Find file {style}+{content}.ext in style_dir. Try .png first, then .jpg."""
    for ext in extensions:
        path = os.path.join(style_dir, f"{style_name}+{content_name}{ext}")
        if os.path.isfile(path):
            return path
    return None


def find_gt_path(style_dir, style_name, content_name):
    return _find_file(style_dir, style_name, content_name)


def find_style_char_path(style_dir, style_name, char):
    return _find_file(style_dir, style_name, char)


def collect_content_images(content_dir):
    """Collect (content_name, content_path) from ContentImage/."""
    items = []
    if not os.path.isdir(content_dir):
        return items
    for f in os.listdir(content_dir):
        p = Path(f)
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            items.append((p.stem, os.path.join(content_dir, f)))
    return items


def collect_styles(target_dir):
    """Collect style names (subdirs) from TargetImage/."""
    if not os.path.isdir(target_dir):
        return []
    return [d for d in os.listdir(target_dir)
            if os.path.isdir(os.path.join(target_dir, d))]


def build_tasks(data_root, style_char):
    """
    Build list of (content_name, content_path, style_name, style_path, gt_path).
    Skip samples without GT.
    """
    content_dir = os.path.join(data_root, "ContentImage")
    target_dir = os.path.join(data_root, "TargetImage")
    contents = collect_content_images(content_dir)
    styles = collect_styles(target_dir)
    if not contents or not styles:
        return []

    tasks = []
    for content_name, content_path in contents:
        for style_name in styles:
            style_dir = os.path.join(target_dir, style_name)
            gt_path = find_gt_path(style_dir, style_name, content_name)
            if gt_path is None:
                continue  # Skip without GT

            if style_char is not None:
                style_path = find_style_char_path(style_dir, style_name, style_char)
                if style_path is None:
                    continue  # Skip if this style has no char
            else:
                # Random pick one image from style dir (excluding gt if we want diff ref)
                files = [f for f in os.listdir(style_dir)
                         if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
                if not files:
                    continue
                style_path = os.path.join(style_dir, random.choice(files))

            tasks.append((content_name, content_path, style_name, style_path, gt_path))

    return tasks


def get_transforms(content_size, style_size):
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
    return content_tf, style_tf


def concat_three_images(result_pil, gt_pil, style_pil, resolution):
    """Horizontal concat: result | gt | style."""
    from PIL import Image
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
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build args for pipeline
    parser = get_parser()
    args = parser.parse_args([])
    args.ckpt_dir = CKPT_DIR
    args.device = DEVICE
    args.guidance_scale = GUIDANCE_SCALE
    args.num_inference_steps = NUM_INFERENCE_STEPS
    args.resolution = RESOLUTION
    args.seed = SEED
    style_size = args.style_image_size
    content_size = args.content_image_size
    args.style_image_size = (style_size, style_size)
    args.content_image_size = (content_size, content_size)

    # Load pipeline
    print("Loading pipeline...")
    pipe = load_fontdiffuer_pipeline(args=args)
    content_tf, style_tf = get_transforms(content_size, style_size)

    # Build tasks
    tasks = build_tasks(DATA_ROOT, STYLE_CHAR)
    print(f"Total tasks (with GT): {len(tasks)}")
    if not tasks:
        print("No tasks to run. Exiting.")
        return

    # Process in batches
    for i in range(0, len(tasks), INFERENCE_BATCH_SIZE):
        batch = tasks[i:i + INFERENCE_BATCH_SIZE]
        content_tensors = []
        style_tensors = []
        for content_name, content_path, style_name, style_path, gt_path in batch:
            content_img = Image.open(content_path).convert("RGB")
            style_img = Image.open(style_path).convert("RGB")
            ct = content_tf(content_img)[None, :]
            st = style_tf(style_img)[None, :]
            content_tensors.append(ct)
            style_tensors.append(st)

        content_batch = torch.cat(content_tensors, dim=0).to(args.device)
        style_batch = torch.cat(style_tensors, dim=0).to(args.device)

        with torch.no_grad():
            images = pipe.generate(
                content_images=content_batch,
                style_images=style_batch,
                batch_size=len(batch),
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

        for j, (content_name, content_path, style_name, style_path, gt_path) in enumerate(batch):
            result_pil = images[j]
            out_name = f"{content_name}_{style_name}"

            if SAVE_RESULT:
                single_path = os.path.join(OUTPUT_DIR, f"{out_name}.png")
                result_pil.save(single_path)

            if SAVE_COMPARE:
                out_subdir = os.path.join(OUTPUT_DIR, out_name)
                os.makedirs(out_subdir, exist_ok=True)
                result_pil.save(os.path.join(out_subdir, "result.png"))
                style_pil = Image.open(style_path).convert("RGB")
                gt_pil = Image.open(gt_path).convert("RGB")
                style_pil.save(os.path.join(out_subdir, "style_input.png"))
                gt_pil.save(os.path.join(out_subdir, "gt.png"))
                concat = concat_three_images(result_pil, gt_pil, style_pil, RESOLUTION)
                concat.save(os.path.join(out_subdir, "concat.png"))

        print(f"Processed {min(i + INFERENCE_BATCH_SIZE, len(tasks))}/{len(tasks)}")

    print("Done.")


if __name__ == "__main__":
    run()
