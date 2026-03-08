"""
Ablation inference script for FontDiffuser.
Runs inference with configurable content/style injection scales, saves images,
and computes content/style similarity vs. ground truth.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.transforms as transforms

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_content_encoder,
    build_ddpm_scheduler,
    build_style_encoder,
    build_unet,
)


def parse_scales(s: str, length: int):
    """Parse comma-separated scale string to tuple of floats.
    Example: "1,0,0" -> (1.0, 0.0, 0.0)
    """
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != length:
        raise ValueError(f"Expected {length} values, got {len(parts)}")
    return tuple(parts)


def load_and_preprocess_image(path, size, device):
    """Load image and apply inference transforms."""
    img = Image.open(path).convert("RGB")
    trans = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return trans(img)[None, :].to(device)


def compute_content_similarity(content_encoder, img_a, img_b):
    """Compute content feature similarity (cosine, L2) between two images."""
    with torch.no_grad():
        feat_a, _ = content_encoder(img_a)
        feat_b, _ = content_encoder(img_b)
    fa = feat_a.flatten(1)
    fb = feat_b.flatten(1)
    cos = F.cosine_similarity(fa, fb, dim=1).item()
    l2 = (fa - fb).norm(p=2, dim=1).item()
    return cos, l2


def compute_style_similarity(style_encoder, img_a, img_b):
    """Compute style feature similarity (cosine, L2) between two images."""
    with torch.no_grad():
        _, h_a, _ = style_encoder(img_a)
        _, h_b, _ = style_encoder(img_b)
    cos = F.cosine_similarity(h_a, h_b, dim=1).item()
    l2 = (h_a - h_b).norm(p=2, dim=1).item()
    return cos, l2


def run_ablation(args):
    from accelerate.utils import set_seed

    if args.seed is not None:
        set_seed(args.seed)

    # Parse scales
    content_scales = parse_scales(args.content_scales, 3)
    style_scales = parse_scales(args.style_scales, 5)

    # Build args for model construction (match configs)
    class BuildArgs:
        pass

    build_args = BuildArgs()
    build_args.resolution = getattr(args, "resolution", 96)
    build_args.unet_channels = getattr(args, "unet_channels", (64, 128, 256, 512))
    build_args.style_image_size = (getattr(args, "style_image_size", 96),)
    build_args.content_image_size = (getattr(args, "content_image_size", 96),)
    build_args.content_encoder_downsample_size = getattr(args, "content_encoder_downsample_size", 3)
    build_args.channel_attn = getattr(args, "channel_attn", True)
    build_args.content_start_channel = getattr(args, "content_start_channel", 64)
    build_args.style_start_channel = getattr(args, "style_start_channel", 64)

    # Load model and pipeline
    unet = build_unet(build_args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location="cpu"))
    style_encoder = build_style_encoder(build_args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location="cpu"))
    content_encoder = build_content_encoder(build_args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location="cpu"))
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
    )
    model.to(args.device)
    model.eval()

    train_scheduler = build_ddpm_scheduler(args)
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=getattr(args, "model_type", "noise"),
        guidance_type=getattr(args, "guidance_type", "classifier-free"),
        guidance_scale=getattr(args, "guidance_scale", 7.5),
    )
    pipe.model.to(args.device)

    size = getattr(args, "content_image_size", 96)
    if isinstance(size, tuple):
        size = size[0]

    content_img = load_and_preprocess_image(args.content_image_path, size, args.device)
    style_img = load_and_preprocess_image(args.style_image_path, size, args.device)
    gt_img = load_and_preprocess_image(args.gt_image_path, size, args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    results_content_cos = []
    results_content_l2 = []
    results_style_cos = []
    results_style_l2 = []

    for run_idx in range(args.num_runs):
        with torch.no_grad():
            imgs = pipe.generate(
                content_images=content_img,
                style_images=style_img,
                batch_size=1,
                order=getattr(args, "order", 2),
                num_inference_step=getattr(args, "num_inference_steps", 20),
                content_encoder_downsample_size=build_args.content_encoder_downsample_size,
                dm_size=(size, size),
                content_scales=content_scales,
                style_scales=style_scales,
                generator=torch.manual_seed(args.seed + run_idx) if args.seed is not None else None,
            )
        pil_img = imgs[0]
        out_path = os.path.join(args.output_dir, f"run_{run_idx:03d}.png")
        pil_img.save(out_path)

        # Convert PIL to tensor for similarity (same transforms as input)
        arr = transforms.functional.to_tensor(pil_img.convert("RGB"))
        arr = transforms.functional.normalize(arr, [0.5], [0.5])
        gen_tensor = arr[None, :].to(args.device)

        cc, cl2 = compute_content_similarity(model.content_encoder, gen_tensor, gt_img)
        sc, sl2 = compute_style_similarity(model.style_encoder, gen_tensor, gt_img)

        results_content_cos.append(cc)
        results_content_l2.append(cl2)
        results_style_cos.append(sc)
        results_style_l2.append(sl2)

    # Write similarity.txt
    lines = [
        f"=== Ablation: content={list(content_scales)} | style={list(style_scales)} ===",
        f"GT Image: {args.gt_image_path}",
        "Generated via: ContentEncoder(final feature) & StyleEncoder(global avg pool)",
        "",
    ]
    for i in range(args.num_runs):
        lines.extend([
            f"--- Run {i} ---",
            f"  Content Cosine Similarity (vs GT):  {results_content_cos[i]:.6f}",
            f"  Content L2 Distance       (vs GT):  {results_content_l2[i]:.6f}",
            f"  Style   Cosine Similarity (vs GT):  {results_style_cos[i]:.6f}",
            f"  Style   L2 Distance       (vs GT):  {results_style_l2[i]:.6f}",
            "",
        ])
    import statistics
    lines.extend([
        "=== Mean over {} runs ===".format(args.num_runs),
        f"  Content Cosine: {statistics.mean(results_content_cos):.4f} ± {statistics.stdev(results_content_cos):.4f}" if args.num_runs > 1 else f"  Content Cosine: {results_content_cos[0]:.4f}",
        f"  Content L2:     {statistics.mean(results_content_l2):.4f} ± {statistics.stdev(results_content_l2):.4f}" if args.num_runs > 1 else f"  Content L2:     {results_content_l2[0]:.4f}",
        f"  Style   Cosine: {statistics.mean(results_style_cos):.4f} ± {statistics.stdev(results_style_cos):.4f}" if args.num_runs > 1 else f"  Style   Cosine: {results_style_cos[0]:.4f}",
        f"  Style   L2:     {statistics.mean(results_style_l2):.4f} ± {statistics.stdev(results_style_l2):.4f}" if args.num_runs > 1 else f"  Style   L2:     {results_style_l2[0]:.4f}",
    ])
    txt_path = os.path.join(args.output_dir, "similarity.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def main():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--content_image_path", type=str, required=True)
    parser.add_argument("--style_image_path", type=str, required=True)
    parser.add_argument("--gt_image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--content_scales", type=str, required=True,
                        help="Comma-separated, length 3: [Down1, Down2, Mid]. 0=disable, 1=full.")
    parser.add_argument("--style_scales", type=str, required=True,
                        help="Comma-separated, length 5: [Down1, Down2, Mid, Up1, Up2]. 0=disable, 1=full.")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
