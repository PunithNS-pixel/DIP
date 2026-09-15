import csv
import math
import random
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# FINAL BLIND DIP — DIP + BLIND-SPOT DETAIL REFINEMENT
# ============================================================
# Final version for the project.
#
# Why this version is different:
#   Earlier DIP versions produced clean but face-soft images because the
#   generator had to reconstruct fine details entirely from a fixed random
#   latent. V15 additionally used a Sobel/median structural loss and produced
#   severe horizontal ringing/banding.
#
# This final version separates the problem into two roles:
#   1) DIP backbone: fixed random input -> globally coherent clean image.
#   2) Blind detail refiner: sees the noisy image only through randomly hidden
#      center pixels and predicts a residual at those hidden pixels.
#
# The refiner is blind-spot/self-supervised: the pixel being predicted is NOT
# available at its own location. Therefore the clean image is never required
# for training or stopping.
#
# The final output is a restrained combination of the DIP base and learned
# blind residual. This is deliberately called a HYBRID BLIND DIP rather than
# claiming it is a pure DIP-only architecture.
#
# Selection:
#   90% noisy pixels -> training
#   10% noisy pixels -> fixed blind validation / stopping
#   clean image      -> evaluation only; oracle is retrospective only
# ============================================================

ROOT = Path("experiments") / "final_blind_dip"

PROCESS_LONG_SIDE = 512
DIP_IN_CHANNELS = 32
DIP_NOISE_SIZE = 256

SIGMAS = [0.30]
SEEDS = [42]
ITERATIONS = 6000
LEARNING_RATE = 1.0e-4

VAL_RATIO = 0.10
VAL_SMOOTH_WINDOW = 31
PATIENCE = 900
MIN_DELTA = 1e-6

# Random blind-spot mask used during training. A small percentage keeps the
# task difficult enough to force neighborhood-based reconstruction while
# leaving enough visible pixels for context.
REFINE_MASK_RATIO = 0.08
REFINE_RESIDUAL_SCALE = 0.28

TV_WEIGHT = 2e-9
GRAD_CLIP = 1.0

PRINT_EVERY = 100
SNAPSHOT_ITERS = [1, 25, 100, 250, 500, 750, 1000, 1500,
                  2000, 3000, 4000, 5000, 6000]

BASELINE_GAUSSIAN_RADIUS = 1.0
BASELINE_MEDIAN_SIZE = 3
SHARPEN_RADIUS = 1.0
SHARPEN_PERCENT = 55
SHARPEN_THRESHOLD = 4

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 78)
print(f"DEVICE: {DEVICE}")
print("FINAL BLIND DIP — DIP + BLIND-SPOT DETAIL REFINEMENT")
print("=" * 78)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path, long_side=PROCESS_LONG_SIDE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path.resolve()}\n"
            f"Put {path.name} beside main.py."
        )
    image = Image.open(path).convert("RGB")
    original_size = image.size
    w, h = image.size
    scale = long_side / max(w, h)
    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))
    image = image.resize((nw, nh), Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    print(f"Original image : {original_size}")
    print(f"Processed image: {tuple(tensor.shape)}")
    return tensor


def add_gaussian_noise(clean, sigma, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator,
                        dtype=clean.dtype, device="cpu") * sigma
    return (clean + noise).clamp(0.0, 1.0)


def tensor_to_uint8(x):
    x = x.detach().float().cpu().squeeze(0).clamp(0, 1)
    arr = x.permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def save_tensor_image(x, path):
    Image.fromarray(tensor_to_uint8(x)).save(path)


def mse(a, b):
    return F.mse_loss(a.float(), b.float()).item()


def psnr(a, b):
    value = mse(a, b)
    return 100.0 if value <= 1e-12 else 10.0 * math.log10(1.0 / value)


def ssim_torch(img1, img2):
    img1 = img1.detach().float()
    img2 = img2.detach().float()
    channels = img1.shape[1]
    k = 11
    sigma = 1.5
    coords = torch.arange(k, device=img1.device, dtype=img1.dtype) - k // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    window2d = kernel[:, None] @ kernel[None, :]
    window = window2d[None, None].expand(channels, 1, k, k)
    mu1 = F.conv2d(img1, window, padding=k // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=k // 2, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    var1 = F.conv2d(img1 * img1, window, padding=k // 2, groups=channels) - mu1_sq
    var2 = F.conv2d(img2 * img2, window, padding=k // 2, groups=channels) - mu2_sq
    cov = F.conv2d(img1 * img2, window, padding=k // 2, groups=channels) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu1_mu2 + c1) * (2 * cov + c2)) / (
        (mu1_sq + mu2_sq + c1) * (var1 + var2 + c2) + 1e-8
    )
    return score.mean().item()


def total_variation(x):
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


def masked_charbonnier(output, target, mask, eps=1e-3):
    error = torch.sqrt((output - target) ** 2 + eps ** 2)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse(output, target, mask):
    error = (output - target) ** 2
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


# ============================================================
# DIP BACKBONE
# ============================================================


class ResBlock(nn.Module):
    def __init__(self, channels, scale=0.10):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.10, inplace=True)
        self.scale = scale

    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.c2(y)
        return self.act(x + self.scale * y)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.10, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.10, inplace=True),
            ResBlock(out_ch, 0.12),
            ResBlock(out_ch, 0.12),
        )

    def forward(self, x):
        return self.body(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = Block(in_ch, out_ch)

    def forward(self, x, size):
        return self.body(F.interpolate(x, size=size, mode="bilinear", align_corners=False))


class FinalDIP(nn.Module):
    """High-resolution DIP synthesis path; no noisy image enters this network."""
    def __init__(self):
        super().__init__()
        self.e1 = Block(DIP_IN_CHANNELS, 64)
        self.e2 = Block(64, 96)
        self.e3 = Block(96, 128)
        self.e4 = Block(128, 160)
        self.e5 = Block(160, 192)
        self.pool = nn.AvgPool2d(2)
        self.b = Block(192, 192)

        self.u4 = Up(192, 160)
        self.d4 = Block(352, 160)
        self.u3 = Up(160, 128)
        self.d3 = Block(288, 128)
        self.u2 = Up(128, 96)
        self.d2 = Block(224, 96)
        self.u1 = Up(96, 64)
        self.d1 = Block(160, 64)

        self.half_fuse = nn.Sequential(
            nn.Conv2d(64 + 96, 96, 3, padding=1),
            nn.LeakyReLU(0.10, inplace=True),
            ResBlock(96, 0.08),
            ResBlock(96, 0.08),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.LeakyReLU(0.10, inplace=True),
        )
        self.full = nn.Sequential(
            nn.Conv2d(64 + 64, 96, 3, padding=1),
            nn.LeakyReLU(0.10, inplace=True),
            ResBlock(96, 0.08),
            ResBlock(96, 0.08),
            ResBlock(96, 0.08),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.LeakyReLU(0.10, inplace=True),
        )
        self.head = nn.Conv2d(64, 3, 3, padding=1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.012)
        nn.init.zeros_(self.head.bias)

    def forward(self, z, target_size):
        e1 = self.e1(z)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))
        b = self.b(self.pool(e5))

        d4 = self.d4(torch.cat([self.u4(b, e5.shape[-2:]), e5], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4, e4.shape[-2:]), e4], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3, e3.shape[-2:]), e3], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2, e2.shape[-2:]), e2], dim=1))

        half = self.half_fuse(torch.cat([d1, e2], dim=1))
        half = F.interpolate(half, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        full = self.full(torch.cat([half, e1], dim=1))
        out = torch.sigmoid(self.head(full))
        return F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)


# ============================================================
# BLIND DETAIL REFINER
# ============================================================


class DetailRefiner(nn.Module):
    """Predicts local residual from a blind-masked observation + DIP base."""
    def __init__(self):
        super().__init__()
        # 3 noisy/masked channels + 3 DIP channels + 1 mask/confidence channel.
        self.in_conv = nn.Conv2d(7, 64, 3, padding=1)
        self.body = nn.Sequential(
            nn.LeakyReLU(0.10, inplace=True),
            ResBlock(64, 0.08),
            ResBlock(64, 0.08),
            ResBlock(64, 0.08),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.10, inplace=True),
            ResBlock(64, 0.06),
        )
        self.head = nn.Conv2d(64, 3, 3, padding=1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.004)
        nn.init.zeros_(self.head.bias)

    def forward(self, masked_noisy, base, hidden_mask):
        x = torch.cat([masked_noisy, base, hidden_mask[:, :1]], dim=1)
        x = self.body(self.in_conv(x))
        return REFINE_RESIDUAL_SCALE * torch.tanh(self.head(x))


# ============================================================
# BLIND MASKING
# ============================================================


def create_train_val_masks(seed, h, w):
    rng = np.random.default_rng(seed)
    total = h * w
    val_count = int(round(total * VAL_RATIO))
    indices = rng.permutation(total)
    val_indices = indices[:val_count]
    val = np.zeros(total, dtype=np.float32)
    val[val_indices] = 1.0
    val = val.reshape(1, 1, h, w)
    train = 1.0 - val
    return torch.from_numpy(train).to(DEVICE), torch.from_numpy(val).to(DEVICE)


def random_refine_mask(seed, h, w, train_mask):
    # MPS-safe deterministic CPU RNG; only training pixels can be hidden.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    r = torch.rand((1, 1, h, w), generator=gen, device="cpu")
    mask = (r < REFINE_MASK_RATIO).float().to(DEVICE)
    mask = mask * train_mask[:, :1]
    return mask


def blind_input(noisy, hidden_mask, median_prior):
    # Replace the center pixels that the refiner must predict. Median is used
    # only as a hole-filling context value; it is not a pixel target.
    return noisy * (1.0 - hidden_mask) + median_prior * hidden_mask


def moving_average(values, window):
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


# ============================================================
# BASELINES
# ============================================================


def filter_baseline(noisy, fn):
    image = Image.fromarray(tensor_to_uint8(noisy))
    out = fn(image)
    arr = np.asarray(out).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def gaussian_baseline(noisy):
    return filter_baseline(noisy, lambda im: im.filter(ImageFilter.GaussianBlur(BASELINE_GAUSSIAN_RADIUS)))


def median_baseline(noisy):
    return filter_baseline(noisy, lambda im: im.filter(ImageFilter.MedianFilter(BASELINE_MEDIAN_SIZE)))


def sharpen_refinement(x):
    image = Image.fromarray(tensor_to_uint8(x))
    image = image.filter(ImageFilter.UnsharpMask(
        radius=SHARPEN_RADIUS,
        percent=SHARPEN_PERCENT,
        threshold=SHARPEN_THRESHOLD,
    ))
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# ============================================================
# VISUALS
# ============================================================


def save_quality_plot(history, blind_iter, oracle_iter, path):
    x = np.arange(1, len(history["psnr"]) + 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, history["psnr"], label="PSNR")
    ax.axvline(blind_iter, linestyle="--", label="Blind selected")
    ax.axvline(oracle_iter, linestyle=":", label="Oracle best")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_snapshots(snapshots, path):
    if not snapshots:
        return
    cols = 4
    rows = math.ceil(len(snapshots) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.7 * rows))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (it, out, score) in zip(axes, snapshots):
        ax.imshow(tensor_to_uint8(out))
        ax.set_title(f"Iteration {it}\nPSNR {score:.2f} dB")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gif(snapshots, path):
    if snapshots:
        imageio.mimsave(path, [tensor_to_uint8(o) for _, o, _ in snapshots], duration=0.7, loop=0)


def save_comparison(clean, noisy, gaussian, median, blind, oracle, sharp, metrics, path):
    fig, axes = plt.subplots(1, 7, figsize=(28, 5))
    images = [clean, noisy, gaussian, median, blind, oracle, sharp]
    titles = [
        "Original Clean",
        f"Noisy\nσ={metrics['sigma']:.2f}\nPSNR {metrics['noisy_psnr']:.2f}",
        f"Gaussian Blur\nPSNR {metrics['gaussian_psnr']:.2f}",
        f"Median Filter\nPSNR {metrics['median_psnr']:.2f}",
        f"Blind Hybrid DIP\nPSNR {metrics['blind_psnr']:.2f}\nSSIM {metrics['blind_ssim']:.4f}\niter {metrics['blind_iter']}",
        f"Oracle Hybrid DIP\nPSNR {metrics['oracle_psnr']:.2f}\nSSIM {metrics['oracle_ssim']:.4f}\niter {metrics['oracle_iter']}",
        f"Blind + mild sharpen\nPSNR {metrics['sharp_psnr']:.2f}\nSSIM {metrics['sharp_ssim']:.4f}",
    ]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(tensor_to_uint8(image))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_detail_crop(clean, noisy, median, blind, sharp, path):
    h, w = clean.shape[-2:]
    # Cat occupies central region for this project's test image. The crop is
    # deliberately generous so eye + ear + scarf can be compared together.
    crop_h = min(h, max(160, int(h * 0.64)))
    crop_w = min(w, max(160, int(w * 0.64)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    y1, x1 = y0 + crop_h, x0 + crop_w

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.7))
    images = [clean, noisy, median, blind, sharp]
    titles = ["Clean crop", "Noisy crop", "Median crop", "Final DIP crop", "Final + sharpen crop"]
    for ax, image, title in zip(axes, images, titles):
        arr = tensor_to_uint8(image)
        ax.imshow(arr[y0:y1, x0:x1])
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# TRAINING
# ============================================================


def run_dip(clean, noisy, seed, output_dir):
    seed_all(seed)
    clean_d = clean.to(DEVICE)
    noisy_d = noisy.to(DEVICE)
    target_size = tuple(noisy.shape[-2:])
    h, w = target_size

    train_mask, val_mask = create_train_val_masks(seed + 999, h, w)
    median_prior = median_baseline(noisy).to(DEVICE)

    dip = FinalDIP().to(DEVICE)
    refiner = DetailRefiner().to(DEVICE)
    z = torch.rand(1, DIP_IN_CHANNELS, DIP_NOISE_SIZE, DIP_NOISE_SIZE, device=DEVICE)

    with torch.no_grad():
        base0 = dip(z, target_size)
        blind0 = base0 + refiner(blind_input(noisy_d, val_mask, median_prior), base0, val_mask)
    if tuple(blind0.shape) != tuple(noisy_d.shape):
        raise RuntimeError(f"Shape mismatch: {blind0.shape} vs {noisy_d.shape}")

    params = sum(p.numel() for p in dip.parameters()) + sum(p.numel() for p in refiner.parameters())
    print(f"    total parameters: {params:,}")
    print(f"    input: {tuple(z.shape)} -> output: {tuple(blind0.shape)}")
    print("    clean image is evaluation-only")
    print("    DIP input is fixed random noise")
    print("    detail refiner uses blind center masking")

    optimizer = torch.optim.Adam(
        list(dip.parameters()) + list(refiner.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ITERATIONS, eta_min=LEARNING_RATE * 0.10
    )

    history = {"train": [], "val": [], "val_smooth": [], "psnr": [], "ssim": [], "lr": []}
    snapshots = []
    oracle_psnr = -float("inf")
    oracle_iter = 0
    oracle_output = None
    best_val = float("inf")
    blind_iter = 0
    blind_output = None
    no_improvement = 0

    for iteration in range(1, ITERATIONS + 1):
        dip.train()
        refiner.train()
        optimizer.zero_grad(set_to_none=True)

        # Fixed DIP z; no random input perturbation is needed. The blind mask
        # itself supplies the stochastic regularization for the detail path.
        base = dip(z, target_size)

        mask_seed = seed * 100000 + iteration
        hidden = random_refine_mask(mask_seed, h, w, train_mask)
        masked_noisy = blind_input(noisy_d, hidden, median_prior)
        residual = refiner(masked_noisy, base, hidden)
        output = (base + residual).clamp(0.0, 1.0)

        # The loss is evaluated only at hidden training pixels. The network
        # therefore cannot simply copy the noisy center pixel.
        refine_loss = masked_charbonnier(output, noisy_d, hidden.expand_as(noisy_d))
        # Keep the DIP backbone itself anchored to the blind 90% observation.
        # This prevents the detail refiner from carrying the entire task and
        # preserves the characteristic DIP global image prior.
        base_loss = masked_charbonnier(base, noisy_d, train_mask.expand_as(noisy_d))
        tv = total_variation(output)
        train_loss = 0.35 * base_loss + 0.65 * refine_loss
        loss = train_loss + TV_WEIGHT * tv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(dip.parameters()) + list(refiner.parameters()), GRAD_CLIP
        )
        optimizer.step()
        scheduler.step()

        # Fixed blind validation mask. This is the only signal used for blind
        # stopping. The clean target is not involved here.
        dip.eval()
        refiner.eval()
        with torch.no_grad():
            base_eval = dip(z, target_size)
            val_input = blind_input(noisy_d, val_mask, median_prior)
            val_residual = refiner(val_input, base_eval, val_mask)
            eval_out = (base_eval + val_residual).clamp(0.0, 1.0)

            val_loss = masked_mse(eval_out, noisy_d, val_mask.expand_as(noisy_d)).item()
            p = psnr(eval_out, clean_d)
            s = ssim_torch(eval_out, clean_d)

        history["train"].append(train_loss.item())
        history["val"].append(val_loss)
        history["psnr"].append(p)
        history["ssim"].append(s)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        smooth = moving_average(history["val"], VAL_SMOOTH_WINDOW)
        history["val_smooth"].append(smooth)

        if p > oracle_psnr:
            oracle_psnr = p
            oracle_iter = iteration
            oracle_output = eval_out.detach().cpu().clone()

        if smooth < best_val - MIN_DELTA:
            best_val = smooth
            blind_iter = iteration
            blind_output = eval_out.detach().cpu().clone()
            no_improvement = 0
        else:
            no_improvement += 1

        if iteration in SNAPSHOT_ITERS:
            snapshots.append((iteration, eval_out.detach().cpu().clone(), p))

        if iteration == 1 or iteration % PRINT_EVERY == 0:
            print(
                f"    {iteration:4d}/{ITERATIONS} | train {train_loss.item():.6f} | "
                f"val {val_loss:.6f} | PSNR {p:.2f} | SSIM {s:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

        if iteration >= max(800, VAL_SMOOTH_WINDOW) and no_improvement >= PATIENCE:
            print(f"    Blind early stop at iteration {iteration}.")
            break

    if blind_output is None or oracle_output is None:
        raise RuntimeError("No valid blind/oracle reconstruction was produced.")

    sharp = sharpen_refinement(blind_output)
    blind_d = blind_output.to(DEVICE)
    oracle_d = oracle_output.to(DEVICE)

    return {
        "blind_output": blind_output,
        "blind_iter": blind_iter,
        "blind_psnr": psnr(blind_d, clean_d),
        "blind_ssim": ssim_torch(blind_d, clean_d),
        "blind_val_loss": best_val,
        "oracle_output": oracle_output,
        "oracle_iter": oracle_iter,
        "oracle_psnr": psnr(oracle_d, clean_d),
        "oracle_ssim": ssim_torch(oracle_d, clean_d),
        "sharp_output": sharp,
        "sharp_psnr": psnr(sharp.to(DEVICE), clean_d),
        "sharp_ssim": ssim_torch(sharp.to(DEVICE), clean_d),
        "history": history,
        "snapshots": snapshots,
        "iterations_completed": len(history["train"]),
    }


# ============================================================
# EXPERIMENT + REPORT
# ============================================================


def run_experiment(clean, sigma, seed, output_dir):
    seed_all(seed)
    noisy = add_gaussian_noise(clean, sigma, seed + 10000)
    gaussian = gaussian_baseline(noisy)
    median = median_baseline(noisy)

    metrics = {
        "sigma": sigma,
        "seed": seed,
        "noisy_psnr": psnr(noisy, clean),
        "noisy_ssim": ssim_torch(noisy.to(DEVICE), clean.to(DEVICE)),
        "gaussian_psnr": psnr(gaussian, clean),
        "gaussian_ssim": ssim_torch(gaussian.to(DEVICE), clean.to(DEVICE)),
        "median_psnr": psnr(median, clean),
        "median_ssim": ssim_torch(median.to(DEVICE), clean.to(DEVICE)),
    }

    print("\n" + "-" * 78)
    print(f"sigma={sigma:.2f} | seed={seed}")
    print(f"Noisy    : {metrics['noisy_psnr']:.2f} dB | SSIM {metrics['noisy_ssim']:.4f}")
    print(f"Gaussian : {metrics['gaussian_psnr']:.2f} dB | SSIM {metrics['gaussian_ssim']:.4f}")
    print(f"Median   : {metrics['median_psnr']:.2f} dB | SSIM {metrics['median_ssim']:.4f}")
    print("Starting FINAL HYBRID BLIND DIP...")

    result = run_dip(clean, noisy, seed, output_dir)
    metrics.update({
        "blind_psnr": result["blind_psnr"],
        "blind_ssim": result["blind_ssim"],
        "blind_iter": result["blind_iter"],
        "blind_val_loss": result["blind_val_loss"],
        "oracle_psnr": result["oracle_psnr"],
        "oracle_ssim": result["oracle_ssim"],
        "oracle_iter": result["oracle_iter"],
        "sharp_psnr": result["sharp_psnr"],
        "sharp_ssim": result["sharp_ssim"],
        "psnr_gap": result["oracle_psnr"] - result["blind_psnr"],
        "ssim_gap": result["oracle_ssim"] - result["blind_ssim"],
        "iteration_gap": abs(result["oracle_iter"] - result["blind_iter"]),
        "iterations_completed": result["iterations_completed"],
    })

    save_tensor_image(noisy, output_dir / "noisy.png")
    save_tensor_image(gaussian, output_dir / "gaussian.png")
    save_tensor_image(median, output_dir / "median.png")
    save_tensor_image(result["blind_output"], output_dir / "blind_output.png")
    save_tensor_image(result["oracle_output"], output_dir / "oracle_best.png")
    save_tensor_image(result["blind_output"], output_dir / "stable_output.png")
    save_tensor_image(result["sharp_output"], output_dir / "blind_sharpened.png")
    save_quality_plot(result["history"], result["blind_iter"], result["oracle_iter"], output_dir / "quality.png")
    save_snapshots(result["snapshots"], output_dir / "snapshots.png")
    save_gif(result["snapshots"], output_dir / "reconstruction.gif")
    save_comparison(clean, noisy, gaussian, median,
                    result["blind_output"], result["oracle_output"], result["sharp_output"],
                    metrics, output_dir / "comparison.png")
    save_detail_crop(clean, noisy, median, result["blind_output"], result["sharp_output"],
                     output_dir / "detail_crop.png")

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(
        f"RESULT | sigma {sigma:.2f} | seed {seed} | "
        f"Blind {metrics['blind_psnr']:.2f} dB | "
        f"Oracle {metrics['oracle_psnr']:.2f} dB | "
        f"Median {metrics['median_psnr']:.2f} dB | "
        f"Gap {metrics['psnr_gap']:.2f} dB | "
        f"blind iter {metrics['blind_iter']} | oracle iter {metrics['oracle_iter']}"
    )
    return metrics


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(image_path=None):
    """Run the research/evaluation pipeline on an explicitly supplied image.

    Deployment does not use this entry point: app.py receives the user's
    uploaded image directly and calls the model utilities in this module.
    """
    if image_path is None:
        raise ValueError("Provide an image path, e.g. main(\"my_image.jpg\")")

    seed_all(SEEDS[0])
    ROOT.mkdir(parents=True, exist_ok=True)
    clean = load_image(image_path)

    (ROOT / "README.txt").write_text(
        "FINAL HYBRID BLIND DIP\n"
        "DIP synthesis backbone + blind-spot residual detail refinement.\n"
        "90% noisy pixels are used for blind training and 10% for blind stopping.\n"
        "The clean image is evaluation-only; oracle selection is retrospective.\n"
        "The detail refiner never receives the center pixel it is asked to predict.\n"
        "Median filtering is retained as a classical baseline, not used as a pixel target.\n",
        encoding="utf-8"
    )

    rows = []
    for sigma in SIGMAS:
        for seed in SEEDS:
            out = ROOT / f"sigma_{sigma:.2f}" / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            rows.append(run_experiment(clean, sigma, seed, out))

    write_csv(rows, ROOT / "all_runs.csv")
    print("\n" + "=" * 78)
    print("FINAL EXPERIMENT COMPLETE")
    print(f"Results directory: {ROOT.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Final Hybrid Blind DIP research pipeline")
    parser.add_argument(
        "image",
        help="Path to the evaluation image used by the research pipeline"
    )
    args = parser.parse_args()
    main(args.image)
