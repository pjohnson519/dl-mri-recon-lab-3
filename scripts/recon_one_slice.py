"""Lab 3 helper: run one score-MRI reconstruction from the command line.

This is a thin CLI wrapper around Algorithm 5 (hybrid per-coil sampler). The
notebook calls the same functions in-process; this script exists mainly for
debugging / batch runs outside the notebook.

Usage:
    python scripts/recon_one_slice.py \
        --ckpt  /gpfs/scratch/.../score_magnitude_knee320_ep95.pth \
        --slice_dir /gpfs/scratch/.../lab3_demo/pd_file1000073 \
        --slice_idx 1 \
        --mode cold --N 500

Args:
    --ckpt        Path to the pretrained magnitude prior (.pth).
    --slice_dir   Directory holding kspace.npy, maps.npy, target.npy, meta.json.
    --slice_idx   Which slice (0-based index within the prepared volume).
    --mode        "cold" (paper default) or "warm" (faster; uses zero-filled init).
    --N           Number of PC steps (500 = paper default; 1000 = diminishing return).
    --out_dir     Where to save recon.npy, label.npy, zf.npy (default: slice_dir/recon_out).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import sigpy.mri as mr
from skimage.metrics import structural_similarity as skssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr

# Make score_mri importable regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "score_mri"))

from configs.ve.fastmri_knee_320_ncsnpp_continuous import get_config
from models import ncsnpp  # noqa: F401  (registers model class)
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from sampling import (ReverseDiffusionPredictor, LangevinCorrector,
                      get_pc_fouriercs_RI_coil_SENSE)
from utils import (fft2_m, ifft2_m, get_data_scaler, get_data_inverse_scaler,
                   restore_checkpoint, normalize_complex, root_sum_of_squares,
                   lambda_schedule_linear)
from hybrid_sampler_warm import get_pc_fouriercs_RI_coil_SENSE_warm


def ifft2c_np(x):
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def gaussian1d_mask(size, acc_factor, center_fraction, seed=0):
    """Reproduce the repo's gaussian1d mask: dense ACS + gaussian-drawn outer columns.

    Returns a (size, size) float32 mask (all rows identical -> column mask).
    """
    rng = np.random.RandomState(seed)
    mask1d = np.zeros(size, dtype=np.float32)
    Nsamp_center = int(size * center_fraction)
    c_from = size // 2 - Nsamp_center // 2
    mask1d[c_from:c_from + Nsamp_center] = 1.0
    Nsamp = size // acc_factor
    samples = rng.normal(loc=size // 2, scale=size * (15.0 / 128),
                         size=int(Nsamp * 1.2))
    int_samples = np.clip(samples.astype(int), 0, size - 1)
    mask1d[int_samples] = 1.0
    return np.tile(mask1d[None, :], (size, 1)).astype(np.float32)


def run_recon(ckpt_path, slice_dir, slice_idx, mode, N,
              warm_sigma=10.0, acc_factor=4, center_fraction=0.08,
              snr=0.16, m_steps=50, lamb_start=1.0, lamb_end=0.2,
              save_filmstrip=False, n_snapshots=10):
    """Run one reconstruction. Returns (recon_n, label_n, zf_n, ssim, psnr, seconds)."""
    device = torch.device("cuda:0")

    cfg = get_config()
    cfg.device = device
    cfg.training.batch_size = 1
    sde = sde_lib.VESDE(sigma_min=cfg.model.sigma_min,
                        sigma_max=cfg.model.sigma_max, N=N)

    score_model = mutils.create_model(cfg)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=cfg.model.ema_rate)
    state = {"step": 0, "model": score_model, "ema": ema}
    state = restore_checkpoint(str(ckpt_path), state, device, skip_sigma=True)
    ema.copy_to(score_model.parameters())
    score_model.eval()

    scaler = get_data_scaler(cfg)
    inverse_scaler = get_data_inverse_scaler(cfg)
    lamb_schedule = lambda_schedule_linear(start_lamb=lamb_start, end_lamb=lamb_end)

    slice_dir = Path(slice_dir)
    kspace_np = np.load(slice_dir / "kspace.npy")[slice_idx]
    H, W = kspace_np.shape[-2:]

    # Label = RSS of fully-sampled coil images
    full_coils = ifft2c_np(kspace_np).astype(np.complex64)
    label_rss = np.sqrt(np.sum(np.abs(full_coils) ** 2, axis=0)).astype(np.float32)
    scale = float(label_rss.max())
    label_n = np.clip(label_rss / scale, 0, 1)

    # Mask + ZF baseline
    mask_np = gaussian1d_mask(W, acc_factor, center_fraction)
    zf_coils = ifft2c_np(kspace_np * mask_np).astype(np.complex64)
    zf_n = np.clip(np.sqrt(np.sum(np.abs(zf_coils) ** 2, axis=0)) / scale, 0, 1)
    mask_t = torch.from_numpy(mask_np).view(1, 1, H, W).to(device)

    # Sampler input: paper-style normalize_complex on coil images
    img = normalize_complex(torch.from_numpy(full_coils)).view(1, 15, H, W).to(device)
    k = fft2_m(img)
    under_k = k * mask_t
    under_img = ifft2_m(under_k)
    mps = mr.app.EspiritCalib(k.cpu().detach().squeeze().numpy(),
                              show_pbar=False).run()
    mps_t = torch.from_numpy(mps).view(1, 15, H, W).to(device)

    if mode == "cold":
        sampler = get_pc_fouriercs_RI_coil_SENSE(
            sde, ReverseDiffusionPredictor, LangevinCorrector, inverse_scaler,
            snr=snr, n_steps=1, m_steps=m_steps, mask=mask_t, sens=mps_t,
            lamb_schedule=lamb_schedule, probability_flow=False,
            continuous=cfg.training.continuous, denoise=True,
        )
    elif mode == "warm":
        sampler = get_pc_fouriercs_RI_coil_SENSE_warm(
            sde, ReverseDiffusionPredictor, LangevinCorrector, inverse_scaler,
            snr=snr, n_steps=1, m_steps=m_steps, mask=mask_t, sens=mps_t,
            lamb_schedule=lamb_schedule, probability_flow=False,
            continuous=cfg.training.continuous, denoise=True,
            warm_start=True, warm_sigma=warm_sigma,
        )
    else:
        raise ValueError(f"mode must be 'cold' or 'warm', got {mode!r}")

    tic = time.time()
    x = sampler(score_model, scaler(under_img), y=under_k)
    dt = time.time() - tic

    recon = np.abs(root_sum_of_squares(x, dim=1).squeeze().cpu().detach().numpy())
    recon_n = np.clip(recon / (recon.max() or 1.0), 0, 1)
    ssim = skssim(label_n, recon_n, data_range=1.0)
    psnr = skpsnr(label_n, recon_n, data_range=1.0)
    return recon_n, label_n, zf_n, mask_np, ssim, psnr, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to score_magnitude_knee320_ep95.pth")
    ap.add_argument("--slice_dir", required=True, help="dir with kspace.npy, maps.npy, target.npy, meta.json")
    ap.add_argument("--slice_idx", type=int, default=0, help="0-based index into the prepared slices")
    ap.add_argument("--mode", choices=["cold", "warm"], default="cold")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--warm_sigma", type=float, default=10.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else Path(args.slice_dir) / "recon_out"
    out.mkdir(parents=True, exist_ok=True)

    recon, label, zf, mask, ssim, psnr, dt = run_recon(
        args.ckpt, args.slice_dir, args.slice_idx, args.mode, args.N, args.warm_sigma,
    )

    np.save(out / "recon.npy", recon)
    np.save(out / "label.npy", label)
    np.save(out / "zf.npy", zf)
    np.save(out / "mask.npy", mask)
    print(f"\n[{args.mode} N={args.N}]  SSIM {ssim:.4f}  PSNR {psnr:.2f} dB  in {dt:.0f} s")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
