"""Unconditional image generation with a score-based diffusion model.

Runs the reverse VE-SDE from pure Gaussian noise down to data, WITHOUT any
data-consistency step. Used by Lab 3 to visualize what the prior alone produces.

The only twist: the pretrained knee-magnitude prior is a 1-channel real image
denoiser, so we sample one single image (shape (1, 1, H, W)).
"""
import numpy as np
import torch
from tqdm import tqdm

from sampling import ReverseDiffusionPredictor, LangevinCorrector
from models import utils as mutils


def unconditional_sample(
    score_model, sde, shape=(1, 1, 320, 320),
    snr=0.16, n_corrector_steps=1, eps=1e-5,
    snap_callback=None, n_snapshots=10, device=None, seed=None,
):
    """Cold-start reverse-SDE sampling with predictor-corrector, no DC.

    Returns a numpy array shape (H, W) float32 — the final generated image (clipped to [0, 1]).
    If `snap_callback` is provided, it will be called as
        snap_callback(step_idx, x_mean_np)
    n_snapshots times during sampling, for filmstrip visualization.
    """
    if device is None:
        device = next(score_model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Build predictor + corrector on top of the model's score function
    score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=True)
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    corrector = LangevinCorrector(sde, score_fn, snr=snr, n_steps=n_corrector_steps)

    with torch.no_grad():
        # Cold start: x_T ~ N(0, sigma_max^2 * I)
        x = sde.prior_sampling(shape).to(device)
        x_mean = x.clone()

        timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
        snap_every = max(sde.N // max(n_snapshots, 1), 1) if snap_callback is not None else None

        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=device) * t
            x, x_mean = predictor.update_fn(x, vec_t)
            x, x_mean = corrector.update_fn(x, vec_t)

            if snap_every is not None and ((i + 1) % snap_every == 0 or i == sde.N - 1):
                snap_callback(i, x_mean.detach().cpu().numpy().squeeze())

        img = x_mean.detach().cpu().numpy().squeeze()
    # The model was trained on [0, 1] magnitudes; clip to a visible range.
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img
