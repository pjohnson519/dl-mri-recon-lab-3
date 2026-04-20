"""Modified copy of get_pc_fouriercs_RI_coil_SENSE with warm-start support.

Warm start: initialize the coil state `x` = `under_coils + warm_sigma * complex_noise`
and run the reverse SDE from `t_start = sigma_to_t(warm_sigma)` down to eps, rather
than from t=T down to eps. `under_coils` = IFFT(kspace * mask) — the zero-filled
coil images.

Drop-in analogue of get_pc_fouriercs_RI_coil_SENSE.
"""
import functools
import numpy as np
import torch
from tqdm import tqdm

from sampling import shared_predictor_update_fn, shared_corrector_update_fn
from utils import fft2_m, ifft2_m, root_sum_of_squares


def get_pc_fouriercs_RI_coil_SENSE_warm(
    sde, predictor, corrector, inverse_scaler, snr,
    n_steps=1, lamb_schedule=None, probability_flow=False, continuous=False,
    denoise=True, eps=1e-5, sens=None, mask=None, m_steps=10,
    warm_start=False, warm_sigma=10.0,
    snap_callback=None, n_snapshots=10,
):
    """Same as get_pc_fouriercs_RI_coil_SENSE but with warm-start + snapshot hook.

    warm_start: if True, init from zero-filled coils + noise at `warm_sigma`;
        otherwise cold-start from pure Gaussian (paper default).
    snap_callback: optional fn(step_idx, rss_image_np) called n_snapshots times
        over the course of sampling, used for filmstrip visualization.
    """
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde, predictor=predictor,
        probability_flow=probability_flow, continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde, corrector=corrector, continuous=continuous,
        snr=snr, n_steps=n_steps,
    )

    def data_fidelity(mask_c, x, x_mean, y):
        x = ifft2_m(fft2_m(x) * (1.0 - mask_c) + y)
        x_mean = ifft2_m(fft2_m(x_mean) * (1.0 - mask_c) + y)
        return x, x_mean

    def A(x, sens=sens, mask=mask):
        return mask * fft2_m(sens * x)

    def A_H(x, sens=sens, mask=mask):
        return torch.sum(torch.conj(sens) * ifft2_m(x * mask), dim=1).unsqueeze(dim=1)

    def kaczmarz(x, x_mean, y, lamb=1.0):
        x = x + lamb * A_H(y - A(x))
        x_mean = x_mean + lamb * A_H(y - A(x_mean))
        return x, x_mean

    def get_coil_update_fn(update_fn):
        def fouriercs_update_fn(model, data, x, t, y=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x_real = torch.real(x)
                x_imag = torch.imag(x)
                x_real, x_real_mean = update_fn(x_real, vec_t, model=model)
                x_imag, x_imag_mean = update_fn(x_imag, vec_t, model=model)
                x = x_real + 1j * x_imag
                x_mean = x_real_mean + 1j * x_imag_mean
                mask_c = mask[0, 0, :, :].squeeze()
                x, x_mean = data_fidelity(mask_c, x, x_mean, y)
                return x, x_mean
        return fouriercs_update_fn

    predictor_coil_update_fn = get_coil_update_fn(predictor_update_fn)
    corrector_coil_update_fn = get_coil_update_fn(corrector_update_fn)

    def pc_fouriercs(model, data, y=None):
        with torch.no_grad():
            if warm_start:
                under_coils = ifft2_m(y)                                 # (1, 15, H, W) complex
                # VESDE marginal at t_start: N(x0, warm_sigma^2 I)
                noise_r = torch.randn_like(under_coils.real) * warm_sigma
                noise_i = torch.randn_like(under_coils.imag) * warm_sigma
                x = under_coils + torch.complex(noise_r, noise_i).to(under_coils.dtype)
                x_mean = x.clone().detach()
                t_start = float(
                    np.log(warm_sigma / sde.sigma_min) / np.log(sde.sigma_max / sde.sigma_min)
                )
                t_start = max(eps, min(t_start, sde.T))
                timesteps = torch.linspace(t_start, eps, sde.N)
            else:
                x_r = sde.prior_sampling(data.shape).to(data.device)
                x_i = sde.prior_sampling(data.shape).to(data.device)
                x = torch.complex(x_r, x_i)
                x_mean = x.clone().detach()
                timesteps = torch.linspace(sde.T, eps, sde.N)

            snap_every = max(sde.N // max(n_snapshots, 1), 1) if snap_callback is not None else None

            for i in tqdm(range(sde.N)):
                for c in range(data.shape[1]):
                    t = timesteps[i]
                    x_c = x[:, c:c + 1, :, :]
                    y_c = y[:, c:c + 1, :, :]
                    x_c, x_c_mean = predictor_coil_update_fn(model, data, x_c, t, y=y_c)
                    x_c, x_c_mean = corrector_coil_update_fn(model, data, x_c, t, y=y_c)
                    x[:, c, :, :] = x_c
                    x_mean[:, c, :, :] = x_c_mean
                if i % m_steps == 0:
                    lamb = lamb_schedule.get_current_lambda(i)
                    x, x_mean = kaczmarz(x, x_mean, y, lamb=lamb)

                if snap_every is not None and ((i + 1) % snap_every == 0 or i == sde.N - 1):
                    rss = root_sum_of_squares(torch.abs(x_mean), dim=1).squeeze().detach().cpu().numpy()
                    snap_callback(i, rss)

            return inverse_scaler(x_mean if denoise else x)

    return pc_fouriercs
