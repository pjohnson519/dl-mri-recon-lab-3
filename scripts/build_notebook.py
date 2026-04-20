"""Generate notebooks/Lab3_ScoreMRI.ipynb from source-controlled Python.

Run once after edits:
    python scripts/build_notebook.py

That writes the .ipynb file, with all output cleared.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "Lab3_ScoreMRI.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(source):
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": source,
    }


cells = []

# ===================== Title =====================
cells.append(md("""# Lab 3 — Score-based diffusion priors for MRI reconstruction

In this lab we reconstruct an undersampled 4× knee k-space using a **score-based
diffusion model as a learned image prior**, following Chung & Ye 2022
(["Score-based diffusion models for accelerated MRI"](https://arxiv.org/abs/2202.04292)).

Unlike Lab 2's VarNet — which interleaves learned denoising and data consistency
inside one network trained end-to-end on undersampled data — score-MRI *separates*
prior and physics:

- A **score network** is trained once, offline, on *clean* knee images. It knows
  nothing about MRI physics, coils, or k-space.
- At inference, a **sampling loop** combines the score network (as a generic image
  denoiser) with classical data-consistency steps (k-space replacement + SENSE)
  to reconstruct the underlying image.

This lab walks through the math, then runs the full pipeline on real fastMRI data.

**Plan for this session:**

1. **Setup** (~2 min). Decide whether you're running **cold-start** or **warm-start**
   sampling — half the class does each so we can compare in discussion.
2. **Unconditional generation** (~20 min wall). Watch a knee image emerge from
   pure noise. No MRI physics involved.
3. **Reconstruction** (~20–30 min wall). Same score model, now with data
   consistency, reconstructs an actual undersampled acquisition.
4. **Discussion**. Compare cold vs warm results across the class.
5. **Homework**. Re-run reconstruction with a different mask type to explore
   generalization.

Cells (2) and (3) are long-running. **Kick them off and read the theory in
Part 1 while they execute.**
"""))

# ===================== Part 0: Setup =====================
cells.append(md("""## Part 0 — Setup

Run the two cells below. The first loads the pretrained score network and verifies
paths. The second is where you set **`SAMPLING_MODE`** for the rest of the lab.
"""))

cells.append(code("""# imports and shared paths
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path.cwd().parent if (Path.cwd().name == "notebooks") else Path.cwd()
sys.path.insert(0, str(REPO / "score_mri"))

# Shared-scratch assets. Ask your TA if any of these paths fail to resolve.
CKPT       = Path("/gpfs/scratch/johnsp23/DLrecon_lab1/pretrained/score_magnitude_knee320_ep95.pth")
DEMO_ROOT  = Path("/gpfs/scratch/johnsp23/DLrecon_lab1/data/lab3_demo")
TORCH_EXT  = Path("/gpfs/scratch/johnsp23/DLrecon_lab1/torch_ext_score")

# Pin the toolchain BigPurple modules for JIT (OnDemand Jupyter starts with
# CUDA 9.0 + GCC 4.8.5 on PATH, both too old for this PyTorch). We need to
# force CUDA 12.6 + GCC 11.2, and point TORCH_EXTENSIONS_DIR at the pre-built
# kernel cache so we don't wait 10 min for nvcc to rebuild.
import os
GCC_BIN  = "/gpfs/share/apps/gcc/11.2.0/bin"
CUDA_DIR = "/gpfs/share/apps/cuda/12.6"

os.environ["CUDA_HOME"] = CUDA_DIR
os.environ["CUDA_PATH"] = CUDA_DIR
os.environ["CC"]  = f"{GCC_BIN}/gcc"
os.environ["CXX"] = f"{GCC_BIN}/g++"

# Strip pre-existing CUDA entries (may point at old CUDA 9.0) then prepend ours.
clean_path = ":".join(p for p in os.environ.get("PATH", "").split(":")
                      if "/cuda/" not in p and "/cuda-" not in p)
os.environ["PATH"] = f"{GCC_BIN}:{CUDA_DIR}/bin:/usr/bin:/bin:{clean_path}"
os.environ["LD_LIBRARY_PATH"] = (
    f"{CUDA_DIR}/lib64:/gpfs/share/apps/gcc/11.2.0/lib64:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["TORCH_EXTENSIONS_DIR"] = str(TORCH_EXT)

assert CKPT.exists(),      f"Missing checkpoint: {CKPT}"
assert DEMO_ROOT.exists(), f"Missing demo data:  {DEMO_ROOT}"
print(f"Checkpoint: {CKPT}  ({CKPT.stat().st_size/1e6:.0f} MB)")
print(f"Demo data:  {sorted(p.name for p in DEMO_ROOT.iterdir())}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — you need a GPU node'}")
"""))

cells.append(code("""# *** IMPORTANT: Choose your sampling mode ***
#
# Your TA will split the class:
#   - Half of you use  SAMPLING_MODE = "cold"   (matches the paper's default)
#   - Half of you use  SAMPLING_MODE = "warm"   (~30% faster, slightly different dynamics)
#
# We'll compare notes across groups in the discussion at the end.

SAMPLING_MODE = "cold"      # <-- edit this to "warm" if your TA assigned you warm

assert SAMPLING_MODE in ("cold", "warm"), "Use either 'cold' or 'warm'"
print(f"This kernel will run SAMPLING_MODE = {SAMPLING_MODE!r}")
"""))

cells.append(code("""# Load the pretrained magnitude-trained score network. Takes ~30-60 s.
from configs.ve.fastmri_knee_320_ncsnpp_continuous import get_config
from models import ncsnpp  # noqa: F401   (registers the model class)
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint
import sde_lib

device = torch.device("cuda:0")
cfg = get_config()
cfg.device = device
cfg.training.batch_size = 1

score_model = mutils.create_model(cfg)
ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.model.ema_rate)
state = {"step": 0, "model": score_model, "ema": ema}
state = restore_checkpoint(str(CKPT), state, device, skip_sigma=True)
ema.copy_to(score_model.parameters())
score_model.eval()

# Variance-exploding SDE (VESDE) — the diffusion process the score model was trained with.
sde = sde_lib.VESDE(sigma_min=cfg.model.sigma_min,
                    sigma_max=cfg.model.sigma_max,
                    N=500)      # number of reverse-SDE steps; overridden later
print(f"Model loaded.  sigma_min={sde.sigma_min:.3f}  sigma_max={sde.sigma_max:.1f}")
"""))

# ===================== Part 2: Unconditional =====================
cells.append(md("""## Part 2 — Unconditional generation (watch knees emerge from noise)

Before doing reconstruction, let's see what the score network alone has learned.

For each run we initialize a 320×320 array with pure Gaussian noise at
σ = σ_max ≈ 378, then run the reverse SDE (predictor-corrector) down to
σ = σ_min = 0.01. **No MRI data is involved** — no k-space, no mask, no coils.
The sampler uses nothing but the prior.

We run **three samples with different random seeds** so you can see that the prior
captures a *distribution* of plausible knees, not one specific image. Different
seed → different patient's knee-like image, all drawn from the learned manifold.

Takes ~3 min per sample on an A100 (~9 min total). While it runs, scroll down and
read **Part 1 — Theory**.
"""))

cells.append(code("""from unconditional import unconditional_sample

# Collect a snapshot trajectory for the first seed only; the filmstrip tells the
# "noise -> knee" story once. For seeds 1 and 2 we keep just the final image.
snapshots = []
def collect(step_idx, img_np):
    snapshots.append((step_idx, img_np))

sde.N = 500
uncond_imgs = []
for i, seed in enumerate([0, 1, 2]):
    print(f"Sampling seed={seed} ...")
    img = unconditional_sample(
        score_model, sde, shape=(1, 1, 320, 320),
        snr=0.16, seed=seed,
        snap_callback=(collect if i == 0 else None), n_snapshots=10,
    )
    uncond_imgs.append(img)
uncond_img = uncond_imgs[0]    # the "canonical" one used elsewhere
print(f"Generated {len(uncond_imgs)} samples; captured {len(snapshots)} filmstrip frames.")
"""))

cells.append(code("""# Filmstrip: show snapshots from noise -> knee
# Each snapshot uses its own symmetric scale so you can see structure emerge
# from random noise even when intermediate iterates are huge in magnitude.
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, (step_idx, img) in zip(axes.ravel(), snapshots):
    vmax = max(abs(img.max()), abs(img.min())) or 1.0
    ax.imshow(img, cmap="gray", vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {step_idx+1}/{sde.N}", fontsize=9)
fig.suptitle("Unconditional reverse-SDE trajectory: noise → knee", y=1.02)
fig.tight_layout()
plt.show()
"""))

cells.append(code("""# All three final samples side-by-side with clean [0, max] grayscale scaling.
# Each image is a completely different "patient" — drawn independently from the
# learned prior distribution. The fact that they all look like plausible knees
# (and NOT like the same knee) is the point.
fig, axes = plt.subplots(1, len(uncond_imgs), figsize=(5 * len(uncond_imgs), 5))
if len(uncond_imgs) == 1: axes = [axes]
for ax, img, seed in zip(axes, uncond_imgs, [0, 1, 2]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=img.max())
    ax.set_title(f"Unconditional sample, seed={seed}")
    ax.set_axis_off()
fig.suptitle("Three plausible knees generated from pure noise by the score prior alone",
             y=1.02, fontsize=12)
plt.tight_layout(); plt.show()
"""))

# ===================== Part 1: Theory =====================
cells.append(md("""## Part 1 — Theory: score models, reverse SDE, and data consistency

While the unconditional sampler runs above (and the reconstruction runs below),
work through this section. The math isn't required to interpret the results, but it's
what makes the method click.

### 1.1 The score network

Given a distribution of clean images `p(x)` (here: fastMRI knee magnitude images),
a **score network** `s_θ(x, σ)` learns to estimate:

$$
s_\\theta(x, \\sigma) \\approx \\nabla_x \\log p_\\sigma(x)
$$

where `p_σ(x)` is the distribution of clean images blurred with Gaussian noise at
scale σ. The training recipe is simple:

1. Pick a clean image `x` and a noise level σ.
2. Add noise: `x̃ = x + σ · z`, with `z ~ N(0, I)`.
3. Ask the network to predict `-z/σ`, i.e. point back toward `x`.
4. Minimize `E[ ‖σ·s_θ(x̃, σ) + z‖² ]`.

No MRI specifics enter the training loss — it's a pure image-denoising task
across a wide range of noise levels. The same network denoises *slightly* noisy
images and *very* noisy images.

### 1.2 The reverse SDE

Once `s_θ` is trained, a **variance-exploding SDE** turns the score into a
generator. The forward SDE drives clean data `x₀ → x_T` (pure noise). Reversing
it:

$$
dx = [-\\sigma^2(t) \\, s_\\theta(x, t)] \\, dt + \\sigma(t) \\, d\\bar w
$$

turns noise `x_T` back into a data-distribution sample. We solve this numerically
with two alternating steps per time-step — the *predictor-corrector* scheme:

- **Predictor** (reverse diffusion): one discretized Euler step of the reverse SDE,
  guided by the score.
- **Corrector** (Langevin): one extra score-guided gradient step at the current σ
  level to sharpen the iterate.

**Cold start vs warm start.** The paper's default is to start `x_T` from pure
Gaussian noise at σ=σ_max ≈ 378. That's what Part 2 does and what *cold-start*
reconstruction does. Warm start initializes `x` = zero-filled recon + noise at some
smaller σ (we use σ=10), and begins the reverse SDE at the correspondingly earlier
timestep. Pros: ~30% faster, potentially fewer outlier samples. Cons: loses some
diversity/robustness and introduces bias if the zero-filled image has strong
aliasing artifacts. Your TA will have half the class run each mode; we'll compare.

### 1.3 Reconstruction = prior + data consistency

For reconstruction we don't just want *a* plausible knee — we want the knee
*consistent with the measured k-space*. This is where **data consistency (DC)**
comes in. Algorithm 5 from the paper alternates, per reverse-SDE step:

1. **Prior step**: run predictor-corrector on each of the 15 coil images'
   real and imaginary parts separately (30 score-network calls total per PC step,
   applied as generic 1-channel denoisers).
2. **Inner DC (per-coil, every step)**: for each coil image `x_c`, compute
   `FFT(x_c)`, then **replace** the frequencies that the scanner actually
   measured with the true measurements `y_c`. Invert back.
3. **Outer DC (SENSE Kaczmarz, every `m_steps=50` PC steps)**:
   `x ← x + λ · A^H(y − Ax)` where `A = mask·F·S` is the full SENSE forward model
   with sensitivity maps `S` (estimated via ESPIRiT from the fully-sampled ACS).

The prior makes the image look like a plausible knee. DC makes it match the
measurements. Iterating interleaves both constraints and converges to a
reconstruction that is both plausible and measurement-consistent.

**Compare to VarNet:**

| | VarNet (Lab 2) | Score-MRI |
|---|---|---|
| Training data | Undersampled k-space + GT images | Clean images only |
| Training sees physics? | Yes (DC layers in the loss) | No |
| Inference time | ~1 second | ~20-30 min |
| Inference FLOPs | ~12 UNet passes | ~15,000 score-model passes |
| One prior, many masks? | Retrain per mask | Same prior, any mask |

The two approaches are complementary. VarNet is fast and task-specialized;
score-MRI is slow but modular and doesn't need paired training data.
"""))

# ===================== Part 3: Reconstruction =====================
cells.append(md("""## Part 3 — Reconstruction of one demo slice

Now the main event. We take a fully-sampled knee k-space, apply a 4× gaussian1d
undersampling mask, and reconstruct using Algorithm 5.

Cell below: load the demo data, build the mask, visualize ground truth and
zero-filled baseline.
"""))

cells.append(code("""import sigpy.mri as mr
from skimage.metrics import structural_similarity as skssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from utils import fft2_m, ifft2_m, get_data_scaler, get_data_inverse_scaler
from utils import normalize_complex, root_sum_of_squares, lambda_schedule_linear
from sampling import ReverseDiffusionPredictor, LangevinCorrector

def ifft2c_np(x):
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2,-1)), norm="ortho"),
        axes=(-2,-1))

def gaussian1d_mask(size=320, acc=4, cf=0.08, seed=0):
    rng = np.random.RandomState(seed)
    m = np.zeros(size, dtype=np.float32)
    Nc = int(size * cf)
    m[size//2 - Nc//2 : size//2 - Nc//2 + Nc] = 1.0
    n_out = size // acc
    samples = rng.normal(loc=size//2, scale=size*(15.0/128), size=int(n_out*1.2))
    m[np.clip(samples.astype(int), 0, size-1)] = 1.0
    return np.tile(m[None,:], (size, 1)).astype(np.float32)

# --- load the PD demo slice (slice index 1 = 2nd central slice) ---
DEMO_VOL  = DEMO_ROOT / "pd_file1000073"
SLICE_IDX = 1
kspace_np = np.load(DEMO_VOL / "kspace.npy")[SLICE_IDX]                      # (15, 320, 320) complex64
H, W = kspace_np.shape[-2:]
full_coils = ifft2c_np(kspace_np).astype(np.complex64)
label_rss  = np.sqrt(np.sum(np.abs(full_coils)**2, axis=0)).astype(np.float32)
scale      = float(label_rss.max())
label_n    = np.clip(label_rss / scale, 0, 1)

mask_np = gaussian1d_mask(W, acc=4, cf=0.08)
zf_coils = ifft2c_np(kspace_np * mask_np).astype(np.complex64)
zf_n     = np.clip(np.sqrt(np.sum(np.abs(zf_coils)**2, axis=0)) / scale, 0, 1)
zf_ssim  = skssim(label_n, zf_n, data_range=1.0)
zf_psnr  = skpsnr(label_n, zf_n, data_range=1.0)

n_cols = int((mask_np.sum(axis=0) > 0).sum())
print(f"{DEMO_VOL.name} slice {SLICE_IDX}")
print(f"Mask: gaussian1d  {n_cols}/{W} cols = effective {W/n_cols:.2f}x acceleration")
print(f"ZF baseline:  SSIM {zf_ssim:.4f}   PSNR {zf_psnr:.2f} dB")

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(label_n, cmap='gray', vmin=0, vmax=1);  ax[0].set_title("Fully-sampled (GT)"); ax[0].set_axis_off()
ax[1].imshow(mask_np, cmap='gray');                   ax[1].set_title("gaussian1d 4x mask"); ax[1].set_axis_off()
ax[2].imshow(zf_n,    cmap='gray', vmin=0, vmax=1);   ax[2].set_title(f"Zero-filled (SSIM {zf_ssim:.3f})"); ax[2].set_axis_off()
plt.tight_layout(); plt.show()
"""))

cells.append(code("""# Set up and run the reconstruction sampler (Algorithm 5).
# Expected wall time on an A100:  ~32 min for cold,  ~22 min for warm.
# Scroll up and keep reading Part 1 while this runs.

from hybrid_sampler_warm import get_pc_fouriercs_RI_coil_SENSE_warm

N = 500
sde.N = N
scaler = get_data_scaler(cfg)
inverse_scaler = get_data_inverse_scaler(cfg)
lamb_schedule = lambda_schedule_linear(start_lamb=1.0, end_lamb=0.2)

mask_t = torch.from_numpy(mask_np).view(1, 1, H, W).to(device)

# Preprocess coil images the same way the paper does (normalize magnitude + phase
# to [0, 1], then FFT back to k-space).
img = normalize_complex(torch.from_numpy(full_coils)).view(1, 15, H, W).to(device)
k = fft2_m(img)
under_k = k * mask_t
under_img = ifft2_m(under_k)

# ESPIRiT sensitivity maps from the fully-sampled k-space (used inside the sampler).
sens = mr.app.EspiritCalib(k.cpu().detach().squeeze().numpy(), show_pbar=False).run()
sens_t = torch.from_numpy(sens).view(1, 15, H, W).to(device)

# Collect filmstrip snapshots as sampling proceeds.
recon_snaps = []
def collect(step_idx, rss):
    recon_snaps.append((step_idx, rss))

# warm_start = True if SAMPLING_MODE == "warm" else False
sampler = get_pc_fouriercs_RI_coil_SENSE_warm(
    sde, ReverseDiffusionPredictor, LangevinCorrector, inverse_scaler,
    snr=0.16, n_steps=1, m_steps=50, mask=mask_t, sens=sens_t,
    lamb_schedule=lamb_schedule, probability_flow=False,
    continuous=cfg.training.continuous, denoise=True,
    warm_start=(SAMPLING_MODE == "warm"), warm_sigma=10.0,
    snap_callback=collect, n_snapshots=10,
)

tic = time.time()
x = sampler(score_model, scaler(under_img), y=under_k)
dt = time.time() - tic

recon = np.abs(root_sum_of_squares(x, dim=1).squeeze().cpu().detach().numpy())
recon_n = np.clip(recon / (recon.max() or 1.0), 0, 1)
recon_ssim = skssim(label_n, recon_n, data_range=1.0)
recon_psnr = skpsnr(label_n, recon_n, data_range=1.0)

print(f"\\n[{SAMPLING_MODE} N={N}]  SSIM {recon_ssim:.4f}  PSNR {recon_psnr:.2f} dB  in {dt:.0f} s "
      f"(Δ SSIM vs ZF: {recon_ssim - zf_ssim:+.4f})")
"""))

cells.append(code("""# Recon filmstrip — how the sampler's RSS estimate evolved.
# Each snapshot uses its own scale so both the noisy early iterates AND the
# clean late iterates are visible. Watch the knee emerge from noise, constrained
# by the measurements, over 10 snapshots across the 500 PC steps.
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, (step_idx, rss) in zip(axes.ravel(), recon_snaps):
    vmax = rss.max() or 1.0
    ax.imshow(rss, cmap='gray', vmin=0, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {step_idx+1}/{sde.N}", fontsize=9)
fig.suptitle(f"Reconstruction trajectory ({SAMPLING_MODE} start) — per-step auto-scaled",
             y=1.02, fontsize=12)
fig.tight_layout(); plt.show()
"""))

cells.append(code("""# Final comparison: GT / ZF / score recon + error maps.
err_zf    = np.abs(zf_n    - label_n)
err_recon = np.abs(recon_n - label_n)

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0,0].imshow(label_n,  cmap='gray', vmin=0, vmax=1); ax[0,0].set_title("Ground truth (RSS)")
ax[0,1].imshow(zf_n,     cmap='gray', vmin=0, vmax=1); ax[0,1].set_title(f"Zero-filled  SSIM {zf_ssim:.4f}")
ax[0,2].imshow(recon_n,  cmap='gray', vmin=0, vmax=1); ax[0,2].set_title(f"Score ({SAMPLING_MODE})  SSIM {recon_ssim:.4f}")

ax[1,0].imshow(mask_np,    cmap='gray');                                  ax[1,0].set_title("Mask")
ax[1,1].imshow(err_zf,     cmap='hot',  vmin=0, vmax=0.1);                 ax[1,1].set_title(f"|ZF − GT|")
ax[1,2].imshow(err_recon,  cmap='hot',  vmin=0, vmax=0.1);                 ax[1,2].set_title(f"|Score − GT|")

for a in ax.ravel(): a.set_axis_off()
fig.suptitle(f"Reconstruction summary  — slice: {DEMO_VOL.name} [{SLICE_IDX}]", y=1.00)
fig.tight_layout(); plt.show()
"""))

# ===================== Part 4: Discussion =====================
cells.append(md("""## Part 4 — Discussion

Compare your numbers with a partner who ran the *other* sampling mode:

1. **Quantitative**: how similar are your cold vs warm SSIMs? Is one consistently higher?
2. **Qualitative**: look at the filmstrips. Does cold start show a cleaner "noise → knee"
   progression, or does warm start converge faster but look samey throughout?
3. **What the method actually did**: the prior never saw an undersampled or multi-coil
   image during training. It just learned "what clean knees look like." The sampler
   did the work of weaving in k-space measurements. How does that decoupling compare
   to VarNet's end-to-end approach?
4. **Computational cost**: ~20-30 min here vs <1 sec for VarNet. What would
   justify using score-MRI over VarNet in practice?

Some things to notice:

- The score sampler typically improves SSIM over ZF by ~+0.03 to +0.08, with
  larger gains on PDFS contrast (where ZF baselines are lower).
- Residual aliasing in the error map means the prior couldn't fully resolve
  ambiguity at unmeasured frequencies — this is where training data quality,
  mask design, and acceleration factor all matter.
- Cold and warm usually agree within ~±0.005 SSIM on the same slice. The
  difference matters more for speed than for quality.
"""))

# ===================== Part 5: Homework =====================
cells.append(md("""## Part 5 — Homework: change the sampling pattern

So far you used the paper's default **gaussian1d** mask with `acc_factor=4` and
`center_fraction=0.08`. That keeps ~26% of k-space columns (84 out of 320), densely
concentrated around the DC line.

**Your task**: reconstruct the second demo slice (`pdfs_file1001090` slice 1) using
**a different mask density**. Just change the two knobs on the existing
`gaussian1d_mask` function and rerun:

- **`acc_factor`** — higher = fewer columns sampled = harder recon. Try `6` or `8`.
- **`center_fraction`** — bigger = more dense low-frequency ACS block. Try `0.04` (harder)
  or `0.16` (easier).

A score prior trained on clean images should, in principle, work with *any* mask
density — that's the big difference from VarNet, which is specifically trained for
one mask. The question: does the SSIM gain over ZF stay roughly constant as you make
the mask harder, or does the gap collapse?

Suggested experiments (run at least one):

| `acc_factor` | `center_fraction` | effective accel | expected difficulty |
|---|---|---|---|
| 4 | 0.08 | ~3.8× | (the default — what you already have) |
| 6 | 0.08 | ~5.4× | harder |
| 8 | 0.04 | ~7.5× | much harder |
| 4 | 0.16 | ~3.5× | easier |

The starter cell below does everything the Part 3 cells did, but takes `acc_factor`
and `center_fraction` as parameters. Just pick values, run it, and compare the
resulting numbers with your cold/warm partner and with the default.
"""))

cells.append(code("""# --- Homework: reconstruct with different acc_factor / center_fraction ---
# Edit these two numbers; leave everything else alone.
HW_ACC            = 6       # try 6 or 8 for harder, or go back to 4 for baseline
HW_CENTER_FRAC    = 0.08    # try 0.04 (harder) or 0.16 (easier)

DEMO_VOL_HW  = DEMO_ROOT / "pdfs_file1001090"
SLICE_IDX_HW = 1

# Rebuild the forward problem with the new mask.
kspace_hw = np.load(DEMO_VOL_HW / "kspace.npy")[SLICE_IDX_HW]
full_coils_hw = ifft2c_np(kspace_hw).astype(np.complex64)
label_rss_hw  = np.sqrt(np.sum(np.abs(full_coils_hw)**2, axis=0)).astype(np.float32)
scale_hw      = float(label_rss_hw.max())
label_n_hw    = np.clip(label_rss_hw / scale_hw, 0, 1)

mask_hw = gaussian1d_mask(W, acc=HW_ACC, cf=HW_CENTER_FRAC)
zf_coils_hw = ifft2c_np(kspace_hw * mask_hw).astype(np.complex64)
zf_n_hw     = np.clip(np.sqrt(np.sum(np.abs(zf_coils_hw)**2, axis=0)) / scale_hw, 0, 1)
zf_ssim_hw  = skssim(label_n_hw, zf_n_hw, data_range=1.0)
zf_psnr_hw  = skpsnr(label_n_hw, zf_n_hw, data_range=1.0)
n_cols_hw   = int((mask_hw.sum(axis=0) > 0).sum())
print(f"HW mask: acc={HW_ACC}, cf={HW_CENTER_FRAC}  ->  {n_cols_hw}/{W} cols = effective {W/n_cols_hw:.2f}x")
print(f"HW ZF baseline: SSIM {zf_ssim_hw:.4f}  PSNR {zf_psnr_hw:.2f} dB")

# Now run the same sampler as Part 3 (cold or warm per SAMPLING_MODE).
mask_t_hw  = torch.from_numpy(mask_hw).view(1, 1, H, W).to(device)
img_hw     = normalize_complex(torch.from_numpy(full_coils_hw)).view(1, 15, H, W).to(device)
k_hw       = fft2_m(img_hw)
under_k_hw = k_hw * mask_t_hw
under_img_hw = ifft2_m(under_k_hw)
sens_hw    = mr.app.EspiritCalib(k_hw.cpu().detach().squeeze().numpy(), show_pbar=False).run()
sens_t_hw  = torch.from_numpy(sens_hw).view(1, 15, H, W).to(device)

sde.N = 500
sampler_hw = get_pc_fouriercs_RI_coil_SENSE_warm(
    sde, ReverseDiffusionPredictor, LangevinCorrector, inverse_scaler,
    snr=0.16, n_steps=1, m_steps=50, mask=mask_t_hw, sens=sens_t_hw,
    lamb_schedule=lamb_schedule, probability_flow=False,
    continuous=cfg.training.continuous, denoise=True,
    warm_start=(SAMPLING_MODE == "warm"), warm_sigma=10.0,
)
tic = time.time()
x_hw = sampler_hw(score_model, scaler(under_img_hw), y=under_k_hw)
dt_hw = time.time() - tic

recon_hw = np.abs(root_sum_of_squares(x_hw, dim=1).squeeze().cpu().detach().numpy())
recon_n_hw = np.clip(recon_hw / (recon_hw.max() or 1.0), 0, 1)
recon_ssim_hw = skssim(label_n_hw, recon_n_hw, data_range=1.0)
recon_psnr_hw = skpsnr(label_n_hw, recon_n_hw, data_range=1.0)
print(f"HW recon:  SSIM {recon_ssim_hw:.4f}  PSNR {recon_psnr_hw:.2f} dB  "
      f"(Δ SSIM vs HW ZF: {recon_ssim_hw - zf_ssim_hw:+.4f})  [{dt_hw:.0f} s]")

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(label_n_hw, cmap='gray', vmin=0, vmax=1); ax[0].set_title("GT"); ax[0].set_axis_off()
ax[1].imshow(zf_n_hw,    cmap='gray', vmin=0, vmax=1); ax[1].set_title(f"ZF ({zf_ssim_hw:.3f})"); ax[1].set_axis_off()
ax[2].imshow(recon_n_hw, cmap='gray', vmin=0, vmax=1); ax[2].set_title(f"Score ({recon_ssim_hw:.3f})"); ax[2].set_axis_off()
plt.tight_layout(); plt.show()
"""))

# ===================== Footer =====================
cells.append(md("""---

### Credits

- Method: Chung & Ye (2022) — *Score-based diffusion models for accelerated MRI* [[arXiv]](https://arxiv.org/abs/2202.04292) [[code]](https://github.com/HJ-harry/score-MRI)
- Foundation: Song et al. (2021) — *Score-Based Generative Modeling through Stochastic Differential Equations* [[arXiv]](https://arxiv.org/abs/2011.13456)
- Data: [fastMRI multi-coil knee dataset](https://fastmri.med.nyu.edu/), NYU Langone Health.
- Lab authors: `dl-mri-recon-lab` team, NYU Langone.
"""))

# ===================== Assemble =====================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3", "version": "3.10"
        },
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
