# Lab 3 — Score-based diffusion priors for MRI reconstruction

A teaching lab on **learned-prior + classical data-consistency** MRI reconstruction, using a pretrained score-based diffusion model on the fastMRI multi-coil knee dataset. Companion to Labs 1–2 (VarNet).

We vendor and lightly adapt code from the score-MRI reference implementation
([Chung & Ye 2022](https://arxiv.org/abs/2202.04292), [code](https://github.com/HJ-harry/score-MRI)) and ship our own trained weights.

---

## What the model actually does (read before the lab)

Lab 2's VarNet interleaves learned denoising and k-space data consistency (DC) inside one 12-cascade network trained end-to-end on undersampled data. Score-MRI takes the **opposite** approach: it trains a *generic image prior* once, offline, on clean knee images only — then at inference time composes that prior with *any* classical forward model.

### The score network (one-time training)

A U-Net-like architecture (NCSN++) is trained to approximate the **score** `∇_x log p_σ(x)` of knee images at every noise level σ ∈ [0.01, 378]. Concretely, the training procedure per minibatch is:

1. Sample a clean fastMRI knee magnitude slice `x`.
2. Sample a noise level σ (log-uniform).
3. Add Gaussian noise: `x̃ = x + σ · z`, `z ~ N(0, I)`.
4. Ask the network to predict `−z / σ`, i.e. point back toward clean `x`.
5. Minimize `E[‖σ · s_θ(x̃, σ) + z‖²]`.

The network never sees MRI physics, coils, k-space, or undersampling. It just learns to denoise generic images at arbitrary noise levels.

### Reverse SDE sampling (inference)

Unconditional generation (Part 2 of the notebook) starts from pure Gaussian noise at σ = σ_max and runs 500 alternating predictor-corrector steps:

- **Predictor** (reverse diffusion): one Euler step of the reverse variance-exploding SDE, using `s_θ` as the drift.
- **Corrector** (Langevin): one extra score-guided MCMC step at the current σ to sharpen.

This is enough to sample a believable-looking knee from noise — with no MRI data involved whatsoever.

### Reconstruction = prior + data consistency

For reconstruction (Part 3), we still run the reverse SDE, but at each PC step we add classical MRI physics on top:

1. **Prior step**: apply the score network as a generic image denoiser to *real and imaginary parts of each coil image separately* — 30 score-network passes per reverse-SDE step (15 coils × {Re, Im}).
2. **Inner DC (every step)**: per-coil hard k-space replacement — keep everything the scanner measured, use the model's prediction for everything else.
3. **Outer DC (every 50 steps)**: SENSE-Kaczmarz step using ESPIRiT sensitivity maps — pushes all 15 coils into joint agreement.

Iterating 500 times converges to an image that is **both** plausible (the prior) and measurement-consistent (DC). Because the prior and the forward model are decoupled, the same checkpoint works on any mask, any acceleration, any coil count — only the DC changes.

### What you'll see in the lab

- **Cold start** (half the class) and **warm start** (the other half) — we compare.
- Unconditional generation trajectory from noise → knee (filmstrip).
- A 4× reconstruction trajectory + SSIM/PSNR vs zero-filled baseline.
- Homework: try a different mask pattern on a second slice.

**Expected wall-clock**: ~15-20 min unconditional, ~22-32 min reconstruction, both on an A100. Read Part 1 (theory) of the notebook while they run.

---

## Quick start

You'll launch Jupyter on a GPU node via the **NYU HPC OnDemand** web portal, same as Labs 1 and 2.

### 1. Clone this repo to your home dir

```bash
ssh YOUR_KID@bigpurple.nyumc.org
cd ~
git clone https://github.com/pjohnson519/dl-mri-recon-lab-3.git
```

### 2. Launch Jupyter on a GPU node (A100 required)

1. In your browser: **[https://ondemand.hpc.nyumc.org](https://ondemand.hpc.nyumc.org)**
2. Log in with your KID.
3. Start **Interactive App → Jupyter** with:
   - GPU: `a100` (the lab's sampler runs here)
   - Time: 2 hours
   - Memory: 32 GB
4. Once running, click **Connect to Jupyter**.

### 3. Open the notebook

- Navigate to `dl-mri-recon-lab-3/notebooks/`
- Open `Lab3_ScoreMRI.ipynb`
- **Before running anything**, edit the `SAMPLING_MODE` line per your TA's instructions (half the class runs `"cold"`, half `"warm"`)
- **Run → Run All**. Then scroll up and read Part 1 theory while cells 2 and 3 run.

---

## Shared scratch assets

All large files live on shared scratch so the repo stays small:

| Path | What |
|---|---|
| `/gpfs/scratch/johnsp23/DLrecon_lab1/envs/score-mri/` | Pre-built conda env (Python 3.10, PyTorch, sigpy, scikit-image, ...) |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/pretrained/score_magnitude_knee320_ep95.pth` | Magnitude-prior score checkpoint (~492 MB) |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/data/lab3_demo/pd_file1000073/` | PD demo slices (kspace.npy, maps.npy, target.npy, meta.json) |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/data/lab3_demo/pdfs_file1001090/` | PDFS demo slices (for homework) |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/torch_ext_score/` | Pre-compiled CUDA kernels for `upfirdn2d` and `fused_act` (saves ~10 min at first run) |

The notebook sets `TORCH_EXTENSIONS_DIR` to that last path automatically so you reuse the compiled kernels instead of triggering a 10-minute nvcc compile on your fresh node.

---

## Repository layout

```
dl-mri-recon-lab-3/
├── README.md                       # you are here
├── LICENSE                         # Apache 2.0, for our additions
├── LICENSE_score-MRI               # Apache 2.0, from the upstream repo
├── NOTICE                          # attribution and citation info — please read
├── notebooks/
│   └── Lab3_ScoreMRI.ipynb         # the lab notebook (run this)
├── scripts/
│   ├── recon_one_slice.py          # CLI helper: reconstruct one slice from the shell
│   └── build_notebook.py           # regenerates the notebook from source-controlled Python
└── score_mri/                      # vendored + lightly adapted from HJ-harry/score-MRI
    ├── configs/ve/fastmri_knee_320_ncsnpp_continuous.py
    ├── configs/default_lsun_configs.py
    ├── models/                     # NCSN++, EMA
    ├── op/                         # custom CUDA ops (JIT-compiled on first use)
    ├── sampling.py                 # Algorithm 5 (get_pc_fouriercs_RI_coil_SENSE)
    ├── hybrid_sampler_warm.py      # our warm-start variant + snapshot hook
    ├── unconditional.py            # Part 2 helper — reverse SDE without DC
    ├── sde_lib.py                  # VESDE definition
    ├── utils.py                    # FFT, masking, scalers, schedules
    ├── losses.py                   # used indirectly via model creation
    └── fastmri_utils.py            # FFT wrappers
```

---

## Credits and license

This repo **vendors code from score-MRI** ([Chung & Ye 2022](https://arxiv.org/abs/2202.04292)) under Apache License 2.0. The original repo is the canonical reference — our copy is tailored for classroom use only. **If you build on this lab, please cite the paper**:

```bibtex
@article{chung2022score,
  title   = {Score-based diffusion models for accelerated MRI},
  author  = {Chung, Hyungjin and Ye, Jong Chul},
  journal = {Medical Image Analysis},
  volume  = {80},
  pages   = {102479},
  year    = {2022},
  doi     = {10.1016/j.media.2022.102479}
}
```

See `NOTICE` for full attribution (including the foundational Song et al. score-SDE paper) and `LICENSE` / `LICENSE_score-MRI` for the Apache 2.0 terms.

### What we added

- Training of a magnitude-prior score network on our slice of fastMRI multi-coil knee data (we didn't have access to the authors' published weights, which aren't shipped with the upstream repo).
- `hybrid_sampler_warm.py`: warm-start variant of the paper's hybrid sampler, plus a snapshot-capture hook for filmstrip visualization.
- `unconditional.py`: a clean reverse-SDE-only helper for Part 2.
- The lab notebook, scripts, and this README.

### Data

The fastMRI knee dataset is distributed by NYU Langone Health for research purposes and is **not** redistributed here. See [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu/).
