# AEPsych — Color Discrimination Experiment Fork

This is a research fork of [AEPsych](https://github.com/facebookresearch/aepsych) extended to implement the adaptive color discrimination experiment described in:

> **Hong et al. (2025)** — *Comprehensive characterization of human color discrimination thresholds*
> bioRxiv. [https://doi.org/10.1101/2025.07.16.665219](https://www.biorxiv.org/content/10.1101/2025.07.16.665219v2)

The companion Unity/Quest app (stimulus presentation) lives here: [github.com/Znasif/VisionTest/tree/fullview](https://github.com/Znasif/VisionTest/tree/fullview)

---

## Motivation

Color discrimination thresholds — the smallest detectable color differences — are fundamental benchmarks for color vision models, disease assessment, and display design. Characterizing them comprehensively across the full stimulus space has long been considered intractable due to the **psychophysical curse of dimensionality**: a naïve grid search over a 4-D stimulus space would require millions of trials.

Hong et al. (2025) solve this with two key innovations:
1. **Adaptive trial placement via AEPsych** — a Gaussian Process model selects the most informative stimulus on every trial, targeting the 66.7%-correct threshold level.
2. **Post-hoc WPPM fitting** — a semi-parametric Wishart Process Psychophysical Model fits the collected data and recovers a continuously varying covariance field over stimulus space.

Together, the full isoluminant-plane discrimination surface can be mapped in ~6,000 trials per participant.

---

## What This Experiment Looks Like

![WPPM fitted threshold ellipses](WPPM.gif)

The GIF above shows the WPPM-fitted discrimination threshold ellipses evolving across the isoluminant color plane as data accumulates. Each ellipse represents the just-noticeable-difference (JND) boundary at one reference point.

---

## System Architecture

The system runs as a **dual-computer pipeline**:

```
Meta Quest 3 (Unity)                    Logic PC (Python / WSL)
─────────────────────                   ───────────────────────
3AFC color discrimination task          AEPsych server (socket)
│                                       │
│  ←── next trial parameters ──────────┤
│  ──── participant response ──────────→│
│                                       │
│  (ngrok TCP tunnel bridges LAN gap)   ├── realtime_visualizer.py
                                        └── post-hoc WPPM fitting
```

See [architecture.md](architecture.md) for the full specification.

---

## Repository Structure

```
aepsych/                        # Core AEPsych library (upstream)
aepsych/server/server.py        # Server entry point used in this experiment
src/
  adaptive_engine/
    aepsych_wrapper.py          # 4-D color discrimination config wrapper
    color.py                    # Model-space ↔ RGB transformations
    wppm_fitter.py              # Post-hoc Wishart Process fitting (JAX)
simulation.py                   # Headless AEPsych simulation
test_integration_with_wppm.py   # Full pipeline test (--display flag for plots)
realtime_visualizer.py          # Live DB polling + matplotlib plot
wppm_visualizer.py              # Standalone WPPM result visualizer
wppm_fitted.npz                 # Example fitted WPPM output
```

---

## Quick Start

### 1. Install dependencies

```bash
conda create -n braille python=3.11
conda activate braille
pip install -e ".[dev]"
pip install jax jaxlib  # for WPPM fitting
```

### 2. Start the AEPsych server

```bash
conda activate braille
python aepsych/server/server.py --port 5555 --ip 0.0.0.0
```

### 3. Expose the server over the internet (for Quest)

```bash
ngrok tcp 5555
```

Copy the generated address (e.g. `tcp://0.tcp.ngrok.io:12345`) and paste it into the Quest app's server URL field before starting the experiment.

### 4. (Optional) Watch trials arrive in real time

```bash
python realtime_visualizer.py
```

This polls `aepsych/server/databases/default.db` every second and updates a live scatter plot of sampled stimuli in model space, colour-coded by correct/incorrect response.

### 5. (Optional) Run a simulation without a headset

```bash
python test_integration_with_wppm.py --display
```

Simulates a full experiment run on your desktop, including WPPM fitting and threshold visualisation at the end.

---

## AEPsych Configuration

The experiment uses a 4-D parameter space:

| Parameter | Description | Range |
|-----------|-------------|-------|
| `x0_dim1` | Reference colour, dimension 1 (model space) | [-0.7, 0.7] |
| `x0_dim2` | Reference colour, dimension 2 (model space) | [-0.7, 0.7] |
| `delta_dim1` | Comparison offset, dimension 1 | [-0.3, 0.3] |
| `delta_dim2` | Comparison offset, dimension 2 | [-0.3, 0.3] |

The binary outcome is **1 = correct** (participant identified the odd-one-out), **0 = incorrect**. AEPsych targets the 66.7%-correct threshold level (`MCLevelSetEstimation`, `target = 0.667`).

Phase 1 (900 trials): Sobol quasi-random initialization.
Phase 2 (5100 trials): GP-based adaptive sampling with `OptimizeAcqfGenerator`.

---

## Post-hoc WPPM Fitting

After data collection, run:

```bash
python wppm_visualizer.py
```

This loads `wppm_fitted.npz` (or re-fits from the collected data) and renders threshold ellipses across the isoluminant plane.

---

## Upstream AEPsych

The original AEPsych library documentation follows below.

AEPsych is a framework and library for adaptive experimentation in psychophysics and related domains.

**Installation:** `pip install aepsych`
**Server:** `aepsych_server --port 5555 --ip 0.0.0.0 --db mydatabase.db`
**Paper:** Owen et al. (2021) — [Adaptive Nonparametric Psychophysics](https://arxiv.org/abs/2104.09549)
**License:** CC-BY-NC 4.0

See the original [README](https://github.com/facebookresearch/aepsych#readme) and [examples](https://github.com/facebookresearch/aepsych/tree/main/examples) for full upstream documentation.
