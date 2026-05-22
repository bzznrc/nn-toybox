# nn-toybox

`nn-toybox` is a compact visual neural-network toybox in PyTorch and Arcade.

The main experience is live training: pick a demo, run `view`, and watch the model learn.

Working slogan: **Learning -> Geometry -> Compression -> Generation**

## Quick Start

Main experience:

```bash
python -m scripts.view --demo grad --dataset moons
python -m scripts.view --demo embed
python -m scripts.view --demo autoencode
python -m scripts.view --demo diffuse --dataset rings
```

Headless training and export:

```bash
python -m scripts.train --demo grad --dataset moons --steps 1000
python -m scripts.train --demo embed --steps 1000
python -m scripts.train --demo autoencode --steps 1000
python -m scripts.train --demo diffuse --dataset rings --steps 1000
```

`view` opens Arcade and trains live. `train` never opens Arcade; it is for CI, reproducibility, checkpoints, and static artifacts.

## Mental Model

- `demo` selects the toy: `grad`, `embed`, `autoencode`, or `diffuse`.
- Arcade is UI only.
- PyTorch and NumPy own model, data, and training.
- A small registry connects each demo config, trainer, and renderer.
- CI uses the shared `train` and trainer paths, not viewer checkpoints.

Common flags work across demos:

```bash
python -m scripts.view --demo grad --dataset moons --seed 0 --steps-per-frame 4
python -m scripts.train --demo diffuse --dataset rings --steps 1000 --seed 1
```

Demo-specific flags stay small:

```bash
python -m scripts.view --demo grad --boundary-resolution 96
python -m scripts.view --demo embed --embedding-dim 2
python -m scripts.view --demo autoencode --latent-dim 4
python -m scripts.view --demo diffuse --timesteps 32
```

## Learning Path

| Step | Demo | Concept | What to look at |
| --- | --- | --- | --- |
| 1 | `grad` | Learning | Decision boundaries, loss, gradients, optimizer behavior |
| 2 | `embed` | Geometry | Similarity, clusters, nearest neighbors, representation quality |
| 3 | `autoencode` | Compression | Bottlenecks, reconstruction, latent structure |
| 4 | `diffuse` | Generation | Noise schedules, denoising, sample formation |

## Display Controls

- `Space`: pause or resume
- `R`: reset with the same seed
- `N`: increment the seed and restart
- `S`: save a checkpoint and artifacts
- `1`, `2`, `3`: change training speed
- `Esc`: quit

## Artifacts

Headless runs write to `runs/<demo>/<run-name>/`:

- `config.json`: resolved config
- `metrics.json`: final metrics
- `checkpoint.pt`: final model checkpoint
- `artifacts/`: NumPy exports and static preview images when implemented

The primary workflow is live training through `view`, not opening a saved run. Checkpoint loading can be added later without becoming the main path.

## Repo Layout

- `scripts/train.py`: shared headless entrypoint
- `scripts/view.py`: shared Arcade live-training entrypoint
- `core/config.py`: shared config/dataclass/CLI helpers
- `core/registry.py`: tiny explicit demo registry
- `demos/<name>/`: per-demo `config`, `trainer`, `renderer`, `model`, and `data`
- `core/`: shared Arcade view, checkpoints, plotting, generated data, and utilities

## Data Policy

V1 is synthetic by default and requires no downloads.

- `grad`: moons, circles, spirals, XOR, blobs, checkerboard
- `embed`: a tiny hand-written semantic relation corpus
- `autoencode`: generated 16x16 icons
- `diffuse`: generated 2D point clouds such as rings, moons, spirals, Gaussian mixtures, and checkerboard

Optional downloaded datasets can be future TODOs, not core behavior.

## Relationship To rl-toybox

`nn-toybox` follows the compact style of `rl-toybox`: one small registry, shared config/train setup, generated toy problems, compact logs, simple run artifacts, and a shared Arcade visual language.

It does not copy RL-specific abstractions. There are no environments, rewards, replay buffers, rollout runners, policies, or curricula in V1. The equivalent organizing idea here is the four-step neural-network path: supervised learning, embeddings, autoencoding, and diffusion.

## Smoke Check

```bash
python -m scripts.train --demo grad --dataset moons --steps 5 --smoke
python -m scripts.train --demo diffuse --dataset rings --steps 5 --smoke
```

Smoke checks use CPU-only trainers and the shared headless run path. They do not open Arcade windows.
