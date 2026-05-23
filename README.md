# nn-toybox

`nn-toybox` is a compact visual neural-network toybox in PyTorch and Arcade.

The main experience is live training: pick a demo, run `view`, and watch the model learn.

Working slogan: **Learning -> Geometry -> Compression -> Generation**

## Clips

<p align="center">
  <img src="media/grad.gif" alt="Grad demo clip" width="50%" /><img src="media/embed.gif" alt="Embed demo clip" width="50%" />
</p>
<p align="center">
  <img src="media/autoencode.gif" alt="Autoencode demo clip" width="50%" /><img src="media/diffuse.gif" alt="Diffuse demo clip" width="50%" />
</p>

## Quick Start

Main experience:

```bash
python -m scripts.view --demo grad --dataset "Distributions - Moons"
python -m scripts.view --demo embed
python -m scripts.view --demo autoencode
python -m scripts.view --demo diffuse --dataset "Distributions - Gaussian Mixtures"
```

Headless training and export:

```bash
python -m scripts.train --demo grad --dataset "Distributions - Moons" --steps 1000
python -m scripts.train --demo embed --steps 1000
python -m scripts.train --demo autoencode --steps 1000
python -m scripts.train --demo diffuse --dataset "Distributions - Gaussian Mixtures" --steps 1000
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
python -m scripts.view --demo grad --dataset "Distributions - Moons" --seed 0 --steps-per-frame 4
python -m scripts.train --demo diffuse --dataset "Distributions - Gaussian Mixtures" --steps 1000 --seed 1
```

Demo-specific flags stay small:

```bash
python -m scripts.view --demo grad --boundary-resolution 96
python -m scripts.view --demo embed --embedding-dim 2
python -m scripts.view --demo autoencode --latent-dim 4
python -m scripts.view --demo diffuse --sample-timesteps 8
python -m scripts.view --demo diffuse --preset nice
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

- `grad`: Distributions - Blobs, Distributions - Gaussian Mixtures, Distributions - Circles, Distributions - Rings, Distributions - Moons, Distributions - XOR, Distributions - Checkerboard, Distributions - Spiral
- `embed`: Text - Tiny Semantics, Text - Big Tiny Semantics
- `autoencode`: Images - Icons, Images - Patterns
- `diffuse`: generated 2D point clouds such as Distributions - Gaussian Mixtures, Distributions - Rings, Distributions - Moons, Distributions - Checkerboard, and Distributions - Spiral

Diffuse defaults to a fast Gaussian Mixtures live preset: 512 training points, 16 training timesteps, 8 preview timesteps, 64 preview samples, and sparse cached drawing. Use `--preset nice` for the heavier/slower preview settings.

Optional downloaded datasets can be future TODOs, not core behavior.

## Smoke Check

```bash
python -m scripts.train --demo grad --dataset "Distributions - Moons" --steps 5 --smoke
python -m scripts.train --demo diffuse --dataset "Distributions - Gaussian Mixtures" --steps 5 --smoke
```

Smoke checks use CPU-only trainers and the shared headless run path. They do not open Arcade windows.
