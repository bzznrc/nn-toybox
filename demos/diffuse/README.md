# diffuse

Purpose: show how generative models can turn noise into structure.

`diffuse` trains a tiny denoising MLP on generated 2D point clouds and samples by reversing a simple noise schedule. The default live Arcade preset is intentionally cheap and favors smooth interaction over sample quality.

## Clip

![Diffuse Demo](../../media/diffuse.gif)

## In Simple Terms

Diffusion starts with clean data, adds noise to it, and trains a model to predict the noise that was added. Once the model can remove noise, generation runs the process backward: start from random noise, repeatedly denoise, and hope a structured sample appears.

The live view reads like the diffusion path: the secondary panel shows the clean target in neutral gray, while the main panel compares noisy corruption on the left in aqua with generated samples on the right in coral. The generated side animates through a cached preview path, so training can keep running in the background without sampling every frame.

## Neural Network Shape

Default topology:

```text
Input: noisy 2D point + 16 time features
Linear 18 -> 64
SiLU
Linear 64 -> 64
SiLU
Linear 64 -> 2
Output: predicted noise for x and y
```

That is `18 -> 64 -> 64 -> 2`, with **5,506 trainable parameters**.

The time features are sinusoidal features of the current diffusion timestep. The model learns to predict the noise vector, not the final point directly.

Defaults:

- Preset: `fast`
- Timesteps: `16`
- Preview Timesteps: `8`
- Noise Schedule: `linear`
- Hidden Dim: `64`
- Time Dim: `16`
- Points: `512`
- Sample Count: `64`
- Sample Refresh Every: `500`
- Max Clean Points Drawn: `128`
- Max Noised Points Drawn: `128`
- Max Generated Points Drawn: `64`
- Learning Rate: `0.002`
- Loss: mean squared error

## Commands

```bash
python -m scripts.display --demo diffuse --dataset gaussian-mixtures
python -m scripts.display --demo diffuse --preset nice --dataset gaussian-mixtures
python -m scripts.display --demo diffuse --dataset rings
python -m scripts.display --demo diffuse --dataset spirals --timesteps 60
python -m scripts.display --demo diffuse --dataset checkerboard --noise-schedule cosine

python -m scripts.run --demo diffuse --dataset gaussian-mixtures --steps 1000
python -m scripts.capture_demo --demo diffuse --dataset gaussian-mixtures
```

## Look For

- Clean target in the secondary panel.
- Noisy corruption on the left side of the main panel.
- Reverse denoising frames on the right side of the main panel.
- Whether generated samples recover the target shape.
- How fewer denoising steps make rougher samples.
- Gaussian mixtures are the fast default target; `--preset nice` raises preview count and preview timesteps for slower, smoother samples.

## Knobs

- `--dataset`: `Distributions` or a friendly point-cloud alias such as `gaussian-mixtures`, `rings`, `moons`, `checkerboard`, `spirals`
- `--distribution`: `gaussian_mixtures`, `rings`, `moons`, `checkerboard`, `spirals`
- `--preset`: `fast` or `nice`
- `--timesteps`
- `--sample-timesteps`
- `--noise-schedule`
- `--hidden-dim`
- `--time-dim`
- `--sample-count`
- `--sample-refresh-every`
- `--max-clean-points`
- `--max-noised-points`
- `--max-generated-points`
- `--noise`
- `--steps-per-frame`

## Failure Cases Worth Trying

```bash
python -m scripts.display --demo diffuse --dataset spirals --timesteps 8
python -m scripts.display --demo diffuse --dataset checkerboard --hidden-dim 16
python -m scripts.run --demo diffuse --dataset rings --steps 20
```
