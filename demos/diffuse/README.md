# diffuse

Purpose: show how generative models can turn noise into structure.

`diffuse` trains a tiny denoising MLP on generated 2D point clouds and samples by reversing a simple noise schedule.

## In Simple Terms

Diffusion starts with clean data, adds noise to it, and trains a model to predict the noise that was added. Once the model can remove noise, generation runs the process backward: start from random noise, repeatedly denoise, and hope a structured sample appears.

The live view shows clean data, noisy data, and a reverse denoising trajectory. Early in training, generated points look scattered. As learning improves, they should begin to follow the dataset shape.

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

- Timesteps: `32`
- Noise Schedule: `linear`
- Hidden Dim: `64`
- Time Dim: `16`
- Points: `1024`
- Sample Count: `512`
- Sample Refresh Every: `100`
- Learning Rate: `0.002`
- Loss: mean squared error

## Commands

```bash
python -m scripts.view --demo diffuse --dataset rings
python -m scripts.view --demo diffuse --dataset spirals --timesteps 60
python -m scripts.view --demo diffuse --dataset checkerboard --noise-schedule cosine

python -m scripts.train --demo diffuse --dataset rings --steps 1000
python -m scripts.capture_demo --demo diffuse --dataset rings
```

## Look For

- Clean data versus fully noised data.
- Reverse denoising frames in the viewer.
- Whether generated samples recover the target shape.
- How fewer denoising steps make rougher samples.

## Knobs

- `--dataset`: `rings`, `moons`, `spirals`, `gaussian_mixtures`, `checkerboard`
- `--timesteps`
- `--noise-schedule`
- `--hidden-dim`
- `--time-dim`
- `--sample-count`
- `--sample-refresh-every`
- `--noise`
- `--steps-per-frame`

## Failure Cases Worth Trying

```bash
python -m scripts.view --demo diffuse --dataset spirals --timesteps 8
python -m scripts.view --demo diffuse --dataset checkerboard --hidden-dim 16
python -m scripts.train --demo diffuse --dataset rings --steps 20
```
