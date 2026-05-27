# VS Code Launch Notes

The Run and Debug menu has four shared entries:

- `Run`: headless run/export through `scripts.run`
- `Display`: live Arcade display through `scripts.display`
- `Train Model`: headless model-training run for pretraining-style demos
- `Capture`: live Arcade GIF capture through `scripts.capture_demo`

The shared `Run`, `Display`, and `Capture` entries ask for:

1. `Demo`
2. `Dataset`

`Train Model` asks for a model demo, the model-training dataset, step count, and save cadence. Run names and capture filenames are automatic. The dataset picker shows only canonical display names such as `Distributions`, `Text - Tiny Semantics`, `Text - Big Tiny Semantics`, `Digits - 8x8 Mini`, and `Optimization - Loss Landscapes`. Distribution variants are selected inside the demo with up/down arrows. Invalid demo/dataset pairings fail before the demo starts.

`Run` is the old `Train` entry renamed around the shared headless entrypoint. `Train Model` is also headless and intentionally opens no Arcade window.
