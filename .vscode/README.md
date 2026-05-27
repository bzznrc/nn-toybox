# VS Code Launch Notes

The Run and Debug menu has four shared entries:

- `Run`: headless run/export through `nn_toybox.run`
- `Display`: live Arcade display through `nn_toybox.display`
- `Train Model`: headless model-training run for pretraining-style demos
- `Capture`: live Arcade GIF capture through `scripts.capture_demo`

The shared `Run`, `Display`, and `Capture` entries ask for:

1. `Demo`
2. `Dataset`

`Train Model` asks for a model demo, the model-training dataset, step count, and save cadence. Run names and capture filenames are automatic. The full dataset list is shared across demos; choose the exact canonical dataset name shown in the picker, such as `Distributions - Moons`, `Text - Tiny Semantics`, `Digits - 8x8 Mini`, `Text - Subject Verb Agreement`, or `Optimization - Loss Landscapes`. Aliases are intentionally not accepted, and invalid demo/dataset pairings fail before the demo starts.

`Run` is the old `Train` entry renamed around the shared headless entrypoint. `Train Model` is also headless and intentionally opens no Arcade window.
