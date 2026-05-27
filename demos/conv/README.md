# conv

Purpose: show how convolutional filters detect visual patterns.

`conv` trains a tiny CNN on the bundled zipped 8x8 digits and displays the original digit, learned filters, feature maps, and class probabilities.

## Clip

![Conv Demo](../../media/conv.gif)

## In Simple Terms

A convolutional filter is a small reusable visual detector. It slides across the image and fires where a local pattern matches: strokes, corners, gaps, or blobs.

The feature maps show where each filter fires. The classifier head turns those local detections into digit probabilities.

## What The Model Does

Default shape:

```text
Input: 1x8x8 digit
Conv 1 -> channels
ReLU
Conv channels -> channels
ReLU
Classifier head -> 10 digit logits
```

Training uses the native 8x8 images from `digits8_mini/train`. The display scales the same native resolution up with nearest-neighbor drawing so the pixel structure stays visible.

## What To Look For Visually

- The original digit in the secondary panel.
- Learned filters as small 3x3 detector tiles.
- Feature maps showing where filters fire on the current digit.
- Strong activations clustering around strokes and corners.
- Output probabilities shifting as the model learns.

## Important Knobs

- `--channels`
- `--noise-amount`
- `--batch-size`
- `--lr`
- `--steps-per-frame`

Display controls:

- Up/down: browse variations of the same digit
- Left/right: switch digit classes
- `G`: random inference example
- `V`: toggle feature layer
- `M`: cycle light input noise

## Failure Cases Worth Trying

```bash
python -m scripts.display --demo conv --channels 2
python -m scripts.display --demo conv --noise-amount 0.2
python -m scripts.run --demo conv --steps 20
```

Too few channels make the filter bank cramped. Too much noise makes the feature maps fire in messy places.

## Display Command

```bash
python -m scripts.display --demo conv
```

## Headless Run Command

```bash
python -m scripts.run --demo conv --steps 1000
```
