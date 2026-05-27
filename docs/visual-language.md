# Visual Language

The visual language for `nn-toybox` is deliberately simple, clear, minimal, and reusable. Each demo should feel like part of one small system, not a separate visualization experiment.

## Philosophy

- Keep the screen quiet and legible.
- Reuse the shared layout, palette, marker shapes, and chart styling from core.
- Prefer one strong visual idea per region.
- Avoid nested frames, decorative panels, and demo-specific chrome.
- Let demos provide data and domain visuals; let `core.arcade_view` provide structure.

## Layout

Live demos use one shared layout from `core.arcade_view`:

- A single square main canvas on the left.
- A right column with an optional secondary visual at the top.
- A text area below the secondary visual.
- The combined main-canvas-plus-sidebar block is centered inside the available window margins.
- Core draws the dark background and the main canvas outline.
- Demos draw only their data into the regions core gives them.

Core layout constants from `core.arcade_style`:

- `TOYBOX_MARGIN = 24.0`
- `TOYBOX_GAP = 18.0`
- `TOYBOX_MAIN_WIDTH_FRACTION = 0.75`
- `TOYBOX_SECONDARY_HEIGHT_FRACTION = 0.42`
- `TOYBOX_INNER_GAP = 24.0`

## Outlines

The main canvas has exactly one outline, drawn by core.

- `TOYBOX_PANEL_OUTLINE_ALPHA = 90`
- `TOYBOX_PANEL_OUTLINE_WIDTH = 3.0`

Demos should not draw nested panel outlines, image outlines, selection boxes, or local frames inside the main canvas.

## Sidebar Text

Sidebar text should be compact but readable. It wraps inside the text region and keeps a right margin.

- `TOYBOX_INFO_LINE_HEIGHT = 25.3`
- `TOYBOX_TEXT_RIGHT_PADDING = 14.0`
- `TOYBOX_TEXT_SCROLL_WIDTH = 8.0`

The `Key:` part of a `Key: Value` line is bold. Wrapped continuation lines start at the normal text left edge.

## Digit Browsing

Any demo browsing the bundled digits dataset should use the shared `Digits8Browser` semantics:

- Left/right arrows move through variations of the current digit.
- Up/down arrows move through digit classes while keeping the same variation slot when possible.
- `G` may select a random example.

## Palette

Use the shared palette exactly. Main/secondary pairs are used together: the outer diamond uses the main color and the inner diamond uses the matching secondary color.

Neutrals:

- `COLOR_DARK_NEUTRAL = (29, 32, 36)`
- `COLOR_SLATE_GRAY = (103, 107, 114)`
- `COLOR_FOG_GRAY = (232, 234, 237)`
- `COLOR_LIGHT_NEUTRAL = (245, 246, 248)`

Class and group pairs:

- Aqua / Deep Teal: `(102, 204, 193)` / `(38, 110, 105)`
- Coral / Brick Red: `(240, 128, 112)` / `(150, 62, 54)`
- Blue / Navy: `(66, 133, 244)` / `(26, 92, 173)`
- Leaf Green / Forest Green: `(102, 187, 106)` / `(56, 142, 60)`
- Purple / Deep Purple: `(171, 71, 188)` / `(123, 31, 162)`
- Sand / Ochre: `(214, 188, 133)` / `(166, 133, 82)`
- Walnut / Bark: `(141, 110, 99)` / `(93, 64, 55)`

## Point Markers

Point markers are four-sided diamonds. They are clipped by the shared renderer: data outside the view range is culled, and marker centers are inset so marker shapes cannot spill outside the main or secondary regions.

Marker sizes:

- `small`: outer radius `5.0`, inner radius `3.0`
- `regular`: outer radius `7.0`, inner radius `4.2`

Embed uses `regular` markers. Grad and Diffuse use `small` markers for denser point clouds.

Diamonds always use the two-layer treatment: outer main color, inner paired secondary color. Do not draw fill-only point diamonds in demos.

## Network Lines

Network connection lines use neutral palette colors only and a fixed design-language width:

- `TOYBOX_CONNECTION_LINE_WIDTH = 1.5`

Use opacity to encode strength instead of changing hue or thickness. Preferred stepped opacity levels are 0%, 25%, 50%, and 75%.

## Charts

Histogram and chart bars use neutral Slate Gray tracks/fills. Do not use class colors for bars unless a chart is explicitly comparing class categories.

- `TOYBOX_CHART_TRACK_ALPHA = 70`
- `TOYBOX_CHART_FILL_ALPHA = 210`

## Demo Guidance

When adding or changing a demo:

- Use `window.layout(...)` and shared drawing helpers from `core.arcade_view`.
- Keep the main canvas to one coherent visual.
- Use the secondary region for one small supporting visual.
- Use the text region for state, metrics, and labels only.
- Prefer existing colors and markers before adding anything new.
- Add new core visual primitives only when multiple demos will benefit from them.
