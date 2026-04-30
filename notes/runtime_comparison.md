# Runtime Comparison — SAM3 vs GS2

## Setup

- Image: `data/images/htx.png`
- Prompt: `"objects."`
- Thresholds: `SAM3_CONF_THRESH=0.2`, `GS2_BOX_THRESH=0.2`, `GS2_TEXT_THRESH=0.2`
- Both timers start **after model loading**, at the first model forward pass (`set_image`)
- Both timers start **after image file I/O** (`Image.open` / `load_image` outside timer)

## Timer Scope

What each timer measures:

| Step | SAM3 | GS2 |
|---|---|---|
| Model loading | ❌ excluded | ❌ excluded |
| Image file I/O | ❌ excluded | ❌ excluded |
| Image encode (`set_image`) | ✅ included | ✅ included |
| Detection | ✅ included (unified pass) | ✅ included (GroundingDINO) |
| Mask decode | ✅ included (unified pass) | ✅ included (SAM2) |

## Order Effect Test

To check whether GPU boost clocks from the first run advantage the second model,
the pipeline was run in both orders.

| Run order | SAM3 runtime | GS2 runtime |
|---|---|---|
| SAM3 first (default) | 0.276s | 0.588s |
| GS2 first (swapped) | 0.297s | 0.577s |
| Delta | +21ms when second | −11ms when first |

Order effect is **real but small** (~10–20ms). Both models are marginally faster
when running second (warm GPU clocks), and the effect is roughly symmetric.

## Conclusion

The ~2× runtime gap (SAM3 ~0.28s vs GS2 ~0.58s) is **architectural**, not a
measurement artifact:

- SAM3 is a single unified model (one forward pass: encode + detect + segment)
- GS2 runs two models in sequence: GroundingDINO (detect) → SAM2 (segment)

The order effect (~20ms) is negligible relative to the actual gap (~300ms).
Timing comparison is fair.

---

## N-Runs Median Benchmark — 2026-04-30 15:20

- **N**: 5 runs per image per model
- **Prompt**: `"objects."`
- **Thresholds**: `SAM3_CONF_THRESH=0.2`, `GS2_BOX_THRESH=0.2`, `GS2_TEXT_THRESH=0.2`
- **Order**: alternated per run (even=SAM3 first, odd=GS2 first) to cancel GPU boost-clock bias
- **Images**: 5 (groceries.jpg, htx.png, htx2.png, kids.jpg, truck.jpg)

| Image | SAM3 median | GS2 median | GS2/SAM3 ratio |
|---|---|---|---|
| `groceries.jpg` | 0.288s | 0.575s | 1.99x |
| `htx.png` | 0.284s | 0.575s | 2.02x |
| `htx2.png` | 0.287s | 0.585s | 2.04x |
| `kids.jpg` | 0.287s | 0.563s | 1.96x |
| `truck.jpg` | 0.289s | 0.564s | 1.95x |
| **Overall** | **0.287s** | **0.575s** | **2.00x** |

Raw per-image times:

- `groceries.jpg`: SAM3=[0.288, 0.296, 0.304, 0.284, 0.284]  GS2=[0.580, 0.575, 0.582, 0.571, 0.574]
- `htx.png`: SAM3=[0.281, 0.276, 0.299, 0.284, 0.284]  GS2=[0.576, 0.575, 0.563, 0.569, 0.575]
- `htx2.png`: SAM3=[0.287, 0.287, 0.277, 0.282, 0.299]  GS2=[0.585, 0.577, 0.596, 0.578, 0.586]
- `kids.jpg`: SAM3=[0.279, 0.290, 0.301, 0.277, 0.287]  GS2=[0.576, 0.567, 0.563, 0.563, 0.558]
- `truck.jpg`: SAM3=[0.291, 0.279, 0.289, 0.282, 0.289]  GS2=[0.562, 0.571, 0.575, 0.562, 0.564]
