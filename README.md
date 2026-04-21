# Pupa Counter (v12)

Automatic silkworm-pupa counter for 300 dpi paper-sheet scans, built on a
lightweight U-Net (466K params). **v12 default** (fresh-trained on 99
hand-corrected scans, ~10,144 sure labels after multiple rounds of label
cleanup + scanner black-border handling, with per-pixel spatial loss
weighting). v6/v7/v11 shipped as reference checkpoints.

One command per scan: you get an annotated PNG and a running Excel log of
counts. No cloud service, no API key, all inference runs locally on CPU,
CUDA, or Apple Silicon (MPS).

```
python pupa_counter.py examples/example_scan.png
```

![example](examples/example_output.png)

## What it does

For every scan you feed it, the program:

1. Runs the v12 CNN and extracts candidate pupa center locations.
2. **Runs a 2nd-stage classifier on each candidate to drop false
   positives** (stains, ink shadows, dirt). Disable with `--no-filter`.
3. Draws a green dot on each confirmed pupa.
4. Computes **three rank percentile lines** at 5%, 25%, 75% of the
   detected-pupa Y range (not the image height — the topmost/bottommost
   detected pupa define 0% / 100%).
5. Reports pupa counts in four bands.
6. Saves the annotated image to `output/`.
7. Appends two rows to `output/pupa_counts.xlsx`:
   - **Summary sheet** (`Pupa counts`): one row per scan with counts and
     band breakdown.
   - **Per-pupa detail sheet** (`Per pupa detail`): one row per detected
     pupa with exact `(x_pixel, y_pixel, rank_pct, band, image_height)`.
8. At the end of a batch run, writes `output/rank_distribution.png` — a
   global rank histogram plus per-scan small multiples so you can see
   how each vial's pupae are distributed vertically.

The rank convention is purely positional — no value judgement:

| Image Y-coord | Rank % | Where the pupa sits in the image |
|---|---|---|
| Bottommost pupa  | 0%   | Image BOTTOM  (largest y) |
| Topmost pupa     | 100% | Image TOP     (smallest y) |

Four regions & counts reported:

| Band       | Meaning |
|---|---|
| Rank 0 - 5%    | Top of ranking — pupae at image bottom (red line) |
| Rank 5 - 25%   | Next 20% |
| Rank 25 - 75%  | Middle 50% |
| Rank 75 - 100% | Bottom of ranking — pupae at image top |

The 5% line is drawn in red; 25% and 75% in orange.

## Accuracy

Two-stage pipeline: CNN detector + peak-level classifier.

| Pipeline | Self-eval F1 on 99 scans (cleaned labels) | Precision | Recall |
|---|---|---|---|
| v12 CNN only | 98.30% | 97.69% | 98.93% |
| **v12 CNN + classifier filter** (default) | **99.46%** | **100.00%** | **98.93%** |
| v11 CNN + classifier v4 (legacy) | 99.42% | 100.00% | 98.85% |
| v6 CNN + classifier v3 (legacy) | 99.23% | 99.87% | 98.60% |

The 2nd-stage classifier (`model/peak_filter_clf.pkl`, ~360 KB) is a
Gradient Boosting model trained on v12's own detection outputs across
all 99 labeled scans. At the default `threshold=0.60` it kills **100% of
false positives** (238 → 0 across 99 scans) while losing **zero** true
positives. Net F1 gain over raw CNN: **+1.16pp**.

The 109 remaining misses break down by precise per-scan black-border
measurement (border width is typically 6-8 px, measured automatically):

| Region | Pupae | Misses | Miss rate |
|---|---|---|---|
| Interior (≥10 px from any edge) | 9,972 | 42 | **0.42%** |
| Paper edge zone (<10 px from edge) | 172 | 67 | **38.95%** |

Interior miss rate of 0.42% is within human-recount agreement. The edge
zone is where most remaining errors concentrate — these pupae either
have very different visual context (background transitions from white
paper to the scanner black border) or are physically truncated (the
scanner cut them in half). Only 1.7% of all pupae sit in this zone, so
the model has seen few examples. If future scans keep pupae fully on
the paper away from the left/right image edges, this error source
disappears entirely.

**Why v12 over v11:** v12 was re-trained fresh (random init, no warm
start) on 99 cleaned scans after the cleanup rounds described below.
The training loss uses a per-pixel spatial underpred weighting (0.75
near the measured left border, ramping to 1.5 on paper interior) so the
model does not waste gradient capacity on the physically unrecoverable
half-cut pupae. The classifier was then retrained on v12's outputs with
the final cleaned labels. Combined gain over v11 on clean self-eval:
**+0.17 pp F1** — small in absolute terms but honest (pre-cleanup v11
numbers were slightly inflated by 26 top-edge artifact labels that have
since been removed).

**Label cleanup rounds applied (all included in the 10,144 sure labels):**
1. Active-learning diff review: 134 model-vs-user disagreements triaged;
   10 accepted (9 deletes + 1 add).
2. Scanner black-border handling: 2 out-of-image labels removed; 61
   labels inside the 6-8 px black border moved to the first paper
   pixel (they represent half-cut pupae whose exact center fell in
   black).
3. Top-edge artifact review: 26 sure labels with gy < 20 (top 0.8% of
   image) were inspected in a dedicated contact-sheet tool and deleted
   — these were scanner timestamps, thin horizontal lines, or faint
   shadows mislabeled as pupae.

v6 (`model/pupa_counter_v6.pt`), v7 (`model/pupa_counter_v7.pt`), and
v11 (`model/pupa_counter_v11.pt`) are retained as reference checkpoints.
Pair them with their matching classifier (`peak_filter_clf_v3.pkl` for
v6, `peak_filter_clf_v4.pkl` for v11) to reproduce prior numbers.

**Runtime on Apple M4 (10-core MPS):** ~0.65 s per scan total (~0.6 s
for CNN + ~0.05 s for classifier feature extraction + inference).

Disable the classifier entirely with `--no-filter` (returns F1 ≈ 98.30%
raw CNN output).

---

## Install

Tested on Python 3.9+. Clone and set up a virtual env:

```bash
git clone https://github.com/sgaofen/pupa_counter_v6.git
cd pupa_counter_v6

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `opencv-python`,
`scikit-image`, `openpyxl`, `matplotlib`. The requirements file pins
minimums only, so any recent version works. (`matplotlib` is only used
to draw the end-of-batch rank-distribution plot; if missing, the plot
is skipped gracefully and everything else still runs.)

> **Apple Silicon users** already get MPS acceleration for free.
> **CUDA users** will pick up GPU automatically if `torch.cuda.is_available()`.
> **CPU only** is supported (takes ~2-3 s per scan instead of 0.6 s).

---

## Usage

### Single image

```bash
python pupa_counter.py path/to/scan.png
```

Outputs:
- `output/scan_counted.png` — the annotated image
- `output/pupa_counts.xlsx` — Excel row appended

### Batch a directory

```bash
python pupa_counter.py path/to/scans_folder/
```

Every image in the folder is processed in filename order; the Excel keeps
accumulating rows.

### Custom paths

```bash
python pupa_counter.py scans/ \
    --out results/ \
    --excel results/my_counts.xlsx \
    --model model/pupa_counter_v12.pt
```

### Output Excel columns

The Excel has **two sheets**. Both are appended to on every run; delete
the file to start fresh.

**Sheet 1 — `Pupa counts`** (one row per scan):

| Column | Meaning |
|---|---|
| `scan_name` | Source filename |
| `timestamp` | ISO-8601 time of processing |
| `total_count` | All pupae detected |
| `top_5_pct_count`        | Pupae at rank 0-5% (image bottom 5%) |
| `rank_5_to_25_pct_count` | Pupae at rank 5-25% |
| `middle_50_pct_count`    | Pupae at rank 25-75% (middle 50%) |
| `bottom_25_pct_count`    | Pupae at rank 75-100% (image top 25%) |
| `y_min_of_pupae`, `y_max_of_pupae` | Pixel Y coordinates used for percentile calc |
| `image_width`, `image_height`      | Original scan dimensions |

**Sheet 2 — `Per pupa detail`** (one row per detected pupa):

| Column | Meaning |
|---|---|
| `scan_name`     | Source filename |
| `pupa_idx`      | 1-based index within its scan |
| `x_pixel`       | X coordinate of detected center (pixels) |
| `y_pixel`       | Y coordinate of detected center (pixels) |
| `rank_pct`      | 0 = image bottom, 100 = image top (float, 2 decimals) |
| `band`          | `"0-5%"` / `"5-25%"` / `"25-75%"` / `"75-100%"` |
| `image_height`  | Original scan height (for normalizing `y_pixel` if desired) |

### End-of-batch distribution plot

After every batch run, `pupa_counter.py` writes
`output/rank_distribution.png`. It has two panels:

1. A global rank-percentile histogram across all pupae in the batch,
   with dashed vertical lines at the 5% / 25% / 75% band boundaries.
2. Per-scan small multiples — one mini-histogram per scan so you can
   see how each vial's pupae cluster along the rank axis (uniform,
   bimodal, skewed, etc.).

---

## How it works (short version)

1. **Tiled inference**: the full scan is tiled into 256×256 patches with 192
   px stride, each patch is fed to the CNN, overlapping predictions are
   averaged into a single heatmap.
2. **Peak extraction**: `skimage.feature.peak_local_max` with
   `min_distance=8` and `threshold_abs=0.15` pulls the centers out of the
   heatmap.
3. **Region count + render**: horizontal split lines at 25% and 75%, counts
   per band, annotations overlaid, saved as PNG and Excel row.

Architecture is a compact U-Net (466K params):

```
Input (RGB patch, 256×256)
  Encoder: 3 → 32 → 64 → 128 channels, two 2× downsamples
  Bottleneck
  Decoder: 128 → 64 → 32 channels, two 2× upsamples with skip connections
Output: 1-channel sigmoid heatmap
```

Training (v12 default): fresh random init on 99 hand-labeled scans
(~10,144 sure labels after three rounds of label cleanup), 50 epochs
with cosine-annealed Adam (lr 7e-4 → 7e-5). Loss is weighted MSE +
under-prediction (recall) + over-prediction (anti-FP) terms, **with a
per-pixel spatial weight on the under-prediction term**: 0.75× near the
measured left black scanner border, ramping to 1.5× on paper interior.
This directs gradient capacity to recoverable pupae instead of the
physically-truncated edge cases. Best checkpoint (ep 48) saved by
lowest non-edge miss count.

Prior checkpoints for reference: v6 was fresh on 60 scans (~6,300
labels) at lr 1e-3 → 1e-4; v11 was warm-started from v6 on 99 cleaned
scans at lr 3e-4 → 3e-5.

---

## Tuning

If the model under- or over-counts on your scans, you can edit the constants
at the top of `pupa_counter.py`:

```python
PEAK_THRESHOLD = 0.15   # lower = more detections (higher recall, more FPs)
PEAK_MIN_DIST  = 8      # smaller = allows closer peaks (better for dense clusters)
PATCH_SIZE     = 256    # don't change unless you retrain
STRIDE         = 192    # smaller stride = slower but smoother heatmap
```

A threshold of `0.10` is a good try if you see many pupae missed; `0.20` if
many blank-area false positives appear.

---

## Troubleshooting

**"No module named 'openpyxl'"**
Install it: `pip install openpyxl`

**"model weights not found"**
The repo ships `model/pupa_counter_v12.pt` (1.9 MB). Check it wasn't skipped
during clone (LFS-style). Alternatively pass `--model /path/to/weights.pt`.

**Really slow on CPU**
Expected — tiled inference over a 1000×2500 scan on CPU is 2-3 s. Use
MPS (Mac) or CUDA for ~0.6 s.

**Results look noisy on my scans**
The model was trained on 300 dpi, white-paper, tan/brown pupae on
blue-ink sheets. Very different lighting, different species, or a
different scanner will degrade accuracy. You'd need to label a few
dozen of your own scans and fine-tune — see `HANDOFF_2026-04-16.md` in
this repo for the training recipe, data paths, and checkpoint
inventory.

---

## License

MIT — use it, fork it, improve it. Acknowledgement is appreciated but not
required.

## Credits

Built by Stephen Yu with AI-assisted iteration. Trained on data hand-
corrected over one night of active-learning labeling.
