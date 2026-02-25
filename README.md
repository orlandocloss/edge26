# Edge26 - On-Device Insect Detection

On-device insect detection, tracking, and classification for Raspberry Pi with Hailo accelerator.

Uses [**BugSpot**](https://github.com/orlandocloss/bugspot) for motion-based detection and tracking, with Hailo HEF models for hierarchical classification (Family / Genus / Species).

## Quick Start

```bash
# Edit configuration
nano config/settings.yaml

# Run pipeline
python main.py --config config/settings.yaml

# Ctrl+C once  → Stop recording, finish processing queue
# Ctrl+C twice → Force stop immediately
```

## Architecture

Two device types work together:

- **FLICK** — Full pipeline: record, detect, track, classify. Also classifies data received from DOTs.
- **DOT** — Detection only: record, detect, track, extract crops/composites. Sends results to a FLICK for classification.

```
Recording Thread                    Processing Thread
      │                                   │
      ▼                                   ▼
┌──────────────┐                 ┌──────────────────────┐
│ Camera       │    .mp4 files   │ Phase 1-4: BugSpot   │
│ → 60s chunks ├────────────────▶│   Detection          │
│              │                 │   Tracking            │
└──────────────┘                 │   Topology            │
                                 │   Crops & Composites  │
                  input_storage  ├──────────────────────┤
                                 │ Phase 5-6: Hailo      │
  DOT devices      DOT dirs     │   Classification      │
  (crops only) ────────────────▶│   Aggregation         │
                                 └──────────┬───────────┘
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │ output/results/       │
                                 │   {device}/{datetime} │
                                 │   crops + composites  │
                                 │   results.json        │
                                 └──────────────────────┘
```

FLICK videos pass through the full pipeline (phases 1–6). DOT directories arrive in the same `input_storage` and skip straight to classification (phases 5–6 only) — they never touch the BugSpot tracker, so continuous tracking across FLICK videos is unaffected.

### Input Storage Naming

The pipeline identifies items in `input_storage` by naming convention:

```
input_storage/
├── flick01_20260204_100000.mp4          # FLICK video: {flick_id}_{YYYYMMDD}_{HHMMSS}.mp4
├── flick01_20260204_100100.mp4
├── dot01_20260204_120000/               # DOT directory: {dot_id}_{YYYYMMDD}_{HHMMSS}/
│   ├── dot01_20260204_120000_crops/     #   {name}_crops/{track_id}/frame_NNNNNN.jpg
│   │   ├── a1b2c3d4/
│   │   │   ├── frame_000042.jpg
│   │   │   └── frame_000085.jpg
│   │   └── e5f6g7h8/
│   │       └── ...
│   └── dot01_20260204_120000_composites/  #   {name}_composites/
│       └── ...
└── .last_recording                      # Tracker reset marker (auto-managed)
```

- **FLICK videos** — `{flick_id}_{YYYYMMDD}_{HHMMSS}.mp4`. Created by the recorder, matched by `.mp4` extension.
- **DOT directories** — `{dot_id}_{YYYYMMDD}_{HHMMSS}/`. Matched by prefix against `device.dot_ids`. Must contain a `{name}_crops/` subdirectory with track folders of crop images.

## Pipeline Phases

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | BugSpot | **Detection** — GMM background subtraction, morphological filtering, shape/cohesiveness filters |
| 2 | BugSpot | **Tracking** — Hungarian algorithm matching with lost track recovery |
| 3 | BugSpot | **Topology** — Path analysis confirms insect-like movement vs plants/noise |
| 4 | BugSpot | **Crops & Composites** — Re-reads video to extract per-track crop images and composite visualisations |
| 5 | Hailo | **Classification** — HEF model predicts Family, Genus, Species per crop |
| 6 | edge26 | **Aggregation** — Hierarchical voting across frames (best family → best genus within → best species within) |

## Continuous Tracking

When `continuous_tracking: true`, the BugSpot tracker persists across video chunk boundaries — the same insect keeps the same track ID even if it spans multiple 60-second videos.

The tracker resets automatically on three boundaries to avoid stale state:

| Trigger | What happens |
|---------|-------------|
| **Day change** | Date in filename changes (e.g. `20260204` → `20260205`) → full reset before processing |
| **Recording stops** | Ctrl+C or end of session → `.last_recording` marker written to disk → tracker resets after that video is processed |
| **Restart after crash** | On startup, reads `.last_recording` marker → resets at the correct boundary (or immediately if the marked video was already processed) |

The `.last_recording` marker file persists in `input_storage/` so the boundary is preserved even if the pipeline crashes and restarts.

Set `continuous_tracking: false` for fully independent per-video processing (full reset between every video).

## Recording Modes

| Mode | Config | Behaviour |
|------|--------|-----------|
| **Continuous** | `recording_mode: "continuous"` | Record non-stop, chunk after chunk, no gaps |
| **Interval** | `recording_mode: "interval"` | Record one chunk, release camera, wait `recording_interval_minutes`, repeat |

## Configuration

All settings in `config/settings.yaml`:

```yaml
device:
  flick_id: "flick01"                 # Device identifier
  dot_ids: ["dot01", "dot02"]         # Known DOT devices

pipeline:
  enable_recording: true              # Record video
  enable_processing: true             # Detect + classify
  enable_classification: true         # false = detection only (N/A predictions)
  continuous_tracking: true           # Tracker persists across video chunks
  recording_mode: "continuous"        # "continuous" or "interval"
  recording_interval_minutes: 5       # Interval mode only

paths:
  input_storage: "/mnt/usb/videos"    # FLICK videos + DOT directories

capture:
  fps: 30
  chunk_duration_seconds: 60

classification:
  model: "data/models/classifier.hef"
  labels: "data/models/labels.txt"

tracking:
  max_lost_frames: 45                 # ~1.5s at 30fps
  cost_threshold: 0.3
```

## Output Structure

Results are organised by device and timestamp:

```
output/results/
├── flick01/
│   └── 20260204_100000/
│       ├── crops/{track_id}/frame_000075.jpg
│       ├── composites/
│       └── results.json
├── dot01/
│   └── 20260204_120000/
│       ├── crops/
│       ├── composites/
│       └── results.json
└── processing_summary.json
```

## Directory Structure

```
edge26/
├── main.py                         # Entry point + Pipeline orchestrator
├── config/
│   └── settings.yaml               # ← EDIT THIS
├── src/
│   ├── capture/recorder.py         # Video recording (continuous/interval)
│   ├── processing/
│   │   ├── processor.py            # Pipeline orchestrator (BugSpot + Hailo)
│   │   └── classifier.py           # Hailo HEF classification + taxonomy
│   └── output/writer.py            # JSON results writer
├── data/models/                    # Place HEF model + labels here
└── output/
    ├── results/                    # Per-device/timestamp results
    └── logs/                       # Daily log files
```

## Model Files

Place in `data/models/`:
- `classifier.hef` — Compiled Hailo model
- `labels.txt` — Species names, one per line, same order as training
