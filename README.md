# Edge26 - On-Device Insect Detection

Continuous video capture with asynchronous insect detection, classification, and tracking.

Uses [**BugSpot**](../bugspot/) for the shared detection and tracking pipeline, with Hailo HEF models for on-device classification.

## Quick Start

```bash
# 1. Edit configuration
nano config/settings.yaml

# 2. Run pipeline
python main.py --config config/settings.yaml

# Ctrl+C once  → Stop recording, finish processing queue
# Ctrl+C twice → Force stop immediately
```

## Architecture

```
Recording Thread                    Processing Thread
      │                                   │
      ▼                                   ▼
┌──────────────┐     Queue      ┌──────────────────────┐
│ Camera       │───────────────▶│ Phase 1: Detection   │
│ → 1min chunk │                │ Phase 2: Topology    │
│ → USB storage│                │ Phase 3: Classify    │
└──────────────┘                │ Phase 4: Aggregate   │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │ JSON + Crops         │
                                │ Delete video         │
                                └──────────────────────┘
```

## Pipeline Phases

1. **Detection** - Motion-based (GMM) with shape/cohesiveness filters *(BugSpot)*
2. **Tracking** - Hungarian algorithm with lost track recovery *(BugSpot)*
3. **Topology Analysis** - Confirms insects by path characteristics *(BugSpot)*
4. **Crop Extraction + Composites** - Per-track outputs *(BugSpot)*
5. **Classification** - Hailo HEF model (Family/Genus/Species) *(edge26)*
6. **Aggregation** - Hierarchical voting across frames *(edge26)*

Tracks persist across video boundaries (same insect = same ID).

## Directory Structure

```
edge26/
├── main.py                     # Entry point
├── config/
│   └── settings.yaml           # ← EDIT THIS
├── src/
│   ├── capture/recorder.py     # Video recording
│   ├── processing/
│   │   ├── detector.py         # Motion detection + topology
│   │   ├── classifier.py       # Hailo classification
│   │   ├── tracker.py          # Hungarian tracking
│   │   └── processor.py        # Pipeline orchestrator
│   └── output/writer.py        # JSON + crops
├── data/models/                # Place HEF weights here
└── output/
    ├── results/                # JSON output
    └── logs/                   # Log files
```

## Configuration

All settings in `config/settings.yaml`. Key sections:

```yaml
pipeline:
  enable_recording: true             # Record video
  enable_processing: true            # Detection/tracking/classification
  # Both = full pipeline (default)
  # Recording only = just capture video
  # Processing only = process existing videos

paths:
  video_storage: "/mnt/usb/videos"   # External USB

capture:
  fps: 30
  chunk_duration_seconds: 60

detection:
  min_area: 200                      # Shape filters
  min_displacement: 50               # Topology filters

classification:
  model: "data/models/classifier.hef"

tracking:
  max_lost_frames: 45                # ~1.5s at 30fps
  cost_threshold: 0.3
```

## Output Format

JSON per video with hierarchical predictions:

```json
{
  "video_file": "edge26_20260202_143000.mp4",
  "summary": {
    "confirmed_tracks": 3,
    "total_tracks": 12
  },
  "tracks": [{
    "track_id": "uuid",
    "final_prediction": {
      "family": "Apidae",
      "genus": "Apis",
      "species": "Apis mellifera",
      "species_confidence": 0.87
    },
    "frames": [{
      "frame_number": 75,
      "timestamp_seconds": 2.5,
      "bbox": [120, 340, 180, 400],
      "prediction": {...}
    }]
  }]
}
```

## Model Files

Place in `data/models/`:
- `classifier.hef` - Hailo model (labels embedded in model metadata)
