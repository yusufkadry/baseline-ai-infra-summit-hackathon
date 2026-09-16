# BASELINE WATCH -- PREVIOUSLY "TRIPEWIRE" BUT REBRANDED AS THERE IS A SECURITY SYSTEM ALREADY CALLED TRIPWIRE. REFER TO TRIPWIRE AS "Baseline"

Baseline WATCH is a fixed-camera scene monitor for SiMa.ai Neat and Modalix. It
divides each frame into a stable grid, asks an image-capable vision-language
model (VLM) to describe permanent physical state during calibration, and then
compares later frames against that baseline.

The application is designed for state changes such as moved or missing objects,
changed orientation, and connected-versus-disconnected cables. A per-tile pixel
difference gate avoids unnecessary VLM requests when a tile has not changed.

## Features

- Deterministic row/column tile IDs such as `T_r2c5`.
- Static-image and live-frame calibration modes.
- Static-image and continuous live WATCH modes.
- One-image/one-baseline WATCH requests for spatial isolation.
- Normalized RMS pixel-difference gating per tile.
- Structured `OBSERVATION`, `REASON`, and `VERDICT` model responses.
- Parse-failure containment so one malformed tile does not abort a pass.
- Atomic JSON status snapshots for a companion application.
- Unit coverage for CLI, grid, calibration, parsing, gating, and live IPC.

## Repository layout

| Path | Purpose |
| --- | --- |
| `tripwire.py` | Main CLI, calibration, inference, parsing, and live-loop implementation. |
| `region_sheet.py` | Compatibility entry point that invokes one-shot WATCH mode. |
| `test_tripwire_live.py` | Unit and IPC contract tests. |
| `baseline.json` | Saved 6x8 grid baseline metadata from this snapshot. |
| `baseline.txt` | Human-readable baseline summary. |
| `calibration_frames/` | Calibration frames preserved from the backup. |
| `live/` | Captured live frame and latest tile-status snapshot. |
| `detections.jsonl` | Preserved runtime detection log. |
| `region_sheet_test.jpg` | Preserved test visualization. |

This repository package intentionally preserves every file from the
`tripwire-CLEAN-0FAIL` backup. Python bytecode caches and a zero-byte temporary
JPEG are present for archival completeness, although `.gitignore` prevents new
copies from being committed.

## Requirements

The inference paths require:

- A SiMa.ai Modalix DevKit, or the Neat Development Environment connected to one.
- The public `pyneat` Python package supplied with Neat.
- Python 3.11 or 3.12.
- OpenCV (`cv2`) and NumPy.
- An image-capable deployed VLM directory.
- A YAML application config containing `grid.rows` and `grid.columns`.

The default model path in this snapshot is:

```text
/media/nvme/llima/models/Qwen3-VL-2B-Instruct-GPTQ-a16w4
```

The default shared application config is:

```text
/workspace/labs/agent-neat/config/config.yaml
```

Use explicit `--model` and `--config` arguments when your paths differ.

## Important snapshot note

The preserved `baseline.json` names `baseline_reference.jpg` as its reference
frame, but that JPEG was not present in the `tripwire-CLEAN-0FAIL` backup.
One-shot or live WATCH requires a valid reference frame next to the baseline.
Run calibration to regenerate both files before relying on the included
baseline:

```bash
python3 tripwire.py calibrate FRAME_1.jpg FRAME_2.jpg FRAME_3.jpg \
  --baseline /workspace/tripwire/baseline.json \
  --config /workspace/labs/agent-neat/config/config.yaml
```

## Grid configuration

The baseline grid must match the companion application's YAML config. The
included baseline uses 6 rows and 8 columns:

```yaml
grid:
  rows: 6
  columns: 8
```

If the grid changes, recalibrate. WATCH intentionally rejects a baseline whose
grid dimensions do not match the shared config.

## Run tests

The test suite can run without a live model. Image-processing tests require the
real NumPy/OpenCV runtime; otherwise those tests are skipped cleanly.

```bash
cd /workspace/tripwire
python3 -m unittest -v test_tripwire_live.py
```

Show all CLI modes:

```bash
python3 tripwire.py --help
python3 tripwire.py watch --help
```

## Run on a Modalix DevKit

From the Neat Development Environment, confirm the DevKit connection:

```bash
bash -lic 'dk status'
```

The project must remain under `/workspace` for the shared SDK/DevKit workflow.
The `dk` runner activates the DevKit's `pyneat` environment and streams
stdout/stderr back to the SDK shell.

### Calibrate from image files

```bash
dk /workspace/tripwire/tripwire.py calibrate \
  /workspace/assets/cal_01.jpg \
  /workspace/assets/cal_02.jpg \
  /workspace/assets/cal_03.jpg \
  --baseline /workspace/tripwire/baseline.json \
  --config /workspace/labs/agent-neat/config/config.yaml
```

Calibration writes `baseline.json` and `baseline_reference.jpg` together.

### Watch one or more image files

```bash
dk /workspace/tripwire/tripwire.py watch \
  /workspace/assets/test_normal.jpg \
  --baseline /workspace/tripwire/baseline.json \
  --config /workspace/labs/agent-neat/config/config.yaml \
  --diff-threshold 0.25
```

Set `--diff-threshold 0.0` to force VLM inference for every tile. The default
is `0.25`.

### Calibrate from live frame snapshots

A companion producer must atomically publish JPEG frames to the configured
frame path.

```bash
dk /workspace/tripwire/tripwire.py calibrate-live \
  --frame-source /workspace/tripwire/live/latest_frame.jpg \
  --baseline /workspace/tripwire/baseline.json \
  --config /workspace/labs/agent-neat/config/config.yaml \
  --frame-count 3
```

### Run continuous WATCH

```bash
dk /workspace/tripwire/tripwire.py watch-live \
  --frame-source /workspace/tripwire/live/latest_frame.jpg \
  --status-output /workspace/tripwire/live/tile_status.json \
  --baseline /workspace/tripwire/baseline.json \
  --config /workspace/labs/agent-neat/config/config.yaml \
  --watch-interval 5
```

For a smoke test, append `--max-passes 1`.

## Output model

Each tile produces:

- `observation`: the current visible permanent state.
- `reason`: comparison with the matching baseline statement.
- `verdict`: `MATCH`, `DEVIATION`, or `UNCERTAIN`.
- `status`: `normal`, `deviation`, or `uncertain`.
- `normalized_bbox`: tile bounds in normalized `xywh` coordinates.

WATCH prints a complete `WATCH_RESULT_JSON` record after each pass. Live mode
also atomically replaces `live/tile_status.json`, allowing another process to
consume only complete snapshots.

## Data size and GitHub

The preserved backup is about 356 MB, of which about 351 MB is calibration
JPEGs. No individual file exceeds GitHub's 100 MB per-file limit, but cloning
the repository will still be relatively heavy. If the image history will grow,
move `calibration_frames/*.jpg` to Git LFS before adding more data.

## License

No license was included in the source backup. Add an appropriate `LICENSE`
file before publishing if you intend to grant reuse or redistribution rights.
