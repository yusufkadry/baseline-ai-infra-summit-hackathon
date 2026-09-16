#!/usr/bin/env python3
"""Calibrate and watch a fixed scene using stable spatial grid tiles."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

try:
    import cv2
    import numpy as np
except ImportError:
    # Keep CLI discovery and IPC-only tests usable on hosts without the
    # Modalix vision packages. Image-processing modes still require them.
    cv2 = None
    np = None


DEFAULT_MODEL = Path(
    "/media/nvme/llima/models/Qwen3-VL-2B-Instruct-GPTQ-a16w4"
)
DEFAULT_BASELINE = Path("/workspace/tripwire/baseline.json")
DEFAULT_LIVE_FRAME = Path("/workspace/tripwire/live/latest_frame.jpg")
DEFAULT_STATUS_OUTPUT = Path("/workspace/tripwire/live/tile_status.json")
DEFAULT_APP_CONFIG = Path("/workspace/labs/agent-neat/config/config.yaml")
DEFAULT_TILE_SIZE = 448
DEFAULT_DIFF_THRESHOLD = 0.25
MAX_IMAGES_PER_REQUEST = 12
DEFAULT_LIVE_CALIBRATION_FRAMES = 3
DEFAULT_CALIBRATION_INTERVAL_SECONDS = 5.0
DEFAULT_WATCH_INTERVAL_SECONDS = 5.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.1
MAX_GENERATION_TOKENS = 512
CALIBRATION_TOKENS_PER_TILE = 48
WATCH_MAX_NEW_TOKENS = 200
MERGE_TOKENS = 80
VALID_VERDICTS = frozenset({"MATCH", "DEVIATION", "UNCERTAIN"})

CALIBRATION_LINE_RE = re.compile(
    r"^\s*(T_r\d+c\d+)\s*\|\s*DESCRIPTION\s*=\s*(.+?)\s*$",
    re.IGNORECASE,
)
WATCH_VERDICT_LINE_RE = re.compile(r"^\s*VERDICT\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
WATCH_REASON_LINE_RE = re.compile(r"^\s*REASON\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
WATCH_OBSERVATION_LINE_RE = re.compile(
    r"^\s*OBSERVATION\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)
WATCH_REASON_PREFIX_RE = re.compile(
    r"^(CONTRADICTS|DOES NOT CONTRADICT|UNCERTAIN):\s*(.+)$",
    re.IGNORECASE,
)
WATCH_COMPLETE_VERDICT_RE = re.compile(
    r"^\s*VERDICT\s*[:=]\s*"
    r"(?:DOES NOT CONTRADICT|CONTRADICTS|DEVIATION|UNCERTAIN|MATCH)"
    r"(?=\s*(?::|$))",
    re.IGNORECASE | re.MULTILINE,
)
WATCH_VERDICT_ALIASES = {
    "MATCH": "MATCH",
    "DEVIATION": "DEVIATION",
    "UNCERTAIN": "UNCERTAIN",
    "CONTRADICTS": "DEVIATION",
    "DOES NOT CONTRADICT": "MATCH",
}
CONNECTED_STATE_RE = re.compile(
    r"\b(?:connected|plugged(?:\s+in)?|attached)\b",
    re.IGNORECASE,
)
DISCONNECTED_STATE_RE = re.compile(
    r"\b(?:disconnected|unplugged|unattached|detached|loose|hanging loose|"
    r"not connected|not plugged(?:\s+in)?)\b",
    re.IGNORECASE,
)
COMPLETED_SENTENCE_RE = re.compile(r".*?(?:[.!?](?=\s|$)|\r?\n)", re.DOTALL)
CABLE_TERM_RE = re.compile(r"\b(?:cable|cord|plug|connector)\b", re.IGNORECASE)
CONNECTION_STATE_RE = re.compile(
    r"\b(?:connected|disconnected|plugged(?:\s+in)?|unplugged|attached|detached|"
    r"unattached|loose|hanging|connection state (?:is )?(?:unclear|unknown)|"
    r"connection state cannot be determined|ends? (?:leave|leaves|exit|exits)|"
    r"ends? (?:is|are) (?:hidden|occluded|outside))\b",
    re.IGNORECASE,
)
SPATIAL_STATE_RE = re.compile(
    r"\b(?:left|right|upper|lower|top|bottom|center|middle|horizontal|vertical|"
    r"diagonal|upright|flat|facing|oriented|position|runs?|routed?|crosses|lies|"
    r"sits?|stands?|occupies|beside|next to|above|below|under|over|between|"
    r"against|near)\b",
    re.IGNORECASE,
)
NO_PERMANENT_CONTENT_RE = re.compile(
    r"\bno permanent physical (?:content|contents|object|objects)\b",
    re.IGNORECASE,
)


class VisionLanguageModel(Protocol):
    def run(self, request: Any) -> Any: ...

    def stream(self, request: Any) -> Any: ...

    def accepts_image(self) -> bool: ...


@dataclass(frozen=True)
class GridTile:
    tile_id: str
    row: int
    column: int
    normalized_bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class BaselineTile:
    tile_id: str
    normalized_bbox: tuple[float, float, float, float]
    description: str
    observation_count: int


@dataclass(frozen=True)
class GridBaseline:
    rows: int
    columns: int
    image_width: int
    image_height: int
    tiles: dict[str, BaselineTile]
    reference_frame: Path | None = None


@dataclass(frozen=True)
class SharedGridConfig:
    rows: int
    columns: int


@dataclass
class TileAccumulator:
    descriptions: list[str] = field(default_factory=list)
    observation_count: int = 0

    def add(self, description: str) -> None:
        if description not in self.descriptions:
            self.descriptions.append(description)
        self.observation_count += 1


@dataclass(frozen=True)
class TileVerdict:
    verdict: str
    reason: str
    observation: str


@dataclass(frozen=True)
class BatchResult:
    values: dict[str, Any]
    mode: str
    request_count: int
    inference_seconds: float
    raw_responses: tuple[str, ...]
    missing: frozenset[str]


@dataclass(frozen=True)
class FrameStamp:
    """Identity and write time of one atomically published live frame."""

    device: int
    inode: int
    size: int
    mtime_ns: int


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def add_vlm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"deployed VLM directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"grid baseline JSON (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_int,
        default=DEFAULT_TILE_SIZE,
        help=f"square VLM input size for each tile (default: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--force-row-batches",
        action="store_true",
        help=(
            "calibration only: skip its all-tile request and submit one "
            "multi-image request per row; WATCH always uses one tile per request"
        ),
    )


def add_shared_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_APP_CONFIG,
        help=(
            "shared C++/Python application config containing grid.rows and "
            f"grid.columns (default: {DEFAULT_APP_CONFIG})"
        ),
    )


def add_diff_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--diff-threshold",
        type=nonnegative_float,
        default=DEFAULT_DIFF_THRESHOLD,
        metavar="FRACTION",
        help=(
            "normalized per-tile RMS pixel-difference required for VLM inference "
            f"(0.0 sends every tile; default: {DEFAULT_DIFF_THRESHOLD})"
        ),
    )


def add_live_frame_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frame-source",
        type=Path,
        default=DEFAULT_LIVE_FRAME,
        help=f"atomic JPEG published by the C++ app (default: {DEFAULT_LIVE_FRAME})",
    )
    parser.add_argument(
        "--poll-interval",
        type=positive_float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=(
            "seconds between checks while waiting for a new frame "
            f"(default: {DEFAULT_POLL_INTERVAL_SECONDS})"
        ),
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        metavar="IMAGE",
        help="one or more input image paths",
    )
    add_vlm_arguments(parser)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate or watch a fixed scene using stable grid tiles."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="build a grid baseline from calibration frames",
    )
    add_common_arguments(calibrate_parser)
    add_shared_config_argument(calibrate_parser)

    watch_parser = subparsers.add_parser(
        "watch",
        help="compare each image's grid tiles with a saved baseline",
    )
    add_common_arguments(watch_parser)
    add_shared_config_argument(watch_parser)
    add_diff_arguments(watch_parser)

    calibrate_live_parser = subparsers.add_parser(
        "calibrate-live",
        help="sample live C++ frame snapshots and build a grid baseline",
    )
    add_vlm_arguments(calibrate_live_parser)
    add_shared_config_argument(calibrate_live_parser)
    add_live_frame_arguments(calibrate_live_parser)
    calibrate_live_parser.add_argument(
        "--frame-count",
        type=positive_int,
        default=DEFAULT_LIVE_CALIBRATION_FRAMES,
        help=(
            "number of distinct live frames to calibrate from "
            f"(default: {DEFAULT_LIVE_CALIBRATION_FRAMES})"
        ),
    )
    calibrate_live_parser.add_argument(
        "--sample-interval",
        type=positive_float,
        default=DEFAULT_CALIBRATION_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=(
            "minimum interval between captured calibration frames "
            f"(default: {DEFAULT_CALIBRATION_INTERVAL_SECONDS})"
        ),
    )

    watch_live_parser = subparsers.add_parser(
        "watch-live",
        help="periodically judge live C++ frames and publish tile statuses",
    )
    add_vlm_arguments(watch_live_parser)
    add_shared_config_argument(watch_live_parser)
    add_live_frame_arguments(watch_live_parser)
    add_diff_arguments(watch_live_parser)
    watch_live_parser.add_argument(
        "--status-output",
        type=Path,
        default=DEFAULT_STATUS_OUTPUT,
        help=(
            "atomic tile-status snapshot consumed by the C++ app "
            f"(default: {DEFAULT_STATUS_OUTPUT})"
        ),
    )
    watch_live_parser.add_argument(
        "--watch-interval",
        type=positive_float,
        default=DEFAULT_WATCH_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=(
            "minimum interval between WATCH pass starts; inference may take longer "
            f"(default: {DEFAULT_WATCH_INTERVAL_SECONDS})"
        ),
    )
    watch_live_parser.add_argument(
        "--max-passes",
        type=positive_int,
        help="stop after this many completed WATCH passes (default: run forever)",
    )
    return parser.parse_args(argv)


def validate_paths(args: argparse.Namespace) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(f"shared application config does not exist: {args.config}")
    if not args.model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model}")
    if args.mode in {"calibrate", "watch"}:
        for image_path in args.images:
            if not image_path.is_file():
                raise FileNotFoundError(f"image does not exist: {image_path}")
    if args.mode in {"watch", "watch-live"} and not args.baseline.is_file():
        raise FileNotFoundError(
            f"baseline does not exist: {args.baseline}; run calibrate first"
        )


def load_shared_grid_config(path: Path) -> SharedGridConfig:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"could not read shared application config: {path}") from exc

    grid_indent: int | None = None
    values: dict[str, int] = {}
    for raw_line in lines:
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip())
        stripped = content.strip()
        if grid_indent is None:
            if stripped == "grid:":
                grid_indent = indent
            continue
        if indent <= grid_indent:
            break
        match = re.fullmatch(r"(rows|columns)\s*:\s*([+-]?\d+)", stripped)
        if match is None:
            continue
        key = match.group(1)
        if key in values:
            raise ValueError(f"duplicate grid.{key} in shared config: {path}")
        values[key] = int(match.group(2))

    if grid_indent is None:
        raise ValueError(f"shared config has no grid section: {path}")
    missing = {"rows", "columns"} - set(values)
    if missing:
        raise ValueError(
            "shared config is missing "
            + ", ".join(f"grid.{key}" for key in sorted(missing))
        )
    if values["rows"] <= 0 or values["columns"] <= 0:
        raise ValueError("shared grid rows and columns must be greater than zero")
    return SharedGridConfig(values["rows"], values["columns"])


def require_matching_grid(
    baseline: GridBaseline,
    shared_grid: SharedGridConfig | None,
) -> None:
    if shared_grid is None:
        return
    if (baseline.rows, baseline.columns) != (
        shared_grid.rows,
        shared_grid.columns,
    ):
        raise ValueError(
            "baseline grid does not match shared config: "
            f"baseline={baseline.columns}x{baseline.rows}, "
            f"config={shared_grid.columns}x{shared_grid.rows}; run calibrate"
        )


def build_grid(rows: int, columns: int) -> tuple[GridTile, ...]:
    return tuple(
        GridTile(
            tile_id=f"T_r{row}c{column}",
            row=row,
            column=column,
            normalized_bbox=(
                column / columns,
                row / rows,
                1.0 / columns,
                1.0 / rows,
            ),
        )
        for row in range(rows)
        for column in range(columns)
    )


def resize_tile(tile_bgr: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = tile_bgr.shape[:2]
    scale = min(tile_size / width, tile_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        tile_bgr,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    square = np.full((tile_size, tile_size, 3), 114, dtype=np.uint8)
    offset_x = (tile_size - resized_width) // 2
    offset_y = (tile_size - resized_height) // 2
    square[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return np.ascontiguousarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))


def add_tile_id_banner(tile_rgb: np.ndarray, tile_id: str) -> np.ndarray:
    annotated = tile_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, annotated.shape[1] / 440.0)
    thickness = max(2, round(annotated.shape[1] / 180.0))
    (text_width, text_height), baseline = cv2.getTextSize(
        tile_id,
        font,
        font_scale,
        thickness,
    )
    cv2.rectangle(
        annotated,
        (0, 0),
        (text_width + 18, text_height + baseline + 16),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        annotated,
        tile_id,
        (9, text_height + 8),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return np.ascontiguousarray(annotated)


def decode_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to decode image: {image_path}")
    return image_bgr


def tile_bounds(
    tile: GridTile,
    image_width: int,
    image_height: int,
    rows: int,
    columns: int,
) -> tuple[int, int, int, int]:
    left = image_width * tile.column // columns
    right = image_width * (tile.column + 1) // columns
    top = image_height * tile.row // rows
    bottom = image_height * (tile.row + 1) // rows
    return left, top, right, bottom


def tile_crops_from_image(
    image_bgr: np.ndarray,
    tiles: Sequence[GridTile],
    rows: int,
    columns: int,
    tile_size: int,
    annotate_tile_ids: bool = True,
) -> dict[str, np.ndarray]:
    image_height, image_width = image_bgr.shape[:2]
    crops: dict[str, np.ndarray] = {}
    for tile in tiles:
        left, top, right, bottom = tile_bounds(
            tile,
            image_width,
            image_height,
            rows,
            columns,
        )
        crop = resize_tile(
            image_bgr[top:bottom, left:right],
            tile_size,
        )
        crops[tile.tile_id] = (
            add_tile_id_banner(crop, tile.tile_id)
            if annotate_tile_ids
            else crop
        )
    return crops


def load_tile_crops(
    image_path: Path,
    tiles: Sequence[GridTile],
    rows: int,
    columns: int,
    tile_size: int,
) -> tuple[int, int, dict[str, np.ndarray]]:
    image_bgr = decode_image(image_path)
    image_height, image_width = image_bgr.shape[:2]
    crops = tile_crops_from_image(image_bgr, tiles, rows, columns, tile_size)
    return image_width, image_height, crops


def tile_difference_scores(
    current_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    tiles: Sequence[GridTile],
    rows: int,
    columns: int,
) -> dict[str, float]:
    current_height, current_width = current_bgr.shape[:2]
    reference_height, reference_width = reference_bgr.shape[:2]
    if (reference_width, reference_height) != (current_width, current_height):
        interpolation = (
            cv2.INTER_AREA
            if reference_width > current_width or reference_height > current_height
            else cv2.INTER_CUBIC
        )
        reference_bgr = cv2.resize(
            reference_bgr,
            (current_width, current_height),
            interpolation=interpolation,
        )

    scores: dict[str, float] = {}
    for tile in tiles:
        left, top, right, bottom = tile_bounds(
            tile,
            current_width,
            current_height,
            rows,
            columns,
        )
        current_tile = current_bgr[top:bottom, left:right].astype(np.float32)
        reference_tile = reference_bgr[top:bottom, left:right].astype(np.float32)
        delta = current_tile - reference_tile
        score = float(np.sqrt(np.mean(delta * delta)) / 255.0)
        scores[tile.tile_id] = min(1.0, max(0.0, score))
    return scores


def make_request(
    prompt: str,
    images: Sequence[np.ndarray],
    max_new_tokens: int,
    image_tile_ids: Sequence[str] | None = None,
) -> Any:
    import pyneat as neat

    request = neat.genai.GenerationRequest()
    if images:
        if image_tile_ids is None or len(image_tile_ids) != len(images):
            raise ValueError("each request image must have one tile id")
        messages = []
        for tile_id, image in zip(image_tile_ids, images):
            message = neat.genai.ChatMessage()
            message.role = "user"
            message.content = f"This attached image is spatial tile {tile_id}."
            message.images = [image]
            messages.append(message)
        instruction = neat.genai.ChatMessage()
        instruction.role = "user"
        instruction.content = prompt
        messages.append(instruction)
        request.messages = messages
    else:
        request.prompt = prompt
    request.use_cached_images = False
    request.enable_thinking = False
    request.max_new_tokens = min(MAX_GENERATION_TOKENS, max_new_tokens)
    return request


def make_watch_request(prompt: str, image: np.ndarray) -> Any:
    """Build a single-image WATCH request with no tile identity in its text."""
    import pyneat as neat

    request = neat.genai.GenerationRequest()
    message = neat.genai.ChatMessage()
    message.role = "user"
    message.content = prompt
    message.images = [image]
    request.messages = [message]
    request.use_cached_images = False
    request.enable_thinking = False
    request.max_new_tokens = WATCH_MAX_NEW_TOKENS
    return request


def timed_run(model: VisionLanguageModel, request: Any) -> tuple[str, float]:
    started = time.perf_counter()
    result = model.run(request)
    elapsed = time.perf_counter() - started
    text = (result.text or "").strip()
    if not text:
        raise RuntimeError("VLM returned an empty response")
    return text, elapsed


def normalize_generated_sentence(sentence: str) -> str:
    return " ".join(sentence.split()).casefold()


def timed_watch_run(model: VisionLanguageModel, request: Any) -> tuple[str, float]:
    """Stream WATCH output until a verdict or repeated sentence is complete."""
    started = time.perf_counter()
    stream = model.stream(request)
    accepted: list[str] = []
    pending = ""
    seen_sentences: set[str] = set()
    repeated = False
    completed_response: str | None = None

    for sample in stream:
        if sample.is_final:
            break
        if not sample.text:
            continue
        pending += sample.text
        current_response = "".join(accepted) + pending
        if verdict_match := WATCH_COMPLETE_VERDICT_RE.search(current_response):
            stream.cancel()
            completed_response = current_response[: verdict_match.end()].strip()
            break
        while match := COMPLETED_SENTENCE_RE.match(pending):
            sentence = match.group(0)
            pending = pending[match.end() :]
            normalized = normalize_generated_sentence(sentence)
            if normalized and normalized in seen_sentences:
                stream.cancel()
                repeated = True
                break
            if normalized:
                seen_sentences.add(normalized)
            accepted.append(sentence)
        if repeated:
            break

    if completed_response is not None:
        text = completed_response
    else:
        if not repeated:
            accepted.append(pending)
        text = "".join(accepted).strip()
    elapsed = time.perf_counter() - started
    if not text:
        raise RuntimeError("VLM returned an empty response")
    return text, elapsed


def compact_text(text: str) -> str:
    return " ".join(text.split()).strip()


def valid_description(text: str) -> bool:
    return len(re.findall(r"[A-Za-z][A-Za-z'-]+", text)) >= 3


def valid_calibration_description(text: str) -> bool:
    if not valid_description(text):
        return False
    if NO_PERMANENT_CONTENT_RE.search(text):
        return True
    lowered = text.casefold()
    if (
        "a cable or cord also visible" in lowered
        or "a cable or cord is also visible" in lowered
        or "cables or cords are also visible" in lowered
    ):
        return False
    if SPATIAL_STATE_RE.search(text) is None:
        return False
    if CABLE_TERM_RE.search(text) and CONNECTION_STATE_RE.search(text) is None:
        return False
    return True


def canonical_tile_id(raw_tile_id: str, expected: Sequence[str]) -> str | None:
    lowered = raw_tile_id.casefold()
    return next((tile_id for tile_id in expected if tile_id.casefold() == lowered), None)


def parse_calibration_response(
    response: str,
    expected_tile_ids: Sequence[str],
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    invalid: set[str] = set()
    for raw_line in response.splitlines():
        match = CALIBRATION_LINE_RE.fullmatch(raw_line)
        if match is None:
            continue
        tile_id = canonical_tile_id(match.group(1), expected_tile_ids)
        if tile_id is None:
            continue
        description = compact_text(match.group(2))
        if tile_id in parsed or not valid_calibration_description(description):
            invalid.add(tile_id)
            continue
        parsed[tile_id] = description
    return {
        tile_id: description
        for tile_id, description in parsed.items()
        if tile_id not in invalid
    }


def compact_watch_field(value: str) -> str:
    compacted = compact_text(value)
    if (
        len(compacted) >= 2
        and compacted[0] == compacted[-1]
        and compacted[0] in {"\"", "'"}
    ):
        return compact_text(compacted[1:-1])
    return compacted


def canonical_watch_verdict(value: str) -> tuple[str, str] | None:
    raw_verdict = compact_watch_field(value).upper()
    for candidate in (
        "DOES NOT CONTRADICT",
        "CONTRADICTS",
        "DEVIATION",
        "UNCERTAIN",
        "MATCH",
    ):
        if raw_verdict == candidate or raw_verdict.startswith(f"{candidate}:"):
            return candidate, WATCH_VERDICT_ALIASES[candidate]
    return None


def match_contradicts_connection_state(
    verdict: str,
    baseline_description: str,
    current_observation: str,
) -> bool:
    if verdict != "MATCH":
        return False
    baseline_connected = bool(CONNECTED_STATE_RE.search(baseline_description))
    baseline_disconnected = bool(
        DISCONNECTED_STATE_RE.search(baseline_description)
    )
    current_connected = bool(CONNECTED_STATE_RE.search(current_observation))
    current_disconnected = bool(
        DISCONNECTED_STATE_RE.search(current_observation)
    )
    return (
        baseline_connected
        and not baseline_disconnected
        and current_disconnected
    ) or (
        baseline_disconnected
        and not baseline_connected
        and current_connected
    )


def parse_watch_response(
    response: str,
    baseline_description: str,
) -> TileVerdict | None:
    """Parse one observe-compare-decide response for one image and baseline."""
    lines = [line for line in response.splitlines() if line.strip()]
    if len(lines) != 3:
        return None
    observation_match = WATCH_OBSERVATION_LINE_RE.fullmatch(lines[0])
    reason_match = WATCH_REASON_LINE_RE.fullmatch(lines[1])
    verdict_match = WATCH_VERDICT_LINE_RE.fullmatch(lines[2])
    if observation_match is None or reason_match is None or verdict_match is None:
        return None
    observation = compact_watch_field(observation_match.group(1))
    reason = compact_watch_field(reason_match.group(1))
    parsed_verdict = canonical_watch_verdict(verdict_match.group(1))
    reason_prefix_match = WATCH_REASON_PREFIX_RE.fullmatch(reason)
    if (
        not observation
        or not reason
        or parsed_verdict is None
    ):
        return None
    raw_verdict, verdict = parsed_verdict
    if reason_prefix_match is not None:
        expected_verdict = WATCH_VERDICT_ALIASES[
            reason_prefix_match.group(1).upper()
        ]
        if verdict != expected_verdict:
            return None
    if match_contradicts_connection_state(
        verdict,
        baseline_description,
        observation,
    ):
        return None
    return TileVerdict(verdict, reason, observation)


def print_watch_parse_failure_diagnostic(
    tile_id: str,
    response: str,
    already_reported: int,
    limit: int = 3,
) -> int:
    if already_reported >= limit:
        return already_reported
    reported = already_reported + 1
    print(f"WATCH_PARSE_FAILURE_{reported}_TILE: {tile_id}", file=sys.stderr)
    print(
        f"WATCH_PARSE_FAILURE_{reported}_EXPECTED: "
        "OBSERVATION=<one sentence>, then "
        "REASON=<CONTRADICTS, DOES NOT CONTRADICT, or UNCERTAIN: comparison>, "
        "then VERDICT=<MATCH or DEVIATION or UNCERTAIN>",
        file=sys.stderr,
    )
    print(
        f"WATCH_PARSE_FAILURE_{reported}_RAW_RESPONSE_JSON: "
        f"{json.dumps(response, ensure_ascii=False)}",
        file=sys.stderr,
    )
    return reported


def image_mapping(tile_ids: Sequence[str]) -> str:
    return "\n".join(
        f"IMAGE {index} = {tile_id}"
        for index, tile_id in enumerate(tile_ids, start=1)
    )


def calibration_prompt(tile_ids: Sequence[str]) -> str:
    output_lines = "\n".join(
        f"{tile_id}|DESCRIPTION=<one factual sentence>" for tile_id in tile_ids
    )
    return f"""\
You receive {len(tile_ids)} images. Each image is a fixed spatial tile from the
same stationary-camera scene. Images are attached in exactly this order:
{image_mapping(tile_ids)}

For EACH tile, write a checkable record of the permanent physical state, not an
inventory of visible object names. The sentence must contain enough spatial and
connection detail that a later unplugged, moved, rotated, or missing object
would visibly contradict it.

For every cable, cord, plug, or connector that is visible:
- Trace it as far as the image permits and name what EACH visible end is
  connected to, such as a laptop's left port, development kit, or power strip.
- State whether each visible end is plugged in, attached, loose, or unattached.
- Describe its route and position, such as left-to-right across the lower edge.
- If an end is hidden or leaves the image, explicitly say that its connection
  state cannot be determined; never replace unknown state with "also visible."

For every other permanent object, identify it and state its position and
orientation, such as upright at lower left or horizontal across the center.
Forbid generic inventories such as "a cable or cord is also visible" and "X, Y,
and Z are visible" without connection, position, or orientation state.

Ignore people, hands, screen contents, shadows, reflections, and other transient
activity. If no permanent physical content is clear, say that plainly. Do not
refer to images, tiles, crops, their printed ID banners, or this prompt. Inspect
each differently labelled image independently; never copy a description from a
different tile. Use exactly one factual sentence of at most 45 words per tile.

Return exactly these lines in this order, replacing only the placeholder:
{output_lines}

Output nothing else: no markdown, bullets, preamble, or repeated lines.
"""


def watch_prompt(baseline_description: str) -> str:
    return f"""\
BASELINE_SENTENCE={json.dumps(baseline_description, ensure_ascii=False)}

The baseline records the permanent physical state during calibration, even if
that recorded state looks disconnected, loose, incomplete, or abnormal.

The ONLY question is whether the permanent physical state in the current image
CHANGED from what the quoted baseline recorded. Do not judge whether the scene
looks abnormal, unsafe, or desirable on its own merits.

Reason in this mandatory order:
1. OBSERVATION: State what the current image positively shows about the specific
   physical state claimed by the baseline.
2. REASON: Explicitly test whether that observation contradicts the quoted
   baseline. Begin this line with exactly one of:
   - CONTRADICTS: when a positively observed current state conflicts with it.
   - DOES NOT CONTRADICT: when the current state agrees or is compatible with it.
   - UNCERTAIN: when the baseline claim cannot be checked confidently.
   REASON must be exactly one concise sentence about the baseline's relevant
   claim. Do not enumerate every object, repeat the comparison, or discuss
   unrelated details. After that one sentence, immediately emit VERDICT; never
   write a second REASON sentence.
3. VERDICT: Derive this mechanically from REASON: CONTRADICTS means DEVIATION;
   DOES NOT CONTRADICT means MATCH; UNCERTAIN means UNCERTAIN.

Comparison rules:
- A loose, unplugged, unattached, or disconnected cable is MATCH when the
  baseline already records that same loose, unplugged, unattached, or
  disconnected state.
- Do not invent a change merely because the baseline omits a detail. An
  unmentioned detail does not contradict the baseline.
- Ignore people, hands, screen contents, shadows, reflections, and transient
  activity; they are not changes to permanent physical state.
- If occlusion, framing, blur, low resolution, partial visibility, or ambiguity
  prevents a confident check, use UNCERTAIN, never DEVIATION. Failure to confirm
  the baseline is not evidence that the baseline is false.
- DEVIATION requires positive visual evidence of a different state, such as an
  object removed from a clearly visible expected location or a cable visibly
  detached when the baseline records it attached.

Answer exactly three lines in this order, with no tile ID:
OBSERVATION=<one factual sentence about the relevant current state>
REASON=<exactly one sentence beginning CONTRADICTS:, DOES NOT CONTRADICT:, or UNCERTAIN:>
VERDICT=<MATCH or DEVIATION or UNCERTAIN>
Nothing else: no markdown, preamble, list, or extra lines.
"""


def token_budget(tile_count: int, tokens_per_tile: int) -> int:
    return min(MAX_GENERATION_TOKENS, max(64, tile_count * tokens_per_tile + 16))


def row_tile_ids(tiles: Sequence[GridTile], row: int) -> list[str]:
    return [tile.tile_id for tile in tiles if tile.row == row]


def excessive_copy_forward(values: Mapping[str, Any]) -> bool:
    if len(values) < 6:
        return False
    signatures: list[str] = []
    for value in values.values():
        if isinstance(value, TileVerdict):
            text = f"{value.observation} {value.reason} {value.verdict}"
        else:
            text = str(value)
        signatures.append(normalize_generated_sentence(text))
    largest_repeat = max(signatures.count(signature) for signature in set(signatures))
    return largest_repeat >= math.ceil(len(signatures) * 2 / 3)


def describe_frame_tiles(
    model: VisionLanguageModel,
    tiles: Sequence[GridTile],
    crops: Mapping[str, np.ndarray],
    rows: int,
    force_row_batches: bool,
) -> BatchResult:
    all_tile_ids = [tile.tile_id for tile in tiles]
    raw_responses: list[str] = []
    request_count = 0
    inference_seconds = 0.0

    if not force_row_batches and len(all_tile_ids) <= MAX_IMAGES_PER_REQUEST:
        try:
            response, elapsed = timed_run(
                model,
                make_request(
                    calibration_prompt(all_tile_ids),
                    [crops[tile_id] for tile_id in all_tile_ids],
                    token_budget(len(all_tile_ids), CALIBRATION_TOKENS_PER_TILE),
                    all_tile_ids,
                ),
            )
            request_count += 1
            inference_seconds += elapsed
            raw_responses.append(response)
            parsed = parse_calibration_response(response, all_tile_ids)
            copied = excessive_copy_forward(parsed)
            if len(parsed) == len(all_tile_ids) and not copied:
                return BatchResult(
                    parsed,
                    "all_tiles",
                    request_count,
                    inference_seconds,
                    tuple(raw_responses),
                    frozenset(),
                )
            missing = sorted(set(all_tile_ids) - set(parsed))
            detail = (
                "copy-forward repetition"
                if copied
                else f"missing {', '.join(missing)}"
            )
            print(
                "[warn] all-tile calibration response unusable; "
                f"falling back to row batches: {detail}",
                file=sys.stderr,
            )
        except (RuntimeError, ValueError) as exc:
            request_count += 1
            raw_responses.append(f"ERROR: {exc}")
            print(
                "[warn] all-tile calibration request failed; "
                f"falling back to row batches: {exc}",
                file=sys.stderr,
            )

    parsed: dict[str, str] = {}
    for row in range(rows):
        tile_ids = row_tile_ids(tiles, row)
        try:
            response, elapsed = timed_run(
                model,
                make_request(
                    calibration_prompt(tile_ids),
                    [crops[tile_id] for tile_id in tile_ids],
                    token_budget(len(tile_ids), CALIBRATION_TOKENS_PER_TILE),
                    tile_ids,
                ),
            )
            request_count += 1
            inference_seconds += elapsed
            raw_responses.append(response)
            row_parsed = parse_calibration_response(response, tile_ids)
            if excessive_copy_forward(row_parsed):
                print(
                    f"[warn] calibration row {row} contains copy-forward "
                    "repetition; retrying its tiles individually",
                    file=sys.stderr,
                )
            else:
                parsed.update(row_parsed)
        except (RuntimeError, ValueError) as exc:
            request_count += 1
            raw_responses.append(f"ROW {row} ERROR: {exc}")
            print(f"[warn] calibration row {row} failed: {exc}", file=sys.stderr)

    retry_tile_ids = sorted(set(all_tile_ids) - set(parsed))
    for tile_id in retry_tile_ids:
        try:
            response, elapsed = timed_run(
                model,
                make_request(
                    calibration_prompt([tile_id]),
                    [crops[tile_id]],
                    token_budget(1, CALIBRATION_TOKENS_PER_TILE),
                    [tile_id],
                ),
            )
            request_count += 1
            inference_seconds += elapsed
            raw_responses.append(response)
            parsed.update(parse_calibration_response(response, [tile_id]))
        except (RuntimeError, ValueError) as exc:
            request_count += 1
            raw_responses.append(f"TILE {tile_id} ERROR: {exc}")
            print(f"[warn] calibration tile {tile_id} failed: {exc}", file=sys.stderr)

    missing = frozenset(set(all_tile_ids) - set(parsed))
    return BatchResult(
        parsed,
        "row_batches_with_tile_retries" if retry_tile_ids else "row_batches",
        request_count,
        inference_seconds,
        tuple(raw_responses),
        missing,
    )


def judge_frame_tiles(
    model: VisionLanguageModel,
    baseline: GridBaseline,
    tiles: Sequence[GridTile],
    crops: Mapping[str, np.ndarray],
    force_row_batches: bool,
) -> BatchResult:
    del force_row_batches  # Retained only for CLI compatibility; WATCH never batches.
    all_tile_ids = [tile.tile_id for tile in tiles]
    raw_responses: list[str] = []
    request_count = 0
    inference_seconds = 0.0
    parse_failures_reported = 0
    parsed: dict[str, TileVerdict] = {}
    for tile in tiles:
        tile_id = tile.tile_id
        request_count += 1
        try:
            response, elapsed = timed_watch_run(
                model,
                make_watch_request(
                    watch_prompt(baseline.tiles[tile_id].description),
                    crops[tile_id],
                ),
            )
            inference_seconds += elapsed
            raw_responses.append(response)
            tile_verdict = parse_watch_response(
                response,
                baseline.tiles[tile_id].description,
            )
            if tile_verdict is None:
                parse_failures_reported = print_watch_parse_failure_diagnostic(
                    tile_id,
                    response,
                    parse_failures_reported,
                )
            else:
                parsed[tile_id] = tile_verdict
        except (RuntimeError, ValueError) as exc:
            raw_responses.append(f"TILE {tile_id} ERROR: {exc}")
            print(
                f"[warn] WATCH tile {tile_id} failed: {exc}",
                file=sys.stderr,
            )

    missing = frozenset(set(all_tile_ids) - set(parsed))
    return BatchResult(
        parsed,
        "single_tile_requests",
        request_count,
        inference_seconds,
        tuple(raw_responses),
        missing,
    )


def merge_tile_descriptions(
    model: VisionLanguageModel,
    tile_id: str,
    descriptions: Sequence[str],
) -> str:
    if len(descriptions) == 1:
        return compact_text(descriptions[0])
    observations = "\n".join(
        f"OBSERVATION {index}: {description}"
        for index, description in enumerate(descriptions, start=1)
    )
    prompt = f"""\
Merge these independent observations into a checkable baseline state for
spatial tile {tile_id}. Permanent means it remains after all people leave.
Remove people, clothing, hands, actions, screen contents, drinks, and other
transient details. Preserve object position and orientation. For every cable,
preserve its route, the named object at each visible end, and whether each end
is plugged in, attached, loose, or unattached. If an end is hidden, outside the
image, or observations conflict, explicitly say its connection state cannot be
determined instead of omitting the state.

Never output an object inventory such as "a cable or cord is also visible" or
"X, Y, and Z are visible" without connection, position, or orientation state.

Return exactly: DESCRIPTION=<one factual sentence of at most 45 words>
Do not mention tiles, images, observations, or these instructions.

{observations}
"""
    response, _ = timed_run(model, make_request(prompt, [], MERGE_TOKENS))
    match = re.search(r"(?im)^\s*DESCRIPTION\s*=\s*(.+?)\s*$", response)
    merged = compact_text(match.group(1) if match else response.splitlines()[0])
    sentence_match = re.match(r"^.*?[.!?](?=\s|$)", merged)
    if sentence_match:
        merged = sentence_match.group(0)
    words = merged.split()
    if len(words) > 45:
        merged = " ".join(words[:45]).rstrip(",;:") + "."
    if not valid_calibration_description(merged):
        merged = max(descriptions, key=lambda description: len(description.split()))
    return compact_text(merged)


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def store_reference_frame(image_path: Path, baseline_path: Path) -> Path:
    suffix = image_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    reference_path = baseline_path.with_name(
        f"{baseline_path.stem}_reference{suffix}"
    )
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = reference_path.with_suffix(reference_path.suffix + ".tmp")
    temporary_path.write_bytes(image_path.read_bytes())
    temporary_path.replace(reference_path)
    return reference_path


def frame_stamp_from_stat(stat_result: os.stat_result) -> FrameStamp:
    return FrameStamp(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
    )


def live_frame_stamp(path: Path) -> FrameStamp | None:
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file() or stat_result.st_size <= 0:
        return None
    return frame_stamp_from_stat(stat_result)


def wait_for_new_live_frame(
    path: Path,
    previous: FrameStamp | None,
    not_before: float,
    poll_interval: float,
) -> FrameStamp:
    """Wait for a distinct atomically replaced frame without opening it yet."""
    while True:
        now = time.monotonic()
        if now >= not_before:
            stamp = live_frame_stamp(path)
            if stamp is not None and stamp != previous:
                return stamp
        remaining = max(0.0, not_before - now)
        time.sleep(min(poll_interval, remaining) if remaining else poll_interval)


def capture_live_frame(source: Path, destination: Path) -> FrameStamp:
    """Copy one immutable, decodable frame even if source is replaced meanwhile."""
    try:
        with source.open("rb") as frame_file:
            stamp = frame_stamp_from_stat(os.fstat(frame_file.fileno()))
            payload = frame_file.read()
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"could not read live frame {source}: {exc}") from exc
    if stamp.size <= 0 or len(payload) != stamp.size:
        raise RuntimeError(f"live frame was incomplete: {source}")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    if cv2.imdecode(encoded, cv2.IMREAD_COLOR) is None:
        raise RuntimeError(f"live frame is not a decodable image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return stamp


def capture_next_live_frame(
    source: Path,
    destination: Path,
    previous: FrameStamp | None,
    not_before: float,
    poll_interval: float,
) -> FrameStamp:
    while True:
        wait_for_new_live_frame(source, previous, not_before, poll_interval)
        try:
            captured = capture_live_frame(source, destination)
        except RuntimeError as exc:
            print(f"[warn] {exc}; waiting for the next frame", file=sys.stderr)
            time.sleep(poll_interval)
            continue
        if captured != previous:
            return captured


def calibrate_live(
    model: VisionLanguageModel,
    frame_source: Path,
    baseline_path: Path,
    rows: int,
    columns: int,
    tile_size: int,
    force_row_batches: bool,
    frame_count: int,
    sample_interval: float,
    poll_interval: float,
) -> None:
    print(f"LIVE_FRAME_SOURCE: {frame_source}")
    print(f"LIVE_CALIBRATION_FRAME_COUNT: {frame_count}")
    print(f"LIVE_CALIBRATION_SAMPLE_INTERVAL_SECONDS: {sample_interval:g}")
    with tempfile.TemporaryDirectory(prefix="tripwire-calibrate-") as directory:
        capture_directory = Path(directory)
        captured_paths: list[Path] = []
        # Treat a pre-existing image as stale so clip transitions cannot race
        # the first calibration capture.
        previous = live_frame_stamp(frame_source)
        not_before = time.monotonic()
        for index in range(1, frame_count + 1):
            captured_path = capture_directory / f"calibration_frame_{index:04d}.jpg"
            previous = capture_next_live_frame(
                frame_source,
                captured_path,
                previous,
                not_before,
                poll_interval,
            )
            captured_paths.append(captured_path)
            print(
                f"LIVE_CALIBRATION_CAPTURE: {index}/{frame_count} "
                f"mtime_ns={previous.mtime_ns} size={previous.size}"
            )
            not_before = time.monotonic() + sample_interval

        calibrate(
            model,
            captured_paths,
            baseline_path,
            rows,
            columns,
            tile_size,
            force_row_batches,
        )


def calibrate(
    model: VisionLanguageModel,
    image_paths: Sequence[Path],
    baseline_path: Path,
    rows: int,
    columns: int,
    tile_size: int,
    force_row_batches: bool,
) -> None:
    started = time.perf_counter()
    tiles = build_grid(rows, columns)
    accumulators = {
        tile.tile_id: TileAccumulator()
        for tile in tiles
    }
    source_width = 0
    source_height = 0

    for image_path in image_paths:
        width, height, crops = load_tile_crops(
            image_path,
            tiles,
            rows,
            columns,
            tile_size,
        )
        if source_width == 0:
            source_width, source_height = width, height
        else:
            source_aspect = source_width / source_height
            current_aspect = width / height
            if not math.isclose(source_aspect, current_aspect, rel_tol=0.01):
                raise ValueError(
                    f"calibration image aspect ratio differs: {image_path}"
                )

        result = describe_frame_tiles(
            model,
            tiles,
            crops,
            rows,
            force_row_batches,
        )
        if result.missing:
            print(
                f"RAW_RESPONSES_JSON: {json.dumps(result.raw_responses)}",
                file=sys.stderr,
            )
            print(
                f"[warn] calibration tile(s) still missing after individual "
                f"retry for {image_path}: "
                + ", ".join(sorted(result.missing))
                + "; empty descriptions will be recorded if no other frame "
                "succeeds",
                file=sys.stderr,
            )
        for tile_id, description in result.values.items():
            accumulators[tile_id].add(description)
            print(f"IMAGE: {image_path}")
            print(f"TILE: {tile_id}")
            print("STATUS: calibrating")
            print(f"DESCRIPTION: {description}")
        print(f"CALIBRATION_BATCH_MODE: {result.mode}")
        print(f"CALIBRATION_REQUEST_COUNT: {result.request_count}")
        print(f"CALIBRATION_INFERENCE_SECONDS: {result.inference_seconds:.3f}")

    baseline_tiles: dict[str, dict[str, Any]] = {}
    for tile in tiles:
        accumulator = accumulators[tile.tile_id]
        if accumulator.observation_count == 0:
            description = ""
        else:
            description = merge_tile_descriptions(
                model,
                tile.tile_id,
                accumulator.descriptions,
            )
        baseline_tiles[tile.tile_id] = {
            "normalized_bbox": [
                round(value, 6)
                for value in tile.normalized_bbox
            ],
            "description": description,
            "observation_count": accumulator.observation_count,
            "status": "calibrating",
        }

    reference_path = store_reference_frame(image_paths[0], baseline_path)
    baseline = {
        "version": 3,
        "region_type": "grid",
        "bbox_format": "normalized_xywh",
        "reference_frame": reference_path.name,
        "grid": {
            "rows": rows,
            "columns": columns,
        },
        "source": {
            "width": source_width,
            "height": source_height,
            "aspect_ratio": round(source_width / source_height, 6),
        },
        "calibration_frame_count": len(image_paths),
        "tiles": baseline_tiles,
    }
    write_json_atomic(baseline_path, baseline)
    print(f"BASELINE: {baseline_path}")
    print(f"REFERENCE_FRAME: {reference_path}")
    print(f"TILE_COUNT: {len(tiles)}")
    print(f"CALIBRATION_TOTAL_SECONDS: {time.perf_counter() - started:.3f}")


def _baseline_bbox(value: object, tile_id: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{tile_id}: normalized_bbox must contain four values")
    if len(value) != 4:
        raise ValueError(f"{tile_id}: normalized_bbox must contain four values")
    try:
        bbox = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{tile_id}: normalized_bbox values must be numbers") from exc
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        raise ValueError(f"{tile_id}: normalized_bbox values must be finite")
    if (
        bbox[0] < 0.0
        or bbox[1] < 0.0
        or bbox[2] <= 0.0
        or bbox[3] <= 0.0
        or bbox[0] + bbox[2] > 1.000001
        or bbox[1] + bbox[3] > 1.000001
    ):
        raise ValueError(f"{tile_id}: normalized_bbox is outside the image")
    return bbox  # type: ignore[return-value]


def load_baseline(path: Path) -> GridBaseline:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid baseline JSON: {exc.msg}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("baseline must be a JSON object")
    if document.get("version") != 3 or document.get("region_type") != "grid":
        raise ValueError("baseline must be a version 3 grid baseline; run calibrate")
    if document.get("bbox_format") != "normalized_xywh":
        raise ValueError("baseline bbox_format must be normalized_xywh")

    raw_grid = document.get("grid")
    raw_source = document.get("source")
    raw_tiles = document.get("tiles")
    raw_reference_frame = document.get("reference_frame")
    if not isinstance(raw_grid, Mapping):
        raise ValueError("baseline grid must be an object")
    if not isinstance(raw_source, Mapping):
        raise ValueError("baseline source must be an object")
    if not isinstance(raw_tiles, Mapping):
        raise ValueError("baseline tiles must be an object keyed by tile id")
    if not isinstance(raw_reference_frame, str) or not raw_reference_frame.strip():
        raise ValueError("baseline reference_frame must be a non-empty path")
    reference_frame = Path(raw_reference_frame)
    if not reference_frame.is_absolute():
        reference_frame = path.parent / reference_frame
    if not reference_frame.is_file():
        raise FileNotFoundError(
            f"baseline reference frame does not exist: {reference_frame}"
        )
    try:
        rows = int(raw_grid["rows"])
        columns = int(raw_grid["columns"])
        image_width = int(raw_source["width"])
        image_height = int(raw_source["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("baseline grid/source dimensions are invalid") from exc
    if rows <= 0 or columns <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("baseline grid/source dimensions must be positive")

    expected_tiles = build_grid(rows, columns)
    baseline_tiles: dict[str, BaselineTile] = {}
    for tile in expected_tiles:
        raw_tile = raw_tiles.get(tile.tile_id)
        if not isinstance(raw_tile, Mapping):
            raise ValueError(f"baseline is missing {tile.tile_id}")
        description = compact_text(str(raw_tile.get("description", "")))
        try:
            observation_count = int(raw_tile.get("observation_count", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{tile.tile_id}: observation_count is invalid") from exc
        bbox = _baseline_bbox(raw_tile.get("normalized_bbox"), tile.tile_id)
        if observation_count < 0:
            raise ValueError(
                f"{tile.tile_id}: observation_count cannot be negative"
            )
        if (observation_count == 0 and description) or (
            observation_count > 0 and not description
        ):
            raise ValueError(
                f"{tile.tile_id}: description and observation_count disagree"
            )
        if any(
            not math.isclose(actual, expected, abs_tol=1e-5)
            for actual, expected in zip(bbox, tile.normalized_bbox)
        ):
            raise ValueError(f"{tile.tile_id}: normalized_bbox does not match grid")
        baseline_tiles[tile.tile_id] = BaselineTile(
            tile.tile_id,
            bbox,
            description,
            observation_count,
        )
    if set(raw_tiles) != set(baseline_tiles):
        raise ValueError("baseline contains unexpected tile ids")
    return GridBaseline(
        rows,
        columns,
        image_width,
        image_height,
        baseline_tiles,
        reference_frame,
    )


def status_for_verdict(verdict: str) -> str:
    if verdict == "MATCH":
        return "normal"
    if verdict == "DEVIATION":
        return "deviation"
    return "uncertain"


def retained_tile_result(
    tile: GridTile,
    previous_tiles: Mapping[str, Any] | None,
) -> dict[str, Any]:
    previous = previous_tiles.get(tile.tile_id) if previous_tiles is not None else None
    if (
        isinstance(previous, Mapping)
        and previous.get("status") in {"normal", "uncertain", "deviation"}
        and previous.get("verdict") in VALID_VERDICTS
    ):
        return {
            "normalized_bbox": [round(value, 6) for value in tile.normalized_bbox],
            "observation": str(previous.get("observation", "")),
            "verdict": str(previous["verdict"]),
            "reason": str(previous.get("reason", "")),
            "status": str(previous["status"]),
        }
    return {
        "normalized_bbox": [round(value, 6) for value in tile.normalized_bbox],
        "observation": "No material pixel change is visible from calibration.",
        "verdict": "MATCH",
        "reason": "The difference gate retained the baseline-normal status.",
        "status": "normal",
    }


def watch_pass(
    model: VisionLanguageModel,
    image_path: Path,
    baseline: GridBaseline,
    tiles: Sequence[GridTile],
    tile_size: int,
    force_row_batches: bool,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    previous_tiles: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pass_started = time.perf_counter()
    if not math.isfinite(diff_threshold) or not 0.0 <= diff_threshold <= 1.0:
        raise ValueError("diff threshold must be between 0.0 and 1.0")
    if baseline.reference_frame is None:
        raise ValueError("baseline has no reference frame; run calibrate")
    current_bgr = decode_image(image_path)
    reference_bgr = decode_image(baseline.reference_frame)
    height, width = current_bgr.shape[:2]
    if not math.isclose(
        width / height,
        baseline.image_width / baseline.image_height,
        rel_tol=0.01,
    ):
        raise ValueError(
            f"WATCH image aspect ratio does not match baseline: {image_path}"
        )

    difference_scores = tile_difference_scores(
        current_bgr,
        reference_bgr,
        tiles,
        baseline.rows,
        baseline.columns,
    )
    changed_tiles = [
        tile
        for tile in tiles
        if diff_threshold == 0.0
        or difference_scores[tile.tile_id] > diff_threshold
    ]
    baseline_missing_tiles = [
        tile
        for tile in changed_tiles
        if not baseline.tiles[tile.tile_id].description.strip()
    ]
    baseline_missing_tile_ids = {
        tile.tile_id for tile in baseline_missing_tiles
    }
    sent_tiles = [
        tile
        for tile in changed_tiles
        if tile.tile_id not in baseline_missing_tile_ids
    ]
    changed_tile_ids = {tile.tile_id for tile in changed_tiles}
    gated_tile_count = len(tiles) - len(changed_tiles)
    if sent_tiles:
        crops = tile_crops_from_image(
            current_bgr,
            sent_tiles,
            baseline.rows,
            baseline.columns,
            tile_size,
            False,
        )
        result = judge_frame_tiles(
            model,
            baseline,
            sent_tiles,
            crops,
            force_row_batches,
        )
    else:
        result = BatchResult(
            {},
            (
                "missing_baseline_no_inference"
                if baseline_missing_tiles
                else "diff_gated_no_inference"
            ),
            0,
            0.0,
            (),
            frozenset(),
        )

    tile_results: dict[str, dict[str, Any]] = {}
    for tile in tiles:
        if tile.tile_id not in changed_tile_ids:
            tile_results[tile.tile_id] = retained_tile_result(tile, previous_tiles)
        elif tile.tile_id in baseline_missing_tile_ids:
            tile_results[tile.tile_id] = {
                "normalized_bbox": [
                    round(value, 6) for value in tile.normalized_bbox
                ],
                "observation": "No calibration baseline was recorded for this tile.",
                "verdict": "UNCERTAIN",
                "reason": "A current image cannot be compared without a baseline.",
                "status": "uncertain",
            }
        else:
            parsed = result.values.get(tile.tile_id)
            if isinstance(parsed, TileVerdict):
                verdict = parsed.verdict
                reason = parsed.reason
                observation = parsed.observation
            else:
                verdict = "UNCERTAIN"
                reason = "The VLM response for this tile was missing or malformed."
                observation = "No parseable current-state observation was returned."
            status = status_for_verdict(verdict)
            tile_results[tile.tile_id] = {
                "normalized_bbox": [
                    round(value, 6)
                    for value in tile.normalized_bbox
                ],
                "observation": observation,
                "verdict": verdict,
                "reason": reason,
                "status": status,
            }
        tile_result = tile_results[tile.tile_id]
        print(f"IMAGE: {image_path}")
        print(f"TILE: {tile.tile_id}")
        print(f"PIXEL_DIFFERENCE: {difference_scores[tile.tile_id]:.6f}")
        print(f"OBSERVATION: {tile_result['observation']}")
        print(f"REASON: {tile_result['reason']}")
        print(f"VERDICT: {tile_result['verdict']}")
        print(f"STATUS: {tile_result['status']}")

    wall_seconds = time.perf_counter() - pass_started
    output = {
        "image": str(image_path),
        "grid": {
            "rows": baseline.rows,
            "columns": baseline.columns,
        },
        "batch_mode": result.mode,
        "diff_threshold": diff_threshold,
        "gated_tile_count": gated_tile_count,
        "sent_tile_count": len(sent_tiles),
        "baseline_missing_tile_count": len(baseline_missing_tiles),
        "request_count": result.request_count,
        "parse_failure_count": len(result.missing),
        "inference_seconds": round(result.inference_seconds, 3),
        "wall_clock_seconds": round(wall_seconds, 3),
        "tiles": tile_results,
    }
    print(f"WATCH_BATCH_MODE: {result.mode}")
    print(f"WATCH_DIFF_THRESHOLD: {diff_threshold:.6f}")
    print(f"WATCH_GATED_TILE_COUNT: {gated_tile_count}")
    print(f"WATCH_SENT_TILE_COUNT: {len(sent_tiles)}")
    print(f"WATCH_BASELINE_MISSING_TILE_COUNT: {len(baseline_missing_tiles)}")
    print(f"WATCH_REQUEST_COUNT: {result.request_count}")
    print(f"WATCH_PARSE_FAILURE_COUNT: {len(result.missing)}")
    print(f"WATCH_INFERENCE_SECONDS: {result.inference_seconds:.3f}")
    print(f"WATCH_PASS_TIME_SECONDS: {wall_seconds:.3f}")
    print(f"WATCH_RESULT_JSON: {json.dumps(output, sort_keys=True)}")
    print(f"RAW_RESPONSES_JSON: {json.dumps(result.raw_responses)}")
    print()
    return output


def watch(
    model: VisionLanguageModel,
    image_paths: Sequence[Path],
    baseline_path: Path,
    tile_size: int,
    force_row_batches: bool,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    shared_grid: SharedGridConfig | None = None,
) -> int:
    baseline = load_baseline(baseline_path)
    require_matching_grid(baseline, shared_grid)
    tiles = build_grid(baseline.rows, baseline.columns)
    previous_tiles: Mapping[str, Any] | None = None
    for image_path in image_paths:
        output = watch_pass(
            model,
            image_path,
            baseline,
            tiles,
            tile_size,
            force_row_batches,
            diff_threshold,
            previous_tiles,
        )
        previous_tiles = output["tiles"]

    return 0


def build_status_snapshot(
    output: Mapping[str, Any],
    sequence: int,
    completed_unix_ms: int,
    captured_unix_ms: int,
    frame_source: Path,
    frame_stamp: FrameStamp,
) -> dict[str, Any]:
    raw_grid = output.get("grid")
    raw_tiles = output.get("tiles")
    if not isinstance(raw_grid, Mapping) or not isinstance(raw_tiles, Mapping):
        raise ValueError("completed WATCH result has no grid tile mapping")
    rows = int(raw_grid.get("rows", 0))
    columns = int(raw_grid.get("columns", 0))
    if rows <= 0 or columns <= 0:
        raise ValueError("completed WATCH result has invalid grid dimensions")
    expected_tiles = build_grid(rows, columns)
    expected_ids = {tile.tile_id for tile in expected_tiles}
    if set(raw_tiles) != expected_ids:
        raise ValueError("completed WATCH result does not contain every grid tile")
    for expected_tile in expected_tiles:
        tile_id = expected_tile.tile_id
        raw_tile = raw_tiles[tile_id]
        if not isinstance(raw_tile, Mapping):
            raise ValueError(f"completed WATCH result for {tile_id} is malformed")
        bbox = _baseline_bbox(raw_tile.get("normalized_bbox"), tile_id)
        if any(
            not math.isclose(actual, expected, abs_tol=1e-5)
            for actual, expected in zip(bbox, expected_tile.normalized_bbox)
        ):
            raise ValueError(
                f"completed WATCH result for {tile_id} has the wrong grid bbox"
            )
        if raw_tile.get("status") not in {"normal", "uncertain", "deviation"}:
            raise ValueError(f"completed WATCH result for {tile_id} has invalid status")

    return {
        "version": 1,
        "region_type": "grid",
        "bbox_format": "normalized_xywh",
        "sequence": sequence,
        "completed_unix_ms": completed_unix_ms,
        "captured_unix_ms": captured_unix_ms,
        "source_image": str(frame_source),
        "source_frame": {
            "device": frame_stamp.device,
            "inode": frame_stamp.inode,
            "size": frame_stamp.size,
            "mtime_ns": frame_stamp.mtime_ns,
        },
        "grid": {
            "rows": rows,
            "columns": columns,
        },
        "watch": {
            "batch_mode": output.get("batch_mode"),
            "request_count": output.get("request_count"),
            "parse_failure_count": output.get("parse_failure_count"),
            "inference_seconds": output.get("inference_seconds"),
            "wall_clock_seconds": output.get("wall_clock_seconds"),
        },
        "tiles": dict(raw_tiles),
    }


def watch_live(
    model: VisionLanguageModel,
    frame_source: Path,
    baseline_path: Path,
    status_output: Path,
    tile_size: int,
    force_row_batches: bool,
    watch_interval: float,
    poll_interval: float,
    max_passes: int | None,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    shared_grid: SharedGridConfig | None = None,
) -> int:
    baseline = load_baseline(baseline_path)
    require_matching_grid(baseline, shared_grid)
    tiles = build_grid(baseline.rows, baseline.columns)
    print(f"LIVE_FRAME_SOURCE: {frame_source}")
    print(f"LIVE_STATUS_OUTPUT: {status_output}")
    print(f"LIVE_WATCH_INTERVAL_SECONDS: {watch_interval:g}")
    print("LIVE_WATCH_WAITING_FOR_FRAME: true")

    # A clip-1 frame may still be present while ACTIVE starts. Watermark it and
    # wait for C++ to atomically publish the first post-start clip-2 frame.
    previous = live_frame_stamp(frame_source)
    next_pass_not_before = time.monotonic()
    completed_passes = 0
    previous_tiles: Mapping[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="tripwire-watch-") as directory:
        captured_path = Path(directory) / "watch_frame.jpg"
        while max_passes is None or completed_passes < max_passes:
            captured_stamp = capture_next_live_frame(
                frame_source,
                captured_path,
                previous,
                next_pass_not_before,
                poll_interval,
            )
            previous = captured_stamp
            captured_unix_ms = time.time_ns() // 1_000_000
            pass_started = time.monotonic()
            print(
                f"LIVE_WATCH_CAPTURE: mtime_ns={captured_stamp.mtime_ns} "
                f"size={captured_stamp.size}"
            )
            output = watch_pass(
                model,
                captured_path,
                baseline,
                tiles,
                tile_size,
                force_row_batches,
                diff_threshold,
                previous_tiles,
            )
            previous_tiles = output["tiles"]
            completed_passes += 1
            completed_unix_ms = time.time_ns() // 1_000_000
            snapshot = build_status_snapshot(
                output,
                completed_passes,
                completed_unix_ms,
                captured_unix_ms,
                frame_source,
                captured_stamp,
            )
            write_json_atomic(status_output, snapshot)
            print(f"LIVE_STATUS_SEQUENCE: {completed_passes}")
            print(f"LIVE_STATUS_COMPLETED_UNIX_MS: {completed_unix_ms}")
            print(f"LIVE_STATUS_SNAPSHOT: {status_output}")
            next_pass_not_before = pass_started + watch_interval
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_paths(args)
    shared_grid = load_shared_grid_config(args.config)

    import pyneat as neat

    model = neat.genai.VisionLanguageModel(str(args.model))
    if not model.accepts_image():
        raise RuntimeError(f"model is not image-capable: {args.model}")
    print("VLM_READY: true", flush=True)

    if args.mode == "calibrate":
        calibrate(
            model,
            args.images,
            args.baseline,
            shared_grid.rows,
            shared_grid.columns,
            args.tile_size,
            args.force_row_batches,
        )
        return 0
    if args.mode == "calibrate-live":
        calibrate_live(
            model,
            args.frame_source,
            args.baseline,
            shared_grid.rows,
            shared_grid.columns,
            args.tile_size,
            args.force_row_batches,
            args.frame_count,
            args.sample_interval,
            args.poll_interval,
        )
        return 0
    if args.mode == "watch-live":
        return watch_live(
            model,
            args.frame_source,
            args.baseline,
            args.status_output,
            args.tile_size,
            args.force_row_batches,
            args.watch_interval,
            args.poll_interval,
            args.max_passes,
            args.diff_threshold,
            shared_grid,
        )
    return watch(
        model,
        args.images,
        args.baseline,
        args.tile_size,
        args.force_row_batches,
        args.diff_threshold,
        shared_grid,
    )


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
