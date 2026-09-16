#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


def install_host_test_stubs() -> None:
    """The host lacks DevKit OpenCV/NumPy; IPC tests need only two tiny APIs."""
    try:
        __import__("numpy")
    except ModuleNotFoundError:
        numpy = types.ModuleType("numpy")
        numpy.ndarray = object
        numpy.uint8 = object()
        numpy.frombuffer = lambda payload, dtype: payload
        sys.modules["numpy"] = numpy
    try:
        __import__("cv2")
    except ModuleNotFoundError:
        cv2 = types.ModuleType("cv2")
        cv2.IMREAD_COLOR = 1
        cv2.imdecode = lambda encoded, flags: object()
        sys.modules["cv2"] = cv2


install_host_test_stubs()

import tripwire  # noqa: E402


class LiveIpcTest(unittest.TestCase):
    def test_vlm_ready_marker_is_validated_and_flushed(self) -> None:
        events: list[str] = []

        class FakeModel:
            def __init__(self, model_path: str) -> None:
                events.append("constructed")

            def accepts_image(self) -> bool:
                events.append("validated")
                return True

        fake_pyneat = types.ModuleType("pyneat")
        fake_pyneat.genai = types.SimpleNamespace(VisionLanguageModel=FakeModel)
        args = Namespace(
            mode="watch-live",
            model=Path("/fake/model"),
            config=tripwire.DEFAULT_APP_CONFIG,
            frame_source=Path("/fake/latest_frame.jpg"),
            baseline=Path("/fake/baseline.json"),
            status_output=Path("/fake/tile_status.json"),
            tile_size=448,
            force_row_batches=False,
            diff_threshold=tripwire.DEFAULT_DIFF_THRESHOLD,
            watch_interval=5.0,
            poll_interval=0.1,
            max_passes=1,
        )

        def record_print(*values: object, **options: object) -> None:
            if values == ("VLM_READY: true",):
                self.assertIs(options.get("flush"), True)
                events.append("ready")

        with (
            mock.patch.object(tripwire, "parse_args", return_value=args),
            mock.patch.object(tripwire, "validate_paths"),
            mock.patch.object(
                tripwire,
                "load_shared_grid_config",
                return_value=tripwire.SharedGridConfig(6, 8),
            ),
            mock.patch.object(tripwire, "watch_live", return_value=0),
            mock.patch.dict(sys.modules, {"pyneat": fake_pyneat}),
            mock.patch("builtins.print", side_effect=record_print),
        ):
            self.assertEqual(tripwire.main([]), 0)

        self.assertEqual(events, ["constructed", "validated", "ready"])

    def test_live_cli_defaults(self) -> None:
        calibrate_args = tripwire.parse_args(["calibrate-live"])
        self.assertEqual(calibrate_args.frame_source, tripwire.DEFAULT_LIVE_FRAME)
        self.assertEqual(calibrate_args.frame_count, 3)
        self.assertEqual(calibrate_args.config, tripwire.DEFAULT_APP_CONFIG)
        shared_grid = tripwire.load_shared_grid_config(calibrate_args.config)
        self.assertEqual(shared_grid.columns, 8)
        self.assertEqual(shared_grid.rows, 6)

        watch_args = tripwire.parse_args(["watch-live", "--max-passes", "1"])
        self.assertEqual(watch_args.status_output, tripwire.DEFAULT_STATUS_OUTPUT)
        self.assertEqual(watch_args.max_passes, 1)
        self.assertEqual(watch_args.diff_threshold, tripwire.DEFAULT_DIFF_THRESHOLD)

    def test_baseline_grid_must_match_shared_config(self) -> None:
        baseline = tripwire.GridBaseline(3, 4, 4, 3, {})
        with self.assertRaisesRegex(ValueError, "baseline grid does not match"):
            tripwire.require_matching_grid(
                baseline,
                tripwire.SharedGridConfig(6, 8),
            )

    def test_capture_live_frame_is_immutable_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "latest_frame.jpg"
            destination = root / "captured.jpg"
            payload = b"complete atomic jpeg payload"
            source.write_bytes(payload)

            with mock.patch.object(
                tripwire.cv2,
                "imdecode",
                return_value=object(),
            ):
                stamp = tripwire.capture_live_frame(source, destination)
            replacement = root / "replacement.tmp"
            replacement.write_bytes(b"replacement frame")
            replacement.replace(source)

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(stamp.size, len(payload))
            self.assertNotEqual(tripwire.live_frame_stamp(source), stamp)

    def test_live_modes_watermark_preexisting_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "latest_frame.jpg"
            source.write_bytes(b"stale clip-one frame")
            stale_stamp = tripwire.live_frame_stamp(source)
            fresh_stamp = tripwire.FrameStamp(9, 8, 7, 6)

            with (
                mock.patch.object(
                    tripwire,
                    "capture_next_live_frame",
                    return_value=fresh_stamp,
                ) as capture,
                mock.patch.object(tripwire, "calibrate"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                tripwire.calibrate_live(
                    model=mock.Mock(),
                    frame_source=source,
                    baseline_path=root / "baseline.json",
                    rows=3,
                    columns=4,
                    tile_size=448,
                    force_row_batches=False,
                    frame_count=1,
                    sample_interval=5.0,
                    poll_interval=0.1,
                )
            self.assertEqual(capture.call_args.args[2], stale_stamp)

            baseline = tripwire.GridBaseline(3, 4, 4, 3, {})
            with (
                mock.patch.object(tripwire, "load_baseline", return_value=baseline),
                mock.patch.object(
                    tripwire,
                    "capture_next_live_frame",
                    return_value=fresh_stamp,
                ) as capture,
                mock.patch.object(
                    tripwire,
                    "watch_pass",
                    return_value={"tiles": {}},
                ),
                mock.patch.object(tripwire, "build_status_snapshot", return_value={}),
                mock.patch.object(tripwire, "write_json_atomic"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                tripwire.watch_live(
                    model=mock.Mock(),
                    frame_source=source,
                    baseline_path=root / "baseline.json",
                    status_output=root / "tile_status.json",
                    tile_size=448,
                    force_row_batches=False,
                    watch_interval=5.0,
                    poll_interval=0.1,
                    max_passes=1,
                )
            self.assertEqual(capture.call_args.args[2], stale_stamp)

    def test_reference_frame_is_stored_next_to_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "calibration.jpg"
            image.write_bytes(b"reference image bytes")
            baseline = root / "custom_baseline.json"

            reference = tripwire.store_reference_frame(image, baseline)

            self.assertEqual(reference, root / "custom_baseline_reference.jpg")
            self.assertEqual(reference.read_bytes(), image.read_bytes())
            self.assertFalse(reference.with_suffix(".jpg.tmp").exists())

            tile = tripwire.build_grid(1, 1)[0]
            baseline.write_text(
                json.dumps(
                    {
                        "version": 3,
                        "region_type": "grid",
                        "bbox_format": "normalized_xywh",
                        "reference_frame": reference.name,
                        "grid": {"rows": 1, "columns": 1},
                        "source": {"width": 100, "height": 100},
                        "tiles": {
                            tile.tile_id: {
                                "normalized_bbox": list(tile.normalized_bbox),
                                "description": "A connected power cable is visible.",
                                "observation_count": 1,
                                "status": "calibrating",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = tripwire.load_baseline(baseline)
            self.assertEqual(loaded.reference_frame, reference)

    def test_diff_gate_sends_only_changed_tiles_and_retains_status(self) -> None:
        tiles = tripwire.build_grid(1, 2)
        baseline_tiles = {
            tile.tile_id: tripwire.BaselineTile(
                tile.tile_id,
                tile.normalized_bbox,
                "A permanent device and cable are visible.",
                3,
            )
            for tile in tiles
        }
        baseline = tripwire.GridBaseline(
            1,
            2,
            200,
            100,
            baseline_tiles,
            Path("/fake/reference.jpg"),
        )
        frame = types.SimpleNamespace(shape=(100, 200, 3))
        previous_tiles = {
            tiles[0].tile_id: {
                "normalized_bbox": list(tiles[0].normalized_bbox),
                "observation": "The cable connection could not be seen.",
                "reason": "The previous frame was occluded.",
                "verdict": "UNCERTAIN",
                "status": "uncertain",
            }
        }
        changed_verdict = tripwire.TileVerdict(
            "DEVIATION",
            "The connected baseline cable is now unplugged.",
            "The cable is visibly unplugged from the device.",
        )
        batch = tripwire.BatchResult(
            {tiles[1].tile_id: changed_verdict},
            "single_tile_requests",
            1,
            4.0,
            ("raw",),
            frozenset(),
        )

        with (
            mock.patch.object(tripwire, "decode_image", side_effect=[frame, frame]),
            mock.patch.object(
                tripwire,
                "tile_difference_scores",
                return_value={tiles[0].tile_id: 0.01, tiles[1].tile_id: 0.08},
            ),
            mock.patch.object(
                tripwire,
                "tile_crops_from_image",
                return_value={tiles[1].tile_id: object()},
            ) as cropper,
            mock.patch.object(
                tripwire,
                "judge_frame_tiles",
                return_value=batch,
            ) as judge,
            contextlib.redirect_stdout(io.StringIO()) as output_stream,
        ):
            output = tripwire.watch_pass(
                mock.Mock(),
                Path("/fake/current.jpg"),
                baseline,
                tiles,
                448,
                False,
                0.03,
                previous_tiles,
            )

        sent_tiles = judge.call_args.args[2]
        self.assertEqual([tile.tile_id for tile in sent_tiles], [tiles[1].tile_id])
        self.assertIs(cropper.call_args.args[5], False)
        self.assertEqual(output["gated_tile_count"], 1)
        self.assertEqual(output["sent_tile_count"], 1)
        self.assertEqual(output["tiles"][tiles[0].tile_id]["status"], "uncertain")
        self.assertEqual(output["tiles"][tiles[1].tile_id]["status"], "deviation")
        self.assertIn("WATCH_GATED_TILE_COUNT: 1", output_stream.getvalue())
        self.assertIn("WATCH_SENT_TILE_COUNT: 1", output_stream.getvalue())
        self.assertIn("WATCH_PARSE_FAILURE_COUNT: 0", output_stream.getvalue())
        self.assertIn("WATCH_PASS_TIME_SECONDS:", output_stream.getvalue())

    def test_diff_gate_can_skip_all_vlm_inference(self) -> None:
        tile = tripwire.build_grid(1, 1)[0]
        baseline = tripwire.GridBaseline(
            1,
            1,
            100,
            100,
            {
                tile.tile_id: tripwire.BaselineTile(
                    tile.tile_id,
                    tile.normalized_bbox,
                    "A connected power cable is visible.",
                    1,
                )
            },
            Path("/fake/reference.jpg"),
        )
        frame = types.SimpleNamespace(shape=(100, 100, 3))
        with (
            mock.patch.object(tripwire, "decode_image", side_effect=[frame, frame]),
            mock.patch.object(
                tripwire,
                "tile_difference_scores",
                return_value={tile.tile_id: 0.03},
            ),
            mock.patch.object(tripwire, "judge_frame_tiles") as judge,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            output = tripwire.watch_pass(
                mock.Mock(),
                Path("/fake/current.jpg"),
                baseline,
                [tile],
                448,
                False,
                0.03,
            )

        judge.assert_not_called()
        self.assertEqual(output["batch_mode"], "diff_gated_no_inference")
        self.assertEqual(output["request_count"], 0)
        self.assertEqual(output["gated_tile_count"], 1)
        self.assertEqual(output["sent_tile_count"], 0)
        self.assertEqual(output["tiles"][tile.tile_id]["status"], "normal")

    def test_changed_tile_without_baseline_is_uncertain_and_not_sent(self) -> None:
        tile = tripwire.build_grid(1, 1)[0]
        baseline = tripwire.GridBaseline(
            1,
            1,
            100,
            100,
            {
                tile.tile_id: tripwire.BaselineTile(
                    tile.tile_id,
                    tile.normalized_bbox,
                    "",
                    0,
                )
            },
            Path("/fake/reference.jpg"),
        )
        frame = types.SimpleNamespace(shape=(100, 100, 3))
        with (
            mock.patch.object(tripwire, "decode_image", side_effect=[frame, frame]),
            mock.patch.object(
                tripwire,
                "tile_difference_scores",
                return_value={tile.tile_id: 0.5},
            ),
            mock.patch.object(tripwire, "tile_crops_from_image") as cropper,
            mock.patch.object(tripwire, "judge_frame_tiles") as judge,
            contextlib.redirect_stdout(io.StringIO()) as output_stream,
        ):
            output = tripwire.watch_pass(
                mock.Mock(),
                Path("/fake/current.jpg"),
                baseline,
                [tile],
                448,
                False,
                0.03,
            )

        cropper.assert_not_called()
        judge.assert_not_called()
        self.assertEqual(output["gated_tile_count"], 0)
        self.assertEqual(output["sent_tile_count"], 0)
        self.assertEqual(output["baseline_missing_tile_count"], 1)
        self.assertEqual(output["parse_failure_count"], 0)
        self.assertEqual(output["tiles"][tile.tile_id]["status"], "uncertain")
        self.assertIn(
            "WATCH_BASELINE_MISSING_TILE_COUNT: 1",
            output_stream.getvalue(),
        )

    def test_large_grid_calibration_goes_directly_to_row_batches(self) -> None:
        tiles = tripwire.build_grid(6, 8)
        crops = {tile.tile_id: object() for tile in tiles}
        requests: list[list[str]] = []

        def fake_request(
            prompt: str,
            images: list[object],
            max_new_tokens: int,
            image_tile_ids: list[str],
        ) -> list[str]:
            requests.append(list(image_tile_ids))
            return list(image_tile_ids)

        def fake_run(model: object, tile_ids: list[str]) -> tuple[str, float]:
            response = "\n".join(
                f"{tile_id}|DESCRIPTION=A distinct permanent object at "
                f"position {index} is visible."
                for index, tile_id in enumerate(tile_ids)
            )
            return response, 1.0

        with (
            mock.patch.object(tripwire, "make_request", side_effect=fake_request),
            mock.patch.object(tripwire, "timed_run", side_effect=fake_run),
        ):
            result = tripwire.describe_frame_tiles(
                mock.Mock(),
                tiles,
                crops,
                6,
                False,
            )

        self.assertEqual(result.mode, "row_batches")
        self.assertEqual(result.request_count, 6)
        self.assertFalse(result.missing)
        self.assertEqual(len(requests), 6)
        self.assertTrue(all(len(request) == 8 for request in requests))

    def test_calibration_requires_checkable_spatial_and_connection_state(self) -> None:
        self.assertFalse(
            tripwire.valid_calibration_description(
                "A development kit and electronic device are visible, with a cable "
                "or cord also visible."
            )
        )
        self.assertFalse(
            tripwire.valid_calibration_description("A chair is visible.")
        )
        self.assertTrue(
            tripwire.valid_calibration_description(
                "A white cable runs left-to-right along the lower edge, with its "
                "left end plugged into the laptop and right end plugged into the "
                "power strip."
            )
        )
        self.assertTrue(
            tripwire.valid_calibration_description(
                "A black cable crosses the center horizontally; both ends leave "
                "the image, so their connection state cannot be determined."
            )
        )

    def test_single_calibration_observation_is_not_rewritten_as_inventory(self) -> None:
        description = (
            "A white cable runs from the laptop's left port to the power strip "
            "along the lower edge, with both ends plugged in."
        )
        self.assertEqual(
            tripwire.merge_tile_descriptions(mock.Mock(), "T_r0c0", [description]),
            description,
        )

    def test_calibration_prompt_forbids_generic_inventory(self) -> None:
        prompt = tripwire.calibration_prompt(["T_r0c0"])
        self.assertIn("name what EACH visible end is", prompt)
        self.assertIn("connected to, such as a laptop's left port", prompt)
        self.assertRegex(prompt, r"position and\s+orientation")
        self.assertRegex(prompt, r"connection\s+state cannot be determined")
        self.assertIn("Forbid generic inventories", prompt)

    def test_calibration_records_empty_tile_after_retry_failure(self) -> None:
        tiles = tripwire.build_grid(1, 2)
        successful_tile = tiles[0].tile_id
        missing_tile = tiles[1].tile_id

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "calibration_frame_1.jpg"
            image_path.write_bytes(b"reference image bytes")
            baseline_path = root / "baseline.json"
            result = tripwire.BatchResult(
                {successful_tile: "A permanent device is visible."},
                "row_batches_with_tile_retries",
                3,
                1.0,
                ("partial response",),
                frozenset({missing_tile}),
            )

            with (
                mock.patch.object(
                    tripwire,
                    "load_tile_crops",
                    return_value=(200, 100, {tile.tile_id: object() for tile in tiles}),
                ),
                mock.patch.object(
                    tripwire,
                    "describe_frame_tiles",
                    return_value=result,
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                tripwire.calibrate(
                    mock.Mock(),
                    [image_path],
                    baseline_path,
                    1,
                    2,
                    448,
                    False,
                )

            document = json.loads(baseline_path.read_text(encoding="utf-8"))
            self.assertTrue(document["tiles"][successful_tile]["description"])
            self.assertEqual(
                document["tiles"][successful_tile]["observation_count"],
                1,
            )
            self.assertEqual(document["tiles"][missing_tile]["description"], "")
            self.assertEqual(
                document["tiles"][missing_tile]["observation_count"],
                0,
            )
            loaded = tripwire.load_baseline(baseline_path)
            self.assertEqual(loaded.tiles[missing_tile].description, "")
            self.assertEqual(loaded.tiles[missing_tile].observation_count, 0)

    @unittest.skipUnless(
        hasattr(tripwire.np, "zeros"),
        "requires the real NumPy runtime",
    )
    def test_rms_difference_is_computed_per_tile(self) -> None:
        tiles = tripwire.build_grid(1, 2)
        reference = tripwire.np.zeros((100, 120, 3), dtype=tripwire.np.uint8)
        current = reference.copy()
        current[:, 60:, :] = 255

        scores = tripwire.tile_difference_scores(
            current,
            reference,
            tiles,
            1,
            2,
        )

        self.assertAlmostEqual(scores[tiles[0].tile_id], 0.0)
        self.assertAlmostEqual(scores[tiles[1].tile_id], 1.0)

    def test_watch_prompt_restores_observe_compare_decide_order(self) -> None:
        baseline_description = (
            "A chair is visible, with a cable visibly disconnected or hanging loose."
        )
        prompt = tripwire.watch_prompt(baseline_description)

        self.assertLess(prompt.index("OBSERVATION=<"), prompt.index("REASON=<"))
        self.assertLess(prompt.index("REASON=<"), prompt.index("VERDICT=<"))
        self.assertIn(
            f"BASELINE_SENTENCE={json.dumps(baseline_description)}",
            prompt,
        )
        self.assertIn("The ONLY question", prompt)
        self.assertIn("CHANGED from what the quoted baseline recorded", prompt)
        self.assertIn("DOES NOT CONTRADICT", prompt)
        self.assertIn("already records that same loose", prompt)
        self.assertIn("Do not invent a change", prompt)
        self.assertIn("people, hands, screen contents, shadows, reflections", prompt)
        self.assertIn("use UNCERTAIN, never DEVIATION", prompt)
        self.assertIn("REASON must be exactly one concise sentence", prompt)
        self.assertIn("Do not enumerate every object", prompt)
        self.assertNotIn("T_r", prompt)

    def test_watch_request_contains_one_unlabelled_image_message(self) -> None:
        class FakeRequest:
            pass

        class FakeMessage:
            pass

        fake_pyneat = types.ModuleType("pyneat")
        fake_pyneat.genai = types.SimpleNamespace(
            GenerationRequest=FakeRequest,
            ChatMessage=FakeMessage,
        )
        image = object()
        prompt = tripwire.watch_prompt("A device remains upright in the center.")

        with mock.patch.dict(sys.modules, {"pyneat": fake_pyneat}):
            request = tripwire.make_watch_request(prompt, image)

        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0].content, prompt)
        self.assertEqual(request.messages[0].images, [image])
        self.assertNotIn("T_r", request.messages[0].content)
        self.assertEqual(tripwire.WATCH_MAX_NEW_TOKENS, 200)
        self.assertEqual(request.max_new_tokens, tripwire.WATCH_MAX_NEW_TOKENS)

    def test_watch_rejects_match_that_contradicts_connected_baseline(self) -> None:
        response = (
            "OBSERVATION=The power strip is unplugged and disconnected from the wall.\n"
            "REASON=DOES NOT CONTRADICT: The current state agrees with the baseline.\n"
            "VERDICT=MATCH"
        )

        self.assertIsNone(
            tripwire.parse_watch_response(
                response,
                "A power strip is plugged into the wall outlet.",
            )
        )

    def test_watch_accepts_matching_disconnected_baseline(self) -> None:
        response = (
            "OBSERVATION=The power cable remains loose and disconnected.\n"
            "REASON=DOES NOT CONTRADICT: The baseline records the same loose state.\n"
            "VERDICT=MATCH"
        )

        parsed = tripwire.parse_watch_response(
            response,
            "A loose power cable at lower right is disconnected.",
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.verdict, "MATCH")
        self.assertEqual(
            parsed.observation,
            "The power cable remains loose and disconnected.",
        )
        self.assertNotEqual(parsed.observation, parsed.reason)

    def test_watch_parser_requires_observation_reason_verdict_order(self) -> None:
        response = (
            "REASON=CONTRADICTS: The device changed orientation.\n"
            "OBSERVATION=The device is now lying on its side.\n"
            "VERDICT=DEVIATION"
        )

        self.assertIsNone(
            tripwire.parse_watch_response(
                response,
                "The device stands upright in the center.",
            )
        )

    def test_watch_parser_accepts_colons_and_truncated_reason(self) -> None:
        compact = tripwire.parse_watch_response(
            "OBSERVATION:The chair is lying flat.\n"
            "REASON:CONTRADICTS: The woven chair differs because its\n"
            "VERDICT:DEVIATION",
            "The chair is upright and faces left.",
        )
        spaced = tripwire.parse_watch_response(
            "OBSERVATION: The image shows a dark, flat surface.\n"
            "REASON: DOES NOT CONTRADICT: The recorded surface remains present.\n"
            "VERDICT: MATCH",
            "A dark, flat surface crosses the center.",
        )

        self.assertIsNotNone(compact)
        self.assertEqual(compact.verdict, "DEVIATION")
        self.assertEqual(
            compact.reason,
            "CONTRADICTS: The woven chair differs because its",
        )
        self.assertIsNotNone(spaced)
        self.assertEqual(spaced.verdict, "MATCH")

    def test_watch_parser_rejects_reason_verdict_mismatch(self) -> None:
        response = (
            "OBSERVATION=The cable remains plugged into the device.\n"
            "REASON=DOES NOT CONTRADICT: The recorded connection remains intact.\n"
            "VERDICT=DEVIATION"
        )

        self.assertIsNone(
            tripwire.parse_watch_response(
                response,
                "The cable is plugged into the device.",
            )
        )

    def test_watch_parser_maps_reason_vocabulary_in_verdict_field(self) -> None:
        for raw_verdict, expected in (
            ("CONTRADICTS", "DEVIATION"),
            ("DOES NOT CONTRADICT", "MATCH"),
            ("UNCERTAIN", "UNCERTAIN"),
        ):
            with self.subTest(raw_verdict=raw_verdict):
                response = (
                    "OBSERVATION: The device has a clearly visible current state.\n"
                    "REASON: The current state was compared directly with the "
                    "recorded state.\n"
                    f"VERDICT: {raw_verdict}"
                )
                parsed = tripwire.parse_watch_response(
                    response,
                    "The device has a recorded physical state.",
                )
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.verdict, expected)

        explained = tripwire.parse_watch_response(
            "OBSERVATION: The cable remains plugged into the power strip.\n"
            "REASON: The observed connection agrees with the recorded state.\n"
            "VERDICT: DOES NOT CONTRADICT: The connection is unchanged.",
            "The cable is plugged into the power strip.",
        )
        self.assertIsNotNone(explained)
        self.assertEqual(explained.verdict, "MATCH")

    def test_watch_parser_accepts_canonical_verdict_with_prose_reason(self) -> None:
        parsed = tripwire.parse_watch_response(
            "OBSERVATION: The cable remains plugged into the power outlet.\n"
            "REASON: The current connection is the same as the baseline state.\n"
            "VERDICT: MATCH",
            "The cable is plugged into the power outlet.",
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.verdict, "MATCH")

    def test_timed_watch_run_stops_after_complete_verdict_line(self) -> None:
        class Sample:
            def __init__(self, text: str, is_final: bool = False) -> None:
                self.text = text
                self.is_final = is_final

        class Stream:
            def __init__(self) -> None:
                self.cancelled = False
                self.samples = (
                    Sample(
                        "OBSERVATION: The cable is plugged into the outlet.\n"
                        "REASON: CONTRADICTS: The baseline records it unplugged.\n"
                        "VERDICT: CONTRADICTS"
                    ),
                    Sample("\nThis text must never be consumed."),
                )

            def __iter__(self):
                return iter(self.samples)

            def cancel(self) -> None:
                self.cancelled = True

        stream = Stream()
        model = mock.Mock()
        model.stream.return_value = stream

        response, _ = tripwire.timed_watch_run(model, object())

        self.assertTrue(stream.cancelled)
        self.assertTrue(response.endswith("VERDICT: CONTRADICTS"))
        self.assertNotIn("never be consumed", response)

    def test_watch_uses_one_image_and_one_baseline_per_request(self) -> None:
        tiles = tripwire.build_grid(1, 3)
        baseline = tripwire.GridBaseline(
            1,
            3,
            300,
            100,
            {
                tile.tile_id: tripwire.BaselineTile(
                    tile.tile_id,
                    tile.normalized_bbox,
                    f"Baseline state for object number {index}.",
                    1,
                )
                for index, tile in enumerate(tiles)
            },
        )
        crops = {tile.tile_id: f"crop-{tile.tile_id}" for tile in tiles}
        requests: list[tuple[str, object]] = []

        def fake_request(prompt: str, image: object) -> object:
            requests.append((prompt, image))
            return image

        def fake_run(model: object, request: object) -> tuple[str, float]:
            return (
                "OBSERVATION=The visible object remains in its recorded state.\n"
                "REASON=DOES NOT CONTRADICT: It is compatible with the baseline.\n"
                "VERDICT=MATCH",
                1.0,
            )

        with (
            mock.patch.object(
                tripwire,
                "make_watch_request",
                side_effect=fake_request,
            ),
            mock.patch.object(tripwire, "timed_watch_run", side_effect=fake_run),
        ):
            result = tripwire.judge_frame_tiles(
                mock.Mock(), baseline, tiles, crops, True
            )

        self.assertFalse(result.missing)
        self.assertEqual(result.mode, "single_tile_requests")
        self.assertEqual(result.request_count, 3)
        self.assertEqual(result.inference_seconds, 3.0)
        self.assertEqual([image for _, image in requests], list(crops.values()))
        for index, (prompt, _) in enumerate(requests):
            self.assertIn(f"Baseline state for object number {index}.", prompt)
            self.assertNotIn("T_r", prompt)

    def test_watch_tile_failure_does_not_abort_later_requests(self) -> None:
        tiles = tripwire.build_grid(1, 3)
        baseline = tripwire.GridBaseline(
            1,
            3,
            300,
            100,
            {
                tile.tile_id: tripwire.BaselineTile(
                    tile.tile_id,
                    tile.normalized_bbox,
                    "A device remains upright in the center.",
                    1,
                )
                for tile in tiles
            },
        )
        crops = {tile.tile_id: index for index, tile in enumerate(tiles)}

        def fake_request(prompt: str, image: int) -> int:
            return image

        def fake_run(model: object, request: int) -> tuple[str, float]:
            responses = (
                "OBSERVATION=The device remains upright in the center.\n"
                "REASON=DOES NOT CONTRADICT: Its position matches the baseline.\n"
                "VERDICT=MATCH",
                "not parseable",
                "OBSERVATION=The device is now lying on its side.\n"
                "REASON=CONTRADICTS: Its orientation differs from the baseline.\n"
                "VERDICT=DEVIATION",
            )
            return responses[request], 1.0

        with (
            mock.patch.object(
                tripwire,
                "make_watch_request",
                side_effect=fake_request,
            ),
            mock.patch.object(tripwire, "timed_watch_run", side_effect=fake_run),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = tripwire.judge_frame_tiles(
                mock.Mock(), baseline, tiles, crops, False
            )

        self.assertEqual(result.request_count, 3)
        self.assertEqual(result.missing, frozenset({tiles[1].tile_id}))
        self.assertEqual(result.values[tiles[0].tile_id].verdict, "MATCH")
        self.assertEqual(result.values[tiles[2].tile_id].verdict, "DEVIATION")

    def test_first_three_watch_parse_failures_print_raw_response(self) -> None:
        tile_ids = [f"T_r0c{column}" for column in range(5)]
        raw_response = "The model returned prose instead of two fields."
        error_output = io.StringIO()

        with contextlib.redirect_stderr(error_output):
            reported = 0
            for tile_id in tile_ids:
                reported = tripwire.print_watch_parse_failure_diagnostic(
                    tile_id,
                    raw_response,
                    reported,
                )

        diagnostic = error_output.getvalue()
        self.assertEqual(reported, 3)
        self.assertIn("WATCH_PARSE_FAILURE_1_TILE: T_r0c0", diagnostic)
        self.assertIn("WATCH_PARSE_FAILURE_2_TILE: T_r0c1", diagnostic)
        self.assertIn("WATCH_PARSE_FAILURE_3_TILE: T_r0c2", diagnostic)
        self.assertNotIn("WATCH_PARSE_FAILURE_4_TILE", diagnostic)
        self.assertIn(json.dumps(raw_response, ensure_ascii=False), diagnostic)
        self.assertIn("OBSERVATION=<one sentence>", diagnostic)
        self.assertIn("VERDICT=<MATCH or DEVIATION or UNCERTAIN>", diagnostic)
        self.assertIn("DOES NOT CONTRADICT", diagnostic)

    def test_completed_snapshot_matches_cpp_contract(self) -> None:
        shared_grid = tripwire.load_shared_grid_config(tripwire.DEFAULT_APP_CONFIG)
        tile_results = {}
        for tile in tripwire.build_grid(
            shared_grid.rows,
            shared_grid.columns,
        ):
            tile_results[tile.tile_id] = {
                "normalized_bbox": list(tile.normalized_bbox),
                "observation": "A stable object is visible.",
                "reason": "The object agrees with the baseline.",
                "verdict": "MATCH",
                "status": "normal",
            }
        output = {
            "grid": {
                "rows": shared_grid.rows,
                "columns": shared_grid.columns,
            },
            "batch_mode": "single_tile_requests",
            "request_count": 1,
            "parse_failure_count": 0,
            "inference_seconds": 30.0,
            "wall_clock_seconds": 31.0,
            "tiles": tile_results,
        }
        frame_stamp = tripwire.FrameStamp(1, 2, 3, 4)

        snapshot = tripwire.build_status_snapshot(
            output,
            sequence=7,
            completed_unix_ms=2000,
            captured_unix_ms=1000,
            frame_source=Path("/tmp/latest_frame.jpg"),
            frame_stamp=frame_stamp,
        )

        self.assertEqual(snapshot["version"], 1)
        self.assertEqual(snapshot["region_type"], "grid")
        self.assertEqual(snapshot["bbox_format"], "normalized_xywh")
        self.assertEqual(snapshot["sequence"], 7)
        self.assertEqual(snapshot["completed_unix_ms"], 2000)
        self.assertEqual(len(snapshot["tiles"]), 48)
        self.assertEqual(
            set(snapshot["watch"]),
            {
                "batch_mode",
                "request_count",
                "parse_failure_count",
                "inference_seconds",
                "wall_clock_seconds",
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "tile_status.json"
            tripwire.write_json_atomic(output_path, snapshot)
            reloaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded, snapshot)

            replacement_snapshot = dict(snapshot)
            replacement_snapshot["sequence"] = 8
            tripwire.write_json_atomic(output_path, replacement_snapshot)
            reloaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded, replacement_snapshot)
            self.assertFalse(output_path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
