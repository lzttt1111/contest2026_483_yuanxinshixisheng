from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.wrinkle.report_region_overlays import (
    REGION_GROUPS,
    render_legacy_group_overlays,
)
from src.wrinkle.algorithms.wrinkle_detection_algorithm import (
    FACE_REGION_INDICES,
    build_aesthetic_region_masks,
)


class WrinkleLegacyReportOverlayTests(unittest.TestCase):
    def test_region_builder_keeps_empty_report_domains_for_distinct_overlays(self) -> None:
        shape = (200, 200)
        points = np.full((478, 2), (100, 100), dtype=np.int32)

        def place(indices: list[int], center: tuple[int, int], axes: tuple[int, int]) -> None:
            for position, index in enumerate(indices):
                angle = 2.0 * np.pi * position / len(indices)
                points[index] = (
                    int(round(center[0] + axes[0] * np.cos(angle))),
                    int(round(center[1] + axes[1] * np.sin(angle))),
                )

        place(FACE_REGION_INDICES["face_oval"], (100, 105), (72, 90))
        place(FACE_REGION_INDICES["right_eye"], (68, 82), (17, 7))
        place(FACE_REGION_INDICES["left_eye"], (132, 82), (17, 7))
        place(FACE_REGION_INDICES["right_eyebrow"], (68, 66), (19, 5))
        place(FACE_REGION_INDICES["left_eyebrow"], (132, 66), (19, 5))
        place(FACE_REGION_INDICES["lips"], (100, 145), (24, 8))
        place(FACE_REGION_INDICES["nose_lower"], (100, 118), (12, 8))

        face_skin = np.full(shape, 255, dtype=np.uint8)
        empty = np.zeros(shape, dtype=np.uint8)
        allowed = cv2.rectangle(
            empty.copy(), (35, 20), (165, 75), 255, -1
        )
        filters = {
            "face_skin": face_skin,
            "wrinkle_allowed_zone": allowed,
            "forehead_zone": cv2.rectangle(empty.copy(), (35, 20), (165, 68), 255, -1),
            "glabella_zone": cv2.rectangle(empty.copy(), (88, 62), (112, 100), 255, -1),
            "eyes_forbidden": empty,
            "eyebrows_forbidden": empty,
            "nose_forbidden": empty,
            "lips_forbidden": empty,
            "mustache_forbidden": empty,
        }
        one_fold = empty.copy()
        cv2.line(one_fold, (88, 118), (76, 145), 255, 2)

        region_defs = build_aesthetic_region_masks(
            points,
            FACE_REGION_INDICES,
            filters,
            one_fold,
        )
        returned_keys = {definition[0] for definition in region_defs}

        self.assertTrue(
            set(REGION_GROUPS["07"]["region_keys"]).issubset(returned_keys)
        )
        self.assertTrue(
            set(REGION_GROUPS["08"]["region_keys"]).issubset(returned_keys)
        )
        self.assertTrue(
            set(REGION_GROUPS["09"]["region_keys"]).issubset(returned_keys)
        )

    def test_grouped_overlays_select_only_declared_regions(self) -> None:
        image = np.full((96, 128, 3), 90, dtype=np.uint8)
        keys = [
            key
            for group in REGION_GROUPS.values()
            for key in group["region_keys"]
        ]
        region_defs = []
        for index, key in enumerate(keys):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            line = np.zeros_like(mask)
            x1 = 3 + (index % 5) * 24
            y1 = 3 + (index // 5) * 40
            cv2.rectangle(mask, (x1, y1), (x1 + 16, y1 + 24), 255, -1)
            cv2.line(line, (x1 + 4, y1 + 12), (x1 + 12, y1 + 12), 255, 1)
            region_defs.append((key, key, key, mask, line))

        with tempfile.TemporaryDirectory() as directory:
            manifest = render_legacy_group_overlays(image, region_defs, directory)
            self.assertEqual(set(manifest), {"07", "08", "09"})
            self.assertEqual(
                manifest["07"]["selected_region_keys"],
                ["left_under_eye", "right_under_eye"],
            )
            self.assertEqual(len(manifest["08"]["selected_region_keys"]), 4)
            self.assertEqual(len(manifest["09"]["selected_region_keys"]), 4)
            self.assertEqual(
                len({Path(row["path"]).read_bytes() for row in manifest.values()}),
                3,
            )

    def test_pipeline_keeps_balanced_preset_and_enables_local_extra_images(self) -> None:
        from src.wrinkle.pipeline import Pipeline

        pipeline = object.__new__(Pipeline)
        args = pipeline._build_algorithm_args(
            source_path=Path("face.jpg"),
            output_dir=Path("output"),
            weights_path=Path("weights.pt"),
            face_model_path=Path("face.task"),
            segmenter_model_path=Path("segmenter.tflite"),
            preset="balanced",
        )
        self.assertEqual(args.run_preset, "balanced")
        self.assertFalse(args.save_runs)
        self.assertTrue(args.report_group_overlays)


if __name__ == "__main__":
    unittest.main()
