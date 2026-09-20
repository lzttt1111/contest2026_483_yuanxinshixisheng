from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.engines.visia_unified_contour import (
    _front_template,
    build_unified_visia_scope,
)


class UnifiedVisiaContourTest(unittest.TestCase):
    def test_front_path_is_one_closed_connected_scope(self) -> None:
        landmarks, _ = _front_template()
        shape = (1024, 1024)
        envelope = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(envelope, (512, 540), (360, 455), 0, 0, 360, 255, -1)

        contour, mask, separator = build_unified_visia_scope(
            landmarks,
            shape,
            envelope,
            partial_face=False,
            left_visible_area=100_000,
            right_visible_area=100_000,
        )

        self.assertIsNotNone(contour)
        assert contour is not None
        self.assertTrue(np.array_equal(contour[0], contour[-1]))
        self.assertGreater(len(contour), 100)
        self.assertGreater(np.count_nonzero(mask), 1_000)
        self.assertIsNotNone(separator)
        assert separator is not None
        self.assertFalse(np.array_equal(separator[0], separator[-1]))
        contour_points = contour.reshape(-1, 2)
        self.assertTrue(np.any(np.all(contour_points == separator[0, 0], axis=1)))
        self.assertTrue(np.any(np.all(contour_points == separator[-1, 0], axis=1)))
        components, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
        self.assertEqual(components - 1, 1)

    def test_scope_is_deterministic(self) -> None:
        landmarks, _ = _front_template()
        shape = (1024, 1024)
        envelope = np.full(shape, 255, dtype=np.uint8)
        first_contour, first_mask, first_separator = build_unified_visia_scope(
            landmarks,
            shape,
            envelope,
            partial_face=False,
            left_visible_area=90_000,
            right_visible_area=91_000,
        )
        second_contour, second_mask, second_separator = build_unified_visia_scope(
            landmarks,
            shape,
            envelope,
            partial_face=False,
            left_visible_area=90_000,
            right_visible_area=91_000,
        )
        np.testing.assert_array_equal(first_contour, second_contour)
        np.testing.assert_array_equal(first_mask, second_mask)
        np.testing.assert_array_equal(first_separator, second_separator)

    def test_separator_does_not_cut_the_filled_scope(self) -> None:
        landmarks, _ = _front_template()
        shape = (1024, 1024)
        envelope = np.full(shape, 255, dtype=np.uint8)
        contour, mask, separator = build_unified_visia_scope(
            landmarks,
            shape,
            envelope,
            partial_face=False,
            left_visible_area=90_000,
            right_visible_area=90_000,
        )
        self.assertIsNotNone(contour)
        self.assertIsNotNone(separator)
        assert separator is not None
        points = separator.reshape(-1, 2)
        self.assertTrue(np.all(mask[points[:, 1], points[:, 0]] == 255))

    def test_partial_face_uses_closed_side_scope_without_front_separator(
        self,
    ) -> None:
        landmarks, _ = _front_template()
        shape = (1024, 1024)
        envelope = np.full(shape, 255, dtype=np.uint8)

        contour, mask, separator = build_unified_visia_scope(
            landmarks,
            shape,
            envelope,
            partial_face=True,
            left_visible_area=120_000,
            right_visible_area=20_000,
        )

        self.assertIsNotNone(contour)
        assert contour is not None
        self.assertTrue(np.array_equal(contour[0], contour[-1]))
        self.assertGreater(len(contour), 40)
        self.assertGreater(np.count_nonzero(mask), 1_000)
        self.assertIsNone(separator)


if __name__ == "__main__":
    unittest.main()
