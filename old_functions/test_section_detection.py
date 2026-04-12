import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from old_functions.section_detection import extract_features, build_ssm, detect_boundaries, label_sections


class TestExtractFeatures(unittest.TestCase):
    def test_output_has_12_chroma_rows(self):
        sr = 22050
        duration = 5
        audio = np.random.randn(sr * duration)
        beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        features = extract_features(audio, sr, beats)
        self.assertEqual(features.shape[0], 12)

    def test_output_columns_match_beats(self):
        sr = 22050
        duration = 5
        audio = np.random.randn(sr * duration)
        beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        features = extract_features(audio, sr, beats)
        self.assertGreater(features.shape[1], 0)

    def test_type_error_on_non_numeric_audio(self):
        with self.assertRaises(TypeError):
            extract_features(["a", "b", "c"], 22050, np.array([1.0, 2.0]))

    def test_type_error_on_non_numeric_sr(self):
        with self.assertRaises(TypeError):
            extract_features(np.random.randn(22050), "fast", np.array([1.0, 2.0]))

    def test_value_error_on_non_positive_sr(self):
        with self.assertRaises(ValueError):
            extract_features(np.random.randn(22050), -1, np.array([1.0, 2.0]))

    def test_type_error_on_non_numeric_beats(self):
        with self.assertRaises(TypeError):
            extract_features(np.random.randn(22050), 22050, ["a", "b"])

    def test_value_error_on_empty_beats(self):
        with self.assertRaises(ValueError):
            extract_features(np.random.randn(22050), 22050, np.array([]))


class TestBuildSSM(unittest.TestCase):
    def test_output_is_square(self):
        features = np.random.randn(12, 50)
        ssm = build_ssm(features)
        self.assertEqual(ssm.shape[0], ssm.shape[1])

    def test_output_shape_matches_columns(self):
        features = np.random.randn(12, 50)
        ssm = build_ssm(features)
        self.assertEqual(ssm.shape[0], 50)

    def test_diagonal_values_positive(self):
        features = np.random.randn(12, 50)
        ssm = build_ssm(features)
        self.assertTrue(np.all(np.diag(ssm) >= 0))

    def test_type_error_on_non_numeric(self):
        with self.assertRaises(TypeError):
            build_ssm([["a", "b"], ["c", "d"]])

    def test_value_error_on_empty(self):
        with self.assertRaises(ValueError):
            build_ssm(np.array([]))


class TestDetectBoundaries(unittest.TestCase):
    def test_returns_list(self):
        ssm = np.random.randn(100, 100)
        beats = np.linspace(0, 50, 100)
        boundaries = detect_boundaries(ssm, beats)
        self.assertIsInstance(boundaries, list)

    def test_boundaries_within_beat_range(self):
        ssm = np.random.randn(100, 100)
        beats = np.linspace(0, 50, 100)
        boundaries = detect_boundaries(ssm, beats)
        for b in boundaries:
            self.assertGreaterEqual(b, beats[0])
            self.assertLessEqual(b, beats[-1])

    def test_type_error_on_non_numeric_ssm(self):
        with self.assertRaises(TypeError):
            detect_boundaries([["a", "b"], ["c", "d"]], np.array([1.0, 2.0]))

    def test_type_error_on_non_numeric_beats(self):
        with self.assertRaises(TypeError):
            detect_boundaries(np.random.randn(100, 100), ["a", "b"])

    def test_value_error_on_1d_ssm(self):
        with self.assertRaises(ValueError):
            detect_boundaries(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_value_error_on_non_square_ssm(self):
        with self.assertRaises(ValueError):
            detect_boundaries(np.random.randn(100, 50), np.array([1.0, 2.0]))

    def test_value_error_on_too_small_ssm(self):
        with self.assertRaises(ValueError):
            detect_boundaries(np.random.randn(30, 30), np.linspace(0, 10, 30))


class TestLabelSections(unittest.TestCase):
    def test_returns_list_of_strings(self):
        features = np.random.randn(12, 100)
        beat_times = np.linspace(0, 50, 100)
        boundaries = [0.0, 10.0, 25.0, 50.0]
        sections = label_sections(features, boundaries, beat_times)
        self.assertIsInstance(sections, list)
        for label in sections:
            self.assertIsInstance(label, str)

    def test_first_label_is_a(self):
        features = np.random.randn(12, 100)
        beat_times = np.linspace(0, 50, 100)
        boundaries = [0.0, 10.0, 25.0, 50.0]
        sections = label_sections(features, boundaries, beat_times)
        self.assertEqual(sections[0], 'A')

    def test_number_of_labels_matches_segments(self):
        features = np.random.randn(12, 100)
        beat_times = np.linspace(0, 50, 100)
        boundaries = [0.0, 10.0, 25.0, 50.0]
        sections = label_sections(features, boundaries, beat_times)
        self.assertEqual(len(sections), len(boundaries) - 1)

    def test_similar_sections_get_same_label(self):
        features = np.zeros((12, 100))
        features[:, :50] = 1.0
        features[:, 50:] = 1.0
        beat_times = np.linspace(0, 50, 100)
        boundaries = [0.0, 25.0, 50.0]
        sections = label_sections(features, boundaries, beat_times)
        self.assertEqual(sections[0], sections[1])

    def test_type_error_on_non_numeric_features(self):
        with self.assertRaises(TypeError):
            label_sections([["a", "b"]], [0.0, 1.0], np.array([0.0, 1.0]))

    def test_type_error_on_non_numeric_boundaries(self):
        with self.assertRaises(TypeError):
            label_sections(np.random.randn(12, 100), ["a", "b"], np.array([0.0, 1.0]))

    def test_type_error_on_non_numeric_beat_times(self):
        with self.assertRaises(TypeError):
            label_sections(np.random.randn(12, 100), [0.0, 1.0], ["a", "b"])

    def test_value_error_on_fewer_than_two_boundaries(self):
        with self.assertRaises(ValueError):
            label_sections(np.random.randn(12, 100), [0.0], np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
