import unittest
import numpy as np
import librosa
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from beat_detection import detect_beats, merge_beats, eight_count_grouping

# Load test audio once for all tests
_audio, _sr = librosa.load("test_files/TBH.mp3")


class TestDetectBeats(unittest.TestCase):
    def test_returns_sorted_array(self):
        beats = detect_beats(_audio, _sr)
        self.assertIsInstance(beats, np.ndarray)
        self.assertTrue(np.all(beats[1:] >= beats[:-1]))

    def test_returns_positive_timestamps(self):
        beats = detect_beats(_audio, _sr)
        self.assertTrue(np.all(beats >= 0))

    def test_returns_nonempty_on_real_audio(self):
        beats = detect_beats(_audio, _sr)
        self.assertGreater(len(beats), 0)

    def test_non_numeric_audio_raises_type_error(self):
        with self.assertRaises(TypeError):
            detect_beats(["not", "numeric"], 22050)

    def test_empty_audio_raises_value_error(self):
        with self.assertRaises(ValueError):
            detect_beats(np.array([]), 22050)

    def test_invalid_sr_raises_type_error(self):
        with self.assertRaises(TypeError):
            detect_beats(np.array([0.1, 0.2]), "bad")

    def test_non_positive_sr_raises_value_error(self):
        with self.assertRaises(ValueError):
            detect_beats(np.array([0.1, 0.2]), -1)


class TestMergeBeats(unittest.TestCase):
    def test_fills_gaps(self):
        drum = np.array([1.0, 2.0, 5.0, 6.0])
        mix = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        merged = merge_beats(drum, mix, gap_threshold=1.0)
        self.assertIn(3.0, merged)
        self.assertIn(4.0, merged)

    def test_empty_drum_returns_mix(self):
        drum = np.array([])
        mix = np.array([1.0, 2.0, 3.0])
        merged = merge_beats(drum, mix)
        np.testing.assert_array_equal(merged, mix)

    def test_no_duplicates_when_close(self):
        drum = np.array([1.0, 2.0, 3.0])
        mix = np.array([1.05, 2.1, 3.0])
        merged = merge_beats(drum, mix, gap_threshold=0.5)
        self.assertEqual(len(merged), 3)

    def test_result_is_sorted(self):
        drum = np.array([1.0, 5.0])
        mix = np.array([2.0, 3.0, 4.0])
        merged = merge_beats(drum, mix, gap_threshold=1.0)
        self.assertTrue(np.all(merged[1:] >= merged[:-1]))

    def test_identical_inputs(self):
        beats = np.array([1.0, 2.0, 3.0])
        merged = merge_beats(beats, beats, gap_threshold=0.5)
        self.assertEqual(len(merged), 3)

    def test_gap_threshold_respected(self):
        drum = np.array([1.0, 10.0])
        mix = np.array([5.0])
        merged_tight = merge_beats(drum, mix, gap_threshold=0.5)
        self.assertIn(5.0, merged_tight)
        merged_wide = merge_beats(drum, mix, gap_threshold=6.0)
        self.assertNotIn(5.0, merged_wide)


class TestEightCountGrouping(unittest.TestCase):
    def test_groups_into_eight_counts(self):
        beats = np.arange(1.0, 17.0)  # 16 beats
        groups = eight_count_grouping(beats)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 8)
        self.assertEqual(len(groups[1]), 8)

    def test_partial_last_group(self):
        beats = np.arange(1.0, 11.0)  # 10 beats
        groups = eight_count_grouping(beats)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 8)
        self.assertEqual(len(groups[1]), 2)

    def test_no_subdivision_marks_all_as_beats(self):
        beats = np.array([1.0, 2.0, 3.0])
        groups = eight_count_grouping(beats)
        all_tuples = groups[0]
        for timestamp, is_beat in all_tuples:
            self.assertTrue(is_beat)

    def test_subdivision_creates_intermediate_points(self):
        beats = np.array([0.0, 1.0, 2.0])
        groups = eight_count_grouping(beats, subdivisions=2)
        # With 3 beats and subdivisions=2: each interval gets 1 extra point
        # beat0, sub, beat1, sub, beat2 = 5 entries
        all_tuples = [t for g in groups for t in g]
        self.assertEqual(len(all_tuples), 5)
        # Check intermediate point between beat 0 and 1
        self.assertAlmostEqual(all_tuples[1][0], 0.5)
        self.assertFalse(all_tuples[1][1])

    def test_subdivision_groups_respect_eight_count(self):
        beats = np.arange(0.0, 9.0)  # 9 beats
        groups = eight_count_grouping(beats, subdivisions=2)
        # Each beat interval produces 2 entries, so 8 intervals + final beat = 17 entries
        # First group should have 16 entries (8 * subdivisions)
        self.assertEqual(len(groups[0]), 16)

    def test_single_beat(self):
        beats = np.array([5.0])
        groups = eight_count_grouping(beats)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], [(5.0, True)])

    def test_non_numeric_raises_type_error(self):
        with self.assertRaises(TypeError):
            eight_count_grouping(["not", "numeric"])

    def test_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            eight_count_grouping(np.array([]))

    def test_non_integer_subdivisions_raises_type_error(self):
        with self.assertRaises(TypeError):
            eight_count_grouping(np.array([1.0, 2.0]), subdivisions=1.5)

    def test_non_positive_subdivisions_raises_value_error(self):
        with self.assertRaises(ValueError):
            eight_count_grouping(np.array([1.0, 2.0]), subdivisions=0)


if __name__ == "__main__":
    unittest.main()