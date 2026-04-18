import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import audio_utils as audio_utils


class TestLoadAudio(unittest.TestCase):
    def test_rejects_non_string_filepath(self):
        with self.assertRaises(TypeError):
            audio_utils.load_audio(123)

    def test_rejects_empty_filepath(self):
        with self.assertRaises(ValueError):
            audio_utils.load_audio("")

    def test_rejects_none_filepath(self):
        with self.assertRaises(TypeError):
            audio_utils.load_audio(None)

    def test_nonexistent_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            audio_utils.load_audio("nonexistent_file.wav")

    def test_returns_tuple_of_array_and_sr(self):
        audio, sr = audio_utils.load_audio("test_files/TBH.mp3")
        self.assertIsInstance(audio, np.ndarray)
        self.assertIsInstance(sr, (int, float))

    def test_returns_nonempty_audio(self):
        audio, sr = audio_utils.load_audio("test_files/TBH.mp3")
        self.assertGreater(len(audio), 0)

    def test_returns_positive_sample_rate(self):
        audio, sr = audio_utils.load_audio("test_files/TBH.mp3")
        self.assertGreater(sr, 0)

    def test_audio_is_numeric(self):
        audio, sr = audio_utils.load_audio("test_files/TBH.mp3")
        self.assertTrue(np.issubdtype(audio.dtype, np.floating))


class TestGetFilePath(unittest.TestCase):
    @patch("builtins.input", return_value="test_files/TBH.mp3")
    def test_returns_user_input(self, mock_input):
        result = audio_utils.get_file_path()
        self.assertEqual(result, "test_files/TBH.mp3")
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="")
    def test_empty_input_raises_value_error(self, mock_input):
        with self.assertRaises(ValueError):
            audio_utils.get_file_path()

    @patch("builtins.input", return_value="/some/path/song.wav")
    def test_returns_string(self, mock_input):
        result = audio_utils.get_file_path()
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
