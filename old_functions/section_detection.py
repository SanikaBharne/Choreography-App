import librosa
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def extract_features(audio, sr, merged_beats):
    """Extract beat-synchronous chroma features from audio.

    Args:
        audio: NumPy array of audio samples.
        sr: Sample rate in Hz.
        merged_beats: NumPy array of beat timestamps in seconds.

    Returns:
        A 2-D NumPy array of beat-synchronous chroma features.

    Raises:
        TypeError: If audio or merged_beats are not array-like or contain
            non-numeric values, or if sr is not numeric.
        ValueError: If sr is not positive or merged_beats is empty.
    """
    audio = np.asarray(audio)
    merged_beats = np.asarray(merged_beats)

    if not np.issubdtype(audio.dtype, np.number):
        raise TypeError("audio must contain numeric values")
    if not isinstance(sr, (int, float)):
        raise TypeError("sr must be numeric")
    if sr <= 0:
        raise ValueError("sr must be positive")
    if not np.issubdtype(merged_beats.dtype, np.number):
        raise TypeError("merged_beats must contain numeric values")
    if len(merged_beats) == 0:
        raise ValueError("merged_beats must not be empty")

    chromas = librosa.feature.chroma_cqt(y=audio, sr=sr)
    normalized_chromas = librosa.util.normalize(chromas, axis=0)
    return librosa.util.sync(normalized_chromas, librosa.time_to_frames(merged_beats, sr=sr))

def build_ssm(features):
    """Build a self-similarity matrix from chroma features.

    Args:
        features: 2-D NumPy array of chroma features (rows are features,
            columns are time frames).

    Returns:
        A 2-D NumPy self-similarity matrix.

    Raises:
        TypeError: If features is not array-like or contains non-numeric values.
        ValueError: If features is empty.
    """
    features = np.asarray(features)

    if not np.issubdtype(features.dtype, np.number):
        raise TypeError("features must contain numeric values")
    if features.size == 0:
        raise ValueError("features must not be empty")

    return features.T @ features

def detect_boundaries(ssm, merged_beats):
    """Detect section boundaries using a checkerboard kernel on the SSM.

    Args:
        ssm: 2-D square NumPy array, the self-similarity matrix.
        merged_beats: NumPy array of beat timestamps in seconds.

    Returns:
        A list of boundary timestamps in seconds.

    Raises:
        TypeError: If ssm or merged_beats are not array-like or contain
            non-numeric values.
        ValueError: If ssm is not 2-D, not square, or has fewer than 65
            rows and columns.
    """
    ssm = np.asarray(ssm)
    merged_beats = np.asarray(merged_beats)

    if not np.issubdtype(ssm.dtype, np.number):
        raise TypeError("ssm must contain numeric values")
    if not np.issubdtype(merged_beats.dtype, np.number):
        raise TypeError("merged_beats must contain numeric values")
    if ssm.ndim != 2:
        raise ValueError("ssm must be 2-D")
    if ssm.shape[0] != ssm.shape[1]:
        raise ValueError("ssm must be square")
    if ssm.shape[0] < 65:
        raise ValueError("ssm must have at least 65 rows and columns")

    first_quad = np.ones((32, 32))
    second_quad = np.ones((32, 32)) * -1
    third_quad = np.ones((32, 32)) * -1
    fourth_quad = np.ones((32, 32))

    checkerboard = np.block([[first_quad, second_quad],
                            [third_quad, fourth_quad]])

    novelty = []

    for i in range(32, len(ssm) - 32):
        chunk = ssm[i - 32: i+32, i-32: i +32]
        value = np.sum(chunk * checkerboard)
        novelty.append(value)
        
    times = merged_beats[32:32 + len(novelty)]
    plt.plot(times, novelty)
    plt.xlabel("Time (s)")
    plt.show()

    peaks, _ = find_peaks(novelty, distance=60)
    boundary_times = []

    for peak in peaks:
        boundary_times.append(merged_beats[peak + 32])

    return boundary_times

def label_sections(features, boundaries, merged_beats):
    """Assign letter labels to sections based on chroma similarity.

    Args:
        features: 2-D NumPy array of beat-synchronous chroma features.
        boundaries: List or array of boundary timestamps in seconds.
        beat_times: NumPy array of beat timestamps in seconds.

    Returns:
        A list of section label strings (e.g. ['A', 'B', 'A']).

    Raises:
        TypeError: If features is not array-like or contains non-numeric
            values, or if boundaries or beat_times are not array-like.
        ValueError: If boundaries has fewer than 2 elements.
    """
    features = np.asarray(features)
    boundaries = np.asarray(boundaries)
    merged_beats = np.asarray(merged_beats)

    if not np.issubdtype(features.dtype, np.number):
        raise TypeError("features must contain numeric values")
    if not np.issubdtype(boundaries.dtype, np.number):
        raise TypeError("boundaries must contain numeric values")
    if not np.issubdtype(merged_beats.dtype, np.number):
        raise TypeError("beat_times must contain numeric values")
    if len(boundaries) < 2:
        raise ValueError("boundaries must have at least 2 elements")

    beat_indices = []
    for boundary in boundaries:
        beat_indices.append(np.searchsorted(merged_beats, boundary))

    segment_representatives = []
    for i in range(len(beat_indices) - 1):
        segment_representatives.append(np.mean(features[:, beat_indices[i]:beat_indices[i+1]], axis=1))

    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    sections = []
    for segment in segment_representatives:
        sections.append(None)

    counter = 0
    for i in range(len(sections)):
        if sections[i] == None:
            sections[i] = labels[counter]
            counter += 1
            for j, segment in enumerate(segment_representatives[i:]):
                cosine_norm = np.linalg.norm(segment_representatives[i]) * np.linalg.norm(segment)
                if np.dot(segment_representatives[i], segment)/cosine_norm > 0.9 and not sections[j + i]:
                    sections[j + i] = sections[i]

    return sections
