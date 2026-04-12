import librosa


def get_file_path():
    """Prompt the user for a file path and return it.

    Returns:
        The file path string entered by the user.

    Raises:
        ValueError: If the user enters an empty string.
    """
    file_path = input("Insert your file path: ")
    if not file_path:
        raise ValueError("file path must not be empty")
    return file_path

def load_audio(filepath):
    """Load an audio file and return the audio time series and sample rate.

    Args:
        filepath: Path to the audio file to load.

    Returns:
        A tuple of (audio, sr) where audio is a NumPy array of samples
        and sr is the sample rate.

    Raises:
        TypeError: If filepath is not a string.
        ValueError: If filepath is empty.
        FileNotFoundError: If filepath does not exist.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")
    if not filepath:
        raise ValueError("filepath must not be empty")

    return librosa.load(filepath)