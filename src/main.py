import hpss
import audio_utils
from scipy.io.wavfile import write

def main():
    file_path = input("Insert your file path: ")
    sample_rate, data = audio_utils.load_audio(file_path)
    fourier_transform = hpss.stft(data)
    horizontal, vertical = hpss.compute_masks(fourier_transform)
    harmonic_mask, percussive_mask = hpss.build_masks(horizontal, vertical)
    harmonic_spectrogram, percussive_spectrogram = hpss.apply_masks(harmonic_mask, percussive_mask, fourier_transform)
    h_audio = hpss.istft(harmonic_spectrogram)
    p_audio = hpss.istft(percussive_spectrogram)

    write("harmonic.wav", sample_rate, h_audio)
    write("percussive.wav", sample_rate, p_audio)

if __name__ == "__main__":
    main()
