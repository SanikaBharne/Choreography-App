import source_separation
import beat_detection
import audio_utils


def main():
    #loading audio
    file_path = audio_utils.get_file_path()
    audio, sr = audio_utils.load_audio(file_path)

    #audio separation
    stems = source_separation.separate(file_path)
    source_separation.get_instrumental(stems)

    #beat detection
    drum_audio, drum_sr = audio_utils.load_audio(stems["drums"])
    drum_beats = beat_detection.detect_beats(drum_audio, drum_sr)
    mix_beats = beat_detection.detect_beats(audio, sr)
    merged_beats = beat_detection.merge_beats(drum_beats, mix_beats)
    
    '''print(f"drum beats: {drum_beats}")
    for i in range(3):
        print("\n")
    print(f"mix beats: {mix_beats}")
    for i in range(3):
        print("\n")
    print(f"Merged beat times: {merged_beats}")
    for i in range(3):
        print("\n")'''
    
    grouped_counts = beat_detection.eight_count_grouping(merged_beats, subdivisions=2)
    for i, eight_count in enumerate(grouped_counts):
        print(f"Eight count {i+1}: {eight_count}")
        print("\n")


if __name__ == "__main__":
    main()