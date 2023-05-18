import sys
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from pysofaconventions import SOFAFile

def apply_hrtf(audio, hrtf, direction):
    # Find the closest direction in the HRTF dataset
    closest_direction = np.argmin(np.abs(hrtf.getVariableValue('SourcePosition')[:, 0] - direction))

    # Get the HRTF filters for the closest direction
    left_hrtf = hrtf.Data_IR[closest_direction, 0]
    right_hrtf = hrtf.Data_IR[closest_direction, 1]

    # Apply the HRTF filters to the audio
    left_channel = np.convolve(audio, left_hrtf)
    right_channel = np.convolve(audio, right_hrtf)

    # Combine the left and right channels
    spatial_audio = np.vstack((left_channel, right_channel)).T

    return spatial_audio

def play_spatial_audio(audio_file, hrtf_file, direction):
    # Load the audio file
    audio, sample_rate = sf.read(audio_file)

    # Load the HRTF data
    hrtf = SOFAFile(hrtf_file, 'r')

    # Apply HRTF to the audio
    spatial_audio = apply_hrtf(audio, hrtf, direction)

    # Convert the spatial audio to an AudioSegment
    spatial_audio_segment = AudioSegment(
        spatial_audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=spatial_audio.dtype.itemsize,
        channels=2
    )

    # Play the spatial audio
    play(spatial_audio_segment)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python spatial_audio.py <audio_file> <hrtf_file> <direction>")
        sys.exit(1)

    audio_file = sys.argv[1]
    hrtf_file = sys.argv[2]
    direction = float(sys.argv[3])

    play_spatial_audio(audio_file, hrtf_file, direction)