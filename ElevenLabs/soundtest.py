from pydub import AudioSegment
from pydub.playback import play

audio = AudioSegment.from_file("speech.wav", format="wav")
left_channel = audio.split_to_mono()[0]
play(left_channel)
