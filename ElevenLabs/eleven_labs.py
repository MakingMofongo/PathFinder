"""ElevenLabs speech module"""
import os

import requests
from playsound import playsound


PLACEHOLDERS = {"your-voice-id"}


class ElevenLabsSpeech():
    """ElevenLabs speech class"""

    def setup(self) -> None:
        """Set up the voices, API key, etc.

        Returns:
            None: None
        """

        default_voices = ["21m00Tcm4TlvDq8ikWAM", "MF3mGyEYCl7XYWbV9V6O"]
        voice_options = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Domi": "AZnzlk1XvdvUeBnXmlld",
            "Bella": "EXAVITQu4vr4xnSDxMaL",
            "Antoni": "ErXwobaYiN019PkySvjV",
            "Elli": "MF3mGyEYCl7XYWbV9V6O",
            "Josh": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold": "VR6AewLTigWG4xSOukaG",
            "Adam": "pNInz6obpgDQGcFmaJgB",
            "Sam": "yoZ06aMxZJJ28mfd3POQ",
            "Xina": "bruh"
        }
        self._headers = {
            "Content-Type": "application/json",
            "xi-api-key": '8fd6e8b3f1e2b892fae848f5aee22ebd',
        }
        self._voices = default_voices.copy()
        # self.use_custom_voice(elevenlabs_voice_1_id, 0)
        # self.use_custom_voice(elevenlabs_voice_2_id, 1)

    def use_custom_voice(self, voice, voice_index) -> None:
        """Use a custom voice if provided and not a placeholder

        Args:
            voice (str): The voice ID
            voice_index (int): The voice index

        Returns:
            None: None
        """
        # Placeholder values that should be treated as empty
        if voice and voice not in PLACEHOLDERS:
            self._voices[voice_index] = voice

    def speech(self, text: str, voice_index: int = 0, only_return: bool = False):
        """Speak text using elevenlabs.io's API

        Args:
            text (str): The text to speak
            voice_index (int, optional): The voice to use. Defaults to 0.

        Returns:
            bool: True if the request was successful, False otherwise
        """
        tts_url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voices[voice_index]}/stream"
        )
        response = requests.post(tts_url, headers=self._headers, json={"text": text,
                                                                       "voice_settings":{
                                                                            "stability": .6,
                                                                            "similarity_boost": .8, 
                                                                        }
                                                                       })

        if response.status_code == 200:
            with open("speech.mpeg", "wb") as f:
                f.write(response.content)
                if only_return:
                    return True    
            playsound("speech.mpeg", True)
            os.remove("speech.mpeg")
            return True
        else:
            print("Request failed with status code:", response.status_code)
            print("Response content:", response.content)
            return False
        
def play_audio(input_text):
    tts = ElevenLabsSpeech()
    tts.setup()
    saved = tts.speech(input_text, 0)
    return saved

        
if __name__ == "__main__":
    audio = play_audio("Stop")
    print(audio)


    # tts = ElevenLabsSpeech()
    # tts.setup()
    # tts.speech("30 SECONDS LEFT....... 10 SECONDS LEFT! SPIKE PLANTED!  ACE!!", 0)
