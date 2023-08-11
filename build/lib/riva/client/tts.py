from typing import Generator, Optional, Union

import riva.client.proto.riva_tts_pb2 as rtts
import riva.client.proto.riva_tts_pb2_grpc as rtts_srv
from riva.client import Auth
from riva.client.proto.riva_audio_pb2 import AudioEncoding

from tts_config import riva_tts_config, sample_rate


class SpeechSynthesisService:
    """
    A class for synthesizing speech from text. Provides `synthesize` which returns entire audio for a text
    and `synthesize_online` which returns audio in small chunks as it is becoming available.
    """

    def __init__(self, auth: Auth) -> None:
        """
        Initializes an instance of the class.

        Args:
            auth (Auth): An instance of `riva.client.auth.Auth` used for authentication metadata generation.
        """
        self.auth = auth
        self.stub = rtts_srv.RivaSpeechSynthesisStub(self.auth.channel)

    def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        language_code: str = 'en-US',
        encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
        sample_rate_hz: int = config.sample_rate,
        future: bool = False,
    ) -> Union[rtts.SynthesizeSpeechResponse, _MultiThreadedRendezvous]:
        """
        Synthesizes an entire audio for the input text.

        Args:
            text (str): The input text to be synthesized.
            voice_name (str, optional): The name of the voice, e.g., "English-US-Female-1". If None, the server
                will select the first available model with the correct `language_code`.
            language_code (str): The language to use.
            encoding (AudioEncoding): The output audio encoding, e.g., AudioEncoding.LINEAR_PCM.
            sample_rate_hz (int): The number of frames per second in the output audio.
            future (bool, optional): Whether to return an async result instead of the usual response. You can get
                a response by calling the `result()` method of the future object.

        Returns:
            Union[rtts.SynthesizeSpeechResponse, _MultiThreadedRendezvous]: A response with the output audio.
        """
        request = rtts.SynthesizeSpeechRequest(
            text=text,
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
            encoding=encoding,
        )
        if voice_name is not None:
            request.voice_name = voice_name
        func = self.stub.Synthesize.future if future else self.stub.Synthesize
        return func(request, metadata=self.auth.get_auth_metadata())

    def synthesize_online(
        self,
        text: str,
        voice_name: Optional[str] = None,
        language_code: str = 'en-US',
        encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
        sample_rate_hz: int = config.sample_rate,
    ) -> Generator[rtts.SynthesizeSpeechResponse, None, None]:
        """
        Synthesizes and yields output audio chunks for the input text as the chunks become available.

        Args:
            text (str): The input text to be synthesized.
            voice_name (str, optional): The name of the voice, e.g., "English-US-Female-1". If None, the server
                will select the first available model with the correct `language_code`.
            language_code (str): The language to use.
            encoding (AudioEncoding): The output audio encoding, e.g., AudioEncoding.LINEAR_PCM.
            sample_rate_hz (int): The number of frames per second in the output audio.

        Yields:
            Generator[rtts.SynthesizeSpeechResponse, None, None]: A generator yielding response with output audio chunks.
        """
        request = rtts.SynthesizeSpeechRequest(
            text=text,
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
            encoding=encoding,
        )
        if voice_name is not None:
            request.voice_name = voice_name
        return self.stub.SynthesizeOnline(request, metadata=self.auth.get_auth_metadata())
