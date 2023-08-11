import io
import os
import sys
import time
import warnings
import wave
from pathlib import Path
from typing import Callable, Dict, Generator, Iterable, List, Optional, TextIO, Union

from grpc._channel import _MultiThreadedRendezvous

import riva.client
import riva.client.proto.riva_asr_pb2 as rasr
import riva.client.proto.riva_asr_pb2_grpc as rasr_srv
from riva.client.auth import Auth

import logging
import inspect

from config import riva_asr_config, num_chars_printed, num_chars

# Configure logging with a custom log formatter
log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s'
logging.basicConfig(filename='app_info.log', level=logging.INFO, format=log_format)

# Helper function to log the print statement
def log_print(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    logging.debug(message)

# Replace the print statements with log_print
print = log_print


class ASRService:
    """Provides streaming and offline recognition services. Calls gRPC stubs with authentication metadata."""
    def __init__(self, auth: Auth) -> None:
        """
        Initializes an instance of the class.

        Args:
            auth (:obj:`riva.client.auth.Auth`): an instance of :class:`riva.client.auth.Auth` which is used for
                authentication metadata generation.
        """
        self.auth = auth
        self.stub = rasr_srv.RivaSpeechRecognitionStub(self.auth.channel)

    def streaming_response_generator(
        self, audio_chunks: Iterable[bytes], streaming_config: rasr.StreamingRecognitionConfig
    ) -> Generator[rasr.StreamingRecognizeResponse, None, None]:
        """
        Generates speech recognition responses for fragments of speech audio in :param:`audio_chunks`.
        The purpose of the method is to perform speech recognition "online" - as soon as
        audio is acquired on small chunks of audio.

        All available audio chunks will be sent to a server on the first ``next()`` call.

        Args:
            audio_chunks (:obj:`Iterable[bytes]`): an iterable object that contains raw audio fragments
                of speech. For example, such raw audio can be obtained with

                .. code-block:: python

                    import wave
                    with wave.open(file_name, 'rb') as wav_f:
                        raw_audio = wav_f.readframes(n_frames)

            streaming_config (:obj:`riva.client.proto.riva_asr_pb2.StreamingRecognitionConfig`): a config for streaming.
                You may find a description of config fields in message ``StreamingRecognitionConfig`` in
                `the common repo
                <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/reference/protos/protos.html#riva-proto-riva-asr-proto>`_.
                An example of creating a streaming config:

                .. code-style:: python

                    from riva.client import RecognitionConfig, StreamingRecognitionConfig
                    config = RecognitionConfig(enable_automatic_punctuation=True)
                    streaming_config = StreamingRecognitionConfig(config, interim_results=True)

        Yields:
            :obj:`riva.client.proto.riva_asr_pb2.StreamingRecognizeResponse`: responses for audio chunks in
            :param:`audio_chunks`. You may find a description of response fields in the declaration of
            ``StreamingRecognizeResponse``
            message `here
            <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/reference/protos/protos.html#riva-proto-riva-asr-proto>`_.
        """
        generator = streaming_request_generator(audio_chunks, streaming_config)
        for response in self.stub.StreamingRecognize(generator, metadata=self.auth.get_auth_metadata()):
            yield response

    def offline_recognize(
        self, audio_bytes: bytes, config: rasr.RecognitionConfig, future: bool = False
    ) -> Union[rasr.RecognizeResponse, _MultiThreadedRendezvous]:
        """
        Performs speech recognition for raw audio in :param:`audio_bytes`. This method is for processing of
        huge audio at once - not as it is being generated.

        Args:
            audio_bytes (:obj:`bytes`): a raw audio. For example, it can be obtained with

                .. code-block:: python

                    import wave
                    with wave.open(file_name, 'rb') as wav_f:
                        raw_audio = wav_f.readframes(n_frames)

            config (:obj:`riva.client.proto.riva_asr_pb2.RecognitionConfig`): a config for offline speech recognition.
                You may find a description of config fields in message ``RecognitionConfig`` in
                `the common repo
                <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/reference/protos/protos.html#riva-proto-riva-asr-proto>`_.
                An example of creating a config:

                .. code-style:: python

                    from riva.client import RecognitionConfig
                    config = RecognitionConfig(enable_automatic_punctuation=True)
            future (:obj:`bool`, defaults to :obj:`False`): whether to return an async result instead of the usual
                response. You can get a response by calling the ``result()`` method of the future object.

        Returns:
            :obj:`Union[riva.client.proto.riva_asr_pb2.RecognizeResponse, grpc._channel._MultiThreadedRendezvous]``: a
            response with results of :param:`audio_bytes` processing. You may find a description of response fields in
            the declaration of ``RecognizeResponse`` message `here
            <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/reference/protos/protos.html#riva-proto-riva-asr-proto>`_.
            If :param:`future` is :obj:`True`, then a future object is returned. You may retrieve a response from a
            future object by calling the ``result()`` method.
        """
        request = rasr.RecognizeRequest(config=config, audio=audio_bytes)
        func = self.stub.Recognize.future if future else self.stub.Recognize
        return func(request, metadata=self.auth.get_auth_metadata())