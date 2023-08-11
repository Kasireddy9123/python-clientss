import io
import os
import sys
import time
import warnings
import wave
from contextlib import closing
from pathlib import Path
from typing import Callable, Dict, Generator, Iterable, List, Optional, TextIO, Union

from grpc._channel import _MultiThreadedRendezvous

import riva.client
import riva.client.proto.riva_asr_pb2 as rasr
import riva.client.proto.riva_asr_pb2_grpc as rasr_srv
from riva.client.auth import Auth

from config import num_chars, num_chars_printed

import logging

from asr_config import riva_asr_config, num_chars_printed, num_chars

from asrService import ASRService

# Configure logging with a custom log formatter
log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s'
logging.basicConfig(filename='app.log', level=logging.DEBUG, format=log_format)

# Helper function to log the print statement
def log_print(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    logging.debug(message)

# Replace the print statements with log_print
print = log_print


def get_wav_file_parameters(input_file: Union[str, os.PathLike]) -> Optional[Dict[str, Union[int, float]]]:
    try:
        input_file = Path(input_file).expanduser()
        with wave.open(str(input_file), 'rb') as wf:
            nframes = wf.getnframes()
            rate = wf.getframerate()
            parameters = {
                'nframes': nframes,
                'framerate': rate,
                'duration': nframes / rate,
                'nchannels': wf.getnchannels(),
                'sampwidth': wf.getsampwidth(),
                'data_offset': wf.getfp().size_read + wf.getfp().offset
            }
    except wave.Error:
        # Not a WAV file
        return None
    return parameters


def sleep_audio_length(audio_chunk: bytes, time_to_sleep: float) -> None:
    time.sleep(time_to_sleep)


class AudioChunkFileIterator:
    def __init__(
        self,
        input_file: Union[str, os.PathLike],
        chunk_n_frames: int,
        delay_callback: Optional[Callable[[bytes, float], None]] = None,
    ) -> None:
        self.input_file: Path = Path(input_file).expanduser()
        self.chunk_n_frames = chunk_n_frames
        self.delay_callback = delay_callback
        self.file_parameters = get_wav_file_parameters(self.input_file)
        self.file_object: Optional[io.BufferedIOBase] = open(str(self.input_file), 'rb')
        if self.delay_callback and self.file_parameters is None:
            warnings.warn(f"delay_callback not supported for encoding other than LINEAR_PCM")
            self.delay_callback = None
        self.first_buffer = True

    def close(self) -> None:
        self.file_object.close()
        self.file_object = None

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback) -> None:
        if self.file_object is not None:
            self.file_object.close()

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self.file_parameters:
            data = self.file_object.read(self.chunk_n_frames * self.file_parameters['sampwidth'] * self.file_parameters['nchannels'])
        else:
            data = self.file_object.read(self.chunk_n_frames)
        if not data:
            self.close()
            raise StopIteration
        if self.delay_callback is not None:
            offset = self.file_parameters['data_offset'] if self.first_buffer else 0
            self.delay_callback(
                data[offset:], (len(data) - offset) / self.file_parameters['sampwidth'] / self.file_parameters['framerate']
            )
            self.first_buffer = False
        return data


def add_word_boosting_to_config(
    config: Union[rasr.StreamingRecognitionConfig, rasr.RecognitionConfig],
    boosted_lm_words: Optional[List[str]],
    boosted_lm_score: float,
) -> None:
    inner_config: rasr.RecognitionConfig = config if isinstance(config, rasr.RecognitionConfig) else config.config
    if boosted_lm_words is not None:
        speech_context = rasr.SpeechContext()
        speech_context.phrases.extend(boosted_lm_words)
        speech_context.boost = boosted_lm_score
        inner_config.speech_contexts.append(speech_context)


def add_audio_file_specs_to_config(
    config: Union[rasr.StreamingRecognitionConfig, rasr.RecognitionConfig],
    audio_file: Union[str, os.PathLike],
) -> None:
    inner_config: rasr.RecognitionConfig = config if isinstance(config, rasr.RecognitionConfig) else config.config
    wav_parameters = get_wav_file_parameters(audio_file)
    if wav_parameters is not None:
        inner_config.sample_rate_hertz = wav_parameters['framerate']
        inner_config.audio_channel_count = wav_parameters['nchannels']


def add_speaker_diarization_to_config(
    config: rasr.RecognitionConfig,
    diarization_enable: bool,
) -> None:
    if diarization_enable:
        diarization_config = rasr.SpeakerDiarizationConfig(enable_speaker_diarization=True)
        config.diarization_config.CopyFrom(diarization_config)


PRINT_STREAMING_ADDITIONAL_INFO_MODES = ['no', 'time', 'confidence']


def print_streaming(
    responses: Iterable[rasr.StreamingRecognizeResponse],
    output_file: Optional[Union[Union[os.PathLike, str, TextIO], List[Union[os.PathLike, str, TextIO]]]] = None,
    additional_info: str = 'no',
    word_time_offsets: bool = False,
    show_intermediate: bool = False,
    file_mode: str = 'w',
) -> None:
    """
    Prints streaming speech recognition results to provided files or streams.

    Args:
        responses (:obj:`Iterable[riva.client.proto.riva_asr_pb2.StreamingRecognizeResponse]`): responses acquired during
            streaming speech recognition.
        output_file (:obj:`Union[Union[os.PathLike, str, TextIO], List[Union[os.PathLike, str, TextIO]]]`, `optional`):
            a path to an output file or a text stream or a list of paths/streams. If contains several elements, then
            output will be written to all destinations. If :obj:`None`, then output will be written to STDOUT.
        additional_info (:obj:`str`, defaults to :obj:`"no"`): a string which can take one of three values:
            :obj:`"no"`, :obj:`"time"`, :obj:`"confidence"`.

            If :obj:`"no"`, then partial transcript is prefixed by ">>", and final transcript is prefixed with "##".
            An option :param:`show_intermediate` can be used.

            If :obj:`"time"`, then transcripts are prefixed by time when they were printed. An option
            :param:`word_time_offsets` can be used.

            If :obj:`"confidence"`, then transcript stability and confidence are printed. Finished and updating
            parts of a transcript are shown separately.
        word_time_offsets (:obj:`bool`, defaults to :obj:`False`): If :obj:`True`, then word time stamps are printed.
            Available only if ``additional_info="time"``.
        show_intermediate (:obj:`bool`, defaults to :obj:`False`): If :obj:`True`, then partial transcripts are
            printed. If printing is performed to a stream (e.g. :obj:`sys.stdout`), then partial transcript is updated
            on the same line of a console. Available only if ``additional_info="no"``.
        file_mode (:obj:`str`, defaults to :obj:`"w"`): a mode in which files are opened.

    Raises:
        :obj:`ValueError`: if the wrong :param:`additional_info` value is passed to this function.
    """
    if additional_info not in PRINT_STREAMING_ADDITIONAL_INFO_MODES:
        raise ValueError(
            f"Not allowed value '{additional_info}' of parameter `additional_info`. "
            f"Allowed values are {PRINT_STREAMING_ADDITIONAL_INFO_MODES}"
        )
    if additional_info != PRINT_STREAMING_ADDITIONAL_INFO_MODES[0] and show_intermediate:
        warnings.warn(
            f"`show_intermediate=True` will not work if "
            f"`additional_info != {PRINT_STREAMING_ADDITIONAL_INFO_MODES[0]}`. `additional_info={additional_info}`"
        )
    if additional_info != PRINT_STREAMING_ADDITIONAL_INFO_MODES[1] and word_time_offsets:
        warnings.warn(
            f"`word_time_offsets=True` will not work if "
            f"`additional_info != {PRINT_STREAMING_ADDITIONAL_INFO_MODES[1]}`. `additional_info={additional_info}"
        )
    if output_file is None:
        output_file = [sys.stdout]
    elif not isinstance(output_file, list):
        output_file = [output_file]
    file_opened = [False] * len(output_file)
    try:
        for i, elem in enumerate(output_file):
            if isinstance(elem, io.TextIOBase):
                file_opened[i] = False
            else:
                file_opened[i] = True
                output_file[i] = Path(elem).expanduser().open(file_mode)
        start_time = time.time()  # used in 'time` additional_info
        num_chars_printed = config.num_chars_printed  # used in 'no' additional_info
        for response in responses:
            if not response.results:
                continue
            partial_transcript = ""
            for result in response.results:
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript
                if additional_info == 'no':
                    if result.is_final:
                        if show_intermediate:
                            overwrite_chars = ' ' * (num_chars_printed - len(transcript))
                            for i, f in enumerate(output_file):
                                f.write("## " + transcript + (overwrite_chars if not file_opened[i] else '') + "\n")
                            num_chars_printed = config.num_chars_printed
                        else:
                            for i, alternative in enumerate(result.alternatives):
                                for f in output_file:
                                    f.write(
                                        f'##'
                                        + (f'(alternative {i + 1})' if i > 0 else '')
                                        + f' {alternative.transcript}\n'
                                    )
                    else:
                        partial_transcript += transcript
                elif additional_info == 'time':
                    if result.is_final:
                        for i, alternative in enumerate(result.alternatives):
                            for f in output_file:
                                f.write(
                                    f"Time {time.time() - start_time:.2f}s: Transcript {i}: {alternative.transcript}\n"
                                )
                        if word_time_offsets:
                            for f in output_file:
                                f.write("Timestamps:\n")
                                f.write('{: <40s}{: <16s}{: <16s}\n'.format('Word', 'Start (ms)', 'End (ms)'))
                                for word_info in result.alternatives[0].words:
                                    f.write(
                                        f'{word_info.word: <40s}{word_info.start_time: <16.0f}'
                                        f'{word_info.end_time: <16.0f}\n'
                                    )
                    else:
                        partial_transcript += transcript
                else:  # additional_info == 'confidence'
                    if result.is_final:
                        for f in output_file:
                            f.write(f'## {transcript}\n')
                            f.write(f'Confidence: {result.alternatives[0].confidence:9.4f}\n')
                    else:
                        for f in output_file:
                            f.write(f'>> {transcript}\n')
                            f.write(f'Stability: {result.stability:9.4f}\n')
            if additional_info == 'no':
                if show_intermediate and partial_transcript != '':
                    overwrite_chars = ' ' * (num_chars_printed - len(partial_transcript))
                    for i, f in enumerate(output_file):
                        f.write(">> " + partial_transcript + ('\n' if file_opened[i] else overwrite_chars + '\r'))
                    num_chars_printed = len(partial_transcript) + config.num_chars
            elif additional_info == 'time':
                for f in output_file:
                    if partial_transcript:
                        f.write(f">>>Time {time.time():.2f}s: {partial_transcript}\n")
            else:
                for f in output_file:
                    f.write('----\n')
    finally:
        for fo, elem in zip(file_opened, output_file):
            if fo:
                elem.close()


def print_offline(response: rasr.RecognizeResponse) -> None:
    print(response)
    if len(response.results) > 0 and len(response.results[0].alternatives) > 0:
        final_transcript = ""
        for res in response.results:
            final_transcript += res.alternatives[0].transcript
        print("Final transcript:", final_transcript)


def streaming_request_generator(
    audio_chunks: Iterable[bytes], streaming_config: rasr.StreamingRecognitionConfig
) -> Generator[rasr.StreamingRecognizeRequest, None, None]:
    yield rasr.StreamingRecognizeRequest(streaming_config=streaming_config)
    for chunk in audio_chunks:
        yield rasr.StreamingRecognizeRequest(audio_content=chunk)


# Create Instance of ASRService
obj = ASRService()

# Call methods of ASRService
obj.__init__()
obj.streaming_response_generator()
obj.offline_recognize()