import argparse
import os
import queue
import time
from pathlib import Path
from threading import Thread
from typing import Union, NamedTuple, List, Optional, Generator
from enum import Enum

import riva.client
from riva.client.asr import get_wav_file_parameters
from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters

from config import NUM_CLIENTS, NUM_ITERATIONS, INPUT_FILE, SIMULATE_REALTIME, FILE_STREAMING_CHUNK, FILE_MODE, argument_default, time_sleep

import logging

# Configure logging with a custom log formatter
log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s'
logging.basicConfig(filename='app.log', level=logging.DEBUG, format=log_format)

# Helper function to log the print statement
def log_print(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    logging.debug(message)

# Replace the print statements with log_print
print = log_print

class AdditionalInfo(Enum):
    NO = "no"
    TIME = "time"
    CONFIDENCE = "confidence"


class StreamingTranscriptionWorkerArgs(NamedTuple):
    args: argparse.Namespace
    output_file: Path
    thread_i: int
    exception_queue: queue.Queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming transcription via Riva AI Services. Unlike `scripts/asr/transcribe_file.py` script, "
        "this script can perform transcription several times on the same audio if `--num-iterations` is "
        "greater than 1. If `--num-clients` is greater than 1, then a file will be transcribed independently "
        "in several threads. Unlike other ASR scripts, this script does not print output but saves it in files "
        "which names follow a format `output_<thread_num>.txt`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-clients", default=1, type=int, help="Number of client threads.")
    parser.add_argument("--num-iterations", default=1, type=int, help="Number of iterations over the file.")
    parser.add_argument(
        "--input-file", required=True, type=str, help="Name of the WAV file with LINEAR_PCM encoding to transcribe."
    )
    parser.add_argument(
        "--simulate-realtime",
        action='store_true',
        help="Option to simulate realtime transcription. Audio fragments are sent to a server at a pace that mimics "
        "normal speech.",
    )
    parser.add_argument(
        "--file-streaming-chunk", type=int, default=config.argument_default, help="Number of frames in one chunk sent to the server."
    )
    parser = add_connection_argparse_parameters(parser)
    parser = add_asr_config_argparse_parameters(parser, max_alternatives=True, profanity_filter=True, word_time_offsets=True)
    args = parser.parse_args()
    if args.max_alternatives < 1:
        parser.error("`--max-alternatives` must be greater than or equal to 1")
    return args


def streaming_transcription_worker(args: argparse.Namespace, output_file: Path, thread_i: int, exception_queue: queue.Queue) -> None:
    try:
        auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server)
        asr_service = riva.client.ASRService(auth)
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                language_code=args.language_code,
                max_alternatives=args.max_alternatives,
                profanity_filter=args.profanity_filter,
                enable_automatic_punctuation=args.automatic_punctuation,
                verbatim_transcripts=not args.no_verbatim_transcripts,
                enable_word_time_offsets=args.word_time_offsets,
            ),
            interim_results=True,
        )
        riva.client.add_word_boosting_to_config(config, args.boosted_lm_words, args.boosted_lm_score)
        for _ in range(args.config.NUM_ITERATIONS):
            with riva.client.AudioChunkFileIterator(
                args.config.INPUT_FILE,
                args.config.FILE_STREAMING_CHUNK,
                delay_callback=riva.client.sleep_audio_length if args.config.SIMULATE_REALTIME else None,
            ) as audio_chunk_iterator:
                riva.client.print_streaming(
                    responses=asr_service.streaming_response_generator(
                        audio_chunks=audio_chunk_iterator,
                        streaming_config=config,
                    ),
                    output_file=output_file,
                    additional_info=AdditionalInfo.TIME,
                    file_mode='a',
                    word_time_offsets=args.word_time_offsets,
                )
    except BaseException as e:
        exception_queue.put((e, thread_i))
        raise


def main() -> None:
    args = parse_args()
    print(f"Number of clients: {args.cofig.NUM_CLIENTS}")
    print(f"Number of iterations: {args.config.NUM_ITERATIONS}")
    print(f"Input file: {args.config.INPUT_FILE}")
    threads: List[Thread] = []
    exception_queue: queue.Queue = queue.Queue()
    for i in range(args.cofig.NUM_CLIENTS):
        t = Thread(target=streaming_transcription_worker, args=[args, Path(f"output_{i:d}.txt"), i, exception_queue])
        t.start()
        threads.append(t)
    while True:
        try:
            exc, thread_i = exception_queue.get(block=False)
        except queue.Empty:
            pass
        else:
            raise RuntimeError(f"A thread with index {thread_i} failed with error:\n{exc}")
        all_dead = True
        for t in threads:
            t.join(0.0)
            if t.is_alive():
                all_dead = False
                break
        if all_dead:
            break
        time.sleep(config.time_sleep)
    print(f"{args.cofig.NUM_CLIENTS} threads done, output written to output_<thread_id>.txt")


if __name__ == "__main__":
    main()
