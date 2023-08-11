from typing import List, Tuple, Union, Generator

import riva.client.proto.riva_nlp_pb2 as rnlp
import riva.client.proto.riva_nlp_pb2_grpc as rnlp_srv
from riva.client import Auth

import logging
import inspect

from nlp_config import riva_nlp_config

from nlpService import NLPService

# Configure logging with a custom log formatter
log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s'
logging.basicConfig(filename='app.log', level=logging.DEBUG, format=log_format)

# Helper function to log the print statement
def log_print(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    logging.debug(message)

# Replace the print statements with log_print
print = log_print

def extract_all_text_classes_and_confidences(
    response: rnlp.TextClassResponse
) -> Tuple[List[List[str]], List[List[float]]]:
    text_classes, confidences = [], []
    for batch_elem_result in response.results:
        text_classes.append([lbl.class_name for lbl in batch_elem_result.labels])
        confidences.append([lbl.score for lbl in batch_elem_result.labels])
    return text_classes, confidences


def extract_most_probable_text_class_and_confidence(response: rnlp.TextClassResponse) -> Tuple[List[str], List[float]]:
    intents, confidences = extract_all_text_classes_and_confidences(response)
    return [x[0] for x in intents], [x[0] for x in confidences]


def extract_all_token_classification_predictions(
    response: rnlp.TokenClassResponse
) -> Tuple[
    List[List[str]],
    List[List[List[str]]],
    List[List[List[float]]],
    List[List[List[int]]],
    List[List[List[int]]]
]:
    tokens, token_classes, confidences, starts, ends = [], [], [], [], []
    for batch_elem_result in response.results:
        elem_tokens, elem_token_classes, elem_confidences, elem_starts, elem_ends = [], [], [], [], []
        for token_result in batch_elem_result.results:
            elem_tokens.append(token_result.token)
            elem_token_classes.append([lbl.class_name for lbl in token_result.label])
            elem_confidences.append([lbl.score for lbl in token_result.label])
            elem_starts.append([span.start for span in token_result.span])
            elem_ends.append([span.end for span in token_result.span])
        tokens.append(elem_tokens)
        token_classes.append(elem_token_classes)
        confidences.append(elem_confidences)
        starts.append(elem_starts)
        ends.append(elem_ends)
    return tokens, token_classes, confidences, starts, ends


def extract_most_probable_token_classification_predictions(
    response: rnlp.TokenClassResponse
) -> Tuple[List[List[str]], List[List[str]], List[List[float]], List[List[int]], List[List[int]]]:
    tokens, token_classes, confidences, starts, ends = extract_all_token_classification_predictions(response)
    return (
        tokens,
        [[xx[0] for xx in x] for x in token_classes],
        [[xx[0] for xx in x] for x in confidences],
        [[xx[0] for xx in x] for x in starts],
        [[xx[0] for xx in x] for x in ends],
    )


def extract_all_transformed_texts(response: rnlp.TextTransformResponse) -> List[str]:
    return [t for t in response.text]


def extract_most_probable_transformed_text(response: rnlp.TextTransformResponse) -> str:
    return response.text[0]


def prepare_transform_text_request(
    input_strings: Union[List[str], str], model_name: str, language_code: str = 'en-US'
) -> rnlp.TextTransformRequest:
    if isinstance(input_strings, str):
        input_strings = [input_strings]
    request = rnlp.TextTransformRequest()
    if model_name is not None:
        request.model.model_name = model_name
    request.model.language_code = language_code
    for q in input_strings:
        request.text.append(q)
    return request

# Create Instance of NLPService
obj = NLPService()

# Call methods of NLPService
obj.__init__()
obj.classify_text()
obj.classify_tokens()
obj.transform_text()
obj.analyze_entities()
obj.analyze_intent()
obj.punctuate_text()
obj.natural_query()
obj.batch_generator()
obj.process_batches_async()
obj.check_max_async_requests_to_queue()
obj.classify_text_batch()
obj.classify_tokens_batch()