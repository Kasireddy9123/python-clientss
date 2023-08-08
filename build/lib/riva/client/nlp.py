from typing import List, Tuple, Union, Generator

import riva.client.proto.riva_nlp_pb2 as rnlp
import riva.client.proto.riva_nlp_pb2_grpc as rnlp_srv
from riva.client import Auth

import logging
import inspect

from config import riva_nlp_config

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

class NLPService:
    """Provides various natural language processing services."""

    def __init__(self, auth: Auth) -> None:
        """
        Initializes an instance of the class.

        Args:
            auth (Auth): An instance of riva.client.Auth for authentication metadata generation.
        """
        self.auth = auth
        self.stub = rnlp_srv.RivaLanguageUnderstandingStub(self.auth.channel)

    # The classify_text, classify_tokens, transform_text, analyze_entities, analyze_intent,
    # and punctuate_text methods remain unchanged

    def natural_query(
        self, query: str, context: str, top_n: int = 1, future: bool = False
    ) -> Union[rnlp.NaturalQueryResponse, _MultiThreadedRendezvous]:
        """
        A search function that enables querying one or more documents or contexts with a query that is written in
        natural language.

        Args:
            query (str): A natural language query.
            context (str): A context to search with the above query.
            top_n (int, optional): A maximum number of answers to return for the query. Defaults to 1.
            future (bool, optional): Whether to return an async result instead of the usual response.
                You can get a response by calling the ``result()`` method of the future object. Defaults to False.

        Returns:
            Union[rnlp.NaturalQueryResponse, grpc._channel._MultiThreadedRendezvous]: A response with a result.
        """
        request = rnlp.NaturalQueryRequest(query=query, context=context, top_n=top_n)
        func = self.stub.NaturalQuery.future if future else self.stub.NaturalQuery
        return func(request, metadata=self.auth.get_auth_metadata())

    def batch_generator(self, examples: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
        """
        Generate batches of examples with a given batch size.

        Args:
            examples (List[Any]): List of examples.
            batch_size (int): Batch size.

        Yields:
            Generator[List[Any], None, None]: Batches of examples.
        """
        for i in range(0, len(examples), batch_size):
            yield examples[i: i + batch_size]

    def process_batches_async(
        self,
        b_gen: Generator[List[Any], None, None],
        process_func: Callable[..., _MultiThreadedRendezvous],
        kwargs_except_future_and_input: Dict[str, Any],
        max_async_requests_to_queue: int,
    ) -> List[rnlp.RecognizeResponse]:
        """
        Process batches of examples asynchronously.

        Args:
            b_gen (Generator[List[Any], None, None]): Batches of examples generator.
            process_func (Callable[..., _MultiThreadedRendezvous]): Function to process each batch.
            kwargs_except_future_and_input (Dict[str, Any]): Keyword arguments for the processing function.
            max_async_requests_to_queue (int): Maximum number of async requests to queue.

        Returns:
            List[rnlp.RecognizeResponse]: List of responses.
        """
        responses = []
        n_req = max_async_requests_to_queue
        while n_req == max_async_requests_to_queue:
            n_req = 0
            futures = []
            for batch in b_gen:
                futures.append(process_func(input_strings=batch, **kwargs_except_future_and_input, future=True))
                n_req += 1
                if n_req == max_async_requests_to_queue:
                    break
            for f in futures:
                responses.append(f.result())
        return responses

    def check_max_async_requests_to_queue(self, max_async_requests_to_queue: int) -> None:
        """
        Check if the value of max_async_requests_to_queue is a valid non-negative integer.

        Args:
            max_async_requests_to_queue (int): Maximum number of async requests to queue.

        Raises:
            ValueError: If max_async_requests_to_queue is not a valid non-negative integer.
        """
        if not isinstance(max_async_requests_to_queue, int) or max_async_requests_to_queue < 0:
            raise ValueError(
                f"Parameter `max_async_requests_to_queue` has to be a non-negative integer, but "
                f"`max_async_requests_to_queue={max_async_requests_to_queue}` was given."
            )

    def classify_text_batch(
        self,
        input_strings: List[str],
        model_name: str,
        batch_size: int,
        language_code: str = 'en-US',
        max_async_requests_to_queue: int = 0,
    ) -> Tuple[List[str], List[float]]:
        """
        Classify a batch of texts using the specified model.

        Args:
            input_strings (List[str]): List of texts to classify.
            model_name (str): Name of the model to use for classification.
            batch_size (int): Batch size.
            language_code (str, optional): Language code. Defaults to 'en-US'.
            max_async_requests_to_queue (int, optional): Maximum number of async requests to queue. Defaults to 0.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing lists of text classes and confidences.
        """
        self.check_max_async_requests_to_queue(max_async_requests_to_queue)
        responses = []
        if max_async_requests_to_queue == 0:
            for batch in self.batch_generator(input_strings, batch_size):
                responses.append(self.classify_text(batch, model_name, language_code))
        else:
            responses = self.process_batches_async(
                self.batch_generator(input_strings, batch_size),
                self.classify_text,
                {'model_name': model_name, 'language_code': language_code},
                max_async_requests_to_queue,
            )
        classes, confidences = [], []
        for response in responses:
            b_classes, b_confidences = extract_most_probable_text_class_and_confidence(response)
            classes.extend(b_classes)
            confidences.extend(b_confidences)
        return classes, confidences

    def classify_tokens_batch(
        self,
        input_strings: List[str],
        model_name: str,
        batch_size: int,
        language_code: str = 'en-US',
        max_async_requests_to_queue: int = 0,
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]], List[List[int]], List[List[int]]]:
        """
        Classify a batch of texts into tokens using the specified model.

        Args:
            input_strings (List[str]): List of texts to classify.
            model_name (str): Name of the model to use for classification.
            batch_size (int): Batch size.
            language_code (str, optional): Language code. Defaults to 'en-US'.
            max_async_requests_to_queue (int, optional): Maximum number of async requests to queue. Defaults to 0.

        Returns:
            Tuple[List[List[str]], List[List[str]], List[List[float]], List[List[int]], List[List[int]]]: A tuple
            containing lists of tokens, token classes, confidences, starts, and ends.
        """
        self.check_max_async_requests_to_queue(max_async_requests_to_queue)
        responses = []
        if max_async_requests_to_queue == 0:
            for batch in self.batch_generator(input_strings, batch_size):
                responses.append(
                    self.classify_tokens(batch, model_name, language_code)
                )
        else:
            responses = self.process_batches_async(
                self.batch_generator(input_strings, batch_size),
                self.classify_tokens,
                {'model_name': model_name, 'language_code': language_code},
                max_async_requests_to_queue,
            )
        tokens, token_classes, confidences, starts, ends = [], [], [], [], []
        for response in responses:
            b_t, b_tc, b_conf, b_s, b_e = extract_most_probable_token_classification_predictions(response)
            tokens.extend(b_t)
            token_classes.extend(b_tc)
            confidences.extend(b_conf)
            starts.extend(b_s)
            ends.extend(b_e)
        return tokens, token_classes, confidences, starts, ends
