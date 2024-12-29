"""Tools for accessing API-based models."""

from __future__ import annotations  # noqa FI58

import asyncio
import json
import logging
import re
import time

import aiolimiter
import litellm.utils
import openai
import tiktoken
from aiohttp import ClientSession
from litellm import acompletion, completion
from tqdm.asyncio import tqdm_asyncio

# Note that litellm converts all API errors into openai errors,
# so openai errors are valid even when using other services.
API_ERRORS = (
    openai.APIError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.BadRequestError,
    openai.APIStatusError,
    json.decoder.JSONDecodeError,
    AssertionError,
)

ERROR_ERRORS_TO_MESSAGES = {
    openai.BadRequestError: "API Invalid Request: Prompt was filtered",
    openai.RateLimitError: "API rate limit exceeded. Sleeping for 10 seconds.",
    openai.APIConnectionError: "Error Communicating with API",
    openai.APITimeoutError: "API Timeout Error: API Timeout",
    openai.APIStatusError: "API service unavailable error: {e}",
    openai.APIError: "API error: {e}",
}
BUFFER_DURATION = 2


class APIAgent:
    """A class for accessing API-based models."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        max_tokens: int | None = 4000,
        api_base: str | None = None,
    ):
        """Initialize APIAgent with model_name and max_tokens.

        Args:
            model_name: Name of the model to use (by default, gpt-4).
            max_tokens: The maximum number of tokens to generate. Defaults to the max
                value for the model if available through litellm.
            api_base: Custom endpoint for Hugging Face's inference API.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.api_base = api_base
        if max_tokens is None:
            try:
                self.max_tokens = litellm.utils.get_max_tokens(model_name)
                if isinstance(self.max_tokens, dict):
                    self.max_tokens = self.max_tokens["max_tokens"]
            except Exception:
                pass

    def generate_one_completion(
        self,
        prompt: str,
        temperature: float = 0,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        token_buffer: int = 300,
    ) -> openai.Completion:
        """Generate a chat completion using an API-based model.

        Args:
            prompt: A prompt asking for a response.
            temperature: What sampling temperature to use, between 0 and 2. Higher
                values like 0.8 will make the output more random, while lower values
                like 0.2 will make it more focused and deterministic.
            presence_penalty: Float between -2.0 and 2.0. Positive values penalize new
                tokens based on whether they appear in the text so far, increasing the
                model's likelihood to talk about new topics.
            frequency_penalty: Float between -2.0 and 2.0. Positive values penalize new
                tokens based on their existing frequency in the text so far, decreasing
                the model's likelihood of repeating the same line verbatim.
            token_buffer: Number of tokens below the LLM's limit to generate. In case
                our tokenizer does not exactly match the LLM API service's perceived
                number of tokens, this prevents service errors. On the other hand, this
                may lead to generating fewer tokens in the completion than is actually
                possible.

        Returns:
            An OpenAI-like response object if there were no errors in generation.
            In case of API-specific error, Exception object is captured and returned.
        """
        num_prompt_tokens = count_tokens_from_string(prompt)
        if self.max_tokens:
            max_tokens = self.max_tokens - num_prompt_tokens - token_buffer
        else:
            max_tokens = 3 * num_prompt_tokens
        response = completion(  # completion gets the key from os.getenv
            model=self.model_name,
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ],
            api_base=self.api_base,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
        )
        return response

    async def generate_batch_completion(
        self,
        prompts: list[str],
        temperature: float = 1,
        responses_per_request: int = 5,
        requests_per_minute: int = 80,
        token_buffer: int = 300,
    ) -> list[openai.Completion]:
        """Generate a batch responses from OpenAI Chat Completion API.

        Args:
            prompts: List of prompts to generate from.
            model_config: Model configuration.
            temperature: Temperature to use.
            responses_per_request: Number of responses for each request.
                i.e. the parameter n of API call.
            requests_per_minute: Number of requests per minute to allow.
            token_buffer: Number of tokens below the LLM's limit to generate. In case
                our tokenizer does not exactly match the LLM API service's perceived
                number of tokens, this prevents service errors. On the other hand, this
                may lead to generating fewer tokens in the completion than is actually
                possible.

        Returns:
            List of generated responses.
        """
        async with ClientSession() as _:
            limiter = aiolimiter.AsyncLimiter(requests_per_minute)

            async def _throttled_completion_acreate(
                model: str,
                messages: list[dict[str, str]],
                temperature: float,
                max_tokens: int,
                n: int,
                top_p: float,
                limiter: aiolimiter.AsyncLimiter,
            ):
                async with limiter:
                    for _ in range(3):
                        try:
                            return await acompletion(
                                model=model,
                                messages=messages,
                                api_base=self.api_base,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                n=n,
                                top_p=top_p,
                            )
                        except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                            if isinstance(
                                e,
                                (
                                    openai.APIStatusError,
                                    openai.APIError,
                                ),
                            ):
                                logging.warning(
                                    ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e)
                                )
                            elif isinstance(e, openai.BadRequestError):
                                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                                return {
                                    "choices": [
                                        {
                                            "message": {
                                                "content": "Invalid Request: Prompt was filtered"  # noqa E501
                                            }
                                        }
                                    ]
                                }
                            else:
                                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                            await asyncio.sleep(10)
                    return {"choices": [{"message": {"content": ""}}]}

            num_prompt_tokens = max(
                count_tokens_from_string(prompt) for prompt in prompts
            )
            if self.max_tokens:
                max_tokens = self.max_tokens - num_prompt_tokens - token_buffer
            else:
                max_tokens = 3 * num_prompt_tokens

            async_responses = [
                _throttled_completion_acreate(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": f"{prompt}"},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=responses_per_request,
                    top_p=1,
                    limiter=limiter,
                )
                for prompt in prompts
            ]
            responses = await tqdm_asyncio.gather(*async_responses)
        # Note: will never be none because it's set, but mypy doesn't know that.
        return responses


def handle_api_error(e, backoff_duration=1) -> None:
    """Handle OpenAI errors or related errors that the API may raise.

    Sleeps incase error is some type of timeout, else throws error.

    Args:
        e: The error to handle. This could be an OpenAI error or a related
           non-fatal error, such as JSONDecodeError or AssertionError.
        backoff_duration: The duration (in s) to wait before retrying.

    Raises:
        e: If the error is not an instance of APIError, Timeout, or RateLimitError.

    Returns:
        None
    """
    logging.error(e)

    if not isinstance(e, API_ERRORS):
        raise e

    if isinstance(
        e,
        (openai.APIError, openai.APITimeoutError, openai.RateLimitError),
    ):

        match = re.search(r"Please retry after (\d+) seconds", str(e))
        # If OpenAI mentions how long to sleep, use that. Otherwise, do
        # exponential backoff.
        if match is not None:
            backoff_duration = int(match.group(1)) + BUFFER_DURATION

        logging.info(f"Retrying in {backoff_duration} seconds...")
        time.sleep(backoff_duration)


def count_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Handle count the tokens in a string with OpenAI's tokenizer.

    Args:
        string: The string to count.
        encoding_name: The name of the tokenizer to use.

    Returns:
        The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# This is the default API agent that is used everywhere if a different agent is not
# specified
default_api_agent = APIAgent(max_tokens=4000)
