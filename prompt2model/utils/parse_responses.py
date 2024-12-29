"""Utility file for parsing OpenAI json responses."""
from __future__ import annotations

import json
import re
from typing import Any

import openai

from prompt2model.utils import api_tools, get_formatted_logger
from prompt2model.utils.api_tools import API_ERRORS, handle_api_error

logger = get_formatted_logger("ParseJsonResponses")


def find_and_parse_json(
    response: openai.Completion, required_keys: list, optional_keys: list = []
) -> dict | None:
    """Parse stuctured fields from the API response.

    In case there are multiple JSON objects in the response, take the final one.

    Args:
        response: API response.
        required_keys: Required keys from the response
        optional_keys: Optional keys from the response

    Returns:
        If the API response is a valid JSON object and contains the
        required and optional keys then returns the
        final response as a Dictionary
        Else returns None.
    """
    if not isinstance(response, str) and hasattr(response, "choices"):
        response = response.choices[0]["message"]["content"]
    correct_json = find_rightmost_brackets(response)

    if correct_json is None:
        logger.warning("No valid JSON found in the response.")
        return None

    try:
        response_json = json.loads(correct_json, strict=False)
    except json.decoder.JSONDecodeError:
        logger.warning(f"API response was not a valid JSON: {correct_json}")
        return None

    missing_keys = [key for key in required_keys if key not in response_json]
    if len(missing_keys) != 0:
        logger.warning(f'API response must contain {", ".join(required_keys)} keys')
        return None

    final_response = {}
    for key in required_keys + optional_keys:
        if key not in response_json:
            # This is an optional key, so exclude it from the final response.
            continue
        if type(response_json[key]) == str:
            final_response[key] = response_json[key].strip()
        else:
            final_response[key] = response_json[key]
    return final_response


def find_rightmost_brackets(text: str) -> str | None:
    """Find the rightmost complete set of brackets in a string."""
    stack = []
    for i, char in enumerate(reversed(text)):
        if char == "}":
            stack.append(len(text) - i - 1)
        elif char == "{" and stack:
            start = len(text) - i - 1
            end = stack.pop()
            if not stack:  # Found the rightmost complete set
                return text[start : end + 1]
    return None


def parse_dataset_config_responses(response: openai.ChatCompletion) -> dict:
    """Parse the response to extract relevant information from dataset/configuration.

    LLMs can return the dataset configuration in different formats -
    usually either between ** ** or as a sentence.

    Args:
        response: The response containing the dataset configuration.

    Returns:
        The extracted relevant information from the dataset configuration.
    """
    if not isinstance(response, str) and hasattr(response, "choices"):
        response_str = response.choices[0]["message"]["content"]
    else:
        response_str = response

    pattern = r"\*\*(.*?)\*\*"

    match = re.search(pattern, response_str)
    dataset_config = ""
    if match:
        dataset_config = match.group(1)
    elif len(response_str.split()) >= 1:
        dataset_config = response_str.split()[-1].replace(".", "")
    return {"name": dataset_config}


def parse_prompt_to_fields(
    prompt: str,
    required_keys: list = [],
    optional_keys: list = [],
    max_api_calls: int = 5,
    module_name: str = "col_selection",
) -> dict[str, Any]:
    """Parse prompt into specific fields, and return to the calling function.

    This function calls the required api, has the logic for the retrying,
    passes the response to the parsing function, and return the
    response back or throws an error

    Args:
        prompt: User prompt into specific fields
        required_keys: Fields that need to be present in the response
        optional_keys: Field that may/may not be present in the response
        max_api_calls: Max number of retries, defaults to 5 to avoid
                        being stuck in an infinite loop
        module_name: The module this is to be used for. Currently supports
                        rerank and col_selection

    Returns:
        Parsed Response as a dictionary.

    Raises:
        ValueError: If max_api_calls is not greater than 0.
        RuntimeError: If the maximum number of API calls is reached.

    """
    chat_api = api_tools.default_api_agent
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be > 0.")

    api_call_counter = 0
    last_error = None
    while True:
        api_call_counter += 1
        try:
            response: openai.ChatCompletion | Exception = (
                chat_api.generate_one_completion(
                    prompt,
                    temperature=0.01,
                    presence_penalty=0,
                    frequency_penalty=0,
                )
            )
            extraction: dict[str, Any] | None = None
            if module_name == "col_selection":
                extraction = find_and_parse_json(response, required_keys, optional_keys)

            elif module_name == "rerank":
                extraction = parse_dataset_config_responses(response)
            if extraction is not None:
                return extraction
        except API_ERRORS as e:
            last_error = e
            handle_api_error(e, backoff_duration=2**api_call_counter)

        if api_call_counter >= max_api_calls:
            # In case we reach maximum number of API calls, we raise an error.
            logger.error("Maximum number of API calls reached.")
            raise RuntimeError("Maximum number of API calls reached.") from last_error


def make_single_api_request(prompt: str, max_api_calls: int = 10) -> str:
    """Prompts an LLM using the APIAgent, and returns the response.

    This function calls the required api, has the logic for retrying,
    returns the response back or throws an error
    Args:
        prompt: User prompt into specific fields
        max_api_calls: Max number of retries, defaults to 5 to avoid
                        being stuck in an infinite loop
    Returns:
        Response text or throws error
    """
    chat_api = api_tools.default_api_agent
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be > 0.")

    api_call_counter = 0
    last_error = None
    while True:
        api_call_counter += 1
        try:
            response: openai.ChatCompletion = chat_api.generate_one_completion(
                prompt=prompt, temperature=0.01, presence_penalty=0, frequency_penalty=0
            )
            if response is not None:
                return response.choices[0]["message"]["content"]

        except API_ERRORS as e:
            last_error = e
            handle_api_error(e, backoff_duration=2**api_call_counter)

        if api_call_counter >= max_api_calls:
            # In case we reach maximum number of API calls, we raise an error.
            logger.error("Maximum number of API calls reached.")
            raise RuntimeError("Maximum number of API calls reached.") from last_error
