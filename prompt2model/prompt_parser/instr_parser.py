"""An interface for prompt parsing."""

from __future__ import annotations  # noqa FI58

import json
import logging
import time

import openai

from prompt2model.prompt_parser.base import PromptSpec, TaskType

from prompt2model.prompt_parser.instr_parser_prompt import (  # isort: split
    construct_prompt_for_instruction_parsing,
)
from prompt2model.utils import ChatGPTAgent


class OpenAIInstructionParser(PromptSpec):
    """Parse the prompt to separate instructions from task demonstrations."""

    def __init__(
        self, task_type: TaskType, api_key: str | None = None, max_api_calls: int = None
    ):
        """Initialize the prompt spec with empty parsed fields.

        We initialize the "instruction" and "demonstration" fields with None.
        These fields can be populated with the parse_from_prompt method.

        Args:
            task_type: Set a constant task type to use for all prompts.
            api_key: A valid OpenAI API key. Alternatively, set as None and
                     set the environment variable `OPENAI_API_KEY=<your key>`.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
        """
        self.task_type = task_type
        self.instruction: str | None = None
        self.demonstration: str | None = None
        self.api_key: str | None = api_key
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0

    def extract_response(self, response: openai.Completion) -> tuple[str, str]:
        """Parse stuctured fields from the OpenAI API response.

        Args:
            response: OpenAI API response.

        Returns:
            tuple[str, str | None]: Tuple consisting of:
                1) Instruction: The instruction parsed from the API response.
                2) Demonstrations: (Optional) demonstrations parsed from the
                   API response.
        """
        response_text = response.choices[0]["message"]["content"]
        try:
            response_json = json.loads(response_text, strict=False)
        except json.decoder.JSONDecodeError as e:
            logging.warning("API response was not a valid JSON")
            raise e
        instruction_string = response_json["Instruction"].strip()
        demonstration_string = response_json["Demonstrations"].strip()
        return instruction_string, demonstration_string

    def parse_from_prompt(self, prompt: str) -> None:
        """Parse prompt into specific fields, stored as class member variables.

        Args:
            prompt: User prompt to parse into two specific fields:
                    "instruction" and "demonstrations".

        Returns:
            None: this void function directly stores the parsed fields into
            the class's member variables `instruction` and `demonstration.

        """
        parsing_prompt_for_chatgpt = construct_prompt_for_instruction_parsing(prompt)

        chat_api = ChatGPTAgent(self.api_key)
        while True:
            try:
                response = chat_api.generate_openai_chat_completion(
                    parsing_prompt_for_chatgpt
                )
                self.instruction, self.demonstration = self.extract_response(response)
                break
            except (
                openai.error.APIError,
                openai.error.Timeout,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                json.decoder.JSONDecodeError,
            ) as e:
                self.api_call_counter += 1
                if self.max_api_calls:
                    if self.api_call_counter < self.max_api_calls:
                        if (
                            isinstance(e, openai.error.APIError)
                            or isinstance(e, openai.error.Timeout)
                            or isinstance(e, openai.error.RateLimitError)
                        ):
                            # For these errors, OpenAI recommends waiting before
                            # retrying.
                            time.sleep(1)
                        continue
                    else:
                        logging.error("Maximum number of API calls reached.")
                        raise e
        return None
