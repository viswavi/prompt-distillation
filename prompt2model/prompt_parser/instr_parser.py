"""An interface for prompt parsing."""

from __future__ import annotations  # noqa FI58

import os

from prompt2model.prompt_parser.base import PromptSpec, TaskType

from prompt2model.prompt_parser.instr_parser_prompt import (  # isort: split
    construct_prompt_for_instruction_parsing,
)

from prompt2model.utils.parse_responses import parse_prompt_to_fields

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PromptBasedInstructionParser(PromptSpec):
    """Parse the prompt to separate instructions from task demonstrations."""

    def __init__(self, task_type: TaskType, max_api_calls: int = 5):
        """Initialize the prompt spec with empty parsed fields.

        We initialize the "instruction" and "examples" fields with None.
        These fields can be populated with the parse_from_prompt method.

        Args:
            task_type: Set a constant task type to use for all prompts.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
        """
        self.task_type = task_type
        self._instruction: str | None = None
        self._examples: str | None = None
        self.max_api_calls = max_api_calls

    def parse_from_prompt(self, prompt: str) -> None:
        """Parse prompt into specific fields, stored as class member variables.

        This function directly stores the parsed fields into the class's member
        variables `instruction` and `examples`. So it has no return value.

        Args:
            prompt: User prompt to parse into two specific fields:
                    "instruction" and "demonstrations".
        """
        parsing_prompt_for_chatgpt = construct_prompt_for_instruction_parsing(prompt)
        required_keys = ["Instruction", "Demonstrations"]

        extraction = parse_prompt_to_fields(
            parsing_prompt_for_chatgpt,
            required_keys,
            max_api_calls=self.max_api_calls,
        )
        self._instruction = extraction["Instruction"]
        self._examples = extraction["Demonstrations"]

    def set_instruction_and_examples(
        self, instruction: str = "", examples: str = ""
    ) -> None:
        """Set the instruction and examples directly."""
        self._instruction = instruction
        self._examples = examples
