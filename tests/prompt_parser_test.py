"""Tests for the prompt_parser module."""

import gc
import logging
from unittest.mock import patch

import openai
import pytest

from prompt2model.prompt_parser import PromptBasedInstructionParser, TaskType
from prompt2model.prompt_parser.mock import MockPromptSpec
from prompt2model.utils import api_tools
from test_helpers import MockCompletion, UnknownGpt3Exception
from test_helpers.mock_api import MockAPIAgent
from test_helpers.test_utils import temp_setattr

logger = logging.getLogger("ParseJsonResponses")
GPT3_RESPONSE_WITH_DEMONSTRATIONS = MockCompletion(
    '{"Instruction": "Convert each date from an informal description into a'
    ' MM/DD/YYYY format.", "Demonstrations": "Fifth of November 2024 ->'
    ' 11/05/2024\nJan. 9 2023 -> 01/09/2023\nChristmas 2016 -> 12/25/2016"}'
)
GPT3_RESPONSE_WITHOUT_DEMONSTRATIONS = MockCompletion(
    '{"Instruction": "Turn the given fact into a question by a simple rearrangement'
    " of words. This typically involves replacing some part of the given fact with a"
    " WH word. For example, replacing the subject of the provided fact with the word"
    ' \\"what\\" can form a valid question. Don\'t be creative! You just need to'
    " rearrange the words to turn the fact into a question - easy! Don't just"
    " randomly remove a word from the given fact to form a question. Remember that"
    " your question must evaluate scientific understanding. Pick a word or a phrase"
    " in the given fact to be the correct answer, then make the rest of the question."
    " You can also form a question without any WH words. For example, 'A radio"
    ' converts electricity into?\'", "Demonstrations": "N/A"}'
)
GPT3_RESPONSE_WITH_INVALID_JSON = MockCompletion(
    '{"Instruction": "A", "Demonstrations": "B}'  # Missing final quotation mark
)


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_WITH_DEMONSTRATIONS],
)
def test_instruction_parser_with_demonstration(mocked_parsing_method):
    """Test a prompt-based instruction (with the LLM call mocked).

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    prompt = """Convert each date from an informal description into a MM/DD/YYYY format.
Fifth of November 2024 -> 11/05/2024
Jan. 9 2023 -> 01/09/2023
Christmas 2016 -> 12/25/2016"""
    prompt_spec = PromptBasedInstructionParser(task_type=TaskType.TEXT_GENERATION)
    prompt_spec.parse_from_prompt(prompt)

    assert prompt_spec.task_type == TaskType.TEXT_GENERATION
    correct_instruction = (
        "Convert each date from an informal description into a MM/DD/YYYY format."
    )
    assert prompt_spec.instruction == correct_instruction
    assert prompt_spec.instruction == correct_instruction
    assert (
        prompt_spec.examples
        == """Fifth of November 2024 -> 11/05/2024
Jan. 9 2023 -> 01/09/2023
Christmas 2016 -> 12/25/2016"""
    )
    assert (
        prompt_spec.examples
        == """Fifth of November 2024 -> 11/05/2024
Jan. 9 2023 -> 01/09/2023
Christmas 2016 -> 12/25/2016"""
    )
    assert mocked_parsing_method.call_count == 1


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_WITHOUT_DEMONSTRATIONS],
)
def test_instruction_parser_without_demonstration(mocked_parsing_method):
    """Test a prompt-based instruction (with the LLM call mocked).

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    prompt = """Turn the given fact into a question by a simple rearrangement of words. This typically involves replacing some part of the given fact with a WH word. For example, replacing the subject of the provided fact with the word \"what\" can form a valid question. Don't be creative! You just need to rearrange the words to turn the fact into a question - easy! Don't just randomly remove a word from the given fact to form a question. Remember that your question must evaluate scientific understanding. Pick a word or a phrase in the given fact to be the correct answer, then make the rest of the question. You can also form a question without any WH words. For example, 'A radio converts electricity into?'"""  # noqa: E501
    prompt_spec = PromptBasedInstructionParser(task_type=TaskType.TEXT_GENERATION)
    prompt_spec.parse_from_prompt(prompt)

    assert prompt_spec.task_type == TaskType.TEXT_GENERATION
    correct_instruction = "Turn the given fact into a question by a simple rearrangement of words. This typically involves replacing some part of the given fact with a WH word. For example, replacing the subject of the provided fact with the word \"what\" can form a valid question. Don't be creative! You just need to rearrange the words to turn the fact into a question - easy! Don't just randomly remove a word from the given fact to form a question. Remember that your question must evaluate scientific understanding. Pick a word or a phrase in the given fact to be the correct answer, then make the rest of the question. You can also form a question without any WH words. For example, 'A radio converts electricity into?'"  # noqa: E501
    assert prompt_spec.instruction == correct_instruction
    assert prompt_spec.instruction == correct_instruction
    assert prompt_spec.examples == "N/A"
    assert prompt_spec.examples == "N/A"
    assert mocked_parsing_method.call_count == 1
    gc.collect()


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_WITH_INVALID_JSON] * 3,
)
def test_instruction_parser_with_invalid_json(mocked_parsing_method):
    """Verify that we handle when the API returns a invalid JSON response.

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    prompt = """This prompt will be ignored by the parser in this test."""
    prompt_spec = PromptBasedInstructionParser(
        task_type=TaskType.TEXT_GENERATION, max_api_calls=3
    )
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        with pytest.raises(RuntimeError):
            prompt_spec.parse_from_prompt(prompt)
        mock_info.assert_not_called()
        warning_list = [each.args[0] for each in mock_warning.call_args_list]
        assert (
            warning_list
            == [
                'API response was not a valid JSON: {"Instruction": "A", "Demonstrations": "B}'  # noqa: E501
            ]
            * 3
        )  # noqa: E501
    assert mocked_parsing_method.call_count == 3
    assert prompt_spec._instruction is None
    assert prompt_spec._examples is None
    gc.collect()


@patch("time.sleep")
@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=openai.APITimeoutError("timeout"),
)
def test_instruction_parser_with_timeout(mocked_parsing_method, mocked_sleep_method):
    """Verify that we wait and retry (a set number of times) if the API times out.

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using the
                               API. The mocked API call raises a `openai.error.Timeout`
                               error when we request a chat completion.
        mocked_sleep_method: When `time.sleep` is called, we mock it to do nothing.
                             We simply use this mock to verify that the function waits
                             some time after each API timeout.
    """
    prompt = """This prompt will be ignored by the parser in this test."""
    with pytest.raises(RuntimeError) as exc_info:
        prompt_spec = PromptBasedInstructionParser(
            task_type=TaskType.TEXT_GENERATION, max_api_calls=3
        )
        prompt_spec.parse_from_prompt(prompt)

    # If we allow 3 API calls, we should have 2 sleep calls (1 after each
    # timeout).
    assert mocked_sleep_method.call_count == 3
    assert mocked_parsing_method.call_count == 3

    # Check if the RuntimeError was raised
    assert isinstance(exc_info.value, RuntimeError)
    # Check if the original exception (e) is present as the cause
    original_exception = exc_info.value.__cause__
    assert isinstance(original_exception, openai.APITimeoutError)
    gc.collect()


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=UnknownGpt3Exception(),
)
def test_instruction_parser_with_unexpected_error(mocked_parsing_method):
    """Verify we don't retry the API call if an unexpected exception appears.

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    prompt = """This prompt will be ignored by the parser in this test."""
    with pytest.raises(UnknownGpt3Exception):
        prompt_spec = PromptBasedInstructionParser(
            task_type=TaskType.TEXT_GENERATION, max_api_calls=3
        )
        prompt_spec.parse_from_prompt(prompt)

    # Check that we only tried calling the API once.
    assert mocked_parsing_method.call_count == 1
    gc.collect()


def test_prompt_parser_agent_switch():
    """Test if prompt parser can use a user-set API agent."""
    my_agent = MockAPIAgent(
        default_content='{"Instruction": "test response", "Demonstrations": "test response"}'  # noqa: E501
    )
    with temp_setattr(api_tools, "default_api_agent", my_agent):
        prompt_parser = PromptBasedInstructionParser(
            TaskType.CLASSIFICATION, max_api_calls=3
        )
        prompt_spec = MockPromptSpec(TaskType.CLASSIFICATION)
        prompt_parser.parse_from_prompt(prompt_spec)
    assert my_agent.generate_one_call_counter == 1
