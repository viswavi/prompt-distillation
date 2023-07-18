"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from __future__ import annotations  # noqa FI58

import json
import logging
import os
import random

import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.dataset_generator.openai_gpt_template import construct_meta_prompt
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import OPENAI_ERRORS, ChatGPTAgent, handle_openai_error


class OpenAIDatasetGenerator(DatasetGenerator):
    """A abstract class for NLP dataset generation using OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str | None = None, max_api_calls: int = None):
        """Initialize an OpenAI DatasetGenerator with an API key and max API call.

        Args:
            api_key: A valid OpenAI API key. Alternatively, set as None and set
                the environment variable with `export OPENAI_API_KEY=<your key>`.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
        """
        self.api_key: str | None = api_key if api_key else os.environ["OPENAI_API_KEY"]
        assert self.api_key is not None and self.api_key != "", (
            "API key must be provided"
            + " or set the environment variable with `export OPENAI_API_KEY=<your key>`"
        )
        if max_api_calls:
            assert max_api_calls > 0, "max_api_calls must be > 0"
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0
        self.recent_10_generated_examples = []  # type: list[dict[str, str]]
        # Randomly selected several examples as addtional few-shot examples
        # from the last 10 examples to generate new examples.

    def generate_prompt(
        self,
        instruction: str,
        few_shot_example_string: str = None,
    ) -> tuple[str, str]:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt.

        Returns:
            The generated prompt string and the few-shot examples string.
        """
        # The random_example_string is a string, which contains several random
        # few-shot examples as demonstrations for the DatasetGenertor. If
        # self.recent_10_generated_examples is empty, then the
        # random_example_string is the few-shot examples parsed from the user's
        # prompt.
        if len(self.recent_10_generated_examples) == 0:
            random_example_string = (
                (few_shot_example_string + "\n")
                if (
                    few_shot_example_string is not None
                    and few_shot_example_string != "N/A"
                    and few_shot_example_string != ""
                )
                else "N/A\n"
            )
            # This is the default random_example_string if self.recent_10_generated_examples
            # is empty. few_shot_example_string is the few-shot examples parsed from the
            # user's prompt. But if user does not provide any few-shot examples in the input
            # prompt, the few_shot_example_string will be "N/A"/""/None.
            random_selected_generted_example_num = 0
            # random_selected_generted_example_num is the number of selected
            # random examples from self.recent_10_generated_examples that will
            # be added to random_example_string. If the self.recent_10_generated_examples
            # is empty, then random_selected_generted_example_num is 0.
        else:
            # If self.recent_10_generated_examples is not empty, then the
            # random_example_string is the few-shot examples parsed from the user's
            # input prompt by the PromptParser, together with sveral random generated
            # examples from self.recent_10_generated_examples.

            # To increase the diversity of the random_example_string, we first select
            # several random examples from self.recent_10_generated_examples.
            # And then the few-shot examples parsed from the user's input prompt
            # will be inserted into these random examples in a random index.
            random_example_string = ""
            random_selected_generted_example_num = random.randint(
                1, len(self.recent_10_generated_examples)
            )
            # random_selected_generted_example_num is the number of selected
            # random examples from self.recent_10_generated_examples that will
            # be added to random_example_string.
            random_examples = random.sample(
                self.recent_10_generated_examples, random_selected_generted_example_num
            )
            # If recent_10_generated_examples is not empty, then choose several
            # random examples from self.recent_10_generated_examples to construct
            # new random_example_string, else use the default random_example_string.
            user_examples_insert_index = random.randint(0, len(random_examples) - 1)
            for index, example in enumerate(random_examples):
                random_example_string += (
                    f"input=\"{example['input']}\"\noutput=\"{example['output']}\"\n"
                )
                if (
                    # If the index equals to user_examples_insert_index and the
                    # few_shot_example_string is valid, then add the few-shot
                    # into the random_example_string at the index.
                    index == user_examples_insert_index
                    and few_shot_example_string is not None
                    and few_shot_example_string != "N/A"
                    and few_shot_example_string != ""
                ):
                    random_example_string += few_shot_example_string + "\n"
        # To increase the diversity of the prompt to DatasetGenerator and
        # save the cost of API calls,  the fewer the random_selected_generted_example_num
        # is, the more complex the prompt_type is.
        if 0 <= random_selected_generted_example_num <= 3:
            template_type = "COMPLEX"
        elif 4 <= random_selected_generted_example_num <= 7:
            template_type = "MIDDLE"
        else:
            template_type = "SIMPLE"
        prompt = construct_meta_prompt(
            instruction=instruction,
            few_shot_example_string=random_example_string,
            template_type=template_type,
        )
        return prompt, random_example_string

    def extract_response(self, response: openai.Completion) -> tuple[str, str]:
        """Extracts the generated sample and annotation from an OpenAI API response.

        Args:
            response (openai.Completion): The response object returned by OpenAI API.

        Returns:
            A tuple of (sample, annotation), where:
            - sample is the generated example string extracted from the response.
            - annotation is the generated label string extracted from the response.
        """
        try:
            response_json = json.loads(response.choices[0]["message"]["content"])
        except json.decoder.JSONDecodeError as e:
            logging.warning("API response was not a valid JSON")
            raise e
        required_keys = ["input", "output"]
        missing_keys = [key for key in required_keys if key not in response_json]
        assert (
            len(missing_keys) == 0
        ), f'API response must contain {", ".join(required_keys)} keys'
        input = str(response_json["input"]).strip()
        output = str(response_json["output"]).strip()
        return input, output

    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> Dataset:
        """Generate a single dataset using GPT-3.5.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.

        Returns:
            A single dataset.
        """
        _ = split  # suppress unused variable warnings

        chat_api = ChatGPTAgent(self.api_key)
        input_cols = []  # type: list[str]
        output_cols = []  # type: list[str]

        for _ in tqdm(range(num_examples), desc="Generating examples"):
            while True:
                try:
                    if (
                        self.max_api_calls
                        and self.api_call_counter >= self.max_api_calls
                    ):
                        logging.warning("Maximum number of API calls reached.")
                        return Dataset.from_dict(
                            {"input_col": input_cols, "output_col": output_cols}
                        )
                    else:
                        self.api_call_counter += 1
                        prompt, _ = self.generate_prompt(
                            instruction=prompt_spec.get_instruction,
                            few_shot_example_string=prompt_spec.get_examples,
                        )
                    response = chat_api.generate_openai_chat_completion(prompt)
                    input_col, output_col = self.extract_response(response)
                    logging.info(f"Prompt: \n\n{prompt}\n\n")  # noqa: E501
                    logging.info(f"Input: \n\n{input_col}\n\n")  # noqa: E501
                    logging.info(f"Output: \n\n{output_col}\n\n")  # noqa: E501
                    input_cols.append(input_col)
                    output_cols.append(output_col)
                    assert 0 <= len(self.recent_10_generated_examples) <= 10
                    if len(self.recent_10_generated_examples) == 10:
                        self.recent_10_generated_examples.pop(0)
                    self.recent_10_generated_examples.append(
                        {"input": input_col, "output": output_col}
                    )
                    break
                except OPENAI_ERRORS as e:
                    self.api_call_counter = handle_openai_error(
                        e, self.api_call_counter
                    )

        return Dataset.from_dict({"input_col": input_cols, "output_col": output_cols})
