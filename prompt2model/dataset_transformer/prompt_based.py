"""A simple dataset transformer that uses a plan prompt and transform prompt."""
from __future__ import annotations

import asyncio
from collections.abc import Callable

import datasets

from prompt2model.dataset_retriever.task_expansion_prompt import (
    construct_prompt_for_task_explanation,
)
from prompt2model.dataset_transformer.base import DatasetTransformer
from prompt2model.dataset_transformer.prompt_template import (
    construct_prompt_for_plan,
    construct_prompt_for_transform_data,
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import (
    API_ERRORS,
    api_tools,
    get_formatted_logger,
    handle_api_error,
)
from prompt2model.utils.parse_responses import (
    find_and_parse_json,
    make_single_api_request,
)

logger = get_formatted_logger("DatasetTransformer")


class PromptBasedDatasetTransformer(DatasetTransformer):
    """Transform data based on a transform prompt."""

    def __init__(
        self,
        num_points_to_transform: int = 10,
        max_allowed_failed_transforms: int = 3,
        plan_prompt_fn: Callable[
            [str, str, list[dict]], str
        ] = construct_prompt_for_plan,
        transform_prompt_fn: Callable[
            [str, str, str, str], str
        ] = construct_prompt_for_transform_data,
        num_retries: int = 10,
    ):
        """Initializes an instance of the PromptBasedDatasetTransformer class.

        Args:
            num_points_to_transform: The number of points to transform.
            max_allowed_failed_transforms: The maximum number of
                                                 failed transforms allowed.
            plan_prompt_fn: The function to construct the prompt for plan
            transform_prompt_fn: The function to construct the prompt
                                 for transform data.
            num_retries: The number of retries to attempt for each API call.
        """
        self.plan_prompt_fn = plan_prompt_fn
        self.transform_prompt_fn = transform_prompt_fn
        self.plan: str = ""
        self.num_points_to_transform = num_points_to_transform
        self.curr_failed_transforms = 0
        self.max_allowed_failed_transforms = max_allowed_failed_transforms
        self.num_retries = num_retries

    def generate_task_explanation(self, prompt_spec: PromptSpec) -> str:
        """Generate task explanation."""
        task_explanation_prompt = construct_prompt_for_task_explanation(
            prompt_spec.instruction, prompt_spec.examples
        )
        return make_single_api_request(
            task_explanation_prompt, max_api_calls=self.num_retries
        )

    def generate_plan(
        self, task_explanation: str, dataset: datasets.Dataset, prompt_spec: PromptSpec
    ) -> str:
        """Generate plan for the task."""
        plan_prompt = self.plan_prompt_fn(
            task_explanation, prompt_spec.examples, dataset
        )
        return make_single_api_request(plan_prompt, max_api_calls=self.num_retries)

    def generate_transform_prompts(
        self,
        task_explanation: str,
        dataset: datasets.Dataset,
        prompt_spec: PromptSpec,
    ) -> list[str]:
        """Get transform prompts for each row in the dataset."""
        transform_prompts = []
        for i in range(min(self.num_points_to_transform, len(dataset))):
            row = dataset[i]
            transform_prompt = self.transform_prompt_fn(
                task_explanation, row, self.plan, prompt_spec.examples
            )
            transform_prompts.append(transform_prompt)
        return transform_prompts

    def generate_responses(
        self, transform_prompts_batch: list[str], model_name="gpt-3.5-turbo"
    ) -> list[str]:
        """Generate responses for the given transform prompts.

        Args:
            transform_prompts_batch: A list of transform prompts.
            model_name: The name of the model to use. Defaults to
                    "gpt-3.5-turbo" to save costs.

        Returns:
            A list of generated responses.

        Raises:
            API_ERRORS: If there is an error with the API.

        """
        api_call_counter = 0
        last_error = None
        responses = []
        while True:
            api_call_counter += 1

            async def generate_responses_async(transform_prompts):
                """Generate responses asynchronously using the specified model."""
                responses = await api_tools.APIAgent(
                    model_name=model_name
                ).generate_batch_completion(
                    transform_prompts,
                    temperature=0,
                    responses_per_request=1,
                    requests_per_minute=15,
                )
                return responses

            try:
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(
                    generate_responses_async(transform_prompts_batch)
                )
                break
            except API_ERRORS as e:
                last_error = e
                handle_api_error(e)
            if api_call_counter > self.num_retries:
                # In case we reach maximum number of API calls, we raise an error.
                logger.error("Maximum number of API calls reached.")
                raise RuntimeError(
                    "Maximum number of API calls reached."
                ) from last_error

        return responses

    def process_responses(
        self, responses: list, prompt_spec: PromptSpec
    ) -> tuple[list[str], list[str]]:
        """Process the responses received from the API.

        Args:
            responses: A list of response strings from the API.
            prompt_spec: The PromptSpec object containing the instruction and examples.

        Returns:
            A tuple containing two lists: inputs and outputs.
            - inputs: A list of transformed input strings.
            - outputs: A list of transformed output strings.
        """
        inputs, outputs = [], []
        show_sample_flag = False
        for response in responses:
            try:
                extraction = find_and_parse_json(response, ["input", "output"], [])
                if extraction is not None:
                    if extraction["input"] is None or extraction["output"] is None:
                        raise ValueError("Input or output is None")
                    input = str(extraction["input"]).strip()
                    output = str(extraction["output"]).strip()
                    if input in prompt_spec.examples:
                        raise ValueError("Repeated Task Examples from prompt")

                    inputs.append(input)
                    outputs.append(output)
                    if show_sample_flag:
                        logger.info(f"inputs\n{input}\n\nouputs\n{output}")
                        show_sample_flag = False

            except Exception as e:
                logger.error(f"Error extracting from response: {e}")
                self.curr_failed_transforms += 1
                if self.curr_failed_transforms > self.max_allowed_failed_transforms:
                    break

        return inputs, outputs

    def transform_data(
        self, prompt_spec: PromptSpec, dataset: datasets.Dataset
    ) -> tuple[list[str], list[str]]:
        """Transforms the given dataset based on the provided prompt specification.

        Args:
            prompt_spec: The prompt specification object that defines
                            the transformation rules.
            dataset: The dataset to be transformed.

        Returns:
            A tuple containing two lists: inputs and outputs.
        """
        task_explanation = self.generate_task_explanation(prompt_spec)
        self.plan = self.generate_plan(task_explanation, dataset, prompt_spec)
        logger.info(f"Plan created. Plan: {self.plan}")

        transform_prompts = self.generate_transform_prompts(
            task_explanation, dataset, prompt_spec
        )
        inputs, outputs = [], []
        for batch_indices in range(0, len(transform_prompts), 100):
            transform_prompt_batch = transform_prompts[
                batch_indices : batch_indices + 100
            ]
            responses = self.generate_responses(transform_prompt_batch)
            curr_inputs, curr_outputs = self.process_responses(responses, prompt_spec)
            inputs.extend(curr_inputs)
            outputs.extend(curr_outputs)
            if self.curr_failed_transforms > self.max_allowed_failed_transforms:
                logger.error(
                    f"Exceeded max allowed failed transforms: {self.curr_failed_transforms}"  # noqa: E501
                )
                self.max_allowed_failed_transforms = 0
                break

        logger.info(
            f"Requested length: {self.num_points_to_transform}\nActual length: {len(inputs)}\n"  # noqa: E501
        )
        return inputs, outputs
