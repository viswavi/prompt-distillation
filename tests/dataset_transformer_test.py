"""Tests for dataset_transformer."""

from functools import partial
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from prompt2model.dataset_transformer import PromptBasedDatasetTransformer
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import MockCompletion, mock_batch_api_response_identical_completions

TASK_EXPLANATION = MockCompletion(
    """This is a question answering task. The model is given a context and a question, and it must provide the answer."""  # noqa E501
)
PLAN_RESPONSE = MockCompletion(
    """plan:
1. Combine "context" and "question" into "input".
2. Combine "answer" into "output"."""
)

TRANSFORMED_DATA = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "context: Albany, the state capital, is the sixth-largest city in the State of New York.\nquestion: What is the capital of New York?", "output": "Albany"}',  # noqa E501
)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=TRANSFORMED_DATA,
)
@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[TASK_EXPLANATION, PLAN_RESPONSE],
)
def test_transform_data(mock_batch_completion, mock_one_completion):
    """Test transform_data."""
    dataset_transformer = PromptBasedDatasetTransformer(
        num_points_to_transform=1, max_allowed_failed_transforms=0
    )  # noqa: E501
    prompt_spec = MockPromptSpec(
        TaskType.TEXT_GENERATION,
        instruction="instruction",
        examples="example1\nexample2",
    )
    mock_dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of New York?"],
            "context": [
                "Albany, the state capital, is the sixth-largest city in the State of New York."  # noqa E501
            ],
            "answer": ["Albany"],
        }
    )
    # Create a mock DatasetDict consisting of the same example in each split.
    dataset = DatasetDict(
        {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
    )
    inputs, outputs = dataset_transformer.transform_data(
        prompt_spec=prompt_spec,
        dataset=dataset["train"],
    )
    assert inputs == [
        "context: Albany, the state capital, is the sixth-largest city in the State of New York.\nquestion: What is the capital of New York?"  # noqa E501
    ]  # noqa E501
    assert outputs == ["Albany"]
