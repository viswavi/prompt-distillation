"""Testing a dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import numpy as np
import os
import tempfile
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from prompt2model.dataset_retriever import DescriptionDatasetRetriever, DatasetInfo
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import create_test_search_index, create_test_search_index_class_method

TINY_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"


def test_initialize_dataset_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        # This tiny dataset search index contains 3 datasets.
        assert len(retriever.dataset_infos) == 3


def test_encode_model_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        model_vectors = retriever.encode_dataset_descriptions(f.name)
        assert model_vectors.shape == (3, 128)


def test_canonicalize_dataset_using_columns():
    """Test canonicalizing a dataset with specified column names."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
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
        dataset_splits = DatasetDict(
            {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
        )
        canonicalized_dataset = retriever.canonicalize_dataset_using_columns(
            dataset_splits, ["question", "context"], "answer"
        )
        splits = ["train", "val", "test"]
        assert canonicalized_dataset.keys() == set(splits)

        for split in splits:
            assert len(canonicalized_dataset[split]) == 1
            row = canonicalized_dataset[split][0]
            assert (
                row["input_col"]
                == """question: What is the capital of New York?
context: Albany, the state capital, is the sixth-largest city in the State of New York."""  # noqa E501
            )
            assert row["output_col"] == "Albany"

def mock_choose_dataset(self, top_datasets: list[DatasetInfo]) -> str | None:
    return top_datasets[0].name    

def mock_canonicalize_dataset(self, dataset_name: str) -> DatasetDict:
    assert dataset_name == "squad"
    mock_dataset = Dataset.from_dict(
        {
            "input_col": ["question: What class of animals are pandas.\ncontext: Pandas are mammals."],
            "output_col": ["mammals"],
        }
    )
    # Create a mock DatasetDict consisting of the same example in each split.
    dataset_splits = DatasetDict(
        {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
    )
    return dataset_splits

@patch.object(
    DescriptionDatasetRetriever,
    "encode_dataset_descriptions",
    new=create_test_search_index_class_method,
)
@patch(
    "prompt2model.dataset_retriever.hf_dataset_retriever.encode_text",
    return_value=np.array([[1, 0, 0]]),
)
@patch.object(
    DescriptionDatasetRetriever,
    "choose_dataset",
    new=mock_choose_dataset,
)
@patch.object(
    DescriptionDatasetRetriever,
    "canonicalize_dataset",
    new=mock_canonicalize_dataset,
)
def test_retrieve_dataset_dict_without_existing_search_index(encode_text):
    """Test retrieve dataset without an existing search index."""
    with tempfile.TemporaryDirectory() as tempdir:
        temporary_file = os.path.join(tempdir, "search_index.pkl")
        retriever = DescriptionDatasetRetriever(
            search_index_path=temporary_file,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        retrieved_dataset = retriever.retrieve_dataset_dict(mock_prompt)
        assert encode_text.call_count == 1
