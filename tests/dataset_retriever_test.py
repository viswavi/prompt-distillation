"""Testing a dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import json
import os
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from prompt2model.dataset_retriever import DatasetInfo, DescriptionDatasetRetriever
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import MockCompletion, create_test_search_index

# The following variables are specifically for the
# four automatic column selection and reranking tests.
dataset_info_file = "test_helpers/reranking_dataset_index_tiny.json"
with open(dataset_info_file, "r") as f:
    dataset_info = json.load(f)

SQUAD_DATASET_INFO = dataset_info["squad"]["configs"]["plain_text"]


INSTRUCTION = "Your task is to generate a relevant answer to a given question. You will be provided with context for each question"  # noqa: E501
GPT3_RESPONSE_COL_SELECTION_CORRECT = MockCompletion(
    """{\n        \"input\": [\"context\", \"question\"],\n        \"output\": [\"answers_text\"],\n        \"irrelevant\": [\"id\", \"title\"],\n        \"ambiguous\": []\n}"""  # noqa: E501
)
GPT3_RESPONSE_COL_SELECTION_WITHOUT_REQUIRED_COLS = MockCompletion(
    """{\n       \"output\": [\"answers\"],\n        \"irrelevant\": [\"id\", \"title\"],\n        \"ambiguous\": []\n}"""  # noqa: E501
)
GPT3_RESPONSE_COL_SELECTION_WITH_UNKNOWN_COLS = MockCompletion(
    """{\n   \"input\": [\"comprehension\", \"question\"],\n     \"output\": [\"answers_text\"],\n        \"irrelevant\": [\"id\", \"title\"],\n        \"ambiguous\": []\n}"""  # noqa: E501
)
GPT3_RESPONSE_COL_SELECTION_WITH_MORE_THAN_ONE_OUTPUT = MockCompletion(
    """{\n   \"input\": [\"context\", \"question\"],\n     \"output\": [\"answers\", \"title\"],\n        \"irrelevant\": [\"id\", \"title\"],\n        \"ambiguous\": []\n}"""  # noqa: E501
)

GPT3_RESPONSE_RERANK_DATASETS_CORRECT_DATASET = MockCompletion(
    """**codeparrot/apps**"""
)
GPT3_RESPONSE_RERANK_DATASETS_CORRECT_CONFIG = MockCompletion("""**interview**""")
GPT3_RESPONSE_RERANK_DATASETS_HALLUCINATES_CONFIG = MockCompletion("""not_a_config""")

TINY_DUAL_ENCODER_NAME = "google/bert_uncased_L-2_H-128_A-2"


def test_canonicalize_dataset_using_samples():
    """Test canonicalize_dataset_using_samples."""
    inputs = ["input1", "input2"]
    outputs = ["output1", "output2"]
    dataset_dict = DescriptionDatasetRetriever().make_dataset_from_samples(
        inputs, outputs
    )
    assert dataset_dict["train"].column_names == ["input_col", "output_col"]
    assert dataset_dict["train"].shape == (2, 2)
    assert dataset_dict["train"][0]["input_col"] == "input1"
    assert dataset_dict["train"][0]["output_col"] == "output1"
    assert dataset_dict["train"][1]["input_col"] == "input2"
    assert dataset_dict["train"][1]["output_col"] == "output2"


def test_initialize_dataset_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
            reranking_dataset_info_file="test_helpers/reranking_dataset_index_tiny.json",  # noqa: E501
        )
        # This tiny dataset search index contains 3 datasets.
        assert len(retriever.dataset_infos) == 3


def test_encode_model_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.TemporaryDirectory() as tempdir:
        temporary_file = os.path.join(tempdir, "search_index.pkl")
        retriever = DescriptionDatasetRetriever(
            search_index_path=temporary_file,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
            reranking_dataset_info_file="test_helpers/reranking_dataset_index_tiny.json",  # noqa: E501
        )
        retriever.initialize_search_index()
        with open(temporary_file, "rb") as f:
            model_vectors, _ = pickle.load(f)
        assert model_vectors.shape == (3, 128)


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_COL_SELECTION_CORRECT],
)
def test_automatic_column_selection_correct(mocked_parsing_method):
    """Check that normal automatic column selection runs fine."""
    expected_input_columns = ["context", "question"]
    expected_output_column = "answers_text"
    (
        input_columns,
        output_column,
    ) = DescriptionDatasetRetriever.automatic_column_selection(
        INSTRUCTION,
        SQUAD_DATASET_INFO["dataset_name"],
        SQUAD_DATASET_INFO["dataset_description"],
        SQUAD_DATASET_INFO["columns"],
        SQUAD_DATASET_INFO["sample_row"],
    )

    assert type(input_columns) == list
    assert input_columns == expected_input_columns

    assert type(output_column) == str
    assert output_column == expected_output_column


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_COL_SELECTION_WITH_UNKNOWN_COLS],
)
def test_automatic_column_selection_unknown_cols(mocked_parsing_method):
    """Check error thrown if there are unknown cols returned in input/output."""
    with pytest.raises(RuntimeError) as exc_info:
        _ = DescriptionDatasetRetriever.automatic_column_selection(
            INSTRUCTION,
            SQUAD_DATASET_INFO["dataset_name"],
            SQUAD_DATASET_INFO["dataset_description"],
            SQUAD_DATASET_INFO["columns"],
            SQUAD_DATASET_INFO["sample_row"],
        )
        error_info = exc_info.value.args[0]
        assert error_info == "Incorrect columns being parsed"


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_COL_SELECTION_WITHOUT_REQUIRED_COLS],
)
def test_automatic_column_selection_without_required_cols(mocked_parsing_method):
    """Check that if input/output columns are missing, an error is thrown."""
    with pytest.raises(StopIteration) as exc_info:
        _ = DescriptionDatasetRetriever.automatic_column_selection(
            INSTRUCTION,
            SQUAD_DATASET_INFO["dataset_name"],
            SQUAD_DATASET_INFO["dataset_description"],
            SQUAD_DATASET_INFO["columns"],
            SQUAD_DATASET_INFO["sample_row"],
        )
        error_info = exc_info.value.args[0]
        assert error_info == "Maximum number of API calls reached."


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_COL_SELECTION_WITH_MORE_THAN_ONE_OUTPUT],
)
def test_automatic_column_selection_incorrect_number_of_output_cols(
    mocked_parsing_method,
):
    """Check that if number of input/output columns are wrong, an error is thrown."""
    with pytest.raises(RuntimeError) as exc_info:
        _ = DescriptionDatasetRetriever.automatic_column_selection(
            INSTRUCTION,
            SQUAD_DATASET_INFO["dataset_name"],
            SQUAD_DATASET_INFO["dataset_description"],
            SQUAD_DATASET_INFO["columns"],
            SQUAD_DATASET_INFO["sample_row"],
        )
        error_info = exc_info.value.args[0]
        assert (
            error_info
            == "Input columns length was less than 1 or output column length was not 1"
        )


def test_canonicalize_dataset_using_columns():
    """Test canonicalizing a dataset with specified column names."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
            reranking_dataset_info_file="test_helpers/reranking_dataset_index_tiny.json",  # noqa: E501
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
    """Mock the choose_dataset by always choosing the first choice."""
    # Given the max_search_depth of 3 used below, we should choose among 3 datasets.
    assert len(top_datasets) == 3
    return top_datasets[0].name


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[
        GPT3_RESPONSE_RERANK_DATASETS_CORRECT_DATASET,
        GPT3_RESPONSE_RERANK_DATASETS_CORRECT_DATASET,
        GPT3_RESPONSE_RERANK_DATASETS_CORRECT_CONFIG,
        GPT3_RESPONSE_RERANK_DATASETS_CORRECT_CONFIG,
    ],
)
def test_rerank_datasets_correct(mocked_parsing_method):
    """Test correct working of dataset reranking."""
    prompt_spec = MockPromptSpec(
        task_type=TaskType.TEXT_GENERATION, instruction=INSTRUCTION
    )
    datasets_list = ["squad", "codeparrot/apps"]

    descriptionDatasetRetriever = DescriptionDatasetRetriever(
        reranking_dataset_info_file="test_helpers/reranking_dataset_index_tiny.json",
        num_votes=2,
    )
    dataset_info_dict = descriptionDatasetRetriever.get_all_dataset_infos(datasets_list)
    dataset_name, config_name = descriptionDatasetRetriever.rerank_datasets(
        dataset_info_dict, prompt_spec
    )
    assert dataset_name == "codeparrot/apps"
    assert config_name == "interview"


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[
        GPT3_RESPONSE_RERANK_DATASETS_CORRECT_DATASET,
        GPT3_RESPONSE_RERANK_DATASETS_CORRECT_DATASET,
        GPT3_RESPONSE_RERANK_DATASETS_HALLUCINATES_CONFIG,
        GPT3_RESPONSE_RERANK_DATASETS_HALLUCINATES_CONFIG,
    ],
)
def test_rerank_datasets_hallucinate_config(mocked_parsing_method):
    """Check if dataset reranking returns None if LLM hallucinates config name."""
    prompt_spec = MockPromptSpec(
        task_type=TaskType.TEXT_GENERATION, instruction=INSTRUCTION
    )

    descriptionDatasetRetriever = DescriptionDatasetRetriever(
        reranking_dataset_info_file="test_helpers/reranking_dataset_index_tiny.json",
        num_votes=2,
    )

    datasets_list = ["squad", "codeparrot/apps"]
    dataset_info_dict = descriptionDatasetRetriever.get_all_dataset_infos(datasets_list)

    dataset_name, config_name = descriptionDatasetRetriever.rerank_datasets(
        dataset_info_dict, prompt_spec
    )
    assert dataset_name == "codeparrot/apps"
    assert config_name is None


def mock_rerank_datasets(self, dataset_list, prompt_spec):
    """Mock dataset reranking by just returning squad dataset."""
    return "squad", "plain_text"


def mock_canonicalize_dataset(self, top_dataset_info, prompt_spec) -> DatasetDict:
    """Mock the canonicalize_dataset function by constructing a mock dataset."""
    # Given the dataset of
    # [ [0.9, 0, 0],
    #   [0, 0.9, 0],
    #   [0, 0, 0.9] ]
    # and the query
    # [1, 0, 0]
    # we should return the first dataset, which in our test index is search_qa.
    assert top_dataset_info["dataset_name"] == "squad"
    mock_dataset = Dataset.from_dict(
        {
            "input_col": [
                "question: What class of animals are pandas.\ncontext: Pandas are mammals."  # noqa E501
            ],
            "output_col": ["mammals"],
        }
    )
    # Create a mock DatasetDict consisting of the same example in each split.
    dataset_splits = DatasetDict(
        {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
    )
    return dataset_splits


@patch(
    "prompt2model.dataset_retriever.description_dataset_retriever.encode_text",
    return_value=np.array([[1, 0, 0]]),
)
@patch.object(
    DescriptionDatasetRetriever,
    "rerank_datasets",
    new=mock_rerank_datasets,
)
@patch.object(
    DescriptionDatasetRetriever,
    "canonicalize_dataset_automatically",
    new=mock_canonicalize_dataset,
)
def test_retrieve_dataset_dict_when_search_index_exists(encode_text):
    """Test retrieve dataset without an existing search index."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
            reranking_dataset_info_file="test_helpers/reranking_dataset_index_tiny.json",  # noqa: E501
        )
        assert [info.name for info in retriever.dataset_infos] == [
            "search_qa",
            "squad",
            "trivia_qa",
        ]
        create_test_search_index(f.name)

        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        retrieved_dataset = retriever.retrieve_dataset_dict(mock_prompt)
        assert encode_text.call_count == 1
        for split_name in ["train", "val", "test"]:
            assert split_name in retrieved_dataset
            split = retrieved_dataset[split_name]
            assert len(split) == 1
            assert (
                split[0]["input_col"]
                == "question: What class of animals are pandas.\ncontext: Pandas are mammals."  # noqa E501
            )
            assert split[0]["output_col"] == "mammals"


@patch(
    "prompt2model.dataset_retriever.description_dataset_retriever.encode_text",
    return_value=np.array([[1, 0, 0]]),
)
@patch.object(
    DescriptionDatasetRetriever,
    "rerank_datasets",
    new=mock_rerank_datasets,
)
@patch.object(
    DescriptionDatasetRetriever,
    "canonicalize_dataset_automatically",
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
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
            reranking_dataset_info_file="test_helpers/reranking_dataset_index_tiny.json",  # noqa: E501
        )
        create_test_search_index(temporary_file)
        assert [info.name for info in retriever.dataset_infos] == [
            "search_qa",
            "squad",
            "trivia_qa",
        ]
        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        retrieved_dataset = retriever.retrieve_dataset_dict(mock_prompt)
        assert encode_text.call_count == 2
        for split_name in ["train", "val", "test"]:
            assert split_name in retrieved_dataset
            split = retrieved_dataset[split_name]
            assert len(split) == 1
            assert (
                split[0]["input_col"]
                == "question: What class of animals are pandas.\ncontext: Pandas are mammals."  # noqa E501
            )
            assert split[0]["output_col"] == "mammals"
