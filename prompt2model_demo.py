"""An commend line demo to run the whole system."""

import json
import logging
import os
import time
from pathlib import Path

import datasets
import pyfiglet
import torch
import transformers
import yaml
from datasets import concatenate_datasets, load_from_disk
from termcolor import colored

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.prompt_based import PromptBasedDatasetGenerator
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from prompt2model.dataset_retriever import DescriptionDatasetRetriever
from prompt2model.demo_creator import create_gradio
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.model_retriever import DescriptionModelRetriever
from prompt2model.model_trainer.generate import GenerationModelTrainer
from prompt2model.param_selector import OptunaParamSelector
from prompt2model.prompt_parser import (
    MockPromptSpec,
    PromptBasedInstructionParser,
    TaskType,
)
from prompt2model.utils.config import DEFAULT_HYPERPARAMETERS_SPACE
from prompt2model.utils.logging_utils import get_formatted_logger


def line_print(input_str: str) -> None:
    """Print the given input string surrounded by horizontal lines.

    Args:
        input_str: The string to be printed.
    """
    print(f"{input_str}")


def print_logo():
    """Print the logo of Prompt2Model."""
    figlet = pyfiglet.Figlet(width=200)
    # Create ASCII art for each word and split into lines
    words = ["Prompt", "2", "Model"]
    colors = ["red", "green", "blue"]
    ascii_art_parts = [figlet.renderText(word).split("\n") for word in words]

    # Calculate the maximum height among the words
    max_height = max(len(part) for part in ascii_art_parts)

    # Equalize the height by adding empty lines at the bottom
    for part in ascii_art_parts:
        while len(part) < max_height:
            part.append("")

    # Zip the lines together, color them, and join them with a space
    ascii_art_lines = []
    for lines in zip(*ascii_art_parts):
        colored_line = " ".join(
            colored(line, color) for line, color in zip(lines, colors)
        )
        ascii_art_lines.append(colored_line)

    # Join the lines together to get the ASCII art
    ascii_art = "\n".join(ascii_art_lines)

    # Get the width of the terminal
    term_width = os.get_terminal_size().columns

    # Center the ASCII art
    centered_ascii_art = "\n".join(
        line.center(term_width) for line in ascii_art.split("\n")
    )

    line_print(centered_ascii_art)


def parse_model_size_limit(line: str, default_size=3e9) -> float:
    """Parse the user input for the maximum size of the model.

    Args:
        line: The user input.
        default_size: The default size to use if the user does not specify a size.
    """
    if len(line.strip()) == 0:
        return default_size
    model_units = {"B": 1e0, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "PB": 1e15}
    unit_disambiguations = {
        "KB": ["Kb", "kb", "kilobytes"],
        "MB": ["Mb", "mb", "megabytes"],
        "GB": ["Gb", "gb", "gigabytes"],
        "TB": ["Tb", "tb", "terabytes"],
        "PB": ["Pb", "pb", "petabytes"],
        "B": ["b", "bytes"],
    }
    unit_matched = False
    for unit, disambiguations in unit_disambiguations.items():
        for unit_name in [unit] + disambiguations:
            if line.strip().endswith(unit_name):
                unit_matched = True
                break
        if unit_matched:
            break
    if unit_matched:
        numerical_part = line.strip()[: -len(unit_name)].strip()
    else:
        numerical_part = line.strip()

    if not str.isdecimal(numerical_part):
        raise ValueError(
            "Invalid input. Please enter a number (integer " + "or number with units)."
        )
    scale_factor = model_units[unit] if unit_matched else 1
    return int(numerical_part) * scale_factor


def main():
    """The main function running the whole system."""
    print_logo()
    # Save the status of Prompt2Model for this session,
    # in case the user wishes to stop and continue later.
    if os.path.isfile("status.yaml"):
        with open("status.yaml", "r") as f:
            status = yaml.safe_load(f)
    else:
        status = {}

    while True:
        line_print("Do you want to start from scratch? (y/n)")
        answer = input()
        if answer.lower() == "n":
            if os.path.isfile("status.yaml"):
                with open("status.yaml", "r") as f:
                    status = yaml.safe_load(f)
                    print(f"Current status:\n{json.dumps(status, indent=4)}")
                    break
            else:
                status = {}
                break
        elif answer.lower() == "y":
            status = {}
            break
        else:
            continue

    propmt_has_been_parsed = status.get("prompt_has_been_parsed", False)
    dataset_has_been_retrieved = status.get("dataset_has_been_retrieved", False)
    model_has_been_retrieved = status.get("model_has_been_retrieved", False)
    dataset_has_been_generated = status.get("dataset_has_been_generated", False)
    model_has_been_trained = status.get("model_has_been_trained", False)
    if not propmt_has_been_parsed:
        prompt = ""
        line_print(
            "Enter your task description and few-shot examples (or 'done' to finish):"
        )
        time.sleep(2)
        while True:
            line = input()
            if line == "done":
                break
            prompt += line + "\n"
        line_print("Parsing prompt...")
        prompt_spec = PromptBasedInstructionParser(task_type=TaskType.TEXT_GENERATION)
        prompt_spec.parse_from_prompt(prompt)

        propmt_has_been_parsed = True
        status["instruction"] = prompt_spec.instruction
        status["examples"] = prompt_spec.examples
        status["prompt_has_been_parsed"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        line_print("Prompt parsed.")

    if propmt_has_been_parsed and not dataset_has_been_retrieved:
        retriever_logger = get_formatted_logger("DescriptionDatasetRetriever")
        retriever_logger.setLevel(logging.INFO)
        prompt_spec = MockPromptSpec(
            TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
        )
        line_print("Retrieving dataset...")
        line_print("Do you want to perform data transformation? (y/n)")
        line_print(
            "Data transformation converts retrieved data into the desired format as per the prompt."  # noqa E501
        )
        auto_transform_data = False
        while True:
            line = input()
            if line.lower() == "y":
                auto_transform_data = True
                break
            elif line.lower() == "n":
                auto_transform_data = False
                break
            else:
                line_print("Invalid input. Please enter y or n.")

        if auto_transform_data:
            while True:
                line_print(
                    "Enter the number of data points you want to transform (the remaining data points in the dataset will be discarded):"  # noqa E501
                )
                line = input()
                try:
                    total_num_points_to_transform = int(line)
                except ValueError:
                    line_print("Invalid input. Please enter a number.")
                    continue
                if total_num_points_to_transform <= 0:
                    line_print("Invalid input. Please enter a number greater than 0.")
                    continue
                status["num_transform"] = total_num_points_to_transform
                break
        retriever = DescriptionDatasetRetriever(
            auto_transform_data=auto_transform_data,
            total_num_points_to_transform=total_num_points_to_transform,
        )
        retrieved_dataset_dict = retriever.retrieve_dataset_dict(prompt_spec)

        dataset_has_been_retrieved = True
        if retrieved_dataset_dict is not None:
            retrieved_dataset_dict.save_to_disk("retrieved_dataset_dict")
            status["retrieved_dataset_dict_root"] = "retrieved_dataset_dict"
        else:
            status["retrieved_dataset_dict_root"] = None
        status["dataset_has_been_retrieved"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and not model_has_been_retrieved
    ):
        line_print(
            "Enter the maximum size of the model (by default, enter nothing "
            + "and we will use 3GB as the limit). You can specify a unit (e.g. "
            + "3GB, 300Mb). If no unit is given, we assume the size is given in bytes."
        )
        max_size_line = input()
        max_size = parse_model_size_limit(max_size_line)
        line_print(f"Maximum model size set to {max_size} bytes.")

        line_print("Retrieving model...")
        prompt_spec = MockPromptSpec(
            TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
        )
        retriever = DescriptionModelRetriever(
            model_descriptions_index_path="huggingface_data/huggingface_models/model_info/",  # noqa E501
            use_bm25=True,
            use_HyDE=True,
            model_size_limit_bytes=max_size,
        )
        top_model_name = retriever.retrieve(prompt_spec)
        line_print("Here are the models we retrieved.")
        for idx, each in enumerate(top_model_name):
            line_print(f"# {idx + 1}: {each}")
        while True:
            line_print(
                "Enter the number of the model you want to use. Range from 1 to 5."
            )
            line = input()
            try:
                rank = int(line)
                assert 1 <= rank <= 5
                break
            except Exception:
                line_print("Invalid input. Please enter a number.")
        model_has_been_retrieved = True
        status["model_has_been_retrieved"] = True
        status["model_name"] = top_model_name[rank - 1]
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and model_has_been_retrieved
        and not dataset_has_been_generated
    ):
        prompt_spec = MockPromptSpec(
            TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
        )
        generator_logger = get_formatted_logger("DatasetGenerator")
        generator_logger.setLevel(logging.INFO)
        line_print("The dataset generation has not finished.")
        time.sleep(2)
        line_print(f"Your input instruction:\n\n{prompt_spec.instruction}")
        time.sleep(2)
        line_print(f"Your input few-shot examples:\n\n{prompt_spec.examples}")
        time.sleep(2)
        while True:
            line_print("Enter the number of examples you wish to generate:")
            line = input()
            try:
                num_expected = int(line)
                break
            except ValueError:
                line_print("Invalid input. Please enter a number.")
        while True:
            line_print("Enter the initial temperature:")
            line = input()
            try:
                initial_temperature = float(line)
                assert 0 <= initial_temperature <= 2.0
                break
            except Exception:
                line_print(
                    "Invalid initial temperature. Please enter a number (float) between 0 and 2."  # noqa E501
                )
        while True:
            line_print("Enter the max temperature (we suggest 1.4):")
            line = input()
            try:
                max_temperature = float(line)
                assert 0 <= max_temperature <= 2.0
                break
            except Exception:
                line_print(
                    "Invalid max temperature. Please enter a float between 0 and 2."
                )
        line_print("Starting to generate dataset. This may take a while...")
        time.sleep(2)
        unlimited_dataset_generator = PromptBasedDatasetGenerator(
            initial_temperature=initial_temperature,
            max_temperature=max_temperature,
            responses_per_request=3,
        )
        generated_dataset = unlimited_dataset_generator.generate_dataset_split(
            prompt_spec, num_expected, split=DatasetSplit.TRAIN
        )
        generated_dataset.save_to_disk("generated_dataset")
        dataset_has_been_generated = True
        status["dataset_has_been_generated"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        line_print("The generated dataset is ready.")
        time.sleep(2)

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and model_has_been_retrieved
        and dataset_has_been_generated
        and not model_has_been_trained
    ):
        line_print("The model has not been trained.")
        time.sleep(2)
        dataset_root = Path("generated_dataset")
        if not dataset_root.exists():
            raise ValueError("Dataset has not been generated yet.")
        trained_model_root = Path("result/trained_model")
        trained_tokenizer_root = Path("result/trained_tokenizer")
        RESULT_PATH = Path("result/result")
        trained_model_root.mkdir(parents=True, exist_ok=True)
        trained_tokenizer_root.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.mkdir(parents=True, exist_ok=True)
        dataset = load_from_disk(dataset_root)
        if status["retrieved_dataset_dict_root"] is not None:
            cached_retrieved_dataset_dict = datasets.load_from_disk(
                status["retrieved_dataset_dict_root"]
            )
            dataset_list = [dataset, cached_retrieved_dataset_dict["train"]]
        else:
            dataset_list = [dataset]

        line_print("Processing datasets.")
        instruction = status["instruction"]
        t5_processor = TextualizeProcessor(has_encoder=True)
        t5_modified_dataset_dicts = t5_processor.process_dataset_lists(
            instruction,
            dataset_list,
            train_proportion=0.7,
            val_proportion=0.1,
            maximum_example_num={"train": 3500, "val": 500, "test": 1000},
        )
        processor_logger = get_formatted_logger("DatasetProcessor")
        processor_logger.setLevel(logging.INFO)
        training_datasets = []
        validation_datasets = []
        test_datasets = []
        for idx, modified_dataset_dict in enumerate(t5_modified_dataset_dicts):
            training_datasets.append(modified_dataset_dict["train"])
            validation_datasets.append(modified_dataset_dict["val"])
            test_datasets.append(modified_dataset_dict["test"])
        trainer_logger = get_formatted_logger("ModelTrainer")
        trainer_logger.setLevel(logging.INFO)
        evaluator_logger = get_formatted_logger("ModelEvaluator")
        evaluator_logger.setLevel(logging.INFO)

        train_batch_size = None

        while True:
            line = input(
                "Are you interested to train the model with automatic hyperparameter search? Type 'y' for Yes and 'n' for No. "  # noqa E501
            )
            try:
                assert line in ["y", "n"]
                break
            except Exception:
                line_print("The answer should be either y or n")
        time.sleep(1)

        if line == "y":
            line_print("Starting training with hyperparameter selection.")
            default_min_num_epochs = DEFAULT_HYPERPARAMETERS_SPACE[
                "min_num_train_epochs"
            ]
            min_num_epochs = input(
                f"Enter min number of epochs. Press enter to use default value ({default_min_num_epochs}): "  # noqa E501
            )
            default_max_num_epochs = DEFAULT_HYPERPARAMETERS_SPACE[
                "max_num_train_epochs"
            ]
            max_num_epochs = input(
                f"Enter max number of epochs. Press enter to use default value ({default_max_num_epochs}): "  # noqa E501
            )
            default_num_trials = 10
            num_trials = input(
                f"Enter the number of trials (maximum number of hyperparameter configurations to consider) for hyperparameter search. Press enter to use default value ({default_num_trials}): "  # noqa E501
            )
            default_batch_size = DEFAULT_HYPERPARAMETERS_SPACE[
                "per_device_train_batch_size"
            ]  # noqa E501
            max_batch_size = input(
                "Enter the max batch size. "
                + f"Press enter to use default ({default_batch_size}): "
            )

            min_num_epochs = (
                default_min_num_epochs if min_num_epochs == "" else eval(min_num_epochs)
            )
            max_num_epochs = (
                default_max_num_epochs if max_num_epochs == "" else eval(max_num_epochs)
            )
            num_trials = 1 if num_trials == "" else eval(num_trials)

            max_batch_size = (
                DEFAULT_HYPERPARAMETERS_SPACE["per_device_train_batch_size"]
                if max_batch_size == ""
                else eval(max_batch_size)
            )

            trainer = GenerationModelTrainer(
                status["model_name"],
                has_encoder=True,
                executor_batch_size=max_batch_size,
                tokenizer_max_length=1024,
                sequence_max_length=1280,
            )
            args_output_root = Path("result/training_output")
            args_output_root.mkdir(parents=True, exist_ok=True)
            line_print("Starting training.")

            trained_model, trained_tokenizer = OptunaParamSelector(
                n_trials=num_trials,
                trainer=trainer,
            ).select_from_hyperparameters(
                training_datasets=training_datasets,
                validation=validation_datasets,
                hyperparameters={
                    "min_num_train_epochs": min_num_epochs,
                    "max_num_train_epochs": max_num_epochs,
                    "per_device_train_batch_size": [max_batch_size],
                },
            )
            train_batch_size = max_batch_size

        else:
            line_print("Starting training without hyperparameter selection.")
            while True:
                line = input("Enter the training batch size:")
                try:
                    train_batch_size = int(line)
                    assert 0 < train_batch_size
                    break
                except Exception:
                    line_print("The training batch size must be greater than 0.")
            time.sleep(1)

            while True:
                line = input("Enter the number of epochs to train for:")
                try:
                    num_epochs = int(line)
                    break
                except ValueError:
                    line_print("Invalid input. Please enter a number.")
            time.sleep(1)

            trainer = GenerationModelTrainer(
                status["model_name"],
                has_encoder=True,
                executor_batch_size=train_batch_size,
                tokenizer_max_length=1024,
                sequence_max_length=1280,
            )
            args_output_root = Path("result/training_output")
            args_output_root.mkdir(parents=True, exist_ok=True)
            line_print("Starting training.")
            trained_model, trained_tokenizer = trainer.train_model(
                hyperparameter_choices={
                    "output_dir": str(args_output_root),
                    "save_strategy": "epoch",
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": train_batch_size,
                    "evaluation_strategy": "epoch",
                },
                training_datasets=training_datasets,
                validation_datasets=validation_datasets,
            )

        trained_model.save_pretrained(trained_model_root)
        trained_tokenizer.save_pretrained(trained_tokenizer_root)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            trained_model_root
        ).to(device)
        trained_tokenizer = transformers.AutoTokenizer.from_pretrained(
            trained_tokenizer_root
        )
        line_print("Finished training. Now evaluating on the test set.")
        test_dataset = concatenate_datasets(test_datasets)

        model_executor = GenerationModelExecutor(
            trained_model,
            trained_tokenizer,
            train_batch_size,
            tokenizer_max_length=1024,
            sequence_max_length=1280,
        )
        t5_outputs = model_executor.make_prediction(
            test_set=test_dataset, input_column="model_input"
        )
        evaluator = Seq2SeqEvaluator()
        metric_values = evaluator.evaluate_model(
            test_dataset,
            "model_output",
            t5_outputs,
            encoder_model_name="xlm-roberta-base",
        )
        line_print(metric_values)
        with open(RESULT_PATH / "metric.txt", "w") as result_file:
            for metric_name, metric_value in metric_values.items():
                result_file.write(f"{metric_name}: {metric_value}\n")
        status["model_has_been_trained"] = model_has_been_trained = True
        status["trained_model_root"] = str(trained_model_root)
        status["trained_tokenizer_root"] = str(trained_tokenizer_root)
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        line_print("Model has been trained and evaluated.")

    t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        status["trained_model_root"]
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(
        status["trained_tokenizer_root"]
    )
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, 1, tokenizer_max_length=1024, sequence_max_length=1280
    )
    prompt_spec = MockPromptSpec(
        TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
    )
    interface_t5 = create_gradio(model_executor, prompt_spec)
    interface_t5.launch(share=True)


if __name__ == "__main__":
    main()
