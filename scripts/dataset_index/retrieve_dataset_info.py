from __future__ import annotations  # noqa FI58

import argparse
import gc
import json
import threading
import time
from collections.abc import MutableMapping
from typing import Any

import datasets
import requests


def parse_arguments():
    """Parse command line arguments for the script.

    Used for whether we should create the dataset index
    file with processing of each configuration or not

    Returns:
        argparse: Argument Parser which contains 'loads_configs' attribute,
                            indicating if configurations should be loaded (boolean).
    """
    parser = argparse.ArgumentParser(
        description="Dataset Index File configuration settings."
    )
    parser.add_argument(
        "--loads_configs",
        type=bool,
        default=True,
        help="Indicates if configurations are present (default: True)",
    )

    args = parser.parse_args()
    return args


def get_dataset_validity(dataset_name: str, max_retries: int = 5) -> bool:
    """Check if a given dataset name is valid on HuggingFace's dataset server.

    Also added support for exponential backoff.

    Args:
        dataset_name (str): The name of the dataset to check.
        max_retries (int): Maximum number of retries for the request (default is 5).

    Returns:
        bool: True if dataset is valid and can be previewed and viewed, else False.
    """
    api_url = f"https://datasets-server.huggingface.co/is-valid?dataset={dataset_name}"
    retries = 0
    backoff = 10

    while retries < max_retries:
        response = requests.get(api_url)

        if response.status_code == 200:
            response_json = response.json()
            return (
                "preview" in response_json
                and "viewer" in response_json
                and response_json["preview"] & response_json["viewer"]
            )

        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else backoff
            time.sleep(wait)
            backoff *= 2  # Exponential increase
            retries += 1
        else:
            # Handle other HTTP errors
            break

    return False


def replace_duplicate_columns(original_dataset_columns: list) -> tuple[list, dict]:
    """Replace duplicate column names in a dataset after flattening.

    Args:
        original_dataset_columns: List of original column names in the dataset.

    Returns:
        tuple: A tuple containing two elements:
                1. A list of new column names with duplicates handled.
                2. A dictionary mapping original column names to new column names.
    """
    columns_mapping: dict[str, str] = {}
    new_columns = []
    counter: dict[str, int] = {}
    # convert flattened columns like answer.text -> answer_text
    for col in original_dataset_columns:
        new_col = col.replace(".", "_")
        if new_col in columns_mapping.values():
            counter[new_col] = counter.get(new_col, 0) + 1
            new_col = f"{new_col}_{counter[new_col]}"
        columns_mapping[col] = new_col
        new_columns.append(new_col)
    return new_columns, columns_mapping


def fetch_first_row_with_timeout(
    dataset: datasets.Dataset, timeout: int = 30
) -> dict | None:
    """Fetch the first row of a dataset within a specified timeout period.

    Args:
        dataset: The dataset from which to fetch the first row.
        timeout: The maximum time in seconds to wait for
                 fetching the row (default is 30 seconds).

    Returns:
        dict or None: The first row of the dataset as a dictionary,
                      or None if the operation times out.
    """

    def fetch_sample_row(container: list[dict | Exception]):
        try:
            container.append(next(iter(dataset)))
        except Exception as e:
            container.append(e)

    result_container: list[dict | Exception] = []
    fetch_thread = threading.Thread(target=fetch_sample_row, args=(result_container,))
    fetch_thread.start()
    fetch_thread.join(timeout)

    if fetch_thread.is_alive() or result_container[0] is None:
        # Operation took too long or failed
        return None

    return result_container[0] if isinstance(result_container[0], dict) else None


def truncate_row(example_row: dict, max_length=50) -> str:
    """Truncate each value in a row to a specified maximum length.

    Args:
        example_row (dict): A dictionary representing a row from the dataset.
        max_length (int): Maximum length for each value in the row.

    Returns:
        str: A JSON string representation of the truncated row.
    """
    truncated_row = {}
    for key in example_row.keys():
        curr_row = json.dumps(example_row[key])
        truncated_row[key] = (
            curr_row
            if len(curr_row) <= max_length - 3
            else curr_row[:max_length] + "..."
        )
    return json.dumps(truncated_row)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten the sample rows of streaming dataset.

    Streaming Datasets from HF don't inherently have the flatten function.

    Args:
        d (MutableMapping): The dictionary to flatten.
        parent_key (str): The base key string to use for the flattened keys.
        sep (str): Separator used between nested keys (default is '.').

    Returns:
        dict: A flattened dictionary with no nested structures.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_dataset_configs(dataset_name: str) -> dict:
    """Get all configurations for a given dataset.

    For each configuration:
    Skip the config if any of the following pass:
    1. checking if it has a train split
    2. if any of the columns are Image(and hence Video) or DateTime type.
    3. If streaming the dataset is taking too long

    Then load the dataset for that config, and flatten it.
    Args:
        dataset_name: The name of the dataset for which to get configurations.

    Returns:
        dict: A dictionary with configuration names as keys and configuration
              details as values.
    """
    config_names = datasets.get_dataset_config_names(dataset_name)
    all_configs = {}
    for config_name in config_names:
        if "train" not in datasets.get_dataset_split_names(dataset_name, config_name):
            continue
        dataset = datasets.load_dataset(
            dataset_name,
            config_name,
            split="train",
            streaming=True,
            download_mode="force_redownload",
        )
        sample_rows = fetch_first_row_with_timeout(dataset, timeout=30)
        if not sample_rows:
            continue
        sample_rows = flatten_dict(sample_rows)
        if any(
            "ImageFile" in sample_rows[key].__class__.__name__
            or "DateTime" in sample_rows[key].__class__.__name__
            for key in sample_rows
        ):
            continue
        columns, columns_mapping = replace_duplicate_columns(list(sample_rows.keys()))

        columns_str = ", ".join(columns)
        all_configs[config_name] = {
            "config_name": config_name,
            "sample_row": truncate_row(sample_rows),
            "columns": columns_str,
            "columns_mapping": columns_mapping,
            "dataset_description": dataset.info.description,
            "dataset_name": dataset_name,
        }

    return all_configs


def process_datasets(chunk: list, loads_configs: bool):
    """Process through the chunk of datasets and get dataset info to store."""
    dataset_index = {}
    max_attempts = 3
    for z in range(len(chunk)):
        print(f"currently on {z} out of {len(chunk)}")

        attempt = 0
        delay = 10
        while attempt < max_attempts:
            try:
                dataset_info = chunk[z]
                dataset_name = dataset_info["id"]
                description = dataset_info["description"]

                is_gated = hasattr(dataset_info, "gated") and dataset_info["gated"]
                if hasattr(dataset_info, "disabled") and dataset_info["disabled"]:
                    raise ValueError("dataset is disabled")
                if not get_dataset_validity(dataset_name):
                    raise ValueError("dataset is not valid")

                all_configs: dict[str, dict] = {}
                if loads_configs:
                    all_configs = get_dataset_configs(dataset_name)

                dataset_index[dataset_name] = {
                    "dataset_name": dataset_name,
                    "description": description,
                    "downloads": dataset_info["downloads"],
                    "configs": all_configs,
                    "tags": dataset_info["tags"],
                    "is_gated": is_gated,
                }
                print(
                    f"""completed {z} out of {len(chunk)}, dataset is
                    {dataset_name}, and it has {len(all_configs)} configs in it"""
                )
                del all_configs

                break
            except Exception as e:
                if "429 Client Error" in str(e):
                    time.sleep(delay)
                    delay *= 2
                    attempt += 1
                else:
                    print("Error processing +", dataset_info["id"], ": ", e)
                    break
            except SystemExit as e:
                print("Error processing +", dataset_info["id"], ": ", e)
                break
            gc.collect()

    return dataset_index


if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()
    all_datasets_file = "processed_datasets.json"

    with open(all_datasets_file, "r") as f:
        all_datasets = json.load(f)
        process_datasets(all_datasets, loads_configs=args.loads_configs)
    end_time = time.time()
    print(f"Process took {(end_time-start_time)//60} minutes")
