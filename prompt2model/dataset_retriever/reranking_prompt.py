"""This module contains the functions to generate the prompt for dataset reranking."""
from __future__ import annotations  # noqa FI58

METAPROMPT_BASE_DATASET = """Your objective is to choose the most relevant dataset for a given a task (and few examples of the task). For each dataset, you will be provided with the dataset description, and tags related to the dataset which provide meta-information about the dataset. Please return the most relevant dataset, e.g. squad """  # noqa: E501

METAPROMPT_BASE_CONFIG = """Your objective is to choose the most relevant config of a dataset for a given a task (and few examples of the task). A config of a dataset is a version of that dataset. You will be provided information about this dataset, followed by information about its configs. For each config, you will be provided with the config name, and columns and rows of that config. The columns of the config could be useful at understanding whether this config is relevant to the given task. Another relevant factor is the config_name, this would give information on a high level about what each config represents. Please return the most relevant config"""  # noqa: E501

INPUT_PROMPT_DATASET_TEMPLATE = """The following is the task \n {instruction} \n and these are some examples of the same: \n{examples} \n
There are {num} datasets available for this task. \n
{datasets}.
The name of the most relevant dataset for this task is:"""  # noqa: E501

INPUT_PROMPT_CONFIG_TEMPLATE = """ The following is the task: \n {instruction} \n
The following is the dataset selected: {dataset_name}: {dataset_description}
There are {num} configs available in this dataset for this task. \n
{configs}
The name of the most relevant config from these for this task is:
"""

DATASET_TEMPLATE = """[{counter}] **{dataset_name}**:\nDescription-{dataset_description}.\nThis dataset has the following tags:\n {tags} """  # noqa: E501
CONFIG_TEMPLATE = """\t[{counter}] **{config_name}**\n: The columns in this config are {dataset_columns}.\n An example row from this config is {sample_row}.\n """  # noqa: E501


def build_datasets_prompt(instruction: str, examples: str, datasets_infos: dict):
    """Builds the prompt for dataset reranking.

    Args:
        instruction (str): Task instructions
        examples (str): Task Examples
        datasets_infos (dict): A dictionary containing information about all datasets.

    Returns:
        str: The input prompt for dataset retrieval.
    """
    dataset_string = ""
    for i, (dataset_name, dataset_info) in enumerate(datasets_infos.items(), start=1):
        dataset_string += f"""{DATASET_TEMPLATE.format(
                                                    counter = i,
                                                    dataset_name=dataset_name,
                                                    dataset_description=dataset_info["description"],
                                                    tags = dataset_info["tags"]
                                                  )}\n\n"""

    input_prompt = INPUT_PROMPT_DATASET_TEMPLATE.format(
        instruction=instruction,
        examples=examples,
        datasets=dataset_string,
        num=len(datasets_infos),
    )
    return input_prompt


def build_configs_prompt(instruction: str, examples: str, dataset_info: dict):
    """Builds the prompt for config reranking.

    Args:
        instruction (str): Task instructions
        examples (str): Task Examples
        datasets_infos (dict): A dictionary containing information about
            the specific dataset, which includes config information.

    Returns:
        str: The input prompt for dataset retrieval.
    """
    configs_string = ""
    for j, (config_name, config_info) in enumerate(dataset_info["configs"].items()):
        configs_string += f"""{CONFIG_TEMPLATE.format(
                                            counter = chr(ord('a')+j),
                                            config_name = config_name,
                                            dataset_columns = config_info["columns"],
                                            sample_row = config_info["sample_row"]
                                            )}\n"""  # noqa: E501

    input_prompt = INPUT_PROMPT_CONFIG_TEMPLATE.format(
        instruction=instruction,
        examples=examples,
        dataset_name=dataset_info["dataset_name"],
        dataset_description=dataset_info["description"],
        configs=configs_string,
        num=len(dataset_info["configs"]),
    )
    return input_prompt


def construct_prompt_for_dataset_reranking(
    instruction: str,
    examples: str,
    datasets_infos: dict,
    is_config: bool = False,
):
    """Generate the full prompt for dataset reranking based on the given parameters.

    Args:
        instruction (str): Instruction of the task.
        examples (str): Examples of the task.
        datasets_infos (dict): Dictionary with dataset/config information. Each
                               dataset_info object also has a configs object
                               representing the various configs of that dataset
        is_config (bool): bool: Whether the prompt is for dataset
                                reranking or config reranking

    Returns:
        str: Builds a comprehensive prompt for dataset reranking. This prompt includes
             the base instructions, incontext example and the prompt returned by the
             build_input function.
    """
    if is_config:
        metaprompt_base = METAPROMPT_BASE_CONFIG
        input_prompt = build_configs_prompt(instruction, examples, datasets_infos)
    else:
        metaprompt_base = METAPROMPT_BASE_DATASET
        input_prompt = build_datasets_prompt(instruction, examples, datasets_infos)

    prompt_sections = [metaprompt_base]
    all_prompts = "\n\n------\n\n".join(prompt_sections) + "\n\n------\n\n"
    all_prompts += input_prompt

    return all_prompts
