"""This module contains the functions to construct the prompt for task expansion."""
METAPROMPT_BASE = "Carefully analyse the  task description and examples of the task, and explain the task to give a clearer description. Do not explain each example, but rather capture the general trends. Also place special focus on the format of the input/output examples."  # noqa: E501

TASK = """
Task Description: {task_description}

Task Examples: {examples}
"""


def construct_prompt_for_task_explanation(instruction: str, demonstrations: str):
    """Constructs prompt for task explanation.

    This is useful for clarifying the requirements of a task,
    and providing a clearer description of the task.

    Args:
        instruction (str): The task instruction.
        demonstrations (str): The task demonstrations.

    Returns:
        str: The constructed prompt.
    """
    task = TASK.format(task_description=instruction, examples=demonstrations)
    prompt = "\n--------\n".join([METAPROMPT_BASE, task])
    return prompt
