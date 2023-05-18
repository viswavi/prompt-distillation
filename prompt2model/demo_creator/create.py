"""An interface for creating Gradio demos automatically."""

import gradio as gr

from prompt2model.model_executor import MockModelExecutor
from prompt2model.prompt_parser import PromptSpec


def gradio_create(
    model_executor: MockModelExecutor, prompt_spec: PromptSpec
) -> gr.Interface:
    """Create a Gradio interface automatically.

    Args:
        model: A trained model to expose via a Gradio interface.
        prompt_spec: A PromptSpec to help choose the visual interface.

    Returns:
        A Gradio interface for interacting with the model.

    """
    _ = model_executor, prompt_spec  # suppress unused variable warnings
    dummy_interface = gr.Interface(lambda input: None, "textbox", "label")
    return dummy_interface
