"""An interface for creating Gradio demos automatically from a trained model
and a prompt specification.
"""

import gradio as gr
import transformers

from prompt_parser.base import PromptSpec


def create_gradio(
    model: transformers.PreTrainedModel, prompt_spec: PromptSpec
) -> gr.Interface:
    """
    Create a Gradio interface from a trained model and a prompt specification.
    """
    _ = model, prompt_spec  # suppress unused variable warnings
    dummy_interface = gr.Interface(lambda input: None, "textbox", "label")
    return dummy_interface
