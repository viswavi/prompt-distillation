"""An interface for trainers."""

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers
from transformers import AutoModel, AutoTokenizer


# pylint: disable=too-few-public-methods
class Model_Trainer(ABC):
    """Train a model with a fixed set of hyperparameters."""

    def __init__(self, pretrained_model_name: str):
        """Initialize a model trainer.

        Args:
            pretrained_model_name: A HuggingFace model name to use for training.
        """
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.wandb = None

    @abstractmethod
    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a model with the given hyperparameters and return it."""
