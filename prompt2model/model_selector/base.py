"""An interface for model selection."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers

from prompt2model.prompt_parser.base import PromptSpec
from prompt2model.trainer import Trainer


# pylint: disable=too-few-public-methods
class ModelSelector(ABC):
    """Select a good model from among a set of hyperparameter choices."""

    @abstractmethod
    def select_model(
        self,
        trainer: Trainer,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
        prompt_spec: PromptSpec,
        hyperparameters: dict[str, list[Any]] | None = None,
    ) -> transformers.PreTrainedModel:
        """Select a model among a set of hyperparameters (given or inferred).

        Args:
            trainer: A trainer object.
            training_sets: One or more training datasets for the trainer.
            validation: A dataset for computing validation metrics.
            prompt_spec: (Optional) A prompt to infer hyperparameters from.
            hyperparameters: (Optional) A dictionary of hyperparameter choices.

        Return:
            A model (with hyperparameters selected from the specified range).

        """
