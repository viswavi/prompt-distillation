"""An interface for dataset transformation."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod

import datasets

from prompt2model.prompt_parser import PromptSpec


class DatasetTransformer(ABC):
    """A class for transforming a given dataset to a desired format."""

    @abstractmethod
    def transform_data(
        self,
        prompt_spec: PromptSpec,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Transform a split of data.

        Args:
            prompt_spec: A prompt spec (containing a system description).
            dataset: A dataset split.
            num_points_to_transform: Number of data points you wish to
            transform. Number must be greater than zero. If number is greater
            than size of dataset, whole dataset will be transformed. Ignored
            if data_transform is False.

        Returns:
            A single dataset split.
        """
