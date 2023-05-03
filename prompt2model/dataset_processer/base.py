"""An interface for dataset processer."""

from abc import ABC, abstractmethod

import datasets


# pylint: disable=too-few-public-methods
class BasePrcesser(ABC):
    """A class for post-processing datasets."""

    @abstractmethod
    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """Post-process a list of DatasetDicts.

        Args:
            instruction: The instruction to convert example into a text2text fashion.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
        """
