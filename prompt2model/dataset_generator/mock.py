"""A class for generating empty datasets (for testing purposes)."""

import datasets

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptSpec


class MockDatasetGenerator(DatasetGenerator):
    """A class for generating empty datasets (for testing purposes)."""

    def generate_dataset(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> datasets.Dataset:
        """Create empty versions of the datasets, for testing.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.

        Returns:
            A single dataset split.
        """
        _ = prompt_spec, split  # suppress unused variable warnings
        col_values = [""] * num_examples
        # Construct empty-valued dataframe with length matching num_examples.
        return datasets.Dataset.from_dict(
            {"input_col": col_values, "output_col": col_values}
        )
