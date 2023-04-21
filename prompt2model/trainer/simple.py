"""This module provides a dummy trainer for testing purposes."""

from pathlib import Path
from typing import Any

import datasets
import transformers
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, TrainingArguments, AutoModel, AutoTokenizer, PreTrainedTokenizer

from prompt2model.trainer import Trainer


class SimpleTrainer(Trainer):
    """This is the simple trainer does not actually train anything."""

    def __init__(self, pretrained_model_id: str):
        """Initialize a simple model trainer.

        Args:
            pretrained_model_id: A HuggingFace model ID to use for training.
        """
        self.model = AutoModel.from_pretrained(pretrained_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
        self.wandb = None

    def set_up_weights_and_biases(self) -> None:
        """Set up Weights & Biases logging."""
        self.wandb = None
        raise NotImplementedError

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> transformers.PreTrainedModel:
        """Train a sequence classification model (TODO(Chenyang): update).

        Args:
            training_datasets: A list of training datasets.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A trained HuggingFace model.
        """
        # Set up checkpointing
        output_dir = Path(hyperparameter_choices.get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = hyperparameter_choices.get("checkpoint_file")
        if checkpoint_file is not None:
            checkpoint_dict = hyperparameter_choices.copy()
            checkpoint_dict["output_dir"] = str(output_dir)
            parser = HfArgumentParser(TrainingArguments)
            checkpoint = parser.parse_dict(checkpoint_dict)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                checkpoint_file
            )
            optimizer = transformers.AdamW(
                model.parameters(), lr=checkpoint.learning_rate
            )
            start_epoch = checkpoint.epoch + 1
            global_step = checkpoint.global_step
        else:
            # Load the tokenizer and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                hyperparameter_choices["model"]
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                hyperparameter_choices["model"]
            )

            # Set up the optimizer
            optimizer = transformers.AdamW(
                model.parameters(), lr=hyperparameter_choices["learning_rate"]
            )

            checkpoint = TrainingArguments(output_dir=output_dir)
            start_epoch = 0
            global_step = 0

        # Train the model
        batch_size = hyperparameter_choices.get("batch_size", 32)
        train_data_loader = DataLoader(
            training_datasets, batch_size=batch_size, shuffle=True
        )

        for epoch in range(start_epoch, checkpoint.num_train_epochs):
            for batch in train_data_loader:
                inputs = tokenizer(
                    batch["input_ids"],
                    batch["attention_mask"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                labels = batch["label"]

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()

                # Save checkpoint every save_steps steps
                if (global_step + 1) % checkpoint.save_steps == 0:
                    checkpoint.epoch = epoch
                    checkpoint.global_step = global_step + 1
                    checkpoint_dir = output_dir / f"checkpoint-{global_step + 1}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    checkpoint.save((checkpoint_dir / "checkpoint").as_posix())

                global_step += 1

        # Save final model if checkpointing was enabled
        if checkpoint_file is None:
            model.save_pretrained(output_dir)

        return model
