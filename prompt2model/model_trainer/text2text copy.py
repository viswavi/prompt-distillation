from typing import Any

import torch
import nlp
import datasets
import transformers
from datasets import concatenate_datasets
from transformers import Trainer, TrainingArguments, T5Tokenizer

from prompt2model.model_trainer import ModelTrainer
from datasets import DatasetDict

from functools import partial


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_col'], pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['output_col'], pad_to_max_length=True, max_length=16)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings


class TextToTextTrainer(ModelTrainer):
    """This is a simple trainer for a T5 based text2text model."""

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a text2text T5 based model.

        Args:
            training_datasets: Training `Dataset`s, with `input_col` and `output_col`.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A trained HuggingFace model.
        """
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=hyperparameter_choices.get("num_train_epochs", 1),
            per_device_train_batch_size=hyperparameter_choices.get("batch_size", 1),
            warmup_steps=hyperparameter_choices.get("warmup_steps", 0),
            weight_decay=hyperparameter_choices.get("weight_decay", 0.01),
            logging_dir="./logs",
            logging_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
        )

        # Concatenate and preprocess the training datasets
        training_dataset = concatenate_datasets(training_datasets)
        shuffled_dataset = training_dataset.shuffle(seed=42)
        preprocessed_dataset = preprocess_dataset(shuffled_dataset, self.tokenizer)

        def data_collator(batch, tokenizer):
            inputs = [example["input_ids"] for example in batch]
            attention_masks = [example["attention_mask"] for example in batch]

            return{
                "input_ids": inputs,
                "attention_mask": attention_masks,
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=preprocessed_dataset,
            data_collator=partial(data_collator, tokenizer=self.tokenizer),
        )

        # Train the model
        trainer.train()

        # Return the trained model and tokenizer
        return self.model, self.tokenizer
