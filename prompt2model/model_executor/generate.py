"""Model executor for generative models, including T5-type and GPT-type."""

import datasets
import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from prompt2model.model_executor import ModelExecutor, ModelOutput


class GenerationModelExecutor(ModelExecutor):
    """Model executor for T5-type and GPT-type models."""

    def make_predictions(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        test_set: datasets.Dataset,
        input_column: str,
    ) -> list[ModelOutput]:
        """Evaluate a T5-type or GPT-type model on a test set.

        Args:
            model: The T5-type or GPT-type model to evaluate.
            tokenizer: The model's associated tokenizer.
            test_set: The dataset to make predictions on.
            input_column: The dataset column to use as input to the model.

        Returns:
            A list of model outputs, one for each element in the test set.
        """
        assert input_column == "model_input"

        model_outputs = []
        for example in test_set:
            input_text = example[input_column]
            encoded_input = tokenizer(
                input_text, truncation=True, padding=True, return_tensors="pt"
            )
            if issubclass(model.__class__, transformers.T5ForConditionalGeneration):
                output = model.generate(**encoded_input)
            else:
                output = model.generate(input_ids=encoded_input["input_ids"])
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            logits = output[0].float()
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.mean().item()
            model_output = ModelOutput(
                prediction=decoded_output,
                confidence=confidence,
                auxiliary_info={},
            )
            model_outputs.append(model_output)

        return model_outputs
