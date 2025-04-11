import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast


class BertWrapper(object):
    MODELS_DIR: str = "new_models/"
    MODEL_NAME: str = "model"
    TOKENIZER: str = "tokenizer"

    def __init__(self) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BertWrapper.MODELS_DIR + BertWrapper.MODEL_NAME, torchscript=True
        )
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "blanchefort/rubert-base-cased-sentiment"
        )
        self.id2label: dict[int, str] = {0: "__label__positive", 1: "__label__negative"}

    @torch.no_grad()
    def __call__(self, text: str) -> str:
        max_input_length = (
            self.model.config.max_position_embeddings
        )  # 512 for this model
        inputs = self.tokenizer(
            text,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(
            **inputs, return_dict=True
        )  # output is logits for huggingfcae transformers
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_id = torch.argmax(predicted, dim=1).numpy()[0]
        return self.id2label[predicted_id], predicted[0][predicted_id]
