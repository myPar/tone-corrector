from model_wrapper.bert_wrapper import BertWrapper
from model_wrapper.fasttext_wrapper import FasttextWrapper
from model_wrapper.frida_wrapper import FridaWrapper

from typing import Any

class ModelWrapper(object):
    def __init__(self) -> None:
        self.models_dict: dict[str, Any] = {
            "fasttext": FasttextWrapper(),
            "ru-BERT": BertWrapper(),
            "FRIDA": FridaWrapper(),
        }

    def __call__(self, text: str, model_name: str) -> str:
        return self.models_dict[model_name](text)
