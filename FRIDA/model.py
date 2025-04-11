import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
import os
from typing import List
import re

FRIDA_EMB_DIM = 1536
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]


class FridaClassifier(torch.nn.Module):
    def __init__(self):
        super(FridaClassifier, self).__init__()
        self.frida_embedder = T5EncoderModel.from_pretrained("ai-forever/FRIDA")
        self._freeze_embedder_grad()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=FRIDA_EMB_DIM, out_features=500),
            torch.nn.Dropout(p=0.2),
            torch.nn.SELU(),
            torch.nn.Linear(in_features=500, out_features=100),
            torch.nn.Dropout(p=0.1),
            torch.nn.SELU(),
            torch.nn.Linear(in_features=100, out_features=2)
        )

    def _freeze_embedder_grad(self):
        for param in self.frida_embedder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # no gradients calculation for frida embedder
            outputs = self.frida_embedder(input_ids=input_ids, attention_mask=attention_mask)

            embeddings = pool(
                outputs.last_hidden_state,
                attention_mask,
                pooling_method="cls"  # or try "mean"
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
        out = self.classifier(embeddings)

        return out


# return model and tokenizer
def load_model(head_path: str):
    if not os.path.isfile(head_path):
        raise Exception(f'no model weights with path - {head_path}')
    loaded_model = FridaClassifier()
    loaded_model.classifier.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True))
    loaded_model.eval()
    loaded_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")

    return loaded_model, tokenizer


def infer(model: FridaClassifier, tokenizer: AutoTokenizer, texts: List[str], device):
    with torch.no_grad():
        model.eval()
        texts = ["categorize_sentiment: " + text for text in texts]
        tokenized_data = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        input_ids, attention_masks = tokenized_data['input_ids'].type(torch.LongTensor).to(device), tokenized_data[
            'attention_mask'].type(torch.LongTensor).to(device)
        logits_tensor = model(input_ids, attention_masks)
        sft_max = torch.nn.Softmax(dim=-1)
        pred_probs = sft_max(logits_tensor)

        return pred_probs


labels = {0: 'non-toxic', 1: 'toxic'}


#print('loading model and tokenizer...')
#chkp_dir = './'    # CHANGE ON YOUR DIR WITH HEAD WEIGHTS!
#model, tokenizer = load_model(os.path.join(chkp_dir, "classifier_head.pth"))
#print('loaded.')


from typing import List
from pydantic import BaseModel

# Define DTOs
class ToxicityPrediction(BaseModel):
    text: str
    label: str
    toxicity_rate: float


class ToxicityPredictionResponse(BaseModel):
    predictions: List[ToxicityPrediction]


def generate_resp(texts: List[str],model, tokenizer):
    probs = infer(model, tokenizer, texts, device)
    probs_arr = probs.to('cpu').numpy()
    predictions = torch.argmax(probs, dim=-1).int().to('cpu').numpy()
    predicted_labels = [labels[label] for label in predictions]

    predictions_list = [
        ToxicityPrediction(
            text=texts[i],
            label=predicted_labels[i],
            toxicity_rate=float(probs_arr[i][1])  # Ensure float type
        )
        for i in range(len(texts))
    ]

    return ToxicityPredictionResponse(predictions=predictions_list)