import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
import re
import emoji
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from torch.optim import AdamW
import pandas as pd
import re
import emoji
import os
import gc
from torch.utils.data import Dataset, DataLoader

# load dataset
dataset_path = os.environ['MY_DATASET_PATH']
checkpoint_dir = os.environ['CHECKPOINT_DIR']

main_dataset = pd.read_csv(dataset_path)
print(main_dataset.head())

# train test split:
train_df, test_df = train_test_split(main_dataset, random_state=42, test_size=0.2)
test_df, val_df = train_test_split(test_df, random_state=42, test_size=0.5)

# define model:
FRIDA_EMB_DIM = 1536

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
        self.frida_embedder = T5EncoderModel.from_pretrained("ai-forever/FRIDA", local_files_only=True)
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
        with torch.no_grad():   # no gradients calculation for frida embedder and use only eval mode
            self.frida_embedder.eval()
            outputs = self.frida_embedder(input_ids=input_ids, attention_mask=attention_mask)
            
            embeddings = pool(
                outputs.last_hidden_state, 
                attention_mask,
                pooling_method="cls" # or try "mean"
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
        out = self.classifier(embeddings)

        return out
    

# dataset and dataloader
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, full_gpu_load: bool=False):
        """
            * df - dataframe of the dataset
            * full_gpu_load - weather will full dataset will be loaded to gpu. if no - only single batch will be loaded
        """
        super(ToxicDataset).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.full_gpu_load = full_gpu_load
        #data = df.to_numpy()
        texts = df['comment'].to_numpy()
        # add sentiment prefix
        texts = ["categorize_sentiment: " + text for text in texts]
        labels = df['toxic'].to_numpy()
        tokenized_data = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        self.input_ids, self.attention_masks = tokenized_data['input_ids'].type(torch.LongTensor), tokenized_data['attention_mask'].type(torch.LongTensor)
        self.labels = torch.tensor(labels).type(torch.LongTensor)

        if self.full_gpu_load:
            self.input_ids = self.input_ids.to(self.device)
            self.attention_masks = self.attention_masks.to(self.device)
            self.labels = self.labels.to(self.device)

    def __getitem__(self, index):
        input_ids, attention_mask = self.input_ids[index], self.attention_masks[index]
        label = self.labels[index]

        return input_ids, attention_mask, label
    
    def __len__(self):
        return len(self.input_ids)

print('loading tokenizer...')    
tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA", local_files_only=True)
print('loaded')

train_dataset = ToxicDataset(train_df, tokenizer, full_gpu_load=True)
test_dataset = ToxicDataset(test_df, tokenizer, full_gpu_load=True)
val_dataset = ToxicDataset(val_df, tokenizer, full_gpu_load=True)

batch_size = 64
num_workers = 0

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)
print('dataloaders created.')

# plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_epoch_metrics(epoch_idx, train_loss_list, validation_loss_list,
                       train_metrics_list, val_metrics_list, epoch_validation: bool):
    epochs = list(range(1, len(train_loss_list) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))

    # Accuracy Plot
    fig.add_trace(go.Scatter(x=epochs,
                             y=[m['acc'] for m in train_metrics_list],
                             mode='lines+markers',
                             name='Train Accuracy'),
                  row=1, col=1)

    # Loss Plot
    fig.add_trace(go.Scatter(x=epochs,
                             y=train_loss_list,
                             mode='lines+markers',
                             name='Train Loss'),
                  row=1, col=2)
    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_yaxes(title_text="accuracy", row=1, col=1)
    fig.update_yaxes(title_text="loss", row=1, col=2)

    if epoch_validation:
        # same for validation data:
        fig.add_trace(go.Scatter(x=epochs,
                                 y=validation_loss_list,
                                 mode='lines+markers',
                                 name='Test Loss'),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs,
                                 y=[m['acc'] for m in val_metrics_list],
                                 mode='lines+markers',
                                 name='Test Accuracy'),
                      row=1, col=1)        

    fig.update_layout(title_text=f"Epoch {epoch_idx + 1} Metrics",
                      width=1000, height=400)
    return fig


from tqdm import tqdm


def calc_metrics(probs: torch.tensor, labels: torch.tensor):
    assert len(probs.shape) == 2
    assert len(labels.shape) == 1
    assert len(labels) == len(probs)
    assert len(labels) > 0

    predictions = torch.argmax(probs, dim=-1).int()
    labels = labels.int()

    not_matched = torch.sum(torch.abs(predictions - labels))
    acc = (len(labels) - not_matched) / len(labels)
    acc = acc.item()

    tp = torch.sum(predictions & labels).item()
    fn = torch.sum((predictions == 0) & labels).item()
    fp = torch.sum(predictions & (labels == 0)).item()
    if tp != 0:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    else:
        recall = 0
        precision = 0

    assert recall >= 0
    assert precision >= 0
    assert acc >= 0

    return acc, recall, precision


def train_loop(dataloader, model, loss_fn, optimizer, device):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    epoch_loss = 0
    num_batches = len(dataloader)

    def train_step(input_ids, attention_mask, label):
        optimizer.zero_grad()  # zero gradients to prevent accumulated gradient value

        # Compute prediction and loss
        # print(input_ids)
        # print(attention_mask)
        pred = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(pred, label)

        # Backpropagation
        loss.backward()
        optimizer.step()

        return loss.item()
    pbar = tqdm(total=len(dataloader))

    for _, (input_ids, attention_mask, label) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        epoch_loss += train_step(input_ids, attention_mask, label)
        pbar.update(1)


    epoch_loss = epoch_loss / num_batches  # get average loss

    return epoch_loss

# predictions - tensor of probs, actual - tensor of labels: {0, 1}
def predict(dataloader, model, device):
    model.eval()  # inference - set to eval
    predictions = torch.empty(0).to(device)
    actual = torch.empty(0).type(torch.LongTensor).to(device)

    with torch.no_grad():  # we are making predictions, so it's no need to calc gradients
        pbar = tqdm(total=len(dataloader))
        for _, (input_ids, attention_mask, label) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            sft_max = torch.nn.Softmax(dim=-1)
            pred_probs = sft_max(pred)

            predictions = torch.concat((predictions, pred_probs), 0)
            actual = torch.concat((actual, label), 0)
            pbar.update(1)


    return predictions, actual


def test_loop(dataloader, model, loss_fn):
    model.eval()  # inference - set to eval
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():  # we test model, so it's no need to calc gradients
        for _, (input_ids, attention_mask, label) in enumerate(dataloader):
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            test_loss += loss_fn(pred, label).item()

    test_loss = test_loss / num_batches

    return test_loss


def log_likelihood(predicted_probs, labels):
    log_probs = torch.log(predicted_probs)
    loss = torch.nn.NLLLoss()
    return loss(log_probs, labels).item()


def script_model(model, model_dir: str, metric_item):
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    scripted_model = torch.jit.script(model)
    scripted_model.save(os.path.join(model_dir, 'model.pt'))
    print(f'model is saved: acc={metric_item['acc']}; recall={metric_item['rec']}; precision={metric_item['pre']}')


def save_model_head(model, model_dir: str, metric_item):
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    torch.save(model.classifier.state_dict(), os.path.join(model_dir, 'classifier_head.pth'))
    print(f'model is saved: acc={metric_item['acc']}; recall={metric_item['rec']}; precision={metric_item['pre']}')


def train_model(epoch_count, model, optimizer, loss_function,
                train_loader, test_loader, epoch_validation: bool, 
                train_loss_threshold: float, device:str, chkp_dir:str='models/'):
    max_acc = 0
    train_loss_list = list()
    test_loss_list = list()
    train_metrics_list = list()
    test_metrics_list = list()

    if not os.path.isdir(chkp_dir):
        os.mkdir(chkp_dir)    

    for epoch_count in range(epoch_count):
        train_loss = train_loop(train_loader, model, loss_function, optimizer, device)
        train_loss_list.append(train_loss)
        train_predictions, train_labels = predict(train_loader, model, device)
        train_acc, train_recall, train_precision = calc_metrics(train_predictions, train_labels)
        train_metrics_list.append({'acc': train_acc,'rec': train_recall, 'pre': train_precision})

        if epoch_validation:
            test_predictions, test_labels = predict(test_loader, model, device)
            test_loss = log_likelihood(test_predictions, test_labels)   # predicions are probs so use another loss function to get equivalent
            test_loss_list.append(test_loss)
            test_acc, test_recall, test_precision = calc_metrics(test_predictions, test_labels)

            metric_item = {'acc': test_acc,'rec': test_recall, 'pre': test_precision}
            test_metrics_list.append(metric_item)
            if metric_item['acc'] > max_acc:
                max_acc = metric_item['acc']
                save_model_head(model, chkp_dir, metric_item)

            print(f"Epoch: {epoch_count}; train loss={train_loss:>8f}; validation loss={test_loss:>8f}")
            print(f"train metrics: acc={train_acc:>8f}; recall={train_recall:>8f}; precision={train_precision:>8f};")
            print(f"val metrics: acc={test_acc:>8f}; recall={test_recall:>8f}; precision={test_precision:>8f};")
            print("-----------")
        else:
            metric_item = train_metrics_list[-1]
            if metric_item['acc'] > max_acc:
                max_acc = metric_item['acc']
                save_model_head(model, chkp_dir)
            print(f"Epoch: {epoch_count}; train loss={train_loss:>8f}")
            print(f"train metrics: acc={train_acc:>8f}; recall={train_recall:>8f}; precision={train_precision:>8f};")
            print("-----------")

        if train_loss < train_loss_threshold:
            break
        torch.cuda.empty_cache()
        gc.collect()  

    result_graphic = plot_epoch_metrics(epoch_count, train_loss_list, test_loss_list, train_metrics_list, test_metrics_list, epoch_validation)
    result_graphic.write_image(os.path.join(chkp_dir, 'metrics.pdf'))
    
    return train_loss_list, test_loss_list

from torch.optim import AdamW

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
print('loading model...')
frida_classifier = FridaClassifier()
print("loaded")
frida_classifier.to(device)
lr = 0.0005
loss_function = torch.nn.CrossEntropyLoss()
optimizer = AdamW(params=frida_classifier.classifier.parameters(), lr=lr)
epoch_count = 20

train_loss_list, val_loss_list = train_model(epoch_count, frida_classifier, optimizer, loss_function, train_dataloader, 
            val_dataloader, epoch_validation=True, train_loss_threshold=0, device=device, chkp_dir=checkpoint_dir)

print()
print('BEST MODEL TEST RESULTS:')
gc.collect()
torch.cuda.empty_cache()

loaded_model = FridaClassifier()
loaded_model.classifier.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'classifier_head.pth'), map_location='cpu', weights_only=True))
loaded_model.eval()
loaded_model.to('cuda')
test_predictions, test_labels = predict(test_dataloader, loaded_model, device)
test_loss = log_likelihood(test_predictions, test_labels)   # predicions are probs so use another loss function to get equivalent
test_acc, test_recall, test_precision = calc_metrics(test_predictions, test_labels)
print(f"test loss={test_loss:>8f}")
print(f"test metrics: acc={test_acc:>8f}; recall={test_recall:>8f}; precision={test_precision:>8f};")
