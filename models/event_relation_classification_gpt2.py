
import argparse
import numpy
import torch
from transformers import set_seed, GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification
from utils import *

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--m",
                    default="gpt2",
                    type=str)
parser.add_argument("--i",
                    default="Event-Event-SRL-DEP",
                    type=str)
parser.add_argument("--t",
                    default='binary',
                    type=str)
args = parser.parse_args()
model_name = args.m
input_type = args.i
task = args.t
train_dataset, val_dataset, test_dataset = read_data(model_name=model_name, input=input_type, task=task)

model_config = GPT2Config.from_pretrained('gpt2-large', num_labels=len(list(set(train_dataset["label"].tolist())))) # Binary Classification
model = GPT2ForSequenceClassification.from_pretrained('gpt2-large', config=model_config)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
tokenizer.padding_side = "left" # Very Important
tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

train_dataset = train_dataset[['text', 'label']].to_dict('records')
val_dataset = val_dataset[['text', 'label']].to_dict('records')
test_dataset = test_dataset[['text', 'label']].to_dict('records')



# print(111, test_dataset)
class Gpt2ClassificationCollator(object):
    def __init__(self, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        return

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [int(sequence['label']) for sequence in sequences]
        inputs = self.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)
        inputs.update({'labels': torch.tensor(labels)})

        return inputs

gpt2classificationcollator = Gpt2ClassificationCollator(tokenizer=tokenizer,
                                                        max_seq_len=256)


from torch.utils.data import DataLoader


train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=4,
                              shuffle=True,
                              collate_fn=gpt2classificationcollator)
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=4,
                            shuffle=False,
                            collate_fn=gpt2classificationcollator)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=4,
                             shuffle=False,
                             collate_fn=gpt2classificationcollator)

from transformers import AdamW, get_cosine_schedule_with_warmup

total_epochs = 4

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=1e-5,
                  eps=1e-8)

num_train_steps = len(train_dataloader) * total_epochs
num_warmup_steps = int(num_train_steps * 0.1)

lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=num_warmup_steps,
                                               num_training_steps = num_train_steps)


def train(dataloader, optimizer, scheduler, device_):
    global model
    model.train()

    prediction_labels = []
    true_labels = []

    total_loss = []

    for batch in dataloader:
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k, v in batch.items()}


        outputs = model(**batch)
        loss, logits = outputs[:2]
        logits = logits.detach().cpu().numpy()
        total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradient

        optimizer.step()
        scheduler.step()

        prediction_labels += logits.argmax(axis=-1).flatten().tolist()

    return true_labels, prediction_labels, total_loss

def validation(dataloader, device_):
    global model
    model.eval()

    prediction_labels = []
    true_labels = []
    predicted_logits = []
    total_loss = []

    for batch in dataloader:
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss.append(loss.item())

            prediction_labels += logits.argmax(axis=-1).flatten().tolist()
            predicted_logits += logits.tolist()
    return true_labels, prediction_labels, total_loss, predicted_logits


####train#######
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

all_loss = {'train_loss': [], 'val_loss': []}
all_acc = {'train_acc': [], 'val_acc': []}

for epoch in range(total_epochs):
    y, y_pred, train_loss = train(train_dataloader, optimizer, lr_scheduler, device)
    train_acc = accuracy_score(y, y_pred)

    y, y_pred, val_loss, y_logits = validation(val_dataloader, device)
    val_acc = accuracy_score(y, y_pred)

    all_loss['train_loss'] += train_loss
    all_loss['val_loss'] += val_loss

    all_acc['train_acc'].append(train_acc)
    all_acc['val_acc'].append(val_acc)

    print(f'Epoch: {epoch}, train_loss: {torch.tensor(train_loss).mean():.3f}, train_acc: {train_acc:.3f}, val_loss: {torch.tensor(val_loss).mean():.3f}, val_acc: {val_acc:.3f}')



true_labels, y_pred, l, predicted_logits = validation(test_dataloader, device)


if task == 'binary':
    roc_auc = roc_auc_score(true_labels, y_pred)
    print("For gpt2", input_type, task, "roc_auc is ", roc_auc)

with open('../results/gpt2_report_{}_{}.txt'.format(input_type, task),'w') as f:
    f.writelines(metrics_frame(y_pred, true_labels))



