import argparse, os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import *


#####load data
parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--m",
                    default="roberta",
                    type=str)
parser.add_argument("--i",
                    default="Event-Event-SRL-DEP",
                    type=str)
parser.add_argument("--t",
                    default='binary',
                    type=str)
args = parser.parse_args()
model_use = args.m
input_type = args.i
task = args.t


df_train, df_val, df_test = read_data(model_name="bert", input=input_type, task=task)
####bert: bert_base_uncased, bert_large_uncased,  bert_base_cased, bert_large_cased
###roberta: roberta-base, roberta-large
###xlnet:xlnet-base-cased, xlnet-large-cased


MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, "bert-large-uncased"),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, "xlnet-large-cased"),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-large'),
}
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_use]

tokenizer = tokenizer_class.from_pretrained(config_class, do_lower_case=True)


encoded_data_train = tokenizer.batch_encode_plus(
    df_train.input_text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)


encoded_data_val = tokenizer.batch_encode_plus(
    df_val.input_text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    df_test.input_text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df_train.label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df_val.label.values)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(df_test.label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

print("data size is ", len(dataset_train), len(dataset_val), len(dataset_test))

model = model_class.from_pretrained(config_class,
                                                      num_labels=len(list(set(df_train["label"].tolist()))),
                                                      output_attentions=False,
                                                      output_hidden_states=False)



batch_size = 4
learning_rate = 1e-5
num_epoch = 4

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)
dataloader_val = DataLoader(dataset_val,
                              sampler=RandomSampler(dataset_val),
                              batch_size=batch_size)
dataloader_test = DataLoader(dataset_test,
                                   sampler=SequentialSampler(dataset_test),
                                   batch_size=batch_size)
optimizer = AdamW(model.parameters(),
                  lr=learning_rate,
                  eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*num_epoch)


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='macro')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("using device: ", device)


####evaluation

def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

####training/fintuning

best_epoch = num_epoch
save_val_f1 = 0

if not os.path.exists("../saved_models/finetuned_{}_model/".format(model_use)):
    os.makedirs("../saved_models/finetuned_{}_model/".format(model_use))
for epoch in tqdm(range(1, num_epoch+1)):

    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


    torch.save(model.state_dict(), '../saved_models/finetuned_{}_model/finetuned_{}_epoch_{}.model'.format(model_use, model_use, epoch))

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (macro): {val_f1}')
    #####record the epoch with the best val fi score
    if val_f1 > save_val_f1:
        save_val_f1 = val_f1
        best_epoch = epoch



model = model_class.from_pretrained(config_class,
                                                      num_labels=len(list(set(df_train["label"].tolist()))),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)


model.load_state_dict(torch.load('../saved_models/finetuned_{}_model/finetuned_{}_epoch_{}.model'.format(model_use, model_use, best_epoch), map_location=torch.device('cpu')))

_, predictions, true_test = evaluate(dataloader_test)
preds_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_test.flatten()

# print(accuracy_per_class(predictions, true_test))
if task == 'binary':
    roc_auc = roc_auc_score(labels_flat, preds_flat)
    print("For {}".format(model_use), input_type, task, "roc_auc is ", roc_auc)



with open('../results/{}_report_{}_{}.txt'.format(model_use, input_type, task),'w') as f:
    f.writelines(metrics_frame(preds_flat, labels_flat))



