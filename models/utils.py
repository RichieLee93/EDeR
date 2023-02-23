import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix, \
    roc_auc_score
import json

def read_data(model_name="bert", task="binary", data_path="../data/", input="Event-Event-SRL-DEP"):
    df_train = pd.DataFrame.from_records(load_json(data_path + "train.json"))
    df_dev = pd.DataFrame.from_records(load_json(data_path + "dev.json"))
    df_test = pd.DataFrame.from_records(load_json(data_path + "test.json"))
    if model_name in ["bert","roberta", "xlnet"]:
        if task == "binary":
            convert_binary_numeric(df_train)
            convert_binary_numeric(df_dev)
            convert_binary_numeric(df_test)

        elif task == "three":
            convert_three_numeric(df_train)
            convert_three_numeric(df_dev)
            convert_three_numeric(df_test)

        df_train['input_text'] = df_train[input]
        df_dev['input_text'] = df_dev[input]
        df_test['input_text'] = df_test[input]

        return df_train, df_dev, df_test


    if model_name == "distilbert":
        if task == "binary":
            convert_binary_numeric(df_train)
            convert_binary_numeric(df_dev)
            convert_binary_numeric(df_test)

        elif task == "three":
            convert_three_numeric(df_train)
            convert_three_numeric(df_dev)
            convert_three_numeric(df_test)
        df_train['TITLE'] = df_train[input]
        df_dev['TITLE'] = df_dev[input]
        df_test['TITLE'] = df_test[input]
        df_train = pd.concat([df_train, df_dev])
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        return df_train, df_test

    if model_name == "gpt2":
        if task == "binary":
            convert_binary_numeric(df_train)
            convert_binary_numeric(df_dev)
            convert_binary_numeric(df_test)

        elif task == "three":
            convert_three_numeric(df_train)
            convert_three_numeric(df_dev)
            convert_three_numeric(df_test)

        df_train['text'] = df_train[input]
        df_dev['text'] = df_dev[input]
        df_test['text'] = df_test[input]

        return df_train,df_dev, df_test

def load_json(file):
    with open(file, "r") as outfile:
        data = json.load(outfile)
    return data

def convert_binary_numeric(df):
    numeric_label_dict = {"independent": 0, "condition": 0, "optional argument": 1, "required argument": 1}
    df['label'] = df['label'].replace(numeric_label_dict)


def convert_three_numeric(df):
    numeric_label_dict = {"independent": 0, "condition": 0, "optional argument": 1, "required argument": 2}
    df['label'] = df['label'].replace(numeric_label_dict)

def convert_label_text_binary(df):
    text_label_dict = {"independent": "non-argument", "condition": "non-argument", "optional argument": "argument", "required argument": "argument"}
    df['label'] = df['label'].replace(text_label_dict)

def convert_label_text_three(df):
    text_label_dict = {"independent": "non-argument", "condition": "non-argument"}
    df['label'] = df['label'].replace(text_label_dict)

def metrics_frame(preds, labels):

    recall_micro = recall_score(labels, preds, average="micro")
    recall_macro = recall_score(labels, preds, average="macro")
    precision_micro = precision_score(labels, preds, average="micro")
    precision_macro = precision_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    cr = classification_report(labels, preds, digits=4)
    model_metrics = {"Precision, Micro": precision_micro, "Precision, Macro": precision_macro,
                     "Recall, Micro": recall_micro, "Recall, Macro": recall_macro,
                     "F1 score, Micro": f1_micro, "F1 score, Macro": f1_macro, "Classification report": cr}
    return model_metrics

