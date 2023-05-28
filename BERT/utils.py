import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

def read_labels(label_file_path):
    return list(pd.read_csv(label_file_path, header=None, index_col=0).T)

def get_label_map(label_list):
    return {v: index for index, v in enumerate(label_list)}

def get_inv_label_map(label_list):
    return {i: label for i, label in enumerate(label_list)}

def align_predictions(predictions, label_ids, inv_label_map):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    assert(preds.shape == label_ids.shape)

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i] [j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(inv_label_map[label_ids[i][j]])
                preds_list[i].append(inv_label_map[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(predictions, labels, inv_label_map, generate_report=False):
    preds_list, out_label_list = align_predictions(predictions, labels, inv_label_map)

    if generate_report:
        try:
            print(classification_report(out_label_list, preds_list))
        except:
            print('There was an error while generating the classification report!')

    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
