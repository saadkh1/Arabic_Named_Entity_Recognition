from transformers import AutoTokenizer, BertForTokenClassification
import torch
import numpy as np
import argparse
from typing import List
from config import Config
from utils import read_labels, get_label_map, get_inv_label_map


class NERPredictor:
    def __init__(self, model_path: str):
        self.cfg = Config()
        
        self.label_list = read_labels('/content/label_list.txt')
        self.label_map = get_label_map(self.label_list)
        self.inv_label_map = get_inv_label_map(self.label_list)

        self.model = BertForTokenClassification.from_pretrained(self.cfg.MODEL_NAME,
                                                                return_dict=True,
                                                                num_labels=len(self.label_map),
                                                                output_attentions=False,
                                                                output_hidden_states=False)

        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.MODEL_NAME)

    def predict(self, sentences: str) -> List[str]:
        input_ids = self.tokenizer.encode(sentences, return_tensors='pt')

        with torch.no_grad():
            self.model.to('cpu')
            output = self.model(input_ids)

        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(self.inv_label_map[label_idx])
                new_tokens.append(token)

        return new_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Named Entities in a given sentence.')
    parser.add_argument('--sentence', type=str, required=True, help='input sentence to predict NER labels')
    args = parser.parse_args()

    cfg = Config()
    predictor = NERPredictor(model_path='/content/best_model.pt')
    sentence = args.sentence
    predicted_labels = predictor.predict(sentence)
    for token, label in zip(sentence.split(), predicted_labels):
        print("{}\t{}".format(label, token))
