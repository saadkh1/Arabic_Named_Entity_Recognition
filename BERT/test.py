from tqdm.notebook import tqdm
import torch
import numpy as np
from utils import compute_metrics, read_labels, get_label_map, get_inv_label_map
from transformers import BertForTokenClassification
from preprocess import NERDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from cleaning import DataReader
from config import Config

class Trainer:
    def __init__(self):
        self.cfg = Config()
        self.device = self.cfg.device
        self.data_reader = DataReader("/content/data.txt")
        self.data, _, _ = self.data_reader.read_data_bert()
        self.label_list = read_labels('/content/label_list.txt')
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.test_data, self.eval_data = train_test_split(self.test_data, test_size=0.5, random_state=42)
        self.label_map = get_label_map(self.label_list)
        self.inv_label_map = get_inv_label_map(self.label_list)
        self.test_dataset = NERDataset(
            texts=[x[0] for x in self.test_data],
            tags=[x[1] for x in self.test_data],
            label_list=self.label_list,
            model_name=self.cfg.MODEL_NAME,
            max_length=self.cfg.MAX_LEN
        )

    def model_test(self, test_dl, model, device):
        with torch.no_grad():
            model.to(device)
            model.eval()
            final_loss = 0
            all_predictions = []
            all_labels = []
            for data in tqdm(test_dl, total = len(test_dl)):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids = input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
                predictions = outputs.logits
                all_labels.extend(labels.to('cpu').numpy())
                all_predictions.extend(outputs.logits.to('cpu').numpy())
                final_loss += loss.item()

            metrics = \
            compute_metrics(predictions=np.asarray(all_predictions), labels=np.asarray(all_labels), inv_label_map=self.inv_label_map)

            accuracy_score = metrics['accuracy_score']
            precision= metrics['precision']
            recall= metrics['recall']
            f1= metrics['f1']

            print(f' Accuracy: {accuracy_score}')
            print(f' Precision: {precision}')
            print(f' Recall: {recall}')
            print(f' F1: {f1}')

    def run(self):
        model = BertForTokenClassification.from_pretrained(self.cfg.MODEL_NAME,
                                                      return_dict=True,
                                                      num_labels=len(self.label_map),output_attentions = False,
                                                      output_hidden_states = False)

        test_data_loader = DataLoader(dataset=self.test_dataset,batch_size=self.cfg.TEST_BATCH_SIZE,shuffle= True)

        model.load_state_dict(torch.load('/content/best_model.pt', map_location='cpu'))

        self.model_test(test_dl=test_data_loader, model=model, device=self.cfg.device)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
