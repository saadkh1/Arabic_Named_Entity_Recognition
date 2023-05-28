from preprocess import NERDataset
from cleaning import DataReader
import numpy as np

from utils import compute_metrics, get_label_map, get_inv_label_map, read_labels

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader
import torch
from torch import nn
from config import Config


class NERTrainer:
    def __init__(self):
        self.cfg = Config()
        self.data_reader = DataReader("/content/data.txt")
        self.data, _, _ = self.data_reader.read_data_bert()
        self.label_list = read_labels('/content/label_list.txt')

        self.label_map = get_label_map(self.label_list)
        self.inv_label_map = get_inv_label_map(self.label_list)

        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.test_data, self.eval_data = train_test_split(self.test_data, test_size=0.5, random_state=42)

        self.TOKENIZER = AutoTokenizer.from_pretrained(self.cfg.MODEL_NAME)

        self.train_dataset = NERDataset(
            texts=[x[0] for x in self.train_data],
            tags=[x[1] for x in self.train_data],
            label_list=self.label_list,
            model_name=self.cfg.MODEL_NAME,
            max_length=self.cfg.MAX_LEN
            )

        self.eval_dataset = NERDataset(
            texts=[x[0] for x in self.eval_data],
            tags=[x[1] for x in self.eval_data],
            label_list=self.label_list,
            model_name=self.cfg.MODEL_NAME,
            max_length=self.cfg.MAX_LEN
            )

        self.train_data_loader = DataLoader(dataset=self.train_dataset,batch_size=self.cfg.TRAIN_BATCH_SIZE,shuffle= True)
        self.eval_data_loader = DataLoader(dataset=self.eval_dataset,batch_size=self.cfg.VALID_BATCH_SIZE,shuffle= True)

        self.model = BertForTokenClassification.from_pretrained(self.cfg.MODEL_NAME,
                                                      return_dict=True,
                                                      num_labels=len(self.label_map),
                                                      output_attentions=False,
                                                      output_hidden_states=False).to(self.cfg.device)

        self.optimizer = AdamW(self.model.parameters(), lr=5e-5, correct_bias=False)
        total_steps = len(self.train_data_loader) * self.cfg.EPOCHS

        self.scheduler = get_linear_schedule_with_warmup(
          self.optimizer,
          num_warmup_steps=0,
          num_training_steps=total_steps
        )

        self.best_eval_loss = float('inf')
        self.best_model = None

    def train_epoch(self):

        self.model.train()

        final_loss = 0

        for data in tqdm(self.train_data_loader, total = len(self.train_data_loader)):
            input_ids = data['input_ids'].to(self.cfg.device)
            attention_mask = data['attention_mask'].to(self.cfg.device)
            token_type_ids = data['token_type_ids'].to(self.cfg.device)
            labels = data['labels'].to(self.cfg.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)

            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            final_loss += loss.item()

        loss = final_loss / len(self.train_data)
        print(f"Train loss: {loss}")

        return loss
    
    def eval_epoch(self):

        self.model.eval()

        final_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for data in tqdm(self.eval_data_loader, total=len(self.eval_data_loader)):
                input_ids = data['input_ids'].to(self.cfg.device)
                attention_mask = data['attention_mask'].to(self.cfg.device)
                token_type_ids = data['token_type_ids'].to(self.cfg.device)
                labels = data['labels'].to(self.cfg.device)

                outputs = self.model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)

                loss = outputs[0]
                final_loss += loss.item()

                logits = outputs.logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()

                all_preds.extend(logits)
                all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.asarray(all_labels)

        metrics = compute_metrics(all_preds, all_labels, self.inv_label_map, False)
        final_loss = final_loss / len(self.eval_data_loader)

        print(f"Eval loss: {final_loss}")
        print(f"Eval Metrics: {metrics}")

        return final_loss, metrics


    def train(self):
        for epoch in range(self.cfg.EPOCHS):
            print(f"Training Epoch: {epoch+1}")
            self.train_epoch()

            print(f"Evaluating Epoch: {epoch+1}")
            eval_loss, _ = self.eval_epoch()

            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.best_model = self.model.state_dict()
                torch.save(self.best_model, "best_model.pt")

if __name__ == '__main__':
    ner_trainer = NERTrainer()
    ner_trainer.train()