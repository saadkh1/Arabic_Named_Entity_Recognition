import torch

class Config:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  DATASET_NAME = 'Fine_GRATED'
  MODEL_NAME = 'aubmindlab/bert-base-arabertv02'
  TASK_NAME = 'tokenclassification'
  MAX_LEN = 256
  TRAIN_BATCH_SIZE = 16
  VALID_BATCH_SIZE = 16
  TEST_BATCH_SIZE = 16
  EPOCHS = 5
  recurrent_dropout=0.5
  embed_size=300