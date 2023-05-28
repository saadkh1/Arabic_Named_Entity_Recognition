import numpy as np

class Config:
  tags_pad_val = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
  sentences_pad_val = 0
  MAX_SEQUENCE_LENGTH = 45
  train_size = 0.8
  validation_size = 0.15
  random_state= 42
  lr = 0.0001
  beta_1 = 0.9
  beta_2 = 0.999
  epochs = 50
  batch_size = 10
  dropout = 0.5
  units = 42
  dropout=0.5
  recurrent_dropout=0.5
  embed_size=300