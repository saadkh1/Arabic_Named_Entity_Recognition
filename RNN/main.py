import numpy as np
import pandas as pd
import argparse
import gensim
from cleaning import DataReader
from config import Config
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, GRU, Input, Dense, Embedding, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

class RNN:
    def __init__(self, args, word2vec_model, rnn_type):
        self.cfg = Config()
        self.args = args
        self.word2vec_model = word2vec_model
        self.rnn_type = rnn_type

    def run(self, args):
        data_reader = DataReader("/content/data.txt")
        sentences, tags, vocab = data_reader.read_data_rnn()

        word2id = {word:id for  id, word in enumerate(vocab)}

        label_list = list(pd.read_csv('label_list.txt', header=None, index_col=0).T)
        tags_encoding = { v:index for index, v in enumerate(label_list) }

        embedding_matrix = self.get_embedding_model(embedding_type=args.word2vec, word2id=word2id, vocab=vocab, embed_size=self.cfg.embed_size)

        sentences_encoded = [self.encode_sentence(sentence, word2id) for sentence in sentences]
        tags_encoded = [self.encode_tags(tag, tags_encoding) for tag in tags]

        sentences_padded = pad_sequences(sequences=sentences_encoded, 
                                          maxlen=self.cfg.MAX_SEQUENCE_LENGTH,
                                          dtype='int32', 
                                          padding='post',
                                          truncating='post',
                                          value = 0)
        tags_padded = pad_sequences(sequences=tags_encoded, 
                                          maxlen=self.cfg.MAX_SEQUENCE_LENGTH,
                                          dtype='int32', 
                                          padding='post',
                                          truncating='post',
                                          value=self.cfg.tags_pad_val)

        train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences_padded, 
                                                                                        tags_padded, 
                                                                                        train_size=self.cfg.train_size, 
                                                                                        random_state=self.cfg.random_state)

        model = self.build_model(embedding_matrix, vocab=vocab, embed_size=self.cfg.embed_size, rnn_type=args.model, tags_encoding=tags_encoding)

        model.fit(train_sentences, 
                  train_labels, 
                  validation_split=self.cfg.validation_size, 
                  batch_size=self.cfg.batch_size,
                  epochs=self.cfg.epochs)

        model.evaluate(test_sentences, test_labels)
        model.save('my_model.h5')

    def get_embedding_model(self, embedding_type, word2id, vocab, embed_size):
        if embedding_type == 'cbow':
            weights_path = "/content/word2vec_model/wikipedia_cbow_300"
        elif embedding_type == 'skipgram':
            weights_path = "/content/word2vec_model/wikipedia_sg_300"
        else:
            raise ValueError("Invalid embedding type")

        w2v_model = gensim.models.Word2Vec.load(weights_path)
        num_words = len(vocab)
        embedding_matrix = np.zeros(shape=(num_words, embed_size))
        for word, id in word2id.items():
            try:
                embedding_matrix[id] = w2v_model.wv[word]
            except KeyError:
                embedding_matrix[id] = np.zeros(embed_size)

        return embedding_matrix

    def encode_sentence(self, old_sentence, word2id):
        return [word2id.get(word, 0) for word in old_sentence]

    def encode_tags(self, old_tags, tags_encoding):
        new_tags = [tags_encoding[tag] for tag in old_tags]
        return to_categorical(y=new_tags, num_classes=len(tags_encoding))

    def build_model(self, embedding_matrix, vocab, embed_size, rnn_type, tags_encoding):
        input_layer = Input(shape=(self.cfg.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer = Embedding(input_dim=len(vocab),
                                    output_dim=embed_size,
                                    input_length=self.cfg.MAX_SEQUENCE_LENGTH,
                                    weights=[embedding_matrix],
                                    trainable=True)(input_layer)
        if rnn_type == 'lstm':
            rnn_layer = LSTM(units=embed_size,
                            return_sequences=True,
                            dropout=self.cfg.dropout,
                            recurrent_dropout=self.cfg.recurrent_dropout)(embedding_layer)
        elif rnn_type == 'bilstm':
            rnn_layer = Bidirectional(LSTM(units=embed_size,
                                        return_sequences=True,
                                        dropout=self.cfg.dropout,
                                        recurrent_dropout=self.cfg.recurrent_dropout))(embedding_layer)
        elif rnn_type == 'gru':
            rnn_layer = GRU(units=embed_size,
                            return_sequences=True,
                            dropout=self.cfg.dropout,
                            recurrent_dropout=self.cfg.recurrent_dropout)(embedding_layer)
        elif rnn_type == 'bigru':
            rnn_layer = Bidirectional(GRU(units=embed_size,
                                        return_sequences=True,
                                        dropout=self.cfg.dropout,
                                        recurrent_dropout=self.cfg.recurrent_dropout))(embedding_layer)
        else:
            raise ValueError('Invalid RNN type')
        output_layer = TimeDistributed(Dense(len(tags_encoding), activation='softmax'))(rnn_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self):
        self.model.fit(self.train_sentences, self.train_labels, validation_split=self.cfg.validation_size, batch_size = self.cfg.batch_size, epochs = self.cfg.epochs)

    def evaluate(self):
        self.model.evaluate(self.test_sentences, self.test_labels)

    def save(self):
        self.model.save('my_model.h5')

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--word2vec', type=str, choices=['cbow', 'skipgram'], default='cbow', help = 'choose between CBOW and SKIPGRAM')
  parser.add_argument('--model', type=str, choices=['lstm', 'bilstm', 'gru', 'bigru'], default='lstm', help = 'choose between LSTM, BILSTM, GRU and BIGRU models')
  
  args = parser.parse_args()
  rnn = RNN(args=args, word2vec_model=args.word2vec, rnn_type=args.model)
  rnn.run(args)