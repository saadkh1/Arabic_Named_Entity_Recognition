from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import argparse
import numpy as np
from data_cleaning import DataReader, TextCleaner
from tags import get_tags
from config import Config


class LSTMPredictor:
    def __init__(self, file_path):
        self.model = load_model(file_path)
        self.tags_onehot, self.tags_scores = get_tags()
        self.cfg = Config()
        self.data_reader = DataReader("/content/data.txt")
        _, _, self.vocab = self.data_reader.read_data_rnn()
        self.Cleaner = TextCleaner()
        self.word2id = {word:id for  id, word in enumerate(self.vocab)}
        
    def encode_sentence(self, old_sentence):
        encoded_sentence = [self.word2id.get(word, 0) for word in old_sentence]
        return encoded_sentence

    def predict(self, sentence):
        sentence = sentence.split(sep=' ')
        word_count = len(sentence) 
        # Clean sentence
        ready_sentence = [self.Cleaner.clean_text(word) for word in sentence]
        # Encode sentence
        ready_sentence = self.encode_sentence(ready_sentence)
        # Padding sentence
        ready_sentence = pad_sequences(sequences = [ready_sentence], 
                                  maxlen=self.cfg.MAX_SEQUENCE_LENGTH,
                                  dtype='int32', 
                                  padding='post',
                                  truncating='post',
                                  value = 0)

        predictions = self.model.predict(ready_sentence)[0][0:word_count]
        i = 0
        for prediction in predictions:
            for tag in list(self.tags_onehot.keys()):
                self.tags_scores[tag] = np.linalg.norm(self.tags_onehot[tag] - prediction)

            print(sentence[i],':',min(self.tags_scores, key=self.tags_scores.get))
            i+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='my_model.h5', help='path to save the Keras model')
    parser.add_argument('--sentence', type=str, default='منشئ المسجد هو أحمد بن طولون مؤسس الدولة الطولونية في مصر والشام', help='enter your sentence')
    args = parser.parse_args()

    lstm_predictor = LSTMPredictor(args.file_path)
    lstm_predictor.predict(args.sentence)
