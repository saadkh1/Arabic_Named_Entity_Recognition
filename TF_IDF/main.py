from cleaning import DataReader
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

class TextClassifier:
    def __init__(self, model_type):
        self.vectorizer = CountVectorizer()
        self.tfidf = TfidfTransformer()
        if model_type == 'LinearSVC':
            self.model = LinearSVC()
        elif model_type == 'GaussianNB':
            self.model = GaussianNB()

    def train(self, data):
        train_text = data['text'].astype(str)
        train_label = data['label'].astype(str)

        self.vectorizer.fit(train_text)
        train_vec = self.vectorizer.transform(train_text)

        self.tfidf.fit(train_vec)
        train_tfvec = self.tfidf.transform(train_vec)

        self.model.fit(train_tfvec, train_label)

    def test(self, data):
        test_text = data['text'].astype(str)
        test_label = data['label'].astype(str)

        test_vec = self.vectorizer.transform(test_text)
        test_tfvec = self.tfidf.transform(test_vec)

        y_pred = self.model.predict(test_tfvec)
        accuracy = accuracy_score(test_label, y_pred)
        return accuracy

    def predict(self, phrase):
        arr = phrase.split()
        y = []
        token = []

        for x in arr:
            x = [x]
            test_str = self.vectorizer.transform(x)
            test_tfstr = self.tfidf.transform(test_str)
            token.append(x)
            y.append(self.model.predict(test_tfstr.toarray())[0])

        df = pd.DataFrame(list(zip(token, y)), columns=['token', 'entity_type'])
        return df

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.vectorizer, self.tfidf, self.model), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Classifier')
    parser.add_argument('--model_type', type=str, choices=['LinearSVC', 'GaussianNB'], default='LinearSVC', help='Type of model to use')
    args = parser.parse_args()

    # Training the model
    data_reader = DataReader("../data/data.txt")
    sentences, tags, _ = data_reader.read_data_rnn()

    data = data_reader.read_data_tf_idf(sentences, tags)

    train, test = train_test_split(data, test_size=0.2)

    classifier = TextClassifier(args.model_type)
    classifier.train(train)

    # Testing the model
    accuracy = classifier.test(test)
    print("Accuracy:", accuracy)

    # Saving the model
    classifier.save_model("model.pkl")
