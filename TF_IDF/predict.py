import argparse
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TextClassifier:
    def __init__(self, model_type):
        self.vectorizer = CountVectorizer()
        self.tfidf = TfidfTransformer()
        self.model_type = model_type
        self.model = None

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.vectorizer, self.tfidf, self.model = pickle.load(f)

    def predict(self, phrase):
        if not self.model:
            raise ValueError('Model not loaded. Please load the model first.')

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


# Adding command line arguments
parser = argparse.ArgumentParser(description='Text Classifier')
parser.add_argument('--model_path', type=str, help='Path to the saved model file', required=True)
parser.add_argument('--sentence', type=str, help='Input sentence for prediction', required=True)
args = parser.parse_args()

# Loading the model
classifier = TextClassifier('')
classifier.load_model(args.model_path)

# Making predictions
predictions = classifier.predict(args.phrase)
print(predictions)
