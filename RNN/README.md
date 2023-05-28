# Arabic Named Entity Recognition with RNN

This code implements a recurrent neural network (RNN) model for Arabic Named Entity Recognition (NER). The model uses word embeddings, such as CBOW or Skip-gram, and different types of RNNs (LSTM, BiLSTM, GRU, BiGRU) for sequence tagging.

## Installation

1. Install the required Python packages by running the following command:

```bash
 pip install -r requirements.txt
```
2. Run the word2vec.sh script to install the word2vec models (skipgram and cbow) required for the NER model:

```bash
 sh word2vec.sh
```

## Training and Testing
1. Run the main.py script to train and test the model:

```bash
python main.py --word2vec [embedding_type] --model [rnn_type]
```
* 'embedding_type': Choose between "cbow" (Continuous Bag-of-Words) or "skipgram" for word embeddings (default: "cbow").

* 'rnn_type': Choose between "lstm", "bilstm", "gru", or "bigru" for the type of RNN model (default: "lstm").

The script will preprocess the data, train the model, and save it as 'my_model.h5'.

2. Prediction Example

To predict the named entities in a given sentence, you can use the prediction.py script. For example: 

```bash
python prediction.py --sentence "جامعة بيرزيت وبالتعاون مع مؤسسة ادوارد سعيد تنظم مهرجان للفن الشعبي سيبدأ الساعة الرابعة عصرا، بتاريخ 16/5/2016" --file_path "my_model.h5"
```
This will output the predicted named entities in the sentence.