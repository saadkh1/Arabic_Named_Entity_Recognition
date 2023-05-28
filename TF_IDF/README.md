# TF-IDF Arabic Named Entity Recognition

This code provides a TF-IDF based Arabic Named Entity Recognition (NER) model. It uses a cleaning module (cleaning.py) for text preprocessing and a main module (main.py) for training and testing the model.

## Usage

To train and test the TF-IDF Arabic NER model, follow these steps:

1. Ensure you have the required dependencies installed (pandas, sklearn).
2. Prepare your data file (data.txt) containing labeled sentences and entity tags.
3. Run the following command to train the model:
```bash
python main.py --model_type <model_type>
```
Replace <model_type> with the desired model type (LinearSVC or GaussianNB).

The training process will output the accuracy of the model on the test dataset.

4. To predict named entities in a new phrase, use the following command:
```bash
python predict.py --phrase "<phrase>" --model_path "<model_path>"
```
Replace <phrase> with the input sentence you want to predict the named entities for, and <model_path> with the path to the saved model file. If the model file is named model.pkl and located in the current directory, you can omit the --model_path argument.

Example command:
```bash
python predict.py --sentence "جامعة بيرزيت وبالتعاون مع مؤسسة ادوارد سعيد تنظم مهرجان للفن الشعبي سيبدأ الساعة الرابعة عصرا، بتاريخ 16/5/2016" --model_path "./model.pkl" --model_path "./model.pkl"
```

The output will be a dataframe containing the predicted entities and their corresponding tokens.

Note: The input sentences should be in Arabic.
