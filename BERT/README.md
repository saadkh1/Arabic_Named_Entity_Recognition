# Arabic Named Entity Recognition with BERT

This repository contains code for training and testing an Arabic Named Entity Recognition (NER) model using the BERT model.

## Installation

To install the required dependencies, run the following command:

```bash
 pip install -r requirements.txt
```
## Training

To train the model, make sure you have the training data in a file named data.txt. Then, run the following command:

```bash
 python train.py
```
The training process will print the training loss for each epoch.

## Testing 

To test the trained model, run the following command:

```bash
 python test.py
```
The testing process will print the evaluation loss and metrics.

## Prediction

To use the trained model for prediction, run the following command:

```bash
 python prediction.py --sentence "جامعة بيرزيت وبالتعاون مع مؤسسة ادوارد سعيد تنظم مهرجان للفن الشعبي سيبدأ الساعة الرابعة عصرا، بتاريخ 16/5/2016"
```
Replace the sentence with the desired input sentence in Arabic. The model will predict the named entities in the sentence.

Please note that the model requires the best_model.pt file, which is the saved best model during the training process. Make sure to have this file in the same directory as the prediction.py script.