# Classification-using-LSTM

# MNIST Handwritten Digit Classification using LSTM

Project Overview:
The goal of this project is to classify handwritten digits from the MNIST dataset. An LSTM model is built and trained to recognize the digits, and its performance is evaluated using various metrics.

Dataset:
The project utilizes the MNIST dataset, a large database of handwritten digits that is commonly used for training various image processing systems. It consists of:

60,000 training images and their corresponding labels.
10,000 test images and their corresponding labels.
Each image is a 28x28 grayscale image.

Model Architecture:
The model is a simple recurrent neural network (RNN) built using the Keras Functional API, consisting of:

An Input layer expecting 28x28 images.
An LSTM layer with 128 units to process the sequential nature of the image rows.
A Dense output layer with 10 units (for 10 classes of digits 0-9) and a softmax activation function for multi-class classification.

Training:
The model is compiled with:

Optimizer: Adam
Loss Function: categorical_crossentropy (suitable for multi-class classification with one-hot encoded labels)
Metrics: accuracy
The model was trained for 15 epochs with the following training history:

Epoch	Training Accuracy	Training Loss	Validation Accuracy	Validation Loss:
1	0.7733	0.6747	0.9679	0.1068
...	...	...	...	...
15	0.9962	0.0122	0.9852	0.0558
(Full training history can be found in the notebook output.)

Results:
After training, the model achieved the following performance on the test set:

Test Accuracy: ~98.52%
Test Loss: ~0.0558
Detailed evaluation metrics are provided, including a confusion matrix and a classification report, showcasing the precision, recall, and f1-score for each digit class.

Confusion Matrix
array([[ 979,    0,    0,    0,    0,    0,    0,    1,    0,    0],
       [   1, 1125,    5,    1,    0,    0,    2,    0,    1,    0],
       [   1,    0, 1023,    4,    0,    0,    1,    2,    1,    0],
       [   0,    0,    1, 1004,    0,    2,    0,    1,    2,    0],
       [   0,    1,    1,    0,  956,    0,    0,    3,    2,   19],
       [   0,    1,    0,   12,    0,  872,    3,    3,    0,    1],
       [   9,    3,    0,    1,    3,    0,  939,    0,    2,    1],
       [   0,    2,    8,    0,    0,    0,    0, 1017,    0,    1],
       [   3,    0,    3,    3,    1,    2,    3,    2,  957,    0],
       [   0,    2,    2,    9,    7,    1,    0,    5,    3,  980]])
Classification Report
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.99      0.99      1032
           3       0.97      0.99      0.98      1010
           4       0.99      0.97      0.98       982
           5       0.99      0.98      0.99       892
           6       0.99      0.98      0.99       958
           7       0.98      0.99      0.99      1028
           8       0.99      0.98      0.99       974
           9       0.98      0.97      0.97      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.98      0.99     10000
weighted avg       0.99      0.99      0.99     10000

# MNIST Handwritten Digit Classification using Bidirectional LSTM

Project Overview:
The goal of this project is to build and train a recurrent neural network (RNN) model, specifically a Bidirectional LSTM, to accurately classify handwritten digits from the MNIST dataset. The notebook covers data loading, preprocessing, model definition, training, and evaluation.

Model Architecture:
The model utilizes a Bidirectional LSTM layer, which processes sequences in both forward and backward directions, allowing it to capture dependencies from both past and future contexts. This is followed by a Dense output layer with softmax activation for multi-class classification.

Input Layer: (28, 28) representing the 28x28 pixel images.
Bidirectional LSTM Layer: 128 units, capturing sequential patterns in the image rows.
Output Layer: Dense(10, activation='softmax') for classifying into 10 digit classes.
Results:
After training for 15 epochs, the model achieved the following performance on the test set:

Test Loss: 0.043
Test Accuracy: 0.9889 (approximately 98.89%)
The detailed classification report shows strong performance across all digit classes:

              precision    recall  f1-score   support

           0       0.99      1.00      0.99       980
           1       1.00      0.99      0.99      1135
           2       0.99      0.99      0.99      1032
           3       0.98      0.99      0.99      1010
           4       0.99      0.98      0.99       982
           5       1.00      0.99      0.99       892
           6       0.99      1.00      0.99       958
           7       0.98      0.99      0.99      1028
           8       0.99      0.98      0.99       974
           9       0.98      0.98      0.98      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
