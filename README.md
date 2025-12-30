Mini Neural Network from Scratch (NumPy)
Overview

This project is a simple neural network implemented from scratch using NumPy only.
No deep learning libraries (like TensorFlow or PyTorch) are used.

The goal of this project is to understand how neural networks work internally, including:

Forward propagation

Backward propagation (backpropagation)

Loss functions

Optimizers

Hyperparameter tuning

The network is trained and tested on the Digits dataset (handwritten digits from 0 to 9).

Project Structure
.
├── main.py
├── layers.py
├── losses.py
├── network.py
├── optimizers.py
├── trainer.py
├── hyperparameter_tuning.py
└── README.md

Files Explanation
1. layers.py

Contains the implementation of neural network layers:

Layer (base abstract class)

Dense (fully connected layer)

ReLU activation function

Sigmoid activation function

Each layer has:

forward() for forward propagation

backward() for backpropagation

2. losses.py

Contains loss functions:

MeanSquaredError

SoftmaxCrossEntropy (used for classification)

SoftmaxCrossEntropy combines Softmax + Cross Entropy for numerical stability.

3. network.py

Defines the NeuralNetwork class:

Stores layers

Runs forward pass

Computes loss

Runs backward pass

Calculates accuracy

Collects parameters and gradients

This file represents the core of the neural network.

4. optimizers.py

Implements optimization algorithms:

SGD (Stochastic Gradient Descent)

Momentum

Adam

In this project, SGD is mainly used.

5. trainer.py

Handles the training process:

Mini-batch training

Shuffling data

Printing loss and test accuracy each epoch

This file controls the training loop.

6. hyperparameter_tuning.py

Tries different values for:

Learning rate

Batch size

Number of hidden units

For each combination:

Trains the network for a few epochs

Evaluates test accuracy

Selects the best hyperparameters

7. main.py

This is the entry point of the project.

Steps:

Load Digits dataset

Normalize data

Split data into train and test

Run hyperparameter tuning

Build the final neural network using the best parameters

Train the final model for 20 epochs

Print loss and test accuracy

Dataset

Digits dataset from sklearn

Images are 8×8 pixels (64 features)

10 classes (digits 0–9)

How to Run
Requirements

Python 3.x

NumPy

scikit-learn

Install dependencies:

pip install numpy scikit-learn


Run the project:

python main.py

Example Output

During training, the program prints:

Epoch number

Average loss

Test accuracy

Example:

Epoch 10/20 | Loss: 0.1755 | Test Acc: 0.9361


At the end:

Best hyperparameters:
{'learning_rate': 0.1, 'batch_size': 16, 'hidden_units': 50}
Best test accuracy: 0.90+

Results

The model achieves high accuracy (~96%) on the test set.

Increasing learning rate and hidden units improves performance.

Hyperparameter tuning significantly affects results.

Learning Objectives

This project helped me understand:

How neural networks work internally

How backpropagation is implemented

The role of optimizers

The effect of hyperparameters

Training using mini-batches

Notes

This project is educational, not optimized for speed.

Everything is implemented manually to improve understanding.