# hyperparameter_tuning.py
import numpy as np

from layers import Dense, ReLU
from losses import SoftmaxCrossEntropy
from network import NeuralNetwork
from optimizers import SGD
from trainer import Trainer


class HyperparameterTuning:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        learning_rates = [0.01, 0.05, 0.1]
        batch_sizes = [16, 32, 64]
        hidden_units = [32, 50, 100]

        best_acc = 0.0
        best_params = None

        for lr in learning_rates:
            for batch_size in batch_sizes:
                for hidden in hidden_units:
                    # بناء شبكة جديدة
                    network = NeuralNetwork()
                    network.add(Dense(64, hidden))
                    network.add(ReLU())
                    network.add(Dense(hidden, 10))
                    network.set_loss(SoftmaxCrossEntropy())

                    optimizer = SGD(lr)
                    trainer = Trainer(network, optimizer)

                    # تدريب سريع مشان نجرب الإعدادات
                    trainer.fit(
                        self.x_train, self.y_train,
                        self.x_test, self.y_test,
                        epochs=5,
                        batch_size=batch_size,
                    )

                    acc = trainer.test_accuracy

                    # نطبع نتائج التجربة
                    print(
                        f"lr={lr}, batch={batch_size}, hidden={hidden} "
                        f"=> acc={acc:.4f}"
                    )

                    if acc > best_acc:
                        best_acc = acc
                        best_params = {
                            "learning_rate": lr,
                            "batch_size": batch_size,
                            "hidden_units": hidden
                        }

        return best_params, best_acc
