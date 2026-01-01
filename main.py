# main.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from layers import Dense, ReLU
from losses import SoftmaxCrossEntropy
from network import NeuralNetwork
from optimizers import SGD
from trainer import Trainer
from hyperparameter_tuning import HyperparameterTuning

# نستخدم بيانات الأرقام مشان ندرب الشبكة
digits = load_digits()
x = digits.data / 16.0
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# نجرب إعدادات مختلفة مشان نلاقي أحسن نتائج
tuner = HyperparameterTuning(x_train, y_train, x_test, y_test)
best_params, best_acc = tuner.run()

print("\nBest hyperparameters:")
print(best_params)
print("Best test accuracy:", best_acc)

# بناء الشبكة باستخدام أحسن الإعدادات
network = NeuralNetwork()
network.add(Dense(64, best_params["hidden_units"]))
network.add(ReLU())
network.add(Dense(best_params["hidden_units"], 10))
network.set_loss(SoftmaxCrossEntropy())

optimizer = SGD(best_params["learning_rate"])
trainer = Trainer(network, optimizer)

trainer.fit(
    x_train, y_train,
    x_test, y_test,
    epochs=20,
    batch_size=best_params["batch_size"]
)
