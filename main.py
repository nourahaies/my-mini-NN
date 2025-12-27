#main.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from layers import Dense, ReLU
from losses import SoftmaxCrossEntropy
from network import NeuralNetwork
from optimizers import SGD
from trainer import Trainer

# =====================================
# تحميل البيانات
# =====================================
digits = load_digits()
x = digits.data
y = digits.target

# # تطبيع البيانات
x = x / 16.0

# # تقسيم Train / Test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# =====================================
# بناء الشبكة
# =====================================
network = NeuralNetwork()

network.add(Dense(64, 50))
network.add(ReLU())
network.add(Dense(50, 10))

network.set_loss(SoftmaxCrossEntropy())

# =====================================
# Optimizer
# =====================================
optimizer = SGD(lr=0.1)

# =====================================
# Trainer
# =====================================
trainer = Trainer(network, optimizer)

# =====================================
# التدريب
# =====================================
trainer.fit(
    x_train, y_train,
    x_test, y_test,
    epochs=20,
    batch_size=32
)
