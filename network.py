#network.py
import numpy as np

class NeuralNetwork:
    def __init__(self):
        # قائمة الطبقات
        self.layers = []
        # تابع الخسارة
        self.loss_fn = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def gradient(self, x, y):
        # ---------- Forward ----------
        out = x
        for layer in self.layers:
            out = layer.forward(out)

        # ---------- Loss ----------
        loss = self.loss_fn.forward(out, y)

        # ---------- Backward ----------
        dout = self.loss_fn.backward()

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # ---------- Collect gradients ----------
        grads = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "dW"):
                grads[f"W{i}"] = layer.dW
                grads[f"b{i}"] = layer.db

        return grads
    
    def loss(self, x, y):
        y_pred = self.predict(x)
        return self.loss_fn.forward(y_pred, y)

    def backward(self):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def accuracy(self, x, y):
        y_pred = self.predict(x)
        y_pred_label = np.argmax(y_pred, axis=1)
        return np.mean(y_pred_label == y)

    def params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                params[f"W{i}"] = layer.W
                params[f"b{i}"] = layer.b
        return params
