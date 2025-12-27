#losses.py
import numpy as np

# ========================================
# Mean Squared Error Loss
# ========================================
class MeanSquaredError:
    def __init__(self):
        self.y = None  # # خرج الشبكة
        self.t = None  # # القيم الحقيقية

    def forward(self, y, t):
        # # تخزين القيم لاستخدامها بالـ backward
        self.y = y
        self.t = t

        # # حساب الـ loss
        loss = 0.5 * np.sum((y - t) ** 2) / y.shape[0]
        return loss

    def backward(self):
        # # مشتقة الـ loss بالنسبة للخرج
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# ========================================
# Softmax + Cross Entropy Loss
# ========================================
class SoftmaxCrossEntropy:
    def __init__(self):
        self.y = None  # # softmax output
        self.t = None  # # labels

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x, t):
        # # حساب softmax
        self.y = self.softmax(x)
        self.t = t

        # # إذا كانت التسمية one-hot نحولها
        if t.ndim == 2:
            t = np.argmax(t, axis=1)

        batch_size = x.shape[0]

        # # حساب Cross Entropy
        loss = -np.sum(np.log(self.y[np.arange(batch_size), t] + 1e-7))
        return loss / batch_size

    def backward(self):
        batch_size = self.t.shape[0]

        # # إذا كانت one-hot
        if self.t.ndim == 2:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
