#losses.py
import numpy as np

# دالة الخطأ التربيعية 
class MeanSquaredError:
    def __init__(self):
        self.y = None  # مخرجات الشبكة
        self.t = None  # القيم الحقيقية

    def forward(self, y, t):
        # نخزن القيم لنستخدمها في التفاضل العكسي
        self.y = y
        self.t = t

        # حساب مقدار الخطأ
        loss = 0.5 * np.sum((y - t) ** 2) / y.shape[0]
        return loss

    def backward(self):
        # مشتقة دالة الخطأ بالنسبة للمخرج
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# دالة الخطأ الخاصة بالتصنيف
class SoftmaxCrossEntropy:
    def __init__(self):
        self.y = None  # مخرجات softmax
        self.t = None  # التسميات

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x, t):
        # حساب دالة softmax مشان نحول القيم لاحتمالات
        self.y = self.softmax(x)
        self.t = t

        # لو التسمية one-hot نحولها لرقم فئة
        if t.ndim == 2:
            t = np.argmax(t, axis=1)

        batch_size = x.shape[0]

        # حساب الخطأ باستخدام Cross Entropy
        loss = -np.sum(np.log(self.y[np.arange(batch_size), t] + 1e-7))
        return loss / batch_size

    def backward(self):
        batch_size = self.t.shape[0]

        # لو التسمية بصيغة one-hot
        if self.t.ndim == 2:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
