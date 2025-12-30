#layers.py
import numpy as np

# ========================================
# Base Layer (طبقة مجردة)
# ========================================
class Layer:
    def forward(self, x):
        # # الانتشار الأمامي (لازم ينعمل override)
        raise NotImplementedError

    def backward(self, dout):
        # # الانتشار الخلفي (لازم ينعمل override)
        raise NotImplementedError


# ========================================
# Dense Layer (Affine Layer)
# ========================================
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # # تهيئة الأوزان بقيم صغيرة
        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)

        # # لتخزين القيم أثناء الـ forward
        self.x = None

        # # لتخزين الـ gradients
        self.dW = None
        self.db = None

    def forward(self, x):
        # # تخزين الدخل لاستخدامه بالـ backward
        self.x = x

        # # y = xW + b
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        # # gradient بالنسبة للأوزان
        self.dW = np.dot(self.x.T, dout)

        # # gradient بالنسبة للـ bias
        self.db = np.sum(dout, axis=0)

        # # gradient بالنسبة للدخل (للطبقة السابقة)
        dx = np.dot(dout, self.W.T)
        return dx
    
# ========================================
# ReLU Activation Layer
# ========================================
class ReLU(Layer):
    def __init__(self):
        self.mask = None  # # لتحديد القيم السالبة

    def forward(self, x):
        # # نخزّن أماكن القيم <= 0
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # # نوقف الـ gradient عند القيم السالبة
        dout[self.mask] = 0
        dx = dout
        return dx


# ========================================
# Sigmoid Activation Layer
# ========================================
class Sigmoid(Layer):
    def __init__(self):
        self.out = None  # # نخزّن خرج الـ forward

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # # مشتقة sigmoid
        dx = dout * (1.0 - self.out) * self.out
        return dx
    