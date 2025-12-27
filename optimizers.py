#optimizers.py
import numpy as np

# ========================================
# Base Optimizer (Class مجرد)
# ========================================
class Optimizer:
    def update(self, params, grads):
        # # تابع مجرد لتحديث الأوزان
        raise NotImplementedError


# ========================================
# Stochastic Gradient Descent (SGD)
# ========================================
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]



# ========================================
# Momentum Optimizer
# ========================================
class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = {}  # # السرعة (velocity)

    def update(self, params, grads):
        # # تهيئة السرعات أول مرة
        if not self.v:
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])

        # # تحديث باستخدام الزخم
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# ========================================
# Adam Optimizer
# ========================================
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = {}
        self.v = {}

    def update(self, params, grads):
        # # تهيئة المتغيرات أول مرة
        if not self.m:
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # # تحديث المتوسط الأول
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # # تحديث المتوسط الثاني
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # # تحديث الأوزان
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)