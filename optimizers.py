#optimizers.py
import numpy as np

class Optimizer:
    def update(self, params, grads):
        # دالة لتحديث الأوزان - لازم كل أوبتيمايزر يعدل فيها
        raise NotImplementedError


# خوارزمية التدرج المتناقص البسيطة
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]



# خوارزمية الزخم - بتحافظ على حركة التحديث
class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = {}  # السرعة لنخزن حركة التحديث

    def update(self, params, grads):
        # أول مرة نجهز السرعات
        if not self.v:
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])

        # نحدث الأوزان باستخدام الزخم
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# خوارزمية Adam :
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        # متغيرات لنحسب المتوسطات
        self.m = {}
        self.v = {}

    def update(self, params, grads):
        # أول مرة نجهز المتغيرات
        if not self.m:
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # نحدث المتوسط الأول
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # نحدث المتوسط الثاني
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # نحدث الأوزان
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)