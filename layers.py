#layers.py
import numpy as np

# كلاس الطبقة الأساسية اللي كل الطبقات ترث منه
class Layer:
    def forward(self, x):
        # الانتشار الأمامي - لازم كل طبقة تعدل فيه
        raise NotImplementedError

    def backward(self, dout):
        # الانتشار الخلفي - لازم كل طبقة تعدل فيه
        raise NotImplementedError


# الطبقة الكثيفة (.Dense) اللي بتعمل ضرب المصفوفات
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # نبدأ الأوزان بقيم صغيرة 
        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)

        # نخزن المدخلات مشان نستخدمها في ال backward
        self.x = None

        # نخزن مشتقات الأوزان ال bias
        self.dW = None
        self.db = None

    def forward(self, x):
        # نخزن المدخلات مشان نستخدمها في التفاضل العكسي
        self.x = x

        # الحساب الأساسي: المدخلات × الأوزان + bias
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        # مشتقة loss بالنسبة للأوزان
        self.dW = np.dot(self.x.T, dout)

        # مشتقة loss بالنسبة ل bias
        self.db = np.sum(dout, axis=0)

        # نرجع مشتقة المدخلات للطبقة اللي قبلها
        dx = np.dot(dout, self.W.T)
        return dx
    
# دالة تنشيط ReLU اللي بتحول السالب لصفر
class ReLU(Layer):
    def __init__(self):
        self.mask = None  # لتحديد القيم السالبة

    def forward(self, x):
        # نخزن أماكن القيم السالبة أو صفر
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # نوقف تدفق المشتقة عند القيم السالبة
        dout[self.mask] = 0
        dx = dout
        return dx


# دالة تنشيط Sigmoid       
class Sigmoid(Layer):
    def __init__(self):
        self.out = None  # نخزن مخرجات ال forward

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # قانون مشتقة دالة sigmoid
        dx = dout * (1.0 - self.out) * self.out
        return dx
    