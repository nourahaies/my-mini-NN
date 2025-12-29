# trainer.py
import numpy as np

class Trainer:
    def __init__(self, network, optimizer):
        # الشبكة العصبية
        self.network = network
        # خوارزمية التحديث
        self.optimizer = optimizer
        # لتخزين آخر دقة اختبار (مهم للـ hyperparameter tuning)
        self.test_accuracy = None

    def train_step(self, x, y):
        # =====================================
        # Forward
        # =====================================
        loss = self.network.loss(x, y)

        # =====================================
        # Backward
        # =====================================
        grads = self.network.gradient(x, y)

        # =====================================
        # Update Weights
        # =====================================
        params = self.network.params()
        self.optimizer.update(params, grads)

        return loss

    def fit(self, x_train, y_train,
            x_test=None, y_test=None,
            epochs=10, batch_size=32):

        data_size = x_train.shape[0]
        iter_per_epoch = max(data_size // batch_size, 1)

        for epoch in range(epochs):
            # خلط البيانات
            idx = np.random.permutation(data_size)
            x_train = x_train[idx]
            y_train = y_train[idx]

            loss_sum = 0

            for i in range(iter_per_epoch):
                batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                batch_y = y_train[i * batch_size:(i + 1) * batch_size]

                loss = self.train_step(batch_x, batch_y)
                loss_sum += loss

            avg_loss = loss_sum / iter_per_epoch
            if x_test is not None and y_test is not None:
               acc = self.network.accuracy(x_test, y_test)
               self.test_accuracy = acc
               print(
                 f"Epoch {epoch + 1}/{epochs} | "
                 f"Loss: {avg_loss:.4f} | "
                 f"Test Acc: {acc:.4f}"
                )
            else:
               print(
                 f"Epoch {epoch + 1}/{epochs} | "
                 f"Loss: {avg_loss:.4f}"
                )

