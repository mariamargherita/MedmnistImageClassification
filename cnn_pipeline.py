
from utils import data_feed, RGB2GRAY, plot_accuracy_loss
from cnn_model import cnn_model
import random
from datetime import datetime

print("-------------- Import Data --------------")

X_train, y_train, X_val, y_val, X_test, y_test = data_feed()

print("-------------- Normalize Data --------------")

X_train = X_train / 255.
X_val = X_val / 255.
X_test = X_test / 255.

print("-------------- CNN Model --------------")

random.seed(1234)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

num_classes = len(set(y_train))
input_shape = X_train.shape[1:]

model = cnn_model(input_shape, num_classes)
history = model.fit_model(X_train, y_train, X_val, y_val, timestamp, batch_size=128, epochs=50) # 0.88 on val acc with batch size 64

plot_accuracy_loss(history)

test_scores = model.model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1]) #0.81 with batch 64

print("-------------- End of pipeline --------------")