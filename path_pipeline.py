
from utils import data_feed, RGB2GRAY, plot_accuracy_loss
from cnn_model import cnn_model, full_cnn_model
import random
from datetime import datetime


print("-------------- Import Data --------------")

X_train, y_train, X_val, y_val, X_test, y_test = data_feed(data_flag='pathmnist')

'''
print("-------------- Convert images to grey scale images --------------")

X_train = RGB2GRAY(X_train)
X_val = RGB2GRAY(X_val)
X_test = RGB2GRAY(X_test)

X_train = X_train.reshape(X_train.shape + (1,))
X_val = X_val.reshape(X_val.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

print("Training shape", X_train.shape)
print("Validation shape", X_val.shape)
print("Testing shape", X_test.shape)
'''

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
history = model.fit_model(X_train, y_train, X_val, y_val, timestamp, batch_size=128, epochs=50)

plot_accuracy_loss(history)

test_scores = model.model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

'''
print("-------------- Train model with best parameters on full training set --------------")

X_train_full = np.vstack((X_train, X_val))
y_train_full = np.hstack((y_train, y_val))

random.seed(1234)

epochs = 50
batch_size = 128
num_classes = len(set(y_train))
input_shape = X_train.shape[1:]

final_model = full_cnn_model(input_shape, num_classes)
history = final_model.fit_model(X_train, y_train, batch_size=batch_size, epochs=epochs)

print("------------- Make prediction on test set ----------------")

y_predict = history.model.predict(X_test)
y_pred = np.asarray([np.argmax(y_predict[i]) for i in range(0, len(y_predict))])

final_accuracy = accuracy_score(y_test, y_pred)
print(f"Model loss and accuracy on test set (model trained on full train set): {final_accuracy}")
'''

print("-------------- End of pipeline --------------")