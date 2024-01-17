
"""
!!! This has to be run OUTSIDE the project environment !!!
In our environment we have tensorflow while the medmnist data extraction is optimized on pyTorch so it is better to keep
the data extraction separate
"""


import torch
import torch.utils.data as data
import medmnist
from medmnist import INFO
import pickle
import numpy as np

print("---------------------- Import, preprocess and load data ----------------------")

# Import data
data_flag = 'pathmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# Load data
train_dataset = DataClass(split='train', download=download)
val_dataset = DataClass(split='val', download=download)
test_dataset = DataClass(split='test', download=download)

# Set seed for reproducibility
torch.manual_seed(123)

# Encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# Get out images and information
X_train = train_loader.dataset.imgs
y_train = np.reshape(train_loader.dataset.labels, (len(train_loader.dataset.labels)))
X_val = val_loader.dataset.imgs
y_val = np.reshape(val_loader.dataset.labels, (len(val_loader.dataset.labels)))
X_test = test_loader.dataset.imgs
y_test = np.reshape(test_loader.dataset.labels, (len(test_loader.dataset.labels)))

print("---------- Store data ----------")

with open('X_train.pickle', 'wb') as f:
    pickle.dump(np.array(X_train), f, pickle.HIGHEST_PROTOCOL)

with open('y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)

with open('X_val.pickle', 'wb') as f:
    pickle.dump(X_val, f, pickle.HIGHEST_PROTOCOL)

with open('y_val.pickle', 'wb') as f:
    pickle.dump(y_val, f, pickle.HIGHEST_PROTOCOL)

with open('X_test.pickle', 'wb') as f:
    pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)

with open('y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)


print("---------- End ----------")