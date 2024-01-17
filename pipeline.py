
from utils import data_feed, RGB2GRAY
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


print("-------------- Import Data --------------")

X_train, y_train, X_val, y_val, X_test, y_test = data_feed()

print("-------------- Convert images to grey scale images --------------")

X_train_grey = RGB2GRAY(X_train)
X_val_grey = RGB2GRAY(X_val)
X_test_grey = RGB2GRAY(X_test)

print("Training shape", X_train_grey.shape)
print("Validation shape", X_val_grey.shape)
print("Testing shape", X_test_grey.shape)

print("-------------- Reshape data --------------")

X_train_reshaped = X_train_grey.reshape(X_train_grey.shape[0], X_train_grey.shape[1]*X_train_grey.shape[2])
X_val_reshaped = X_val_grey.reshape(X_val_grey.shape[0], X_val_grey.shape[1]*X_val_grey.shape[2])
X_test_reshaped = X_test_grey.reshape(X_test_grey.shape[0], X_test_grey.shape[1]*X_test_grey.shape[2])

# Change integers to 32-bit floating point numbers
X_train_reshaped = X_train_reshaped.astype('float32')
X_val_reshaped = X_val_reshaped.astype('float32')
X_test_reshaped = X_test_reshaped.astype('float32')

print("Training shape", X_train_reshaped.shape)
print("Validation shape", X_val_reshaped.shape)
print("Testing shape", X_test_reshaped.shape)

print("-------------- Scale data --------------")

X_train_st = StandardScaler().fit_transform(X_train_reshaped)
X_val_st = StandardScaler().fit_transform(X_val_reshaped)
X_test_st = StandardScaler().fit_transform(X_test_reshaped)

print("-------------- Perform PCA --------------")

'''
This part of the code is left for illustration purposes but there is no need to run it since the output plot is stored in 
the plots folder.

pca = PCA(n_components=X_train_st.shape[1])
pca_data = pca.fit_transform(X_train_st)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cum_var_explained = np.cumsum(percentage_var_explained)

plt.plot(cum_var_explained, linewidth=2)
plt.xlabel("n_components")
plt.ylabel("Cumulative_explained_variance")
plt.savefig("plots/PCA.png")
'''

pca = PCA(n_components=.95)
pca.fit(X_train_st)

print(f'Total number of components used after PCA : {pca.n_components_}')

X_train_pca = pca.transform(X_train_st)
X_val_pca = pca.transform(X_val_st)
X_test_pca = pca.transform(X_test_st)

print(f'Training shape: {X_train_pca.shape}')
print(f'Validation shape: {X_val_pca.shape}')
print(f'Test shape: {X_test_pca.shape}')

print("End")