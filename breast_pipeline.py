
from utils import data_feed, RGB2GRAY, plot_accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  #Random Forest algorithm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import joblib
from joblib import Parallel, delayed
from dask.distributed import Client, LocalCluster
from sklearn.model_selection import cross_val_score

'''
Next steps:
- try other models on this 
 
'''

print("-------------- Import Data --------------")

X_train, y_train, X_val, y_val, X_test, y_test = data_feed(data_flag='breastmnist')

print("-------------- Reshape data --------------")

X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2])
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

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

print("-------------- Random forest classifier --------------")

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, None],
    'max_features': ["sqrt", "log2", None],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 4, 6, 8, 10, 12],
    'n_estimators': [100, 200, 300, 400]
}

# Create a based model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=3,
                           verbose=2)

Parallel(n_jobs=-1, verbose=11)(grid_search.fit(X_train_pca, y_train))

y_pred = rf.predict(X_test_pca)
print ("Classification Report")
print(classification_report(y_test, y_pred))

print("-------------- End of pipeline --------------")