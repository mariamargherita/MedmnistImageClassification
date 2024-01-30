
from utils import data_feed, augment_data, fit_grid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from joblib import Parallel, delayed

print("-------------- Import data --------------")

X_train_base, y_train_base, X_val, y_val, X_test, y_test = data_feed(data_flag='breastmnist')

print("Training shape", X_train_base.shape)
print("Validation shape", X_val.shape)
print("Testing shape", X_test.shape)

print("-------------- Train random forest with best parameters on all training set --------------")

X_train = np.vstack((X_train_base, X_val))
y_train = np.hstack((y_train_base, y_val))

print("Training predictors shape", X_train.shape)
print("Training target shape", y_train.shape)
print("Testing predictors shape", X_test.shape)
print("Testing target shape", y_test.shape)

print("-------------- Perform data augmentation on training data --------------")

X_train_aug, y_train_aug = augment_data(X_train, y_train, n_augs=9)

print("Training predictors shape", X_train_aug.shape)
print("Training target shape", y_train_aug.shape)

print("-------------- Reshape data --------------")

X_train_reshaped = X_train_aug.reshape(X_train_aug.shape[0], X_train_aug.shape[1] * X_train_aug.shape[2])
#X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# Change integers to 32-bit floating point numbers
X_train_reshaped = X_train_reshaped.astype('float32')
#X_val_reshaped = X_val_reshaped.astype('float32')
X_test_reshaped = X_test_reshaped.astype('float32')

print("Training shape", X_train_reshaped.shape)
#print("Training shape", X_val_reshaped.shape)
print("Testing shape", X_test_reshaped.shape)

print("-------------- Scale data --------------")

X_train_st = StandardScaler().fit_transform(X_train_reshaped)
#X_val_st = StandardScaler().fit_transform(X_val_reshaped)
X_test_st = StandardScaler().fit_transform(X_test_reshaped)

print("-------------- Perform PCA --------------")

'''
This part of the code is left for illustration purposes but there is no need to run it since the output plot is stored in 
the plots folder.

pca = PCA(n_components=min(X_train_st.shape[0], X_train_st.shape[1]))
pca_data = pca.fit_transform(X_train_st)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cum_var_explained = np.cumsum(percentage_var_explained)

plt.plot(cum_var_explained, linewidth=2)
plt.xlabel("n_components")
plt.ylabel("Cumulative_explained_variance")
plt.savefig("plots/PCA.png")
'''

pca = PCA(n_components=.99)
pca.fit(X_train_st)

print(f'Total number of components used after PCA : {pca.n_components_}')

X_train_pca = pca.transform(X_train_st)
#X_val_pca = pca.transform(X_val_st)
X_test_pca = pca.transform(X_test_st)

print(f'Training shape: {X_train_pca.shape}')
#print(f'Training shape: {X_val_pca.shape}')
print(f'Test shape: {X_test_pca.shape}')

print("-------------- Parallelized Grid Search - Random forest classifier --------------")

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, None],
    'max_features': ["sqrt", "log2"],
    'n_estimators': [200, 300, 400, 500]
}

# Create base model
'''
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=3,
                           verbose=10)
'''

# Generate all possible parameter combinations
param_combinations = list(ParameterGrid(param_grid))

grid_results = Parallel(n_jobs=-1)(delayed(fit_grid)(X_train_pca, y_train_aug, params) for params in tqdm(param_combinations))

# Perform grid search in parallel with progress tracking
sorted_grid_results = sorted(grid_results, key=lambda x: x[0], reverse=True)
best_grid = sorted_grid_results[0]

print(f"CV validation accuracy for best parameters: {best_grid[0]}")
print(f"The best parameters are: {best_grid[1]}")

print("-------------- Make predictions on test set with best model obtained from grid search --------------")

best_model = RandomForestClassifier(**best_grid[1], verbose=1)
best_model.fit(X_train_pca, y_train_aug)
y_pred = best_model.predict(X_test_pca)
print(f"Test accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Test accuracy: {classification_report(y_test, y_pred)}")



print("-------------- End of pipeline --------------")