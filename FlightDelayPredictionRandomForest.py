import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('flight_data_2018_to_2022_departuters.csv')
dataset_delayed_0 = dataset.loc[dataset['DepDel15'] == 0]
dataset_delayed_1 = dataset.loc[dataset['DepDel15'] == 1]

sample_size = 0.5/2
dataset_delayed_0_sample_size = int(len(dataset.index) * sample_size)
dataset_delayed_1_sample_size = int(len(dataset.index) * sample_size)

if (len(dataset_delayed_0) < dataset_delayed_0_sample_size):
    dataset_delayed_0_sample_size = len(dataset_delayed_0)
    
if (len(dataset_delayed_1) < dataset_delayed_1_sample_size):
    dataset_delayed_1_sample_size = len(dataset_delayed_1)

dataset_delayed_0 = dataset_delayed_0.sample(n = dataset_delayed_0_sample_size)
dataset_delayed_1 = dataset_delayed_1.sample(n = dataset_delayed_1_sample_size)

dataset_merged = pd.concat([dataset_delayed_0, dataset_delayed_1])
dataset_merged = dataset_merged.sample(frac=1).reset_index(drop=True)

X = dataset_merged.iloc[:, :-1].values
y = dataset_merged.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer_numeric = SimpleImputer(missing_values=np.nan, strategy='mean')
numeric_columns = [2,7,9,10,11,12,13,15,16,17]
imputer_numeric.fit(X[:, numeric_columns])
X[:, numeric_columns] = imputer_numeric.transform(X[:, numeric_columns])

imputer_string = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
string_columns = [0,1,3,4,5,6,8,14]
imputer_string.fit(X[:, string_columns])
X[:, string_columns] = imputer_string.transform(X[:, string_columns])

imputer_constant = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imputer_constant.fit(y[:,np.newaxis])
y = imputer_constant.transform(y[:,np.newaxis]).flatten()

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
X[:,1] = le.fit_transform(X[:,1])
X[:,3] = le.fit_transform(X[:,3])
X[:,4] = le.fit_transform(X[:,4])
X[:,5] = le.fit_transform(X[:,5])
X[:,6] = le.fit_transform(X[:,6])
X[:,8] = le.fit_transform(X[:,8])
X[:,14] = le.fit_transform(X[:,14])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random Forest Classification model on the Training set
# Accuracy: 92.90 %
# Standard Deviation: 0.37 %
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 800, 
                                    criterion = 'entropy', 
                                    random_state = 0,
                                    min_samples_split = 10,
                                    min_samples_leaf = 1,
                                    min_weight_fraction_leaf = 0)
classifier.fit(X_train, y_train)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10],
               'criterion': ['entropy'],
               'min_samples_split': [10],
               'min_samples_leaf': [1],
               'min_weight_fraction_leaf': [0],
               }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


