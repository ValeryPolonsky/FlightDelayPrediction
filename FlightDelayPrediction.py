import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('flight_data_2018_to_2022_departuters.csv')
dataset_delayed_0 = dataset.loc[dataset['DepDel15'] == 0]
dataset_delayed_1 = dataset.loc[dataset['DepDel15'] == 1]

sample_size = 0.1/2
dataset_delayed_0_sample_size = int(len(dataset.index) * sample_size)
dataset_delayed_1_sample_size = int(len(dataset.index) * sample_size)

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



# Training the Logistic Regression model on the Training set
# Accuracy ~ 62%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Training the K-NN model on the Training set
# Accuracy ~ 63%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 100, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Training the SVM model on the Training set
# Accuracy ~ 80%
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Training the Kernel SVM model on the Training set
# Accuracy ~ 88%
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Training the Naive Bayes model on the Training set
# Accuracy ~ 56%
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Training the Decision Tree Classification model on the Training set
# Accuracy ~ 95%
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Training the Random Forest Classification model on the Training set
# Accuracy ~ 92%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 128, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Training the ANN model on the Training set
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)



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


