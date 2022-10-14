import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras_tuner as kt
import joblib
import math
from tensorflow import keras

# Importing the dataset
dataset = pd.read_csv('flight_data_2018_to_2022_departuters.csv')
dataset_delayed_0 = dataset.loc[dataset['DepDel15'] == 0]
dataset_delayed_1 = dataset.loc[dataset['DepDel15'] == 1]

sample_size = 1/2
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

# Training the ANN model on the Training set
# Accuracy: 99.00 %
def CreateClassifier(X_train, y_train):
    classifier = keras.models.Sequential()
    classifier.add(keras.layers.Dense(units=6, activation='relu'))
    classifier.add(keras.layers.Dense(units=6, activation='relu'))
    classifier.add(keras.layers.Dense(units=6, activation='relu'))
    classifier.add(keras.layers.Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = int(math.sqrt(len(X_train))), epochs = 300, verbose=1)
    return classifier

classifier = CreateClassifier(X_train, y_train)

# Save classifier
classifier.save('ClassifierANN')

# Load classifier
classifier = keras.models.load_model("ClassifierANN")

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cvscores = []

for train, test in kfold.split(X_train, y_train):
	model = CreateClassifier(X_train[train], y_train[train])
	scores = model.evaluate(X_train[test], y_train[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
 
print("Accuracy: {:.2f} %".format(np.mean(cvscores)))
print("Standard Deviation: {:.2f} %".format(np.std(cvscores)))

# Searching the best parameters
def model_builder(hp):
  model = keras.models.Sequential()
  hp_units = hp.Int('units', min_value=2, max_value=10, step=1)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(units=1, activation='sigmoid'))
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


