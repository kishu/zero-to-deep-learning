import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from sklearn.metrics import accuracy_score

df = pd.read_csv('data/user_visit_duration.csv')

model = Sequential()
model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])

X = df[['Time (min)']].values
y = df['Buy'].values

print("fitting")
model.fit(X, y, epochs=25, verbose=0)

# ax = df.plot(kind='scatter', x='Time (min)', y='Buy', 
#             title='Purchase behavior VS time spent on site')

y_pred = model.predict(X)
y_class_pred = y_pred > 0.5 #[[ True], [False], [ True] ...]

print("weight", model.get_weights())
print("accuracy score {:0.3f}".format(accuracy_score(y, y_class_pred)))

# temp = np.linspace(0, 4)
# temp_class = model.predict(temp) > 0.5
# ax.plot(temp, model.predict(temp), color='orange')
# ax.plot(temp, temp_class, color='purple')
# plt.legend(['model', 'class', 'data'])
# plt.show()

#
# Train / Test split
#
print("\n\n### Train / Test split")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = model.get_weights() # [array([[ 1.91573226]], dtype=float32), array([-2.83298182], dtype=float32)]
params = [np.zeros(w.shape) for w in params] # [array([[ 0.]]), array([ 0.])]
model.set_weights(params)

print("weight", model.get_weights())
print("accuracy score {:0.3f}".format(accuracy_score(y, model.predict(X) > 0.5)))

print("fitting train set")
model.fit(X_train, y_train, epochs=25, verbose=0)

print("train accuracy score {:0.3f}".format(accuracy_score(y_train, model.predict(X_train) > 0.5)))
print("test accuracy score {:0.3f}".format(accuracy_score(y_test, model.predict(X_test) > 0.5)))


#
# Cross Validation
#
print('\n\n###Cross Validation')

from keras.wrappers.scikit_learn import KerasClassifier

def build_logistic_regression_model():
    print('build_logistic_regression_model')
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
    model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_logistic_regression_model, epochs=25, verbose=0)

from sklearn.model_selection import cross_val_score, KFold

# Provides train/test indices to split data in train/test sets.
cv = KFold(3, shuffle=True)

# cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’)
# Evaluate a score by cross-validation
scores = cross_val_score(model, X, y, cv=cv)

print('scores', scores)
print('cross validation accuracy is {:0.4f} +- {:0.4f}'.format(scores.mean(), scores.std()))

#
# Confusion Matrix(오류 매트릭스)
# http://operatingsystems.tistory.com/entry/Data-Mining-Measuring-a-classification-model
#
print('\n\n###Confusion Matrix')
from sklearn.metrics import confusion_matrix

# Compute confusion matrix to evaluate the accuracy of a classification
cm = confusion_matrix(y, y_class_pred)
print("confusion matrix", cm)

def pretty_confusion_matrix(y_true, y_pred, labels=["False", "true"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ['Predicted ' + l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df

df = pretty_confusion_matrix(y, y_class_pred, ['Not Buy', 'Buy'])
print('pretty_confusion_matrix')
print(df.head())

from sklearn.metrics import precision_score, recall_score, f1_score

print("\n")
print("Precision:\t{:0.3f}".format(precision_score(y, y_class_pred)))
print("Recall: \t{:0.3f}".format(recall_score(y, y_class_pred)))
print("F1 Score: \t{:0.3f}".format(f1_score(y, y_class_pred)))

from sklearn.metrics import classification_report
print("\n")
print(classification_report(y, y_class_pred))

