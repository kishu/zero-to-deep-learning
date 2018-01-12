import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import make_moons

# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons
# Make two interleaving half circles
# X : array of shape [n_samples, 2]
#     The generated samples.
# y : array of shape [n_samples]
#     The integer labels (0 or 1) for class membership of each sample.
X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)

# plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
# plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
# plt.legend(['0', '1'])
# plt.show()

from sklearn.model_selection import train_test_split

# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Split arrays or matrices into random train and test subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

# The Sequential model is a linear stack of layers.
# Dense : Just your regular densely-connected NN layer.
# keras.layers.Dense(units, **options)
#  units: Positive integer, dimensionality of the output space.
# model = Sequential()
# model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
# Configures the model for training.
# compile(self, optimizer, loss, ...)
#  metrics: List of metrics to be evaluated by the model during training and testing
# model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=200, verbose=1)
# results = model.evaluate(X_test, y_test)
# print("The Accuracy score on the Train set is: \t{:0.3f}".format(results[1]))

def plot_decision_boundary(model, X, y):
    # if X = [[7, 2], [1, 4], [5, 6]]
    # X.min(axis=0) => [1, 2]
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    # amin에서 amax 까지 고른 간격으로 숫자 101개 만들기
    # make_moon 데이터의 가장 작은 x, y 에서 가장 큰 x,y 까지 101개 좌표 생성
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    # 그리드 포인터 생성
    aa, bb = np.meshgrid(hticks, vticks)
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.c_.html
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ravel.html
    ab = np.c_[aa.ravel(), bb.ravel()]

    c = model.predict(ab)
    cc = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    # 윤곽을 그린다.
    plt.contourf(aa, bb, cc, cmap='bwr', alpha=0.2)
    plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
    plt.legend(['0', '1'])
    plt.show()

#plot_decision_boundary(model, X, y)

#
# Deep Model
#

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='tanh'))
model.add(Dense(2, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

from sklearn.metrics import accuracy_score, confusion_matrix
y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

print("The Accuracy score on the Train set is: \t{:0.3f}".format(accuracy_score(y_train, y_train_pred)))
print("The Accuracy score on the Test set is: \t{:0.3f}".format(accuracy_score(y_test, y_test_pred)))

plot_decision_boundary(model, X, y)





    

