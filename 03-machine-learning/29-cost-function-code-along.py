import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data/weight-height.csv')
# print(df.head())
# df.plot(kind='scatter',
#         x='Height',
#         y='Weight',
#         title='Weight and Height in adults')

def line(x, w=0, b=0):
    return x * w + b

# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)[source]
# Return evenly spaced numbers over a specified interval.
# 균일한 가격의 숫자를 반환
# x = np.linspace(55, 80, 100) # 55에서 80사이 균일한 숫자 100개를 반환
# yhat = line(x, w=0, b=0)

def mean_squared_error(y_true, y_pred):
    s = (y_true - y_pred) ** 2
    return s.mean()

X = df['Height'].values
y_true = df['Weight'].values

#y_pred = line(X, 2.9, 0)
#cost = mean_squared_error(y_true, y_pred)
#print(cost)

# plt.plot(X, y_pred, color='red', linewidth=3)
# plt.show()

# plt.figure(figsize=(10, 5))

# we are going to draw 2 plots in the same figure
# first plot, data and a few lines
# ax1 = plt.subplot(121)
# df.plot(kind='scatter',
#         x='Height',
#         y='Weight',
#         title='Weight and Height in adults',
#         ax=ax1)

# let's explore the cost function for a few values of between -100 and +150
# bbs = np.array([-100, -50, 0, 100, 150])
# mses = [] # we will append the values of the cost hear, for each line
# for b in bbs:
#     y_pred = line (X, w=2, b=b)
#     mse = mean_squared_error(y_true, y_pred)
#     mses.append(mse)
#     plt.plot(X, y_pred)

# second plot: Cost function
# ax2 = plt.subplot(122)
# plt.plot(bbs, mses, 'o-')
# plt.title('Cost as a function of b')
# plt.xlabel('b')

# plt.show()


#
# Linear Regression with Keras
#
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
# model.summary()

model.compile(Adam(lr=0.8), 'mean_squared_error')
# model.fit(X, y_true, epochs=40)

# W, B = model.get_weights()
# print(W, B)

# y_pred = model.predict(X)

# df.plot(kind='scatter',
#     x='Height',
#     y='Weight',
#     title='Weight and Height in adults')
# plt.plot(X, y_pred, color='red')
# plt.show()

#
# Evaluation Model Performance
#
from sklearn.metrics import r2_score
# print("R2S is {:0.3f}".format(r2_score(y_true, y_pred)))

#
# Train Test Split
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2)

# W[0, 0] = 0.0
# B[0] = 0

# model.set_weights((W, B))
model.fit(X_train, y_train, epochs=50)

y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

from sklearn.metrics import mean_squared_error as mse
print("MSE on the Train set is:\t{:0.1f}".format(mse(y_train, y_train_pred)))
print("MSE on the Test set is :\t{:0.1f}".format(mse(y_test, y_test_pred)))
print("R2S on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("R2S on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))