import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles

# Make a large circle containing a smaller circle in 2d.
# A simple toy dataset to visualize clustering and classification algorithms.

# X : array of shape [n_samples, 2]. The generated samples.
# y : array of shape [n_samples]. The integer labels (0 or 1) for class membership of each sample.
X, y = make_circles(n_samples = 1000,
                    noise = 0.1,
                    factor = 0.2,
                    random_state = 0)

# figsize : tuple of integers, optional, default: None
#    width, height in inches. If not provided, defaults to rc figure.figsize.
# plt.figure(figsize = (5, 5))
# plt.plot(X[y == 0, 0], X[y == 0, 1], 'ob', alpha = 0.5)
# plt.plot(X[y == 1, 0], X[y == 1, 1], 'xr', alpha = 0.5)
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# plt.legend(['0', '1'])
# plt.title("Blue circles and Red crosses")
# plt.show()

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# The core data structure of Keras is a model, a way to organize layers.
# The simplest type of model is the Sequential model, a linear stack of layers.
model = Sequential()

# Stacking layers is as easy as .add()
# What is Dense?
# The dense layer is fully connected layer, so all the neurons in a layer are connected to those in a next layer.
# https://www.quora.com/In-Keras-what-is-a-dense-and-a-dropout-layer
model.add(Dense(4, input_shape = (2, ), activation = 'tanh'))
model.add(Dense(1, activation = 'sigmoid'))

# Once your model looks good, configure its learning process with .compile()
# SGD: Stochastic gradient descent optimizer. See http://sanghyukchun.github.io/74/
model.compile(SGD(lr = 0.5), 'binary_crossentropy', metrics = ['accuracy'])

# epoch: a particular period of time marked by distinctive features, events, etc.   
model.fit(X, y, epochs = 20)





# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# Returns num evenly spaced samples, calculated over the interval [start, stop].
hticks = np.linspace(-1.5, 1.5, 101) # [-1.5  -1.47 ... 1.47, 1.5]
vticks = np.linspace(-1.5, 1.5, 101) # [-1.5  -1.47 ... 1.47, 1.5]
# numpy.meshgrid(*xi, **kwargs)
# Return coordinate matrices from coordinate vectors
aa, bb = np.meshgrid(hticks, vticks)
# numpy.c_ = <numpy.lib.index_tricks.CClass object>
# Translates slice objects to concatenation along the second axis.
# numpy.ravel(a, order='C')
# Return a contiguous flattened array.
ab = np.c_[aa.ravel(),  bb.ravel()]
# predict(self, x, batch_size=32, verbose=0)
# Generates output predictions for the input samples.
c = model.predict(ab)
# numpy.reshape(a, newshape, order='C')
# Gives a new shape to an array without changing its data.
cc = c.reshape(aa.shape)


plt.figure(figsize = (5, 5))
plt.contourf(aa, bb, cc, cmap = 'bwr', alpha = 0.2)
plt.plot(X[y == 0, 0], X[y == 0, 1], 'ob', alpha = 0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha = 0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(['0', '1'])
plt.title("Blue circles and Red crosses")
plt.show()
