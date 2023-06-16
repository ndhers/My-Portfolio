import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Make the dataset
N = 1000000
X = np.random.random((N, 2)) * 6 - 3 # uniformly distributed between (-3, +3)
Y = (1-X[:,0])**2 + 100* ((X[:,1]-X[:,0]**2))**2


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(32, input_shape=(2,), activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1)
])

opt = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=opt, loss='mse')
r = model.fit(X, Y, epochs=40)

#plt.plot(r.history['loss'], label='loss')


# Let's test by checking whether (1,1) does in fact give us 0 with the surrogate model

test = np.array([1,1])
test = test.reshape(1,2)
Testresult = model.predict(test)

#print('The predicted value for the exact solution is: {}'.format(Testresult))

# Loop through the predictions and find the minimum value: output corresponding x and y

Yhat = model.predict(X)

k = 50

for i in range(0,1000000,1):
	if Yhat[i] <= k:
		k = Yhat[i]
		index = i

#print('The neural network thinks ({0},{1}) is the solution to the global optimization problem'.format(X[index,0],X[index,1]))


# Let's evaluate the model for an even greater dataset

Nb = 10000000

min_dataset = np.random.random((Nb, 2)) * 6 - 3

Yhat2 = model.predict(min_dataset)

p = 50

for j in range(0,10000000,1):
	if Yhat2[j] <= p:
		p = Yhat2[j]
		index2 = j

#print('The neural network thinks ({0},{1}) is the solution to the global optimization problem'.format(min_dataset[index2,0],min_dataset[index2,1]))


'''

# Plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# surface plot
line = np.linspace(-3, 3, 50)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
#plt.show()

'''
