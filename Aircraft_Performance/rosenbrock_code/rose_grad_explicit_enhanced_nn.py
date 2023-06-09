# -*- coding: utf-8 -*-
"""rose_grad_nn_v4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1br5kdgL8woxm5Plw_znKv2BTr8xsZye0
"""

# Dataset building

import random

N = 1000
X = []
max_z = 0
min_z = 1000

for j in range(N):
  X_local = []
  for i in range(2):
    X_rand = random.uniform(-3, 3) # uniformly distributed between (-3, +3)
    X_local.append(X_rand)
  X_local.append((1-X_local[0])**2 + 100* ((X_local[1]-X_local[0]**2))**2)
  #X_local.append(X_local[0]**2+2*X_local[1]**2)
  if X_local[2] > max_z:
    max_z = X_local[2]
  if X_local[2] < min_z:
    min_z = X_local[2]
  X.append(X_local)

#print(max_z)

for j in range(N):
  X[j][2] = (X[j][2]-min_z)/(max_z-min_z)

max_grad_x = 0
max_grad_y = 0
min_grad_x = 1000
min_grad_y = 1000

for k in range(N):
  X[k].append(2*(200*X[k][0]**3-200*X[k][0]*X[k][1]+X[k][0]-1))
  #X[k].append(2*X[k][0])
  if 200*X[k][0]**3-200*X[k][0]*X[k][1]+X[k][0]-1 > max_grad_x:
    max_grad_x = 200*X[k][0]**3-200*X[k][0]*X[k][1]+X[k][0]-1
  if 200*X[k][0]**3-200*X[k][0]*X[k][1]+X[k][0]-1 < min_grad_x:
    min_grad_x = 200*X[k][0]**3-200*X[k][0]*X[k][1]+X[k][0]-1
  X[k].append(200*(X[k][1]-X[k][0]**2))
  #X[k].append(4*X[k][1])
  if 200*(X[k][1]-X[k][0]**2)> max_grad_y:
    max_grad_y = 200*(X[k][1]-X[k][0]**2)
  if 200*(X[k][1]-X[k][0]**2)< min_grad_y:
    min_grad_y = 200*(X[k][1]-X[k][0]**2)

for j in range(N):
  X[j][3] = (X[j][3]-min_grad_x)/(max_grad_x-min_grad_x)*0.25
  X[j][4] = (X[j][4]-min_grad_y)/(max_grad_y-min_grad_y)*0.25

training_dataset = X

#for i in range(6):
#  print(training_dataset[i][0:2])

import math
from math import exp
from random import seed
from random import random
import matplotlib
from matplotlib import pyplot
 
# Let's start by setting up the network (i.e. number of layers and neurons per layer)
def network_setup(number_input_nodes, number_hlayer1_nodes, number_hlayer2_nodes, number_output_nodes):
  network = []
  hidden_layer1 = [{'weights':[random() for i in range(number_input_nodes)]} for i in range(number_hlayer1_nodes)]
  network.append(hidden_layer1)
  hidden_layer2 = [{'weights':[random() for i in range(number_hlayer1_nodes)]} for i in range(number_hlayer2_nodes)]
  network.append(hidden_layer2)
  output_layer = [{'weights':[random() for i in range(number_hlayer2_nodes)]} for i in range(number_output_nodes)]
  network.append(output_layer)
  return network

# Let's define a function to evaluate the response signal that is fed to the network (feedfoward)
def forward_propagation_response(network, sample):
	inputs = sample
	i = 0
	for layer in network:
		following_inputs = []
		for neuron in layer:
			h = activate(neuron['weights'], inputs)
			if i == 2:
				neuron['output_response'] = activation_function_sigmoid(h)
			else:
				neuron['output_response'] = activation_function_sigmoid(h)				
			following_inputs.append(neuron['output_response'])
		inputs = following_inputs
		i = i+1
	return inputs

def forward_propagation_gradient(network, sample):
	inputs = sample
	i = 0
	for layer in network:
		following_inputs = []
		following_inputs_grad_x = []
		following_inputs_grad_y = []
		for neuron in layer:
			h = activate(neuron['weights'], inputs)
			if i == 2:
				v = activation_function_sigmoid(h)
				neuron['output_gradient_x'] = activation_function_deriv_sigmoid(v)*neuron['gradient_h_x']
				neuron['output_gradient_y'] = activation_function_deriv_sigmoid(v)*neuron['gradient_h_y']
			else:
				v = activation_function_sigmoid(h)
				neuron['output_gradient_x'] = activation_function_deriv_sigmoid(v)*neuron['gradient_h_x']
				neuron['output_gradient_y'] = activation_function_deriv_sigmoid(v)*neuron['gradient_h_y']				
			following_inputs.append(v)
			following_inputs_grad_x.append(neuron['output_gradient_x'])
			following_inputs_grad_y.append(neuron['output_gradient_y'])
		inputs = following_inputs
		grad_x = following_inputs_grad_x
		grad_y = following_inputs_grad_y
		i = i+1
	return [grad_x,grad_y]

def h_gradient_xp(network,inputs):
	net_length = len(network)
	input_length = len(inputs)
	for i in range(net_length):
		grad_list_x = []
		grad_list_y = []
		layer = network[i]
		if i == 0: # i.e. first layer
			for neuron in network[i]:
				for j in range(input_length):
					if j == 0:
						grad_value_x = neuron['weights'][0]
						grad_value_y = 0.0
					else:
						grad_value_y = neuron['weights'][1]
						grad_value_x = 0.0
				grad_list_x.append(grad_value_x)
				grad_list_y.append(grad_value_y)			

		else:
				
			for neuron in network[i]:
				grad_value_x = 0.0
				grad_value_y = 0.0
				k = 0
				for neuron_prec in network[i-1]:
					grad_value_x += (neuron['weights'][k] * neuron_prec['gradient_h_x']) * activation_function_deriv_sigmoid(neuron_prec['output_response'])
					grad_value_y += (neuron['weights'][k] * neuron_prec['gradient_h_y']) * activation_function_deriv_sigmoid(neuron_prec['output_response'])
					k = k+1
				grad_list_x.append(grad_value_x)
				grad_list_y.append(grad_value_y)
					

		
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['gradient_h_x'] = grad_list_x[j]
			neuron['gradient_h_y'] = grad_list_y[j]

# Let's define a function that computes the input to each neuron
def activate(weights, inputs):
	#bias = weights[-1]
	h = 0.0
	for i in range(len(weights)):
		h += weights[i] * inputs[i]
	return h
 
# Let's define the activation function, here we use the sigmoid
def activation_function_sigmoid(h):
	G = 1.0 / (1.0 + exp(-h))
	return G
 
 # Let's define the activation function, here we use the tanh for output between [-1,1]
def activation_function_tanh(h):
	G = math.sinh(h)/math.cosh(h)
	return G

 
# For gradient descent, we need derivative of our sigmoid function g in hidden layers
def activation_function_deriv_sigmoid(output):
	dG = output * (1.0 - output)
	return dG

# For gradient descent, we need derivative of our tanh function g in last layer
def activation_function_deriv_tanh(output):
	dG = (1.0 - output**2)
	return dG

#Second derivatives:
def activation_function_sec_deriv_sigmoid(output):
	d2G = output*(1-output)*(1-2*output)
	return d2G

def activation_function_sec_deriv_tanh(output):
	d2G = -2*output*(1-output**2)
	return d2G

# Let's define a function for the backpropagation algorithm and to store the delta_response in each neuron
def backpropagation(network, target):
	net_length = len(network)
	for i in reversed(range(net_length)):
		error_list = []
		layer = network[i]
		if i == len(network)-1: # i.e. last layer
			for j in range(len(layer)):
				neuron = layer[j]
				error_list.append(target - neuron['output_response'])
		else:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				error_list.append(error)

		for j in range(len(layer)):
			neuron = layer[j]
			if i == len(network)-1:
				neuron['delta'] = error_list[j] * activation_function_deriv_sigmoid(neuron['output_response'])
			else:
				neuron['delta'] = error_list[j] * activation_function_deriv_sigmoid(neuron['output_response'])

def backpropagation_gradient(network,target):
	gradient_x_tar = target[-2]
	gradient_y_tar = target[-1]
	net_length = len(network)
	for i in reversed(range(net_length)):
		gradient_x_list1 = []
		gradient_y_list1 = []
		gradient_x_list2 = []
		gradient_y_list2 = []
		gradient_x_list11 = []
		gradient_x_list12 = []
		gradient_y_list11 = []
		gradient_y_list12 = []		
		layer = network[i]
		if i == len(network)-1: # i.e. last layer
			for j in range(len(layer)):
				neuron = layer[j]
				gradient_x_list1.append(gradient_x_tar - neuron['output_gradient_x'])
				gradient_y_list1.append(gradient_y_tar - neuron['output_gradient_y'])
				gradient_x_list2.append(gradient_x_tar - neuron['output_gradient_x'])
				gradient_y_list2.append(gradient_y_tar - neuron['output_gradient_y'])
		else:
			for j in range(len(layer)):
				delta2_x = 0.0
				delta2_y = 0.0
				delta1_x_1 = 0.0
				delta1_x_2 = 0.0
				delta1_y_1 = 0.0
				delta1_y_2 = 0.0

				for neuron in network[i + 1]:
					delta2_x += (neuron['weights'][j] * neuron['delta2_gradient_x'])
					delta2_y += (neuron['weights'][j] * neuron['delta2_gradient_y'])
					delta1_x_1 += (neuron['weights'][j] * neuron['delta1_gradient_x'])
					delta1_x_2 += (neuron['weights'][j] * neuron['delta2_gradient_x'])
					delta1_y_1 += (neuron['weights'][j] * neuron['delta1_gradient_y'])
					delta1_y_2 += (neuron['weights'][j] * neuron['delta2_gradient_y'])
				gradient_x_list2.append(delta2_x)
				gradient_y_list2.append(delta2_y)
				gradient_x_list11.append(delta1_x_1)
				gradient_x_list12.append(delta1_x_2)
				gradient_y_list11.append(delta1_y_1)
				gradient_y_list12.append(delta1_y_2)

		for j in range(len(layer)):
			neuron = layer[j]
			if i == len(network)-1:
				neuron['delta1_gradient_x'] = gradient_x_list1[j] * activation_function_sec_deriv_sigmoid(neuron['output_response'])*neuron['gradient_h_x']
				neuron['delta1_gradient_y'] = gradient_y_list1[j] * activation_function_sec_deriv_sigmoid(neuron['output_response'])*neuron['gradient_h_y']
				neuron['delta2_gradient_x'] = gradient_x_list2[j] * activation_function_deriv_sigmoid(neuron['output_response'])
				neuron['delta2_gradient_y'] = gradient_y_list2[j] * activation_function_deriv_sigmoid(neuron['output_response'])

			else:
				neuron['delta1_gradient_x'] = gradient_x_list11[j] * activation_function_deriv_sigmoid(neuron['output_response']) + gradient_x_list12[j]*activation_function_sec_deriv_sigmoid(neuron['output_response'])*neuron['gradient_h_x']
				neuron['delta1_gradient_y'] = gradient_y_list11[j] * activation_function_deriv_sigmoid(neuron['output_response']) + gradient_y_list12[j]*activation_function_sec_deriv_sigmoid(neuron['output_response'])*neuron['gradient_h_y']
				neuron['delta2_gradient_x'] = gradient_x_list2[j] * activation_function_deriv_sigmoid(neuron['output_response'])
				neuron['delta2_gradient_y'] = gradient_y_list2[j] * activation_function_deriv_sigmoid(neuron['output_response'])

# Using learning rate, delta value computed in backpropagation algorithm, we can update our weights
def weight_update(network, sample, learning_rate):
	for i in range(len(network)):
		inputs = sample[:-3]
		if i != 0:
			inputs = [neuron['output_response'] for neuron in network[i - 1]]
			gradients_x = [neuron['output_gradient_x'] for neuron in network[i - 1]]
			gradients_y = [neuron['output_gradient_y'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				if i == 0:
					if j == 0:
						neuron['weights'][j] = neuron['weights'][j] + (learning_rate * (neuron['delta'] * inputs[j]+inputs[j]*neuron['delta1_gradient_x']+1*neuron['delta2_gradient_x']+inputs[j]*neuron['delta1_gradient_y']+0*neuron['delta2_gradient_y']))
					else:
						neuron['weights'][j] = neuron['weights'][j] + (learning_rate * (neuron['delta'] * inputs[j]+inputs[j]*neuron['delta1_gradient_x']+0*neuron['delta2_gradient_x']+inputs[j]*neuron['delta1_gradient_y']+1*neuron['delta2_gradient_y']))
				else:
					neuron['weights'][j] = neuron['weights'][j] + (learning_rate * (neuron['delta'] * inputs[j]+inputs[j]*neuron['delta1_gradient_x']+gradients_x[j]*neuron['delta2_gradient_x']+inputs[j]*neuron['delta1_gradient_y']+gradients_y[j]*neuron['delta2_gradient_y']))
			#neuron['weights'][-1] += learning_rate * neuron['delta']

# Let's train the network for a given number of epochs (i.e. passes over the entire dataset)
def network_training(network, training_set, learning_rate, epoch_total):
	error_list = []
	for epoch in range(epoch_total):
		error_sum = 0
		i = 0
		for sample in training_set:
			sample_response = sample[0:-3]
			outputs_response = forward_propagation_response(network, sample_response)
			h_gradient_xp(network, sample_response)
			outputs_grad_x_y = forward_propagation_gradient(network, sample_response)
			target_response = sample[-3]
			error_sum += (1/2)*(target_response-outputs_response[0])**2 * (1/2)*((outputs_grad_x_y[0][0]-sample[-2])**2+(outputs_grad_x_y[1][0]-sample[-1])**2)
			backpropagation(network, target_response)
			backpropagation_gradient(network,sample)
			#if i%16 == 0: # batch_size of 16
			weight_update(network, sample, learning_rate)
			i = i + 1
		error_sum = error_sum/i
		error_list.append(error_sum)
		print('>epoch=%d, lrate=%.3f, error=%.6f' % (epoch, learning_rate, error_sum))
	return error_list

# Let's call the appropriate functions to set up and train the network

number_input_nodes = len(training_dataset[0]) - 3
number_output_nodes = 1
number_hlayer1_nodes = 10
number_hlayer2_nodes = 10
learning_rate = 0.42
epoch_total = 100
network = network_setup(number_input_nodes, number_hlayer1_nodes, number_hlayer2_nodes, number_output_nodes)
error_list = network_training(network, training_dataset, learning_rate, epoch_total)

import matplotlib

epoch_list = []
n_epochs = 100
for i in range(n_epochs):
  epoch_list.append(i)

matplotlib.pyplot.plot(epoch_list,error_list)
matplotlib.pyplot.title('Error vs epochs')
matplotlib.pyplot.xlabel('Epochs')
matplotlib.pyplot.ylabel('Error')

# predictions

def predict(network, row):
	outputs = forward_propagation_response(network, row)
	return outputs[0]

X = [1,1]

prediction = predict(network,X)

print(prediction*(max_z-min_z)+min_z)

matplotlib.pyplot.show()