# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:05:33 2018

@author: 蔡永聿

references:
    http://darren1231.pixnet.net/blog/post/339526256-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%AF%A6%E4%BD%9C%E5%87%BA%E9%A1%9E%E7%A5%9E%E7%B6%93%E5%85%AC%E5%BC%8F-with-ipython-notebook
    https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
"""
from random import seed
from random import random
import numpy as np

learning_rate=0.1

#the simulation function
def simfunc(x,y):
    function=np.sin(x)+2*(y**2)
    return function
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs)],'bias':random()} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden)],'bias':random()} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights,bias, inputs):
	activation = bias
	for i in range(len(weights)):
		activation += weights[i] * inputs[i]
	return activation

#sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#derivative sigmoid function
def dsigmoid(output):
    return output*(1.0-output)

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'],neuron['bias'], inputs)
			neuron['output'] = sigmoid(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * dsigmoid(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['bias'] += l_rate * neuron['delta']
            
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.2f, error=%.3f' % (epoch, l_rate, sum_error))

seed(1)
network = initialize_network(2, 5, 1)
for layer in network:
	print(layer)
    
row = [1, 0]
output = forward_propagate(network, row)
print(output)

expected = [1]
backward_propagate_error(network, expected)
for layer in network:
	print(layer)
    

#data generation
training_data=[]
for i in range(400):
    x_tmp=random.uniform(1, 10)
    y_tmp=random.uniform(1, 10)
    func_tmp=simfunc(x_tmp,y_tmp)
    data_dict_tmp={'x':x_tmp,'y':y_tmp,'func':func_tmp}
    training_data.append(data_dict_tmp)
    
validation_data=[]
for i in range(200):
    x_tmp=random.uniform(1, 10)
    y_tmp=random.uniform(1, 10)
    func_tmp=simfunc(x_tmp,y_tmp)
    data_dict_tmp={'x':x_tmp,'y':y_tmp,'func':func_tmp}
    validation_data.append(data_dict_tmp)
    
testing_data=[]
for i in range(100):
    x_tmp=random.uniform(1, 10)
    y_tmp=random.uniform(1, 10)
    func_tmp=simfunc(x_tmp,y_tmp)
    data_dict_tmp={'x':x_tmp,'y':y_tmp,'func':func_tmp}
    testing_data.append(data_dict_tmp)
