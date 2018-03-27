# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:05:33 2018

@author: 蔡永聿

references:
    http://darren1231.pixnet.net/blog/post/339526256-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%AF%A6%E4%BD%9C%E5%87%BA%E9%A1%9E%E7%A5%9E%E7%B6%93%E5%85%AC%E5%BC%8F-with-ipython-notebook
    https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
"""

import random
import numpy as np

learning_rate=0.1
epoches=200
input_neurons=2
hidden_neurons=5
output_neurons=1

#the simulation function
def simfunc(x,y):
    function=np.sin(x)+2*(y**2)
    return function
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random.random() for i in range(n_inputs)],'bias':random.random()} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random.random() for i in range(n_hidden)],'bias':random.random()} for i in range(n_outputs)]
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
			outputs = forward_propagate(network, row[:-n_outputs])
			expected = [row[-i-1] for i in reversed(range(n_outputs))]
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row[:-n_outputs], l_rate)
		print('>epoch=%d, learning rate=%.2f, loss=%.8f' % (epoch, l_rate, sum_error))

# Make a prediction with a network
def predict(network, row):
    for i in range(len(row)):
        row[i]=(row[i]-1)/9
    outputs = forward_propagate(network, row)
    for i in range(len(outputs)):
        outputs[i]=outputs[i]*199+2
    return outputs

def network_summary(network):
    print('======================================')
    print('Network Summary')
    print('======================================')
    print('')
    print('layer 1(input layer)')
    print('')
    for layer in range(len(network)):
        print('======================================')
        print('')
        print('Layer %d' % (layer+2))
        print('')
        for neuron in range(len(network[layer])):
            print('    neuron %d' % (neuron+1))
            print('        weights:',end='')
            print(network[layer][neuron]['weights'])
            print('        bias   :',end='')
            print(network[layer][neuron]['bias'])
            print('        output :',end='')
            print(network[layer][neuron]['output'])
            print('        delta  :',end='')
            print(network[layer][neuron]['delta'])
        print('')

#input normalization
def input_normalization(data_dict):
    normalized_input=[]
    for i in range(len(data_dict['func'])):
        normalized_x=(data_dict['x'][i]-1)/9
        normalized_y=(data_dict['y'][i]-1)/9
        normalized_func=(data_dict['func'][i]-2)/199
        normalized_input.append([normalized_x,normalized_y,normalized_func])
    return normalized_input

#network = initialize_network(2, 5, 1)
#for layer in network:
#	print(layer)
#    
#row = [1, 0]
#output = forward_propagate(network, row)
#print(output)
#
#expected = [1]
#backward_propagate_error(network, expected)
#for layer in network:
#	print(layer)
    
random.seed(1)
#data generation
training_data={'x':[],'y':[],'func':[]}
for i in range(400):
    x_tmp=random.uniform(1, 10)
    y_tmp=random.uniform(1, 10)
    func_tmp=simfunc(x_tmp,y_tmp)
    training_data['x'].append(x_tmp)
    training_data['y'].append(y_tmp)
    training_data['func'].append(func_tmp)
    
validation_data={'x':[],'y':[],'func':[]}
for i in range(200):
    x_tmp=random.uniform(1, 10)
    y_tmp=random.uniform(1, 10)
    func_tmp=simfunc(x_tmp,y_tmp)
    validation_data['x'].append(x_tmp)
    validation_data['y'].append(y_tmp)
    validation_data['func'].append(func_tmp)
    
testing_data={'x':[],'y':[],'func':[]}
for i in range(100):
    x_tmp=random.uniform(1, 10)
    y_tmp=random.uniform(1, 10)
    func_tmp=simfunc(x_tmp,y_tmp)
    testing_data['x'].append(x_tmp)
    testing_data['y'].append(y_tmp)
    testing_data['func'].append(func_tmp)
    
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def draw3Dplot(plot_name,data_dict,color,marker):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(data_dict['func'])):
        ax.scatter(data_dict['x'][i],data_dict['y'][i],data_dict['func'][i], c=color, marker=marker)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F(X,Y)')
    plt.title(plot_name)
    
    plt.show()
    
#draw3Dplot('training data',training_data,'b','o')
#draw3Dplot('validation data',validation_data,'b','o')
#draw3Dplot('testing data',testing_data,'b','o')

training_set = input_normalization(training_data)
validation_set = input_normalization(validation_data)

network = initialize_network(input_neurons, hidden_neurons, output_neurons)
train_network(network, training_set, learning_rate, epoches, output_neurons)
network_summary(network)
