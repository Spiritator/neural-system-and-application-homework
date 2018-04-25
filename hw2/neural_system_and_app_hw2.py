# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:05:33 2018

@author: 蔡永聿

my homework of the class "neural network and appplication" homework 2

references:
    http://darren1231.pixnet.net/blog/post/339526256-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%AF%A6%E4%BD%9C%E5%87%BA%E9%A1%9E%E7%A5%9E%E7%B6%93%E5%85%AC%E5%BC%8F-with-ipython-notebook
    https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    
question: Use radial basis function ANN to simulate the function f(x)=2(x^2)+1/4(y^2) with radial basis function is Gaussian function and standard deviation is dmax/sqrt(m1), x,y are between -2 to 2.
    
"""

import time
import numpy as np
import scipy as scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

input_neurons=2
hidden_neurons=5
output_neurons=1
weight_l_rate=0.1
center_l_rate=0.1
delta_l_rate=0.1
momentum_rate=0.5
epoches=2000

#the simulation function
def simfunc(x,y):
    function=2*np.power(x, 2.)+np.power(y, 2.)/4
    return function
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'centers':np.array([np.random.random(n_inputs)])} for i in range(n_hidden)]
    coordinates=np.zeros((n_hidden,n_inputs))
    for i in range(n_hidden):
        neuron=hidden_layer[i]
        coordinates[i]=neuron['centers']
    dmax=np.max(scipy.spatial.distance.cdist(coordinates, coordinates, 'euclidean'))
    sigma=dmax/np.sqrt(n_hidden)
    for i in range(n_hidden):
        neuron=hidden_layer[i]
        neuron['sigma']=sigma
    network.append(hidden_layer)
    output_layer = [{'weights':np.array([np.random.random(n_hidden)]),'bias':np.random.random()} for i in range(n_outputs)]
    network.append(output_layer)
    
    return network

# Calculate neuron activation for an input
def activate(weights, bias, inputs):
	activation = np.dot(weights,np.transpose(inputs))
	activation = activation[0][0] + bias
	return activation

#Gaussion Function
def gaussian(distance, sigma):
    return np.exp(-np.power(distance, 2.) / (2 * np.power(sigma, 2.)))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for i in range(len(network)):
        layer = network[i]
        new_inputs = np.array([])
        if i == 0:
            for neuron in layer:
                neuron['output'] = gaussian(np.linalg.norm(row - neuron['centers']), neuron['sigma'])
                new_inputs=np.append(new_inputs,neuron['output'])
        elif i == len(network)-1:
            for neuron in layer:
                neuron['output'] = activate(neuron['weights'],neuron['bias'], inputs)
                new_inputs=np.append(new_inputs,neuron['output'])
        else:
            pass
        inputs = np.expand_dims(new_inputs,0)
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
			neuron['delta'] = errors[j]

# Update network weights with error
def update_weights(network, row, w_l_rate, c_l_rate, sig_l_rate):
    for i in reversed(range(len(network))):
        inputs = row
        if i==1:
            for neuron in network[i]:
                delta_weight=np.array([])
                for j in range(len(network[i-1])):
                    delta_weight = np.append(delta_weight, w_l_rate * neuron['delta'] * network[i-1][j]['output'])
                neuron['delta_weight']=np.expand_dims(delta_weight,0)
                neuron['delta_bias'] = w_l_rate * neuron['delta']
        else:
            for neuron in network[i]:
                neuron['delta_center'] = c_l_rate * neuron['delta'] / np.power(neuron['sigma'],2.) * neuron['output'] * (inputs - neuron['centers'])
                neuron['delta_sigma'] = sig_l_rate * neuron['delta'] / np.power(neuron['sigma'],3.) * neuron['output'] * np.power(inputs - neuron['centers'],2.)
            
    neuron['weights'] += neuron['delta_weight']
    neuron['bias'] += neuron['delta_bias']
    neuron['centers'] += neuron['delta_center']
    neuron['sigma'] += neuron['delta_sigma']
            
# Train a network for a fixed number of epochs
def train_network(network, train, validation, w_l_rate, c_l_rate, sig_l_rate, n_epoch, n_outputs):
    train_loss=[]
    validation_loss=[]
    runtime = time.time()
    
    for epoch in range(n_epoch):
        train_sum_error = 0
        for row in train:
            outputs = forward_propagate(network, np.expand_dims(np.array(row[:-n_outputs]),0))
            expected = np.expand_dims(np.array([row[-i-1] for i in reversed(range(n_outputs))]),0)
            train_sum_error += np.sum((np.power(expected[i]-outputs[i],2.) / 2))
            backward_propagate_error(network, expected)                
            update_weights(network, np.expand_dims(np.array(row[:-n_outputs]),0), w_l_rate, c_l_rate, sig_l_rate)
        train_loss.append(train_sum_error/len(train))
        
        validation_sum_error = 0
        for row in validation:
            outputs = forward_propagate(network, np.expand_dims(np.array(row[:-n_outputs]),0))
            expected = np.expand_dims(np.array([row[-i-1] for i in reversed(range(n_outputs))]),0)
            validation_sum_error += np.sum((np.power(expected[i]-outputs[i],2.) / 2))
        validation_loss.append(validation_sum_error/len(validation))

        print('>epoch=%d, loss=%.4f, val_loss=%.4f' % (epoch, train_sum_error, validation_sum_error))
    return {'train':train_loss,'validation':validation_loss,'runtime':time.time()-runtime}

# Make a prediction with a network
def predict(network, row):
    row=np.expand_dims(np.array(row),0)
    outputs = forward_propagate(network, row)
    return outputs[0]

#evaluate testing result
def testing_network(network,testing_data):
    test_result_data={'x':[],'y':[],'func':[],'predict':[]}
    for i in range(len(testing_data['func'])):
        predict_tmp=predict(network,[testing_data['x'][i],testing_data['y'][i]])
        test_result_data['x'].append(testing_data['x'][i])
        test_result_data['y'].append(testing_data['y'][i])
        test_result_data['func'].append(testing_data['func'][i])
        test_result_data['predict'].append(predict_tmp[0])
    
    return test_result_data
    
#print network weight
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
            print(network[layer][neuron]['weights'][0])
            print('        bias   :',end='')
            print(network[layer][neuron]['bias'])
#            print('        output :',end='')
#            print(network[layer][neuron]['output'])
#            print('        delta  :',end='')
#            print(network[layer][neuron]['delta'])
        print('')
        
def save_network(network):
    saved_network=[]
    for i in range(len(network)):
        layer = network[i]
        saved_layer = []
        for j in range(len(layer)):
            neuron = layer[j]
            saved_neuron={}
            saved_neuron['weights'] = neuron['weights']
            saved_neuron['bias'] = neuron['bias']
            saved_layer.append(saved_neuron)
        saved_network.append(saved_layer)
    
    return saved_network

#input normalization
def input_normalization(data_dict):
    normalized_input=[]
    for i in range(len(data_dict['func'])):
        normalized_x=(data_dict['x'][i]-1)/9
        normalized_y=(data_dict['y'][i]-1)/9
        normalized_func=(data_dict['func'][i]-2)/199
        normalized_input.append([normalized_x,normalized_y,normalized_func])
    return normalized_input

#plot loss descent during training            
def show_train_history(train,validation,title,ylabel):
    plt.plot([i for i in range(len(train))],train)
    plt.plot([i for i in range(len(validation))],validation)
    plt.title(title)
    plt.yscale('log')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

#draw 3D plot of datasets
def draw3Dplot(plot_name,data_dict,color,marker):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data_dict['x'],data_dict['y'],data_dict['func'], c=color, marker=marker, depthshade=False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F(X,Y)')
    plt.title(plot_name)
    
    plt.show()
    
#view testing result in 3D plot
def draw_test_result_3Dplot(plot_name,data_dict,color,marker):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data_dict['x'],data_dict['y'],data_dict['func'], c=color[0], marker=marker[0], label='label', depthshade=False)
    
    ax.scatter(data_dict['x'],data_dict['y'],data_dict['predict'], c=color[1], marker=marker[1], label='predict', depthshade=False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F(X,Y)')
    ax.legend(loc='upper right')
    plt.title(plot_name)
    
    plt.show()
    

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
    
np.random.seed(1)
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
    

#draw3Dplot('training data',training_data,'b','o')
#draw3Dplot('validation data',validation_data,'g','^')
#draw3Dplot('testing data',testing_data,'r','s')

training_set = input_normalization(training_data)
validation_set = input_normalization(validation_data)

network = initialize_network(input_neurons, hidden_neurons, output_neurons)
print('Initial Network\n')
network_summary(network)
train_summary=train_network(network, training_set, validation_set, learning_rate, momentum_rate, epoches, output_neurons)
print('Trained Network\n')
network_summary(network)
print('>train loss=%.5g, validation loss=%.5g, runtime=%.2fs' % (train_summary['train'][-1], train_summary['validation'][-1], train_summary['runtime']))
show_train_history(train_summary['train'],train_summary['validation'],'loss','MSE (log)')
test_result_data=testing_network(network,testing_data)
draw_test_result_3Dplot('Test Result',test_result_data,['r','b'],['o','^'])
saved_network=save_network(network)
