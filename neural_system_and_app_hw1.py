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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

input_neurons=2
hidden_neurons=5
output_neurons=1
learning_rate=0.1
momentum_rate=0.5
epoches=200
batch_size=1

#the simulation function
def simfunc(x,y):
    function=np.sin(x)+2*(y**2)
    return function
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random.random() for i in range(n_inputs)],'momentum':[0.0 for i in range(n_inputs)],'bias':random.random(),'bias_momentum':0.0} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random.random() for i in range(n_hidden)],'momentum':[0.0 for i in range(n_hidden)],'bias':random.random(),'bias_momentum':0.0} for i in range(n_outputs)]
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
			neuron['batch_delta_sum'] += neuron['delta']

# Update network weights with error
def update_weights(network, row, l_rate, m_rate):
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                delta_weight = l_rate * neuron['delta'] * inputs[j] + m_rate * neuron['momentum'][j]
                neuron['weights'][j] += delta_weight
                neuron['momentum'][j] = delta_weight
            delta_bias = l_rate * neuron['delta'] + m_rate * neuron['bias_momentum']     
            neuron['bias'] += delta_bias
            neuron['bias_momentum'] = delta_bias
            
# Train a network for a fixed number of epochs
def train_network(network, train, validation, l_rate, m_rate, n_epoch, n_outputs, batch_size):
    train_loss=[]
    validation_loss=[]
    
    for epoch in range(n_epoch):
        batch_train=[train[i:i+batch_size] for i in range(0,len(train),batch_size)]
        
        train_sum_error = 0
        for batch in batch_train:
            for i in range(len(network)):
                layer = network[i]
                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron['batch_delta_sum'] = 0.0
            
            for row in batch:
                outputs = forward_propagate(network, row[:-n_outputs])
                expected = [row[-i-1] for i in reversed(range(n_outputs))]
                train_sum_error += sum([((expected[i]-outputs[i])**2)/2 for i in range(len(expected))])
                backward_propagate_error(network, expected)
                
            for i in range(len(network)):
                layer = network[i]
                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron['delta']=neuron['batch_delta_sum']/batch_size
                
            update_weights(network, row[:-n_outputs], l_rate, m_rate)
        train_loss.append(train_sum_error)
        
        validation_sum_error = 0
        for row in validation:
            outputs = forward_propagate(network, row[:-n_outputs])
            expected = [row[-i-1] for i in reversed(range(n_outputs))]
            validation_sum_error += sum([((expected[i]-outputs[i])**2)/2 for i in range(len(expected))])
        validation_loss.append(validation_sum_error)

        print('>epoch=%d, loss=%.4f, val_loss=%.4f' % (epoch, train_sum_error, validation_sum_error))
    return {'train':train_loss,'validation':validation_loss}

# Make a prediction with a network
def predict(network, row):
    for i in range(len(row)):
        row[i]=(row[i]-1)/9
    outputs = forward_propagate(network, row)
    for i in range(len(outputs)):
        outputs[i]=outputs[i]*199+2
    return outputs

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
            print(network[layer][neuron]['weights'])
            print('        bias   :',end='')
            print(network[layer][neuron]['bias'])
#            print('        output :',end='')
#            print(network[layer][neuron]['output'])
#            print('        delta  :',end='')
#            print(network[layer][neuron]['delta'])
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
    

draw3Dplot('training data',training_data,'b','o')
draw3Dplot('validation data',validation_data,'g','^')
draw3Dplot('testing data',testing_data,'r','s')

training_set = input_normalization(training_data)
validation_set = input_normalization(validation_data)

network = initialize_network(input_neurons, hidden_neurons, output_neurons)
print('Initial Network\n')
network_summary(network)
train_loss_summary=train_network(network, training_set, validation_set, learning_rate, momentum_rate, epoches, output_neurons, batch_size)
print('Trained Network\n')
network_summary(network)
show_train_history(train_loss_summary['train'],train_loss_summary['validation'],'loss','MSE (log)')
test_result_data=testing_network(network,testing_data)
draw_test_result_3Dplot('Test Result',test_result_data,['r','b'],['o','^'])
