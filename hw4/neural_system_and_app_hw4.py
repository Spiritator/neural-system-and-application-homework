# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:04:37 2018

@author: 蔡永聿

my homework of the class "neural network and appplication" homework 4

references:
    http://darren1231.pixnet.net/blog/post/339526256-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%AF%A6%E4%BD%9C%E5%87%BA%E9%A1%9E%E7%A5%9E%E7%B6%93%E5%85%AC%E5%BC%8F-with-ipython-notebook
    https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    
question: Use counterpropagation neural network and concept of fuzzy-control to estimate funtion y=x/2+sqrt(x).
    
"""
#%%
#parameters
input_neurons=1
output_neurons=1
delta=7
alpha=0.5
beta=0.5
fuzzy_control=False

#%%
#set up
import time
import numpy as np
import matplotlib.pyplot as plt
 
#the simulation function
def simfunc(x):
    y=x/2+np.sqrt(x)
    return y
 
# Initialize a network
def initialize_network(n_outputs):
    network = list()
    hidden_layer = list()
    network.append(hidden_layer)
    output_layer = [{'pi':np.array([])} for i in range(n_outputs)]
    network.append(output_layer)
    
    return network

# Add a neuron criterion to hidden layer
def add_hidden_neuron(network, x, y):
    hidden_layer = network[0]
    hidden_layer.append({'weights':x})
    output_layer = network[1]
    for neuron in output_layer:
        neuron['pi']=np.append(neuron['pi'],y)

# Forward propagate input to a network output
def forward_propagate(network, inputs):
    hidden_layer = network[0]
    distance = np.array([])
    for neuron in hidden_layer:
        neuron['distance'] = np.linalg.norm(inputs - neuron['weights'])
        distance=np.append(distance,neuron['distance'])
    Dmin_arg=distance.argmin()
    output_layer = network[1]
    outputs = np.array([])
    for neuron in output_layer:
        outputs=np.append(outputs, neuron['pi'][Dmin_arg])
    return Dmin_arg,hidden_layer[Dmin_arg]['distance'],outputs

# Update network weights with error
def update_weights(network, Dmin_arg, x, y, alpha, beta):
    hidden_layer = network[0]        
    hidden_layer[Dmin_arg]['weights']+=alpha*(x-hidden_layer[Dmin_arg]['weights'])
    output_layer = network[1]
    for neuron in output_layer:
        neuron['pi'][Dmin_arg]+=beta*(y-neuron['pi'][Dmin_arg])
            
# Train a network for a fixed number of epochs
def train_network(network, train, delta, alpha, beta, n_outputs):
    train_loss=[]
    runtime = time.time()
    criterion=[]
    epoch=0
    
    while True:
        train_sum_error = 0
        for row in train:
            if len(network[0])==0:
                add_hidden_neuron(network, row[:-n_outputs], row[-n_outputs:])
            else:
                Dmin_arg,distance,outputs = forward_propagate(network, row[:-n_outputs])
                expected = row[-n_outputs:]
                train_sum_error += np.sum((np.power(expected-outputs,2.) / 2))
                if distance <= delta:
                    update_weights(network, Dmin_arg, row[:-n_outputs], expected, alpha, beta)
                else:
                    add_hidden_neuron(network, row[:-n_outputs], expected)
        criterion.append(len(network[0]))
        train_loss.append(train_sum_error/len(train))
        print('>epoch=%d, loss=%.4f' % (epoch, train_loss[-1]))
        epoch+=1
        
        if len(criterion)>=2 and criterion[-2]==criterion[-1]:
            break
        
    return {'loss':train_loss,'runtime':time.time()-runtime}

# Make a prediction with a network
def predict(network, row):
    row=np.array([row])
    Dmin_arg,distance,outputs = forward_propagate(network, row)
    return outputs

#evaluate testing result
def testing_network(network,testing_data):
    test_result_data={'x':[],'y':[],'predict':[]}
    for i in range(len(testing_data['y'])):
        predict_tmp=predict(network,testing_data['x'][i])
        test_result_data['x'].append(testing_data['x'][i])
        test_result_data['y'].append(testing_data['y'][i])
        test_result_data['predict'].append(predict_tmp)
    
    return test_result_data
    
#print network weight
def network_summary(network):
    print('======================================')
    print('Network Summary')
    print('======================================')
    print('')
    print('layer 1(input layer)')
    print('')
    print('======================================')
    print('')
    print('Layer 2(hidden layer)')
    print('')
    for neuron in range(len(network[0])):
        print('    neuron %d' % (neuron+1))
        print('        weights:',end='')
        print(network[0][neuron]['weights'])
    print('======================================')
    print('')
    print('Layer 3(output layer)')
    print('')
    for neuron in range(len(network[1])):
        print('    neuron %d' % (neuron+1))
        print('        pi:',end='')
        print(network[1][neuron]['pi'])
    print('')

    print('')

#input normalization
def input_normalization(data_dict):
    normalized_input=[]
    for i in range(len(data_dict['y'])):
        normalized_x=data_dict['x'][i]
        normalized_y=data_dict['y'][i]
        normalized_input.append([normalized_x,normalized_y])
    return np.array(normalized_input)

#plot loss descent and accuracy during training            
def show_train_history(loss,title,ylabel):
    plt.xlabel('Epoches')
    plt.ylabel(ylabel)
    plt.plot([i for i in range(len(loss))],loss, color='b', label='loss')
        
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()
    
#view input data in 2D scatter plot
def draw_class_scatter_plot(plot_name,data_dict,color,marker):            
    plt.scatter(data_dict['x'], data_dict['y'], c=color, marker=marker)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(plot_name)
    plt.show()
    
#view trained result in 2D scatter plot
def draw_result_scatter_plot(plot_name,data_dict,color,marker):            
    plt.scatter(data_dict['x'], data_dict['y'], c=color[0], marker=marker[0], label='label')
    plt.scatter(data_dict['x'], data_dict['predict'], c=color[1], marker=marker[1], label='predict')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc=4)
    plt.title(plot_name)
    plt.show()
    
#view testing result in 2D plot
def draw_test_result_FIT_plot(plot_name,data_dict,color,marker):
    x = np.array([0,9])
    plt.plot(x,x,lw=2, c='tab:gray', linestyle=':' ,label='Y=T')
    plt.scatter(data_dict['label'], data_dict['predict'], s=20, c=color, marker=marker,label='Data')
    FIT_line = np.polyfit(data_dict['label'], data_dict['predict'], 1)
    FIT_line = np.poly1d(FIT_line) 
    plt.plot(x,FIT_line(x),lw=2, c='k', linestyle='-' ,label='FIT')
    plt.xlabel("Target")
    plt.ylabel("Output ~= %.2f T + %.4f" % (FIT_line.coefficients[0],FIT_line.coefficients[1]))
    plt.legend(loc='upper right')
    plot_name+='%.5f' % np.min(np.corrcoef(data_dict['label'], data_dict['predict']))
    plt.title(plot_name)
    plt.show()
    
    
#%%
#prepare data set
training_data={'x':[        9,         16,         25,         32,         49,         56],\
               'y':[simfunc(9),simfunc(16),simfunc(25),simfunc(32),simfunc(49),simfunc(56)]}

        
#%%
#training

draw_class_scatter_plot('Training Data Distribution',training_data,'b','o')
training_set = input_normalization(training_data)
network = initialize_network(output_neurons)
print('Initial Network\n')
network_summary(network)
train_summary=train_network(network, training_set, delta, alpha, beta, output_neurons)
print('Trained Network\n')
network_summary(network)
print('>train loss=%.5g, runtime=%.2fs' % (train_summary['loss'][-1], train_summary['runtime']))
show_train_history(train_summary['loss'],'Train History','MSE (log)')
test_result_data=testing_network(network,training_data)
draw_result_scatter_plot('Test Result',test_result_data,['r','b'],['o','^'])
#draw_test_result_FIT_plot('Regression R = ',test_result_data,'b','o')

