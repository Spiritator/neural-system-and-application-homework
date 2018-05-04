# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:05:33 2018

@author: 蔡永聿

my homework of the class "neural network and appplication" homework 3

references:
    http://darren1231.pixnet.net/blog/post/339526256-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%AF%A6%E4%BD%9C%E5%87%BA%E9%A1%9E%E7%A5%9E%E7%B6%93%E5%85%AC%E5%BC%8F-with-ipython-notebook
    https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    
question: Use LVQ neural network to classify 20 2D vector out of 3 classes.
    
"""
#%%
#parameters
class_numbers=3
input_neurons=2
output_neurons=class_numbers
learning_rate=0.01
epoches=200


#%%
#declaration
import time
import numpy as np
import matplotlib.pyplot as plt
 
# Initialize a network
def initialize_network(n_inputs, n_outputs):
    network = list()
    output_layer = [{'weights':np.array([np.random.random(n_inputs)])} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for i in range(len(network)):
        layer = network[i]
        new_inputs = np.array([])
        for neuron in layer:
            neuron['output'] = np.linalg.norm(inputs - neuron['weights'])
            new_inputs=np.append(new_inputs,neuron['output'])
        inputs = np.expand_dims(new_inputs,0)
    return np.argmin(inputs)+1,inputs

# Backpropagate error and store in neurons
def backward_propagate_error(network, row, pred):
    for i in reversed(range(len(network))):
        layer = network[i]
        neuron = layer[pred-1]
        neuron['delta'] = row-neuron['weights']

# Update network weights with error
def update_weights(network, l_rate, pred, hit):
    for i in range(len(network)):
        layer = network[i]
        neuron = layer[pred-1]
        if hit:
            neuron['weights']+=l_rate*neuron['delta']
        else:
            neuron['weights']-=l_rate*neuron['delta']
            
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    train_loss=[]
    accuracy=[]
    runtime = time.time()
    
    for epoch in range(n_epoch):
        train_sum_error = 0
        hit_counter = 0
        miss_counter = 0
        for row in train:
            pred_class,outputs = forward_propagate(network, row[...,:-1])
            expected = row[...,-1:]
            if expected==pred_class:
                hit_counter+=1
                hit=True
            else:
                miss_counter+=1
                hit=False
            train_sum_error += np.sum((np.power(outputs,2.) / 2))
            backward_propagate_error(network, row[...,:-1], pred_class)                
            update_weights(network, l_rate, pred_class, hit)
        train_loss.append(train_sum_error/len(train))
        accuracy.append(hit_counter/(hit_counter+miss_counter))

        print('>epoch=%d, acc=%.4f, loss=%.4f ' % (epoch, accuracy[-1], train_loss[-1]))
    return {'loss':train_loss,'accuracy':accuracy,'runtime':time.time()-runtime}

# Make a prediction with a network
def predict(network, row):
    row=np.array([row])
    pred_class,outputs = forward_propagate(network, row)
    return pred_class

#evaluate testing result
def testing_network(network,testing_data):
    test_result_data={'x1':[],'x2':[],'label':[],'predict':[]}
    for i in range(len(testing_data['label'])):
        predict_tmp=predict(network,[testing_data['x1'][i],testing_data['x2'][i]])
        test_result_data['x1'].append(testing_data['x1'][i])
        test_result_data['x2'].append(testing_data['x2'][i])
        test_result_data['label'].append(testing_data['label'][i])
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
    print('Layer 2(output layer)')
    print('')
    for neuron in range(len(network[0])):
        print('    neuron %d' % (neuron+1))
        print('        weights:',end='')
        print(network[0][neuron]['weights'][0])

    print('')
        
def save_network(network):
    saved_network=[]
    saved_layer = []
    layer = network[0]
    saved_layer = []
    for j in range(len(layer)):
        neuron = layer[j]
        saved_neuron={}
        saved_neuron['weights'] = neuron['weights']
        saved_layer.append(saved_neuron)
    saved_network.append(saved_layer)
    
    return saved_network

#input normalization
def input_normalization(data_dict):
    normalized_input=[]
    for i in range(len(data_dict['label'])):
        normalized_x=data_dict['x1'][i]
        normalized_y=data_dict['x2'][i]
        normalized_label=data_dict['label'][i]
        normalized_input.append([normalized_x,normalized_y,normalized_label])
    return np.expand_dims(np.array(normalized_input),1)

#plot loss descent and accuracy during training            
def show_train_history(loss,acc,title,ylabel):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoches')
    ax1.set_ylabel(ylabel[0], color=color)
    ax1.plot([i for i in range(len(loss))],loss, color=color, label='loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel(ylabel[1], color=color)  # we already handled the x-label with ax1
    ax2.plot([i for i in range(len(acc))],acc, color=color, label='accuracy', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
        
    fig.legend(loc='upper right')
    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
#view trained centers in 2D scatter plot
def draw_class_scatter_plot(plot_name,data_dict,color,marker):  
    class1_x1=[]
    class1_x2=[]
    class2_x1=[]
    class2_x2=[]
    class3_x1=[]
    class3_x2=[]
    for i in range(len(data_dict['label'])):
        if data_dict['label'][i]==1:
            class1_x1.append(data_dict['x1'][i])
            class1_x2.append(data_dict['x2'][i])
        elif data_dict['label'][i]==2:
            class2_x1.append(data_dict['x1'][i])
            class2_x2.append(data_dict['x2'][i])
        elif data_dict['label'][i]==3:
            class3_x1.append(data_dict['x1'][i])
            class3_x2.append(data_dict['x2'][i])
            
    plt.scatter(class1_x1, class1_x2, c=color[0], marker=marker[0], label='class 1')
    plt.scatter(class2_x1, class2_x2, c=color[1], marker=marker[1], label='class 2')
    plt.scatter(class3_x1, class3_x2, c=color[2], marker=marker[2], label='class 3')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc=3)
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
np.random.seed(1)
#data generation
training_data={   'x1':[3.23,0.27,1.94,1.92,1.89,1.41,0.02,3.49,1.18,3.76,1.06,0.93,3.61,1.41,0.20,3.46,3.89,3.95,0.45,0.75],\
                  'x2':[0.82,3.47,2.01,2.14,2.20,2.81,3.85,0.62,2.35,0.92,2.60,3.67,0.44,2.20,3.42,0.74,0.79,0.02,3.20,3.53],\
               'label':[   2,   1,   3,   3,   3,   3,   1,   2,   3,   2,   3,   1,   2,   3,   1,   2,   2,   2,   1,   1]}

        
#%%
#training

draw_class_scatter_plot('Training Data Distribution',training_data,['g','r','b'],['o','+','^'])
training_set = input_normalization(training_data)
network = initialize_network(input_neurons, output_neurons)
print('Initial Network\n')
network_summary(network)
train_summary=train_network(network, training_set, learning_rate,  epoches, output_neurons)
print('Trained Network\n')
network_summary(network)
print('>train loss=%.5g, validation loss=%.5g, runtime=%.2fs' % (train_summary['loss'][-1], train_summary['accuracy'][-1], train_summary['runtime']))
show_train_history(train_summary['loss'],train_summary['accuracy'],'Train History',['MSE (log)','acc'])
test_result_data=testing_network(network,training_data)
#draw_test_result_FIT_plot('Regression R = ',test_result_data,'b','o')
saved_network=save_network(network)

