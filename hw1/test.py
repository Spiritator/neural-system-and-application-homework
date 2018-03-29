# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:05:33 2018

@author: 蔡永聿

references:
    http://darren1231.pixnet.net/blog/post/339526256-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%AF%A6%E4%BD%9C%E5%87%BA%E9%A1%9E%E7%A5%9E%E7%B6%93%E5%85%AC%E5%BC%8F-with-ipython-notebook
    https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
"""

import numpy as np

learning_rate=0.1

#the simulation function
def simfunc(x,y):
    function=np.sin(x)+2*(y**2)
    return function

sizes=[2,3,1]
num_layers = len(sizes)
biases = [np.random.randn(y, 1) for y in sizes[1:]]  #輸入層沒有bias
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  #23 31

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

    
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

#weight generation    
weight_hidden=np.random.uniform(low=0, high=1, size=(5,2))
bias_hidden=np.random.uniform(low=0, high=1, size=(5,1))
weight_output=np.random.uniform(low=0, high=1, size=(5,1))

#front probagation
def frontprob(x,y):
    input_arr=np.array([x,y])
    hidden_multiply_arr=np.reshape(np.sum(input_arr*weight_hidden,axis=1),(5,1))
    hidden_output_arr=hidden_multiply_arr+bias_hidden
    for i in range(len(hidden_output_arr)):
        hidden_output_arr[i]=sigmoid(hidden_output_arr[i])
    output_arr=hidden_output_arr*weight_output
    output_sum=np.sum(output_arr,axis=0)
    return output_sum[0]

#back probagation
def backprob(x,y,label):
    input_arr=np.array([x,y])
    #front probagation
    hidden_multiply_arr=np.reshape(np.sum(input_arr*weight_hidden,axis=1),(5,1))
    hidden_output_arr=hidden_multiply_arr+bias_hidden
    for i in range(len(hidden_output_arr)):
        hidden_output_arr[i]=sigmoid(hidden_output_arr[i])
    output_arr=hidden_output_arr*weight_output
    output_sum=np.sum(output_arr,axis=0)
    #weight update
    error_output=label-output_sum[0]
    
    weight_output=weight_output+learning_rate*error_output*hidden_output_arr
    
    desigmoid_hidden_output_arr=hidden_output_arr
    for i in range(len(hidden_output_arr)):
        desigmoid_hidden_output_arr[i]=desigmoid(hidden_output_arr[i])
    desigmoid_hidden_output_arr=np.swapaxes(desigmoid_hidden_output_arr,0,1)
    weight_hidden=weight_hidden+learning_rate*error_output*input_arr*hidden_output_arr*desigmoid_hidden_output_arr)
    
    bias_hidden=bias_hidden+learning_rate*error_output*hidden_multiply_arr*desigmoid(hidden_multiply_arr)
    
    
    

output_arrr=frontprob(1,3,weight_hidden,bias_hidden,weight_output)
