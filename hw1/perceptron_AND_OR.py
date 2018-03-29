# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np#載入NumPy函式庫
import matplotlib.pyplot as plt#載入Matpltlib函式庫

#設計嚴格活化函數
def f(net):
    return 1* (net >= 0);

#輸入資料與標準答案
x_train = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y_label = np.array([[0], [0], [0], [1]])

#亂數產生權重與偏權
W = np.random.random_sample([2, 1])
B = np.random.random_sample([1])

#參數設定與初始化
interations = 10000#迭代次數
threshold = 0.0001#門檻植
learning_rate = 0.01#學習速率
Loss = np.zeros(interations)#Loss function 初始化


for i in range(interations):
    net = np.dot(x_train, W) + B#計算net
    output = f(net)#透過活化函數作用產生預測值
    error = y_label - output#計算誤差量
    
    Loss[i] = np.sum(error**2)/(2*4)#計算Loss function(本例題採用MSE)，除以4是因為一次代入4組資料。
    if Loss[i] < threshold:#當誤差值小於門檻
        Loss.resize(i+1)
        break
    #權重與偏權調整
    W = W + learning_rate*np.dot(x_train.transpose(), error)/4
    B = B + learning_rate*np.sum(error)/4
    
    
#繪製數據    
plt.plot(range(i+1), Loss[0:i+1])
plt.xlabel('interations')
plt.ylabel('MSE')
plt.title('Comparsion of MSE and Interations')