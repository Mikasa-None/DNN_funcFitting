# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

class Net():
    def __init__(self, L, N, func, cost_func):
        self.hidden_Layers = L
        self.hidden_Layers_node_number = N
        self.hidden_Layers_function = func
        self.costfunction = cost_func
        self.total_Layers = L + 1
        self.total_Layers_node_number = np.array([])
        self.w = {}
        self.b = {}
        self.A = {}
        self.Z = {}
        self.dA = {}
        self.dZ = {}
        self.dw = {}
        self.db = {}
        self.alpha = 0.01
        self.maxIter = 5000
        self.cost = []


    def train(self, X, Y):
        print('train')
        self.total_Layers_node_number = np.append([X.shape[0]], N)
        k = self.maxIter
        self.init_net()
        for i in range(k):
            A = self.forward(X)
            self.cost.append(self.cost_func(A, Y))
            if i > 1:
                if (self.cost[i] < 0.001 or abs(self.cost[i] - self.cost[i - 1]) < 0.000001):
                    print('cost break')
                    break

            self.backward(Y)
        return self.cost

    def test(self, X, Y):
        print('test')
        Y_test = self.forward(X)
        cost_test = self.cost_func(Y_test, Y)
        return cost_test


    def predict(self, X):
        print('predict')
        Y_predict = self.forward(X)
        return Y_predict

    def cost_func(self, A, Y):
        print('cost')
        if (self.costfunction == 'CrossEntropy'):
            m = Y.shape[1]
            cost = 1.0 / m * np.sum((-np.dot(Y, np.log(A.transpose())) - np.dot((np.ones(Y.shape) - Y), (
                np.log(np.ones(A.shape) - A)).transpose())))
            print(cost)
        if (self.costfunction == 'QuadraticDifference'):
            m = Y.shape[1]
            cost = 1.0 / m * 0.5 * np.sum(np.power(A - Y, 2))
            print(cost)
        return cost

    def init_net(self):
        L = self.hidden_Layers
        np.random.seed(1)
        for i in range(L):
            self.w[i + 1] = np.random.randn(self.total_Layers_node_number[i + 1],
                                            self.total_Layers_node_number[i]) #* 0.01
            '''self.w[i + 1] = np.random.normal(0,0.1*(i+1),size=(self.total_Layers_node_number[i + 1],
                                            self.total_Layers_node_number[i]))'''
            self.b[i + 1] = np.zeros((self.total_Layers_node_number[i + 1], 1))
            # np.zeros((self.total_Layers_node_number[i + 1], m))

    def forward(self, X):
        print('forward')
        self.A[0] = X
        L = self.hidden_Layers
        for i in range(L):
            self.Z[i + 1] = np.dot(self.w[i + 1], self.A[i]) + self.b[i + 1]
            if (self.hidden_Layers_function[i + 1] == 'sigmoid'):
                self.A[i + 1] = 1.0 / (np.ones(self.Z[i + 1].shape) + np.exp(-self.Z[i + 1]))
            elif (self.hidden_Layers_function[i + 1] == 'x'):
                self.A[i + 1] = self.Z[i + 1]
            elif (self.hidden_Layers_function[i + 1] == 'tanh'):
                self.A[i + 1] = (np.exp(self.Z[i + 1]) - np.exp(-self.Z[i + 1])) / (np.exp(self.Z[i + 1]) + np.exp(-self.Z[i + 1]))
        return self.A[L]

    def backward(self, Y):
        print('backward')
        m = Y.shape[1]
        L = self.hidden_Layers
        if (self.costfunction == 'CrossEntropy'):
            self.dA[L] = -Y / self.A[L] + (np.ones(Y.shape) - Y) / (np.ones(self.A[L].shape) - self.A[L])
        if (self.costfunction == 'QuadraticDifference'):
            self.dA[L] = self.A[L] - Y

        for i in range(L, 0, -1):
            if (self.hidden_Layers_function[i] == 'sigmoid'):
                self.dZ[i] = self.dA[i] * self.A[i] * (np.ones(self.A[i].shape) - self.A[i])
            elif (self.hidden_Layers_function[i] == 'tanh'):
                self.dZ[i] = self.dA[i]*(np.ones(self.A[i].shape) - np.power(self.A[i],2))
            elif (self.hidden_Layers_function[i] == 'x'):
                self.dZ[i] = self.dA[i]
            self.dw[i] = 1.0 / m * np.dot(self.dZ[i], self.A[i - 1].transpose())
            self.db[i] = 1.0 / m * np.sum(self.dZ[i],axis=1, keepdims=True)
            self.dA[i - 1] = np.dot(self.w[i].transpose(), self.dZ[i])
            self.w[i] = self.w[i] - self.alpha * self.dw[i]
            self.b[i] = self.b[i] - self.alpha * self.db[i]
        '''self.w[1]=self.w[1]-self.alpha*self.dw[1]
        self.b[1]=self.b[1]-self.alpha*self.db[1]'''


#生成数据
m = 4000
X = np.random.rand(4, m)
#Y = np.random.randint(0, 2, (1, 8000))
#Y=0.5*x1+2*x2-5*x3+7*x4
para=np.array([0.5,2,-5,7])
para=para.reshape((1,4))
#para = para.repeat(m,axis=0)
print(para.shape)
Y_real=np.dot(para,X)
noise = np.random.normal(0, 0.05, Y_real.shape)
Y=Y_real+noise
print(X)
'''
#归一化X
for i in range(len(X)):
    X[i,:]=(X[i,:]-np.min(X[i,:]))/(np.max(X[i,:])-np.min(X[i,:]))
#X=(X-np.min(X))/(np.max(X)-np.min(X))
for i in range(len(Y)):
    Y=(Y-np.min(Y))/(np.max(Y)-np.min(Y))
# print(Y)'''
# 根据比例计算测试、训练和预测样本的划分个数
ratio = [0.6, 0.2, 0.2]
cols = [int(m * ratio[0]), int(m * (ratio[0] + ratio[1])), m]

#构建网络模型
N = np.array([4, 4, 1])
func = {1: 'sigmoid', 2: 'sigmoid', 3: 'x'}
#func = {1: 'tanh', 2: 'x', 3: 'x'}
#func = {1: 'sigmoid', 2: 'x'}
net = Net(len(func), N, func, 'QuadraticDifference')

#训练、测试和预测
train_cost=net.train(X[:, 0:cols[0]], Y[:, 0:cols[0]])
test_cost=net.test(X[:, cols[0]:cols[1]], Y[:, cols[0]:cols[1]])
print(test_cost)
predict=net.predict(X[:, cols[1]:m])
print('predict')
print(predict)
Y_true=Y[:, cols[1]:m]
error=list((predict-Y_true).transpose())
plt.figure(1)
plt.subplot(211)
plt.plot(range(len(train_cost)), train_cost)
plt.title('cost & predict_error')
plt.subplot(212)
plt.plot(range(len(error)),error)
plt.show()