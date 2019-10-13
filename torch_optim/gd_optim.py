'''
优化算法:
以线性回归为例:(平方误差损失函数)
J(X)=1 / 2m * (f(X) - y)**2, 其中f(X)=wX, m为样本总数
J(X)对W的梯度为：
d(J(X)) / d(W) = X(wX - y) / m
'''

import numpy as np
from numpy.linalg import norm, inv
from itertools import cycle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['bgd', 'sgd', 'mbgd', 'momentum', 'nesterov', 'adagrad', 'adadelta', 'RMSprop', 'adam', 'adaMax', 'nadam', 'AMSGrad']

def eval_grad(X, y, W, n_sample):
    return X.T @ (X @ W - y) / n_sample

def data_prepare(train_data):
    X = train_data[:, :-1]
    y = train_data[:, [-1]]

    return X, y

def get_batch(train_data, n_sample, batch_size):
    # 小批量梯度下降初始化
    assert 1 < batch_size < n_sample
    train_data_split = np.array_split(train_data, batch_size)
    return train_data_split

'''
Batch gradient descent:

for i in range(nb_epochs):
    params_grad = evaluate_gradient(loss_function, data, params)
    params = params - learning_rate * params_grad

'''
def bgd(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001):
    #bgd
    W = np.ones((n_feature, 1))
    X, y = data_prepare(train_data)        

    for i in range(nb_epochs):
        grad = eval_grad(X, y, W, n_sample)
        if norm(grad) < epsilon:
            break
        W -= learning_rate * grad

    return W

'''
Stochastic gradient descent：
for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
        params_grad = evaluate_gradient(loss_function, example, params)
        params = params - learning_rate * params_grad
'''
def sgd(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001):
    #sgd
    W = np.ones((n_feature, 1))    

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            W -= learning_rate * grad

    return W

'''
Mini-batch gradient descent:
for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batch(data, batch_size=50):
        params_grad = evaluate_gradient(loss_function, batch, params)
        params = params - learning_rate * params_grad
'''
def mbgd(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, batch_size=4, learning_rate=0.001):
    #mbgd
    W = np.ones((n_feature, 1))

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        for batch_data in get_batch(train_data, n_sample, batch_size):
            X, y = data_prepare(batch_data)
            grad = eval_grad(X, y, W, n_sample)
            if norm(grad) < epsilon:
                break
            W -= learning_rate * grad

    return W

def momentum(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001, gamma=0.9):
    #sgd with momentum
    W = np.ones((n_feature, 1))
    v = np.zeros((n_feature, 1))   

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            v = gamma * v + learning_rate * grad
            W -= v

    return W

def nesterov(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001, gamma=0.9):
    #sgd with nesterov
    W = np.ones((n_feature, 1))
    v = np.zeros((n_feature, 1))   

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            W -= gamma * v
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            v = gamma * v + learning_rate * grad
            W -= v

    return W

def adagrad(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001, epsilon_station=1e-7):
    #adagrad
    W = np.ones((n_feature, 1))
    G = np.zeros((n_feature, 1)) #历史梯度的平方和

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            G += grad ** 2
            W -= learning_rate *grad / np.sqrt(G + epsilon_station)

    return W

def adadelta(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, epsilon_station=1e-7, gamma=0.9):
    #adadelta
    W = np.ones((n_feature, 1))
    G = np.zeros((n_feature, 1))#取决于历史梯度均值和当前梯度
    W_eta = np.zeros((n_feature, 1))#参数平方更新

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            W_old = W 
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            G = gamma * G + (1 - gamma) * grad ** 2
            W_eta = gamma * W_eta + (1 - gamma) * (W - W_old) ** 2
            W -=  np.sqrt(W_eta + epsilon_station) * grad / np.sqrt(G + epsilon_station)  

    return W

def RMSprop(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001, epsilon_station=1e-7, gamma=0.9):
    #RMSprop
    W = np.ones((n_feature, 1))
    G = np.zeros((n_feature, 1))#取决于历史梯度均值和当前梯度

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            G = gamma * G + (1 - gamma) * grad ** 2
            W -= learning_rate * grad / np.sqrt(G + epsilon_station)

    return W

def adam(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001, epsilon_station=1e-7, beta1=0.9, beta2=0.999):
    #adam
    W = np.ones((n_feature, 1))
    m = np.zeros((n_feature, 1))
    v = np.zeros((n_feature, 1))

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_j = m / (1 - beta1 ** (i + 1))
            v_j = v / (1 - beta2 ** (i + 1))
            W -= learning_rate * m_j / (np.sqrt(v_j) + epsilon_station)

    return W

def adaMax(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001, epsilon_station=1e-7, beta1=0.9, beta2=0.999):
    #adaMax
    W = np.ones((n_feature, 1))
    m = np.zeros((n_feature, 1))
    u = np.zeros((n_feature, 1))#adam中v的简化(无穷范式)

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            m = beta1 * m + (1 - beta1) * grad
            u = np.max(np.column_stack((beta2 * u, np.abs(grad))), axis=1).reshape(n_feature, 1)
            m_j = m / (1 - beta1 ** (i + 1))
            W -= learning_rate * m_j / u

    return W

def Nadam(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001, epsilon_station=1e-7, beta1=0.9, beta2=0.999):
    #adam
    W = np.ones((n_feature, 1))
    m = np.zeros((n_feature, 1))
    v = np.zeros((n_feature, 1))

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_j = m / (1 - beta1 ** (i + 1))
            v_j = v / (1 - beta2 ** (i + 1))
            g_j = ((1 - beta1) * grad) / (1 - beta1 ** (i + 1))
            W -= learning_rate * (beta1 * m_j + g_j) / (np.sqrt(v_j) + epsilon_station)

    return W

def AMSGrad(train_data, n_sample, n_feature, epsilon=1e-2, nb_epochs=1000, learning_rate=0.001, epsilon_station=1e-7, beta1=0.9, beta2=0.999):
    #adaMax
    W = np.ones((n_feature, 1))
    m = np.zeros((n_feature, 1))
    v = np.zeros((n_feature, 1))
    v_j = np.zeros((n_feature, 1))

    for i in range(nb_epochs):
        np.random.shuffle(train_data)
        X, y = data_prepare(train_data)
        for x_s, y_s in zip(X, y):
            v_p_j = v_j
            grad = eval_grad(x_s.reshape(1, n_feature), y_s, W, n_sample)
            if norm(grad) < epsilon:
                break
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            v_j = np.max(np.column_stack((v_p_j, v)), axis=1).reshape(n_feature, 1)
            W -= learning_rate * m / (np.sqrt(v_j) + epsilon_station)

    return W

def predict(X, W):
    return X @ W


if __name__ == '__main__':

    train_data = np.array([[1.1, 1.5, 2.5],
                           [1.3, 1.9, 3.2],
                           [1.5, 2.3, 3.9],
                           [1.7, 2.7, 4.6],
                           [1.9, 3.1, 5.3],
                           [2.1, 3.5, 6.0],
                           [2.3, 3.9, 6.7],
                           [2.5, 4.3, 7.4],
                           [2.7, 4.7, 8.1],
                           [2.9, 5.1, 8.8]])

    test_data = np.array([[3.1, 5.5, 9.5],
                          [3.3, 5.9, 10.2],
                          [3.5, 6.3, 10.9],
                          [3.7, 6.7, 11.6],
                          [3.9, 7.1, 12.3]])
    X_test, y_test = test_data[:, :-1], test_data[:, [-1]]

    n_sample = train_data.shape[0]
    n_feature = train_data.shape[1] - 1

    W = []
    y_pred_result=[]
    y_loss=[]

    W.append(bgd(train_data, n_sample, n_feature))
    W.append(sgd(train_data, n_sample, n_feature))
    W.append(mbgd(train_data, n_sample, n_feature))
    W.append(momentum(train_data, n_sample, n_feature))
    W.append(nesterov(train_data, n_sample, n_feature))
    W.append(adagrad(train_data, n_sample, n_feature))
    W.append(adadelta(train_data, n_sample, n_feature))
    W.append(RMSprop(train_data, n_sample, n_feature))
    W.append(adam(train_data, n_sample, n_feature))
    W.append(adaMax(train_data, n_sample, n_feature))
    W.append(Nadam(train_data, n_sample, n_feature))
    W.append(AMSGrad(train_data, n_sample, n_feature))

    for w in W:
        y_pred_result.append(predict(X_test, w))

    for y_pred in y_pred_result:
        y_loss.append(mean_squared_error(y_test, y_pred))

    print(W, '\n', y_pred_result, '\n', y_loss)
