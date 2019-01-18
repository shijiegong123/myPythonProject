#!/usr/bin/env python
# -*- coding: utf-8 -*-





import random
import numpy as np
from activators import SigmoidActivator
from functools import reduce

W=np.random.uniform(-0.1, 0.1 ,(5,6))
b=np.zeros((5,1))

print("W=%s\nb=%s"%(W,b))
print("W[-1]:")
print(W[-1])
print("W[:-1]:")
print(W[:-1])
print("W[::-1]:")
print(W[::-1])




"""向量化编程,实现全连接神经网络"""

# 全连接层实现类,实现了全连接层的前向和后向计算,输入对象x、神经层输出a、输出y均为列向量
# 包含上层所有节点 下层所有节点 两层间的权重数组W
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        """构造函数
        input_size:本层输入向量的维度,本层节点数
        output_size:本层输出向量的维度,下层节点数
        activator:激活函数
        """
        self.input_size=input_size
        self.output_size=output_size
        self.activator=activator
        #权重数组
        self.W=np.random.uniform(-0.1, 0.1,
                                 (output_size, input_size))  #初始化为-0.1~0.1之间的数。权重的大小。行数=输出个数，列数=输入个数。a=w*x，a和x都是列向量
        #偏置项b
        self.b=np.zeros((output_size,1))    #全0列向量偏置项
        #输出向量
        self.output=np.zeros((output_size,1))  #初始化为全0列向量


    def forward(self, input_array):
        """前向计算,预测输出
        input_array:输入向量,维度必须等于input_size
        """
        self.input=input_array
        self.output=self.activator.forward(
            np.dot(self.W, input_array) + self.b)


    def backward(self, delta_array):
        """反向计算W和b的梯度
        delta_array:从上层传递过来的误差项,列向量
        """
        # self.delta=self.activator.backward(self.input)*np.dot(
        #     self.W.T, delta_array)
        self.delta=np.multiply(self.activator.backward(self.input),
                            np.dot(self.W.T, delta_array))      #计算当前层的误差
        self.W_grad=np.dot(delta_array, self.input.T)              #计算W的梯度.梯度=误差*输入
        self.b_grad=delta_array                                    #计算b的梯度


    def update(self, learning_rate):
        """使用梯度下降算法更新权重"""
        self.W+=learning_rate*self.W_grad
        self.b+=learning_rate*self.b_grad

    def dump(self):
        print('W:%s\nb:%s'%(self.W, self.b))

#上面这个类一举取代了原先的Layer、Node、Connection等类，不但代码更加容易理解，而且运行速度也快了几百倍



#神经网络类
class Network(object):
    def __init__(self, layers):
        """初始化一个全连接神经网络
        layers:数组,描述神经网络每层节点数,包含输入层节点个数、隐藏层节点个数、输出层节点个数"""
        self.layers=[]
        for i in range(len(layers)-1):     #一个FullConnetctedLayer-全连接层对象包含了上层和下层两层的联系,所以长度需要-1
            self.layers.append(FullConnectedLayer(layers[i], layers[i+1], SigmoidActivator()))


    def predict(self, sample):
        """使用神经网络实现预测
        sample:输入样本
        """
        sample=sample.reshape(-1,1)     #将样本转化为列向量
        output=sample                   #输入样本作为输入层的输出
        for layer in self.layers:
            layer.forward(output)                 #逐层向后计算预测值,因为每层都是线性回归
            output=layer.output
        return  output

    def train(self, labels, data_set, rate, epoch):
        """训练函数
        labels:样本标签
        data_set:输入样本
        rate:学习速率
        epoch:训练轮数
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d].reshape(-1,1),
                                      data_set[d].reshape(-1,1), rate)  #将输入对象和输出标签转化为列向量


    def train_one_sample(self, label, sample, rate):
        """内部函数，用一个样本训练网络"""
        self.predict(sample)                   #根据样本对象预测值
        self.calc_gradient(label)              #计算梯度
        self.update_weight(rate)               #更新权重


    def calc_gradient(self, label):
        """内部函数,计算每个节点的误差
        lable为一个样本的输出向量,也就对应了最后一个所有输出节点输出的值"""
        # delta=self.layers[-1].activator.backward(
        #     self.layers[-1].output)*(label-self.layers[-1].output)
        delta=np.multiply(self.layers[-1].activator.backward(self.layers[-1].output),
                          (label-self.layers[-1].output))    #计算输出误差

        for layer in self.layers[::-1]:        #将网络连接层反过来排序
            layer.backward(delta)              #逐层向前计算误差.计算神经网络层和输入层误差
            delta=layer.delta
        return delta


    def update_weight(self, rate):
        for layer in self.layers:             #逐层更新权重
            layer.update(rate)


    def dump(self):
        for layer in self.layers:
            layer.dump()


    def loss(self, output, label):
        return 0.5*((label-output)*(label-output)).sum()


    def gradient_check(self, sample_feature, sample_label):
        """梯度检查
        network:神经网络对象
        sample_feature:样本的特征
        sample_label:样本的标签
        """

        #获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        #检查梯度
        epsilon=10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j]+=epsilon
                    output=self.predict(sample_feature)
                    err1=self.loss(sample_label, output)
                    fc.W[i,j]-=2*epsilon
                    output=self.predict(sample_feature)
                    err2=self.loss(sample_label,output)
                    expect_grad=(err1-err2)/(2*epsilon)
                    fc.W[i,j]+=epsilon
                    print('weights(%d,%d):expected-actual %.4e - %.4e'%(i,j,expect_grad, fc.W_grad[i,j]))






"""下面使用随机数对神经网络进行测试"""
def transpose(args):
    return list(map(
        lambda arg: list(map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg))
        , args
    ))


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        data = list(map(lambda m: 0.9 if number & m else 0.1, self.mask))
        return np.array(data).reshape(8, 1)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec[:,0]))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)


def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels, data_set

def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print( 'correct_ratio: %.2f%%' % (correct / 256 * 100))




def test():
    labels, data_set = transpose(train_data_set())
    net = Network([8, 3, 8])
    rate = 0.5
    mini_batch = 20
    epoch = 10
    for i in range(epoch):
        net.train(labels, data_set, rate, mini_batch)
        print( 'after epoch %d loss: %f' % (
            (i + 1),
            net.loss(labels[-1], net.predict(data_set[-1]))
        ))
        rate /= 2
    correct_ratio(net)


def gradient_check():
    '''
    梯度检查
    '''
    labels, data_set = transpose(train_data_set())
    net = Network([8, 3, 8])
    net.gradient_check(data_set[0], labels[0])
    return net


def train(network):
    labels, data_set=train_data_set()
    network.train(labels, data_set, 0.3, 50)


if __name__=='__main__':
    net=Network([8,7,8])
    train(net)
    net.dump()
    correct_ratio(net)

    # test()
    # gradient_check()

# 由于使用了逻辑回归函数，所以只能进行分类识别。识别ont-hot编码的结果
# if __name__ == '__main__':
#     # 使用神经网络实现and运算
#     data_set = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     labels = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
#     net = Network([2, 1, 2])  # 输入节点2个（偏量b会自动加上），神经元1个，输出节点2个。
#     net.train(labels, data_set, 2, 100)
#     for layer in net.layers:  # 网络层总不包含输出层
#         print('W:', layer.W)
#         print('b:', layer.b)
#
#     # 对结果进行预测
#     sample = np.mat([[0, 1]])
#     y = net.predict(sample)
#     print('y=%s'%y)








