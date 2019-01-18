#!/usr/bin/env python
# -*- coding: utf-8 -*-


#获取手写数据
#28*28的图片对象。每个图片对象根据需求是否转化为长度为784的横向量
#每个对象的标签为0-9的数字，one-hot编码成10维的向量

import struct
#from bp import *
from datetime import datetime
from fc import *


#数据加载器基类.派生出图片加载器和标签加载器
class Loader(object):
    def __init__(self,path, count):
        """初始化加载器
        path:数据文件路径
        count:文件中的样本个数
        """
        self.path=path
        self.count=count


    def get_file_content(self):
        """提取文件内容"""
        f=open(self.path,'rb')
        content=f.read()        #读取字节流
        f.close()
        return content          #字节数组

    #将unsigned byte字符转换为整数.python3中bytes的每个分量读取就会变成int
    # def to_int(self,byte):
    #     """将unsigned byte字符转换为整数"""
    #     return struct.unpack('B', byte)[0]



#图像数据加载器
class ImageLoader(Loader):
    """内部函数,从文件字节数组中获取第index个图像数据.文件中包含所有样本图片的数据"""
    def get_picture(self, content, index):
        start=index*28*28+16              #文件头16字节,后面每28*28个字节为一个图片数据
        picture=[]
        for i in range(28):
            picture.append([])            #图片添加一行像素
            for j in range(28):
                byte1=content[start+i*28+j]
                picture[i].append(byte1)             #python3中本来就是int
                #picture[i].append(self.to_int(content[start+i*28+j:start+i*28+j+1]))   #添加一行的每一个像素

        return picture                    #图片为[[x,x,x...][x,x,x...][x,x,x...]]的列表


    #将图像数据转化为784的行向量形式
    def get_one_sample(self,picture):
        """内部函数,将图像转化为样本的输入向量"""
        sample=[]
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])

        return sample


    #加载数据文件,获得全部样本的输入向量.onerow表示是否将每张图片转化为行向量,to2表示是否转化为0,1矩阵
    def load(self, onerow=False):
        """加载数据文件,获得全部样本的输入向量"""
        content=self.get_file_content()
        data_set=[]
        for index in range(self.count):            #遍历每一个样本
            onepic=self.get_picture(content,index)   #从样本数据集中获取第index个样本的图片数据,返回的是二维数组
            if onerow: onepic=self.get_one_sample(onepic)   #将图像转化为一维向量
            data_set.append(onepic)

        return data_set


#标签数据加载器
class LabelLoader(Loader):
    def load(self):
        """加载数据文件,获得全部样本的标签向量"""
        content=self.get_file_content()       #获取文件字节数组
        labels=[]
        """
        struct.unpack causing TypeError:'int' does not support the buffer interface
        That's because you're passing it the contents of data[34] which is an int.
        Try using data[34:35] instead, which is a one element byte array.
        """
        for index in range(self.count):       #遍历每一个样本
            onelabel=content[index+8]         #文件头有8个字节,在python3中contents[*]就是一个int
            onelabelvec=self.norm(onelabel)   #one-hot编码
            labels.append(onelabelvec)
            #labels.append(self.norm(content[index+8:index+9]))


        return labels


    def norm(self, label):
        """内部函数,one-hot编码.将一个值转换为10维的标签向量"""
        label_vec=[]
        #label_value=self.to_int(label)
        label_value=label    #python3中直接就是int
        for i in range(10):
            if i==label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)

        return label_vec


def get_training_data_set(onerow=True):
    """获得训练数据集,onerow表示是否将每张图片转化为行向量"""
    image_loader=ImageLoader('train-images-idx3-ubyte',600)
    label_loader=LabelLoader('train-labels-idx1-ubyte',600)
    return image_loader.load(onerow), label_loader.load()

def get_test_data_set(onerow=True):
    """获得测试数据集,onerow表示是否将每张图片转化为行向量"""
    image_loader=ImageLoader('t10k-images-idx3-ubyte',100)
    label_loader=LabelLoader('t10k-labels-idx1-ubyte',100)
    return image_loader.load(onerow), label_loader.load()


def show(sample):
    str=''
    for i in range(28):
        for j in range(28):
            if sample[i*28+j]!=0:
                str+='*'
            else:
                str+=''
        str +='\n'
    return str



def get_result(vec):
    """网络的输出是一个10维向量,这个向量第n个(从0开始编号)元素的值最大,那么n就是网络的识别结果"""
    max_value_index=0
    max_value=0
    for i in range(len(vec)):
        if vec[i]>max_value:
            max_value=vec[i]
            max_value_index=i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    """使用错误率来对神经网络进行评估"""
    error=0
    total=len(test_data_set)

    for i in range(total):
        label=get_result(test_labels[i])
        predict=get_result(network.predict(test_data_set[i]))
        if label!=predict:
            error+=1

    return float(error)/float(total)


def now():
    return datetime.now().strftime('%c')

def train_and_evaluate():
    """训练策略：每训练10轮，评估一次准确率，当准确率开始下降时终止训练"""
    last_error_ratio=1.0
    epoch=0
    train_data_set, train_labels=get_training_data_set()
    test_data_set, test_labels=get_test_data_set()
    network=Network([784,300,10])
    print("train_labels[0]:\n")
    print(train_labels[0])
    while True:
        epoch+=1
        network.train(train_labels,train_data_set, 0.3, 1)
        print('%s epoch %d finished' %(now(), epoch))
        if epoch%10==0:
            error_ratio=evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f '%(now(),epoch,error_ratio))
            if error_ratio>last_error_ratio:
                break
            else:
                last_error_ratio=error_ratio



if __name__=='__main__':
    train_and_evaluate()

