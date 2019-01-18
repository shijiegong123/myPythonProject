#!/usr/bin/env python
# -*- coding: utf-8 -*-
from operator import le

import numpy as np
from activators import RuleActivator
import math

#获取卷积区域.
#input_array为单通道或多通道的矩阵。i为横向偏移,j为纵向偏移,stride为步幅,filter_width为过滤器宽度,filter_height为过滤器高度
def get_patch(input_array, i, j, filter_width,
              filter_height, stride):
    """从输入数组中获取本次卷积的区域,
    自动适配输入为2D和3D的情况
    """
    start_i=i*stride                  #下标i——行数——图像的高度
    start_j=j*stride                  #下标j——列数——图像的宽度

    if input_array.ndim==2:           #如果只有一个通道
        return input_array[
            start_i:start_i+filter_height,
            start_j:start_j+filter_width]

    elif input_array.ndim==3:         #如果有多个通道,也就是深度上全选
        return input_array[:,
               start_i:start_i+filter_height,
               start_j:start_j+filter_width]


#获取一个2D区域的最大值所在的索引方法一——Max Pooling
def get_max_index(array):
    max_i=0
    max_j=0
    max_value=array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j]>max_value:
                max_i, max_j=i, j
                max_value=array[i,j]

    return max_i, max_j


#获取一个2D区域的最大值所在的索引方法二——Max Pooling
def get_mas_index2(array):
    location=np.where(array==np.max(array))
    return location[0], location[1]


#计算一个过滤器的卷积运算,输出一个二维数据。每个通道的输入是图片,但是可能不是一个通道，所以这里自动适配输入为2D和3D的情况
def conv(input_array, kernel_array, output_array, stride, bias):
    """计算卷积,自动适配输入为2D和3D的情况"""
    channel_number=input_array.ndim        #获取输入数据的的通道
    output_width=output_array.shape[1]     #获取输出的宽度。一个过滤器产生的输出一定是一个通道
    output_height=output_array.shape[0]    #获取输出的高度
    kernel_width=kernel_array.shape[-1]    #过滤器的宽度。有可能有多个通道。多通道时shape=[深度、高度、宽度],单通道时shape=[高度、宽度]
    kernel_height=kernel_array.shape[-2]   #过滤器的高度。有可能有多个通道。多通道时shape=[深度、高度、宽度],单通道时shape=[高度、宽度]
    for i in range(output_height):
        for j in range(output_width):
            juanjiqu=get_patch(input_array, i, j, kernel_width,kernel_height, stride) #获取输入的卷积区(单通道或多通道)
            #这里是对每个通道的两个矩阵对应元素相乘求和,再将每个通道的和值求和
            kernel_values=(np.multiply(juanjiqu,kernel_array)).sum()    #卷积区与过滤器卷积运算。1，一个通道内，卷积区矩阵与过滤器矩阵对应点相乘后，求和值。2、将每个通道的和值再求和。
            output_array[i][j]=kernel_values+bias        #将卷积结果加上偏量
            # output_array[i,j]=(get_patch(input_array, i, j, kernel_width,
            #                              kernel_height, stride)*kernel_array).sum()+bias


#为数组增加Zero padding,zp为补零层数,自动适配输入为2D和3D的情况
def padding(input_array, zp):
    """为数组增加Zero padding,自动适配输入为2D和3D的情况"""
    if zp==0:   #不补零
        return input_array
    else:
        if input_array.ndim==3:                         #如果输入有多个通道
            input_width=input_array.shape[2]            #获取输入的宽度
            input_height=input_array.shape[1]           #获取输入的高度
            input_depth=input_array.shape[0]            #获取输入的深度
            padded_array=np.zeros((input_depth,
                                   input_height+2*zp,
                                   input_width+2*zp))   #先定义一个补0后大小的全矩阵
            padded_array[:,
                  zp:zp+input_height,
                  zp:zp+input_width]=input_array        #每个通道上,将中间部分替换成输入,这样就变成了原矩阵周围补0的形式
            return padded_array
        elif input_array.ndim==2:                       #输入只有一个通道
            input_width=input_array.shape[1]            #获取输入的宽度
            input_height=input_array.shape[0]           #获取输入的高度
            padded_array=np.zeros((
                input_height+2*zp,
                input_width+2*zp))                      #先定义一个补0后大小的全0矩阵
            padded_array[zp:zp+input_height,
                zp:zp+input_width]=input_array          #将中间部分替换成输入,这样就变成了员矩阵周围补0的形式
            return padded_array



#对numpu数组进行逐个元素的操作.op为函数.element_wise_op函数实现了对numpy数组进行按元素操作,并将返回值写回数组中
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...]=op(i)      #将元素i传入op函数,进行修改后在返回给i



#Filter类保存了卷积层的参数以及梯度,并且实现了用梯度下降算法来更新参数
class Filter(object):
    def __init__(self, width, height, depth, filter_num):
        #卷积核随机初始化为[-sqrt(6/(fan_in+fan_out)),sqrt(6/(fan_in+fan_out))]之间的数
        #其中fan_in为输入通道数与滤波器宽高的乘积，即width*height*depth
        #其中fan_out为输出通道数与滤波器宽高的乘积,即width*height*filter_num
        wimin=-math.sqrt(6/(width*height*depth+width*height*filter_num))
        wimax=-wimin
        self.weights=np.random.uniform(wimin, wimax,(depth, height, width))  #随机初始化卷积层权重一个很小的值
        #self.weights=np.random.uniform(-1e-4, 1e-4, (depth, height, width))  #随机初始化卷积层权重一个很小的值
        self.bias=0    #初始化偏量为0
        self.weights_grad=np.zeros(self.weights.shape)     #初始化权重梯度
        self.bias_grad=0                                   #初始化偏量梯度

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' %(repr(self.weights),repr(self.bias))


    #读取权重
    def get_weights(self):
        return self.weights


    #读取偏量
    def get_bias(self):
        return self.bias


    #更新权重和偏量
    def update(self, learning_rate):
        self.weights-=learning_rate*self.weights_grad
        self.bias-=learning_rate*self.bias_grad


#用ConvLayer类来实现一个卷积层.下面的代码是初始化一个卷积层,可以在构造函数中设置卷积层的超参数
class ConvLayer(object):
    #初始化构造卷积层：输入宽度、输入高度、通道数、滤波器宽度、滤波器高度、滤波器数目、补零数目、步长、激活器、学习速率
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, filter_number,
                 zero_padding, stride, activator, learing_rate):
        self.input_width=input_width              #输入宽度
        self.input_height=input_height            #输入高度
        self.channle_number=channel_number        #通道数=输入的深度=过滤器的深度
        self.filter_width=filter_width            #过滤器的宽度
        self.filter_height=filter_height          #过滤器的高度
        self.filter_number=filter_number          #过滤器的数量
        self.zero_padding=zero_padding            #补0的圈数
        self.stride=stride                        #步幅
        self.output_width=
        self.output_height=
        self.output_array=
        self.filters=[]                           #卷积层的每个过滤器
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height, self.channle_number, filter_number))
        self.activator=activator                  #使用rule激活器
        self.learning_rate=learing_rate           #学习速率


    #计算卷积层色输出.输出结果保存在self.output_array
    def forward(self, inpu_array):
        self.input_array=inpu_array                    #多个通道的图片,每个通道为一个二维图片
        self.padded_input_array=padding(inpu_array, self.zero_padding)    #先将输入补0
        for i in range(self.filter_number):            #每个过滤器产生一个二维数组的输出
            filter=self.filters[i]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[i], self.stride, filter.get_bias())

        #element_wise_op()函数实现了对numpy数组进行按元素操作,并将返回值写回到数组中
        element_wise_op(self.output_array, self.activator.forward)


    #反向传播。input_array为该层的输入,sensitivity_array为当前层的输出误差(和输出的维度相同),activator为激活函数
    def backward(self, input_array, sensitivity_array, activator):
        """
        计算传递给前一层的误差项,以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Fiter对象的weights_grad
        """
        self.forward(input_array)   #先根据输入计算经过该卷积层后的输出.(卷积层有几个过滤器,输出层的深度就是多少,输出每一层为一个二维数组)
        self.bp_sensitivity_map(sensitivity_array, activator)   #将误差先传递到前一层,self.delta_array存储上一层的误差
        self.bp_gradient(sensitivity_array)          #计算每个过滤器的w和b的梯度


    #按照梯度下降,更新权重
    def update(self):
        for filter in self.filters:
            filter.update(self.learning_rate)     #每个过滤器都更新权重


    #将误差项传递给上一层.sensitivity_array:本层的误差. activator:上一层的激活函数
    def bp_sensitivity_map(self, sensitivity_array, activator):   #公式9
        #根据卷积步长,对原始sensitivity_map进行补0扩展,扩展成步长为1的输出误差形状.再用公式8求解
        expanded_error_array=self.expand_sensitivity_map(sensitivity_array)










