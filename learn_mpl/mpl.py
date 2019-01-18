#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt

input_values=[1,2,3,4,5]
squares1=[1,4,9,16,25]
squares2=[1,8,27,64,125]

"""使用Plot绘制图形"""
# plt.plot(input_values,squares1,linewidth=5)
#
# #设置标题,并给坐标轴加上标签
# plt.title("Square Number", fontsize=30)
# plt.xlabel("Value", fontsize=15)
# plt.ylabel("Square of Value", fontsize=15)
#
# #设置刻度标记的大小
# plt.tick_params(axis='both', labelsize=15)

# fig=plt.figure()
# subfig1=fig.add_subplot(221)
# subfig1.plot(input_values,squares1,linewidth=2)
#
# subfig2=fig.add_subplot(222)
# subfig2.plot(input_values,squares2)

"""使用Scatter绘制散点"""
x_values=list(range(1,100))
y_values=[x**2 for x in x_values]

# plt.scatter(x_values,y_values,c=(0.8,0,0),edgecolor='none',s=10)
plt.scatter(x_values,y_values,c=y_values,cmap=plt.cm.Reds,edgecolor='none',s=10)

#设置每个坐标轴的取值范围
plt.axis([0,120,0,12000])


plt.savefig('plot.png',bbox_inches='tight')
plt.show()